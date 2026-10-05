use crate::{
    backend::{Decoder, MemoryRmError},
    llama_cpp::checkpoint::{
        CheckpointBudget, Checkpointing, Checkpoints, SeqMemory,
    },
    Batch, LlamaCppModel, Token,
};

use std::{path::PathBuf, sync::Mutex};

use llama_cpp_sys_3::{
    ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED, llama_backend_free,
    llama_backend_init, llama_context, llama_context_params, llama_decode,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED, llama_free,
    llama_get_embeddings_ith, llama_get_logits_ith, llama_get_memory,
    llama_init_from_model, llama_memory_clear, llama_memory_seq_add,
    llama_memory_seq_cp, llama_memory_seq_div, llama_memory_seq_keep,
    llama_memory_seq_pos_max, llama_memory_seq_pos_min, llama_memory_seq_rm,
    llama_model_n_embd_out, llama_n_batch, llama_n_ctx, llama_n_seq_max,
    llama_n_ubatch, llama_numa_init, llama_perf_context,
    llama_perf_context_data, llama_perf_context_reset, llama_pos, llama_seq_id,
    llama_set_n_threads, llama_state_get_data, llama_state_get_size,
    llama_state_seq_flags, llama_state_seq_get_data_ext,
    llama_state_seq_get_size_ext, llama_state_seq_set_data_ext,
    llama_state_set_data,
};

use thiserror::Error;

/// `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (llama.h): serialize only the
/// state a KV truncate cannot rewind — sliding-window cells, recurrent
/// state. A `#define`, so bindgen does not carry it.
const STATE_SEQ_PARTIAL_ONLY: llama_state_seq_flags = 1;

/// Global engine count. When this drops to 0, the llama backend is freed in
/// the last [`LlamaCppDecoder`]'s `Drop` implementation.
pub(super) static ENGINE_COUNT: Mutex<usize> = Mutex::new(0);

/// Lock [`ENGINE_COUNT`], ignoring poisoning.
///
/// The guarded region is a single `usize` increment/decrement plus the
/// backend init/free calls; there is no invariant a panicking holder
/// could leave half-established that a later caller would misread. But
/// `Drop` also takes this lock, so propagating a poison error would
/// turn every subsequent teardown into a panic-during-unwind — an
/// abort. Recovering the inner value is strictly the safer failure
/// mode here.
fn engine_count() -> std::sync::MutexGuard<'static, usize> {
    ENGINE_COUNT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Possible errors when creating a new [`crate::Engine`] or
/// [`LlamaCppDecoder`].
///
/// [`Self::is_resource`] splits them in two: failures found before the
/// backend allocated anything (a missing file, an unsupported
/// architecture), after which the process is as it was, and failures
/// after it began to (out of memory loading the weights or creating
/// the KV cache), after which llama.cpp's state may be partial.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum NewError {
    /// The model file could not be opened: missing, a directory, or
    /// not readable. Checked before llama.cpp sees the path.
    #[error("Could not open model file {path}: {source}")]
    Unreadable {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// llama.cpp could not read the model's header, metadata or
    /// vocabulary — not a GGUF, an unsupported architecture, a
    /// malformed vocabulary — in a vocab-only pass that allocates no
    /// backend memory.
    #[error("Could not read model metadata from {path}")]
    Metadata { path: PathBuf },
    /// The full load failed after the vocab-only pass succeeded: while
    /// the backend was allocating and filling the weights. Out of
    /// memory, most likely; llama.cpp does not say.
    #[error("Could not load model from file: {path}")]
    Model { path: PathBuf },
    /// `llama_init_from_model` failed: most likely the KV cache or
    /// compute buffers could not be allocated; llama.cpp does not say.
    #[error("Could not create context")]
    Context,
    /// An mmproj sidecar exists next to the model but failed to load.
    /// Hard error by design: continuing text-only would silently drop
    /// images.
    #[cfg(feature = "mtmd")]
    #[error("Could not load mmproj sidecar {path}: {source}")]
    Mtmd {
        path: PathBuf,
        #[source]
        source: crate::llama_cpp::mtmd::MtmdNewError,
    },
}

impl NewError {
    /// Whether the failure came after the backend began allocating, so
    /// that llama.cpp may be left with partial state: a resource
    /// failure, out of memory most likely. `false` only for failures
    /// found before any backend allocation ([`Self::Unreadable`],
    /// [`Self::Metadata`], an mmproj path C cannot take); where
    /// llama.cpp does not say why, the answer is `true`.
    pub fn is_resource(&self) -> bool {
        match self {
            Self::Unreadable { .. } | Self::Metadata { .. } => false,
            Self::Model { .. } | Self::Context => true,
            #[cfg(feature = "mtmd")]
            Self::Mtmd { source, .. } => matches!(
                source,
                crate::llama_cpp::mtmd::MtmdNewError::LoadFailed { .. }
            ),
        }
    }
}

static_assertions::assert_impl_all!(NewError: Send, Sync);

/// Possible errors when calling [`LlamaCppDecoder::decode`].
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum DecodeError {
    #[error("Could not find a KV slot for the Batch. Try reducing the size of the batch or increase the context size.")]
    NoKvSlot,
    /// `llama_decode` rejected the batch before touching the KV cache.
    #[error("`llama_decode` rejected the batch as invalid (-1)")]
    InvalidBatch,
    /// Aborted mid-batch. **The KV cache is dirty**: llama.h:952 —
    /// "processed ubatches will remain in the context's memory".
    #[error("`llama_decode` was aborted (2); processed ubatches remain in the KV cache")]
    Aborted,
    /// Fatal error mid-batch. **The KV cache is dirty**, same as
    /// [`Self::Aborted`].
    #[error("`llama_decode` failed fatally ({code}); processed ubatches remain in the KV cache")]
    Fatal { code: i32 },
    /// A return code llama.cpp did not document at the version we were
    /// built against. Treated as KV-dirty, because we cannot know.
    #[error("`llama_decode` returned an unrecognized code: {code}")]
    ErrorCode { code: i32 },
    /// Caught before the call. llama.cpp asserts
    /// `n_tokens_all <= cparams.n_batch` with a non-`NDEBUG`-gated
    /// `GGML_ASSERT`, i.e. it aborts the *process* rather than
    /// returning, so this has to be checked on our side.
    #[error(
        "batch of {n_tokens} tokens exceeds the context's n_batch of {n_batch}"
    )]
    BatchTooLarge { n_tokens: usize, n_batch: u32 },
    /// `llama_decode` reported success but the logits it produced
    /// contain a non-finite value (NaN/Inf).
    ///
    /// This is a *decode* failure wearing a success return code, and
    /// without this check it surfaces far from its cause: every
    /// comparison against NaN is false, so the first thing to notice is
    /// `partial_cmp(…).unwrap()` panicking inside
    /// [`Candidates::sort`](crate::Candidates::sort) — which reads like
    /// a sampler bug when the sampler is the one component that did
    /// nothing wrong.
    ///
    /// Causes seen in practice are backend-side, not ours: a kernel
    /// that has no correct path for the model's shapes or quantization
    /// on the active device. The 2026-07-28 case was Mistral Small 4
    /// (`mistral4`, DeepSeek-2 MLA graph) on Metal, where prefills of
    /// ≥32 tokens returned an entirely NaN vocabulary while the same
    /// prompt on CPU was clean — 32 being `ne21_mm_id_min`, the
    /// `mul_mv_id` → `mul_mm_id` switch for the MoE matmul. When this
    /// fires, compare against `no_gpu` / a different backend before
    /// suspecting anything in this crate.
    ///
    /// Treated as KV-dirty: NaN is contagious through the KV cache, so
    /// the cells this decode wrote must be wiped rather than reused.
    #[error(
        "decode produced a non-finite logit ({value}) at index {index}; \
         the KV cache is poisoned and must be wiped. This is a backend \
         failure, not a sampling one — try a different backend (e.g. \
         `LlamaCppOptions::cpu_only`) to confirm"
    )]
    NonFinite { index: usize, value: f32 },
}

/// Locate the first non-finite value in a logit slice, or `None` when
/// every value is finite.
///
/// Split out from the decode path so the predicate is testable without
/// a model, and shaped for the hot loop: the common case is one
/// vectorizable `all` pass, and only a slice that has already failed
/// pays for the second pass that locates the offender.
pub(crate) fn first_non_finite(logits: &[f32]) -> Option<(usize, f32)> {
    if logits.iter().all(|l| l.is_finite()) {
        return None;
    }
    logits
        .iter()
        .position(|l| !l.is_finite())
        .map(|i| (i, logits[i]))
}

impl DecodeError {
    /// Whether the KV cache may hold partially-decoded ubatches.
    ///
    /// llama.h:950-958 splits decode failures in two: `1` and `-1`
    /// restore "the state before this call", while `2` and `< -1`
    /// leave whatever ubatches already completed sitting in memory.
    /// After a KV-dirty error, a caller's own position bookkeeping is
    /// no longer trustworthy — reconcile against
    /// [`LlamaCppDecoder::memory_seq_pos_max`] or wipe the sequence
    /// before reusing it.
    pub fn kv_dirty(&self) -> bool {
        match self {
            Self::NoKvSlot
            | Self::InvalidBatch
            | Self::BatchTooLarge { .. } => false,
            Self::Aborted
            | Self::Fatal { .. }
            | Self::ErrorCode { .. }
            | Self::NonFinite { .. } => true,
        }
    }
}

static_assertions::assert_impl_all!(DecodeError: Send, Sync);

/// Flash Attention policy for a new [`crate::LlamaCppEngine`] context.
///
/// llama.cpp's default is [`Self::Auto`] — it enables Flash Attention
/// when the active backend supports it (typical on Metal, CUDA, Vulkan).
/// [`Self::Disabled`] is useful as a diagnostic: FA uses a fused softmax
/// kernel that can produce slightly different logits than the non-FA
/// attention path on close-race token distributions, and toggling it off
/// rules that out as a source of divergence against other runners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum FlashAttention {
    /// Let llama.cpp decide based on backend capabilities (default).
    Auto,
    /// Force-disable Flash Attention.
    Disabled,
    /// Force-enable. Errors at context creation if the backend doesn't
    /// support it.
    Enabled,
}

impl FlashAttention {
    /// Map to the raw llama.cpp enum value.
    pub(super) fn as_raw(self) -> llama_cpp_sys_3::llama_flash_attn_type {
        match self {
            Self::Auto => llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
            Self::Disabled => {
                llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED
            }
            Self::Enabled => {
                llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED
            }
        }
    }
}

/// llama.cpp-backed decoder: owns a `llama_context`, manages the KV
/// cache, runs decode passes, and exposes logits / embeddings.
///
/// Implements [`crate::backend::Decoder`]. `LlamaCppDecoder::new`
/// handles backend lifecycle (`llama_backend_init` + `llama_numa_init`
/// on the first-ever decoder; `llama_backend_free` on the last
/// dropped).
#[derive(Debug)]
pub struct LlamaCppDecoder {
    pub(crate) context: *mut llama_context,
    /// The model this context was created from.
    ///
    /// A `llama_context` holds `const llama_model &` for its entire
    /// life, so it must never outlive the weights. Holding a handle
    /// (a refcount bump — see [`LlamaCppModel`]) is what makes that
    /// structural instead of a drop-order convention the caller could
    /// break without writing `unsafe` (issue #54).
    model: LlamaCppModel,
    /// Cached vocab size from the source model — used to size logit
    /// slices. Cached rather than read from `model` per call purely to
    /// keep an FFI hop out of [`Self::logits`], which the sampling
    /// loop hits once per token.
    n_vocab: usize,
    /// Cached embedding dimension from the source model. Reported by
    /// [`Self::embedding_size`]; **not** the slice stride — see
    /// [`Self::embedding_size_out`].
    embedding_size: usize,
    /// Row stride for [`Self::embeddings`], from
    /// `llama_model_n_embd_out`. Equals `embedding_size` unless the
    /// GGUF publishes `%s.embedding_length_out` (or the arch overrides
    /// it, e.g. WavTokenizer-dec). llama.cpp indexes `embd.data` by
    /// this, so sizing slices with `embedding_size` would over-read.
    embedding_size_out: usize,
    /// Host-RAM checkpoints backing [`Decoder::checkpoint_pos`] /
    /// [`Decoder::restore_to`], in the [`Checkpointing`] mode the model
    /// needs: none for dense attention, which truncates losslessly at
    /// any position; partial ones for sliding-window and recurrent /
    /// hybrid layers, which a truncate cannot rewind. Force on via
    /// [`Self::set_seq_snapshots`] (tests, or callers wanting rewind
    /// insurance on a dense model).
    checkpoints: Checkpoints,
}

unsafe impl Send for LlamaCppDecoder {}

impl LlamaCppDecoder {
    /// Create a decoder bound to `model` with the given context params.
    ///
    /// The decoder keeps a handle on `model`, so the weights outlive
    /// the context no matter what the caller does with its own handle.
    /// Two decoders built from clones of one handle share a single
    /// copy of the weights and pay only for their own KV caches.
    ///
    /// Handles the llama.cpp backend lifecycle: on the first-ever
    /// decoder (`ENGINE_COUNT` transitions 0→1) runs
    /// `llama_backend_init` + `llama_numa_init`. Subsequent decoders
    /// just increment the count.
    ///
    /// If context creation fails, the count is rolled back (and the
    /// backend torn down if we were the first). The caller can
    /// retry without double-init.
    pub fn new(
        model: &LlamaCppModel,
        context_params: llama_context_params,
        numa_strategy: Option<u32>,
    ) -> Result<Self, NewError> {
        // `ggml_numa_strategy` is `c_uint` (== u32), so this is a plain
        // pass-through — resolved before the guard on principle: no
        // fallible work belongs inside a region whose panic would
        // poison ENGINE_COUNT and turn every later `Drop` into an
        // abort. (The prior code did a `u32 -> u32` `try_into().unwrap()`
        // here, which looked like a panic risk but was infallible.)
        let numa = numa_strategy
            .unwrap_or(ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED);

        {
            let mut count = engine_count();
            *count += 1;
            if *count == 1 {
                // SAFETY: first live decoder in the process; the guard
                // serializes this against any concurrent init/free.
                unsafe {
                    llama_backend_init();
                    llama_numa_init(numa);
                }
            }
        }

        // SAFETY: `model` is live for the call, and the handle we
        // store below keeps it live for as long as the context exists.
        // `as_ptr_mut` is sound here specifically because
        // `llama_init_from_model` only reads the model before binding
        // it to the context's `const llama_model &` — the non-const in
        // its signature is vestigial (see `ModelInner`'s `Sync` note).
        let context = unsafe {
            llama_init_from_model(model.as_ptr_mut(), context_params)
        };
        if context.is_null() {
            // Roll back the count we just reserved.
            let mut count = engine_count();
            *count -= 1;
            if *count == 0 {
                unsafe { llama_backend_free() };
            }
            return Err(NewError::Context);
        }

        // Sliding-window cells and recurrent layer state cannot be
        // rewound by a KV truncate alone, so those models checkpoint
        // what it cannot rewind at every cache anchor.
        let checkpointing = Checkpointing::for_model(
            model.is_recurrent(),
            model.is_hybrid(),
            model.n_swa(),
        );

        Ok(Self {
            context,
            n_vocab: model.n_vocab() as usize,
            embedding_size: model.embedding_size() as usize,
            // SAFETY: `model` is live for the call and this only reads
            // cached hparams.
            embedding_size_out: unsafe {
                llama_model_n_embd_out(model.as_ptr())
            } as usize,
            model: model.clone(),
            // A full set of anchors per sequence, so one cache slot's
            // checkpoints never evict another's.
            checkpoints: Checkpoints::new(
                checkpointing,
                model.n_swa(),
                crate::snapshot_store::cap_for_sequences(
                    // SAFETY: `context` was checked non-null above.
                    unsafe { llama_n_seq_max(context) } as usize,
                ),
            ),
        })
    }

    /// The model this decoder's context was created from.
    ///
    /// The decoder holds its own handle, so this stays valid even
    /// after every other handle has been dropped — which is what makes
    /// a standalone decoder usable on its own (tokenize, look up
    /// special tokens) without threading a second `LlamaCppModel`
    /// alongside it.
    pub fn model(&self) -> &LlamaCppModel {
        &self.model
    }

    /// Raw pointer to the underlying llama.cpp context (const). Cast to
    /// `*mut` to change memory and it bypasses the checkpoint
    /// bookkeeping as [`Self::context_ptr_mut`] does.
    pub fn context_ptr(&self) -> *const llama_context {
        self.context
    }

    /// Raw pointer to the underlying llama.cpp context (mut).
    ///
    /// Takes `&mut self` so the exclusivity the pointer implies is
    /// actually held — the same reason [`Self::decode`] does.
    ///
    /// Bypasses the checkpoint bookkeeping: change a sequence's memory
    /// through it, and its partial checkpoints may describe a different
    /// history. Follow such a change with [`Self::memory_seq_rm`] over
    /// the range it touched (or [`Self::set_seq_snapshots`]`(false)`),
    /// which drops them.
    pub fn context_ptr_mut(&mut self) -> *mut llama_context {
        self.context
    }

    /// Vocabulary size seen by this decoder (cached from model).
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Embedding dimension seen by this decoder (cached from model).
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    /// Context window size (tokens).
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_n_ctx(self.context) }
    }

    /// Max batch size configured on this context.
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_n_batch(self.context) }
    }

    /// Micro-batch size configured on this context.
    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_n_ubatch(self.context) }
    }

    /// Size of the serialized global state (logits, embedding, memory).
    pub fn state_size(&self) -> usize {
        unsafe { llama_state_get_size(self.context) }
    }

    /// Serialize the global state.
    pub fn get_state(&self) -> Vec<u8> {
        let len = self.state_size();
        let mut buf = vec![0u8; len];
        let copied = unsafe {
            llama_state_get_data(self.context, buf.as_mut_ptr(), len)
        };
        assert_eq!(copied, len);
        buf
    }

    /// Deserialize the global state (bytes from [`Self::get_state`]).
    /// Replaces every sequence, so partial checkpoints are dropped.
    ///
    /// Note [`Self::state_size`] is *content-dependent* — the KV
    /// portion grows with what the cache holds — so a valid saved
    /// state routinely differs in length from the context's current
    /// `state_size` (e.g. restoring after `memory_clear`). llama.cpp
    /// reads the buffer's own header; no length precondition exists.
    ///
    /// # Panics
    /// * If llama.cpp does not consume `state` fully — corrupt bytes
    ///   or a state saved from a different model / context shape.
    pub fn set_state(&mut self, state: &[u8]) {
        self.checkpoints.invalidate_from(-1, -1);
        let read = unsafe {
            llama_state_set_data(self.context, state.as_ptr(), state.len())
        };
        assert_eq!(read, state.len(), "llama.cpp rejected saved state");
    }

    /// Size of the serialized state for a single sequence.
    pub fn state_seq_size(&self, seq_id: llama_seq_id) -> usize {
        ContextMemory(self.context).state_size(seq_id, 0)
    }

    /// Serialize the state of a single sequence (its KV cells plus any
    /// recurrent layer state). The bytes restore via
    /// [`Self::set_state_seq`] — into this context or another one on
    /// the same model.
    pub fn get_state_seq(&self, seq_id: llama_seq_id) -> Vec<u8> {
        let buf = ContextMemory(self.context).state(seq_id, 0);
        assert!(
            !buf.is_empty(),
            "llama.cpp failed to serialize seq {seq_id}"
        );
        buf
    }

    /// Restore a single sequence's state from bytes produced by
    /// [`Self::get_state_seq`], loading them as `dest_seq_id`. Returns
    /// `false` if llama.cpp rejects the payload (wrong model, corrupt
    /// bytes, insufficient KV room) — the destination sequence is left
    /// cleared in that case. Either way the sequence's partial
    /// checkpoints are dropped: the KV they sat on is gone.
    pub fn set_state_seq(
        &mut self,
        state: &[u8],
        dest_seq_id: llama_seq_id,
    ) -> bool {
        self.checkpoints.invalidate_from(dest_seq_id, -1);
        ContextMemory(self.context).load_state(state, dest_seq_id, 0)
    }

    /// How [`Decoder::checkpoint_pos`] / [`Decoder::restore_to`] rewind
    /// this model's sequences: [`Checkpointing::Off`] for dense
    /// attention, [`Checkpointing::Partial`] for sliding-window and
    /// recurrent / hybrid models.
    pub fn checkpointing(&self) -> Checkpointing {
        self.checkpoints.mode()
    }

    /// Whether [`Decoder::checkpoint_pos`] stores anything — see
    /// [`Self::checkpointing`].
    pub fn seq_snapshots_enabled(&self) -> bool {
        self.checkpoints.mode() != Checkpointing::Off
    }

    /// Force checkpointing on or off. On, a dense model stores
    /// [`Checkpointing::Whole`] snapshots; a model that needs
    /// checkpoints keeps its own mode. Disabling drops all stored
    /// checkpoints — and on a sliding-window or recurrent model, the
    /// ability to rewind anywhere but the head.
    pub fn set_seq_snapshots(&mut self, enabled: bool) {
        self.checkpoints.force(enabled);
    }

    /// Number of per-sequence checkpoints currently held.
    pub fn seq_snapshot_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Host RAM the checkpoints currently hold, in bytes.
    pub fn seq_snapshot_bytes(&self) -> usize {
        self.checkpoints.bytes()
    }

    /// Bound the host RAM the checkpoints may hold, evicting the least
    /// recently used ones now if they are over it. See
    /// [`CheckpointBudget`] for the default.
    pub fn set_checkpoint_budget(&mut self, budget: CheckpointBudget) {
        self.checkpoints.set_budget(budget);
    }

    /// Performance information.
    pub fn get_timings(&self) -> llama_perf_context_data {
        unsafe { llama_perf_context(self.context) }
    }

    /// Reset performance information.
    pub fn reset_timings(&mut self) {
        unsafe { llama_perf_context_reset(self.context) };
    }

    /// Set the number of threads used for generation and batch processing.
    pub fn set_n_threads(&mut self, n_gen: i32, n_batch: i32) {
        unsafe { llama_set_n_threads(self.context, n_gen, n_batch) }
    }

    /// Record that `seq_id` just decoded a media chunk whose positions
    /// run up to `head` (start + `n_pos`). An M-RoPE image's cells all
    /// sit at its start, so `pos_max + 1` falls short of the head; this
    /// lets [`Decoder::checkpoint_pos`] / [`Decoder::restore_to`] treat
    /// the image-end boundary as one. See `llama_cpp::checkpoint`.
    #[cfg_attr(not(feature = "mtmd"), allow(dead_code))]
    pub(crate) fn note_media_head(&mut self, seq_id: llama_seq_id, head: i32) {
        self.checkpoints.note_media(
            &mut ContextMemory(self.context),
            seq_id,
            head,
        );
    }

    /// Clear the KV cache, and every checkpoint with it.
    ///
    /// This and the other `memory_*` mutators take `&mut self` and keep
    /// the checkpoints in step with the KV: a partial checkpoint
    /// restores on top of the KV below it, so one left behind a change
    /// to that KV would load another history's window or recurrent
    /// state on the next [`Decoder::restore_to`]. They are what the
    /// [`Decoder`] impl calls; only the raw context pointer
    /// ([`Self::context_ptr_mut`], or [`Self::context_ptr`] cast to
    /// `*mut`, unsafe either way) reaches the memory around them.
    pub fn memory_clear(&mut self) {
        self.checkpoints.clear();
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_clear(mem, true) }
    }

    /// Remove KV entries for `seq_id` in position range `[p0, p1)`
    /// (negative bounds are unbounded; `seq_id < 0` matches every
    /// sequence). `false` when llama.cpp refuses the range.
    ///
    /// Drops the partial checkpoints above `p0` first — even when the
    /// range is refused: that costs at most a checkpoint, a stale one
    /// could cost #91.
    pub fn memory_seq_rm(
        &mut self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> bool {
        self.checkpoints.invalidate_from(seq_id, p0);
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_rm(mem, seq_id, p0, p1) }
    }

    /// Copy KV entries between sequences in `[p0, p1)`. Drops `dst`'s
    /// partial checkpoints above `p0`: its KV there is no longer the one
    /// they were taken over.
    pub fn memory_seq_cp(
        &mut self,
        src: llama_seq_id,
        dst: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) {
        self.checkpoints.invalidate_from(dst, p0);
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_cp(mem, src, dst, p0, p1) }
    }

    /// Keep only `seq_id`'s entries, drop all others — and every other
    /// sequence's partial checkpoints.
    pub fn memory_seq_keep(&mut self, seq_id: llama_seq_id) {
        self.checkpoints.keep_only(seq_id);
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_keep(mem, seq_id) }
    }

    /// Add `delta` to positions of `seq_id` in `[p0, p1)`. Drops the
    /// partial checkpoints above the lowest position a cell left or
    /// landed on — `p0 + delta` for a shift back, `p0` otherwise.
    pub fn memory_seq_add(
        &mut self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
    ) {
        self.checkpoints.invalidate_shift(seq_id, p0, delta);
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_add(mem, seq_id, p0, p1, delta) }
    }

    /// Integer-divide positions of `seq_id` in `[p0, p1)` by `d > 1`.
    /// Drops the partial checkpoints above `p0 / d`, the lowest
    /// position a moved cell lands on.
    pub fn memory_seq_div(
        &mut self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        d: i32,
    ) {
        self.checkpoints.invalidate_div(seq_id, p0, d);
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_div(mem, seq_id, p0, p1, d) }
    }

    /// Largest position present in KV for `seq_id`.
    pub fn memory_seq_pos_max(&self, seq_id: llama_seq_id) -> llama_pos {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_pos_max(mem, seq_id) }
    }

    /// Smallest position present in KV for `seq_id`, `-1` when empty.
    /// Above `0` once a sliding window's masked cells were recycled;
    /// on a hybrid model, the recurrent state's last position.
    pub fn memory_seq_pos_min(&self, seq_id: llama_seq_id) -> llama_pos {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_pos_min(mem, seq_id) }
    }

    /// Run one batch through `llama_decode`.
    ///
    /// Takes `&mut self` because it mutates C-side context state — the
    /// KV cache and the logits buffer. That is also what makes the
    /// borrows handed out by [`Self::logits`] / [`Self::embeddings`]
    /// sound: `llama_decode` reallocates the logits buffer when
    /// `n_outputs` grows (`output_reserve` frees `buf_output`), so a
    /// live slice must not survive across this call, and `&mut self`
    /// is what stops it.
    ///
    /// On error, check [`DecodeError::kv_dirty`] before trusting any
    /// position bookkeeping.
    pub fn decode(&mut self, batch: &Batch) -> Result<(), DecodeError> {
        // SAFETY: `self.context` is a live context (non-null since
        // construction, freed only in `Drop`); `batch.batch` is a
        // `llama_batch` owned by `Batch`, valid for the call and not
        // retained by llama.cpp.
        let ret = unsafe { llama_decode(self.context, batch.batch) };
        // llama.h:950-958. Codes 2 and < -1 leave processed ubatches
        // in the KV cache; 1 and -1 restore the pre-call state.
        match ret {
            0 => Ok(()),
            1 => Err(DecodeError::NoKvSlot),
            2 => Err(DecodeError::Aborted),
            -1 => Err(DecodeError::InvalidBatch),
            code if code < -1 => Err(DecodeError::Fatal { code }),
            code => Err(DecodeError::ErrorCode { code }),
        }
    }

    /// Decode `tokens` into the KV cache at positions
    /// `[start_pos, start_pos + tokens.len())` for `seq_id`.
    ///
    /// Resumable prefill primitive: does **not** clear the KV cache.
    /// Caller owns KV placement. Only the final token has logits
    /// enabled. Empty `tokens` is a no-op.
    pub fn prefill_inherent(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
        seq_id: llama_seq_id,
    ) -> Result<(), DecodeError> {
        if tokens.is_empty() {
            return Ok(());
        }
        // llama.cpp asserts `n_tokens_all <= cparams.n_batch` with a
        // GGML_ASSERT, which is *not* NDEBUG-gated — overflowing it
        // aborts the process instead of returning an error. Check on
        // our side so the signature's `Result` means something.
        let n_batch = self.n_batch();
        if tokens.len() > n_batch as usize {
            return Err(DecodeError::BatchTooLarge {
                n_tokens: tokens.len(),
                n_batch,
            });
        }
        let mut batch = Batch::new(tokens.len(), 0, 1)
            .expect("prefill batch allocation failed");
        let seq_ids = [seq_id];
        let last = tokens.len() - 1;
        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add_token(token, start_pos + i, Some(&seq_ids), i == last)
                .expect("prefill add_token failed (should be unreachable)");
        }
        self.decode(&batch)
    }

    /// Get logits for the i'th token of the most recent decode.
    ///
    /// The returned slice borrows the context's logits buffer, which
    /// [`Self::decode`] reallocates — `&self` here against `&mut self`
    /// there is what keeps that borrow sound.
    ///
    /// # Panics
    /// If `i` is not an output row of the last decode (i.e. the batch
    /// did not set `logits[i]`). llama.h:1009 documents a NULL return
    /// for invalid ids; debug builds of llama.cpp `GGML_ABORT` first,
    /// release builds return NULL and we panic here rather than
    /// building a slice over it.
    pub fn logits(&self, i: usize) -> &[f32] {
        let ptr = unsafe {
            llama_get_logits_ith(self.context, i.try_into().unwrap())
        };
        assert!(
            !ptr.is_null(),
            "llama_get_logits_ith({i}) returned NULL: no logits for \
             that row. The batch must set logits[{i}] = true before \
             decoding."
        );
        // SAFETY: non-null per the assert; llama.cpp guarantees
        // `n_vocab` contiguous floats per output row (the same stride
        // it uses internally, `model.vocab.n_tokens()`); the borrow is
        // tied to `&self` and `decode` takes `&mut self`.
        unsafe { std::slice::from_raw_parts(ptr, self.n_vocab) }
    }

    /// [`Self::logits`], rejecting a row that came back non-finite.
    ///
    /// This is the guard on the decode path — see
    /// [`DecodeError::NonFinite`] for why a NaN logit must be caught
    /// here rather than allowed to reach the sampler, and what tends to
    /// cause one.
    pub fn logits_checked(&self, i: usize) -> Result<&[f32], DecodeError> {
        let logits = self.logits(i);
        match first_non_finite(logits) {
            None => Ok(logits),
            Some((index, value)) => {
                Err(DecodeError::NonFinite { index, value })
            }
        }
    }

    /// Mutable logits for the i'th token.
    ///
    /// # Panics
    /// Same contract as [`Self::logits`].
    pub fn logits_mut(&mut self, i: i32) -> &mut [f32] {
        let ptr = unsafe { llama_get_logits_ith(self.context, i) };
        assert!(
            !ptr.is_null(),
            "llama_get_logits_ith({i}) returned NULL: no logits for \
             that row. The batch must set logits[{i}] = true before \
             decoding."
        );
        // SAFETY: as `logits`, and `&mut self` makes the mutable
        // aliasing exclusive.
        unsafe { std::slice::from_raw_parts_mut(ptr, self.n_vocab) }
    }

    /// Get embeddings for the i'th sequence.
    ///
    /// # Panics
    /// If the context was not created with embeddings enabled, or `i`
    /// is not an output row. Note that a plain generative context
    /// fails this on the *first* call — llama.cpp throws "no
    /// embeddings" and returns NULL when `embd.data` is unset.
    pub fn embeddings(&self, i: i32) -> &[f32] {
        let ptr = unsafe { llama_get_embeddings_ith(self.context, i) };
        assert!(
            !ptr.is_null(),
            "llama_get_embeddings_ith({i}) returned NULL: the context \
             has no embeddings. Create it with `embeddings = true`."
        );
        // SAFETY: non-null per the assert. Stride is `n_embd_out`, not
        // `n_embd` — llama-context.cpp advances by
        // `hparams.n_embd_out()`, and the two differ for models
        // publishing `%s.embedding_length_out`. The header comment
        // still says n_embd; the implementation is the contract.
        unsafe { std::slice::from_raw_parts(ptr, self.embedding_size_out) }
    }

    /// Mutable embeddings for the i'th sequence.
    ///
    /// # Panics
    /// Same contract as [`Self::embeddings`].
    pub fn embeddings_mut(&mut self, i: i32) -> &mut [f32] {
        let ptr = unsafe { llama_get_embeddings_ith(self.context, i) };
        assert!(
            !ptr.is_null(),
            "llama_get_embeddings_ith({i}) returned NULL: the context \
             has no embeddings. Create it with `embeddings = true`."
        );
        // SAFETY: as `embeddings`, with exclusive access via `&mut`.
        unsafe { std::slice::from_raw_parts_mut(ptr, self.embedding_size_out) }
    }
}

impl Drop for LlamaCppDecoder {
    fn drop(&mut self) {
        // Teardown order is airtight without depending on field
        // order: a manual `Drop::drop` runs to completion *before* any
        // field is dropped, so the context is freed here and only then
        // is the `model` handle released (freeing the weights if this
        // was the last handle).
        //
        // `llama_free` deliberately runs *outside* the guard: holding
        // it across context teardown would let a concurrent `new`
        // race init against free.
        unsafe { llama_free(self.context) };
        let mut count = engine_count();
        *count -= 1;
        if *count == 0 {
            unsafe { llama_backend_free() };
        }
    }
}

// llama.cpp-backed [`Decoder`] trait impl. `step` allocates a 1-slot
// `Batch` each call; `prefill` wraps the inherent `prefill_inherent`
// and reads `logits(tokens.len() - 1)`.
impl Decoder for LlamaCppDecoder {
    type Error = DecodeError;

    fn prefill(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
        seq_id: i32,
    ) -> Result<&[f32], Self::Error> {
        LlamaCppDecoder::prefill_inherent(self, tokens, start_pos, seq_id)?;
        if tokens.is_empty() {
            Ok(&[])
        } else {
            self.logits_checked(tokens.len() - 1)
        }
    }

    fn step(
        &mut self,
        token: Token,
        pos: usize,
        seq_id: i32,
    ) -> Result<&[f32], Self::Error> {
        let mut batch =
            Batch::new(1, 0, 1).expect("step batch allocation failed");
        let seq_ids = [seq_id];
        batch
            .add_token(token, pos, Some(&seq_ids), true)
            .expect("step add_token failed (should be unreachable)");
        self.decode(&batch)?;
        self.logits_checked(0)
    }

    fn n_ctx(&self) -> u32 {
        LlamaCppDecoder::n_ctx(self)
    }

    fn n_seq_max(&self) -> u32 {
        unsafe { llama_n_seq_max(self.context) }
    }

    /// [`LlamaCppDecoder::memory_clear`]: the checkpoints go with the
    /// KV.
    fn memory_clear(&mut self) {
        LlamaCppDecoder::memory_clear(self);
    }

    /// [`LlamaCppDecoder::memory_seq_rm`]: also drops the partial
    /// checkpoints above `p0`.
    fn memory_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool {
        LlamaCppDecoder::memory_seq_rm(self, seq_id, p0, p1)
    }

    /// [`LlamaCppDecoder::memory_seq_cp`]: also drops `dst`'s partial
    /// checkpoints above `p0`.
    fn memory_seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) {
        LlamaCppDecoder::memory_seq_cp(self, src, dst, p0, p1);
    }

    /// [`LlamaCppDecoder::memory_seq_keep`]: also drops every other
    /// sequence's partial checkpoints.
    fn memory_seq_keep(&mut self, seq_id: i32) {
        LlamaCppDecoder::memory_seq_keep(self, seq_id);
    }

    fn memory_seq_pos_max(&mut self, seq_id: i32) -> i32 {
        LlamaCppDecoder::memory_seq_pos_max(self, seq_id)
    }

    /// Checkpoint the sequence at `pos`, its head, in the model's
    /// [`Checkpointing`] mode: a no-op on a dense model, where a
    /// truncate is already a lossless rewind to any position; the
    /// sliding-window cells or recurrent state otherwise — see
    /// `llama_cpp::checkpoint`.
    fn checkpoint_pos(&mut self, seq_id: i32, pos: i32) {
        self.checkpoints.checkpoint(
            &mut ContextMemory(self.context),
            seq_id,
            pos,
        );
    }

    /// Rewind `seq_id` to `pos`: the plain KV truncate when that alone
    /// leaves the state the model had at `pos` — always on a dense
    /// model; on a sliding-window one while the window below `pos`
    /// survives — else the checkpoint stored at `pos`, loaded under the
    /// truncate. Checkpoints above `pos` are dropped per the trait
    /// contract.
    ///
    /// Media (#31): M-RoPE images break position density — all
    /// ~n_tokens cells share the chunk's start position and positions
    /// (start, start + n_pos) are a gap — so `pos_max + 1` is not the
    /// head after one. The vision path reports each chunk it decodes
    /// (`note_media_head`), and the image-end boundary then
    /// checkpoints and rewinds like a text one (validated by
    /// `mtmd::tests::mrope_kv_semantics_probe`). Should that report be
    /// lost, the boundary fails closed: a checkpoint / full-reprefill
    /// fallback, never corruption.
    fn restore_to(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        self.checkpoints
            .restore(&mut ContextMemory(self.context), seq_id, pos)
    }

    /// `true` on a dense model ([`Checkpointing::Off`]) for any
    /// position the sequence holds — what [`Self::restore_to`] then
    /// does is the plain truncate.
    fn truncate_restores(&mut self, seq_id: i32, pos: i32) -> bool {
        self.checkpoints.truncate_restores(
            &mut ContextMemory(self.context),
            seq_id,
            pos,
        )
    }

    /// Drop the checkpoint at `(seq_id, pos)`, if one exists.
    /// Idempotent; trivially `Ok` when nothing is stored.
    fn forget_pos(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        self.checkpoints.forget(seq_id, pos);
        Ok(())
    }
}

/// A llama.cpp context's sequence memory, as
/// [`Checkpoints`](crate::llama_cpp::checkpoint) drives it. Borrows
/// nothing: it is the decoder's context pointer, so the decoder can
/// lend it out while it mutates its own checkpoint store.
#[derive(Clone, Copy)]
struct ContextMemory(*mut llama_context);

impl ContextMemory {
    fn flags(partial: bool) -> llama_state_seq_flags {
        if partial {
            STATE_SEQ_PARTIAL_ONLY
        } else {
            0
        }
    }

    fn state_size(self, seq: i32, flags: llama_state_seq_flags) -> usize {
        // SAFETY: the context is live for the decoder's lifetime, and
        // llama.cpp only reads it here.
        unsafe { llama_state_seq_get_size_ext(self.0, seq, flags) }
    }

    /// The serialized state; empty when llama.cpp fails to write it.
    fn state(self, seq: i32, flags: llama_state_seq_flags) -> Vec<u8> {
        let len = self.state_size(seq, flags);
        let mut buf = vec![0u8; len];
        // SAFETY: `buf` is `len` writable bytes, which llama.cpp never
        // writes past (it returns 0 instead).
        let copied = unsafe {
            llama_state_seq_get_data_ext(
                self.0,
                buf.as_mut_ptr(),
                len,
                seq,
                flags,
            )
        };
        buf.truncate(if copied == len { len } else { 0 });
        buf
    }

    fn load_state(
        self,
        state: &[u8],
        seq: i32,
        flags: llama_state_seq_flags,
    ) -> bool {
        // SAFETY: `state` is a valid slice for the call; llama.cpp
        // reads at most `state.len()` bytes of it.
        let read = unsafe {
            llama_state_seq_set_data_ext(
                self.0,
                state.as_ptr(),
                state.len(),
                seq,
                flags,
            )
        };
        read != 0
    }
}

impl SeqMemory for ContextMemory {
    fn seq_rm(&mut self, seq: i32, p0: i32, p1: i32) -> bool {
        // SAFETY: as for every `memory_*` call on the decoder.
        unsafe { llama_memory_seq_rm(llama_get_memory(self.0), seq, p0, p1) }
    }

    fn seq_pos_min(&mut self, seq: i32) -> i32 {
        // SAFETY: as above.
        unsafe { llama_memory_seq_pos_min(llama_get_memory(self.0), seq) }
    }

    fn seq_pos_max(&mut self, seq: i32) -> i32 {
        // SAFETY: as above.
        unsafe { llama_memory_seq_pos_max(llama_get_memory(self.0), seq) }
    }

    fn save(&mut self, seq: i32, partial: bool) -> Vec<u8> {
        self.state(seq, Self::flags(partial))
    }

    fn load(&mut self, seq: i32, bytes: &[u8], partial: bool) -> bool {
        self.load_state(bytes, seq, Self::flags(partial))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_cpp::LlamaCppEngine;

    const PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

    /// The NaN guard's pure core, pinned model-free. The fast path must
    /// pass a clean slice through, and both flavours of non-finite must
    /// be caught — `is_nan()` alone would let an `inf` logit reach the
    /// sampler, where it is just as wrong and much quieter (it sorts,
    /// it just always wins).
    #[test]
    fn first_non_finite_finds_both_flavours() {
        assert_eq!(first_non_finite(&[]), None);
        assert_eq!(first_non_finite(&[-1.0, 0.0, 21.5]), None);

        let (i, v) = first_non_finite(&[1.0, f32::NAN, 3.0]).expect("nan");
        assert_eq!(i, 1);
        assert!(v.is_nan(), "{v}");

        assert_eq!(
            first_non_finite(&[1.0, 2.0, f32::INFINITY]),
            Some((2, f32::INFINITY))
        );
        assert_eq!(
            first_non_finite(&[f32::NEG_INFINITY, 2.0]),
            Some((0, f32::NEG_INFINITY))
        );

        // The whole-slice case: the Mistral-on-Metal shape, where every
        // logit is NaN. The reported index is the first, not the last.
        let all_nan = [f32::NAN; 8];
        let (i, _) = first_non_finite(&all_nan).expect("nan");
        assert_eq!(i, 0);
    }

    /// A non-finite decode leaves poisoned cells behind, so callers
    /// must wipe rather than reuse — same contract as `Aborted`.
    #[test]
    fn non_finite_is_kv_dirty() {
        assert!(DecodeError::NonFinite {
            index: 0,
            value: f32::NAN
        }
        .kv_dirty());
    }

    fn model() -> LlamaCppModel {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/model.gguf");
        LlamaCppModel::from_file(path, None).expect("model load failed")
    }

    /// A decoder keeps its own handle on the model, so the context
    /// stays valid after the caller drops the handle it built from
    /// (issue #54). Before the model became refcounted this was a
    /// use-after-free: `llama_context` holds `const llama_model &` for
    /// its whole life, and dropping the last `LlamaCppModel` freed the
    /// weights out from under it.
    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn decoder_outlives_model_handle() {
        let model = model();
        let mut decoder = LlamaCppDecoder::new(
            &model,
            LlamaCppEngine::default_context_params(),
            None,
        )
        .expect("context creation failed");

        // Consume the caller's handle entirely; only the decoder's
        // remains. `into_raw` returning `None` is the *deterministic*
        // half of this test — it proves the decoder holds a strong
        // reference of its own. The prefill below is the behavioural
        // half, but a use-after-free is UB and is not guaranteed to
        // manifest as a crash we could assert on, so it cannot carry
        // the test alone.
        assert!(
            model.into_raw().is_none(),
            "the decoder does not hold a handle on the model — its \
             context can outlive the weights (issue #54)",
        );

        let tokens = decoder.model().tokenize(PROMPT, true);
        assert!(tokens.len() >= 4, "prompt tokenization too short");
        decoder
            .prefill_inherent(&tokens, 0, 0)
            .expect("prefill failed");

        let logits = decoder.logits(tokens.len() - 1);
        assert_eq!(logits.len(), decoder.n_vocab());
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "non-finite logits: the context is reading freed weights",
        );
    }

    /// Two contexts over one copy of the weights — what the refcounted
    /// handle buys beyond the safety fix. Loading the GGUF twice would
    /// cost two full weight allocations; cloning the handle costs each
    /// decoder only its own KV cache. Also exercises `ENGINE_COUNT`
    /// with two live contexts (backend init once, freed on the last
    /// drop).
    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    fn two_decoders_share_one_model() {
        let model = model();
        let params = LlamaCppEngine::default_context_params();
        let mut a = LlamaCppDecoder::new(&model, params, None)
            .expect("first context creation failed");
        let mut b = LlamaCppDecoder::new(&model, params, None)
            .expect("second context creation failed");

        let tokens = model.tokenize(PROMPT, true);
        a.prefill_inherent(&tokens, 0, 0).expect("prefill a failed");
        b.prefill_inherent(&tokens, 0, 0).expect("prefill b failed");

        // Same weights, same prompt, same positions: the two contexts
        // must agree. (Same-backend, same-process, so this is an
        // equality check, not a tolerance one.)
        assert_eq!(
            a.logits(tokens.len() - 1),
            b.logits(tokens.len() - 1),
            "two contexts over one model disagreed on the same prompt",
        );
    }
}
