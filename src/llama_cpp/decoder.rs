use crate::{
    backend::{Decoder, MemoryRmError},
    snapshot_store::SnapshotStore,
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
    llama_memory_seq_pos_max, llama_memory_seq_rm, llama_model_is_hybrid,
    llama_model_is_recurrent, llama_model_n_embd_out, llama_n_batch,
    llama_n_ctx, llama_n_seq_max, llama_numa_init, llama_perf_context,
    llama_perf_context_data, llama_perf_context_reset, llama_pos, llama_seq_id,
    llama_set_n_threads, llama_state_get_data, llama_state_get_size,
    llama_state_seq_get_data, llama_state_seq_get_size,
    llama_state_seq_set_data, llama_state_set_data,
};

use thiserror::Error;

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
#[derive(Error, Debug)]
pub enum NewError {
    #[error("Could not load model from file: {path}")]
    Model { path: PathBuf },
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

static_assertions::assert_impl_all!(NewError: Send, Sync);

/// Possible errors when calling [`LlamaCppDecoder::decode`].
#[derive(Error, Debug)]
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
            Self::Aborted | Self::Fatal { .. } | Self::ErrorCode { .. } => true,
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
    /// Host-RAM per-sequence state snapshots backing
    /// [`Decoder::checkpoint_pos`] / [`Decoder::restore_to`]. Only
    /// populated when [`Self::seq_snapshots_enabled`].
    seq_snapshots: SnapshotStore,
    /// Whether [`Decoder::checkpoint_pos`] takes real snapshots.
    /// Defaults to `llama_model_is_recurrent || llama_model_is_hybrid`
    /// at construction: pure-attention KV truncates losslessly at any
    /// position, so snapshots are redundant there, but recurrent /
    /// hybrid layer state cannot be unwound by position and needs
    /// them. Force on via [`Self::set_seq_snapshots`] (tests, or
    /// callers wanting rewind insurance on attention models).
    seq_snapshots_enabled: bool,
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

        // Recurrent / hybrid layer state (Mamba-style SSM, RWKV, ...)
        // cannot be rewound by KV-position truncation, so those
        // architectures need real snapshots at cache breakpoints.
        let needs_snapshots = unsafe {
            llama_model_is_recurrent(model.as_ptr())
                || llama_model_is_hybrid(model.as_ptr())
        };

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
            seq_snapshots: SnapshotStore::default(),
            seq_snapshots_enabled: needs_snapshots,
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

    /// Raw pointer to the underlying llama.cpp context (const).
    pub fn context_ptr(&self) -> *const llama_context {
        self.context
    }

    /// Raw pointer to the underlying llama.cpp context (mut).
    ///
    /// Takes `&mut self` so the exclusivity the pointer implies is
    /// actually held — the same reason [`Self::decode`] does.
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
        let read = unsafe {
            llama_state_set_data(self.context, state.as_ptr(), state.len())
        };
        assert_eq!(read, state.len(), "llama.cpp rejected saved state");
    }

    /// Size of the serialized state for a single sequence.
    pub fn state_seq_size(&self, seq_id: llama_seq_id) -> usize {
        unsafe { llama_state_seq_get_size(self.context, seq_id) }
    }

    /// Serialize the state of a single sequence (its KV cells plus any
    /// recurrent layer state). The bytes restore via
    /// [`Self::set_state_seq`] — into this context or another one on
    /// the same model.
    pub fn get_state_seq(&self, seq_id: llama_seq_id) -> Vec<u8> {
        let len = self.state_seq_size(seq_id);
        let mut buf = vec![0u8; len];
        let copied = unsafe {
            llama_state_seq_get_data(
                self.context,
                buf.as_mut_ptr(),
                len,
                seq_id,
            )
        };
        assert_eq!(copied, len);
        buf
    }

    /// Restore a single sequence's state from bytes produced by
    /// [`Self::get_state_seq`], loading them as `dest_seq_id`. Returns
    /// `false` if llama.cpp rejects the payload (wrong model, corrupt
    /// bytes, insufficient KV room) — the destination sequence is left
    /// cleared in that case.
    pub fn set_state_seq(
        &mut self,
        state: &[u8],
        dest_seq_id: llama_seq_id,
    ) -> bool {
        let copied = unsafe {
            llama_state_seq_set_data(
                self.context,
                state.as_ptr(),
                state.len(),
                dest_seq_id,
            )
        };
        copied != 0
    }

    /// Whether [`Decoder::checkpoint_pos`] takes real per-sequence
    /// snapshots (recurrent / hybrid models: on by default; pure
    /// attention: off, truncation already rewinds losslessly).
    pub fn seq_snapshots_enabled(&self) -> bool {
        self.seq_snapshots_enabled
    }

    /// Force per-sequence snapshotting on or off. Disabling drops all
    /// stored snapshots.
    pub fn set_seq_snapshots(&mut self, enabled: bool) {
        self.seq_snapshots_enabled = enabled;
        if !enabled {
            self.seq_snapshots.clear();
        }
    }

    /// Number of per-sequence snapshots currently held.
    pub fn seq_snapshot_count(&self) -> usize {
        self.seq_snapshots.len()
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

    /// Clear the KV cache.
    pub fn memory_clear(&self) {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_clear(mem, true) }
    }

    /// Remove KV entries for `seq_id` in position range `[p0, p1)`.
    pub fn memory_seq_rm(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> bool {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_rm(mem, seq_id, p0, p1) }
    }

    /// Copy KV entries between sequences in `[p0, p1)`.
    pub fn memory_seq_cp(
        &self,
        src: llama_seq_id,
        dst: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_cp(mem, src, dst, p0, p1) }
    }

    /// Keep only `seq_id`'s entries, drop all others.
    pub fn memory_seq_keep(&self, seq_id: llama_seq_id) {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_keep(mem, seq_id) }
    }

    /// Add `delta` to positions of `seq_id` in `[p0, p1)`.
    pub fn memory_seq_add(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
    ) {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_add(mem, seq_id, p0, p1, delta) }
    }

    /// Integer-divide positions of `seq_id` in `[p0, p1)` by `d > 1`.
    pub fn memory_seq_div(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        d: i32,
    ) {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_div(mem, seq_id, p0, p1, d) }
    }

    /// Largest position present in KV for `seq_id`.
    pub fn memory_seq_pos_max(&self, seq_id: llama_seq_id) -> llama_pos {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_pos_max(mem, seq_id) }
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
            Ok(self.logits(tokens.len() - 1))
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
        Ok(self.logits(0))
    }

    fn n_ctx(&self) -> u32 {
        LlamaCppDecoder::n_ctx(self)
    }

    fn n_seq_max(&self) -> u32 {
        unsafe { llama_n_seq_max(self.context) }
    }

    fn memory_clear(&mut self) {
        LlamaCppDecoder::memory_clear(self);
        // Session clears on full re-prefill; the old positions are
        // never referenced again, so free the snapshots with the KV.
        self.seq_snapshots.clear();
    }

    fn memory_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool {
        LlamaCppDecoder::memory_seq_rm(self, seq_id, p0, p1)
    }

    fn memory_seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) {
        LlamaCppDecoder::memory_seq_cp(self, src, dst, p0, p1);
    }

    fn memory_seq_keep(&mut self, seq_id: i32) {
        LlamaCppDecoder::memory_seq_keep(self, seq_id);
    }

    fn memory_seq_pos_max(&mut self, seq_id: i32) -> i32 {
        LlamaCppDecoder::memory_seq_pos_max(self, seq_id)
    }

    /// Snapshot the sequence state at `pos` when
    /// [`seq_snapshots_enabled`](LlamaCppDecoder::seq_snapshots_enabled)
    /// — required for recurrent / hybrid models, whose layer state
    /// cannot be rewound by KV truncation. A no-op for pure-attention
    /// models (the default there), where truncation is already a
    /// lossless rewind to any position.
    fn checkpoint_pos(&mut self, seq_id: i32, pos: i32) {
        if !self.seq_snapshots_enabled {
            return;
        }
        let bytes = self.get_state_seq(seq_id);
        self.seq_snapshots.insert((seq_id, pos), bytes);
    }

    /// Rewind `seq_id` to `pos`. Tries the plain KV truncate
    /// (`llama_memory_seq_rm(seq_id, pos, -1)`) first — lossless and
    /// copy-free on attention models. llama.cpp refuses partial-range
    /// removal on recurrent / hybrid memory, and then a stored
    /// snapshot (if any) is reloaded instead: the sequence is dropped
    /// wholesale (whole-sequence removal is supported everywhere) and
    /// re-populated via `llama_state_seq_set_data`. Either way,
    /// snapshots at positions `> pos` are dropped per the trait
    /// contract.
    fn restore_to(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        // The pos_max check catches truncates that "succeed" without
        // the sequence actually holding [0, pos) — removing an empty
        // range is a success to llama.cpp, but reporting it as a
        // lossless rewind would resume generation over missing KV.
        //
        // Position-density caveat (media, #31): the check assumes a
        // cell exists at `pos - 1`. M-RoPE images break density —
        // all ~n_tokens cells share the chunk's start position and
        // positions (start, start + n_pos) are a gap — so a truncate
        // to a boundary just past an M-RoPE image sees `pos_max ==
        // image_start != pos - 1` and fails closed here even though
        // the prefix is intact (validated by
        // `mtmd::tests::mrope_kv_semantics_probe`). That is
        // acceptable: Session boundaries land in text (breakpoints
        // are message-granular and message-close text follows every
        // image), and a false failure only costs the snapshot /
        // full-reprefill fallback, never correctness.
        if LlamaCppDecoder::memory_seq_rm(self, seq_id, pos, -1)
            && LlamaCppDecoder::memory_seq_pos_max(self, seq_id) == pos - 1
        {
            self.seq_snapshots.invalidate_after(seq_id, pos);
            return Ok(());
        }
        let Some(bytes) = self.seq_snapshots.take((seq_id, pos)) else {
            return Err(MemoryRmError::NoCheckpoint { pos });
        };
        LlamaCppDecoder::memory_seq_rm(self, seq_id, -1, -1);
        if self.set_state_seq(&bytes, seq_id) {
            // Still valid — the snapshot survives its own restore so
            // Session can rewind to the same breakpoint repeatedly.
            self.seq_snapshots.insert((seq_id, pos), bytes);
            self.seq_snapshots.invalidate_after(seq_id, pos);
            Ok(())
        } else {
            // llama.cpp rejected bytes we serialized ourselves — a
            // llama.cpp-internal inconsistency. The sequence is left
            // cleared; BackendUnsupported routes Session to its
            // memory_clear + full-reprefill fallback.
            Err(MemoryRmError::BackendUnsupported { pos })
        }
    }

    /// Drop the snapshot at `(seq_id, pos)`, if one exists. Idempotent;
    /// a no-op (and trivially `Ok`) when snapshotting is disabled.
    fn forget_pos(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        self.seq_snapshots.forget((seq_id, pos));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_cpp::LlamaCppEngine;

    const PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

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
