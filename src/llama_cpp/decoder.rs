use crate::{
    backend::{Decoder, MemoryRmError},
    Batch, LlamaCppModel, Token,
};

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::Mutex,
};

use llama_cpp_sys_3::{
    ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED, llama_backend_free,
    llama_backend_init, llama_context, llama_context_params, llama_decode,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED, llama_free,
    llama_get_embeddings_ith, llama_get_logits_ith, llama_get_memory,
    llama_memory_clear, llama_memory_seq_add, llama_memory_seq_cp,
    llama_memory_seq_div, llama_memory_seq_keep, llama_memory_seq_pos_max,
    llama_memory_seq_rm, llama_model_is_hybrid, llama_model_is_recurrent,
    llama_n_batch, llama_n_ctx, llama_new_context_with_model, llama_numa_init,
    llama_perf_context, llama_perf_context_data, llama_perf_context_reset,
    llama_pos, llama_seq_id, llama_set_n_threads, llama_state_get_data,
    llama_state_get_size, llama_state_seq_get_data, llama_state_seq_get_size,
    llama_state_seq_set_data, llama_state_set_data,
};

use thiserror::Error;

/// Global engine count. When this drops to 0, the llama backend is freed in
/// the last [`LlamaCppDecoder`]'s `Drop` implementation.
pub(super) static ENGINE_COUNT: Mutex<usize> = Mutex::new(0);

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
    #[error("`llama_decode` returned an error code: {code}")]
    ErrorCode { code: i32 },
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

/// Max host-RAM sequence-state snapshots retained per decoder.
/// Anthropic's cache budget is 4 explicit breakpoints; `Session` adds
/// one internal tip. 16 leaves generous slack for multi-sequence use
/// before the LRU starts evicting.
const MAX_SEQ_SNAPSHOTS: usize = 16;

/// Bounded LRU of serialized per-sequence decoder states, keyed by
/// `(seq_id, pos)`. Pure bookkeeping — FFI-free, so the eviction /
/// invalidation logic is unit-testable without a model.
///
/// A stored value is the *entire* serialized state for that sequence
/// (`llama_state_seq_get_data` wire format), taken when the sequence
/// held exactly positions `[0, pos)`. Restoring one replaces the
/// sequence wholesale, so entries stay restorable regardless of later
/// KV mutations; invalidation exists to keep the trait's rewind
/// semantics ("futures are dropped") uniform across backends, not
/// because the bytes go stale.
#[derive(Debug, Default)]
struct SnapshotStore {
    map: HashMap<(llama_seq_id, llama_pos), Vec<u8>>,
    /// Insertion order, oldest first. Re-inserting an existing key
    /// refreshes its position.
    order: VecDeque<(llama_seq_id, llama_pos)>,
}

impl SnapshotStore {
    /// Insert (or replace) the snapshot at `key`, evicting the oldest
    /// entries beyond [`MAX_SEQ_SNAPSHOTS`].
    fn insert(&mut self, key: (llama_seq_id, llama_pos), bytes: Vec<u8>) {
        if self.map.insert(key, bytes).is_some() {
            self.order.retain(|k| *k != key);
        }
        self.order.push_back(key);
        while self.map.len() > MAX_SEQ_SNAPSHOTS {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.map.remove(&oldest);
        }
    }

    /// Remove and return the snapshot at `key`, if any.
    fn take(&mut self, key: (llama_seq_id, llama_pos)) -> Option<Vec<u8>> {
        let bytes = self.map.remove(&key)?;
        self.order.retain(|k| *k != key);
        Some(bytes)
    }

    /// Drop the snapshot at `key`. Idempotent.
    fn forget(&mut self, key: (llama_seq_id, llama_pos)) {
        if self.map.remove(&key).is_some() {
            self.order.retain(|k| *k != key);
        }
    }

    /// Drop every snapshot on `seq_id` at positions strictly greater
    /// than `pos` — the "futures are invalid after a rewind" rule from
    /// [`Decoder::restore_to`].
    fn invalidate_after(&mut self, seq_id: llama_seq_id, pos: llama_pos) {
        self.map.retain(|&(s, p), _| s != seq_id || p <= pos);
        self.order.retain(|&(s, p)| s != seq_id || p <= pos);
    }

    /// Drop everything.
    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

/// llama.cpp-backed decoder: owns a `llama_context`, manages the KV
/// cache, runs decode passes, and exposes logits / embeddings.
///
/// Implements [`crate::backend::Decoder`]. `LlamaCppDecoder::new`
/// handles backend lifecycle (`llama_backend_init` + `llama_numa_init`
/// on the first-ever decoder; `llama_backend_free` on the last
/// dropped). `n_vocab` and `embedding_size` are cached at construction
/// so the decoder can produce correctly-sized slices without holding a
/// reference to the [`LlamaCppModel`] that produced it.
#[derive(Debug)]
pub struct LlamaCppDecoder {
    pub(crate) context: *mut llama_context,
    /// Cached vocab size from the source model — used to size logit slices.
    n_vocab: usize,
    /// Cached embedding dimension from the source model — used to size
    /// embedding slices.
    embedding_size: usize,
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
    /// Handles the llama.cpp backend lifecycle: on the first-ever
    /// decoder (`ENGINE_COUNT` transitions 0→1) runs
    /// `llama_backend_init` + `llama_numa_init`. Subsequent decoders
    /// just increment the count.
    ///
    /// If context creation fails, the count is rolled back (and the
    /// backend torn down if we were the first). The caller can
    /// retry without double-init.
    pub fn new(
        model: &mut LlamaCppModel,
        context_params: llama_context_params,
        numa_strategy: Option<u32>,
    ) -> Result<Self, NewError> {
        {
            let mut count = ENGINE_COUNT.lock().unwrap();
            *count += 1;
            if *count == 1 {
                unsafe {
                    llama_backend_init();
                    llama_numa_init(
                        numa_strategy
                            .unwrap_or(
                                ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED
                                    .try_into()
                                    .unwrap(),
                            )
                            .try_into()
                            .unwrap(),
                    );
                }
            }
        }

        let context = unsafe {
            llama_new_context_with_model(model.as_ptr_mut(), context_params)
        };
        if context.is_null() {
            // Roll back the count we just reserved.
            let mut count = ENGINE_COUNT.lock().unwrap();
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
            seq_snapshots: SnapshotStore::default(),
            seq_snapshots_enabled: needs_snapshots,
        })
    }

    /// Raw pointer to the underlying llama.cpp context (const).
    pub fn context_ptr(&self) -> *const llama_context {
        self.context
    }

    /// Raw pointer to the underlying llama.cpp context (mut).
    pub fn context_ptr_mut(&self) -> *mut llama_context {
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
    pub fn decode(&self, batch: &Batch) -> Result<(), DecodeError> {
        let ret = unsafe { llama_decode(self.context, batch.batch) };
        match ret {
            0 => Ok(()),
            1 => Err(DecodeError::NoKvSlot),
            _ => Err(DecodeError::ErrorCode { code: ret }),
        }
    }

    /// Decode `tokens` into the KV cache at positions
    /// `[start_pos, start_pos + tokens.len())` for `seq_id`.
    ///
    /// Resumable prefill primitive: does **not** clear the KV cache.
    /// Caller owns KV placement. Only the final token has logits
    /// enabled. Empty `tokens` is a no-op.
    pub fn prefill_inherent(
        &self,
        tokens: &[Token],
        start_pos: usize,
        seq_id: llama_seq_id,
    ) -> Result<(), DecodeError> {
        if tokens.is_empty() {
            return Ok(());
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
    /// # Panics
    /// - If the index is invalid (panics come from the C side).
    pub fn logits(&self, i: usize) -> &[f32] {
        let ptr = unsafe {
            llama_get_logits_ith(self.context, i.try_into().unwrap())
        };
        unsafe { std::slice::from_raw_parts(ptr, self.n_vocab) }
    }

    /// Mutable logits for the i'th token.
    pub fn logits_mut(&mut self, i: i32) -> &mut [f32] {
        let ptr = unsafe { llama_get_logits_ith(self.context, i) };
        unsafe { std::slice::from_raw_parts_mut(ptr, self.n_vocab) }
    }

    /// Get embeddings for the i'th sequence.
    pub fn embeddings(&self, i: i32) -> &[f32] {
        let ptr = unsafe { llama_get_embeddings_ith(self.context, i) };
        unsafe { std::slice::from_raw_parts(ptr, self.embedding_size) }
    }

    /// Mutable embeddings for the i'th sequence.
    pub fn embeddings_mut(&mut self, i: i32) -> &mut [f32] {
        let ptr = unsafe { llama_get_embeddings_ith(self.context, i) };
        unsafe { std::slice::from_raw_parts_mut(ptr, self.embedding_size) }
    }
}

impl Drop for LlamaCppDecoder {
    fn drop(&mut self) {
        unsafe { llama_free(self.context) };
        let mut count = ENGINE_COUNT.lock().unwrap();
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

    fn store_with(keys: &[(i32, i32)]) -> SnapshotStore {
        let mut s = SnapshotStore::default();
        for &k in keys {
            s.insert(k, vec![0u8; 4]);
        }
        s
    }

    #[test]
    fn snapshot_store_insert_take_forget_roundtrip() {
        let mut s = SnapshotStore::default();
        s.insert((0, 10), vec![1, 2, 3]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.take((0, 10)), Some(vec![1, 2, 3]));
        assert_eq!(s.len(), 0);
        assert_eq!(s.take((0, 10)), None);
        // forget is idempotent on absent keys.
        s.forget((0, 10));
        s.forget((0, 10));
    }

    #[test]
    fn snapshot_store_replace_refreshes_lru_position() {
        // Fill to capacity, then re-insert the oldest key. The next
        // eviction must hit the second-oldest, not the refreshed one.
        let keys: Vec<(i32, i32)> =
            (0..MAX_SEQ_SNAPSHOTS as i32).map(|i| (0, i)).collect();
        let mut s = store_with(&keys);
        s.insert((0, 0), vec![9]); // refresh oldest
        s.insert((0, 999), vec![8]); // overflow → evict (0, 1)
        assert_eq!(s.len(), MAX_SEQ_SNAPSHOTS);
        assert_eq!(s.take((0, 1)), None, "second-oldest should be evicted");
        assert_eq!(s.take((0, 0)), Some(vec![9]), "refreshed key survives");
    }

    #[test]
    fn snapshot_store_evicts_oldest_beyond_cap() {
        let keys: Vec<(i32, i32)> = (0..(MAX_SEQ_SNAPSHOTS as i32 + 3))
            .map(|i| (0, i))
            .collect();
        let mut s = store_with(&keys);
        assert_eq!(s.len(), MAX_SEQ_SNAPSHOTS);
        // The three oldest are gone; the newest three are present.
        assert_eq!(s.take((0, 0)), None);
        assert_eq!(s.take((0, 1)), None);
        assert_eq!(s.take((0, 2)), None);
        assert!(s.take((0, MAX_SEQ_SNAPSHOTS as i32 + 2)).is_some());
    }

    #[test]
    fn snapshot_store_invalidate_after_is_per_sequence() {
        let mut s = store_with(&[(0, 5), (0, 10), (0, 20), (1, 15)]);
        s.invalidate_after(0, 10);
        // (0, 20) dropped: strictly greater than pos on seq 0.
        assert_eq!(s.take((0, 20)), None);
        // (0, 10) kept: boundary is inclusive.
        assert!(s.take((0, 10)).is_some());
        assert!(s.take((0, 5)).is_some());
        // Other sequences untouched.
        assert!(s.take((1, 15)).is_some());
    }

    #[test]
    fn snapshot_store_clear_empties_map_and_order() {
        let mut s = store_with(&[(0, 1), (0, 2)]);
        s.clear();
        assert_eq!(s.len(), 0);
        // Insert after clear must not resurrect stale order entries.
        s.insert((0, 3), vec![1]);
        assert_eq!(s.len(), 1);
        assert!(s.take((0, 3)).is_some());
    }
}
