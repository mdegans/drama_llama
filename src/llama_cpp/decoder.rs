use crate::{
    backend::{Decoder, MemoryRmError},
    Batch, LlamaCppModel, Token,
};

use std::{path::PathBuf, sync::Mutex};

use llama_cpp_sys_3::{
    ggml_log_callback, ggml_log_level, ggml_log_set,
    ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED, llama_backend_free,
    llama_backend_init, llama_context, llama_context_params, llama_decode,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED, llama_free,
    llama_get_embeddings_ith, llama_get_logits_ith, llama_get_memory,
    llama_log_set, llama_memory_clear, llama_memory_seq_add,
    llama_memory_seq_cp, llama_memory_seq_div, llama_memory_seq_keep,
    llama_memory_seq_pos_max, llama_memory_seq_rm,
    llama_n_batch, llama_n_ctx, llama_new_context_with_model, llama_numa_init,
    llama_perf_context, llama_perf_context_data, llama_perf_context_reset,
    llama_pos, llama_seq_id, llama_set_n_threads, llama_state_get_data,
    llama_state_get_size, llama_state_set_data,
};

use thiserror::Error;

/// Global engine count. When this drops to 0, the llama backend is freed in
/// the last [`LlamaCppDecoder`]'s `Drop` implementation.
pub(super) static ENGINE_COUNT: Mutex<usize> = Mutex::new(0);

/// Silence `llama.cpp` + `ggml` log output.
///
/// Installs a no-op callback on both loggers. llama.cpp and ggml maintain
/// separate log sinks — Metal pipeline compile chatter comes from the ggml
/// side, model-load prose from the llama side — so both need to be hushed
/// for quiet generation.
///
/// Idempotent. Safe to call before or after creating an
/// [`crate::Engine`]. Call [`restore_default_logs`] to undo.
pub fn silence_logs() {
    unsafe {
        llama_log_set(Some(discard_log), std::ptr::null_mut());
        ggml_log_set(Some(discard_log), std::ptr::null_mut());
    }
}

/// Restore default (stderr) logging. Inverse of [`silence_logs`].
pub fn restore_default_logs() {
    unsafe {
        llama_log_set(None, std::ptr::null_mut());
        ggml_log_set(None, std::ptr::null_mut());
    }
}

/// No-op log callback used by [`silence_logs`]. Matches the
/// `ggml_log_callback` / `llama_log_set` C signature.
unsafe extern "C" fn discard_log(
    _level: ggml_log_level,
    _msg: *const std::os::raw::c_char,
    _user_data: *mut std::ffi::c_void,
) {
}

/// Possible errors when creating a new [`crate::Engine`] or
/// [`LlamaCppDecoder`].
#[derive(Error, Debug)]
pub enum NewError {
    #[error("Could not load model from file: {path}")]
    Model { path: PathBuf },
    #[error("Could not create context")]
    Context,
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

        Ok(Self {
            context,
            n_vocab: model.n_vocab() as usize,
            embedding_size: model.embedding_size() as usize,
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

    /// Deserialize the global state.
    ///
    /// # Panics
    /// * If the length of `state` is not equal to [`Self::state_size`].
    pub fn set_state(&mut self, state: &[u8]) {
        let len = self.state_size();
        assert_eq!(state.len(), len);
        let copied = unsafe {
            llama_state_set_data(self.context, state.as_ptr(), len)
        };
        assert_eq!(copied, len);
    }

    /// Performance information.
    pub fn get_timings(&self) -> llama_perf_context_data {
        unsafe { llama_perf_context(self.context) }
    }

    /// Reset performance information.
    pub fn reset_timings(&mut self) {
        unsafe { llama_perf_context_reset(self.context) };
    }

    /// Set the llama.cpp log callback. Does NOT touch the ggml logger
    /// — use [`silence_logs`] to hush both at once.
    pub fn set_log_callback(
        &mut self,
        callback: ggml_log_callback,
        callback_data: Option<*mut std::ffi::c_void>,
    ) {
        unsafe {
            llama_log_set(
                callback,
                callback_data.unwrap_or(std::ptr::null_mut()),
            );
        }
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
        let mut batch = Batch::new(1, 0, 1)
            .expect("step batch allocation failed");
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
    }

    fn memory_seq_rm(
        &mut self,
        seq_id: i32,
        p0: i32,
        p1: i32,
    ) -> bool {
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

    /// llama.cpp preserves recurrent state per-cell, so arbitrary-
    /// position truncate is already lossless. No snapshot needed.
    fn checkpoint_pos(&mut self, _seq_id: i32, _pos: i32) {}

    /// Maps to `llama_kv_cache_seq_rm(seq_id, pos, -1)`. Returns
    /// `BackendUnsupported` only when the underlying call returns
    /// false (invalid seq_id, which Session never passes).
    fn restore_to(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError> {
        if LlamaCppDecoder::memory_seq_rm(self, seq_id, pos, -1) {
            Ok(())
        } else {
            Err(MemoryRmError::BackendUnsupported { pos })
        }
    }
}
