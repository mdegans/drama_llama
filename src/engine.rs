use crate::{
    model::{Vocab, VocabKind},
    predictor::{CandidatePredictor, PiecePredictor, TokenPredictor},
    Batch, Model, PredictOptions, Predictor,
};

use std::{num::NonZeroUsize, path::PathBuf, sync::Mutex};

use llama_cpp_sys_3::{
    ggml_log_callback, ggml_log_level, ggml_log_set,
    ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED, llama_backend_free,
    llama_backend_init, llama_context, llama_context_default_params,
    llama_context_params, llama_decode,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED,
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED, llama_free,
    llama_get_embeddings_ith, llama_get_logits_ith, llama_get_memory,
    llama_log_set, llama_memory_clear, llama_memory_seq_add,
    llama_memory_seq_cp, llama_memory_seq_div, llama_memory_seq_keep,
    llama_memory_seq_pos_max, llama_memory_seq_rm, llama_model_default_params,
    llama_model_params, llama_n_batch, llama_n_ctx,
    llama_new_context_with_model, llama_numa_init, llama_perf_context,
    llama_perf_context_data, llama_perf_context_reset, llama_pos, llama_seq_id,
    llama_set_n_threads, llama_state_get_data, llama_state_get_size,
    llama_state_set_data, llama_supports_gpu_offload, llama_supports_mlock,
    llama_supports_mmap, llama_token,
};

use thiserror::Error;

/// Global engine count. When this drops to 0, the llama backend is freed in
/// the last [`Engine`]'s `Drop` implementation.
static ENGINE_COUNT: Mutex<usize> = Mutex::new(0);

/// Silence `llama.cpp` + `ggml` log output.
///
/// Installs a no-op callback on both loggers. llama.cpp and ggml maintain
/// separate log sinks — Metal pipeline compile chatter comes from the ggml
/// side, model-load prose from the llama side — so both need to be hushed
/// for quiet generation.
///
/// Idempotent. Safe to call before or after creating an [`Engine`]. Call
/// [`restore_default_logs`] to undo.
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

/// Possible errors when creating a new [`Engine`].
#[derive(Error, Debug)]
pub enum NewError {
    #[error("Could not load model from file: {path}")]
    Model { path: PathBuf },
    #[error("Could not create context")]
    Context,
}

static_assertions::assert_impl_all!(NewError: Send, Sync);

/// Possible errors when calling [`Engine::decode`].
#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("Could not find a KV slot for the Batch. Try reducing the size of the batch or increase the context size.")]
    NoKvSlot,
    #[error("`llama_decode` returned an error code: {code}")]
    ErrorCode { code: i32 },
}

static_assertions::assert_impl_all!(DecodeError: Send, Sync);

/// Flash Attention policy for a new [`Engine`] context.
///
/// llama.cpp's default is [`Self::Auto`] — it enables Flash Attention
/// when the active backend supports it (typical on Metal, CUDA, Vulkan).
/// [`Self::Disabled`] is useful as a diagnostic: FA uses a fused softmax
/// kernel that can produce slightly different logits than the non-FA
/// attention path on close-race token distributions, and toggling it off
/// rules that out as a source of divergence against other runners
/// (notably ollama's Go-native `--ollama-engine`, which does not use
/// llama.cpp's FA kernel at all).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttention {
    /// Let llama.cpp decide based on backend capabilities (default).
    Auto,
    /// Force-disable Flash Attention. Slower but numerically closer to
    /// the un-fused softmax path in other runners.
    Disabled,
    /// Force-enable. Errors at context creation if the backend doesn't
    /// support it.
    Enabled,
}

impl FlashAttention {
    /// Map to the raw llama.cpp enum value.
    fn as_raw(self) -> llama_cpp_sys_3::llama_flash_attn_type {
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

/// An `Engine` encompasses everything needed to run inferences. It contains the
/// model and the context. It is the main entry point for running inferences.
#[derive(Debug)]
pub struct Engine {
    /// The llama.cpp context.
    pub(crate) context: *mut llama_context,
    /// The llama.cpp model.
    pub model: Model,
    /// Vocabulary
    pub(crate) vocab: Vocab,
}

impl Engine {
    /// Create a new `Engine` from common command line arguments.
    #[cfg(feature = "cli")]
    pub fn from_cli(
        args: crate::cli::Args,
        numa_strategy: Option<u32>,
    ) -> Result<Self, NewError> {
        let model_params = Some(args.model_params());
        let context_params = Some(args.context_params());
        Self::new(
            args.model,
            model_params,
            context_params,
            numa_strategy,
            Some(args.vocab),
        )
    }

    /// Create a new `Engine` from a model `path`, `model_params`,
    /// `context_params` and `numa_strategy`. The path is the only required
    /// argument. The others are optional.
    pub fn new(
        path: PathBuf,
        model_params: Option<llama_model_params>,
        context_params: Option<llama_context_params>,
        numa_strategy: Option<u32>,
        vocab: Option<VocabKind>,
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

        let mut model = match Model::from_file(path.clone(), model_params) {
            Some(model) => model,
            None => return Err(NewError::Model { path }),
        };
        let context_params =
            context_params.unwrap_or(unsafe { llama_context_default_params() });
        let context = unsafe {
            llama_new_context_with_model(model.as_ptr_mut(), context_params)
        };
        if context.is_null() {
            return Err(NewError::Context);
        }

        // for the moment we're only enabling safe vocab
        let vocab = Vocab::new([vocab.unwrap_or(VocabKind::Safe)], &model);

        Ok(Self {
            context,
            model,
            vocab,
        })
    }

    /// Set Vocab
    pub fn set_vocab(&mut self, vocab: VocabKind) {
        self.vocab = Vocab::new([vocab], &self.model);
    }

    /// Create a new engine from a model `path`. Default model and context
    /// parameters are used.
    pub fn from_path(path: PathBuf) -> Result<Self, NewError> {
        Self::new(path, None, None, None, None)
    }

    /// Create a new engine from a model `path` forcing CPU-only
    /// inference (zero GPU layers).
    ///
    /// Diagnostic escape hatch: CPU kernels are standardized across
    /// ggml implementations where Metal/CUDA/Vulkan kernels are not;
    /// forcing CPU rules out GPU-kernel divergence as the cause of
    /// output differences against other runners.
    ///
    /// Expect this to be dramatically slower than the default
    /// (GPU-offloaded) path — useful for one-off verification, not
    /// production inference.
    pub fn from_path_cpu_only(path: PathBuf) -> Result<Self, NewError> {
        let mut mp = unsafe { llama_model_default_params() };
        mp.n_gpu_layers = 0;
        Self::new(path, Some(mp), None, None, None)
    }

    /// Create a new engine from a model `path` with an explicit Flash
    /// Attention policy.
    ///
    /// Diagnostic hatch: when debugging output divergence between llama.cpp
    /// and ollama's engine (or any other GGML-based runner), forcing FA off
    /// isolates whether the Flash Attention softmax path is producing
    /// different logits on close-race token distributions. ollama's Go-
    /// native runner (`--ollama-engine`) uses a different attention
    /// implementation entirely; comparing against it with FA on vs off in
    /// llama.cpp can pin the numerical divergence to the softmax kernel.
    ///
    /// The default (via [`Self::from_path`]) is [`FlashAttention::Auto`] —
    /// llama.cpp picks based on the backend's capabilities.
    pub fn from_path_with_flash_attention(
        path: PathBuf,
        fa: FlashAttention,
    ) -> Result<Self, NewError> {
        let mut cp = unsafe { llama_context_default_params() };
        cp.flash_attn_type = fa.as_raw();
        Self::new(path, None, Some(cp), None, None)
    }

    /// Returns true if mmap is supported.
    pub fn supports_mmap() -> bool {
        unsafe { llama_supports_mmap() }
    }

    /// Returns true if mlock is supported.
    pub fn supports_mlock() -> bool {
        unsafe { llama_supports_mlock() }
    }

    /// Returns true if GPU offload is supported.
    pub fn supports_gpu_offload() -> bool {
        unsafe { llama_supports_gpu_offload() }
    }

    /// Get a raw pointer to the underlying llama.cpp context.
    pub fn context_ptr(&self) -> *const llama_context {
        // Safety:
        // The same as for `model`.
        self.context
    }

    /// Get a raw pointer to the underlying llama.cpp context.
    pub fn context_ptr_mut(&self) -> *mut llama_context {
        self.context
    }

    // TODO: document, because it's not in llama.cpp
    // Is it the number of contexts? Is it the number of tokens consumed?
    // Sampled?
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_n_ctx(self.context) }
    }

    // TODO: same as `n_ctx`
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_n_batch(self.context) }
    }

    /// Get the size of the global state (logits, embedding, and memory).
    pub fn state_size(&self) -> usize {
        unsafe { llama_state_get_size(self.context) }
    }

    /// Get the llama.cpp global state (logits, embedding, and memory).
    pub fn get_state(&self) -> Vec<u8> {
        let len = self.state_size();
        let mut buf = vec![0u8; len];
        let copied = unsafe {
            llama_state_get_data(self.context, buf.as_mut_ptr(), len)
        };
        assert_eq!(copied, len);

        buf
    }

    /// Set the llama.cpp global state (logits, embedding, and memory).
    ///
    /// # Panics
    /// * If the length of `state` is not equal to [`Engine::state_size`].
    pub fn set_state(&mut self, state: &[u8]) {
        let len = self.state_size();
        assert_eq!(state.len(), len);

        let copied =
            unsafe { llama_state_set_data(self.context, state.as_ptr(), len) };
        assert_eq!(copied, len);
    }

    /// Performance information
    pub fn get_timings(&self) -> llama_perf_context_data {
        unsafe { llama_perf_context(self.context) }
    }

    /// Reset performance information
    pub fn reset_timings(&mut self) {
        unsafe { llama_perf_context_reset(self.context) };
    }

    /// Set the llama.cpp log callback. Does NOT touch the ggml logger —
    /// use [`silence_logs`] to hush both at once.
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

    /// Silence both llama.cpp and ggml log output for the remainder of
    /// this process. Convenience wrapper around [`silence_logs`] that
    /// returns `self` for chaining on construction, e.g.:
    ///
    /// ```no_run
    /// # use drama_llama::Engine;
    /// let engine = Engine::from_path("models/model.gguf").unwrap().quiet();
    /// ```
    pub fn quiet(self) -> Self {
        silence_logs();
        self
    }

    /// Clear the memory (KV cache).
    pub fn memory_clear(&self) {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_clear(mem, true) }
    }

    /// Removes all tokens from memory that belong to the specified
    /// sequence and have positions in [p0, p1)
    /// seq_id < 0 : match any sequence
    /// p0 < 0     : [0,  p1]
    /// p1 < 0     : [p0, inf)
    pub fn memory_seq_rm(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> bool {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_rm(mem, seq_id, p0, p1) }
    }

    /// Copy all tokens that belong to the specified sequence in memory to
    /// another sequence.
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
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

    /// Removes all tokens that do not belong to the specified sequence.
    pub fn memory_seq_keep(&self, seq_id: llama_seq_id) {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_keep(mem, seq_id) }
    }

    /// Adds relative position "delta" to all tokens that belong to the
    /// specified sequence and have positions in [p0, p1)
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
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

    /// Integer division of the positions by factor of `d > 1`
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
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

    /// Returns the largest position present in memory for the specified
    /// sequence.
    pub fn memory_seq_pos_max(&self, seq_id: llama_seq_id) -> llama_pos {
        let mem = unsafe { llama_get_memory(self.context) };
        unsafe { llama_memory_seq_pos_max(mem, seq_id) }
    }

    /// Decode
    pub fn decode(&self, batch: &Batch) -> Result<(), DecodeError> {
        let ret = unsafe { llama_decode(self.context, batch.batch) };

        match ret {
            0 => Ok(()),
            1 => Err(DecodeError::NoKvSlot),
            _ => Err(DecodeError::ErrorCode { code: ret }),
        }
    }

    /// Set the number of threads used for generation and batch processing.
    pub fn set_n_threads(&mut self, n_gen: i32, n_batch: i32) {
        unsafe { llama_set_n_threads(self.context, n_gen, n_batch) }
    }

    /// Get logits for the i'th token.
    ///
    /// # Panics
    /// - If the index is invalid. This comes from the c++ side.
    /// - If the index exceedes i32::MAX. This is a limitation of the c++ API.
    // TODO: This is a terrible API. The tokens are kept separate from the
    // logits. So you have to keep track of the index in the batch and if you
    // make a mistake, you'll get a panic. We may be able to make this better by
    // accepting the batch as an argument as well as index.
    pub fn logits(&self, i: usize) -> &[f32] {
        let len = self.model.n_vocab() as usize;
        let ptr = unsafe {
            llama_get_logits_ith(self.context, i.try_into().unwrap())
        };

        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get logits for the i'th token.
    ///
    /// # Panics
    /// - If the index is invalid. This comes from the c++ side.
    pub fn logits_mut(&mut self, i: i32) -> &mut [f32] {
        let len = self.model.n_vocab() as usize;
        let ptr = unsafe { llama_get_logits_ith(self.context, i) };

        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Get embeddings for the i'th sequence.
    ///
    /// # Panics
    /// - If the index is invalid. This comes from the c++ side.
    pub fn embeddings(&self, i: i32) -> &[f32] {
        let len = self.model.embedding_size() as usize;
        let ptr = unsafe { llama_get_embeddings_ith(self.context, i) };

        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Get embeddings for the i'th sequence.
    ///
    /// # Panics
    /// - If the index is invalid. This comes from the c++ side.
    pub fn embeddings_mut(&mut self, i: i32) -> &mut [f32] {
        let len = self.model.embedding_size() as usize;
        let ptr = unsafe { llama_get_embeddings_ith(self.context, i) };

        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Return an iterator that yields candidates for the next token in a
    /// sequence until `n` tokens have been predicted or the end of context is
    /// reached.
    ///
    /// # Note
    /// * The [`record_choice`] method must be called on the returned iterator
    ///   to record the choice made by the user (or iteration will end).
    /// * The tokens given will be available as the `tokens` field of the
    ///   iterator. For convenience, when finished, the [`CandidatePredictor`]
    ///   can be converted back `into` the tokens, including any choices made.
    ///
    /// [`record_choice`]: crate::predictor::CandidatePredictor::record_choice
    pub fn predict_candidates<'a>(
        &'a mut self,
        tokens: Vec<llama_token>,
        n: NonZeroUsize,
    ) -> CandidatePredictor<'a> {
        // TODO: We technically do not need to clear the cache here. If we keep
        // track of sequence ids, we can clear the cache when the sequence id
        // is no longer in use. This would be more efficient, but requires more
        // bookkeeping.
        self.memory_clear();
        CandidatePredictor::new(self, tokens, n)
    }

    /// Return an iterator that predicts a sequence of tokens until `options.n`
    /// tokens have been predicted, the end of context is reached, or stop
    /// conditions are met.
    ///
    /// # Note
    /// * The tokens given will be available as the `tokens` field of the
    ///   iterator. For convenience, when finished, the [`TokenPredictor`] can
    ///   be converted back `into` the tokens, including any predicted.
    pub fn predict_tokens<'a>(
        &'a mut self,
        tokens: Vec<llama_token>,
        options: PredictOptions,
    ) -> TokenPredictor<'a> {
        self.memory_clear();
        TokenPredictor::new(self, tokens, options)
    }

    /// Return an iterator that predicts a sequence of pieces until `options.n`
    /// tokens have been predicted, the end of context is reached, or stop
    /// conditions are met.
    ///
    /// # Note
    /// * The last piece is not truncated, however the `text` field of the
    ///   predictor will be truncated to a stop string if one is provided and
    ///   found at the end of the text. In this case it may be desirable to use
    ///   a `while let` loop rather than a `for` loop since a for loop will
    ///   consume the iterator.
    /// * The tokens given will be available as the `tokens` field of the
    ///   iterator. For convenience, when finished, the [`PiecePredictor`] can
    ///   be converted back `into` the tokens, including any predicted. It can
    ///   also be converted into the predicted text. See the [`PiecePredictor`]
    ///   for additional conversion and collection methods.
    pub fn predict_pieces<'a>(
        &'a mut self,
        tokens: Vec<llama_token>,
        options: PredictOptions,
    ) -> PiecePredictor<'a> {
        self.memory_clear();
        PiecePredictor::new(self, tokens, options)
    }

    /// Return an iterator that predicts both token sand peices until `
    /// tokens have been predicted, the end of context is reached, or stop
    /// conditions are met.
    ///
    /// # Note
    /// * The tokens and generated text are available from the
    ///   [`Predictor::into_tokens_and_text`] method.
    pub fn predict<'a>(
        &'a mut self,
        tokens: Vec<llama_token>,
        options: PredictOptions,
    ) -> Predictor<'a> {
        self.memory_clear();
        Predictor::new(self, tokens, options)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.context);
        }

        let mut count = ENGINE_COUNT.lock().unwrap();
        *count -= 1;
        if *count == 0 {
            unsafe { llama_backend_free() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "long running"]
    /// Test engine can be constructed and destructed many times. Because there
    /// is global state in llama.cpp, this is a stress test to ensure that there
    /// are no resource leaks. It's not comprehensive, but it's something.
    // TODO: find a way to test for increased memory usage. The test also might
    // be better upstream in the bindings.
    fn construct_destruct_stress_test() {
        // Thanks to Bing's Copilot for helping me quickly find how to reference
        // the absolute path of the model file.
        use std::path::PathBuf;
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("models/model.gguf");
        for i in 0..1000 {
            let engine = Engine::new(path.clone(), None, None, None, None);
            if engine.is_err() {
                println!(
                    "Failed to create engine after {} iterations because: {}",
                    i,
                    engine.unwrap_err()
                );
            }
        }
    }
}
