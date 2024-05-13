use crate::{
    model::{Vocab, VocabKind},
    predictor::{CandidatePredictor, PiecePredictor, TokenPredictor},
    Batch, Model, PredictOptions, Predictor,
};

use std::{num::NonZeroUsize, path::PathBuf, sync::Mutex};

use llama_cpp_sys_3::{
    ggml_log_callback, ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED,
    llama_backend_free, llama_backend_init, llama_context,
    llama_context_default_params, llama_context_params, llama_copy_state_data,
    llama_decode, llama_free, llama_get_embeddings_ith,
    llama_get_kv_cache_token_count, llama_get_kv_cache_used_cells,
    llama_get_logits_ith, llama_get_state_size, llama_get_timings,
    llama_kv_cache_clear, llama_kv_cache_defrag, llama_kv_cache_seq_add,
    llama_kv_cache_seq_cp, llama_kv_cache_seq_div, llama_kv_cache_seq_keep,
    llama_kv_cache_seq_pos_max, llama_kv_cache_seq_rm, llama_kv_cache_update,
    llama_log_set, llama_model_params, llama_n_batch, llama_n_ctx,
    llama_new_context_with_model, llama_numa_init, llama_pos,
    llama_reset_timings, llama_seq_id, llama_set_n_threads, llama_set_rng_seed,
    llama_set_state_data, llama_supports_gpu_offload, llama_supports_mlock,
    llama_supports_mmap, llama_timings, llama_token,
};

use thiserror::Error;

/// Global engine count. When this drops to 0, the llama backend is freed in
/// the last [`Engine`]'s `Drop` implementation.
static ENGINE_COUNT: Mutex<usize> = Mutex::new(0);

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

/// An `Engine` encompasses everything needed to run inferences. It contains the
/// model and the context. It is the main entry point for running inferences.
///
/// # Note:
/// There can only be one `Engine` per process at a time. This may change in the
/// future.
// TODO: It's possible to swap models with llama.cpp. We could add a method to
// set the model for the context. That being said, there is a lot of other
// global state in llama.cpp, so it wouldn't be easy or straightforward to do
// so.
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
                    llama_numa_init(numa_strategy.unwrap_or(
                        ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED,
                    ));
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

    /// Set the rng seed.
    pub fn set_rng_seed(&self, seed: u32) {
        unsafe { llama_set_rng_seed(self.context, seed) };
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

    /// Get the size of the global state (rng, logits, embedding, and kv_cache).
    pub fn state_size(&self) -> usize {
        unsafe { llama_get_state_size(self.context) as usize }
    }

    /// Get the llama.cpp global state (rng, logits, embedding, and kv_cache).
    pub fn get_state(&self) -> Vec<u8> {
        let len = self.state_size();
        let mut buf = Vec::with_capacity(len);
        let copied =
            unsafe { llama_copy_state_data(self.context, buf.as_mut_ptr()) };
        assert_eq!(copied, len);

        buf
    }

    /// Set the llama.cpp global state (rng, logits, embedding, and kv_cache).
    ///
    /// # Panics
    /// * If the length of `state` is not equal to [`Engine::state_size`].
    pub fn set_state(&mut self, state: &[u8]) {
        let len = self.state_size();
        assert_eq!(state.len(), len);

        let copied =
            unsafe { llama_set_state_data(self.context, state.as_ptr()) };
        assert_eq!(copied, len);
    }

    /// Performance information
    pub fn get_timings(&self) -> llama_timings {
        unsafe { llama_get_timings(self.context) }
    }

    /// Reset performance information
    pub fn reset_timings(&self) {
        unsafe { llama_reset_timings(self.context) };
    }

    /// Set callback for all future logging.
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

    /// The number of tokens in the KV cache. (slow, use only for debugging)
    pub fn kv_cache_token_count(&self) -> i32 {
        unsafe { llama_get_kv_cache_token_count(self.context) }
    }

    /// The number of use KV cells.
    pub fn kv_cache_used_cells(&self) -> i32 {
        unsafe { llama_get_kv_cache_used_cells(self.context) }
    }

    /// Clear the KV cache.
    pub fn kv_cache_clear(&self) {
        unsafe { llama_kv_cache_clear(self.context) }
    }

    /// Removes all tokens from the KV cache that belong to the specified
    /// sequence and have positions in [p0, p1)
    /// seq_id < 0 : match any sequence
    /// p0 < 0     : [0,  p1]
    /// p1 < 0     : [p0, inf)
    pub fn kv_cache_seq_rm(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> bool {
        unsafe { llama_kv_cache_seq_rm(self.context, seq_id, p0, p1) }
    }

    /// Copy all tokens that belong to the specified sequence in the KV cache to
    /// another sequence.
    /// Note that this does not allocate extra KV cache memory - it simply
    /// assigns the tokens to the new sequence
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    pub fn kv_cache_seq_cp(
        &self,
        src: llama_seq_id,
        dst: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) {
        unsafe { llama_kv_cache_seq_cp(self.context, src, dst, p0, p1) }
    }

    /// Removes all tokens that do not belong to the specified sequence from the
    /// KV cache.
    pub fn kv_cache_seq_keep(&self, seq_id: llama_seq_id) {
        unsafe { llama_kv_cache_seq_keep(self.context, seq_id) }
    }

    /// Adds relative position "delta" to all tokens that belong to the
    /// specified sequence and have positions in [p0, p1)
    /// If the KV cache is RoPEd, the KV data is updated accordingly:
    ///   - lazily on next llama_decode()
    ///   - explicitly with llama_kv_cache_update()
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    pub fn kv_cache_seq_add(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
    ) {
        unsafe { llama_kv_cache_seq_add(self.context, seq_id, p0, p1, delta) }
    }

    /// Integer division of the positions by factor of `d > 1`
    /// If the KV cache is RoPEd, the KV data is updated accordingly:
    ///   - lazily on next llama_decode()
    ///   - explicitly with llama_kv_cache_update()
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    pub fn kv_cache_seq_div(
        &self,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        d: i32,
    ) {
        unsafe { llama_kv_cache_seq_div(self.context, seq_id, p0, p1, d) }
    }

    /// Returns the largest position present in the KV cache for the specified
    /// sequence.
    pub fn kv_cache_seq_pos_max(&self, seq_id: llama_seq_id) -> llama_pos {
        unsafe { llama_kv_cache_seq_pos_max(self.context, seq_id) }
    }

    /// Defragment the KV cache.
    ///
    /// This will be applied:
    ///   - lazily on next llama_decode()
    ///   - explicitly with llama_kv_cache_update()
    pub fn kv_cache_defrag(&self) {
        unsafe { llama_kv_cache_defrag(self.context) }
    }

    /// Update the KV cache (such as K-shifts, defragmentation, etc.)
    pub fn kv_cache_update(&self) {
        unsafe { llama_kv_cache_update(self.context) }
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
    pub fn set_n_threads(&mut self, n_gen: u32, n_batch: u32) {
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
        self.kv_cache_clear();
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
        self.kv_cache_clear();
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
        self.kv_cache_clear();
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
        self.kv_cache_clear();
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
