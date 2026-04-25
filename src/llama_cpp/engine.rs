use crate::{
    llama_cpp::{
        decoder::{silence_logs, FlashAttention, LlamaCppDecoder, NewError, DecodeError},
        LlamaCppBackend,
    },
    Batch, Engine, LlamaCppModel,
};

use std::path::PathBuf;

use llama_cpp_sys_3::{
    ggml_log_callback, llama_context, llama_context_default_params,
    llama_context_params, llama_model_default_params, llama_model_params,
    llama_perf_context_data, llama_seq_id, llama_supports_gpu_offload,
    llama_supports_mlock, llama_supports_mmap, llama_token,
};

/// Convenience alias for the llama.cpp-backed pair. Use
/// `LlamaCppEngine::from_path(...)` etc. when you want the default
/// backend without turbofish.
pub type LlamaCppEngine = Engine<LlamaCppBackend>;

impl LlamaCppEngine {
    /// Create a new `LlamaCppEngine` from common command line
    /// arguments.
    #[cfg(feature = "cli")]
    pub fn from_cli(
        args: crate::cli::Args,
        numa_strategy: Option<u32>,
    ) -> Result<Self, NewError> {
        let model_params = Some(args.model_params());
        let context_params = Some(args.context_params());
        Self::new(args.model, model_params, context_params, numa_strategy)
    }

    /// Create a new `LlamaCppEngine` from a model `path`, `model_params`,
    /// `context_params` and `numa_strategy`. The path is the only
    /// required argument.
    pub fn new(
        path: PathBuf,
        model_params: Option<llama_model_params>,
        context_params: Option<llama_context_params>,
        numa_strategy: Option<u32>,
    ) -> Result<Self, NewError> {
        let mut model =
            match LlamaCppModel::from_file(path.clone(), model_params) {
                Some(m) => m,
                None => return Err(NewError::Model { path }),
            };
        let context_params =
            context_params.unwrap_or(unsafe { llama_context_default_params() });
        let decoder =
            LlamaCppDecoder::new(&mut model, context_params, numa_strategy)?;
        Ok(Self { decoder, model })
    }

    /// Create a new engine from a model `path`. Default model and
    /// context parameters are used.
    pub fn from_path(path: PathBuf) -> Result<Self, NewError> {
        Self::new(path, None, None, None)
    }

    /// Create a new engine from a model `path` forcing CPU-only
    /// inference (zero GPU layers).
    pub fn from_path_cpu_only(path: PathBuf) -> Result<Self, NewError> {
        let mut mp = unsafe { llama_model_default_params() };
        mp.n_gpu_layers = 0;
        Self::new(path, Some(mp), None, None)
    }

    /// Create a new engine from a model `path` with an explicit Flash
    /// Attention policy.
    pub fn from_path_with_flash_attention(
        path: PathBuf,
        fa: FlashAttention,
    ) -> Result<Self, NewError> {
        let mut cp = unsafe { llama_context_default_params() };
        cp.flash_attn_type = fa.as_raw();
        Self::new(path, None, Some(cp), None)
    }

    /// Create a new engine from a model `path` with an explicit KV
    /// context size.
    pub fn from_path_with_n_ctx(
        path: PathBuf,
        n_ctx: u32,
    ) -> Result<Self, NewError> {
        let mut cp = unsafe { llama_context_default_params() };
        cp.n_ctx = n_ctx;
        cp.n_batch = n_ctx;
        cp.n_ubatch = cp.n_ubatch.min(n_ctx);
        Self::new(path, None, Some(cp), None)
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

    /// Raw pointer to the underlying llama.cpp context (const).
    pub fn context_ptr(&self) -> *const llama_context {
        self.decoder.context_ptr()
    }

    /// Raw pointer to the underlying llama.cpp context (mut).
    pub fn context_ptr_mut(&self) -> *mut llama_context {
        self.decoder.context_ptr_mut()
    }

    /// Max batch size configured on this context.
    pub fn n_batch(&self) -> u32 {
        self.decoder.n_batch()
    }

    /// Size of the serialized global state (logits, embedding, memory).
    pub fn state_size(&self) -> usize {
        self.decoder.state_size()
    }

    /// Get the llama.cpp global state.
    pub fn get_state(&self) -> Vec<u8> {
        self.decoder.get_state()
    }

    /// Set the llama.cpp global state.
    pub fn set_state(&mut self, state: &[u8]) {
        self.decoder.set_state(state)
    }

    /// Performance information.
    pub fn get_timings(&self) -> llama_perf_context_data {
        self.decoder.get_timings()
    }

    /// Reset performance information.
    pub fn reset_timings(&mut self) {
        self.decoder.reset_timings()
    }

    /// Set the llama.cpp log callback. Does NOT touch the ggml logger
    /// — use [`silence_logs`] to hush both at once.
    pub fn set_log_callback(
        &mut self,
        callback: ggml_log_callback,
        callback_data: Option<*mut std::ffi::c_void>,
    ) {
        self.decoder.set_log_callback(callback, callback_data)
    }

    /// Silence both llama.cpp and ggml log output for the remainder of
    /// this process. Convenience wrapper around [`silence_logs`] that
    /// returns `self` for chaining on construction, e.g.:
    ///
    /// ```no_run
    /// # use drama_llama::LlamaCppEngine;
    /// let engine = LlamaCppEngine::from_path("models/model.gguf".into()).unwrap().quiet();
    /// ```
    pub fn quiet(self) -> Self {
        silence_logs();
        self
    }

    /// Set the number of threads used for generation and batch processing.
    pub fn set_n_threads(&mut self, n_gen: i32, n_batch: i32) {
        self.decoder.set_n_threads(n_gen, n_batch)
    }

    /// Run one batch through `llama_decode`.
    pub fn decode(&self, batch: &Batch) -> Result<(), DecodeError> {
        self.decoder.decode(batch)
    }

    /// Decode `tokens` into the KV cache at positions
    /// `[start_pos, start_pos + tokens.len())` for `seq_id`.
    pub fn prefill(
        &self,
        tokens: &[llama_token],
        start_pos: usize,
        seq_id: llama_seq_id,
    ) -> Result<(), DecodeError> {
        self.decoder.prefill_inherent(tokens, start_pos, seq_id)
    }

    /// Get logits for the i'th token.
    pub fn logits(&self, i: usize) -> &[f32] {
        self.decoder.logits(i)
    }

    /// Get mutable logits for the i'th token.
    pub fn logits_mut(&mut self, i: i32) -> &mut [f32] {
        self.decoder.logits_mut(i)
    }

    /// Get embeddings for the i'th sequence.
    pub fn embeddings(&self, i: i32) -> &[f32] {
        self.decoder.embeddings(i)
    }

    /// Get mutable embeddings for the i'th sequence.
    pub fn embeddings_mut(&mut self, i: i32) -> &mut [f32] {
        self.decoder.embeddings_mut(i)
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
        use std::path::PathBuf;
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("models/model.gguf");
        for i in 0..1000 {
            let engine = LlamaCppEngine::new(path.clone(), None, None, None);
            if engine.is_err() {
                println!(
                    "Failed to create engine after {} iterations because: {}",
                    i,
                    engine.unwrap_err()
                );
            }
        }
    }

    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    /// The resuming prediction path (prefill + `predict_pieces_resuming`)
    /// must produce the same token stream as the fresh path
    /// (`predict_pieces`) under greedy sampling.
    fn test_predict_pieces_resuming_matches() {
        use std::path::PathBuf;
        const PROMPT: &str = "The quick brown fox jumps over the lazy dog.";
        let model_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");

        // --- A: fresh path ---
        let mut engine_a = LlamaCppEngine::from_path(model_path.clone()).unwrap();
        let tokens_a = engine_a.model.tokenize(PROMPT, true);
        assert!(tokens_a.len() >= 4, "prompt tokenization too short");
        let k = tokens_a.len() / 2;

        let mut opts = crate::PredictOptions::greedy().add_stop(".".to_owned());
        opts.n = std::num::NonZeroUsize::new(16).unwrap();

        let fresh: Vec<String> = engine_a
            .predict_pieces(tokens_a.clone(), opts.clone())
            .collect();

        drop(engine_a);

        // --- B: resuming path on a fresh engine ---
        let mut engine_b = LlamaCppEngine::from_path(model_path).unwrap();
        let tokens_b = engine_b.model.tokenize(PROMPT, true);
        assert_eq!(tokens_a, tokens_b, "tokenization drift between engines");

        let (prefix, suffix) = tokens_b.split_at(k);

        engine_b.memory_clear();
        engine_b
            .prefill(prefix, 0, 0)
            .expect("priming prefill failed");

        let resumed: Vec<String> = engine_b
            .predict_pieces_resuming(suffix.to_vec(), k, 0, opts)
            .collect();

        assert_eq!(
            fresh, resumed,
            "resuming path diverged from fresh path under greedy sampling",
        );
    }
}
