use crate::{
    llama_cpp::{
        decoder::{DecodeError, FlashAttention, LlamaCppDecoder, NewError},
        LlamaCppBackend,
    },
    log::silence_logs,
    Batch, Engine, LlamaCppModel,
};

use std::path::PathBuf;

use llama_cpp_sys_3::{
    llama_context, llama_context_default_params, llama_context_params,
    llama_model_default_params, llama_model_params, llama_perf_context_data,
    llama_seq_id, llama_supports_gpu_offload, llama_supports_mlock,
    llama_supports_mmap, llama_token,
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

    /// llama.cpp's `llama_context_default_params()` with a usable
    /// thread count. The upstream library default is a hard-coded 4
    /// threads (ggml's `GGML_DEFAULT_N_THREADS`, marked "TODO: better
    /// default" upstream), which cripples CPU inference on larger
    /// machines — every llama.cpp *runner* overrides it, and so do we.
    /// Uses all available logical cores for both generation and batch
    /// processing; tune after construction with
    /// [`Self::set_n_threads`].
    pub fn default_context_params() -> llama_context_params {
        let mut cp = unsafe { llama_context_default_params() };
        if let Ok(n) = std::thread::available_parallelism() {
            let n = n.get() as i32;
            cp.n_threads = n;
            cp.n_threads_batch = n;
        }
        cp
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
            context_params.unwrap_or_else(Self::default_context_params);
        let decoder =
            LlamaCppDecoder::new(&mut model, context_params, numa_strategy)?;
        #[allow(unused_mut)]
        let mut engine = Self {
            vision: None,
            decoder,
            model,
            probe_hook: None,
        };
        // mmproj sidecar convention: a sibling `<model>.mmproj.gguf`
        // opts the model into vision by existing. A present-but-broken
        // sidecar is a hard error — silently continuing text-only
        // would be the silent image drop this feature exists to kill.
        #[cfg(feature = "mtmd")]
        {
            use crate::llama_cpp::mtmd::{Mtmd, MtmdParams};
            if let Some(mmproj) = crate::sidecar::mmproj_path(&path) {
                let mtmd = Mtmd::from_path(
                    &mmproj,
                    &engine.model,
                    MtmdParams::default(),
                )
                .map_err(|source| NewError::Mtmd {
                    path: mmproj,
                    source,
                })?;
                engine.vision = Some(mtmd);
            }
        }
        Ok(engine)
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

    /// Load a multimodal projector (mmproj GGUF) from an arbitrary
    /// path, replacing any current vision capability. The sidecar
    /// convention (`<model>.mmproj.gguf`, see
    /// [`crate::sidecar::mmproj_path`]) is auto-loaded at
    /// construction; this is for projectors living elsewhere.
    #[cfg(feature = "mtmd")]
    pub fn load_mmproj(
        &mut self,
        path: impl AsRef<std::path::Path>,
        params: crate::llama_cpp::mtmd::MtmdParams,
    ) -> Result<(), crate::llama_cpp::mtmd::MtmdNewError> {
        let mtmd =
            crate::llama_cpp::mtmd::Mtmd::from_path(path, &self.model, params)?;
        self.vision = Some(mtmd);
        Ok(())
    }

    /// Create a new engine from a model `path` with an explicit Flash
    /// Attention policy.
    pub fn from_path_with_flash_attention(
        path: PathBuf,
        fa: FlashAttention,
    ) -> Result<Self, NewError> {
        let mut cp = Self::default_context_params();
        cp.flash_attn_type = fa.as_raw();
        Self::new(path, None, Some(cp), None)
    }

    /// Create a new engine from a model `path` with an explicit KV
    /// context size.
    pub fn from_path_with_n_ctx(
        path: PathBuf,
        n_ctx: u32,
    ) -> Result<Self, NewError> {
        let mut cp = Self::default_context_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = n_ctx;
        cp.n_ubatch = cp.n_ubatch.min(n_ctx);
        Self::new(path, None, Some(cp), None)
    }

    /// [`Self::from_path_with_n_ctx`] plus multi-sequence support:
    /// `n_seq_max` KV sequences over one **unified** cell pool of
    /// `n_ctx` (llama.cpp's recommended shape for sequences sharing
    /// large prefixes). This is what the multi-slot prefix cache
    /// wants: `Session::with_prefix_cache` sizes its slot count from
    /// [`crate::backend::Decoder::n_seq_max`], so a session built on
    /// this engine caches up to `n_seq_max` distinct conversation
    /// prefixes (agents) concurrently instead of thrashing one slot.
    /// On recurrent / hybrid models `n_seq_max` also sizes the
    /// per-sequence recurrent-state slots.
    pub fn from_path_with_n_ctx_and_seqs(
        path: PathBuf,
        n_ctx: u32,
        n_seq_max: u32,
    ) -> Result<Self, NewError> {
        let mut cp = Self::default_context_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = n_ctx;
        cp.n_ubatch = cp.n_ubatch.min(n_ctx);
        cp.n_seq_max = n_seq_max.max(1);
        cp.kv_unified = true;
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

    /// Size of the serialized state for a single sequence.
    pub fn state_seq_size(&self, seq_id: i32) -> usize {
        self.decoder.state_seq_size(seq_id)
    }

    /// Serialize a single sequence's state (KV cells plus recurrent
    /// layer state). Restores via [`Self::set_state_seq`].
    pub fn get_state_seq(&self, seq_id: i32) -> Vec<u8> {
        self.decoder.get_state_seq(seq_id)
    }

    /// Restore a single sequence's state from
    /// [`Self::get_state_seq`] bytes, loading as `dest_seq_id`.
    /// Returns `false` if llama.cpp rejects the payload (the
    /// destination sequence is left cleared).
    pub fn set_state_seq(&mut self, state: &[u8], dest_seq_id: i32) -> bool {
        self.decoder.set_state_seq(state, dest_seq_id)
    }

    /// Whether checkpointing takes real per-sequence snapshots. On by
    /// default for recurrent / hybrid models (whose layer state cannot
    /// be rewound by KV truncation); off for pure attention.
    pub fn seq_snapshots_enabled(&self) -> bool {
        self.decoder.seq_snapshots_enabled()
    }

    /// Force per-sequence snapshotting on or off. See
    /// [`LlamaCppDecoder::set_seq_snapshots`](crate::LlamaCppDecoder::set_seq_snapshots).
    pub fn set_seq_snapshots(&mut self, enabled: bool) {
        self.decoder.set_seq_snapshots(enabled)
    }

    /// Number of per-sequence snapshots currently held.
    pub fn seq_snapshot_count(&self) -> usize {
        self.decoder.seq_snapshot_count()
    }

    /// Performance information.
    pub fn get_timings(&self) -> llama_perf_context_data {
        self.decoder.get_timings()
    }

    /// Reset performance information.
    pub fn reset_timings(&mut self) {
        self.decoder.reset_timings()
    }

    /// Silence both llama.cpp and ggml log output for the remainder of
    /// this process. Convenience wrapper around
    /// [`crate::log::silence_logs`] that returns `self` for chaining on
    /// construction, e.g.:
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

    /// Resident set size of this process in bytes (via `ps`, so KiB
    /// granularity). On Apple Silicon, Metal buffers are unified-memory
    /// mappings inside the process, so leaked model weights or contexts
    /// show up here. Coarse, but a leaked engine is hundreds of MB to
    /// GBs per iteration — loud enough for a coarse check.
    #[cfg(unix)]
    fn rss_bytes() -> u64 {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .expect("ps failed");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .expect("unparsable rss")
            * 1024
    }

    #[test]
    #[ignore = "long running, requires models/model.gguf"]
    /// Engine can be constructed and destructed repeatedly without leaking.
    /// llama.cpp has global state (`backend_init`/`backend_free` refcounted
    /// by `ENGINE_COUNT`), so cycling the full lifecycle catches gross
    /// init/free bugs. The leak check replaces the original version's
    /// 1000-iteration watch-Activity-Monitor protocol: baseline RSS after
    /// the first cycle (global init + page-cache warmup land there), then
    /// assert the remaining cycles don't accumulate. A real per-iteration
    /// leak (model, context, or KV cache) is ≥ hundreds of MB × 9, far
    /// over the threshold; the threshold is generous because allocator
    /// and Metal-driver caches genuinely retain some memory.
    fn construct_destruct_stress_test() {
        use std::path::PathBuf;
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("models/model.gguf");

        let construct = |i: usize| {
            drop(
                LlamaCppEngine::new(path.clone(), None, None, None)
                    .unwrap_or_else(|e| {
                        panic!(
                            "engine construction failed on iteration {i}: {e}"
                        )
                    }),
            );
        };

        construct(0);
        let baseline = rss_bytes();
        for i in 1..10 {
            construct(i);
        }
        let grown = rss_bytes().saturating_sub(baseline);

        const LIMIT: u64 = 2 << 30; // 2 GiB
        assert!(
            grown < LIMIT,
            "RSS grew {} MiB over 9 construct/destruct cycles (limit {} MiB) \
             — per-iteration leak?",
            grown >> 20,
            LIMIT >> 20,
        );
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
        let mut engine_a =
            LlamaCppEngine::from_path(model_path.clone()).unwrap();
        let tokens_a = engine_a.model.tokenize(PROMPT, true);
        assert!(tokens_a.len() >= 4, "prompt tokenization too short");
        let k = tokens_a.len() / 2;

        let mut opts = crate::PredictOptions::greedy().add_stop(".".to_owned());
        opts.n = std::num::NonZeroUsize::new(16).unwrap();

        let fresh: Vec<String> = engine_a
            .predict_pieces(tokens_a.clone(), opts.clone(), None)
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
            .predict_pieces_resuming(suffix.to_vec(), k, 0, opts, None)
            .collect();

        assert_eq!(
            fresh, resumed,
            "resuming path diverged from fresh path under greedy sampling",
        );
    }
}
