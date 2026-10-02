use crate::{
    llama_cpp::{
        decoder::{DecodeError, LlamaCppDecoder, NewError},
        LlamaCppBackend, LlamaCppOptions,
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
    /// llama.cpp's `llama_context_default_params()` with the two
    /// defaults every llama.cpp *runner* overrides, and so do we:
    ///
    /// - a usable thread count. The upstream library default is a
    ///   hard-coded 4 threads (ggml's `GGML_DEFAULT_N_THREADS`, marked
    ///   "TODO: better default" upstream), which cripples CPU inference
    ///   on larger machines. Uses all available logical cores for both
    ///   generation and batch processing; tune after construction with
    ///   [`Self::set_n_threads`].
    /// - a window-sized sliding-window cache (`swa_full = false`), as
    ///   llama.cpp's server and CLI ship; the library's full-size one
    ///   would size Gemma 4's window layers at the whole context. See
    ///   [`LlamaCppOptions::swa_full`].
    pub fn default_context_params() -> llama_context_params {
        let mut cp = unsafe { llama_context_default_params() };
        if let Ok(n) = std::thread::available_parallelism() {
            let n = n.get() as i32;
            cp.n_threads = n;
            cp.n_threads_batch = n;
        }
        cp.swa_full = false;
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
        let model = Self::load_model(path.clone(), model_params)?;
        Self::with_model(path, model, context_params, numa_strategy)
    }

    /// [`Self::new`] past the model load: the context, then the mmproj
    /// sidecar.
    fn with_model(
        path: PathBuf,
        model: LlamaCppModel,
        context_params: Option<llama_context_params>,
        numa_strategy: Option<u32>,
    ) -> Result<Self, NewError> {
        let context_params =
            context_params.unwrap_or_else(Self::default_context_params);
        let decoder =
            LlamaCppDecoder::new(&model, context_params, numa_strategy)?;
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

    /// Load the model at `path`, classifying a failure
    /// ([`NewError::is_resource`]) by when it came. llama.cpp answers
    /// every failed load with the same null, so the file is first
    /// opened and then read vocab-only — header, metadata, vocabulary,
    /// no backend allocation — and only then loaded in full: a file
    /// that fails either check is [`NewError::Unreadable`] or
    /// [`NewError::Metadata`], and a full load that fails after both
    /// passed failed allocating ([`NewError::Model`]).
    fn load_model(
        path: PathBuf,
        params: Option<llama_model_params>,
    ) -> Result<LlamaCppModel, NewError> {
        let unreadable = |source| NewError::Unreadable {
            path: path.clone(),
            source,
        };
        let meta = std::fs::File::open(&path)
            .and_then(|file| file.metadata())
            .map_err(unreadable)?;
        if meta.is_dir() {
            return Err(unreadable(std::io::Error::new(
                std::io::ErrorKind::IsADirectory,
                "is a directory",
            )));
        }
        // SAFETY: returns a plain struct by value; no preconditions.
        let params =
            params.unwrap_or_else(|| unsafe { llama_model_default_params() });
        let vocab_only = llama_model_params {
            vocab_only: true,
            ..params
        };
        // Dropped at once: it only proves the metadata reads.
        LlamaCppModel::from_file(path.clone(), Some(vocab_only))
            .ok_or_else(|| NewError::Metadata { path: path.clone() })?;
        LlamaCppModel::from_file(path.clone(), Some(params))
            .ok_or(NewError::Model { path })
    }

    /// Create a new engine from a model `path` and load-time
    /// [`LlamaCppOptions`] — context size, KV slots, Flash Attention
    /// policy, GPU offload, NUMA.
    ///
    /// This is the constructor; [`Self::from_path`] is it with every
    /// option left at llama.cpp's default.
    pub fn from_path_with(
        path: PathBuf,
        options: LlamaCppOptions,
    ) -> Result<Self, NewError> {
        Self::from_path_with_n_ctx_override(path, options, None)
    }

    /// [`Self::from_path_with`], with a per-model `n_ctx` (a load
    /// sidecar's) that beats `options.n_ctx` once capped at the model's
    /// trained window — known only after the model loads, hence here.
    /// See [`crate::sidecar::effective_n_ctx`]. Logs the context the
    /// model is served with.
    pub(crate) fn from_path_with_n_ctx_override(
        path: PathBuf,
        options: LlamaCppOptions,
        n_ctx: Option<u32>,
    ) -> Result<Self, NewError> {
        let model =
            Self::load_model(path.clone(), Some(options.model_params()))?;
        let n_ctx_train = model.context_size().max(0) as u32;
        let effective =
            crate::sidecar::effective_n_ctx(options.n_ctx, n_ctx, n_ctx_train);
        if let Some(requested) = n_ctx.filter(|&n| Some(n) != effective) {
            tracing::warn!(
                path = %path.display(),
                requested,
                n_ctx_train,
                "per-model n_ctx exceeds the trained window; capped",
            );
        }
        let options = LlamaCppOptions {
            n_ctx: effective,
            ..options
        };
        let mut engine = Self::with_model(
            path.clone(),
            model,
            Some(options.context_params()),
            options.numa,
        )?;
        engine.set_checkpoint_budget(options.checkpoint_budget());
        tracing::info!(
            event = "context_size",
            path = %path.display(),
            n_ctx = engine.n_ctx(),
            n_ctx_train,
            source = if n_ctx.is_some() { "sidecar" } else { "default" },
            "serving with n_ctx {}",
            engine.n_ctx(),
        );
        Ok(engine)
    }

    /// Create a new engine from a model `path`. Default model and
    /// context parameters are used — note that llama.cpp's default
    /// `n_ctx` is 512, which is rarely what you want; see
    /// [`LlamaCppOptions::n_ctx`].
    pub fn from_path(path: PathBuf) -> Result<Self, NewError> {
        Self::from_path_with(path, LlamaCppOptions::default())
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

    /// Create a new engine from a model `path` with an explicit KV
    /// context size. Shorthand for the one option almost every caller
    /// sets; everything else goes through [`Self::from_path_with`].
    pub fn from_path_with_n_ctx(
        path: PathBuf,
        n_ctx: u32,
    ) -> Result<Self, NewError> {
        Self::from_path_with(path, LlamaCppOptions::default().with_n_ctx(n_ctx))
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
    pub fn context_ptr_mut(&mut self) -> *mut llama_context {
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

    /// How this model's sequences rewind — see
    /// [`LlamaCppDecoder::checkpointing`](crate::LlamaCppDecoder::checkpointing).
    pub fn checkpointing(&self) -> crate::Checkpointing {
        self.decoder.checkpointing()
    }

    /// Whether checkpointing takes real per-sequence snapshots. On by
    /// default for sliding-window and recurrent / hybrid models (whose
    /// state a KV truncate cannot rewind); off for dense attention.
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

    /// Host RAM the per-sequence snapshots currently hold, in bytes.
    pub fn seq_snapshot_bytes(&self) -> usize {
        self.decoder.seq_snapshot_bytes()
    }

    /// Bound the host RAM the checkpoints may hold. See
    /// [`LlamaCppDecoder::set_checkpoint_budget`](crate::LlamaCppDecoder::set_checkpoint_budget).
    pub fn set_checkpoint_budget(&mut self, budget: crate::CheckpointBudget) {
        self.decoder.set_checkpoint_budget(budget)
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
    ///
    /// `&mut self` is load-bearing: it prevents a slice from
    /// [`Self::logits`] staying live across a decode, which
    /// reallocates the buffer that slice points into.
    pub fn decode(&mut self, batch: &Batch) -> Result<(), DecodeError> {
        self.decoder.decode(batch)
    }

    /// Decode `tokens` into the KV cache at positions
    /// `[start_pos, start_pos + tokens.len())` for `seq_id`.
    ///
    /// See [`Self::decode`] for why this takes `&mut self`.
    pub fn prefill(
        &mut self,
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

    /// The failures a load finds before the backend allocates anything
    /// are not resource failures: a missing path, a directory, a file
    /// llama.cpp cannot read as a model. Real llama.cpp calls; no
    /// weights.
    #[test]
    fn load_failures_before_allocation_are_not_resource_failures() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("missing.gguf");
        let garbage = dir.path().join("garbage.gguf");
        std::fs::write(&garbage, b"not a gguf, not even close").expect("write");

        let error = |path: PathBuf| match LlamaCppEngine::from_path(path) {
            Ok(_) => panic!("loaded a model that isn't one"),
            Err(e) => e,
        };
        let not_found = error(missing);
        assert!(
            matches!(&not_found, NewError::Unreadable { source, .. }
                if source.kind() == std::io::ErrorKind::NotFound),
            "{not_found:?}"
        );
        let directory = error(dir.path().to_path_buf());
        assert!(
            matches!(directory, NewError::Unreadable { .. }),
            "{directory:?}"
        );
        let metadata = error(garbage);
        assert!(
            matches!(metadata, NewError::Metadata { .. }),
            "{metadata:?}"
        );
        for e in [not_found, directory, metadata] {
            assert!(!e.is_resource(), "{e}");
        }
    }

    /// Failures after the backend began allocating are resource
    /// failures — and so is every one llama.cpp leaves unexplained.
    #[test]
    fn load_failures_after_allocation_are_resource_failures() {
        let path = PathBuf::from("model.gguf");
        assert!(NewError::Model { path: path.clone() }.is_resource());
        assert!(NewError::Context.is_resource());
        #[cfg(feature = "mtmd")]
        {
            use crate::llama_cpp::mtmd::MtmdNewError;
            let mtmd = |source| NewError::Mtmd {
                path: path.clone(),
                source,
            };
            let load_failed = MtmdNewError::LoadFailed { path: path.clone() };
            assert!(mtmd(load_failed).is_resource());
            let bad_path = MtmdNewError::BadPath { path: path.clone() };
            assert!(!mtmd(bad_path).is_resource());
        }
    }

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
