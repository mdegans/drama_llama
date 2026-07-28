//! Load-time options for the llama.cpp backend.
//!
//! Everything llama.cpp needs to be told *while* a model and its context
//! are being built, in one place. This is what
//! [`FromPath::Options`](crate::FromPath) resolves to for
//! [`LlamaCppBackend`](crate::LlamaCppBackend), so generic code — code
//! that has a `B: Backend` and no idea which one — can still ask for a
//! context size. That was impossible while every knob lived on an
//! inherent constructor.
//!
//! Every field is "unset" by default and unset means *llama.cpp's*
//! default, not ours. A `Default::default()` load is byte-for-byte the
//! load you would have got from the old `from_path`.

use llama_cpp_sys_3::{
    llama_context_params, llama_model_default_params, llama_model_params,
};

use crate::llama_cpp::{FlashAttention, LlamaCppEngine};

/// Load-time configuration for [`LlamaCppEngine`] and
/// [`Session<LlamaCppBackend>`](crate::Session).
///
/// Serializable (with the `serde` feature) so a deployment can keep its
/// model configuration in a file, and `clap::Args` (with `cli`) so a
/// single-backend binary can `#[command(flatten)]` it straight into its
/// own arguments. A binary that picks its backend at *runtime* cannot
/// flatten this — clap needs one concrete type — and should flatten
/// [`BackendArgs`](crate::cli::BackendArgs) instead, then convert.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[cfg_attr(feature = "cli", derive(clap::Args))]
#[non_exhaustive]
pub struct LlamaCppOptions {
    /// KV context size in tokens. `None` inherits llama.cpp's default,
    /// which is **512** — small enough to truncate a chat or a
    /// structured-output run well before it finishes. Set this to
    /// something realistic (4096 – 16384) for anything reasoning-shaped.
    #[cfg_attr(feature = "cli", arg(long))]
    pub n_ctx: Option<u32>,

    /// Number of KV sequences to keep over one unified cell pool —
    /// llama.cpp's recommended shape for sequences sharing a large
    /// prefix, and what makes an N-agent workload cache N prefixes
    /// instead of thrashing one.
    /// [`Session::with_prefix_cache`](crate::Session::with_prefix_cache)
    /// sizes its slot count from this.
    ///
    /// `Some(1)` is **not** the same as `None`: it also switches the KV
    /// cache to unified. Ask for slots only when you want both.
    #[cfg_attr(feature = "cli", arg(long))]
    pub cache_slots: Option<u32>,

    /// Flash Attention policy. `None` inherits llama.cpp's
    /// [`Auto`](FlashAttention::Auto). A diagnostic knob — see
    /// [`FlashAttention`] for when forcing it off tells you something.
    #[cfg_attr(feature = "cli", arg(long, value_enum))]
    pub flash_attn: Option<FlashAttention>,

    /// Micro-batch size (llama.cpp's `n_ubatch`) — how many tokens one
    /// forward pass evaluates. `None` inherits the library default
    /// (512, capped to [`Self::n_ctx`] when that is set).
    ///
    /// Normally you want this large: it is the prefill parallelism
    /// knob. It is settable because it is also a **correctness escape
    /// hatch** for backend kernels that only misbehave above a batch
    /// threshold.
    ///
    /// The case that motivated exposing it: Mistral Small 4
    /// (`mistral4`) on Metal returns an entirely NaN vocabulary for any
    /// micro-batch of >=32 tokens. At 32 the MoE matmul switches from
    /// `mul_mv_id` to the half-precision simdgroup-MMA `mul_mm_id`
    /// (`ne21_mm_id_min`), and this model's layer-32 activations exceed
    /// f16's 65504 ceiling — they overflow to inf, and inf arithmetic
    /// yields NaN. The `mul_mv_id` path carries the same values in f32
    /// and is correct. `n_ubatch = 31` therefore trades prefill speed
    /// for a working model; see
    /// `.claude/memory/mistral4_support_and_metal_nan.md`.
    ///
    /// Note this is the *micro*-batch: `n_batch` stays large, so a
    /// prompt of any length still submits in one call and is simply
    /// evaluated in smaller chunks.
    #[cfg_attr(feature = "cli", arg(long))]
    pub n_ubatch: Option<u32>,

    /// Keep every layer on the CPU. llama.cpp offloads all layers by
    /// default (`n_gpu_layers = -1`); this forces zero. Diagnostic path
    /// for isolating GPU-kernel divergence.
    #[cfg_attr(feature = "cli", arg(long))]
    pub no_gpu: bool,

    /// NUMA strategy passed through to `llama_numa_init`. Raw
    /// `ggml_numa_strategy` value; `None` leaves llama.cpp alone.
    ///
    /// Not a flag: it is an unlabelled C enum ordinal, so `--numa 3` is
    /// worse than useless to a human. Set it in code or via serde.
    #[cfg_attr(feature = "cli", arg(skip))]
    pub numa: Option<u32>,
}

impl LlamaCppOptions {
    /// Set the KV context size. See [`Self::n_ctx`].
    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = Some(n_ctx);
        self
    }

    /// Ask for `slots` KV sequences over a unified cell pool. See
    /// [`Self::cache_slots`] — note this is not a no-op at `1`.
    pub fn with_cache_slots(mut self, slots: u32) -> Self {
        self.cache_slots = Some(slots);
        self
    }

    /// Force a Flash Attention policy. See [`Self::flash_attn`].
    pub fn with_flash_attention(mut self, fa: FlashAttention) -> Self {
        self.flash_attn = Some(fa);
        self
    }

    /// Set the micro-batch size. See [`Self::n_ubatch`] — this is a
    /// correctness escape hatch as much as a perf knob.
    pub fn with_n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.n_ubatch = Some(n_ubatch);
        self
    }

    /// Keep every layer on the CPU. See [`Self::no_gpu`].
    pub fn cpu_only(mut self) -> Self {
        self.no_gpu = true;
        self
    }

    /// Set the NUMA strategy. See [`Self::numa`].
    pub fn with_numa(mut self, numa: u32) -> Self {
        self.numa = Some(numa);
        self
    }

    /// Build the `llama_model_params` these options describe.
    pub fn model_params(&self) -> llama_model_params {
        // SAFETY: returns POD and allocates nothing for the pointer
        // fields, which are optional and initialized to null.
        let mut mp = unsafe { llama_model_default_params() };
        if self.no_gpu {
            mp.n_gpu_layers = 0;
        }
        mp
    }

    /// Build the `llama_context_params` these options describe, on top
    /// of [`LlamaCppEngine::default_context_params`] (which is the
    /// library default plus a usable thread count).
    pub fn context_params(&self) -> llama_context_params {
        let mut cp = LlamaCppEngine::default_context_params();
        if let Some(n_ctx) = self.n_ctx {
            cp.n_ctx = n_ctx;
            // Batch sizes track the context: a batch larger than the
            // context can never be submitted, and `n_ubatch` above
            // `n_ctx` over-allocates compute buffers.
            cp.n_batch = n_ctx;
            cp.n_ubatch = cp.n_ubatch.min(n_ctx);
        }
        if let Some(slots) = self.cache_slots {
            cp.n_seq_max = slots.max(1);
            cp.kv_unified = true;
        }
        if let Some(fa) = self.flash_attn {
            cp.flash_attn_type = fa.as_raw();
        }
        // Applied last so an explicit micro-batch wins over the
        // `n_ctx` clamp above — the whole point of the knob is to force
        // a *small* ubatch under a large context.
        if let Some(n_ubatch) = self.n_ubatch {
            cp.n_ubatch = n_ubatch.max(1);
        }
        cp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole "unset means llama.cpp's default" contract: a default
    /// `LlamaCppOptions` must not perturb anything the library chose for
    /// us. If this breaks, every existing `from_path` caller silently
    /// changed behaviour.
    #[test]
    fn default_options_perturb_nothing() {
        let opts = LlamaCppOptions::default();
        let mp = opts.model_params();
        // SAFETY: POD, no allocation.
        let default_mp = unsafe { llama_model_default_params() };
        assert_eq!(mp.n_gpu_layers, default_mp.n_gpu_layers);

        let cp = opts.context_params();
        let default_cp = LlamaCppEngine::default_context_params();
        assert_eq!(cp.n_ctx, default_cp.n_ctx);
        assert_eq!(cp.n_batch, default_cp.n_batch);
        assert_eq!(cp.n_ubatch, default_cp.n_ubatch);
        assert_eq!(cp.n_seq_max, default_cp.n_seq_max);
        assert_eq!(cp.kv_unified, default_cp.kv_unified);
        assert_eq!(cp.flash_attn_type, default_cp.flash_attn_type);
    }

    /// `cache_slots: Some(1)` is not the identity — it flips the KV
    /// cache to unified. This was a documented footgun back when it was
    /// reachable only via a three-argument constructor; keep it pinned
    /// now that it is a field.
    #[test]
    fn one_cache_slot_still_unifies_the_kv_cache() {
        let plain = LlamaCppOptions::default().with_n_ctx(4096);
        let one_slot = plain.with_cache_slots(1);
        assert!(!plain.context_params().kv_unified);
        assert!(one_slot.context_params().kv_unified);
        assert_eq!(one_slot.context_params().n_seq_max, 1);
    }

    /// `n_ubatch` is clamped, never raised: a small context must not
    /// grow the physical batch past the library default.
    #[test]
    fn n_ctx_clamps_but_never_raises_n_ubatch() {
        let default_ubatch = LlamaCppEngine::default_context_params().n_ubatch;
        let small = LlamaCppOptions::default().with_n_ctx(64).context_params();
        assert_eq!(small.n_ubatch, default_ubatch.min(64));
        let big = LlamaCppOptions::default()
            .with_n_ctx(32768)
            .context_params();
        assert_eq!(big.n_ubatch, default_ubatch);
        assert_eq!(big.n_batch, 32768);
    }

    #[test]
    fn cpu_only_zeroes_gpu_layers() {
        assert_eq!(
            LlamaCppOptions::default()
                .cpu_only()
                .model_params()
                .n_gpu_layers,
            0
        );
    }
}
