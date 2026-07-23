//! Load-time options for the moeflux backend.
//!
//! Deliberately almost empty. moeflux is compile-time variant-selected —
//! its context length is `variants::MAX_SEQ_LEN`, a constant baked into
//! the build, and its MoE top-K is model shape rather than a parameter —
//! so there is exactly one thing a caller gets to choose at load time.
//!
//! This being nearly vacant is the honest answer, not a gap to fill.
//! Resist the urge to mirror `LlamaCppOptions`
//! here: a field moeflux ignores would be a lie told in a type
//! signature. `n_ctx` earns a place the day moeflux can size its KV
//! allocation at runtime, and not before.

/// Load-time configuration for [`MoefluxEngine`](crate::MoefluxEngine)
/// and [`Session<MoefluxBackend>`](crate::Session).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[cfg_attr(feature = "cli", derive(clap::Args))]
#[non_exhaustive]
pub struct MoefluxOptions {
    /// Select the 2-bit packed-experts layout. Defaults to `false` — the
    /// 4-bit Qwen3 MoE setup, which is what the on-disk artifacts use.
    ///
    /// Must match how the experts were packed: reading a 4-bit
    /// `packed_experts/` as 2-bit does not fail, it produces garbage.
    #[cfg_attr(feature = "cli", arg(long))]
    pub use_2bit: bool,
}

impl MoefluxOptions {
    /// Select the 2-bit packed-experts layout. See [`Self::use_2bit`].
    pub fn with_use_2bit(mut self, use_2bit: bool) -> Self {
        self.use_2bit = use_2bit;
        self
    }
}
