//! moeflux-backed [`crate::backend::Decoder`] and
//! [`crate::backend::Model`] implementations.
//!
//! Gated behind the `moeflux` feature + `target_os = "macos"`.
//! Phase 4 of the v0.8.0 backend-split plan: plugs the Metal
//! streaming-MoE runtime (see
//! [moeflux](https://github.com/mdegans/moeflux)) into drama_llama's
//! backend-agnostic Engine. Tokenization happens Rust-side via the
//! HuggingFace `tokenizers` crate; moeflux only sees token IDs.

pub mod decoder;
pub mod engine;
pub mod model;

pub use decoder::{MoefluxDecoder, MoefluxError, PrefetchStats};
pub use engine::MoefluxEngine;
pub use model::{MoefluxModel, MoefluxModelError};

/// Zero-sized [`crate::Backend`] tag for the moeflux backend. Use as
/// the type parameter in `Engine<MoefluxBackend>` or
/// `Session<MoefluxBackend>` (or via the `MoefluxEngine` alias) to
/// monomorphize against the moeflux Metal decoder + HF tokenizer
/// pair.
#[derive(Debug, Clone, Copy)]
pub struct MoefluxBackend;

impl crate::backend::Backend for MoefluxBackend {
    const NAME: &'static str = "moeflux";

    type Decoder = MoefluxDecoder;
    type Model = MoefluxModel;

    fn is_supported(_: &str, meta: &std::fs::Metadata) -> bool {
        // This is cheap and if the required files aren't here we'll fail later
        // with an std::io::Error.
        meta.is_dir()
    }
}
