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

pub use decoder::{MoefluxDecoder, MoefluxError};
pub use engine::MoefluxEngine;
pub use model::{MoefluxModel, MoefluxModelError};
