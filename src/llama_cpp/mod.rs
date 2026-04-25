//! llama.cpp-backed [`crate::backend::Decoder`] and
//! [`crate::backend::Model`] implementations.
//!
//! This module contains every direct dependency on `llama_cpp_sys_3`.
//! Gated by the `llama-cpp` cargo feature (wired in a follow-up commit;
//! currently always compiled).

pub mod decoder;
pub mod engine;
pub mod model;

pub use decoder::{
    restore_default_logs, silence_logs, DecodeError, FlashAttention,
    LlamaCppDecoder, NewError,
};
pub use engine::LlamaCppEngine;
pub use model::{llama_quantize, LlamaCppModel};

/// Zero-sized [`crate::Backend`] tag for the llama.cpp backend.
/// Use as the type parameter in `Engine<LlamaCppBackend>` or
/// `Session<LlamaCppBackend>` (or via the `LlamaCppEngine` alias) to
/// monomorphize against the llama.cpp decoder + model pair.
#[derive(Debug, Clone, Copy)]
pub struct LlamaCppBackend;

impl crate::backend::Backend for LlamaCppBackend {
    type Decoder = LlamaCppDecoder;
    type Model = LlamaCppModel;
}
