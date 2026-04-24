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
