//! [`llama_cpp_sys_3`] [`Decoder`] and [`Model`] [`Backend`].
//!
//! [`Decoder`]: crate::Decoder
//! [`Model`]: crate::Model

pub mod decoder;
pub mod engine;
pub mod model;
#[cfg(feature = "mtmd")]
pub mod mtmd;

pub use crate::Backend;
pub use decoder::{DecodeError, FlashAttention, LlamaCppDecoder, NewError};
pub use engine::LlamaCppEngine;
pub use model::{llama_quantize, LlamaCppModel};
#[cfg(feature = "mtmd")]
pub use mtmd::{Mtmd, MtmdParams};

/// Tag for the llama-cpp [`Backend`]. Use as a type parameter for [`Engine`] or
/// [`Session`].
///
/// [`Engine`]: crate::Engine
/// [`Session`]: crate::Session
#[derive(Debug, Clone, Copy)]
pub struct LlamaCppBackend;

impl Backend for LlamaCppBackend {
    const NAME: &'static str = "llama-cpp";
    type Decoder = LlamaCppDecoder;
    type Model = LlamaCppModel;
    #[cfg(feature = "mtmd")]
    type Vision = mtmd::Mtmd;
    #[cfg(not(feature = "mtmd"))]
    type Vision = crate::NoVision;

    fn is_supported_model(name: &str, meta: &std::fs::Metadata) -> bool {
        meta.is_file() && name.ends_with(".gguf")
    }
}
