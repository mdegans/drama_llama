//! [`llama_cpp_sys_3`] [`Decoder`] and [`Model`] [`Backend`].
//!
//! [`Decoder`]: crate::Decoder
//! [`Model`]: crate::Model

pub mod decoder;
pub mod engine;
pub mod model;
#[cfg(feature = "mtmd")]
pub mod mtmd;
pub mod options;

pub use crate::Backend;
pub use decoder::{DecodeError, FlashAttention, LlamaCppDecoder, NewError};
pub use engine::LlamaCppEngine;
pub use model::{llama_quantize, LlamaCppModel};
#[cfg(feature = "mtmd")]
pub use mtmd::{Mtmd, MtmdParams};
pub use options::LlamaCppOptions;

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
        // `<model>.mmproj.gguf` is a vision *projector* sidecar, not a
        // standalone model — it auto-loads alongside its base model and
        // fails if asked to load on its own. Exclude it so it never
        // surfaces in `/api/tags` (or gets picked as a default).
        meta.is_file()
            && name.ends_with(".gguf")
            && !name.ends_with(".mmproj.gguf")
    }

    /// Routes both llama.cpp's and ggml's sinks — they are separate
    /// globals upstream and this installs the same trampoline in each.
    fn set_log_callback<F>(f: F) -> Result<(), crate::backend::NotImplemented>
    where
        F: Fn(crate::LogLevel, &str) + Send + Sync + 'static,
    {
        crate::log::set_log_callback(f);
        Ok(())
    }

    fn clear_log_callback() -> Result<(), crate::backend::NotImplemented> {
        crate::log::clear_log_callback();
        Ok(())
    }
}
