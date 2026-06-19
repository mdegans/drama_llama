//! [`moeflux`] [`Decoder`] and [`Model`] [`Backend`].
//!
//! [`Decoder`]: crate::Decoder
//! [`Model`]: crate::Model

pub mod decoder;
pub mod engine;
pub mod model;

use crate::Backend;

pub use decoder::{MoefluxDecoder, MoefluxError, PrefetchStats};
pub use engine::MoefluxEngine;
pub use model::{MoefluxModel, MoefluxModelError};

/// Tag for the moeflux [`Backend`]. Use as a type parameter for [`Engine`] or
/// [`Session`].
///
/// [`Engine`]: crate::Engine
/// [`Session`]: crate::Session
#[derive(Debug, Clone, Copy)]
pub struct MoefluxBackend;

impl Backend for MoefluxBackend {
    const NAME: &'static str = "moeflux";

    type Decoder = MoefluxDecoder;
    type Model = MoefluxModel;

    fn is_supported_model(_: &str, meta: &std::fs::Metadata) -> bool {
        // This is cheap and if the required files aren't here we'll fail later
        // with an std::io::Error.
        meta.is_dir()
    }
}
