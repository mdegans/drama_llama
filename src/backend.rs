//! Backend-agnostic primitives shared across decoder/model implementations.
//!
//! This module defines the types every backend sees: [`Token`] (the
//! canonical token identifier) and [`TokenData`] (a candidate slot
//! carrying an id, logit, and softmaxed probability). Under
//! `cfg(feature = "llama-cpp")` the layout of [`TokenData`] is a
//! contract with llama.cpp's `llama_token_data`: same size, same
//! alignment, same field order, so `&[TokenData]` and
//! `&[llama_token_data]` are transmute-compatible.
//!
//! Traits ([`Decoder`], [`Model`]) will be added here in a later
//! commit.

/// Canonical token identifier used across the crate. Alias for `i32`
/// so it is ABI-compatible with llama.cpp's `llama_token`.
pub type Token = i32;

/// A candidate slot: token id, raw logit, softmaxed probability.
///
/// `#[repr(C)]` with field order identical to llama.cpp's
/// `llama_token_data`. Under `cfg(feature = "llama-cpp")`
/// [`static_assertions`] verify size and alignment match so raw-pointer
/// casts between `*mut TokenData` and `*mut llama_token_data` are
/// sound.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenData {
    pub id: Token,
    pub logit: f32,
    pub p: f32,
}

#[cfg(feature = "llama-cpp")]
mod llama_cpp_abi {
    use super::TokenData;
    use llama_cpp_sys_3::llama_token_data;

    static_assertions::assert_eq_size!(TokenData, llama_token_data);
    static_assertions::assert_eq_align!(TokenData, llama_token_data);
}
