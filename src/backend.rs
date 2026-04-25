//! Backend-agnostic primitives shared across decoder/model implementations.
//!
//! This module defines the types and traits every backend sees:
//! [`Token`] (the canonical token identifier), [`TokenData`] (a
//! candidate slot carrying id, logit, and softmaxed probability),
//! [`Decoder`] (produces logits, manages KV cache), and [`Model`]
//! (tokenization, vocab introspection, metadata). Under
//! `cfg(feature = "llama-cpp")` the layout of [`TokenData`] is a
//! contract with llama.cpp's `llama_token_data`: same size, same
//! alignment, same field order, so `&[TokenData]` and
//! `&[llama_token_data]` are transmute-compatible.

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

/// Decode backend: produces logits and manages the KV cache.
///
/// Implementors own whatever session state is required to advance a
/// single sequence (e.g. llama.cpp's `llama_context`). The trait is
/// deliberately narrow — only what the sampling loop and prefix-cache
/// machinery in `Session` / `Predictor` need. Everything else
/// (diagnostics, thread knobs, state ser/de, construction) stays as
/// inherent methods on the concrete implementor.
pub trait Decoder {
    /// Backend-specific decode error (KV-full, bad position, etc.).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Prefill `tokens` at positions
    /// `[start_pos, start_pos + tokens.len())` on `seq_id`. Returns
    /// logits for the last prefilled token (index `tokens.len() - 1`).
    /// Does not clear the KV cache. An empty `tokens` slice is a
    /// no-op returning an empty slice.
    ///
    /// The returned slice borrows from the decoder's internal logit
    /// buffer and is invalidated by the next mutating call (another
    /// `prefill`, `step`, or `memory_*`).
    fn prefill(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
        seq_id: i32,
    ) -> Result<&[f32], Self::Error>;

    /// Advance one token at `pos` on `seq_id`. Returns the next-token
    /// logits. Same borrow-invalidation rule as [`Decoder::prefill`].
    fn step(
        &mut self,
        token: Token,
        pos: usize,
        seq_id: i32,
    ) -> Result<&[f32], Self::Error>;

    /// Context length in tokens.
    fn n_ctx(&self) -> u32;

    /// Clear the entire KV cache.
    fn memory_clear(&mut self);

    /// Remove KV entries for `seq_id` in position range `[p0, p1)`.
    /// `p0 < 0` means from 0; `p1 < 0` means to the end. `seq_id < 0`
    /// matches any sequence. Returns `true` on success.
    fn memory_seq_rm(
        &mut self,
        seq_id: i32,
        p0: i32,
        p1: i32,
    ) -> bool;

    /// Copy KV entries from `src` seq to `dst` seq in `[p0, p1)`.
    fn memory_seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32);

    /// Keep only `seq_id`'s entries; drop all others.
    fn memory_seq_keep(&mut self, seq_id: i32);

    /// Return the largest position present in the KV cache for
    /// `seq_id`, or a negative value if the sequence is empty.
    fn memory_seq_pos_max(&mut self, seq_id: i32) -> i32;
}

/// Model backend: tokenization, vocab introspection, chat-template
/// source, and non-decode-specific metadata.
///
/// Implementors are immutable in the happy path — a loaded model's
/// vocabulary and tokenizer do not change. Methods take `&self`
/// throughout. An [`Error`] associated type is provided for future
/// fallible operations (e.g. tokenizer streaming) but every current
/// method is infallible.
///
/// [`Error`]: Model::Error
pub trait Model {
    /// Backend-specific model error. For infallible backends, use
    /// `std::convert::Infallible`.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Vocabulary size.
    fn n_vocab(&self) -> i32;

    /// Beginning-of-sequence token.
    fn bos(&self) -> Token;

    /// End-of-sequence token.
    fn eos(&self) -> Token;

    /// End-of-turn token (e.g. `<|eot_id|>` for Llama-3-style
    /// chat-tuned models). May equal `eos()` for models without a
    /// distinct end-of-turn marker.
    fn eot(&self) -> Token;

    /// All control + user-defined special tokens. Used by repetition
    /// penalties to avoid penalizing format tokens.
    fn special_tokens(&self) -> Vec<Token>;

    /// Longest piece length in the vocabulary. Used to size
    /// stop-string search windows.
    fn max_token_len(&self) -> usize;

    /// Tokenize `input`. If `special` is true, the tokenizer emits
    /// special / control tokens when they appear verbatim in the
    /// input; otherwise they are treated as plain text.
    fn tokenize(&self, input: &str, special: bool) -> Vec<Token>;

    /// Convert a single token to its UTF-8 piece (may be empty).
    fn token_to_piece(&self, token: Token) -> String;

    /// Write the raw bytes for `token`'s piece into `buf`, resizing
    /// `buf` to exactly the piece length. Hot-path variant of
    /// [`Model::token_to_piece`] used by grammar / JSON filters:
    /// avoids the UTF-8 validation round-trip through `String` and
    /// lets the caller reuse a single byte buffer across a whole
    /// candidate sweep. Bytes are not required to be valid UTF-8 on
    /// their own (multi-byte characters can split across pieces).
    fn token_to_piece_ref(&self, token: Token, buf: &mut Vec<u8>);

    /// Context length the model was trained with.
    fn context_size(&self) -> i32;

    /// Raw Jinja source for the chat template (GGUF
    /// `tokenizer.chat_template` metadata), if any.
    fn chat_template_source(&self) -> Option<String>;

    /// Generic GGUF-style metadata lookup by string key. Returns the
    /// value as a string, or `None` if missing.
    fn get_meta(&self, key: &str) -> Option<String>;

    /// Human-readable identifier for this loaded model. Used by
    /// servers (e.g. `blallama`) to populate the `model` field of API
    /// responses.
    ///
    /// Convention: each backend returns a basename derived from the
    /// load path. `LlamaCppModel` uses the GGUF file basename;
    /// `MoefluxModel` uses the MLX-export directory basename. Returns
    /// `None` when the model was constructed without a known load
    /// path.
    fn display_name(&self) -> Option<String> {
        None
    }
}

/// Bundle of decoder + model implementations that together form a
/// backend. Plugs the typed pair into [`crate::Engine`] and
/// [`crate::Session`] via a single generic parameter.
///
/// Implementors are zero-sized type tags (e.g. [`crate::LlamaCppBackend`])
/// — the actual work is in the associated [`Decoder`] and [`Model`]
/// implementations. Compile-time monomorphization: every method on
/// `Engine<B>` and `Session<B>` resolves to the concrete `B::Decoder`
/// / `B::Model` impl with no runtime dispatch.
///
/// Decoders need `Send` so an `Engine` can move across `await` points
/// (e.g. into `tokio::task::spawn_blocking`). `Sync` is intentionally
/// *not* required: llama.cpp's `*mut llama_context` is internally
/// mutable and unsound to share across threads, and decoders are only
/// ever accessed through `&mut` anyway.
///
/// Models need both `Send + Sync` because the Predictor family's
/// `Iterator` impls hand `&Model` to grammar / sampling code that may
/// observe it from multiple positions. Both real-world model impls
/// (`LlamaCppModel`, `MoefluxModel`) carry explicit
/// `unsafe impl Sync` declarations.
///
/// Baking these bounds here lets consumers (`Engine<B>`, `Session<B>`,
/// the Predictor family) drop per-site where-clauses.
pub trait Backend {
    /// Concrete decoder type for this backend. `Send` so the engine
    /// can move between threads; not `Sync` (decode mutates state).
    type Decoder: Decoder + Send;
    /// Concrete model type for this backend. `Send + Sync` —
    /// vocab and tokenizer are read concurrently by Iterator impls.
    type Model: Model + Send + Sync;
}
