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
// TODO: This doesn't need to be generic for now but should likely be a an
// associated type. Where? Decoder, Model, or Backend? Not sure. If Model never
// uses Token anywhere the choice is clear. If both Model and Decoder need Token
// we have no choice but to put it in both places and add bounds to Backend that
// Model's Token is the same as Decoder's if that's expressable.
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
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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
/// Implementors own whatever session state is required to advance a single
/// sequence (e.g. llama.cpp's `llama_context`). The trait is deliberately
/// narrow — only what the sampling loop and prefix-cache machinery in `Session`
/// / `Predictor` need. Everything else (diagnostics, thread knobs, state
/// ser/de, construction) stays as inherent methods on the concrete implementor.
pub trait Decoder: Send {
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

    /// Maximum number of distinct KV sequences this decoder supports.
    /// `Session`'s multi-slot prefix cache clamps its slot count to
    /// this.
    ///
    /// **No default impl: forces every backend to make an explicit
    /// choice.** llama.cpp reports the context's `n_seq_max` (cells
    /// are physically partitioned or unified per context params);
    /// moeflux reports a large value because its `seq_id` is a
    /// *namespace label only* — one physical stream, realized as
    /// whole-state snapshots per `(seq, pos)` — so any number of
    /// sequences can exist, just never concurrently resident.
    fn n_seq_max(&self) -> u32;

    /// Clear the entire KV cache.
    fn memory_clear(&mut self);

    /// Remove KV entries for `seq_id` in position range `[p0, p1)`.
    /// `p0 < 0` means from 0; `p1 < 0` means to the end. `seq_id < 0`
    /// matches any sequence. Returns `true` on success.
    fn memory_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> bool;

    /// Copy KV entries from `src` seq to `dst` seq in `[p0, p1)`.
    fn memory_seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32);

    /// Keep only `seq_id`'s entries; drop all others.
    fn memory_seq_keep(&mut self, seq_id: i32);

    /// Return the largest position present in the KV cache for
    /// `seq_id`, or a negative value if the sequence is empty.
    fn memory_seq_pos_max(&mut self, seq_id: i32) -> i32;

    /// Snapshot the current decoder state at sequence position `pos`.
    /// Subsequent [`Decoder::restore_to`] with the same `(seq_id,
    /// pos)` rewinds to exactly this state — enabling lossless
    /// breakpoint rewinds for backends (like moeflux) whose
    /// recurrent state cannot be unwound by position alone.
    ///
    /// Backends that already preserve recurrent state per-position
    /// (e.g. llama.cpp) implement this as a no-op.
    fn checkpoint_pos(&mut self, seq_id: i32, pos: i32);

    /// Restore decoder state to a previously-snapshotted position.
    /// On success, KV state matches what was current immediately
    /// after the [`Decoder::checkpoint_pos`] call at `(seq_id, pos)`,
    /// and all snapshots stored at positions `> pos` are dropped
    /// (their futures are now invalid).
    ///
    /// Backends without per-pos snapshots may implement this by
    /// truncating their KV cache directly (llama.cpp does this — its
    /// recurrent state is preserved per-cell, so a truncate is
    /// already lossless).
    ///
    /// Returns [`MemoryRmError::NoCheckpoint`] when no snapshot
    /// exists at exactly `pos` (caller should fall back to
    /// `memory_clear` + full reprefill), or
    /// [`MemoryRmError::BackendUnsupported`] when the backend cannot
    /// honor the request at all.
    fn restore_to(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError>;

    /// Drop the snapshot stored at `(seq_id, pos)` without otherwise
    /// disturbing decoder state.
    ///
    /// `Session`'s prefix-cache uses this to free orphaned snapshots —
    /// the previous "internal tip" set by an earlier `complete_*` call
    /// that is being replaced by a fresher tip, or breakpoints from a
    /// prior call's prompt that are no longer set in the current call.
    /// Without this, snapshots accumulate in backends that maintain a
    /// bounded LRU (moeflux), eventually evicting the system+tools
    /// anchor — the most valuable cross-session prefix.
    ///
    /// **No default impl: forces every backend to make an explicit
    /// choice.** Backends with per-cell recurrent state (llama.cpp)
    /// implement this as a no-op; backends with explicit per-pos
    /// snapshots (moeflux) drop the entry from the snapshot map.
    ///
    /// Returns `Ok(())` even if no snapshot exists at `pos`
    /// (idempotent — `Session` may forget the same position twice
    /// across the eviction trigger paths). Returns
    /// [`MemoryRmError::BackendUnsupported`] only when the backend
    /// fundamentally cannot honor a single-snapshot delete at all.
    fn forget_pos(
        &mut self,
        seq_id: i32,
        pos: i32,
    ) -> Result<(), MemoryRmError>;
}

/// Errors from [`Decoder::restore_to`]. Surfaced by the
/// prefix-cache machinery in `Session` so it can fall back to a full
/// re-prefill when a snapshot is missing.
#[derive(Debug, thiserror::Error)]
pub enum MemoryRmError {
    /// No snapshot at the requested position. Caller should
    /// `memory_clear` and full-reprefill.
    #[error("no checkpoint stored at position {pos}")]
    NoCheckpoint { pos: i32 },
    /// The backend cannot honor a rewind to this position at all
    /// (invalid seq_id, unsupported semantics, etc.). Treated the
    /// same as `NoCheckpoint` by the cache fallback path.
    #[error("backend cannot restore to position {pos}")]
    BackendUnsupported { pos: i32 },
}

/// A frozen, decoded image crossing the backend-agnostic media
/// boundary: tightly-packed RGB8 pixels plus a memoized content hash.
///
/// This is deliberately a plain record, not an image abstraction —
/// `image::DynamicImage` is the pixel source of truth (decode, scale,
/// convert live there), and an `Image` is the hashed RGB8 snapshot of
/// one, produced at the boundary via the `media`-gated `TryFrom`
/// impls. Keeping `image` types out of the trait layer keeps the
/// codec stack out of builds that never touch media.
///
/// `id` is the sha256 of the RGB8 pixel buffer, computed once at
/// construction. Bundling identity with the pixels (instead of
/// passing hashes alongside) makes "pixels and cache identity
/// disagree" unrepresentable — prefix-cache correctness hangs off
/// this hash, and a false hit would serve image A's KV cells for
/// image B.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    rgb8: Vec<u8>,
    width: u32,
    height: u32,
    id: [u8; 32],
}

impl Image {
    /// Construct from a tightly-packed RGB8 buffer (`RGBRGBRGB…`,
    /// `width * height * 3` bytes). Rejects zero dimensions and
    /// length mismatches; computes and memoizes the content hash.
    pub fn from_rgb8(
        rgb8: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<Self, ImageNewError> {
        if width == 0 || height == 0 {
            return Err(ImageNewError::ZeroDim { width, height });
        }
        let expected = width as usize * height as usize * 3;
        if rgb8.len() != expected {
            return Err(ImageNewError::LengthMismatch {
                width,
                height,
                expected,
                actual: rgb8.len(),
            });
        }
        let id = {
            use sha2::{Digest, Sha256};
            Sha256::digest(&rgb8).into()
        };
        Ok(Self {
            rgb8,
            width,
            height,
            id,
        })
    }

    /// Tightly-packed RGB8 pixels, `width * height * 3` bytes.
    pub fn rgb8(&self) -> &[u8] {
        &self.rgb8
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// sha256 of the RGB8 pixel buffer. Feeds prefix-cache identity.
    pub fn id(&self) -> &[u8; 32] {
        &self.id
    }

    /// Dims + identity without the pixels — what the counting /
    /// cache-compare paths consume.
    pub fn info(&self) -> ImageInfo {
        ImageInfo {
            width: self.width,
            height: self.height,
            id: self.id,
        }
    }
}

/// Failure constructing an [`Image`].
#[derive(Debug, thiserror::Error)]
pub enum ImageNewError {
    #[error("image dimensions must be non-zero, got {width}x{height}")]
    ZeroDim { width: u32, height: u32 },
    #[error(
        "RGB8 buffer length {actual} != width {width} * height {height} \
         * 3 = {expected}"
    )]
    LengthMismatch {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
}

/// Dimensions and identity of an [`Image`], without the pixels.
///
/// [`Vision::tokenize_image`] takes these instead of full [`Image`]s:
/// token counts depend only on dimensions, so prompts can be counted,
/// hashed, and cache-compared without decoding or re-preprocessing
/// pixels. The inverse — [`Vision::prefill_image`] requiring a full
/// [`Image`] — enforces "can't encode a placeholder" in the
/// signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    /// sha256 of the RGB8 pixels (see [`Image::id`]).
    pub id: [u8; 32],
}

impl From<&Image> for ImageInfo {
    fn from(image: &Image) -> Self {
        image.info()
    }
}

/// Token/position extent of one media item in the KV cache.
///
/// These differ on M-RoPE models (e.g. Qwen-VL): an image can occupy
/// ~1024 KV cells (`n_tokens`) while advancing the position counter
/// by only ~32 (`n_pos`). Context-fit checks are cell-space
/// (`n_tokens`); position cursors advance by `n_pos`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaSpan {
    pub n_tokens: u32,
    pub n_pos: u32,
}

/// One chunk of a media-aware tokenization: a run of ordinary text
/// tokens, or one media item identified by its content hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaChunk {
    Text(Vec<Token>),
    Media {
        /// Content hash of the source image (see [`Image::id`]).
        id: [u8; 32],
        span: MediaSpan,
    },
}

/// Vision (image-input) capability of a backend.
///
/// Small by design: it traffics only in backend-agnostic types
/// ([`ImageInfo`], [`MediaChunk`], [`Image`]) so generic `Session`
/// code can tokenize and prefill media without naming a concrete
/// backend — and it is deliberately *not* a general multimodal trait
/// family; with one implementor (mtmd) the abstraction stays as small
/// as what `Session` needs from it.
///
/// The decoder is a type parameter rather than an associated type so
/// the uninhabited [`NoVision`] can implement `Vision<D>` for every
/// backend. [`Vision::prefill_image`] takes the decoder as an
/// argument: encoder state lives on the implementor (`&mut self` —
/// mtmd's context owns the encoder output buffer and is not
/// thread-safe, which is also why the instance hangs off [`Engine`]
/// rather than the `Send + Sync` [`Model`]), while the KV write needs
/// the decoder. `Session` owns both via [`Engine`].
///
/// [`Engine`]: crate::Engine
pub trait Vision<D: Decoder>: Send {
    /// Backend-specific media error (segment/count mismatch,
    /// preprocessing failure, encode/decode failure, etc.).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Whether the loaded projector actually supports image input
    /// (a projector can in principle be audio-only).
    fn supports_images(&self) -> bool;

    /// Tokenize ONE image into its chunk sequence: any wrapper text
    /// chunks the model frames media with (`<|vision_start|>` …
    /// `<|vision_end|>`) plus the media chunk itself, in order.
    /// Placeholder-typed: an [`ImageInfo`] carries dims + identity
    /// but no pixels, so counting/cache paths never pay preprocessing.
    ///
    /// This is deliberately image-only — prompt TEXT never crosses
    /// this trait. The caller (`Session`) tokenizes text segments
    /// through [`Model::tokenize`] / [`Model::tokenize_special`] and
    /// interleaves; the vision backend sees nothing it could
    /// misinterpret as a media marker, making marker injection
    /// structurally impossible (content containing mtmd's literal
    /// `<__media__>` is inert prose).
    fn tokenize_image(
        &self,
        image: &ImageInfo,
        parse_special: bool,
    ) -> Result<Vec<MediaChunk>, Self::Error>;

    /// Encode `image` and write its embeddings into the KV cache
    /// starting at `start_pos` on `seq_id`. Returns the extent
    /// written; the caller advances its position cursor by
    /// [`MediaSpan::n_pos`] (not `n_tokens` — they differ on M-RoPE
    /// models).
    fn prefill_image(
        &mut self,
        decoder: &mut D,
        image: &Image,
        start_pos: usize,
        seq_id: i32,
    ) -> Result<MediaSpan, Self::Error>;
}

/// [`Vision`] type for backends without one. Uninhabited, so an
/// `Option<NoVision>` is statically `None`: the "media unsupported"
/// branch in generic code is free and cannot be misused.
#[derive(Debug, Clone, Copy)]
pub enum NoVision {}

impl<D: Decoder> Vision<D> for NoVision {
    type Error = std::convert::Infallible;

    fn supports_images(&self) -> bool {
        match *self {}
    }

    fn tokenize_image(
        &self,
        _image: &ImageInfo,
        _parse_special: bool,
    ) -> Result<Vec<MediaChunk>, Self::Error> {
        match *self {}
    }

    fn prefill_image(
        &mut self,
        _decoder: &mut D,
        _image: &Image,
        _start_pos: usize,
        _seq_id: i32,
    ) -> Result<MediaSpan, Self::Error> {
        match *self {}
    }
}

/// Conversions from the two upstream image sources: the `image` crate
/// (pixel source of truth) and misanthropic's API-shaped
/// [`Image`](misanthropic::prompt::message::Image) blocks. Gated on
/// `media` so the codec stack stays out of builds that never touch
/// images; decoding always goes through the `image` crate, never
/// mtmd's bundled stb_image.
#[cfg(feature = "media")]
mod media_convert {
    use super::{Image, ImageNewError};

    /// Failure decoding an API image block into an [`Image`].
    #[derive(Debug, thiserror::Error)]
    pub enum ImageDecodeError {
        /// Base64/codec failure, or a URL source (we never fetch —
        /// download it yourself and re-embed as base64).
        #[error(transparent)]
        Decode(#[from] misanthropic::prompt::message::ImageDecodeError),
        /// Decoded pixels failed [`Image`] validation.
        #[error(transparent)]
        Image(#[from] ImageNewError),
    }

    impl TryFrom<image::DynamicImage> for Image {
        type Error = ImageNewError;

        /// Snapshot any `DynamicImage` into a hashed RGB8 record.
        /// Fallible only for zero-dimension images.
        fn try_from(image: image::DynamicImage) -> Result<Self, Self::Error> {
            let rgb8 = image.to_rgb8();
            let (width, height) = rgb8.dimensions();
            Image::from_rgb8(rgb8.into_raw(), width, height)
        }
    }

    impl TryFrom<&misanthropic::prompt::message::Image> for Image {
        type Error = ImageDecodeError;

        fn try_from(
            image: &misanthropic::prompt::message::Image,
        ) -> Result<Self, Self::Error> {
            let rgba = image.decode()?;
            Ok(image::DynamicImage::ImageRgba8(rgba).try_into()?)
        }
    }
}

#[cfg(feature = "media")]
pub use media_convert::ImageDecodeError;

/// Model state: tokenization, vocab introspection, chat-template source, and
/// non-decode-specific metadata. Immutable.
pub trait Model: Send + Sync {
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

    /// [`Model::tokenize`] with explicit control of the tokenizer's
    /// automatic leading specials (BOS on models whose vocab requests
    /// it). Needed by callers assembling one token stream from
    /// multiple text pieces — the media segment path — where only the
    /// FIRST piece may carry the leading specials; `tokenize` would
    /// re-add them to every piece on BOS-adding models.
    ///
    /// Default delegates to `tokenize` ignoring `add_special`,
    /// correct for backends whose tokenizer never auto-prepends
    /// (moeflux) — backends where it can (llama.cpp) must override.
    fn tokenize_special(
        &self,
        input: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<Token> {
        let _ = add_special;
        self.tokenize(input, parse_special)
    }

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

    /// Every token that ENDS GENERATION for this model — the complete
    /// set, `eos` and `eot` included. This is the single authority for
    /// "should I stop here?" and "may this token be masked while a
    /// constraint is open?"; ask it, never `eos()`/`eot()`.
    ///
    /// It is deliberately NOT `{eos} ∪ {eot} ∪ extras`. A model's EOT
    /// is whatever its vocab *calls* an end-of-turn marker, which is
    /// not the same question as whether emitting it ends the turn.
    /// gpt-oss is the case in point: llama.cpp auto-detects EOT by
    /// text and `<|end|>` is on that list, so `eot()` is `<|end|>` —
    /// the Harmony *in-stream channel separator*, which upstream then
    /// removes from its EOG set precisely so the model can close an
    /// analysis channel and keep going. Union `eot` back in and a
    /// Harmony turn dies right after the reasoning block, or (under a
    /// tool grammar, where this set is also the mask-while-incomplete
    /// set) the model can never close the channel at all and burns to
    /// `max_tokens`. Multi-EOS vocabs are covered too: Qwen3 declares
    /// `eos_token_id` as `[<|im_end|>, <|endoftext|>]`, and the
    /// secondary decodes to an empty piece, so missing it means an
    /// invisible loop.
    ///
    /// Backends must report their own truth rather than derive it:
    /// `LlamaCppModel` returns libllama's `special_eog_ids` verbatim
    /// (`llama_vocab_is_eog`), quirks and workarounds included.
    ///
    /// Consumed by `PredictOptions::add_model_stops` (stop sequences),
    /// the grammar / JSON candidate filters (mid-constraint EOG mask),
    /// and Session's piece filter (sentinels dropped from the surfaced
    /// output).
    fn eog_tokens(&self) -> Vec<Token>;

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

/// Bundle of [`Decoder`] and [`Model`] implementations that together form a
/// `Backend`. Used with [`Engine`] and [`Session`].
///
/// [`Engine`]: crate::Engine
/// [`Session`]: crate::Session
pub trait Backend {
    const NAME: &'static str;

    /// Concrete [`Decoder`] type for this [`Backend`]. `Send` so the engine can
    /// move between threads; not `Sync` (decode mutates state).
    type Decoder: Decoder;
    /// Concrete [`Model`] type for this [`Backend`]. `Send + Sync` — vocab and
    /// tokenizer are read concurrently by Iterator impls.
    type Model: Model;
    /// Concrete [`Vision`] capability type for this [`Backend`].
    /// Backends without image support use [`NoVision`]. The instance
    /// (if any) is owned by [`Engine`], declared to drop before the
    /// decoder and model it references.
    ///
    /// [`Engine`]: crate::Engine
    type Vision: Vision<Self::Decoder>;

    /// Return `true` if a file or directory is supported by the [`Backend`].
    fn is_supported_model(name: &str, meta: &std::fs::Metadata) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_from_rgb8_validates() {
        assert!(matches!(
            Image::from_rgb8(vec![], 0, 1),
            Err(ImageNewError::ZeroDim { .. })
        ));
        assert!(matches!(
            Image::from_rgb8(vec![0; 5], 1, 2),
            Err(ImageNewError::LengthMismatch { .. })
        ));
        let img = Image::from_rgb8(vec![7; 12], 2, 2).unwrap();
        assert_eq!((img.width(), img.height()), (2, 2));
        assert_eq!(img.rgb8().len(), 12);
    }

    #[test]
    fn image_id_tracks_pixels() {
        let a = Image::from_rgb8(vec![0; 12], 2, 2).unwrap();
        let b = Image::from_rgb8(vec![0; 12], 2, 2).unwrap();
        let c = Image::from_rgb8(vec![1; 12], 2, 2).unwrap();
        assert_eq!(a.id(), b.id());
        assert_ne!(a.id(), c.id());
        assert_eq!(a.info().id, *a.id());
        // Identity is pixel-buffer based; geometry travels alongside
        // in ImageInfo, so same bytes / different dims share an id but
        // never an info.
        let d = Image::from_rgb8(vec![0; 12], 4, 1).unwrap();
        assert_eq!(d.id(), a.id());
        assert_ne!(d.info(), a.info());
    }

    #[cfg(feature = "media")]
    #[test]
    fn image_try_from_dynamic() {
        let img =
            Image::try_from(image::DynamicImage::new_rgba8(3, 2)).unwrap();
        assert_eq!((img.width(), img.height()), (3, 2));
        assert!(matches!(
            Image::try_from(image::DynamicImage::new_rgba8(0, 0)),
            Err(ImageNewError::ZeroDim { .. })
        ));
    }

    #[cfg(feature = "media")]
    #[test]
    fn image_from_misanthropic_roundtrip() {
        use misanthropic::prompt::message;

        // One opaque red pixel, one semi-transparent green one — PNG
        // is lossless and to_rgb8 drops alpha without compositing.
        let mut rgba = image::RgbaImage::new(2, 1);
        rgba.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        rgba.put_pixel(1, 0, image::Rgba([0, 255, 0, 128]));
        let api = message::Image::encode(message::MediaType::Png, rgba)
            .expect("png encode");

        let img = Image::try_from(&api).unwrap();
        assert_eq!((img.width(), img.height()), (2, 1));
        assert_eq!(img.rgb8(), &[255, 0, 0, 0, 255, 0]);

        // URL sources are never fetched — typed error.
        let url = message::Image::Url {
            url: "https://example.com/x.png".into(),
        };
        assert!(Image::try_from(&url).is_err());
    }
}
