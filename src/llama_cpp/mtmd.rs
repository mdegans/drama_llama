//! Safe wrapper around llama.cpp's multimodal library (`libmtmd`).
//!
//! [`Mtmd`] implements the backend-agnostic [`Vision`] trait for the
//! llama.cpp backend: it turns marker-bearing prompt text plus
//! [`ImageInfo`]s into [`MediaChunk`]s (tokenization / counting, no
//! pixels needed) and encodes real [`Image`]s into the KV cache
//! (prefill). All mtmd handle types (`mtmd_bitmap`,
//! `mtmd_input_chunks`) stay private to this module — the public
//! surface traffics only in the generic types from [`crate::backend`].
//!
//! Image *decoding* (JPEG/PNG → RGB8) never happens here: pixels
//! arrive pre-decoded in an [`Image`], produced by the `image` crate
//! upstream. mtmd's bundled stb_image is intentionally unused
//! (CVE history; adversarial-input posture).

use std::{
    ffi::{CStr, CString},
    path::Path,
    ptr::NonNull,
};

use llama_cpp_sys_3::{
    llama_n_batch, llama_pos, mtmd_bitmap, mtmd_bitmap_free, mtmd_bitmap_init,
    mtmd_bitmap_set_id, mtmd_context, mtmd_context_params_default,
    mtmd_decode_use_mrope, mtmd_free, mtmd_get_marker,
    mtmd_helper_eval_chunk_single, mtmd_init_from_file, mtmd_input_chunk,
    mtmd_input_chunk_get_id, mtmd_input_chunk_get_n_pos,
    mtmd_input_chunk_get_n_tokens, mtmd_input_chunk_get_tokens_text,
    mtmd_input_chunk_get_type,
    mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT, mtmd_input_chunks,
    mtmd_input_chunks_free, mtmd_input_chunks_get, mtmd_input_chunks_init,
    mtmd_input_chunks_size, mtmd_input_text, mtmd_support_audio,
    mtmd_support_vision, mtmd_tokenize,
};

use crate::{
    backend::{Image, ImageInfo, MediaChunk, MediaSpan, Vision},
    llama_cpp::{decoder::LlamaCppDecoder, model::LlamaCppModel},
};

/// Construction parameters for [`Mtmd`] — the small, stable subset of
/// `mtmd_context_params` we expose. Everything else stays upstream
/// default; notably the media marker, which `Session` assumes is the
/// default `<__media__>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtmdParams {
    /// Offload the projector/encoder to GPU.
    pub use_gpu: bool,
    /// Encoder thread count.
    pub n_threads: i32,
    /// Run a warmup encode pass after load (upstream default).
    pub warmup: bool,
}

impl Default for MtmdParams {
    fn default() -> Self {
        Self {
            use_gpu: true,
            // Same rationale as `LlamaCppEngine::default_context_params`:
            // upstream's hard-coded small default cripples CPU encode.
            n_threads: std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4),
            warmup: true,
        }
    }
}

/// Failure constructing an [`Mtmd`].
#[derive(Debug, thiserror::Error)]
pub enum MtmdNewError {
    /// The path cannot be handed to C (not UTF-8, or embedded NUL).
    #[error("mmproj path {path:?} is not valid UTF-8 or contains a NUL")]
    BadPath { path: std::path::PathBuf },
    /// `mtmd_init_from_file` returned null — missing/corrupt mmproj,
    /// or a projector incompatible with the text model.
    #[error("could not load mmproj from {path:?}")]
    LoadFailed { path: std::path::PathBuf },
}

static_assertions::assert_impl_all!(MtmdNewError: Send, Sync);

/// Failure tokenizing marker-bearing text into [`MediaChunk`]s.
#[derive(Debug, thiserror::Error)]
pub enum MtmdTokenizeError {
    /// Text contains an interior NUL and cannot cross the C boundary.
    #[error("prompt text contains a NUL byte: {0}")]
    Nul(#[from] std::ffi::NulError),
    /// Marker occurrences in the text != number of images supplied.
    /// Checked in Rust before calling C — substring counting, the
    /// same semantics `mtmd_tokenize` splits with. (Upstream reports
    /// this too, but its return codes conflate it with preprocessing
    /// failures: the header documents mismatch as code 1, while the
    /// implementation throws → code 2 on one path and returns 1 on
    /// another, so the C codes cannot be classified reliably.)
    #[error(
        "text contains {markers} media markers but {images} images \
         were supplied"
    )]
    MarkerMismatch { markers: usize, images: usize },
    /// Nonzero `mtmd_tokenize` return code: preprocessing failure,
    /// unsupported modality, or internal error — upstream's codes are
    /// not reliably classifiable (see [`Self::MarkerMismatch`]);
    /// consult the llama.cpp log output for the specific cause.
    #[error("mtmd_tokenize failed with code {code}")]
    Code { code: i32 },
    /// `mtmd_bitmap_init` returned null (allocation failure).
    #[error("could not allocate an mtmd bitmap")]
    Bitmap,
    /// The tokenization produced a different number of media chunks
    /// than images supplied — e.g. a model that merges consecutive
    /// bitmaps into one chunk (qwen-style video batching), which the
    /// entry-wise prefix cache cannot represent yet.
    #[error("expected {expected} media chunks, got {actual}")]
    ChunkMismatch { expected: usize, actual: usize },
    /// A media chunk carried an id that is not the 64-hex-digit
    /// sha256 this module stamps on every bitmap.
    #[error("media chunk id {id:?} is not a sha256 this module set")]
    BadChunkId { id: String },
}

static_assertions::assert_impl_all!(MtmdTokenizeError: Send, Sync);

/// Failure encoding + decoding an [`Image`] into the KV cache.
#[derive(Debug, thiserror::Error)]
pub enum MtmdPrefillError {
    #[error(transparent)]
    Tokenize(#[from] MtmdTokenizeError),
    /// Tokenizing a lone marker with one bitmap did not yield exactly
    /// one media chunk — should be unreachable for image projectors.
    #[error("marker tokenization produced no single media chunk")]
    NoMediaChunk,
    /// Encode or decode failed inside the eval loop
    /// (`mtmd_helper_eval_chunk_single` nonzero).
    #[error("media chunk eval failed with code {code}")]
    Eval { code: i32 },
}

static_assertions::assert_impl_all!(MtmdPrefillError: Send, Sync);

/// Umbrella error for the [`Vision`] impl, which carries one error
/// type across both operations. The per-op enums remain the precise
/// types on the inherent methods.
#[derive(Debug, thiserror::Error)]
pub enum MtmdError {
    #[error(transparent)]
    Tokenize(#[from] MtmdTokenizeError),
    #[error(transparent)]
    Prefill(#[from] MtmdPrefillError),
}

static_assertions::assert_impl_all!(MtmdError: Send, Sync);

/// A loaded multimodal projector (mmproj): llama.cpp's `mtmd_context`.
///
/// `Send` but **not** `Sync`: encoding writes the context's internal
/// output-embedding buffer. (`mtmd_tokenize` is documented
/// thread-safe on a shared context, but one non-`Sync` type for the
/// whole handle keeps the story simple.)
///
/// # Lifetime
///
/// The context holds a pointer to the text model it was initialized
/// with (vocab reads during tokenize) — an `Mtmd` must not outlive
/// its [`LlamaCppModel`]. This is the same informal contract as
/// [`LlamaCppDecoder::new`]; for engine-owned instances,
/// [`crate::Engine`]'s field order (vision drops first) enforces it.
pub struct Mtmd {
    ctx: NonNull<mtmd_context>,
    /// Owned copy of the context's media marker (`<__media__>`).
    marker: String,
}

unsafe impl Send for Mtmd {}
static_assertions::assert_impl_all!(Mtmd: Send);

impl std::fmt::Debug for Mtmd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mtmd")
            .field("marker", &self.marker)
            .field("supports_images", &self.supports_images())
            .field("supports_audio", &self.supports_audio())
            .finish_non_exhaustive()
    }
}

impl Drop for Mtmd {
    fn drop(&mut self) {
        unsafe { mtmd_free(self.ctx.as_ptr()) }
    }
}

impl Mtmd {
    /// Load a multimodal projector GGUF for `model`.
    ///
    /// Most callers want the `<model>.mmproj.gguf` sidecar auto-load
    /// in `LlamaCppEngine`'s constructors instead; this is the
    /// arbitrary-path building block (see also
    /// [`crate::LlamaCppEngine::load_mmproj`]).
    pub fn from_path(
        mmproj: impl AsRef<Path>,
        model: &LlamaCppModel,
        params: MtmdParams,
    ) -> Result<Self, MtmdNewError> {
        let mmproj = mmproj.as_ref();
        let bad_path = || MtmdNewError::BadPath {
            path: mmproj.to_path_buf(),
        };
        let path_c = mmproj
            .to_str()
            .ok_or_else(bad_path)
            .and_then(|s| CString::new(s).map_err(|_| bad_path()))?;

        let mut cp = unsafe { mtmd_context_params_default() };
        cp.use_gpu = params.use_gpu;
        cp.n_threads = params.n_threads;
        cp.warmup = params.warmup;

        let ctx =
            unsafe { mtmd_init_from_file(path_c.as_ptr(), model.inner, cp) };
        let ctx =
            NonNull::new(ctx).ok_or_else(|| MtmdNewError::LoadFailed {
                path: mmproj.to_path_buf(),
            })?;

        let marker = unsafe { CStr::from_ptr(mtmd_get_marker(ctx.as_ptr())) }
            .to_string_lossy()
            .into_owned();

        Ok(Self { ctx, marker })
    }

    /// Whether the projector supports image input.
    pub fn supports_images(&self) -> bool {
        unsafe { mtmd_support_vision(self.ctx.as_ptr()) }
    }

    /// Whether the projector supports audio input. (drama_llama does
    /// not feed audio — no upstream `Block` variant — but the
    /// capability is worth surfacing.)
    pub fn supports_audio(&self) -> bool {
        unsafe { mtmd_support_audio(self.ctx.as_ptr()) }
    }

    /// Whether the text model decodes media with M-RoPE (n_tokens ≠
    /// n_pos; e.g. Qwen-VL). Informational — [`MediaSpan`] already
    /// carries both numbers.
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { mtmd_decode_use_mrope(self.ctx.as_ptr()) }
    }

    /// The media marker substring (`<__media__>` by default).
    pub fn marker(&self) -> &str {
        &self.marker
    }

    /// Tokenize `text` (containing one [`Self::marker`] per image,
    /// in order) into [`MediaChunk`]s using placeholder bitmaps —
    /// dims and identity only, no pixels touched. See
    /// [`Vision::tokenize`] for the flag semantics.
    pub fn tokenize(
        &self,
        text: &str,
        images: &[ImageInfo],
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<MediaChunk>, MtmdTokenizeError> {
        let markers = text.matches(&self.marker).count();
        if markers != images.len() {
            return Err(MtmdTokenizeError::MarkerMismatch {
                markers,
                images: images.len(),
            });
        }
        let bitmaps = images
            .iter()
            .map(Bitmap::placeholder)
            .collect::<Result<Vec<_>, _>>()?;
        let chunks =
            self.tokenize_raw(text, &bitmaps, add_special, parse_special)?;
        chunks.to_media_chunks(images.len())
    }

    /// Shared C-boundary crossing for [`Self::tokenize`] and
    /// [`Vision::prefill_image`]. Returns the raw chunk list.
    fn tokenize_raw(
        &self,
        text: &str,
        bitmaps: &[Bitmap],
        add_special: bool,
        parse_special: bool,
    ) -> Result<Chunks, MtmdTokenizeError> {
        let text_c = CString::new(text)?;
        let input = mtmd_input_text {
            text: text_c.as_ptr(),
            add_special,
            parse_special,
        };
        let chunks = Chunks::new()?;
        // bindgen renders C's `const mtmd_bitmap **` as
        // `*mut *const mtmd_bitmap`, hence the `mut` binding.
        let mut ptrs: Vec<*const mtmd_bitmap> = bitmaps
            .iter()
            .map(|b| b.0.as_ptr() as *const mtmd_bitmap)
            .collect();
        // Safety: all pointers live for the duration of the call;
        // `mtmd_tokenize` is documented thread-safe on a shared ctx.
        let ret = unsafe {
            mtmd_tokenize(
                self.ctx.as_ptr(),
                chunks.0.as_ptr(),
                &input,
                ptrs.as_mut_ptr(),
                ptrs.len(),
            )
        };
        match ret {
            0 => Ok(chunks),
            code => Err(MtmdTokenizeError::Code { code }),
        }
    }
}

impl Vision<LlamaCppDecoder> for Mtmd {
    type Error = MtmdError;

    fn marker(&self) -> &str {
        Mtmd::marker(self)
    }

    fn supports_images(&self) -> bool {
        Mtmd::supports_images(self)
    }

    fn tokenize(
        &self,
        text: &str,
        images: &[ImageInfo],
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<MediaChunk>, Self::Error> {
        Ok(Mtmd::tokenize(
            self,
            text,
            images,
            add_special,
            parse_special,
        )?)
    }

    /// Phase B implementation: tokenize a lone marker with the one
    /// real bitmap, then hand the media chunk to
    /// `mtmd_helper_eval_chunk_single`, which batches the encode +
    /// embedding decode and handles the non-causal attention toggle
    /// (Gemma-style) internally.
    ///
    /// Phase C+D replaces these internals with the Rust-owned eval
    /// loop (`EmbdBatch`, pre-KV NaN scan, M-RoPE positions) and
    /// differential-tests it against this helper path; the signature
    /// does not change. Until then there is no pre-KV NaN guard — the
    /// helper offers no hook between encode and decode.
    fn prefill_image(
        &mut self,
        decoder: &mut LlamaCppDecoder,
        image: &Image,
        start_pos: usize,
        seq_id: i32,
    ) -> Result<MediaSpan, Self::Error> {
        let bitmap = Bitmap::real(image)?;
        // Marker-only text: the media chunk it produces is identical
        // to the one a full-prompt tokenization yields for this image
        // (preprocessing is per-bitmap); any wrapper tokens the model
        // adds around markers live in *text* chunks, which the caller
        // prefills via the ordinary text path.
        let chunks = self.tokenize_raw(
            &self.marker,
            std::slice::from_ref(&bitmap),
            false,
            true,
        )?;

        let mut media: Option<*const mtmd_input_chunk> = None;
        for i in 0..chunks.len() {
            let chunk = chunks.get(i);
            let ty = unsafe { mtmd_input_chunk_get_type(chunk) };
            if ty != mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT {
                media = Some(chunk);
            }
        }
        let chunk = media.ok_or(MtmdPrefillError::NoMediaChunk)?;

        let n_tokens = unsafe { mtmd_input_chunk_get_n_tokens(chunk) } as u32;
        let n_pos = unsafe { mtmd_input_chunk_get_n_pos(chunk) } as u32;

        let n_batch = unsafe { llama_n_batch(decoder.context) } as i32;
        let mut new_n_past: llama_pos = 0;
        let ret = unsafe {
            mtmd_helper_eval_chunk_single(
                self.ctx.as_ptr(),
                decoder.context,
                chunk,
                start_pos as llama_pos,
                seq_id,
                n_batch,
                false, // logits_last: nothing reads logits after an embd decode
                &mut new_n_past,
            )
        };
        if ret != 0 {
            return Err(MtmdPrefillError::Eval { code: ret }.into());
        }
        debug_assert_eq!(
            new_n_past,
            start_pos as llama_pos + n_pos as llama_pos,
            "helper position advance disagrees with chunk n_pos"
        );

        Ok(MediaSpan { n_tokens, n_pos })
    }
}

/// Owned `mtmd_bitmap`. Private: the generic signatures enforce what
/// a public placeholder typestate would have — counting paths take
/// [`ImageInfo`] (placeholder, data == null, real dims as upstream
/// requires), encoding takes [`Image`] (pixels guaranteed).
struct Bitmap(NonNull<mtmd_bitmap>);

impl Drop for Bitmap {
    fn drop(&mut self) {
        unsafe { mtmd_bitmap_free(self.0.as_ptr()) }
    }
}

impl Bitmap {
    /// Pixel-carrying bitmap for encoding. Copies the RGB8 buffer.
    fn real(image: &Image) -> Result<Self, MtmdTokenizeError> {
        let ptr = unsafe {
            mtmd_bitmap_init(
                image.width(),
                image.height(),
                image.rgb8().as_ptr(),
            )
        };
        Self::with_id(ptr, image.id())
    }

    /// Placeholder bitmap for counting: null data, real dims
    /// (upstream tokenizes/counts these; encode rejects them).
    fn placeholder(info: &ImageInfo) -> Result<Self, MtmdTokenizeError> {
        let ptr = unsafe {
            mtmd_bitmap_init(info.width, info.height, std::ptr::null())
        };
        Self::with_id(ptr, &info.id)
    }

    fn with_id(
        ptr: *mut mtmd_bitmap,
        id: &[u8; 32],
    ) -> Result<Self, MtmdTokenizeError> {
        let bitmap = Self(NonNull::new(ptr).ok_or(MtmdTokenizeError::Bitmap)?);
        let hex = CString::new(hex_encode(id)).expect("hex has no NUL");
        unsafe { mtmd_bitmap_set_id(bitmap.0.as_ptr(), hex.as_ptr()) };
        Ok(bitmap)
    }
}

/// Owned `mtmd_input_chunks` list.
struct Chunks(NonNull<mtmd_input_chunks>);

impl Drop for Chunks {
    fn drop(&mut self) {
        unsafe { mtmd_input_chunks_free(self.0.as_ptr()) }
    }
}

impl Chunks {
    fn new() -> Result<Self, MtmdTokenizeError> {
        NonNull::new(unsafe { mtmd_input_chunks_init() })
            .map(Self)
            // Reuse the bitmap-allocation variant; both are OOM-shaped.
            .ok_or(MtmdTokenizeError::Bitmap)
    }

    fn len(&self) -> usize {
        unsafe { mtmd_input_chunks_size(self.0.as_ptr()) }
    }

    /// Borrow chunk `i`. Valid while `self` lives.
    fn get(&self, i: usize) -> *const mtmd_input_chunk {
        unsafe { mtmd_input_chunks_get(self.0.as_ptr(), i) }
    }

    /// Convert to the generic representation. `n_images` is the
    /// number of bitmaps that went into tokenization — media chunks
    /// must match it one-to-one for the entry-wise prefix cache.
    fn to_media_chunks(
        &self,
        n_images: usize,
    ) -> Result<Vec<MediaChunk>, MtmdTokenizeError> {
        let mut out = Vec::with_capacity(self.len());
        let mut n_media = 0usize;
        for i in 0..self.len() {
            let chunk = self.get(i);
            let ty = unsafe { mtmd_input_chunk_get_type(chunk) };
            if ty == mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT {
                let mut n_tokens: usize = 0;
                let ptr = unsafe {
                    mtmd_input_chunk_get_tokens_text(chunk, &mut n_tokens)
                };
                let tokens = if n_tokens == 0 {
                    Vec::new()
                } else {
                    unsafe { std::slice::from_raw_parts(ptr, n_tokens) }
                        .to_vec()
                };
                out.push(MediaChunk::Text(tokens));
            } else {
                n_media += 1;
                let id_c = unsafe { mtmd_input_chunk_get_id(chunk) };
                let id = if id_c.is_null() {
                    String::new()
                } else {
                    unsafe { CStr::from_ptr(id_c) }
                        .to_string_lossy()
                        .into_owned()
                };
                let id = hex_decode(&id)
                    .ok_or_else(|| MtmdTokenizeError::BadChunkId { id })?;
                let n_tokens =
                    unsafe { mtmd_input_chunk_get_n_tokens(chunk) } as u32;
                let n_pos = unsafe { mtmd_input_chunk_get_n_pos(chunk) } as u32;
                out.push(MediaChunk::Media {
                    id,
                    span: MediaSpan { n_tokens, n_pos },
                });
            }
        }
        if n_media != n_images {
            return Err(MtmdTokenizeError::ChunkMismatch {
                expected: n_images,
                actual: n_media,
            });
        }
        Ok(out)
    }
}

fn hex_encode(id: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(64);
    for b in id {
        write!(s, "{b:02x}").expect("writing to String cannot fail");
    }
    s
}

fn hex_decode(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve the local test model and its mmproj sidecar. The
    /// `models/model.gguf` symlink is canonicalized first so the
    /// sidecar is discovered next to the *target* file (the symlink
    /// itself has no `model.mmproj.gguf` sibling).
    fn local_vision_paths() -> (std::path::PathBuf, std::path::PathBuf) {
        let model = std::fs::canonicalize("models/model.gguf")
            .expect("models/model.gguf must point to a valid GGUF");
        let mmproj = crate::sidecar::mmproj_path(&model)
            .expect("test model needs a sibling <model>.mmproj.gguf sidecar");
        (model, mmproj)
    }

    #[test]
    #[ignore = "long running; requires local model + mmproj sidecar"]
    fn tokenize_chunk_structure() {
        let (model_path, mmproj) = local_vision_paths();
        let model =
            LlamaCppModel::from_file(model_path, None).expect("model load");
        let mtmd = Mtmd::from_path(
            &mmproj,
            &model,
            MtmdParams {
                use_gpu: false,
                warmup: false,
                ..Default::default()
            },
        )
        .expect("mmproj load");

        assert!(mtmd.supports_images());
        assert_eq!(mtmd.marker(), "<__media__>");

        // Placeholder tokenization: dims + identity only, no pixels.
        let info = ImageInfo {
            width: 640,
            height: 480,
            id: [42; 32],
        };
        let text = format!("Describe this image: {} thanks.", mtmd.marker());
        let chunks = mtmd.tokenize(&text, &[info], false, true).unwrap();

        let media: Vec<_> = chunks
            .iter()
            .filter_map(|c| match c {
                MediaChunk::Media { id, span } => Some((id, span)),
                MediaChunk::Text(_) => None,
            })
            .collect();
        assert_eq!(media.len(), 1, "one marker, one media chunk");
        let (id, span) = media[0];
        assert_eq!(*id, [42; 32], "bitmap id round-trips through mtmd");
        assert!(span.n_tokens > 0);
        assert!(span.n_pos > 0);
        assert!(span.n_pos <= span.n_tokens, "n_pos never exceeds cells");
        assert!(
            matches!(chunks.first(), Some(MediaChunk::Text(t)) if !t.is_empty()),
            "leading prose becomes a text chunk"
        );
        assert!(
            matches!(chunks.last(), Some(MediaChunk::Text(t)) if !t.is_empty()),
            "trailing prose becomes a text chunk"
        );

        // Marker/image count mismatch is a typed error, both ways.
        assert!(matches!(
            mtmd.tokenize("no marker here", &[info], false, true),
            Err(MtmdTokenizeError::MarkerMismatch {
                markers: 0,
                images: 1
            })
        ));
        assert!(matches!(
            mtmd.tokenize(&text, &[], false, true),
            Err(MtmdTokenizeError::MarkerMismatch {
                markers: 1,
                images: 0
            })
        ));
    }

    /// End-to-end Phase B smoke: engine construction auto-loads the
    /// mmproj sidecar, and `prefill_image` lands a real image's
    /// embeddings in the KV cache. CPU encode of a large projector is
    /// slow — the fixture is downscaled first (which also exercises
    /// the DynamicImage-as-source-of-truth path).
    #[test]
    #[ignore = "long running; requires local model + mmproj sidecar"]
    fn prefill_image_smoke() {
        use crate::{backend::Vision as _, Decoder as _};

        let (model_path, _) = local_vision_paths();
        let mut cp = crate::LlamaCppEngine::default_context_params();
        cp.n_ctx = 8192;
        let mut engine =
            crate::LlamaCppEngine::new(model_path, None, Some(cp), None)
                .expect("engine + mmproj sidecar load");
        assert!(engine.vision().is_some(), "mmproj sidecar should auto-load");

        let jpg = std::fs::read("tests/data/images/samoyed.jpg")
            .expect("committed fixture");
        let decoded = image::load_from_memory(&jpg).expect("jpeg decode");
        let image =
            crate::Image::try_from(decoded.thumbnail(512, 512)).unwrap();

        let (vision, decoder) = engine.vision_and_decoder();
        let span = vision
            .expect("vision loaded above")
            .prefill_image(decoder, &image, 0, 0)
            .expect("image prefill");
        assert!(span.n_tokens > 0);
        assert!(span.n_pos > 0);

        let pos_max = decoder.memory_seq_pos_max(0);
        assert!(pos_max >= 0, "KV cache holds the image cells");
        assert!(pos_max < span.n_tokens as i32 + span.n_pos as i32);
    }

    #[test]
    fn hex_roundtrip() {
        let mut id = [0u8; 32];
        for (i, b) in id.iter_mut().enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        assert_eq!(hex_decode(&hex_encode(&id)), Some(id));
        assert_eq!(hex_decode("not hex"), None);
        assert_eq!(hex_decode(&"a".repeat(63)), None);
        // Correct length, bad digits.
        assert_eq!(hex_decode(&"zz".repeat(32)), None);
    }
}
