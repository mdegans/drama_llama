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
    llama_context, llama_decode, llama_get_model, llama_model,
    llama_model_n_embd_inp, llama_n_batch, llama_n_ubatch, llama_pos,
    llama_seq_id, llama_set_causal_attn, mtmd_bitmap, mtmd_bitmap_free,
    mtmd_bitmap_init, mtmd_bitmap_set_id, mtmd_context,
    mtmd_context_params_default, mtmd_decode_use_mrope,
    mtmd_decode_use_non_causal, mtmd_decoder_pos, mtmd_encode_chunk, mtmd_free,
    mtmd_get_marker, mtmd_get_output_embd, mtmd_helper_eval_chunk_single,
    mtmd_helper_image_get_decoder_pos, mtmd_init_from_file, mtmd_input_chunk,
    mtmd_input_chunk_get_id, mtmd_input_chunk_get_n_pos,
    mtmd_input_chunk_get_n_tokens, mtmd_input_chunk_get_tokens_image,
    mtmd_input_chunk_get_tokens_text, mtmd_input_chunk_get_type,
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
    /// The image occupies more cells than one micro-batch holds and
    /// the model requires non-causal (single-pass) image attention
    /// (Gemma-style) — the decode cannot be split. Raise `n_ubatch`
    /// or downscale the image. Checked before the expensive encode.
    #[error(
        "image needs {n_tokens} cells but n_ubatch is {n_ubatch} and the \
         model requires single-pass (non-causal) image decode"
    )]
    ExceedsUbatch { n_tokens: u32, n_ubatch: u32 },
    /// The projector encode failed (`mtmd_encode_chunk` nonzero).
    #[error("media encode failed with code {code}")]
    Encode { code: i32 },
    /// This [`Mtmd`] was built against a different model than the
    /// decoder it was handed. mtmd validates projector/text-model
    /// agreement (embedding width, RoPE type) only at construction,
    /// so pairing it with a foreign decoder would size buffers from
    /// one model and index them with another's dimensions.
    #[error("projector was loaded for a different model than this decoder's")]
    ModelMismatch,
    /// The encoder produced a non-finite value (NaN/Inf), caught
    /// BEFORE any KV write — NaN is maximally contagious in the KV
    /// cache (one poisoned cell makes every later logit NaN and the
    /// damage survives until the cells are wiped). `id` is the
    /// offending bitmap's content hash (hex sha256 of its RGB8
    /// pixels); `index` is the first bad element in the encoder
    /// output. The KV cache is untouched; the caller decides whether
    /// to drop the image or abort.
    #[error(
        "media encoder output contains a non-finite value at element \
         {index} (image id {id}); rejected before any KV write"
    )]
    NonFinite { id: String, index: usize },
    /// `llama_decode` of an embedding batch view failed. KV state for
    /// this image may be partially written — callers should wipe
    /// (Session routes this through its cache-miss error path).
    #[error("media embedding decode failed with code {code}")]
    Decode { code: i32 },
    /// Encode or decode failed inside the bound upstream helper
    /// (`mtmd_helper_eval_chunk_single` nonzero). Only reachable via
    /// the crate-private helper path kept for differential testing.
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
    /// The model this projector was validated against at
    /// construction. Compared (never dereferenced) against the
    /// decoder's model on every eval — see [`Mtmd::eval_media_chunk`].
    model: *const llama_model,
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

        Ok(Self {
            ctx,
            marker,
            model: model.inner,
        })
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
    /// [`Vision::tokenize_image`](crate::Vision::tokenize_image) for the flag semantics.
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

    fn supports_images(&self) -> bool {
        Mtmd::supports_images(self)
    }

    /// One image = lone marker + placeholder bitmap through
    /// `mtmd_tokenize` — exactly the call `prefill_image` makes for
    /// the real encode (per-bitmap preprocessing identity, validated
    /// in Phase B), so the placeholder span always matches the
    /// encode-time span. Wrapper tokens the model frames media with
    /// (`<|vision_start|>` …) come back as text chunks around the
    /// media chunk and are returned in place. Prompt text NEVER
    /// enters this function — only the marker constant does — so no
    /// content byte can ever be interpreted as a marker.
    fn tokenize_image(
        &self,
        image: &ImageInfo,
        parse_special: bool,
    ) -> Result<Vec<MediaChunk>, Self::Error> {
        let placeholder =
            Bitmap::placeholder(image).map_err(MtmdError::Tokenize)?;
        let chunks = self
            .tokenize_raw(
                &self.marker,
                std::slice::from_ref(&placeholder),
                false,
                parse_special,
            )
            .map_err(MtmdError::Tokenize)?;
        Ok(chunks.to_media_chunks(1).map_err(MtmdError::Tokenize)?)
    }

    /// Rust-owned eval loop (Phase C+D): tokenize a lone marker with
    /// the one real bitmap, encode on the projector, scan the
    /// encoder output for non-finite values BEFORE any KV write,
    /// then decode the embeddings through `EmbdBatch` views with
    /// normal or M-RoPE positions and the non-causal attention
    /// toggle (Gemma-style) guarded on every error path.
    /// Differential-tested against the bound upstream helper
    /// (`Mtmd::prefill_image_via_helper`).
    fn prefill_image(
        &mut self,
        decoder: &mut LlamaCppDecoder,
        image: &Image,
        start_pos: usize,
        seq_id: i32,
    ) -> Result<MediaSpan, Self::Error> {
        let bitmap = Bitmap::real(image).map_err(MtmdPrefillError::Tokenize)?;
        // Marker-only text: the media chunk it produces is identical
        // to the one a full-prompt tokenization yields for this image
        // (preprocessing is per-bitmap); any wrapper tokens the model
        // adds around markers live in *text* chunks, which the caller
        // prefills via the ordinary text path.
        let chunks = self
            .tokenize_raw(
                &self.marker.clone(),
                std::slice::from_ref(&bitmap),
                false,
                true,
            )
            .map_err(MtmdPrefillError::Tokenize)?;
        let chunk =
            chunks.find_media().ok_or(MtmdPrefillError::NoMediaChunk)?;
        let span = self.eval_media_chunk(decoder, chunk, start_pos, seq_id)?;
        Ok(span)
    }
}

impl Mtmd {
    /// Encode one media chunk on the projector and decode its
    /// embeddings into the KV cache at `start_pos` on `seq_id`. The
    /// core of [`Vision::prefill_image`]; `chunk` must outlive the
    /// call (it borrows from a live [`Chunks`]).
    fn eval_media_chunk(
        &mut self,
        decoder: &mut LlamaCppDecoder,
        chunk: *const mtmd_input_chunk,
        start_pos: usize,
        seq_id: i32,
    ) -> Result<MediaSpan, MtmdPrefillError> {
        // mtmd enforces projector/text-model agreement (n_embd, RoPE
        // type) once, inside `mtmd_init_from_file`. That guarantee is
        // void the moment this projector meets a different model — and
        // `Engine::set_vision` makes that reachable from safe code.
        // Below we read `n_embd` from the *decoder's* model to size a
        // slice over a buffer mtmd allocated from *its* model, and take
        // the M-RoPE decision from the mtmd context while llama.cpp
        // reads `pos` using the decoder's `n_pos_per_embd`. Mismatched,
        // any of those three is an out-of-bounds access.
        if !std::ptr::eq(self.model, unsafe {
            llama_get_model(decoder.context)
        }) {
            return Err(MtmdPrefillError::ModelMismatch);
        }
        let n_tokens = unsafe { mtmd_input_chunk_get_n_tokens(chunk) } as usize;
        let n_pos = unsafe { mtmd_input_chunk_get_n_pos(chunk) } as u32;

        // Fit check up front, before the expensive encode: non-causal
        // image attention (Gemma-style) means every image cell must
        // attend to every other, so the decode cannot be split across
        // micro-batches. (Upstream's helper has a TODO here and would
        // decode garbage; we refuse instead.)
        let non_causal =
            unsafe { mtmd_decode_use_non_causal(self.ctx.as_ptr(), chunk) };
        let n_ubatch = unsafe { llama_n_ubatch(decoder.context) } as usize;
        if non_causal && n_tokens > n_ubatch {
            return Err(MtmdPrefillError::ExceedsUbatch {
                n_tokens: n_tokens as u32,
                n_ubatch: n_ubatch as u32,
            });
        }

        // Encode. Output lands in the mtmd context's internal buffer
        // (the reason `Mtmd` is `Send` but not `Sync`), valid until
        // the next encode.
        let ret = unsafe { mtmd_encode_chunk(self.ctx.as_ptr(), chunk) };
        if ret != 0 {
            return Err(MtmdPrefillError::Encode { code: ret });
        }
        let embd = unsafe { mtmd_get_output_embd(self.ctx.as_ptr()) };
        if embd.is_null() {
            return Err(MtmdPrefillError::Encode { code: -1 });
        }
        let n_embd =
            unsafe { llama_model_n_embd_inp(llama_get_model(decoder.context)) }
                as usize;
        let embd_slice =
            unsafe { std::slice::from_raw_parts(embd, n_tokens * n_embd) };

        // Pre-KV non-finite guard: this is the hook the upstream
        // helper doesn't have. Nothing has touched the KV cache yet,
        // so a poisoned encode is rejected with state fully intact.
        if let Some(index) = embd_slice.iter().position(|v| !v.is_finite()) {
            let id_c = unsafe { mtmd_input_chunk_get_id(chunk) };
            let id = if id_c.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(id_c) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(MtmdPrefillError::NonFinite { id, index });
        }

        // Positions: M-RoPE images get 4 plane-major position planes
        // from upstream's own layout function; everything else is the
        // ordinary dense single plane.
        let mrope = self.decode_use_mrope();
        let mut batch = EmbdBatch::new(
            embd,
            n_tokens,
            if mrope { 4 } else { 1 },
            n_embd,
            seq_id,
        );
        if mrope {
            let image_tokens =
                unsafe { mtmd_input_chunk_get_tokens_image(chunk) };
            if image_tokens.is_null() {
                return Err(MtmdPrefillError::NoMediaChunk);
            }
            let mut rel = vec![
                mtmd_decoder_pos {
                    t: 0,
                    x: 0,
                    y: 0,
                    z: 0
                };
                n_tokens
            ];
            unsafe {
                mtmd_helper_image_get_decoder_pos(
                    image_tokens,
                    start_pos as llama_pos,
                    rel.as_mut_ptr(),
                )
            };
            batch.set_position_mrope_2d(&rel);
        } else {
            batch.set_position_normal(start_pos as llama_pos);
        }

        // Decode in n_batch slices. The guard restores causal
        // attention on every exit path, including decode errors.
        let n_batch =
            (unsafe { llama_n_batch(decoder.context) } as usize).max(1);
        let _causal = CausalAttnGuard::disable_if(non_causal, decoder.context);
        let mut offset = 0;
        while offset < n_tokens {
            let len = n_batch.min(n_tokens - offset);
            let view = batch.view(offset, len);
            let ret = unsafe { llama_decode(decoder.context, view) };
            if ret != 0 {
                return Err(MtmdPrefillError::Decode { code: ret });
            }
            offset += len;
        }

        Ok(MediaSpan {
            n_tokens: n_tokens as u32,
            n_pos,
        })
    }

    /// The Phase B implementation of [`Vision::prefill_image`],
    /// delegating encode + decode + position assignment + non-causal
    /// toggle to upstream's `mtmd_helper_eval_chunk_single`. Kept
    /// crate-private as the reference the Rust-owned eval loop is
    /// differential-tested against (same context, identical logits).
    /// No pre-KV NaN guard — the helper has no hook between encode
    /// and decode.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn prefill_image_via_helper(
        &mut self,
        decoder: &mut LlamaCppDecoder,
        image: &Image,
        start_pos: usize,
        seq_id: i32,
    ) -> Result<MediaSpan, MtmdError> {
        let bitmap = Bitmap::real(image).map_err(MtmdPrefillError::Tokenize)?;
        let chunks = self
            .tokenize_raw(
                &self.marker.clone(),
                std::slice::from_ref(&bitmap),
                false,
                true,
            )
            .map_err(MtmdPrefillError::Tokenize)?;
        let chunk =
            chunks.find_media().ok_or(MtmdPrefillError::NoMediaChunk)?;

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

/// Hand-assembled embedding batch for media decode.
///
/// `llama_batch_init` hard-allocates `pos` at `n_tokens` entries, but
/// M-RoPE image decode needs `n_tokens × 4` plane-major position
/// entries — extending [`crate::Batch`] is structurally insufficient
/// (upstream's `decode_embd_batch` hand-assembles for the same
/// reason). Every buffer is a Rust-owned `Vec`; the `llama_batch`
/// handed to `llama_decode` is a borrowed VIEW and is never passed to
/// `llama_batch_free`. Logits stay all-false: nothing reads logits
/// after an embedding decode.
struct EmbdBatch {
    /// Plane-major positions, `n_tokens * n_pos_per_embd` entries.
    pos: Vec<llama_pos>,
    n_seq_id: Vec<i32>,
    /// The one sequence id every token targets. Heap storage so the
    /// pointers in `seq_ids` survive moves of the struct itself.
    seq_id_0: Vec<llama_seq_id>,
    seq_ids: Vec<*mut llama_seq_id>,
    logits: Vec<i8>,
    /// Scratch for M-RoPE views — plane slices of `pos` are
    /// non-contiguous, so each view gathers them here.
    pos_view: Vec<llama_pos>,
    /// Encoder output, borrowed from the mtmd context (valid until
    /// the next encode; `EmbdBatch` never outlives the eval call).
    embd: *mut f32,
    n_tokens: usize,
    n_pos_per_embd: usize,
    n_embd: usize,
}

impl EmbdBatch {
    fn new(
        embd: *mut f32,
        n_tokens: usize,
        n_pos_per_embd: usize,
        n_embd: usize,
        seq_id: llama_seq_id,
    ) -> Self {
        let mut batch = Self {
            pos: vec![0; n_tokens * n_pos_per_embd],
            n_seq_id: vec![1; n_tokens],
            seq_id_0: vec![seq_id],
            seq_ids: Vec::with_capacity(n_tokens),
            logits: vec![0; n_tokens],
            pos_view: Vec::new(),
            embd,
            n_tokens,
            n_pos_per_embd,
            n_embd,
        };
        let seq_ptr = batch.seq_id_0.as_mut_ptr();
        batch.seq_ids = vec![seq_ptr; n_tokens];
        batch
    }

    /// Dense single-plane positions `[pos_0, pos_0 + n_tokens)`.
    fn set_position_normal(&mut self, pos_0: llama_pos) {
        debug_assert_eq!(self.n_pos_per_embd, 1);
        for (i, p) in self.pos.iter_mut().enumerate() {
            *p = pos_0 + i as llama_pos;
        }
    }

    /// M-RoPE image positions: plane-major `(t, y, x, z)` — the same
    /// layout upstream's `decode_embd_batch::set_position_mrope_2d`
    /// writes.
    fn set_position_mrope_2d(&mut self, rel: &[mtmd_decoder_pos]) {
        debug_assert_eq!(self.n_pos_per_embd, 4);
        debug_assert_eq!(rel.len(), self.n_tokens);
        let n = self.n_tokens;
        // `mtmd_decoder_pos` fields are declared `uint32_t` upstream
        // but hold `llama_pos` values (C++ converts implicitly).
        for (i, r) in rel.iter().enumerate() {
            self.pos[i] = r.t as llama_pos;
            self.pos[i + n] = r.y as llama_pos;
            self.pos[i + n * 2] = r.x as llama_pos;
            self.pos[i + n * 3] = r.z as llama_pos;
        }
    }

    /// A `llama_batch` view over tokens `[offset, offset + len)`.
    /// Borrows this struct's buffers — do not free, do not outlive.
    fn view(
        &mut self,
        offset: usize,
        len: usize,
    ) -> llama_cpp_sys_3::llama_batch {
        debug_assert!(offset + len <= self.n_tokens);
        let pos_ptr = if self.n_pos_per_embd > 1 {
            // Gather each plane's slice: source layout is
            // `tttt…yyyy…xxxx…zzzz…` over n_tokens, the view needs
            // the same plane-major layout over len.
            self.pos_view.clear();
            self.pos_view.reserve(len * self.n_pos_per_embd);
            for plane in 0..self.n_pos_per_embd {
                let src = plane * self.n_tokens + offset;
                self.pos_view.extend_from_slice(&self.pos[src..src + len]);
            }
            self.pos_view.as_mut_ptr()
        } else {
            self.pos[offset..].as_mut_ptr()
        };
        llama_cpp_sys_3::llama_batch {
            n_tokens: len as i32,
            token: std::ptr::null_mut(),
            embd: unsafe { self.embd.add(offset * self.n_embd) },
            pos: pos_ptr,
            n_seq_id: self.n_seq_id[offset..].as_mut_ptr(),
            seq_id: self.seq_ids[offset..].as_mut_ptr(),
            logits: self.logits[offset..].as_mut_ptr(),
        }
    }
}

/// Disables causal attention for the lifetime of the guard and
/// restores it on drop — error paths inside the media decode loop
/// can't leave the context stuck non-causal.
struct CausalAttnGuard {
    ctx: *mut llama_context,
    active: bool,
}

impl CausalAttnGuard {
    fn disable_if(non_causal: bool, ctx: *mut llama_context) -> Self {
        if non_causal {
            unsafe { llama_set_causal_attn(ctx, false) };
        }
        Self {
            ctx,
            active: non_causal,
        }
    }
}

impl Drop for CausalAttnGuard {
    fn drop(&mut self) {
        if self.active {
            unsafe { llama_set_causal_attn(self.ctx, true) };
        }
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

    /// The first non-text (media) chunk, if any. Valid while `self`
    /// lives.
    fn find_media(&self) -> Option<*const mtmd_input_chunk> {
        (0..self.len()).map(|i| self.get(i)).find(|&chunk| {
            let ty = unsafe { mtmd_input_chunk_get_type(chunk) };
            ty != mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT
        })
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

    /// Segment-assembly differential (plan #31 C+D, work item 1):
    /// [`Vision::tokenize`] on pre-split segments must produce the
    /// same chunk stream as a full-text `mtmd_tokenize` of the
    /// marker-joined prompt — proving our out-of-band split is
    /// byte-identical to mtmd's own marker splitting (mtmd also
    /// tokenizes each inter-marker piece separately, so there is no
    /// BPE seam to diverge on).
    #[test]
    #[ignore = "long running; requires local model + mmproj sidecar"]
    fn segment_tokenize_differential() {
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

        let infos = [
            ImageInfo {
                width: 640,
                height: 480,
                id: [7; 32],
            },
            ImageInfo {
                width: 320,
                height: 240,
                id: [9; 32],
            },
        ];
        let segments = ["<|im_start|>user\nCompare ", " with ", "<|im_end|>\n"];

        // Session-style assembly: text through the MODEL tokenizer
        // (segment 0 like a full render, later segments without
        // leading specials), each image through tokenize_image.
        #[derive(Debug, PartialEq)]
        enum Flat {
            Tok(crate::Token),
            Media([u8; 32], MediaSpan),
        }
        let flatten = |chunks: &[MediaChunk], out: &mut Vec<Flat>| {
            for c in chunks {
                match c {
                    MediaChunk::Text(ts) => {
                        out.extend(ts.iter().map(|t| Flat::Tok(*t)))
                    }
                    MediaChunk::Media { id, span } => {
                        out.push(Flat::Media(*id, *span))
                    }
                }
            }
        };

        let mut assembled: Vec<Flat> = Vec::new();
        for (i, segment) in segments.iter().enumerate() {
            use crate::backend::Model as _;
            let tokens = if i == 0 {
                crate::backend::Model::tokenize(&model, segment, true)
            } else {
                model.tokenize_special(segment, false, true)
            };
            assembled.extend(tokens.into_iter().map(Flat::Tok));
            if let Some(info) = infos.get(i) {
                let chunks = Vision::tokenize_image(&mtmd, info, true)
                    .expect("image tokenize");
                flatten(&chunks, &mut assembled);
            }
        }

        let joined = format!(
            "{}{m}{}{m}{}",
            segments[0],
            segments[1],
            segments[2],
            m = mtmd.marker()
        );
        let via_marker = mtmd
            .tokenize(&joined, &infos, false, true)
            .expect("marker tokenize");
        let mut reference: Vec<Flat> = Vec::new();
        flatten(&via_marker, &mut reference);

        assert_eq!(
            assembled, reference,
            "segment assembly must equal mtmd's own marker splitting"
        );
        // Sanity: the two media items round-trip their ids, in order.
        let ids: Vec<_> = assembled
            .iter()
            .filter_map(|f| match f {
                Flat::Media(id, _) => Some(*id),
                Flat::Tok(_) => None,
            })
            .collect();
        assert_eq!(ids, vec![[7; 32], [9; 32]]);
    }

    /// Differential test for the Rust-owned eval loop (plan #31 C+D,
    /// work item 4): same context, same prefix + image + trailing
    /// text, once through upstream's `mtmd_helper_eval_chunk_single`
    /// and once through [`Vision::prefill_image`]'s `EmbdBatch` loop.
    /// CPU decode of identical batches is bit-deterministic, so the
    /// trailing-text logits must match exactly — any position-plane
    /// or batching mistake in the Rust loop shows up here.
    #[test]
    #[ignore = "long running; requires local model + mmproj sidecar"]
    fn eval_loop_differential_vs_helper() {
        use crate::Decoder as _;

        let (model_path, _) = local_vision_paths();
        let mut cp = crate::LlamaCppEngine::default_context_params();
        cp.n_ctx = 8192;
        let mut engine =
            crate::LlamaCppEngine::new(model_path, None, Some(cp), None)
                .expect("engine + mmproj sidecar load");

        let prefix = engine.model.tokenize("Describe this image:", false);
        let trail = engine.model.tokenize(" What breed is shown?", false);
        let jpg = std::fs::read("tests/data/images/samoyed.jpg")
            .expect("committed fixture");
        let image = crate::Image::try_from(
            image::load_from_memory(&jpg)
                .expect("jpeg decode")
                .thumbnail(512, 512),
        )
        .unwrap();

        let run = |engine: &mut crate::LlamaCppEngine,
                   use_helper: bool|
         -> (MediaSpan, Vec<f32>) {
            engine.memory_clear();
            engine.prefill_chunk(&prefix, 0, 0).expect("prefix prefill");
            let (vision, decoder) = engine.vision_and_decoder();
            let vision = vision.expect("mmproj sidecar should auto-load");
            let span = if use_helper {
                vision
                    .prefill_image_via_helper(decoder, &image, prefix.len(), 0)
                    .expect("helper-path image prefill")
            } else {
                Vision::prefill_image(vision, decoder, &image, prefix.len(), 0)
                    .expect("rust-loop image prefill")
            };
            let after = prefix.len() + span.n_pos as usize;
            let logits = decoder
                .prefill(&trail, after, 0)
                .expect("trail prefill")
                .to_vec();
            (span, logits)
        };

        let (span_helper, logits_helper) = run(&mut engine, true);
        let (span_rust, logits_rust) = run(&mut engine, false);

        assert_eq!(span_helper, span_rust, "spans must agree");
        assert_eq!(logits_helper.len(), logits_rust.len());
        let n_diff = logits_helper
            .iter()
            .zip(&logits_rust)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            n_diff, 0,
            "logits diverge between helper and Rust eval loop",
        );
    }

    /// M-RoPE KV-semantics probe (plan #31 C+D, work item 4 —
    /// front-loaded before the media KV walk is built on top).
    ///
    /// Upstream source facts this validates against a live model
    /// (`mtmd.cpp:1898` `mtmd_image_tokens_get_decoder_pos`,
    /// `:1957` `mtmd_image_tokens_get_n_pos`):
    ///
    /// * Every M-RoPE image cell carries `t = pos_0` — the whole
    ///   image sits at ONE tracked KV position; only the y/x RoPE
    ///   planes vary.
    /// * `n_pos = max(nx, ny)`, so positions
    ///   `[pos_0 + 1, pos_0 + n_pos)` form a GAP no cell ever
    ///   occupies (until later text lands at `pos_0 + n_pos`).
    ///
    /// And one fact about the local Qwen 3.6 in particular: it is a
    /// HYBRID model (recurrent + attention layers), so partial-range
    /// `memory_seq_rm` is refused wholesale and `restore_to` only
    /// ever succeeds through the per-position state snapshots that
    /// `checkpoint_pos` records (`seq_snapshots_enabled`). The
    /// recurrent memory also logs benign "non-consecutive token
    /// position" warnings on every M-RoPE position jump — its
    /// `cell.pos` bookkeeping expects dense positions, but tokens
    /// are still processed in sequence order.
    ///
    /// Consequences the Session KV walk must respect:
    ///
    /// * `memory_seq_pos_max` after an image at `P` reports `P`,
    ///   not `P + n_pos - 1` — position-space "KV length" cannot be
    ///   derived from pos_max when the tip is a media chunk.
    /// * Boundaries the walk wants to rewind to MUST be
    ///   checkpointed at prefill time (Session already does this) —
    ///   including a boundary at image-end, which works via
    ///   snapshot even though the truncate-based path would fail
    ///   its dense-position check there (pure-attention M-RoPE
    ///   models fall back to full reprefill at such a boundary).
    /// * A restore to an uncheckpointed position fails closed; on
    ///   this hybrid the KV is left untouched (truncate refused
    ///   up front), while on pure-attention models the truncate
    ///   half runs first — either way Session's fallback is
    ///   `memory_clear` + full reprefill.
    #[test]
    #[ignore = "long running; requires local model + mmproj sidecar"]
    fn mrope_kv_semantics_probe() {
        use crate::backend::Vision as _;

        let (model_path, _) = local_vision_paths();
        let mut cp = crate::LlamaCppEngine::default_context_params();
        cp.n_ctx = 8192;
        let mut engine =
            crate::LlamaCppEngine::new(model_path, None, Some(cp), None)
                .expect("engine + mmproj sidecar load");

        let mrope = engine
            .vision()
            .expect("mmproj sidecar should auto-load")
            .decode_use_mrope();
        assert!(
            mrope,
            "probe requires an M-RoPE model (Qwen-VL family); the local \
             model symlink points elsewhere"
        );

        // Text prefix [0, T).
        let prefix = engine.model.tokenize("Describe this image:", false);
        let t = prefix.len();
        engine.prefill_chunk(&prefix, 0, 0).expect("text prefill");
        assert_eq!(engine.memory_seq_pos_max(0), t as i32 - 1);

        // Image at position T.
        let jpg = std::fs::read("tests/data/images/samoyed.jpg")
            .expect("committed fixture");
        let decoded = image::load_from_memory(&jpg).expect("jpeg decode");
        let image =
            crate::Image::try_from(decoded.thumbnail(512, 512)).unwrap();
        let (vision, decoder) = engine.vision_and_decoder();
        let snapshots = decoder.seq_snapshots_enabled();
        let span = vision
            .expect("vision loaded above")
            .prefill_image(decoder, &image, t, 0)
            .expect("image prefill");
        eprintln!(
            "probe: image span n_tokens={} n_pos={} at P={t}, \
             seq_snapshots_enabled={snapshots}",
            span.n_tokens, span.n_pos
        );
        assert!(
            span.n_tokens > span.n_pos,
            "M-RoPE image should occupy more cells than positions"
        );

        // THE load-bearing surprise: all image cells share t = P.
        assert_eq!(
            engine.memory_seq_pos_max(0),
            t as i32,
            "M-RoPE image cells all carry the chunk start position"
        );

        // Checkpoint the image-end boundary, exactly as the Session
        // walk will after a media chunk.
        let after = t + span.n_pos as usize;
        engine.checkpoint_pos(0, after as i32);

        // Trailing text at P + n_pos (the position-counter jump),
        // with a mid-text checkpointed boundary.
        let trail = engine.model.tokenize(" What breed is shown?", false);
        assert!(trail.len() >= 3, "probe needs a few trailing tokens");
        let b_text = after + 2;
        engine
            .prefill_chunk(&trail[..2], after, 0)
            .expect("trail prefill (head)");
        engine.checkpoint_pos(0, b_text as i32);
        engine
            .prefill_chunk(&trail[2..], b_text, 0)
            .expect("trail prefill (tail)");
        let full_end = after + trail.len();
        assert_eq!(engine.memory_seq_pos_max(0), full_end as i32 - 1);

        // Rewind to the checkpointed text boundary and re-extend:
        // the core prefix-cache maneuver, now with an image in the
        // retained prefix.
        engine
            .restore_to(0, b_text as i32)
            .expect("restore to a checkpointed text boundary");
        assert_eq!(engine.memory_seq_pos_max(0), b_text as i32 - 1);
        engine
            .prefill_chunk(&trail[2..], b_text, 0)
            .expect("re-extend after restore");
        assert_eq!(engine.memory_seq_pos_max(0), full_end as i32 - 1);

        // Rewind to the image-end boundary. On this hybrid it rides
        // the snapshot; the retained tip is the media chunk, so
        // pos_max reports P — the walk must track the boundary
        // position itself (EntryPos) and never infer it from pos_max.
        engine
            .restore_to(0, after as i32)
            .expect("restore to the checkpointed image-end boundary");
        assert_eq!(engine.memory_seq_pos_max(0), t as i32);
        engine
            .prefill_chunk(&trail, after, 0)
            .expect("re-extend from image-end boundary");
        assert_eq!(engine.memory_seq_pos_max(0), full_end as i32 - 1);

        // Uncheckpointed position (mid-gap, worst case) fails closed.
        let mid_gap = t + (span.n_pos as usize / 2);
        assert!(
            engine.restore_to(0, mid_gap as i32).is_err(),
            "restore to an uncheckpointed mid-gap position must fail"
        );
        if snapshots {
            // Hybrid: the truncate half was refused up front, so the
            // failed restore left KV untouched.
            assert_eq!(engine.memory_seq_pos_max(0), full_end as i32 - 1);
            // ... and raw partial-range removal is likewise refused.
            assert!(!engine.memory_seq_rm(0, t as i32, -1));
        }
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
