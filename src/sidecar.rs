//! Per-model sidecar files.
//!
//! A sidecar is a file colocated with a model on disk that overrides
//! one aspect of how it is served: sampling defaults
//! (`<model>.sampling.toml`, [`SamplerConfig`]), the tool-call
//! dialect (`<model>.dialect.toml`,
//! [`CallSyntax`](crate::CallSyntax)), the chat template itself
//! (`<model>.template.jinja`, raw Jinja), the multimodal
//! projector (`<model>.mmproj.gguf`, [`mmproj_path`] — enables image
//! input under the `mtmd` feature), or load-time options
//! (`<model>.load.toml`, [`LoadSidecar`] — the KV context size and
//! cache type, micro-batch and RoPE scaling). [`crate::LlamaCppSession::from_path*`]
//! looks for each when loading a model. For sampling, if no sidecar
//! exists one is written so the user has a starting point to edit —
//! seeded from the model's own recommendation where it has one (see
//! below).
//!
//! ## Sampling precedence
//!
//! ```text
//! request temperature/top_p/top_k     (per-call, see `apply_request_sampling`)
//!   └─ <model>.sampling.toml sidecar  (per-model, editable — this module)
//!        └─ general.sampling.* GGUF metadata  (seeds the sidecar)
//!             └─ SamplerConfig::default()
//! ```
//!
//! The metadata tier **seeds** the sidecar rather than applying
//! invisibly at load. That keeps exactly one authority for a model's
//! defaults — the file on disk — and makes the model's own
//! recommendation visible and editable instead of a hidden layer the
//! user has to know about to explain their own output.
//!
//! **Not every model advertises sampling metadata**, and that is a
//! normal case, not an error: gpt-oss carries no `general.sampling.*`
//! keys at all, and moeflux has no such namespace to begin with. When
//! [`Model::recommended_sampling`](crate::backend::Model::recommended_sampling)
//! comes back empty the seed is plain [`SamplerConfig::default()`] —
//! the same file that was written before this tier existed. Partial
//! metadata is honored partially: a model advertising only `top_k`
//! seeds a one-mode chain rather than filling the gaps with invented
//! numbers. Either way the written file states in its header comment
//! which tier it came from, so a user comparing two models' sidecars
//! can tell a recommendation from a fallback.
//!
//! Only `modes` is ever model-derived. `repetition` stays at the
//! crate default: upstream's `penalty_repeat` / `penalty_last_n` are
//! scalars, while [`RepetitionOptions`](crate::RepetitionOptions) is
//! n-gram-based with windowed decay, and there is no honest mapping
//! between the two.
//!
//! Backends differ in what they can offer here. The llama.cpp backend
//! reads llama.cpp's typed `general.sampling.*` namespace; moeflux has
//! no equivalent (its config is HF `config.json`) and answers from a
//! per-variant constant. Either way the question is asked through
//! [`Model::recommended_sampling`](crate::backend::Model::recommended_sampling),
//! never by key.
//!
//! ## Where sidecars live
//!
//! - **GGUF (llama-cpp backend)**: sibling file at
//!   `<model>.sampling.toml`. So `model.gguf` →
//!   `model.sampling.toml`.
//! - **Moeflux backend**: `parent/sampling.toml`, alongside the
//!   `mlx`/`artifacts`/`root` symlinks. Not inside any of those —
//!   `parent/` is the blallama-owned dir; the subdirs are
//!   model-canonical content.
//!
//! ## What lives in a sidecar
//!
//! Everything in [`SamplerConfig`] that is `Serialize` /
//! `Deserialize`:
//! - `modes` — the sampling-mode chain
//!   ([`SamplingMode::TopP`](crate::SamplingMode::TopP),
//!   [`SamplingMode::Mirostat`](crate::SamplingMode::Mirostat), etc.)
//! - `repetition` — `Some(RepetitionOptions)` to enable, `None` to
//!   disable. Its `id_patterns` name what an identifier looks like;
//!   every match in the prompt is a *known id*, never penalized when
//!   copied faithfully. `id_copy_lock` (default `true`, inert without
//!   `id_patterns`) also holds a copy to its id: once eight characters
//!   of exactly one known hex id (a UUID, not a sentinel like
//!   `00000000-…-0001`) are written, the next tokens must continue it
//!   until it is complete
//!   ([`RepetitionOptions::id_copy_lock`](crate::RepetitionOptions::id_copy_lock)).
//! - `max_tool_calls_per_turn` — the most client tool calls one turn
//!   may make ([`SamplerConfig::max_tool_calls_per_turn`]); absent is
//!   unlimited. The sampler ends a turn on the model's own EOG once its
//!   last call completes ([`ToolCallCap`](crate::ToolCallCap)), and a
//!   request's `disable_parallel_tool_use` still caps it at one. Never
//!   seeded: a model that loops on parallel calls is found in service,
//!   so the key is added by hand (`models/cogito-32b.sampling.toml`).
//!
//! Excluded:
//! - `deferred_grammar` — runtime per-request state, `#[serde(skip)]`.
//! - [`SamplingMode::Json`] / [`SamplingMode::Grammar`] /
//!   [`SamplingMode::Deny`] — runtime per-request constraints.
//!   Including them in a sidecar would freeze a particular grammar
//!   into the model's defaults; almost never what you want.
//!
//! ## Reset / tweak
//!
//! - To **reset**: delete the sidecar file. The next load rewrites it,
//!   re-seeding from the model's metadata. Note that an existing
//!   sidecar is *never* overwritten, so a file written before the
//!   model-metadata seeding landed keeps its old contents until it is
//!   deleted.
//! - To **tweak** something: edit the sidecar, save, restart.
//!
//! [`crate::LlamaCppSession::from_path*`]: crate::FromPath::from_path
//! [`Session::with_sample_options`]: crate::Session::with_sample_options
//! [`SamplingMode::Json`]: crate::SamplingMode::Json
//! [`SamplingMode::Grammar`]: crate::SamplingMode::Grammar
//! [`SamplingMode::Deny`]: crate::SamplingMode::Deny

use std::path::Path;

#[cfg(feature = "toml")]
use crate::SamplerConfig;

/// Failure mode for sidecar I/O.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SidecarError {
    #[error("sidecar I/O at {path:?}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[cfg(feature = "toml")]
    #[error("sidecar TOML parse at {path:?}: {source}")]
    Parse {
        path: std::path::PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[cfg(feature = "toml")]
    #[error("sidecar TOML serialize: {0}")]
    Serialize(#[from] toml::ser::Error),
}

static_assertions::assert_impl_all!(SidecarError: Send, Sync);

/// Read a sidecar from `path` if it exists and parse it as
/// [`SamplerConfig`].
///
/// Returns:
/// - `Ok(Some(opts))` — sidecar found and parsed.
/// - `Ok(None)` — sidecar does not exist (the common
///   first-time-loading-a-model case).
/// - `Err(SidecarError::Io)` — file exists but couldn't be read
///   (permissions, etc.).
/// - `Err(SidecarError::Parse)` — file exists but contains malformed
///   TOML or TOML that doesn't deserialize into [`SamplerConfig`].
#[cfg(feature = "toml")]
pub fn load_sample_options(
    path: &Path,
) -> Result<Option<SamplerConfig>, SidecarError> {
    let bytes = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(SidecarError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let opts: SamplerConfig =
        toml::from_str(&bytes).map_err(|source| SidecarError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(Some(opts))
}

/// Write `opts` to `path` as TOML so the user has a starting point to
/// edit. Best-effort: if the parent dir doesn't exist or the file
/// isn't writable, returns the underlying IO error and the caller
/// decides whether to log + continue.
///
/// Does *not* overwrite an existing file — call
/// [`load_sample_options`] first to detect existence; the
/// [`crate::LlamaCppSession::from_path*`] integration only writes when the read
/// returned `Ok(None)`.
///
/// `from_metadata` only selects the header comment. Pass `true` when
/// `opts` was derived from the model's own
/// [`recommended_sampling`](crate::backend::Model::recommended_sampling)
/// so the file says where its numbers came from — otherwise a user
/// comparing two models' sidecars has no way to tell a model's
/// recommendation from the crate default.
///
/// [`crate::LlamaCppSession::from_path*`]: crate::FromPath::from_path
#[cfg(feature = "toml")]
pub fn write_sample_options(
    path: &Path,
    opts: &SamplerConfig,
    from_metadata: bool,
) -> Result<(), SidecarError> {
    let body = toml::to_string_pretty(opts)?;
    let provenance = if from_metadata {
        "# The mode chain below was seeded from this model's own\n\
         # `general.sampling.*` metadata — what the model asks for.\n"
    } else {
        "# The model advertised no `general.sampling.*` metadata, so\n\
         # the mode chain below is SamplerConfig::default().\n"
    };
    let header = format!(
        "# drama_llama per-model sampling sidecar.\n\
         # Edit to tune sampling for this model. Delete to reset; the\n\
         # next load will rewrite this file.\n\
         #\n\
         {provenance}\
         #\n\
         # See drama_llama::sidecar module docs for the precedence\n\
         # ladder and what's intentionally excluded (Json, Grammar,\n\
         # Deny modes — those are per-request runtime, not per-model\n\
         # defaults).\n\n"
    );
    std::fs::write(path, format!("{header}{body}")).map_err(|source| {
        SidecarError::Io {
            path: path.to_path_buf(),
            source,
        }
    })
}

/// The sampling config to seed a fresh sidecar with for `model`:
/// its own [`recommended_sampling`](crate::backend::Model::recommended_sampling)
/// compiled into a mode chain, or [`SamplerConfig::default()`] when
/// the model recommends nothing.
///
/// Returns the config plus whether it came from metadata (for
/// [`write_sample_options`]'s header).
///
/// Only `modes` is model-derived. `repetition` and friends stay at
/// the crate default: upstream's `penalty_repeat` / `penalty_last_n`
/// are scalars, while [`RepetitionOptions`](crate::RepetitionOptions)
/// is n-gram-based with
/// windowed decay, and there is no honest mapping between them.
#[cfg(feature = "toml")]
pub fn seed_config_for<M: crate::backend::Model>(
    model: &M,
) -> (SamplerConfig, bool) {
    let params = model.recommended_sampling();
    if params.is_empty() {
        return (SamplerConfig::default(), false);
    }
    let modes: Vec<crate::SamplingMode> = params.into();
    // `is_empty` was false, so the chain is non-empty by construction.
    debug_assert!(!modes.is_empty());
    (
        SamplerConfig {
            modes,
            ..SamplerConfig::default()
        },
        true,
    )
}

/// Read a dialect sidecar from `path` if it exists and parse it as
/// [`CallSyntax`](crate::CallSyntax).
///
/// Discovery convention mirrors the sampling sidecar: sibling file at
/// `<model>.dialect.toml` for GGUF (`model.gguf` →
/// `model.dialect.toml`), `parent/dialect.toml` for moeflux. Unlike
/// sampling, **no default is auto-written**: the template analyzer's
/// output *is* the default, and a sidecar exists only to override a
/// misdetected finetune. All fields are `#[serde(default)]`, so a
/// sidecar may specify only the fields it corrects — but note the
/// merge is whole-struct replacement, not per-field patching over the
/// analysis (simpler to reason about; a partial sidecar plus analyzer
/// output would make round-trip failures very hard to attribute).
#[cfg(feature = "toml")]
pub fn load_call_syntax(
    path: &Path,
) -> Result<Option<crate::CallSyntax>, SidecarError> {
    let bytes = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(SidecarError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let syntax: crate::CallSyntax =
        toml::from_str(&bytes).map_err(|source| SidecarError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(Some(syntax))
}

/// Read a chat-template sidecar from `path` if it exists — raw Jinja
/// source overriding the model's embedded `tokenizer.chat_template`.
///
/// Discovery convention mirrors the other sidecars: sibling file at
/// `<model>.template.jinja` for GGUF (`model.gguf` →
/// `model.template.jinja`), `parent/template.jinja` for moeflux. No
/// default is auto-written — a recognized model gets its baked
/// replacement automatically ([`crate::baked`], rung 2 of the
/// loading ladder) and the embedded template is the fallback; a
/// sidecar is the explicit per-install override (rung 1) for
/// patching a template neither of those got right. The dialect
/// analyzer re-runs against the override so grammar/parse/render
/// stay in lockstep.
pub fn load_template_source(
    path: &Path,
) -> Result<Option<String>, SidecarError> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(SidecarError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// Per-model load-time overrides, from a sibling `<model>.load.toml`
/// (`model.gguf` → `model.load.toml`). A field set here beats the
/// server-wide option of the same name (`LlamaCppOptions`, blallama's
/// flags); an unset one inherits it. Never auto-written. llama-cpp
/// only: moeflux sizes its context at compile time.
///
/// ```toml
/// # Qwen3.8-27B-UD-Q8_K_XL.load.toml — hybrid attention keeps KV
/// # small (~64 KiB/token), so this model can afford its trained
/// # window (with `--n-ctx 262144`; see `effective_n_ctx`).
/// n_ctx = 262144
/// n_ubatch = 2048
/// ```
///
/// Every context bound applies at once — `--n-ctx`, this `n_ctx`, and
/// the model's window — so a sidecar can lower a model's context below
/// the server-wide one but never raise it past `--n-ctx`.
///
/// Unknown keys are a parse error (a misspelled `n-ctx` must not
/// silently fall back to the default). A sidecar that fails to read
/// or parse is logged and ignored, like the other sidecars.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize,
)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct LoadSidecar {
    /// KV context size in tokens, capped at `--n-ctx` and at the
    /// model's window (see [`effective_n_ctx`]).
    pub n_ctx: Option<u32>,
    /// Micro-batch size (llama.cpp's `n_ubatch`, default 512), clamped
    /// to `n_batch`; an explicit `LlamaCppOptions::n_ubatch` wins (see
    /// [`effective_n_ubatch`]). Bigger buys a little prefill
    /// speed (~1–2% at 1024–4096 on Metal, flat above) for a bigger
    /// compute buffer (325–737 MiB at 512), so it spends Metal
    /// working-set headroom.
    pub n_ubatch: Option<u32>,
    /// YaRN factor over the model's original window: turns RoPE
    /// scaling on (or re-scales a GGUF that ships it) and stretches the
    /// window to `original × rope_scale` (see [`n_ctx_window`]). For a
    /// model trained without scaling — Qwen3.5+ at 262144, `4.0` for
    /// 1M — static YaRN costs some quality on short prompts, which is
    /// why it is per-model and opt-in. Leave it unset for a GGUF whose
    /// metadata already carries `rope.scaling.*` (gpt-oss, Mistral
    /// Small 4): those were trained *with* that factor. Values below
    /// `1.0`, or not finite, are ignored.
    pub rope_scale: Option<f32>,
    /// The K cache's element type; beats `LlamaCppOptions::cache_type_k`
    /// (`--cache-type-k`). See [`KvCacheType`].
    pub cache_type_k: Option<KvCacheType>,
    /// The V cache's element type; beats `LlamaCppOptions::cache_type_v`
    /// (`--cache-type-v`). See [`KvCacheType`].
    pub cache_type_v: Option<KvCacheType>,
}

/// An element type for the KV cache — llama.cpp's `--cache-type-k` /
/// `--cache-type-v` set, named as llama.cpp names them (`q8_0`).
///
/// KV is what bounds a long context: a dense model at 128k holds as
/// much cache as weights (cogito: 256 KiB per token at `f16`, 32 GiB).
/// `q8_0` halves that and is close to lossless; `q4_0` quarters it at a
/// measurable cost, more on K than on V. A quantized V cache needs
/// Flash Attention: llama.cpp turns it on under the default
/// [`FlashAttention::Auto`](crate::FlashAttention::Auto) and refuses
/// the context if it is forced off.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[non_exhaustive]
pub enum KvCacheType {
    /// 32-bit float.
    F32,
    /// 16-bit float: llama.cpp's default.
    F16,
    /// bfloat16.
    Bf16,
    /// 8-bit blocks of 32: half of `f16`, close to lossless.
    #[cfg_attr(feature = "cli", value(name = "q8_0"))]
    #[serde(rename = "q8_0")]
    Q8_0,
    /// 4-bit blocks of 32.
    #[cfg_attr(feature = "cli", value(name = "q4_0"))]
    #[serde(rename = "q4_0")]
    Q4_0,
    /// 4-bit blocks of 32, with a minimum.
    #[cfg_attr(feature = "cli", value(name = "q4_1"))]
    #[serde(rename = "q4_1")]
    Q4_1,
    /// 4-bit non-linear blocks of 32.
    #[cfg_attr(feature = "cli", value(name = "iq4_nl"))]
    #[serde(rename = "iq4_nl")]
    Iq4Nl,
    /// 5-bit blocks of 32.
    #[cfg_attr(feature = "cli", value(name = "q5_0"))]
    #[serde(rename = "q5_0")]
    Q5_0,
    /// 5-bit blocks of 32, with a minimum.
    #[cfg_attr(feature = "cli", value(name = "q5_1"))]
    #[serde(rename = "q5_1")]
    Q5_1,
}

impl KvCacheType {
    /// The llama.cpp name (`q8_0`), as `--cache-type-k` takes it.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Iq4Nl => "iq4_nl",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
        }
    }
}

impl std::fmt::Display for KvCacheType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl LoadSidecar {
    /// [`Self::rope_scale`] if it is a usable YaRN factor.
    pub fn yarn_factor(&self) -> Option<f32> {
        self.rope_scale.filter(|f| f.is_finite() && *f >= 1.0)
    }
}

/// Read a load sidecar from `path`, if it exists. Same contract as
/// [`load_sample_options`]: `Ok(None)` when absent.
#[cfg(feature = "toml")]
pub fn load_load_options(
    path: &Path,
) -> Result<Option<LoadSidecar>, SidecarError> {
    let bytes = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(SidecarError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    toml::from_str(&bytes)
        .map(Some)
        .map_err(|source| SidecarError::Parse {
            path: path.to_path_buf(),
            source,
        })
}

/// The window a model can attend over: its trained context
/// `n_ctx_train`, or with a YaRN `factor`, the `original` pre-scaling
/// window (the GGUF's `rope.scaling.original_context_length`, else
/// `n_ctx_train`) times the factor. `0` = unknown.
pub fn n_ctx_window(
    n_ctx_train: u32,
    original: Option<u32>,
    factor: Option<f32>,
) -> u32 {
    match factor {
        None => n_ctx_train,
        Some(factor) => {
            let original = original.filter(|&n| n != 0).unwrap_or(n_ctx_train);
            // Saturating: an absurd factor caps at u32::MAX rather
            // than wrapping.
            (original as f64 * factor as f64).min(u32::MAX as f64) as u32
        }
    }
}

/// The KV context a model is served with: the smallest of the
/// server-wide `default`, the load sidecar's `n_ctx` and the model's
/// `window` ([`n_ctx_window`]; `0` = unknown, uncapped). `None` only
/// when neither context is set, which leaves llama.cpp's default.
///
/// Capping the default too is what keeps one server-wide `--n-ctx`
/// from over-allocating a smaller model: past its window a model
/// produces garbage, not a longer answer, and the KV cells cost the
/// same either way (cogito, 131072 trained, OOM'd at `--n-ctx 262144`).
pub fn effective_n_ctx(
    default: Option<u32>,
    sidecar: Option<u32>,
    window: u32,
) -> Option<u32> {
    let n_ctx = [default, sidecar].into_iter().flatten().min()?;
    Some(match window {
        0 => n_ctx,
        window => n_ctx.min(window),
    })
}

/// The micro-batch a model is served with: an `explicit` option as-is
/// (tests pin it to the ubatch grid, so a sidecar must never move it),
/// else the load sidecar's clamped to `n_batch` (`0` is invalid and
/// ignored), else `None` — llama.cpp's default.
pub fn effective_n_ubatch(
    explicit: Option<u32>,
    sidecar: Option<u32>,
    n_batch: u32,
) -> Option<u32> {
    match (explicit, sidecar) {
        (Some(n_ubatch), _) => Some(n_ubatch),
        (None, None | Some(0)) => None,
        (None, Some(n_ubatch)) => Some(n_ubatch.min(n_batch.max(1))),
    }
}

/// Multimodal-projector sidecar convention: sibling
/// `<model>.mmproj.gguf` next to the `.gguf` file (`model.gguf` →
/// `model.mmproj.gguf`). Returns `Some(path)` only when the file
/// exists — unlike the sampling sidecar, nothing is auto-written; the
/// projector opts the model into vision *by existing*. Consumed by
/// `LlamaCppEngine`'s constructors (under the `mtmd` feature); a
/// present-but-unloadable projector is a hard error there, because
/// continuing text-only would silently drop images.
/// Symlinked models resolve through the link: if no sidecar sits next
/// to the link itself, the canonical target's sibling is checked —
/// the projector belongs with the real weights (`models/model.gguf →
/// /big/disk/qwen.gguf` finds `/big/disk/qwen.mmproj.gguf`).
pub fn mmproj_path(model_path: &Path) -> Option<std::path::PathBuf> {
    let path = model_path.with_extension("mmproj.gguf");
    if path.is_file() {
        return Some(path);
    }
    let canonical = std::fs::canonicalize(model_path).ok()?;
    if canonical == model_path {
        return None;
    }
    let path = canonical.with_extension("mmproj.gguf");
    path.is_file().then_some(path)
}

/// Serialize `syntax` to `path` as TOML. Utility for pinning an
/// analyzer result into an editable override (e.g. via a future CLI
/// `--dump-dialect`); nothing calls this automatically.
#[cfg(feature = "toml")]
pub fn write_call_syntax(
    path: &Path,
    syntax: &crate::CallSyntax,
) -> Result<(), SidecarError> {
    let body = toml::to_string_pretty(syntax)?;
    let header = "# drama_llama per-model tool-call dialect sidecar.\n\
         # Overrides the template analyzer's derived CallSyntax\n\
         # entirely (whole-struct replacement, not per-field patch).\n\
         # Delete to fall back to analysis.\n\n";
    std::fs::write(path, format!("{header}{body}")).map_err(|source| {
        SidecarError::Io {
            path: path.to_path_buf(),
            source,
        }
    })
}

#[cfg(all(test, feature = "toml"))]
mod tests {
    use super::*;

    /// CallSyntax dialect sidecar round-trips through TOML with all
    /// marker whitespace intact (newlines in markers are the trained
    /// format — losing one breaks round-trip byte-stability).
    #[test]
    fn call_syntax_roundtrip() {
        let dir = tempfile_dir();
        let path = dir.join("dialect.toml");

        assert!(load_call_syntax(&path).unwrap().is_none());

        for syntax in [
            crate::CallSyntax::qwen_xml(),
            crate::CallSyntax::hermes_json(),
            crate::CallSyntax::llama31_json(),
        ] {
            write_call_syntax(&path, &syntax).unwrap();
            let loaded = load_call_syntax(&path).unwrap().expect("written");
            assert_eq!(loaded, syntax);
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// Measured effort levels survive the TOML round-trip, and a
    /// sidecar written before the field existed still loads (as "no
    /// knob").
    #[test]
    fn call_syntax_efforts_roundtrip() {
        let dir = tempfile_dir();
        let path = dir.join("dialect.toml");

        let syntax = crate::CallSyntax::gpt_oss();
        assert!(!syntax.reasoning.efforts.is_empty());
        write_call_syntax(&path, &syntax).unwrap();
        let loaded = load_call_syntax(&path).unwrap().expect("written");
        assert_eq!(loaded, syntax);

        // An empty set is not written at all — which is exactly what an
        // older sidecar looks like.
        let syntax = crate::CallSyntax::qwen_xml();
        assert!(syntax.reasoning.efforts.is_empty());
        write_call_syntax(&path, &syntax).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(!body.contains("efforts"), "{body}");
        let loaded = load_call_syntax(&path).unwrap().expect("written");
        assert_eq!(loaded, syntax);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// Round-trip the default through `write_default → load`. Catches
    /// any field that can't be serialized (e.g. an `f32::NaN` slipping
    /// into a default) or any deserialize-side schema drift.
    #[test]
    fn default_roundtrip() {
        let dir = tempfile_dir();
        let path = dir.join("sampling.toml");

        // Sanity: load on empty dir returns Ok(None).
        let loaded = load_sample_options(&path).unwrap();
        assert!(loaded.is_none(), "no file should be Ok(None)");

        // Write default, then load — should round-trip equal.
        write_sample_options(&path, &SamplerConfig::default(), false).unwrap();
        let loaded = load_sample_options(&path).unwrap().expect("file written");
        assert_eq!(loaded, SamplerConfig::default());

        // Cleanup.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// A metadata-seeded chain survives the TOML round-trip intact —
    /// the whole point of seeding is that the numbers reach the file
    /// the user edits. Uses the exact triple Qwen3.6 advertises.
    #[test]
    fn metadata_seeded_roundtrip() {
        let dir = tempfile_dir();
        let path = dir.join("sampling.toml");

        let params = crate::SamplingParams {
            temp: Some(1.0),
            top_p: crate::Probability::from_f(0.95).ok(),
            top_k: std::num::NonZeroUsize::new(20),
            min_p: None,
            mirostat: None,
        };
        let seeded = SamplerConfig {
            modes: params.into(),
            ..SamplerConfig::default()
        };
        assert_ne!(
            seeded,
            SamplerConfig::default(),
            "test is vacuous if the seed matches the crate default"
        );

        write_sample_options(&path, &seeded, true).unwrap();
        let loaded = load_sample_options(&path).unwrap().expect("file written");
        assert_eq!(loaded, seeded);

        // The provenance header is the only way a user can tell a
        // model's recommendation from the crate default on disk.
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(
            raw.contains("general.sampling.*"),
            "seeded sidecar must say where its numbers came from"
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// The tool-call cap reads from the sidecar beside the rest of the
    /// config; absent is no cap, and a cap of zero calls is refused
    /// (`tool_choice: none` is how a turn makes none). It is never
    /// written unless set, so a seeded sidecar does not gain the key.
    #[test]
    fn max_tool_calls_per_turn_parses() {
        let dir = tempfile_dir();
        let path = dir.join("sampling.toml");

        std::fs::write(&path, "modes = []\nmax_tool_calls_per_turn = 3\n")
            .unwrap();
        let loaded = load_sample_options(&path).unwrap().expect("written");
        assert_eq!(
            loaded.max_tool_calls_per_turn,
            std::num::NonZeroU32::new(3)
        );
        assert_eq!(loaded.tool_call_cap, None, "runtime wiring only");

        std::fs::write(&path, "modes = []\n").unwrap();
        let loaded = load_sample_options(&path).unwrap().expect("written");
        assert_eq!(loaded.max_tool_calls_per_turn, None);

        std::fs::write(&path, "modes = []\nmax_tool_calls_per_turn = 0\n")
            .unwrap();
        assert!(matches!(
            load_sample_options(&path),
            Err(SidecarError::Parse { .. })
        ));

        write_sample_options(&path, &SamplerConfig::default(), false).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("max_tool_calls_per_turn"), "{raw}");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// The tracked sidecar cogito serves with caps its calls: it escalated
    /// parallel calls into loops that ran to `max_tokens` (2026-10-04).
    #[test]
    fn cogito_sidecar_caps_tool_calls() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/cogito-32b.sampling.toml");
        let loaded = load_sample_options(&path).unwrap().expect("tracked");
        assert_eq!(
            loaded.max_tool_calls_per_turn,
            std::num::NonZeroU32::new(3)
        );
    }

    /// Malformed TOML reports a Parse error tagged with the path.
    #[test]
    fn malformed_toml_reports_parse_error() {
        let dir = tempfile_dir();
        let path = dir.join("bad.toml");
        std::fs::write(&path, b"this is = not [valid toml").unwrap();

        let err = load_sample_options(&path).unwrap_err();
        match err {
            SidecarError::Parse { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected Parse, got {other:?}"),
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// The load sidecar the docs show parses; absent is `Ok(None)`, an
    /// empty file sets nothing, and a misspelled key is an error rather
    /// than a silent fall back to the default.
    #[test]
    fn load_sidecar_parses() {
        let dir = tempfile_dir();
        let path = dir.join("model.load.toml");
        assert!(load_load_options(&path).unwrap().is_none());

        std::fs::write(&path, "# Qwen3.8\nn_ctx = 262144\n").unwrap();
        let sidecar = load_load_options(&path).unwrap().expect("written");
        assert_eq!(sidecar.n_ctx, Some(262144));
        assert_eq!(sidecar.n_ubatch, None);

        std::fs::write(&path, "n_ctx = 131072\nn_ubatch = 2048\n").unwrap();
        let sidecar = load_load_options(&path).unwrap().expect("written");
        assert_eq!(sidecar.n_ctx, Some(131072));
        assert_eq!(sidecar.n_ubatch, Some(2048));

        std::fs::write(&path, "n_ubatch = 1024").unwrap();
        let sidecar = load_load_options(&path).unwrap().expect("written");
        assert_eq!((sidecar.n_ctx, sidecar.n_ubatch), (None, Some(1024)));

        std::fs::write(&path, "n_ctx = 1048576\nrope_scale = 4.0\n").unwrap();
        let sidecar = load_load_options(&path).unwrap().expect("written");
        assert_eq!(sidecar.rope_scale, Some(4.0));
        std::fs::write(&path, "rope_scale = 4").unwrap();
        let sidecar = load_load_options(&path).unwrap().expect("written");
        assert_eq!(sidecar.rope_scale, Some(4.0), "an integer factor");

        std::fs::write(
            &path,
            "cache_type_k = \"q8_0\"\ncache_type_v = \"iq4_nl\"",
        )
        .unwrap();
        let sidecar = load_load_options(&path).unwrap().expect("written");
        assert_eq!(sidecar.cache_type_k, Some(KvCacheType::Q8_0));
        assert_eq!(sidecar.cache_type_v, Some(KvCacheType::Iq4Nl));

        std::fs::write(&path, "").unwrap();
        let sidecar = load_load_options(&path).unwrap().expect("written");
        assert_eq!(sidecar, LoadSidecar::default());

        for bad in [
            "n-ctx = 262144",
            "n_ctx = -1",
            "n_ctx = \"256k\"",
            "n-ubatch = 2048",
            "n_ubatch = -1",
            "n_ubatch = \"2k\"",
            "rope-scale = 4.0",
            "rope_scale = \"4x\"",
            "cache_type_k = \"q8\"",
            "cache_type_k = \"Q8_0\"",
            "cache-type-v = \"q8_0\"",
        ] {
            std::fs::write(&path, bad).unwrap();
            assert!(
                matches!(
                    load_load_options(&path),
                    Err(SidecarError::Parse { .. })
                ),
                "{bad}"
            );
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// Every bound applies, with the fleet's numbers at
    /// `--n-ctx 262144`: the smallest of the default, the sidecar and
    /// the window wins, and an unknown window caps nothing.
    #[test]
    fn effective_n_ctx_takes_the_smallest_bound() {
        const DEFAULT: Option<u32> = Some(262144);
        // Qwen3.8: its sidecar matches the default and its window.
        assert_eq!(
            effective_n_ctx(DEFAULT, Some(262144), 262144),
            Some(262144)
        );
        // cogito: no sidecar, the default capped at its window.
        assert_eq!(effective_n_ctx(DEFAULT, None, 131072), Some(131072));
        // Mistral Small 4 (1M window): its sidecar lowers it.
        assert_eq!(
            effective_n_ctx(DEFAULT, Some(131072), 1 << 20),
            Some(131072)
        );
        // A sidecar can't raise past the default.
        assert_eq!(
            effective_n_ctx(Some(131072), Some(262144), 262144),
            Some(131072)
        );
        // Past the window: capped.
        assert_eq!(
            effective_n_ctx(DEFAULT, Some(1 << 20), 262144),
            Some(262144)
        );
        // Window unknown: the smaller context stands.
        assert_eq!(effective_n_ctx(DEFAULT, Some(1 << 20), 0), DEFAULT);
        assert_eq!(effective_n_ctx(None, Some(1 << 20), 0), Some(1 << 20));
        // Neither context set: llama.cpp's default.
        assert_eq!(effective_n_ctx(None, None, 40960), None);
        // A sidecar works without any default.
        assert_eq!(effective_n_ctx(None, Some(8192), 40960), Some(8192));
    }

    /// The window is the trained context unless YaRN is asked for,
    /// then the original window times the factor.
    #[test]
    fn n_ctx_window_scales_the_original() {
        // No factor: the trained context, whatever the original.
        assert_eq!(n_ctx_window(262144, None, None), 262144);
        assert_eq!(n_ctx_window(1 << 20, Some(8192), None), 1 << 20);
        // Qwen3.8 → 1M: no scaling metadata, so the trained context
        // is the original.
        assert_eq!(n_ctx_window(262144, None, Some(4.0)), 1 << 20);
        // Mistral Small 4 re-scaled: the GGUF's original, not its
        // already-scaled context.
        assert_eq!(n_ctx_window(1 << 20, Some(8192), Some(16.0)), 131072);
        // An original of 0 means absent.
        assert_eq!(n_ctx_window(262144, Some(0), Some(2.0)), 524288);
        // Saturates instead of wrapping.
        assert_eq!(n_ctx_window(262144, None, Some(1e9)), u32::MAX);
    }

    /// Every KV cache type serializes as its llama.cpp name.
    #[test]
    fn kv_cache_type_names_match_llama_cpp() {
        use KvCacheType::*;
        // Exhaustive, so a new variant has to be named here.
        let all = [F32, F16, Bf16, Q8_0, Q4_0, Q4_1, Iq4Nl, Q5_0, Q5_1];
        for t in all {
            match t {
                F32 | F16 | Bf16 | Q8_0 | Q4_0 | Q4_1 | Iq4Nl | Q5_0 | Q5_1 => {
                }
            }
            let json = serde_json::to_string(&t).unwrap();
            assert_eq!(json, format!("\"{}\"", t.name()));
            assert_eq!(serde_json::from_str::<KvCacheType>(&json).unwrap(), t);
            assert_eq!(t.to_string(), t.name());
        }
    }

    /// Only a finite factor of at least 1 turns YaRN on.
    #[test]
    fn yarn_factor_filters_unusable_values() {
        let factor = |rope_scale| {
            LoadSidecar {
                rope_scale,
                ..Default::default()
            }
            .yarn_factor()
        };
        assert_eq!(factor(Some(4.0)), Some(4.0));
        assert_eq!(factor(Some(1.0)), Some(1.0));
        assert_eq!(factor(None), None);
        assert_eq!(factor(Some(0.5)), None);
        assert_eq!(factor(Some(0.0)), None);
        assert_eq!(factor(Some(f32::NAN)), None);
        assert_eq!(factor(Some(f32::INFINITY)), None);
    }

    /// An explicit option beats the sidecar, which beats llama.cpp's
    /// default; the sidecar's value is clamped to `n_batch` and `0`
    /// is ignored. blallama sets `n_batch = n_ctx`.
    #[test]
    fn effective_n_ubatch_precedence_and_clamp() {
        const N_BATCH: u32 = 131072;
        // Sidecar over the default.
        assert_eq!(effective_n_ubatch(None, Some(2048), N_BATCH), Some(2048));
        assert_eq!(effective_n_ubatch(None, None, N_BATCH), None);
        // An explicit option (the #126 grid pin) beats the sidecar,
        // either way, and is never clamped here.
        assert_eq!(effective_n_ubatch(Some(31), Some(2048), N_BATCH), Some(31));
        assert_eq!(
            effective_n_ubatch(Some(4096), Some(512), N_BATCH),
            Some(4096)
        );
        assert_eq!(effective_n_ubatch(Some(64), None, 32), Some(64));
        // Clamped to n_batch.
        assert_eq!(effective_n_ubatch(None, Some(4096), 1024), Some(1024));
        assert_eq!(effective_n_ubatch(None, Some(1024), 1024), Some(1024));
        // Invalid: ignored, so the default stands.
        assert_eq!(effective_n_ubatch(None, Some(0), N_BATCH), None);
        // A degenerate n_batch still yields a valid micro-batch.
        assert_eq!(effective_n_ubatch(None, Some(2048), 0), Some(1));
    }

    /// Test-local tempfile dir that doesn't depend on the `tempfile`
    /// crate (which isn't in the dev-dependencies list).
    fn tempfile_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("drama_llama_sidecar_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
