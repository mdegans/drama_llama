//! Per-model sidecar files.
//!
//! A sidecar is a file colocated with a model on disk that overrides
//! one aspect of how it is served: sampling defaults
//! (`<model>.sampling.toml`, [`SamplerConfig`]), the tool-call
//! dialect (`<model>.dialect.toml`,
//! [`CallSyntax`](crate::CallSyntax)), the chat template itself
//! (`<model>.template.jinja`, raw Jinja), or the multimodal
//! projector (`<model>.mmproj.gguf`, [`mmproj_path`] — enables image
//! input under the `mtmd` feature). [`crate::LlamaCppSession::from_path*`]
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
//!   disable.
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
//! [`crate::LlamaCppSession::from_path*`]: crate::LlamaCppSession::from_path_sync
//! [`Session::with_sample_options`]: crate::Session::with_sample_options
//! [`SamplingMode::Json`]: crate::SamplingMode::Json
//! [`SamplingMode::Grammar`]: crate::SamplingMode::Grammar
//! [`SamplingMode::Deny`]: crate::SamplingMode::Deny

use std::path::Path;

#[cfg(feature = "toml")]
use crate::SamplerConfig;

/// Failure mode for sidecar I/O.
#[derive(Debug, thiserror::Error)]
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
/// [`crate::LlamaCppSession::from_path*`]: crate::LlamaCppSession::from_path_sync
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
/// default is auto-written — the embedded template *is* the default;
/// a sidecar exists to patch serving-side template bugs (e.g. the
/// vendored `gemma4-cache-stable.jinja`, which fixes Gemma 4's
/// re-ingest path dropping the thinking channel and breaking
/// KV-cache byte-stability). The dialect analyzer re-runs against
/// the override so grammar/parse/render stay in lockstep.
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

    /// Test-local tempfile dir that doesn't depend on the `tempfile`
    /// crate (which isn't in the dev-dependencies list).
    fn tempfile_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("drama_llama_sidecar_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
