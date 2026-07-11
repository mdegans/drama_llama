//! Per-model sidecar files.
//!
//! A sidecar is a file colocated with a model on disk that overrides
//! one aspect of how it is served: sampling defaults
//! (`<model>.sampling.toml`, [`SampleOptions`]), the tool-call
//! dialect (`<model>.dialect.toml`,
//! [`CallSyntax`](crate::CallSyntax)), the chat template itself
//! (`<model>.template.jinja`, raw Jinja), or the multimodal
//! projector (`<model>.mmproj.gguf`, [`mmproj_path`] — enables image
//! input under the `mtmd` feature). [`crate::LlamaCppSession::from_path*`]
//! looks for each when loading a model. For sampling, if no sidecar
//! exists a default is written so the user has a starting point to
//! edit.
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
//! Everything in [`SampleOptions`] that is `Serialize` /
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
//! - To **reset** to defaults: delete the sidecar file. The next load
//!   will rewrite the default.
//! - To **tweak** something: edit the sidecar, save, restart.
//!
//! [`crate::LlamaCppSession::from_path*`]: crate::crate::LlamaCppSession::from_path
//! [`Session::with_sample_options`]: crate::Session::with_sample_options
//! [`SamplingMode::Json`]: crate::SamplingMode::Json
//! [`SamplingMode::Grammar`]: crate::SamplingMode::Grammar
//! [`SamplingMode::Deny`]: crate::SamplingMode::Deny

use std::path::Path;

#[cfg(feature = "toml")]
use crate::SampleOptions;

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
/// [`SampleOptions`].
///
/// Returns:
/// - `Ok(Some(opts))` — sidecar found and parsed.
/// - `Ok(None)` — sidecar does not exist (the common
///   first-time-loading-a-model case).
/// - `Err(SidecarError::Io)` — file exists but couldn't be read
///   (permissions, etc.).
/// - `Err(SidecarError::Parse)` — file exists but contains malformed
///   TOML or TOML that doesn't deserialize into [`SampleOptions`].
#[cfg(feature = "toml")]
pub fn load_sample_options(
    path: &Path,
) -> Result<Option<SampleOptions>, SidecarError> {
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
    let opts: SampleOptions =
        toml::from_str(&bytes).map_err(|source| SidecarError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(Some(opts))
}

/// Write [`SampleOptions::default()`] to `path` as TOML so the user
/// has a starting point to edit. Best-effort: if the parent dir
/// doesn't exist or the file isn't writable, returns the underlying
/// IO error and the caller decides whether to log + continue.
///
/// Does *not* overwrite an existing file — call
/// [`load_sample_options`] first to detect existence; the
/// [`crate::LlamaCppSession::from_path*`] integration only writes when the read
/// returned `Ok(None)`.
///
/// [`crate::LlamaCppSession::from_path*`]: crate::crate::LlamaCppSession::from_path
#[cfg(feature = "toml")]
pub fn write_default_sample_options(path: &Path) -> Result<(), SidecarError> {
    let opts = SampleOptions::default();
    let body = toml::to_string_pretty(&opts)?;
    let header = "# drama_llama per-model sampling sidecar.\n\
         # Edit to tune sampling for this model. Delete to reset to\n\
         # SampleOptions::default(); the next load will rewrite this\n\
         # file.\n\
         #\n\
         # See drama_llama::sidecar module docs for the layout\n\
         # convention and what's intentionally excluded (Json,\n\
         # Grammar, Deny modes — those are per-request runtime, not\n\
         # per-model defaults).\n\n";
    std::fs::write(path, format!("{header}{body}")).map_err(|source| {
        SidecarError::Io {
            path: path.to_path_buf(),
            source,
        }
    })
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
pub fn mmproj_path(model_path: &Path) -> Option<std::path::PathBuf> {
    let path = model_path.with_extension("mmproj.gguf");
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
        write_default_sample_options(&path).unwrap();
        let loaded = load_sample_options(&path).unwrap().expect("file written");
        assert_eq!(loaded, SampleOptions::default());

        // Cleanup.
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
