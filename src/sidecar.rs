//! Per-model sampling sidecar files.
//!
//! A sidecar is a TOML file colocated with a model on disk that holds
//! that model's preferred [`SampleOptions`] — sampling-mode chain,
//! repetition-penalty config, etc. [`Session::from_path*`] looks for
//! one when loading a model and applies it via
//! [`Session::with_sample_options`]. If no sidecar exists, a default is
//! written so the user has a starting point to edit.
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
//! - `modes` — the sampling-mode chain ([`SamplingMode::TopP`],
//!   [`SamplingMode::Mirostat`], etc.)
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
//! [`Session::from_path*`]: crate::Session::from_path
//! [`Session::with_sample_options`]: crate::Session::with_sample_options
//! [`SamplingMode::Json`]: crate::SamplingMode::Json
//! [`SamplingMode::Grammar`]: crate::SamplingMode::Grammar
//! [`SamplingMode::Deny`]: crate::SamplingMode::Deny

use std::path::Path;

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
    #[error("sidecar TOML parse at {path:?}: {source}")]
    Parse {
        path: std::path::PathBuf,
        #[source]
        source: toml::de::Error,
    },
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
/// [`Session::from_path*`] integration only writes when the read
/// returned `Ok(None)`.
///
/// [`Session::from_path*`]: crate::Session::from_path
pub fn write_default_sample_options(
    path: &Path,
) -> Result<(), SidecarError> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let dir = std::env::temp_dir().join(format!(
            "drama_llama_sidecar_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
