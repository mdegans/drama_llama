//! [`Catalog`] — the models a directory can serve, each described by a
//! [`ModelInfo`] read from disk once per process.
//!
//! A [`Session`] is one loaded model, so
//! [`Transport::models`](misanthropic::Transport::models) on a
//! [`SessionTransport`](crate::SessionTransport) rightly answers with one
//! entry. A server that swaps models in and out of a directory
//! (blallama) needs the other question answered — *what could I serve?*
//! — for every model on disk, loaded or not, and without paying a weight
//! load to find out. That is [`FromPath::peek`], and this is the cache in
//! front of it.
//!
//! Both answers are built by the same (private) `Advertised` mapping,
//! so a model's listing and its loaded [`Session::model_info`] never
//! disagree.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Mutex,
    time::SystemTime,
};

use chrono::{DateTime, Utc};
use misanthropic::model::{
    Capabilities, Capability, ModelInfo, Models, ThinkingSupport,
};

use crate::{prompt::AnthropicError, Backend, FromPath, Session};

/// What a model advertises about itself — the facts behind a
/// [`ModelInfo`], gathered either from a loaded [`Session`] or from a
/// weightless [`FromPath::peek`]. `From<Advertised> for ModelInfo` is the
/// one place the capability mapping is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Advertised {
    /// The API id — the directory entry name a request names.
    pub id: String,
    /// [`Model::title`](crate::backend::Model::title); the id stands in
    /// when absent.
    pub title: Option<String>,
    /// The KV context the model is (or would be) served with.
    pub n_ctx: u32,
    /// [`Model::context_size`](crate::backend::Model::context_size) — the
    /// trained context; `0` when the backend doesn't know.
    pub n_ctx_train: u32,
    /// Whether images can be attached (a vision projector is present).
    pub image_input: bool,
    /// Whether the dialect has a reasoning syntax — whether
    /// `thinking: {type: enabled}` does anything.
    pub thinking: bool,
    /// The model file's modification time; `None` when unknown.
    pub modified: Option<SystemTime>,
}

impl Advertised {
    /// The token ceiling advertised for both input and output:
    /// `min(n_ctx, n_ctx_train)`, matching what llama.cpp's server reports
    /// as `n_ctx` (`n_ctx_slot()`), because a KV context larger than the
    /// trained window is honored by the decoder but not by the model —
    /// past `n_ctx_train` the output is garbage, not a longer answer.
    ///
    /// The known under-advertisement: models whose long context comes
    /// from RoPE scaling declare the native window (Qwen3: 40960; 131072
    /// under YaRN). llama.cpp makes the same call, and it is the honest
    /// side to err on.
    fn ceiling(&self) -> u32 {
        if self.n_ctx_train == 0 {
            self.n_ctx
        } else {
            self.n_ctx.min(self.n_ctx_train)
        }
    }
}

impl From<Advertised> for ModelInfo {
    fn from(a: Advertised) -> Self {
        let ceiling = a.ceiling();
        let display_name = a.title.clone().unwrap_or_else(|| a.id.clone());
        let mut info = ModelInfo::new(a.id.clone(), display_name);
        info.max_input_tokens = ceiling;
        info.max_tokens = ceiling;
        if let Some(modified) = a.modified {
            info.created_at = DateTime::<Utc>::from(modified);
        }
        info.capabilities = Capabilities {
            // Grammar-constrained decoding: every session can honor an
            // `output_config` schema.
            structured_outputs: Capability::from(true),
            image_input: Capability::from(a.image_input),
            thinking: ThinkingSupport {
                supported: a.thinking,
                types: if a.thinking {
                    [("enabled".to_string(), Capability::from(true))].into()
                } else {
                    Default::default()
                },
            },
            // No batches endpoint, no citations, no server-side tools, no
            // context editing, no effort levels, no PDF ingestion.
            ..Default::default()
        };
        info
    }
}

/// Identity of a model file at one moment — enough to notice it was
/// replaced. Follows symlinks (`fs::metadata`), so a re-pointed link
/// re-peeks too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    modified: Option<SystemTime>,
    len: u64,
}

impl Stamp {
    fn of(meta: &std::fs::Metadata) -> Self {
        Self {
            modified: meta.modified().ok(),
            len: meta.len(),
        }
    }
}

struct Cached {
    stamp: Stamp,
    info: ModelInfo,
}

/// The models under one directory, with a per-process cache of their
/// [`ModelInfo`]. See the [module docs](self).
///
/// Listing (`read_dir`) is fresh on every call — it is cheap, and a model
/// dropped into the directory should appear without a restart. The
/// metadata read ([`FromPath::peek`]) is what's cached, keyed by the
/// entry name and invalidated when the file's stamp (mtime + size) changes.
///
/// Sync by design: the peek is blocking I/O (and, for llama.cpp, a vocab
/// load — a second or two per model), so an async server calls this
/// from its blocking pool. `Sync` via two mutexes: the cache's, held
/// only to read or insert, and a *peek lane* that serializes misses.
/// Cached reads never wait on a peek in flight; a second miss for the
/// same model waits for the first and then finds its result. A server
/// that wants a warm first response calls [`models`](Self::models) once
/// at startup, off the request path.
pub struct Catalog<B: Backend>
where
    Session<B>: FromPath,
{
    root: PathBuf,
    options: <Session<B> as FromPath>::Options,
    cache: Mutex<HashMap<String, Cached>>,
    /// Held by whoever is peeking; see the type docs.
    peek_lane: Mutex<()>,
}

impl<B: Backend> std::fmt::Debug for Catalog<B>
where
    Session<B>: FromPath,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Catalog")
            .field("backend", &B::NAME)
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl<B: Backend> Catalog<B>
where
    Session<B>: FromPath,
{
    /// A catalog over `root`, describing each model as `options` would
    /// load it (the served context size comes from there).
    pub fn new(
        root: impl Into<PathBuf>,
        options: <Session<B> as FromPath>::Options,
    ) -> Self {
        Self {
            root: root.into(),
            options,
            cache: Mutex::new(HashMap::new()),
            peek_lane: Mutex::new(()),
        }
    }

    /// The cached entry for `name`, if it is still `stamp`.
    fn cached(&self, name: &str, stamp: Stamp) -> Option<ModelInfo> {
        let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .get(name)
            .filter(|c| c.stamp == stamp)
            .map(|c| c.info.clone())
    }

    /// The directory being served.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The load options every model is described (and loaded) with.
    pub fn options(&self) -> &<Session<B> as FromPath>::Options {
        &self.options
    }

    /// The path a listed `name` loads from.
    pub fn path_of(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Directory entries the backend accepts as models, per
    /// [`Backend::is_supported_model`] — unsorted, straight from
    /// `read_dir`.
    ///
    /// Uses `fs::metadata(path)` rather than `DirEntry::metadata()` or
    /// `file_type()`. **Only the first of the three follows symlinks** —
    /// this is a genuine trap, because `DirEntry::metadata()` reads like
    /// the one that would: it does not, and returns `is_symlink() == true`
    /// / `is_file() == false`, so a symlinked model is silently dropped
    /// from the listing. Mike's test layout symlinks `mlx` / `artifacts` /
    /// `root` into a single moeflux model dir, CI symlinks every `.gguf`
    /// in from a shared read-only `/models`, and the dir itself can be a
    /// symlink — all of those forms must enumerate.
    pub fn list(&self) -> std::io::Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            // NOT `entry.metadata()`: that one does not traverse the link
            // (it is `symlink_metadata` in all but name). Skip entries
            // whose target is missing or unreadable — a dangling link is
            // not a servable model.
            let Ok(meta) = std::fs::metadata(entry.path()) else {
                continue;
            };
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            if B::is_supported_model(&name, &meta) {
                names.push(name);
            }
        }
        Ok(names)
    }

    /// Resolve a requested model id to the one to serve: `requested` if
    /// it is on disk, else `default` if *that* is (unmodified
    /// Anthropic-SDK clients request `claude-*` ids), else the `404`
    /// payload. Names only — nothing is peeked.
    pub fn resolve(
        &self,
        requested: &str,
        default: Option<&str>,
    ) -> Result<String, AnthropicError> {
        let names = self.list().map_err(|e| AnthropicError::NotFound {
            message: format!("Models could not be loaded: {e}"),
        })?;
        if names.iter().any(|n| n == requested) {
            return Ok(requested.to_string());
        }
        if let Some(d) = default {
            if names.iter().any(|n| n == d) {
                return Ok(d.to_string());
            }
        }
        Err(AnthropicError::NotFound {
            message: format!("model not found: {requested}"),
        })
    }

    /// Stamp a listed model; `None` if `name` is not a plain entry name,
    /// is missing, or isn't a model the backend accepts.
    fn stamp(&self, name: &str) -> Option<Stamp> {
        // A bare entry name, not a path: `/v1/models/{id}` is
        // client-supplied, and `..` must not walk out of `root`.
        if Path::new(name).file_name() != Some(name.as_ref()) {
            return None;
        }
        let meta = std::fs::metadata(self.path_of(name)).ok()?;
        B::is_supported_model(name, &meta).then(|| Stamp::of(&meta))
    }

    /// The model file's size in bytes (`0` for a directory-shaped
    /// model), if `name` is listed.
    pub fn size(&self, name: &str) -> Option<u64> {
        self.stamp(name).map(|s| s.len)
    }

    /// [`ModelInfo`] for a listed model, from the cache when the file is
    /// unchanged since it was read, else via [`FromPath::peek`].
    ///
    /// A peek that fails yields a degraded, name-only entry (the file is
    /// on disk, so it stays listable and resolvable; the load will report
    /// the real error) and is **not** cached, so the next call retries.
    /// `None` only when `name` isn't a listed model.
    pub fn info(&self, name: &str) -> Option<ModelInfo> {
        let stamp = self.stamp(name)?;
        if let Some(hit) = self.cached(name, stamp) {
            return Some(hit);
        }
        // A miss. Queue behind any peek in flight, then look again: the
        // one we waited on may have been this model.
        let _lane = self.peek_lane.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(hit) = self.cached(name, stamp) {
            return Some(hit);
        }
        let path = self.path_of(name);
        // Stage-logged (#95 style): if a read ever wedges — a model on a
        // network mount, a GGUF that makes the loader loop — the last
        // "reading" line without its "read" names the file.
        tracing::info!(
            event = "read_model_metadata",
            backend = B::NAME,
            model = name,
            path = %path.display(),
            "reading model metadata",
        );
        let started = std::time::Instant::now();
        let info = match <Session<B> as FromPath>::peek(&path, &self.options) {
            Ok(mut info) => {
                if let Some(modified) = stamp.modified {
                    info.created_at = DateTime::<Utc>::from(modified);
                }
                tracing::info!(
                    event = "model_metadata_read",
                    model = name,
                    display_name = %info.display_name,
                    max_input_tokens = info.max_input_tokens,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "model metadata read",
                );
                let mut cache =
                    self.cache.lock().unwrap_or_else(|e| e.into_inner());
                cache.insert(
                    name.to_string(),
                    Cached {
                        stamp,
                        info: info.clone(),
                    },
                );
                info
            }
            Err(e) => {
                tracing::warn!(
                    event = "model_metadata_failed",
                    backend = B::NAME,
                    model = name,
                    error = %e,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "could not read model metadata; listing it by name only",
                );
                Advertised {
                    id: name.to_string(),
                    title: None,
                    n_ctx: 0,
                    n_ctx_train: 0,
                    image_input: false,
                    thinking: false,
                    modified: stamp.modified,
                }
                .into()
            }
        };
        Some(info)
    }

    /// Every listed model's [`ModelInfo`], sorted by id. An unreadable
    /// directory lists as empty (the failure is logged).
    ///
    /// Also the warm-up: called once at startup, off the request path,
    /// it fills the cache so the first client listing is served from
    /// memory.
    pub fn models(&self) -> Models {
        let mut names = match self.list() {
            Ok(names) => names,
            Err(e) => {
                tracing::error!(
                    root = %self.root.display(),
                    error = %e,
                    "could not list models",
                );
                Vec::new()
            }
        };
        names.sort_unstable();
        names.iter().filter_map(|name| self.info(name)).collect()
    }

    /// Replace the cached entry for a listed model with what a *loaded*
    /// session reports ([`Session::model_info`]) — the decoder's real
    /// context size (llama.cpp pads `n_ctx`) beats the peek's estimate.
    /// `created_at` is filled from the file, since a session doesn't know
    /// its own. A no-op if `name` isn't listed.
    pub fn refresh(&self, name: &str, mut info: ModelInfo) {
        let Some(stamp) = self.stamp(name) else {
            return;
        };
        if let Some(modified) = stamp.modified {
            info.created_at = DateTime::<Utc>::from(modified);
        }
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.insert(name.to_string(), Cached { stamp, info });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceiling_is_min_of_served_and_trained_unless_trained_unknown() {
        let mut a = Advertised {
            id: "m.gguf".into(),
            title: None,
            n_ctx: 8192,
            n_ctx_train: 4096,
            image_input: false,
            thinking: false,
            modified: None,
        };
        assert_eq!(a.ceiling(), 4096);
        a.n_ctx = 2048;
        assert_eq!(a.ceiling(), 2048);
        a.n_ctx_train = 0;
        assert_eq!(a.ceiling(), 2048);
    }

    #[test]
    fn model_info_mapping() {
        let modified = SystemTime::UNIX_EPOCH
            + std::time::Duration::from_secs(1_700_000_000);
        let info: ModelInfo = Advertised {
            id: "qwen.gguf".into(),
            title: Some("Qwen3.6 35B A3B".into()),
            n_ctx: 16384,
            n_ctx_train: 40960,
            image_input: true,
            thinking: true,
            modified: Some(modified),
        }
        .into();
        assert_eq!(info.id, "qwen.gguf");
        assert_eq!(info.display_name, "Qwen3.6 35B A3B");
        assert_eq!(info.max_input_tokens, 16384);
        assert_eq!(info.max_tokens, 16384);
        assert_eq!(info.created_at.timestamp(), 1_700_000_000);
        assert!(info.capabilities.structured_outputs == true);
        assert!(info.capabilities.image_input == true);
        assert!(info.capabilities.thinking.supported);
        assert!(info.capabilities.thinking.types["enabled"] == true);
        assert!(info.capabilities.batch == false);
        assert!(info.capabilities.pdf_input == false);
        assert!(!info.capabilities.effort.supported);

        // Title falls back to the id; no thinking means no types either.
        let info: ModelInfo = Advertised {
            id: "plain.gguf".into(),
            title: None,
            n_ctx: 512,
            n_ctx_train: 0,
            image_input: false,
            thinking: false,
            modified: None,
        }
        .into();
        assert_eq!(info.display_name, "plain.gguf");
        assert_eq!(info.max_input_tokens, 512);
        assert!(!info.capabilities.thinking.supported);
        assert!(info.capabilities.thinking.types.is_empty());
        assert_eq!(info.created_at, DateTime::<Utc>::UNIX_EPOCH);
    }

    #[cfg(feature = "llama-cpp")]
    mod llama_cpp {
        use super::*;
        use crate::{LlamaCppBackend, LlamaCppOptions};

        fn catalog(dir: &Path) -> Catalog<LlamaCppBackend> {
            Catalog::new(dir, LlamaCppOptions::default().with_n_ctx(1024))
        }

        /// Empty files are enough for listing: `is_supported_model` is
        /// name + `is_file`. Symlinks must enumerate; projector sidecars
        /// and dangling links must not.
        #[test]
        fn list_and_resolve() {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path();
            std::fs::write(root.join("a.gguf"), b"").unwrap();
            std::fs::write(root.join("b.gguf"), b"").unwrap();
            std::fs::write(root.join("b.mmproj.gguf"), b"").unwrap();
            std::fs::write(root.join("notes.txt"), b"").unwrap();
            std::fs::create_dir(root.join("dir.gguf")).unwrap();
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(
                    root.join("a.gguf"),
                    root.join("link.gguf"),
                )
                .unwrap();
                std::os::unix::fs::symlink(
                    root.join("missing.gguf"),
                    root.join("dangling.gguf"),
                )
                .unwrap();
            }

            let catalog = catalog(root);
            let mut names = catalog.list().unwrap();
            names.sort();
            #[cfg(unix)]
            assert_eq!(names, ["a.gguf", "b.gguf", "link.gguf"]);
            #[cfg(not(unix))]
            assert_eq!(names, ["a.gguf", "b.gguf"]);

            assert_eq!(catalog.resolve("a.gguf", None).unwrap(), "a.gguf");
            assert_eq!(
                catalog.resolve("claude-opus-5", Some("b.gguf")).unwrap(),
                "b.gguf"
            );
            assert!(matches!(
                catalog.resolve("claude-opus-5", None),
                Err(AnthropicError::NotFound { .. })
            ));
            assert!(matches!(
                catalog.resolve("claude-opus-5", Some("nope.gguf")),
                Err(AnthropicError::NotFound { .. })
            ));
            // Sidecars and non-models are not resolvable even by name.
            assert!(catalog.resolve("b.mmproj.gguf", None).is_err());
            assert!(catalog.resolve("notes.txt", None).is_err());

            assert_eq!(catalog.size("a.gguf"), Some(0));
            assert_eq!(catalog.size("notes.txt"), None);
            assert_eq!(catalog.size("../a.gguf"), None);
        }

        /// A file that isn't a GGUF peeks to an error: it lists by name
        /// only, isn't cached (a later call re-peeks), and `refresh`
        /// after a "load" replaces it with the richer entry until the
        /// file changes.
        #[test]
        fn failed_peek_is_degraded_and_uncached() {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path();
            std::fs::write(root.join("bogus.gguf"), b"not a gguf").unwrap();
            let catalog = catalog(root);

            let info = catalog.info("bogus.gguf").expect("listed");
            assert_eq!(info.id, "bogus.gguf");
            assert_eq!(info.display_name, "bogus.gguf");
            assert_eq!(info.max_input_tokens, 0);
            assert!(catalog.cache.lock().unwrap().is_empty());
            assert!(catalog.info("nope.gguf").is_none());
            assert!(catalog.info("../bogus.gguf").is_none());

            let rich: ModelInfo = Advertised {
                id: "bogus.gguf".into(),
                title: Some("Bogus".into()),
                n_ctx: 1024,
                n_ctx_train: 4096,
                image_input: false,
                thinking: true,
                modified: None,
            }
            .into();
            catalog.refresh("bogus.gguf", rich);
            let info = catalog.info("bogus.gguf").unwrap();
            assert_eq!(info.display_name, "Bogus");
            assert_eq!(info.max_input_tokens, 1024);
            assert_ne!(info.created_at, DateTime::<Utc>::UNIX_EPOCH);

            // The models() view carries the cached entry too.
            let models = catalog.models();
            assert_eq!(models.len(), 1);
            assert_eq!(models[0].display_name, "Bogus");

            // Replacing the file invalidates: back to the degraded entry.
            std::fs::write(
                root.join("bogus.gguf"),
                b"still not a gguf, longer",
            )
            .unwrap();
            let info = catalog.info("bogus.gguf").unwrap();
            assert_eq!(info.display_name, "bogus.gguf");

            // Refreshing an unlisted name is a no-op.
            catalog.refresh("nope.gguf", ModelInfo::new("nope", "nope"));
            assert!(catalog.info("nope.gguf").is_none());
        }

        /// The invariant the whole design rests on: a weightless peek
        /// and a real load advertise the same thing. Uses a context
        /// size that is a multiple of 256 so llama.cpp's padding cannot
        /// make the ceilings differ.
        #[test]
        #[ignore = "long running; requires models/model.gguf"]
        fn peek_agrees_with_load() {
            use crate::LlamaCppSession;
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
            let options = LlamaCppOptions::default().with_n_ctx(2048);
            let catalog: Catalog<LlamaCppBackend> =
                Catalog::new(&root, options);
            assert!(catalog.list().unwrap().contains(&"model.gguf".into()));

            let started = std::time::Instant::now();
            let peeked = catalog.info("model.gguf").expect("listed");
            eprintln!("peek: {:?}", started.elapsed());
            assert_eq!(peeked.id, "model.gguf");
            assert!(!peeked.display_name.is_empty());
            assert!(peeked.max_input_tokens > 0);
            assert!(peeked.max_input_tokens <= 2048);
            assert_ne!(peeked.created_at, DateTime::<Utc>::UNIX_EPOCH);
            // Second read is the cached one (`ModelInfo` isn't
            // `PartialEq`; compare the fields that could differ).
            let again = catalog.info("model.gguf").unwrap();
            assert_eq!(again.display_name, peeked.display_name);
            assert_eq!(again.max_input_tokens, peeked.max_input_tokens);
            assert_eq!(again.created_at, peeked.created_at);

            let session = LlamaCppSession::from_path_with(
                root.join("model.gguf"),
                *catalog.options(),
            )
            .unwrap()
            .quiet();
            let loaded = session.model_info();
            assert_eq!(loaded.id, peeked.id);
            assert_eq!(loaded.display_name, peeked.display_name);
            assert_eq!(loaded.max_input_tokens, peeked.max_input_tokens);
            assert_eq!(loaded.max_tokens, peeked.max_tokens);
            assert_eq!(loaded.capabilities, peeked.capabilities);

            catalog.refresh("model.gguf", loaded);
            let refreshed = catalog.info("model.gguf").unwrap();
            assert_eq!(refreshed.created_at, peeked.created_at);
            assert_eq!(refreshed.capabilities, peeked.capabilities);
        }
    }
}
