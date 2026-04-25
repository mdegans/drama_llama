//! moeflux-backed [`Model`]: tokenizer, vocab introspection, and
//! metadata from the MLX artifacts directory.
//!
//! moeflux's C side only consumes token IDs; tokenization lives
//! Rust-side via the HuggingFace [`tokenizers`] crate. This module
//! loads the standard HF tokenizer artifacts (`tokenizer.json`,
//! `tokenizer_config.json`, `config.json`, `chat_template.jinja`) from
//! the same directory an MLX export produces — the directory
//! `extract_weights.py` reads from.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::Value as JsonValue;
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::{backend::Model, Token};

/// Errors constructing a [`MoefluxModel`].
#[derive(Debug, Error)]
pub enum MoefluxModelError {
    /// A required artifact was missing from `mlx_dir`.
    #[error("missing artifact {0}")]
    MissingArtifact(&'static str),
    /// `tokenizers` failed to parse `tokenizer.json`.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// `tokenizer_config.json` or `config.json` was not valid JSON.
    #[error("config parse: {0}")]
    Config(#[from] serde_json::Error),
    /// I/O error reading an artifact.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// moeflux-backed model. Owns the HF tokenizer and a cached
/// `serde_json::Value` for each of `config.json` and
/// `tokenizer_config.json` so metadata lookups don't re-parse.
///
/// Constructed via [`Self::from_mlx_dir`]. The MLX artifacts
/// directory is the canonical "HF export" form — the same directory
/// `tools/extract_weights.py` consumes when producing
/// `model_weights.bin` for [`crate::moeflux::MoefluxDecoder`].
#[derive(Debug)]
pub struct MoefluxModel {
    tokenizer: Tokenizer,
    /// Parsed `config.json` — architecture metadata (vocab_size,
    /// hidden_size, num_hidden_layers, max_position_embeddings, etc.).
    config: JsonValue,
    /// Full `chat_template.jinja` text if present, else falls back to
    /// `tokenizer_config.chat_template`.
    chat_template: Option<String>,
    /// Resolved EOS token id. Qwen3 sets `eos_token_id` as a two-element
    /// array in `config.json` (`[<|im_end|>, <|endoftext|>]`); we pick
    /// the first (chat-EOS). For single-EOS models this degenerates.
    eos: Token,
    /// Resolved BOS token id, or -1 if the model has no BOS (Qwen).
    bos: Token,
    /// Resolved end-of-turn token id. For Qwen3-family models this
    /// equals EOS (`<|im_end|>` doubles as eot).
    eot: Token,
    /// Lazy maximum decoded piece length across the vocabulary.
    /// Populated on first call to [`Self::max_token_len`].
    max_token_len: OnceLock<usize>,
    /// Basename of the directory the model was loaded from (e.g.
    /// `qwen3-6-35b-a3b-mlx-4bit`). Used by [`Model::display_name`]
    /// for human-readable identification in API responses.
    name: Option<String>,
}

unsafe impl Send for MoefluxModel {}
// `tokenizers::Tokenizer` is `Send + Sync`; `JsonValue` and `String`
// are trivially so. Matches LlamaCppModel's Sync.
unsafe impl Sync for MoefluxModel {}

impl MoefluxModel {
    /// Load a model from the MLX export directory.
    ///
    /// The directory must contain:
    /// - `tokenizer.json` (tokenizers-crate format)
    /// - `tokenizer_config.json` (special-token strings)
    /// - `config.json` (architecture + eos/bos token ids)
    ///
    /// Optional:
    /// - `chat_template.jinja` — preferred chat template source.
    ///   Falls back to `tokenizer_config.chat_template` if absent.
    pub fn from_mlx_dir(
        mlx_dir: &Path,
    ) -> Result<Self, MoefluxModelError> {
        let tokenizer_path = mlx_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(MoefluxModelError::MissingArtifact("tokenizer.json"));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| MoefluxModelError::Tokenizer(e.to_string()))?;

        let config = read_json(&mlx_dir.join("config.json"))
            .map_err(|_| MoefluxModelError::MissingArtifact("config.json"))?;
        let tokenizer_config =
            read_json(&mlx_dir.join("tokenizer_config.json")).map_err(
                |_| MoefluxModelError::MissingArtifact("tokenizer_config.json"),
            )?;

        let chat_template =
            read_optional_text(&mlx_dir.join("chat_template.jinja"))
                .or_else(|| {
                    tokenizer_config
                        .get("chat_template")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                });

        let eos = resolve_eos(&config, &tokenizer_config, &tokenizer);
        let bos = resolve_bos(&config, &tokenizer_config, &tokenizer);
        let eot = eos;

        let name = mlx_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned());

        Ok(Self {
            tokenizer,
            config,
            chat_template,
            eos,
            bos,
            eot,
            max_token_len: OnceLock::new(),
            name,
        })
    }

    /// Borrow the underlying `tokenizers::Tokenizer`. Useful for
    /// batch-encoding helpers that don't exist on the trait.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Directory-less access to the parsed `config.json`.
    pub fn config(&self) -> &JsonValue {
        &self.config
    }

    /// Override the name returned by [`Model::display_name`]. By
    /// default `from_mlx_dir` captures the MLX directory's basename
    /// (e.g. `Qwen3.5-397B-A17B-4bit`). Callers using the parent-dir
    /// convention via [`crate::MoefluxEngine::from_path`] override
    /// to the parent's basename so server discovery dirs (where
    /// `mlx/` is just a fixed sub-name) round-trip cleanly through
    /// the API's `model` field.
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }
}

/// Tokenize via the HF pipeline. Errors from the tokenizer indicate
/// malformed input or pretokenizer bugs; we panic rather than
/// propagate because `Model::tokenize` is infallible and callers
/// treat tokenization as total.
fn encode_to_tokens(
    tokenizer: &Tokenizer,
    input: &str,
    add_special_tokens: bool,
) -> Vec<Token> {
    let encoding = tokenizer
        .encode(input, add_special_tokens)
        .expect("tokenizer::encode failed — malformed input?");
    encoding
        .get_ids()
        .iter()
        .map(|&id| id as Token)
        .collect()
}

impl Model for MoefluxModel {
    type Error = std::convert::Infallible;

    fn n_vocab(&self) -> i32 {
        // Prefer config.vocab_size — that's the lm_head dimension and
        // the size moeflux's logit buffer arrives in. Some HF configs
        // (Qwen3 multimodal) nest it under `text_config`; check both.
        // Falling back to the tokenizer's size is wrong: HF tokenizers
        // can report fewer tokens than the embedding matrix has rows
        // (the tail are unused/padding ids), so sampling against a
        // tokenizer-sized buffer would truncate real logits.
        config_i64(&self.config, "vocab_size")
            .or_else(|| {
                self.config
                    .get("text_config")
                    .and_then(|tc| config_i64(tc, "vocab_size"))
            })
            .map(|v| v as i32)
            .unwrap_or_else(|| {
                self.tokenizer.get_vocab_size(true) as i32
            })
    }

    fn bos(&self) -> Token {
        self.bos
    }

    fn eos(&self) -> Token {
        self.eos
    }

    fn eot(&self) -> Token {
        self.eot
    }

    fn special_tokens(&self) -> Vec<Token> {
        // HF tokenizers expose added tokens (both control and
        // user-defined specials) via the added-vocab decoder map.
        self.tokenizer
            .get_added_tokens_decoder()
            .keys()
            .map(|&id| id as Token)
            .collect()
    }

    fn max_token_len(&self) -> usize {
        *self.max_token_len.get_or_init(|| {
            let n = self.n_vocab();
            let mut max_len = 0usize;
            for id in 0..n {
                let Some(piece) = self.tokenizer.id_to_token(id as u32)
                else {
                    continue;
                };
                max_len = max_len.max(piece.len());
            }
            max_len
        })
    }

    fn tokenize(&self, input: &str, special: bool) -> Vec<Token> {
        encode_to_tokens(&self.tokenizer, input, special)
    }

    fn token_to_piece(&self, token: Token) -> String {
        if token < 0 {
            return String::new();
        }
        self.tokenizer
            .decode(&[token as u32], false)
            .unwrap_or_default()
    }

    fn token_to_piece_ref(&self, token: Token, buf: &mut Vec<u8>) {
        buf.clear();
        if token < 0 {
            return;
        }
        if let Ok(s) = self.tokenizer.decode(&[token as u32], false) {
            buf.extend_from_slice(s.as_bytes());
        }
    }

    fn context_size(&self) -> i32 {
        config_i64(&self.config, "max_position_embeddings")
            .or_else(|| {
                self.config
                    .get("text_config")
                    .and_then(|tc| config_i64(tc, "max_position_embeddings"))
            })
            .map(|v| v as i32)
            .unwrap_or(0)
    }

    fn chat_template_source(&self) -> Option<String> {
        self.chat_template.clone()
    }

    fn get_meta(&self, key: &str) -> Option<String> {
        // Look up dotted paths through `config.json`. `model_type`,
        // `vocab_size`, `hidden_size` etc. resolve at the top level;
        // nested keys (`quantization.bits`) walk through. Scalar
        // values are stringified; non-scalar values are JSON-encoded.
        let mut current = &self.config;
        for part in key.split('.') {
            current = current.get(part)?;
        }
        Some(match current {
            JsonValue::String(s) => s.clone(),
            JsonValue::Number(n) => n.to_string(),
            JsonValue::Bool(b) => b.to_string(),
            JsonValue::Null => "null".to_string(),
            other => other.to_string(),
        })
    }

    fn display_name(&self) -> Option<String> {
        self.name.clone()
    }
}

/// Parse `eos_token_id` from config.json. Qwen3 emits an array
/// (`[<|im_end|>, <|endoftext|>]`); single-EOS models emit a scalar.
/// Falls back to looking up the `eos_token` string from
/// tokenizer_config.json against the tokenizer's added vocab.
fn resolve_eos(
    config: &JsonValue,
    tokenizer_config: &JsonValue,
    tokenizer: &Tokenizer,
) -> Token {
    if let Some(val) = config.get("eos_token_id") {
        match val {
            JsonValue::Number(n) => {
                if let Some(id) = n.as_i64() {
                    return id as Token;
                }
            }
            JsonValue::Array(arr) => {
                if let Some(first) =
                    arr.iter().find_map(|v| v.as_i64())
                {
                    return first as Token;
                }
            }
            _ => {}
        }
    }
    if let Some(tok) = tokenizer_config
        .get("eos_token")
        .and_then(|v| v.as_str())
    {
        if let Some(id) = tokenizer.token_to_id(tok) {
            return id as Token;
        }
    }
    -1
}

/// Parse `bos_token_id` from config.json or look up the `bos_token`
/// string against the tokenizer. Returns `-1` if the model has no BOS
/// (Qwen3 sets `bos_token: null`).
fn resolve_bos(
    config: &JsonValue,
    tokenizer_config: &JsonValue,
    tokenizer: &Tokenizer,
) -> Token {
    if let Some(n) = config.get("bos_token_id").and_then(|v| v.as_i64()) {
        return n as Token;
    }
    if let Some(tok) = tokenizer_config
        .get("bos_token")
        .and_then(|v| v.as_str())
    {
        if let Some(id) = tokenizer.token_to_id(tok) {
            return id as Token;
        }
    }
    -1
}

fn read_json(path: &Path) -> Result<JsonValue, MoefluxModelError> {
    let bytes = std::fs::read(path)?;
    let value = serde_json::from_slice(&bytes)?;
    Ok(value)
}

fn read_optional_text(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn config_i64(value: &JsonValue, key: &str) -> Option<i64> {
    value.get(key).and_then(|v| v.as_i64())
}

/// Convenience: locate the standard MLX export directory for a model
/// name below a root. Example: `mlx_dir(&root, "qwen3-6-35b-a3b")`
/// returns `<root>/qwen3-6-35b-a3b-mlx-4bit`. Useful in tests that
/// locate artifacts from an env var.
pub fn mlx_dir(root: &Path, model_stem: &str) -> PathBuf {
    root.join(format!("{model_stem}-mlx-4bit"))
}
