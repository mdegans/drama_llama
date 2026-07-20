use derive_more::From;
use llama_cpp_sys_3::{
    llama_model, llama_model_default_params, llama_model_desc,
    llama_model_free, llama_model_get_vocab, llama_model_load_from_file,
    llama_model_meta_count, llama_model_meta_key_by_index,
    llama_model_meta_val_str, llama_model_meta_val_str_by_index,
    llama_model_n_ctx_train, llama_model_n_embd, llama_model_n_params,
    llama_model_params, llama_model_quantize,
    llama_model_quantize_default_params, llama_model_quantize_params,
    llama_model_rope_freq_scale_train, llama_model_rope_type, llama_model_size,
    llama_token, llama_token_attr_LLAMA_TOKEN_ATTR_CONTROL,
    llama_token_attr_LLAMA_TOKEN_ATTR_USER_DEFINED, llama_token_get_attr,
    llama_token_to_piece, llama_tokenize, llama_vocab, llama_vocab_bos,
    llama_vocab_eos, llama_vocab_eot, llama_vocab_fim_mid, llama_vocab_fim_pre,
    llama_vocab_fim_suf, llama_vocab_get_add_bos, llama_vocab_get_add_eos,
    llama_vocab_get_score, llama_vocab_get_text, llama_vocab_is_eog,
    llama_vocab_n_tokens, llama_vocab_nl, llama_vocab_type,
};
use std::{
    collections::BTreeMap,
    ffi::{c_char, CStr, CString, OsStr, OsString},
    num::{NonZeroU32, NonZeroUsize},
    path::PathBuf,
};

/// Convert a token to it's string representation.
///
/// Adapted from `llama.cpp/common/common.cpp`
///
/// # Panics
/// * If the token's piece is not valid UTF-8.
fn token_to_piece(token: llama_token, model: &LlamaCppModel) -> String {
    // Out-of-range ids (notably LLAMA_TOKEN_NULL = -1, returned by
    // vocab accessors like `eot()` when the model lacks the token)
    // make llama.cpp throw a C++ exception, which aborts the process
    // at the FFI boundary. Guard here so every piece-conversion
    // caller is covered. Observed with gemma-4 (no EOT token).
    if token < 0 || token >= model.n_vocab() {
        return format!("[invalid token {token}]");
    }
    let mut buf = vec![0; 8];
    token_to_piece_ref(token, model, &mut buf);

    String::from_utf8(buf).unwrap_or("[Invalid UTF-8]".to_string())
}

/// Same as `token_to_piece`, but allows reusable buffers.
fn token_to_piece_ref(
    token: llama_token,
    model: &LlamaCppModel,
    buf: &mut Vec<u8>,
) {
    // Safety: If the Vec isn't big enough, the function will return the number
    // of bytes needed. We resize the buffer and call the function again. There
    // is no risk of a buffer overflow.
    let n_tokens = unsafe {
        llama_token_to_piece(
            model.vocab,
            token,
            buf.as_mut_ptr() as *mut i8,
            buf.len().try_into().unwrap(),
            0,
            true,
        )
    };

    if n_tokens < 0 {
        buf.resize(-n_tokens as usize, 0);

        let check = unsafe {
            llama_token_to_piece(
                model.vocab,
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len().try_into().unwrap(),
                0,
                true,
            )
        };

        assert_eq!(check, -n_tokens);
    } else {
        buf.resize(n_tokens as usize, 0);
    }
}

/// Convert a PathBuf to a CString
fn path_to_cstring(path: PathBuf) -> CString {
    let os_str: &std::ffi::OsStr = path.as_os_str();
    CString::new(os_str.as_encoded_bytes()).unwrap()
}

/// Quantize a Llama model.
pub fn llama_quantize(
    in_file: PathBuf,
    out_file: PathBuf,
    params: Option<llama_model_quantize_params>,
) -> Result<(), NonZeroU32> {
    let in_file = path_to_cstring(in_file);
    let out_file = path_to_cstring(out_file);

    // Safety: this returns POD
    let params =
        params.unwrap_or(unsafe { llama_model_quantize_default_params() });

    // Safety: The input and output files are null-terminated strings.
    let code = unsafe {
        llama_model_quantize(
            in_file.as_c_str().as_ptr(),
            out_file.as_c_str().as_ptr(),
            &params,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(code.try_into().unwrap())
    }
}

/// An ergonomic wrapper for a `llama.cpp` model.
#[derive(Debug)]
pub struct LlamaCppModel {
    pub(crate) inner: *mut llama_model,
    /// The vocabulary associated with this model. This pointer is valid for the
    /// lifetime of the model and does not need to be freed separately.
    pub(crate) vocab: *const llama_vocab,
    /// LlamaCppModel basename
    pub(crate) file_name: Option<OsString>,
    /// Cached longest-token length. The predictor calls this per generated
    /// token (stop-string windowing), and the naive compute is O(vocab), so
    /// we memoize the first call.
    max_token_len: std::sync::OnceLock<usize>,
    /// Cached end-of-generation token set (libllama's
    /// `special_eog_ids`). The sampler consults this per candidate
    /// sweep and the naive compute is O(vocab), so memoize.
    eog: std::sync::OnceLock<Vec<llama_token>>,
}

unsafe impl Send for LlamaCppModel {}

// SAFETY:
// `LlamaCppModel` holds a raw `*mut llama_model`. llama.cpp's model data is
// immutable after `llama_load_model_from_file` returns — weights,
// vocab tables, and metadata are allocated once and read-only
// thereafter. All `&LlamaCppModel` call sites in this crate only invoke
// vocab-read APIs (`llama_token_to_piece`, `llama_n_vocab`, metadata
// getters, etc.), which llama.cpp documents as safe to call
// concurrently against a `const llama_model*`. Mutating operations
// (tokenization of *batches* into a context, decoding) target the
// separately-owned `*mut llama_context`, not the model.
//
// The borrow checker ensures nothing drops or re-assigns the model
// while `&LlamaCppModel` references are alive, so the "immutable after load"
// invariant holds for every shared reference. Adding any method that
// mutates llama_model state would invalidate this impl.
unsafe impl Sync for LlamaCppModel {}

#[derive(Debug, From)]
pub enum MetaKey<'a> {
    Int(i32),
    String(&'a str),
}

impl LlamaCppModel {
    /// Load a model from a file.
    pub fn from_file(
        path: PathBuf,
        params: Option<llama_model_params>,
    ) -> Option<Self> {
        if path.file_name().is_some_and(|fname| {
            fname
                .to_string_lossy()
                .to_lowercase()
                .contains("uncensored")
        }) {
            // This is a naive check, but will ensure that the user is aware of
            // the TOS and the prohibition on racists, bigots, and other
            // unsavory content. Smut is just fine. Meta fed in erotic fiction
            // for a reason. Eric Hartford's models are terrible for that.
            eprintln!("Eric Hartford's `Uncensored` models are not supported. Read the TOS. If you want smut, use the foundation models and an n-shot prompt. Example: https://huggingface.co/NousResearch/Meta-Llama-3-70B-GGUF/");
            return None;
        }

        let file_name = path.file_name()?.to_owned();
        let path = path_to_cstring(path);
        // Safety: What's returned is POD
        let params = params.unwrap_or(unsafe { llama_model_default_params() });
        // Safety: The model is owned by the caller. We free it in the `Drop`
        // implementation. The path is a null-terminated string.
        let model = unsafe {
            llama_model_load_from_file(path.as_c_str().as_ptr(), params)
        };

        if model.is_null() {
            None
        } else {
            let vocab = unsafe { llama_model_get_vocab(model) };
            Some(Self {
                inner: model,
                vocab,
                file_name: Some(file_name),
                max_token_len: std::sync::OnceLock::new(),
                eog: std::sync::OnceLock::new(),
            })
        }
    }

    /// Create a new model from a raw pointer. It will return `None` if the
    /// pointer is null.
    ///
    /// # Safety
    /// This will take ownership of the pointer and free it when the model is
    /// dropped.
    pub unsafe fn from_raw(ptr: *mut llama_model) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            let vocab = llama_model_get_vocab(ptr);
            Some(Self {
                inner: ptr,
                vocab,
                file_name: None,
                max_token_len: std::sync::OnceLock::new(),
                eog: std::sync::OnceLock::new(),
            })
        }
    }

    /// Return the base filename if the `LlamaCppModel` was loaded [`from_file`]
    ///
    /// [`from_file`]: LlamaCppModel::from_file
    pub fn file_name<'a>(&'a self) -> Option<&'a OsStr> {
        self.file_name.as_deref()
    }

    /// Unwrap the model and return the raw pointer.
    ///
    /// # Safety
    /// The caller is responsible for freeing the model using `llama_free_model`
    /// or `LlamaCppModel::from_raw` and then dropping it.
    pub fn into_raw(self) -> *mut llama_model {
        let ptr = self.inner;
        std::mem::forget(self);
        ptr
    }

    /// Return the inner model.
    pub fn as_ptr(&self) -> *const llama_model {
        debug_assert_eq!(self.inner.is_null(), false);
        self.inner as *const llama_model
    }

    /// Return the inner model mutably.
    pub fn as_ptr_mut(&mut self) -> *mut llama_model {
        debug_assert_eq!(self.inner.is_null(), false);
        self.inner
    }

    // Safety: The getters that follow are safe because they are simple accessor
    // methods that return POD.

    /// Return the Beginning of Sequence (BOS) token.
    pub fn bos(&self) -> llama_token {
        unsafe { llama_vocab_bos(self.vocab) }
    }

    /// Return the End of Sequence (EOS) token.
    pub fn eos(&self) -> llama_token {
        unsafe { llama_vocab_eos(self.vocab) }
    }

    /// Return the next-line token.
    pub fn next_line(&self) -> llama_token {
        unsafe { llama_vocab_nl(self.vocab) }
    }

    /// Return the infill prefix token.
    pub fn infill_prefix(&self) -> llama_token {
        unsafe { llama_vocab_fim_pre(self.vocab) }
    }

    /// Return the infill middle token.
    pub fn infill_middle(&self) -> llama_token {
        unsafe { llama_vocab_fim_mid(self.vocab) }
    }

    /// Return the end of turn token.
    pub fn eot(&self) -> llama_token {
        unsafe { llama_vocab_eot(self.vocab) }
    }

    /// Return the infill suffix token.
    pub fn infill_suffix(&self) -> llama_token {
        unsafe { llama_vocab_fim_suf(self.vocab) }
    }

    /// Collect every token with the `CONTROL` or `USER_DEFINED` attribute
    /// set — i.e. the full set of chat-template and structural specials
    /// (`<|eot_id|>`, `<|start_header_id|>`, `<|python_tag|>`, `<|eom_id|>`,
    /// BOS/EOS/EOT, tool-call markers, …). Callers can feed this into
    /// [`RepetitionOptions::extend_ignored`] so a repetition penalty never
    /// suppresses the tokens the template needs to close a turn.
    ///
    /// O(vocab_size); intended to be called once at session setup, not
    /// per-token.
    ///
    /// [`RepetitionOptions::extend_ignored`]: crate::RepetitionOptions::extend_ignored
    pub fn special_tokens(&self) -> Vec<llama_token> {
        let n = self.n_vocab();
        let mask = llama_token_attr_LLAMA_TOKEN_ATTR_CONTROL
            | llama_token_attr_LLAMA_TOKEN_ATTR_USER_DEFINED;
        (0..n)
            .filter(|&t| {
                let attr = unsafe { llama_token_get_attr(self.vocab, t) };
                (attr & mask) != 0
            })
            .collect()
    }

    /// libllama's `special_eog_ids` set, verbatim: every token for
    /// which `llama_vocab_is_eog` holds. This is the set libllama's
    /// own generation loop stops on, so it is the one we stop on.
    ///
    /// libllama fills it by token-text matching (`<|im_end|>`,
    /// `<|return|>`, `<|call|>`, …), then applies model-aware fixups.
    /// Those fixups are exactly why this must be read rather than
    /// derived from [`eos`](Self::eos) / [`eot`](Self::eot): for
    /// Harmony vocabs it *removes* `<|end|>` from the set so the
    /// in-stream channel separator stays generatable — while leaving
    /// `special_eot_id` pointing at `<|end|>` (llama-vocab.cpp: EOT is
    /// auto-detected by text, and `"<|end|>"` is on that list). So
    /// gpt-oss has an EOT that does not end generation, and only
    /// `<|return|>` (final), `<|call|>` (tool call) and
    /// `<|endoftext|>` do.
    ///
    /// Memoized — first call is O(vocab), subsequent calls O(1).
    pub fn eog_tokens(&self) -> Vec<llama_token> {
        self.eog
            .get_or_init(|| {
                (0..self.n_vocab())
                    .filter(|&t| unsafe { llama_vocab_is_eog(self.vocab, t) })
                    .collect()
            })
            .clone()
    }

    /// Longest token length in this model's vocabulary. Memoized — first
    /// call is O(k) where k is the vocab size; subsequent calls are O(1).
    /// The predictor hits this per generated token for stop-string
    /// windowing, so the cache matters.
    pub fn max_token_len(&self) -> usize {
        *self.max_token_len.get_or_init(|| {
            let mut max_len = 0;
            for i in 0..self.n_vocab() {
                max_len = max_len.max(self.token_to_text(i).len());
            }
            max_len
        })
    }

    /// Return whether BOS token should be added.
    pub fn add_bos(&self) -> bool {
        unsafe { llama_vocab_get_add_bos(self.vocab) }
    }

    /// Return whether the EOS token should be added.
    pub fn add_eos(&self) -> bool {
        unsafe { llama_vocab_get_add_eos(self.vocab) }
    }

    /// Vocab type.
    pub fn vocab_type(&self) -> llama_vocab_type {
        unsafe { llama_vocab_type(self.vocab) }
    }

    /// Vocab size.
    pub fn n_vocab(&self) -> i32 {
        unsafe { llama_vocab_n_tokens(self.vocab) }
    }

    /// Context size the model was trained with.
    pub fn context_size(&self) -> i32 {
        unsafe { llama_model_n_ctx_train(self.inner) }
    }

    /// Embedding size.
    pub fn embedding_size(&self) -> i32 {
        unsafe { llama_model_n_embd(self.inner) }
    }

    /// Rotary Position Encoding (RoPE) type.
    pub fn rope_type(&self) -> i32 {
        unsafe { llama_model_rope_type(self.inner) }
    }

    /// RoPE frequency scaling factor.
    pub fn rope_freq_scale(&self) -> f32 {
        unsafe { llama_model_rope_freq_scale_train(self.inner) }
    }

    /// Get the number of metadata entries.
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_model_meta_count(self.inner) }
    }

    /// The total size of all the tensors in the model in bytes.
    pub fn size(&self) -> u64 {
        unsafe { llama_model_size(self.inner) }
    }

    /// The total number of parameters in the model.
    pub fn n_params(&self) -> u64 {
        unsafe { llama_model_n_params(self.inner) }
    }

    /// A string describing the model type.
    pub fn desc(&self) -> String {
        let mut buf: Vec<u8> = vec![0; 8];
        // Safety: The buffer is properly aligned and has the correct length.
        // The string will be null-terminated.
        let required = unsafe {
            llama_model_desc(
                self.inner,
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            )
        } + 1;

        if required < 1 {
            panic!("Error reading model description.");
        } else {
            if required as usize > buf.len() {
                buf.resize(required as usize, 0);
                let check = unsafe {
                    llama_model_desc(
                        self.inner,
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len(),
                    )
                };
                assert_eq!(required, check + 1);
                buf.truncate(required as usize - 1);
            } else {
                buf.resize(required as usize - 1, 0);
            }

            // This could fail if the model has junk in the description. It's
            // not a programmer error, so we'll just return an error string.
            return String::from_utf8(buf)
                .unwrap_or("[Invalid UTF-8]".to_string());
        }
    }

    /// The sampling settings this GGUF recommends for itself, read
    /// from `llama.cpp`'s typed `general.sampling.*` metadata
    /// namespace (`enum llama_model_meta_key` in `llama.h`). Absent
    /// keys stay `None`; a model that carries none of them yields
    /// [`SamplingParams::default`](crate::SamplingParams::default).
    ///
    /// # Why parsing
    ///
    /// Upstream exposes no typed value getter —
    /// `llama_model_meta_val_str` is the only accessor, and every
    /// value arrives pre-stringified through `std::to_string`
    /// (`"%f"`, six decimals). So `top_p` reads back as `"0.950000"`,
    /// `temp` as `"1.000000"`, `top_k` as `"20"`. The rounding is
    /// harmless and mildly helpful: it recovers the intended `0.95`
    /// from the `float32` `0.949999988` actually stored in the file.
    ///
    /// # Malformed values
    ///
    /// A key that is present but unparseable, or a `top_p`/`min_p`
    /// outside `0.0..=1.0`, is treated as absent rather than as an
    /// error. A junk metadata value must not stop a model from
    /// loading — the user can always override in the sidecar.
    ///
    /// # Not read
    ///
    /// `xtc_*` (no corresponding [`SamplingMode`](crate::SamplingMode)),
    /// `penalty_repeat` / `penalty_last_n` (this crate's repetition
    /// penalty is n-gram-based with windowed decay and has no honest
    /// scalar preimage), and `sequence` (defined upstream but never
    /// written — `llama-model-saver.cpp` still has it commented out).
    pub fn recommended_sampling(&self) -> crate::SamplingParams {
        // Parse a metadata value, treating both "absent" and
        // "present but junk" as None.
        let num = |key: &str| -> Option<f64> {
            LlamaCppModel::get_meta(self, key)?.trim().parse().ok()
        };
        let prob_f64 = |key: &str| crate::Probability::from_f(num(key)?).ok();
        let prob_f32 =
            |key: &str| crate::Probability::from_f(num(key)? as f32).ok();

        // Mirostat is terminal, so it is read as a unit: the mode
        // selector plus its two parameters, defaulting to llama.cpp's
        // own tau/eta when the model names an algorithm but not its
        // tuning.
        let mirostat = match num("general.sampling.mirostat") {
            Some(v) if v >= 1.0 => {
                let tau =
                    num("general.sampling.mirostat_tau").unwrap_or(5.0) as f32;
                let eta =
                    num("general.sampling.mirostat_eta").unwrap_or(0.1) as f32;
                if v >= 2.0 {
                    Some(crate::Mirostat::V2 { tau, eta })
                } else {
                    Some(crate::Mirostat::V1 { tau, eta })
                }
            }
            _ => None,
        };

        crate::SamplingParams {
            temp: num("general.sampling.temp").map(|t| t as f32),
            top_p: prob_f64("general.sampling.top_p"),
            top_k: num("general.sampling.top_k")
                .filter(|k| *k >= 1.0)
                .and_then(|k| NonZeroUsize::new(k as usize)),
            min_p: prob_f32("general.sampling.min_p"),
            mirostat,
        }
    }

    /// Get all metadata entries.
    ///
    /// Calling this is less efficient than calling `get_meta` for specific
    /// keys.
    pub fn meta(&self) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        for i in 0..self.meta_count() {
            // Safety: The buffer is properly aligned and has the correct
            // length.
            let key_str = unsafe {
                let mut buf: Vec<u8> = vec![0; 8];
                let required = llama_model_meta_key_by_index(
                    self.inner,
                    i,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len(),
                ) + 1;
                if required < 0 {
                    continue;
                }
                if buf.len() != required as usize {
                    buf.resize(required as usize, 0);
                    let check = llama_model_meta_key_by_index(
                        self.inner,
                        i,
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len(),
                    );
                    assert_eq!(required, check + 1);
                    buf.truncate(required as usize - 1);
                } else {
                    buf.resize(required as usize - 1, 0);
                }
                String::from_utf8(buf).unwrap_or("[Invalid UTF-8]".to_string())
            };

            if let Some(val) = self.get_meta(i) {
                map.insert(key_str, val);
            }
        }
        map
    }

    /// Get model metadata value by key (string or int).
    ///
    /// Returns `None` if the key is not found or if the value is invalid UTF-8.
    pub fn get_meta<'a, K>(&self, key: K) -> Option<String>
    where
        K: Into<MetaKey<'a>>,
    {
        self.get_meta_by_key(key.into())
    }

    fn get_meta_by_key(&self, key: MetaKey) -> Option<String> {
        let mut buf: Vec<u8> = vec![0; 8];

        match key {
            // Safety: The buffer is properly initialized, aligned, and has the
            // correct length.
            MetaKey::Int(i) => unsafe {
                let required = llama_model_meta_val_str_by_index(
                    self.inner,
                    i,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len(),
                ) + 1;

                if required < 1 {
                    return None;
                }

                if buf.len() != required as usize {
                    buf.resize(required as usize, 0);
                    let check = llama_model_meta_val_str_by_index(
                        self.inner,
                        i,
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len(),
                    );
                    assert_eq!(required, check + 1);
                    // Because the buffer is null-terminated, we need to remove
                    // the null terminator or it will be included in the string
                    // which will cause a panic.
                    buf.truncate(required as usize - 1);
                } else {
                    buf.resize(required as usize - 1, 0);
                }
            },
            MetaKey::String(s) => {
                let key = CString::new(s).unwrap();
                let required = unsafe {
                    llama_model_meta_val_str(
                        self.inner,
                        key.as_c_str().as_ptr(),
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len(),
                    )
                } + 1;

                if required < 1 {
                    return None;
                }
                if buf.len() != required as usize {
                    buf.resize(required as usize, 0);
                    let check = unsafe {
                        llama_model_meta_val_str(
                            self.inner,
                            key.as_c_str().as_ptr(),
                            buf.as_mut_ptr() as *mut c_char,
                            buf.len(),
                        )
                    };
                    assert_eq!(required, check + 1);
                    buf.truncate(required as usize - 1);
                } else {
                    buf.resize(required as usize - 1, 0);
                }
            }
        };

        String::from_utf8(buf).ok()
    }
    /// Tokenize a string into a Vec of tokens.
    pub fn tokenize(&self, input: &str, special: bool) -> Vec<llama_token> {
        let mut result =
            self.tokenize_raw_special(input, self.add_bos(), special);
        if self.add_eos() {
            result.push(self.eos());
        }
        result
    }

    /// Raw `llama_tokenize` with explicit `add_special` (controls the
    /// vocab-driven automatic BOS) and no EOS append. Building block
    /// for [`Self::tokenize`] and the trait's `tokenize_special` —
    /// the media segment path needs mid-stream pieces tokenized
    /// without leading specials.
    fn tokenize_raw_special(
        &self,
        input: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<llama_token> {
        // Adapted from `llama.cpp/common/common.cpp` which is not exposed to
        // the public API.

        // Guess a reasonable number of tokens to allocate. This is not
        // guaranteed to be enough, but it will probably be enough in most
        // cases.
        let mut n_tokens: i32 = (input.as_bytes().len()
            + if add_special { 1 } else { 0 })
        .try_into()
        .unwrap();
        n_tokens /= 3;

        let mut result = vec![0; n_tokens as usize];

        // Safety: The function has a length paramter, so null-termination is
        // not required. Input is valid UTF-8 and outlives the function call.
        // The result buffer is properly aligned and has the correct length It
        // should be large enough. If it's not, it's not a problem, because the
        // function will return the number of tokens needed.
        n_tokens = unsafe {
            llama_tokenize(
                self.vocab,
                input.as_bytes().as_ptr() as *const i8,
                input.len().try_into().unwrap(),
                result.as_mut_ptr(),
                result.len().try_into().unwrap(),
                add_special,
                parse_special,
            )
        };

        if n_tokens < 0 {
            // this shouldn't happen, because there should be enough space, but
            // if not, `-n_tokens` indicates the number of tokens that are
            // needed.
            result.resize(-n_tokens as usize, 0);
            // Safety: Same as above, but we double-check the length below.
            let check = unsafe {
                llama_tokenize(
                    self.vocab,
                    input.as_bytes().as_ptr() as *const i8,
                    input.len().try_into().unwrap(),
                    result.as_mut_ptr(),
                    result.len().try_into().unwrap(),
                    add_special,
                    parse_special,
                )
            };
            assert_eq!(check, -n_tokens);
        } else {
            result.resize(n_tokens as usize, 0);
        }

        result
    }

    /// Convert a single token to a piece.
    ///
    /// # Panics
    /// * If the token's piece is not valid UTF-8.
    pub fn token_to_piece(&self, token: llama_token) -> String {
        token_to_piece(token, &self)
    }

    /// Convert tokens to text.
    ///
    /// # Panics
    /// * If any token's piece is not valid UTF-8.
    pub fn tokens_to_pieces<'a, Ts>(
        &'a self,
        tokens: Ts,
    ) -> impl Iterator<Item = String> + 'a
    where
        Ts: IntoIterator<Item = llama_token> + 'a,
    {
        tokens.into_iter().map(|token| self.token_to_piece(token))
    }

    /// Convert tokens to a single string. Does not strip any prefix or suffix.
    ///
    /// # Panics
    /// * If any token's piece is not valid UTF-8.
    pub fn tokens_to_string<Ts>(&self, tokens: Ts) -> String
    where
        Ts: IntoIterator<Item = llama_token>,
    {
        self.tokens_to_pieces(tokens).collect()
    }

    /// Get text for a given token.
    ///
    /// This calls `llama_token_get_text`. It does not copy the underlying
    /// string, but whitespace is not converted.
    ///
    /// # Panics
    /// * If the token text is invalid UTF-8
    // It's unclear how this differs from `token_to_piece` other than returning
    // a c_str() ptr to the underlying c++ std::string
    pub fn token_to_text<'a>(&'a self, token: llama_token) -> &'a str {
        let ptr = unsafe { llama_vocab_get_text(self.vocab, token) };
        return unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
    }

    /// Convert tokens to text.
    ///
    /// # Panics
    /// * If any token's piece is not valid UTF-8.
    pub fn tokens_to_text<'a, Ts>(
        &'a self,
        tokens: Ts,
    ) -> impl Iterator<Item = &'a str> + 'a
    where
        Ts: IntoIterator<Item = &'a llama_token> + 'a,
    {
        tokens.into_iter().map(|&token| self.token_to_text(token))
    }

    /// Get score for a given token.
    // TODO: it's not very clear what score is retuned. More documentation is
    // required. It's not the logits or probabilities, because that's stored in
    // the context. It may be a multiplier for the token's probability. The
    // constructor's configuration includes overrides for KV pairs.
    pub fn token_to_score(&self, token: llama_token) -> f32 {
        unsafe { llama_vocab_get_score(self.vocab, token) }
    }

    /// Get scores for a given slice of tokens.
    pub fn tokens_to_scores<'a, Ts>(
        &'a self,
        tokens: Ts,
    ) -> impl Iterator<Item = f32> + 'a
    where
        Ts: IntoIterator<Item = &'a llama_token> + 'a,
    {
        tokens
            .into_iter()
            .map(move |&token| self.token_to_score(token))
    }
}

impl Drop for LlamaCppModel {
    fn drop(&mut self) {
        unsafe { llama_model_free(self.inner) };
    }
}

// Transitional pass-through impl of the backend-agnostic `Model`
// trait. Every method forwards directly to the inherent method of the
// same name. Commit 4 threads `<M: Model>` generics through Session
// and Predictor; until then this impl mainly exists so trait
// signatures can be exercised against the concrete llama.cpp model.
impl crate::backend::Model for LlamaCppModel {
    type Error = std::convert::Infallible;

    fn n_vocab(&self) -> i32 {
        LlamaCppModel::n_vocab(self)
    }

    fn bos(&self) -> crate::Token {
        LlamaCppModel::bos(self)
    }

    fn eos(&self) -> crate::Token {
        LlamaCppModel::eos(self)
    }

    fn eot(&self) -> crate::Token {
        LlamaCppModel::eot(self)
    }

    fn special_tokens(&self) -> Vec<crate::Token> {
        LlamaCppModel::special_tokens(self)
    }

    fn max_token_len(&self) -> usize {
        LlamaCppModel::max_token_len(self)
    }

    fn tokenize(&self, input: &str, special: bool) -> Vec<crate::Token> {
        LlamaCppModel::tokenize(self, input, special)
    }

    fn tokenize_special(
        &self,
        input: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Vec<crate::Token> {
        let mut result =
            self.tokenize_raw_special(input, add_special, parse_special);
        if add_special && self.add_eos() {
            result.push(LlamaCppModel::eos(self));
        }
        result
    }

    fn token_to_piece(&self, token: crate::Token) -> String {
        LlamaCppModel::token_to_piece(self, token)
    }

    fn token_to_piece_ref(&self, token: crate::Token, buf: &mut Vec<u8>) {
        token_to_piece_ref(token, self, buf)
    }

    fn context_size(&self) -> i32 {
        LlamaCppModel::context_size(self)
    }

    fn chat_template_source(&self) -> Option<String> {
        self.get_meta("tokenizer.chat_template")
    }

    fn recommended_sampling(&self) -> crate::SamplingParams {
        LlamaCppModel::recommended_sampling(self)
    }

    fn eog_tokens(&self) -> Vec<crate::Token> {
        LlamaCppModel::eog_tokens(self)
    }

    fn display_name(&self) -> Option<String> {
        self.file_name
            .as_deref()
            .map(|s| s.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_sys_3::llama_vocab_type_LLAMA_VOCAB_TYPE_BPE;

    use crate::{ChatTemplate, Prompt};

    use super::*;

    /// Load `models/model.gguf` with zero GPU layers. The metadata
    /// tests only inspect vocab / metadata — CPU-only loading keeps
    /// them VRAM-free so the default parallel test runner can't stack
    /// several full models onto one card (observed: two of the three
    /// failing with "unable to allocate CUDA0 buffer" on a 24 GB card
    /// under `--features cuda`).
    fn load_test_model_cpu() -> LlamaCppModel {
        use std::path::PathBuf;

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("models/model.gguf");
        let mut params =
            unsafe { llama_cpp_sys_3::llama_model_default_params() };
        params.n_gpu_layers = 0;
        LlamaCppModel::from_file(path, Some(params)).unwrap()
    }

    /// Smoke-test of model loading + introspection. Uses whatever
    /// `models/model.gguf` points to; assertions are model-agnostic so
    /// swapping the symlink (Llama → Qwen → Cogito) doesn't break the
    /// test. Specific architectures used to have hard-coded vocab /
    /// embedding / bos values — those belong in a model-specific
    /// fixture test, not here.
    #[test]
    fn test_model() {
        let model = load_test_model_cpu();

        // Architecture-agnostic invariants every loaded model must
        // satisfy. BPE vocab is the common case (Llama, Qwen, Cogito,
        // Mistral all use it); if a future test model is WordPiece /
        // SPM this assertion will need widening.
        assert_eq!(model.vocab_type(), llama_vocab_type_LLAMA_VOCAB_TYPE_BPE);
        assert!(model.n_vocab() > 0);
        assert!(model.context_size() > 0);
        assert!(model.embedding_size() > 0);
        assert_eq!(model.rope_freq_scale(), 1.0);
        assert!(!model.desc().is_empty());
        assert!(model.meta_count() > 0);
        assert!(model.get_meta("tokenizer.chat_template").is_some());
        assert!(model.size() > 8192 && model.size() < 200_000_000_000);
        assert!(
            model.n_params() > 1_000_000_000
                && model.n_params() < 1_000_000_000_000
        );

        // Tokenization round-trip.
        const EXPECTED: &str = "Hello, world!";
        let tokens = model.tokenize(EXPECTED, false);
        assert!(!tokens.is_empty());
        let text = model.tokens_to_string(tokens);
        assert!(text.ends_with(EXPECTED));

        // Per-token text roundtrip: at least one tokenization of
        // "Hello" must decode to a string containing "Hello" or
        // "hello" (sub-word splits are fine).
        let hello_tokens = model.tokenize("Hello", false);
        assert!(!hello_tokens.is_empty());
        let any_hello = hello_tokens.iter().any(|&t| {
            let text = model.token_to_text(t);
            text.contains("Hello") || text.contains("hello")
        });
        assert!(
            any_hello,
            "No token in the tokenized 'Hello' contains the expected text"
        );

        // Jinja chat-template render. Every chat-tuned model must be
        // able to echo the user/assistant turns we feed it.
        let prompt = Prompt::default()
            .system("A conversation between a user and an assistant.")
            .add_message((crate::Role::User, "Hello, world!"))
            .unwrap()
            .add_message((crate::Role::Assistant, "Hi!"))
            .unwrap()
            .add_message((crate::Role::User, "So, how's it going?"))
            .unwrap();

        let tmpl = ChatTemplate::from_model(&model).unwrap();
        let result = tmpl.render(&prompt, true).unwrap();
        assert!(result.contains("Hello, world!"));
        assert!(result.contains("Hi!"));
        assert!(result.contains("how's it going"));
    }

    #[test]
    fn test_metadata() {
        let model = load_test_model_cpu();
        let meta = model.meta();
        for (key, val) in meta.iter() {
            println!("{}: {}", key, val);
        }
        println!("{:#?}", meta);
    }

    /// The model's own `general.sampling.*` recommendation is read
    /// and parsed. This is the check whose absence let #35 sit: the
    /// values were sitting in the GGUF the whole time, unread.
    ///
    /// Asserted loosely on purpose — `models/model.gguf` is a symlink
    /// the developer repoints, so this pins the *contract* (keys are
    /// found, parse, and land in range) rather than one model's
    /// numbers. The exact Qwen3.6 triple (20 / 0.95 / 1.0) is covered
    /// by the seeding check in the sidecar tests.
    #[test]
    fn test_recommended_sampling() {
        let model = load_test_model_cpu();
        let params = model.recommended_sampling();
        // Printed like `test_metadata` does: when the symlink is
        // repointed, this is how you see what the new model asks for.
        println!("recommended_sampling: {params:#?}");

        // Whatever the symlink points at, a model that advertises a
        // key must round-trip it into range. A model that advertises
        // none yields the empty params — also valid (gpt-oss).
        if let Some(t) = params.temp {
            assert!(t.is_finite() && t >= 0.0, "temp out of range: {t}");
        }
        if let Some(p) = params.top_p {
            let p = p.into_f();
            assert!((0.0..=1.0).contains(&p), "top_p out of range: {p}");
        }

        // Guard the stringified-float parse specifically: llama.cpp
        // hands back `"0.950000"` (std::to_string, "%f"), and a naive
        // integer parse of top_k would silently yield None. If the
        // model advertises top_k at all, it must have parsed.
        if LlamaCppModel::get_meta(&model, "general.sampling.top_k").is_some() {
            assert!(
                params.top_k.is_some(),
                "model advertises general.sampling.top_k but it did not parse"
            );
        }
        if LlamaCppModel::get_meta(&model, "general.sampling.top_p").is_some() {
            assert!(
                params.top_p.is_some(),
                "model advertises general.sampling.top_p but it did not parse"
            );
        }
    }

    /// Cross-model check of the `general.sampling.*` reader against
    /// real files, including the **absent** case — a model that
    /// advertises nothing must yield empty params so the sidecar
    /// falls back to `SamplerConfig::default()` rather than to a
    /// half-populated chain.
    ///
    /// `#[ignore]`d because it needs specific models beyond the
    /// `model.gguf` symlink; each is skipped individually if missing,
    /// so it stays useful on a partial model dir.
    #[test]
    #[ignore = "long running, requires the full models/ dir"]
    fn test_recommended_sampling_across_models() {
        use std::path::PathBuf;

        // (file, expected top_k, expected top_p, expected temp)
        let cases: &[(&str, Option<usize>, Option<f64>, Option<f32>)] = &[
            (
                "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
                Some(20),
                Some(0.95),
                Some(1.0),
            ),
            (
                "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
                Some(64),
                Some(0.95),
                Some(1.0),
            ),
            // Advertises no sampling metadata at all.
            ("gpt-oss-20b-UD-Q8_K_XL.gguf", None, None, None),
        ];

        for (file, want_k, want_p, want_t) in cases {
            let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.push("models");
            path.push(file);
            if !path.is_file() {
                eprintln!("skipping absent model {file}");
                continue;
            }
            let mut params =
                unsafe { llama_cpp_sys_3::llama_model_default_params() };
            params.n_gpu_layers = 0;
            let model = LlamaCppModel::from_file(path, Some(params))
                .unwrap_or_else(|| panic!("load {file}"));

            let got = model.recommended_sampling();
            assert_eq!(got.top_k.map(|k| k.get()), *want_k, "{file} top_k");
            assert_eq!(got.top_p.map(|p| p.into_f()), *want_p, "{file} top_p");
            assert_eq!(got.temp, *want_t, "{file} temp");

            // The empty case must be *fully* empty — a model with no
            // recommendation seeds the crate default, and `is_empty`
            // is what that decision keys off.
            if want_k.is_none() && want_p.is_none() && want_t.is_none() {
                assert!(got.is_empty(), "{file} should recommend nothing");
            }
        }
    }

    #[test]
    fn test_model_desc() {
        let model = load_test_model_cpu();
        let desc = model.desc();
        assert!(!desc.is_empty());
        assert!(!desc.ends_with("\0"));
    }

    #[test]
    fn test_path_to_cstring() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cs = path_to_cstring(path.clone());
        assert_eq!(path.display().to_string(), cs.to_str().unwrap().to_owned());
    }
}
