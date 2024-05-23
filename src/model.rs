use derive_more::From;
use llama_cpp_sys_3::{
    ggml_tensor, llama_add_bos_token, llama_add_eos_token,
    llama_chat_apply_template, llama_chat_message, llama_free_model,
    llama_get_model_tensor, llama_load_model_from_file, llama_model,
    llama_model_apply_lora_from_file, llama_model_default_params,
    llama_model_desc, llama_model_meta_count, llama_model_meta_key_by_index,
    llama_model_meta_val_str, llama_model_meta_val_str_by_index,
    llama_model_n_params, llama_model_params, llama_model_quantize,
    llama_model_quantize_default_params, llama_model_quantize_params,
    llama_model_size, llama_n_ctx_train, llama_n_embd, llama_n_vocab,
    llama_rope_freq_scale_train, llama_rope_type, llama_token, llama_token_bos,
    llama_token_eos, llama_token_eot, llama_token_get_score,
    llama_token_get_text, llama_token_middle, llama_token_nl,
    llama_token_prefix, llama_token_suffix, llama_token_to_piece,
    llama_tokenize, llama_vocab_type,
};
use std::{
    collections::BTreeMap,
    ffi::{c_char, CStr, CString},
    num::{NonZeroI32, NonZeroU32},
    os::unix::ffi::OsStringExt,
    path::PathBuf,
};

mod vocab;

pub use vocab::Vocab;
pub use vocab::VocabKind;

use crate::Prompt;

/// Convert a token to it's string representation.
///
/// Adapted from `llama.cpp/common/common.cpp`
///
/// # Panics
/// * If the token's piece is not valid UTF-8.
fn token_to_piece(token: llama_token, model: &Model) -> String {
    let mut buf = vec![0; 8];
    token_to_piece_ref(token, model, &mut buf);

    String::from_utf8(buf).unwrap_or("[Invalid UTF-8]".to_string())
}

/// Same as `token_to_piece`, but allows reusable buffers.
pub(crate) fn token_to_piece_ref(
    token: llama_token,
    model: &Model,
    buf: &mut Vec<u8>,
) {
    // Safety: If the Vec isn't big enough, the function will return the number
    // of bytes needed. We resize the buffer and call the function again. There
    // is no risk of a buffer overflow.
    let n_tokens = unsafe {
        llama_token_to_piece(
            model.as_ptr(),
            token,
            buf.as_mut_ptr() as *mut i8,
            buf.len().try_into().unwrap(),
        )
    };

    if n_tokens < 0 {
        buf.resize(-n_tokens as usize, 0);

        let check = unsafe {
            llama_token_to_piece(
                model.as_ptr(),
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len().try_into().unwrap(),
            )
        };

        assert_eq!(check, -n_tokens);
    } else {
        buf.resize(n_tokens as usize, 0);
    }
}

/// Quantize a Llama model.
pub fn llama_quantize(
    in_file: PathBuf,
    out_file: PathBuf,
    params: Option<llama_model_quantize_params>,
) -> Result<(), NonZeroU32> {
    let in_file = CString::new(in_file.into_os_string().into_vec()).unwrap();
    let out_file = CString::new(out_file.into_os_string().into_vec()).unwrap();
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
#[repr(transparent)]
pub struct Model {
    pub(crate) inner: *mut llama_model,
}

#[derive(Debug, From)]
pub enum MetaKey<'a> {
    Int(i32),
    String(&'a str),
}

impl Model {
    // TODO: make compile-time configurable(?)
    /// If unspecified, prefix the BOS token to a tokenized sequence.
    pub const DEFAULT_ADD_BOS: bool = true;
    /// If unspecified, append the EOS token to a tokenized sequence.
    pub const DEFAULT_ADD_EOS: bool = false;

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
        let path = CString::new(path.into_os_string().into_vec()).unwrap();
        // Safety: What's returned is POD
        let params = params.unwrap_or(unsafe { llama_model_default_params() });
        // Safety: The model is owned by the caller. We free it in the `Drop`
        // implementation. The path is a null-terminated string.
        let model = unsafe {
            llama_load_model_from_file(path.as_c_str().as_ptr(), params)
        };

        if model.is_null() {
            None
        } else {
            Some(Self { inner: model })
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
            Some(Self { inner: ptr })
        }
    }

    /// Unwrap the model and return the raw pointer.
    ///
    /// # Safety
    /// The caller is responsible for freeing the model using `llama_free_model`
    /// or `Model::from_raw` and then dropping it.
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
        unsafe { llama_token_bos(self.inner) }
    }

    /// Return the End of Sequence (EOS) token.
    pub fn eos(&self) -> llama_token {
        unsafe { llama_token_eos(self.inner) }
    }

    /// Return the next-line token.
    pub fn next_line(&self) -> llama_token {
        unsafe { llama_token_nl(self.inner) }
    }

    /// Return the infill prefix token.
    pub fn infill_prefix(&self) -> llama_token {
        unsafe { llama_token_prefix(self.inner) }
    }

    /// Return the infill middle token.
    pub fn infill_middle(&self) -> llama_token {
        unsafe { llama_token_middle(self.inner) }
    }

    /// Return the end of infill middle token.
    pub fn eot(&self) -> llama_token {
        unsafe { llama_token_eot(self.inner) }
    }

    /// Return the infill suffix token.
    pub fn infill_suffix(&self) -> llama_token {
        unsafe { llama_token_suffix(self.inner) }
    }

    /// Calculate the longest token length. Useful for optimizing searches.
    ///
    /// Time complexity is O(k) where k is the vocab size.
    pub fn max_token_len(&self) -> usize {
        let mut max_len = 0;
        for i in 0..self.n_vocab() {
            max_len = max_len.max(self.token_to_text(i).len());
        }

        max_len
    }

    /// Return whether BOS token is enabled.
    ///
    /// Returns None if unknown.
    pub fn add_bos(&self) -> Option<bool> {
        let code = unsafe { llama_add_bos_token(self.inner) };
        match code {
            -1 => None,
            0 => Some(false),
            1 => Some(true),
            _ => unreachable!(),
        }
    }

    /// Return whether the EOS token is enabled.
    ///
    /// Returns None if unknown.
    pub fn add_eos(&self) -> Option<bool> {
        let code = unsafe { llama_add_eos_token(self.inner) };
        match code {
            -1 => None,
            0 => Some(false),
            1 => Some(true),
            _ => unreachable!(),
        }
    }

    /// Vocab type.
    pub fn vocab_type(&self) -> llama_vocab_type {
        unsafe { llama_vocab_type(self.inner) }
    }

    /// Vocab size.
    pub fn n_vocab(&self) -> i32 {
        unsafe { llama_n_vocab(self.inner) }
    }

    /// Context size the model was trained with.
    pub fn context_size(&self) -> i32 {
        unsafe { llama_n_ctx_train(self.inner) }
    }

    /// Embedding size.
    pub fn embedding_size(&self) -> i32 {
        unsafe { llama_n_embd(self.inner) }
    }

    /// Rotary Position Encoding (RoPE) type.
    pub fn rope_type(&self) -> i32 {
        unsafe { llama_rope_type(self.inner) }
    }

    /// RoPE frequency scaling factor.
    pub fn rope_freq_scale(&self) -> f32 {
        unsafe { llama_rope_freq_scale_train(self.inner) }
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
        let written = unsafe {
            llama_model_desc(
                self.inner,
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            )
        };

        if written < 0 {
            panic!("snprintf encoding error.");
        } else {
            if written as usize > buf.len() {
                buf.resize(written as usize, 0);
                let check = unsafe {
                    llama_model_desc(
                        self.inner,
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len(),
                    )
                };
                assert_eq!(written, check);
            } else {
                buf.resize(written as usize, 0);
            }

            // This could fail if the model has junk in the description. It's
            // not a programmer error, so we'll just return an error string.
            return String::from_utf8(buf)
                .unwrap_or("[Invalid UTF-8]".to_string());
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
                );
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
                    assert_eq!(required, check);
                } else {
                    buf.resize(required as usize, 0);
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
                );

                if required < 0 {
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
                    assert_eq!(required, check);
                } else {
                    buf.resize(required as usize, 0);
                }
            },
            MetaKey::String(s) => {
                let key = CString::new(s).unwrap();
                let written = unsafe {
                    llama_model_meta_val_str(
                        self.inner,
                        key.as_c_str().as_ptr(),
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len(),
                    )
                };

                if written < 0 {
                    return None;
                }
                if buf.len() != written as usize {
                    buf.resize(written as usize, 0);
                    let check = unsafe {
                        llama_model_meta_val_str(
                            self.inner,
                            key.as_c_str().as_ptr(),
                            buf.as_mut_ptr() as *mut c_char,
                            buf.len(),
                        )
                    };
                    assert_eq!(written, check);
                } else {
                    buf.resize(written as usize, 0);
                }
            }
        };

        match String::from_utf8(buf).ok() {
            Some(mut s) => {
                // strip null terminators
                while s.ends_with('\0') {
                    let _ = s.pop();
                }

                Some(s)
            }
            None => None,
        }
    }
    /// Get a tensor by name.
    pub fn get_tensor<'a>(&'a self, name: &str) -> Option<&'a ggml_tensor> {
        let name = CString::new(name).unwrap();
        // Safety: The name is a null-terminated string.
        let tensor = unsafe {
            llama_get_model_tensor(
                self.inner,
                name.as_bytes_with_nul().as_ptr() as *const c_char,
            )
        };
        if tensor.is_null() {
            None
        } else {
            // Safety: The pointer is non-null and properly aligned. The
            // lifetime is tied to the model and documented in the function
            // signature.
            Some(unsafe { tensor.as_ref().unwrap() })
        }
    }

    /// Apply a LoRA adapter to a loaded model.
    ///
    /// In the case of an error, the function returns a non-zero error code.
    ///
    /// Parameters:
    /// * `lora` - Path to the LoRA adapter.
    /// * `scale` - Scaling factor for the adapter.
    /// * `hq_model` - Path to the high-quality model (optional).
    /// * `n_threads` - Number of threads to use.
    pub fn apply_lora(
        &mut self,
        lora: PathBuf,
        scale: f32,
        hq_model: Option<PathBuf>,
        n_threads: i32,
    ) -> Result<(), NonZeroI32> {
        let lora = CString::new(lora.into_os_string().into_vec()).unwrap();
        let hq_model = match hq_model {
            Some(hq) => {
                Some(CString::new(hq.into_os_string().into_vec()).unwrap())
            }
            None => None,
        };

        // Safety: The paths are null-terminated strings.
        let code = unsafe {
            llama_model_apply_lora_from_file(
                self.inner,
                lora.as_bytes_with_nul().as_ptr() as *const c_char,
                scale,
                hq_model
                    .map(|s| s.as_bytes_with_nul().as_ptr() as *const c_char)
                    .unwrap_or(std::ptr::null()),
                n_threads,
            )
        };
        if code == 0 {
            Ok(())
        } else {
            Err(code.try_into().unwrap())
        }
    }

    /// Tokenize a string into a Vec of tokens.
    pub fn tokenize(&self, input: &str, special: bool) -> Vec<llama_token> {
        // Adapted from `llama.cpp/common/common.cpp` which is not exposed to
        // the public API.

        // Guess a reasonable number of tokens to allocate. This is not
        // guaranteed to be enough, but it will probably be enough in most
        // cases.
        let mut n_tokens: i32 = (input.as_bytes().len()
            + if self.add_bos().unwrap_or(Model::DEFAULT_ADD_BOS) {
                1
            } else {
                0
            })
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
                self.inner,
                input.as_bytes().as_ptr() as *const i8,
                input.len().try_into().unwrap(),
                result.as_mut_ptr(),
                result.len().try_into().unwrap(),
                self.add_bos().unwrap_or(Self::DEFAULT_ADD_BOS),
                special,
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
                    self.inner,
                    input.as_bytes().as_ptr() as *const i8,
                    input.len().try_into().unwrap(),
                    result.as_mut_ptr(),
                    result.len().try_into().unwrap(),
                    self.add_bos().unwrap_or(Self::DEFAULT_ADD_BOS),
                    special,
                )
            };
            assert_eq!(check, -n_tokens);
        } else {
            result.resize(n_tokens as usize, 0);
        }

        if self.add_eos().unwrap_or(Self::DEFAULT_ADD_EOS) {
            result.push(self.eos());
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

    /// Apply chat template to a [`Prompt`] using `llama.cpp`'s
    /// `llama_chat_apply_template`. If template is `None`, the model's default
    /// template is used (metadata key `tokenizer.chat_template`).
    ///
    /// This can return `None` if the template is not supported by llama.cpp.
    /// This is equivalent to a return code of -1 from the C++ function. In this
    /// case, use one of the [`Prompt::format`] methods which cannot fail.
    ///
    /// `add_ass` is a boolean that determines whether to add the assistant's
    /// prefix to the output. This forces the model to generate the next message
    /// from the assistant's perspective which is usually the desired behavior.
    pub fn apply_chat_template(
        &self,
        template: Option<&str>,
        prompt: &Prompt,
        add_ass: bool,
    ) -> Option<String> {
        let template = template.map(|s| CString::new(s).unwrap());
        let template_ptr = template
            .map(|s| s.as_bytes_with_nul().as_ptr() as *const c_char)
            // the model's default will be used if template_ptr is null
            .unwrap_or(std::ptr::null_mut());

        // The recommended buffer allocation size is the number of characters in
        // the input messages * 2. This seems like overkill.
        let mut buf_len = 0;
        let mut messages: Vec<llama_chat_message> = Vec::new();
        for message in &prompt.transcript {
            let role = CString::new(match message.role {
                crate::Role::Human => prompt.human.as_bytes(),
                crate::Role::Agent => prompt.agent.as_bytes(),
                crate::Role::System => match &prompt.system {
                    Some(system) => system.as_bytes(),
                    // FIXME: System's name is a per-model thing, but it's not
                    // available on the model metadata. We do need to guess at
                    // this. This is a temporary solution. `Prompt` should use
                    // heuristics to determine the system's name.
                    None => "system".as_bytes(),
                },
            })
            .unwrap();

            let text = CString::new(message.text.as_bytes()).unwrap();

            buf_len += text.as_bytes().len();
            buf_len += role.as_bytes().len();

            // We are leaking memory here. We need to clean up after we're done
            // with the call to `llama_chat_apply_template`.
            messages.push(llama_chat_message {
                role: role.into_raw(),
                content: text.into_raw(),
            });
        }

        let mut buf = vec![0u8; buf_len];

        // Safety: The messages are valid UTF-8, null terminated, and outlive
        // the function call. It is very likely that the buffer will be too
        // small. We'll resize it and call the function again. This is fine
        // because the function will return the required length and not overflow
        // the buffer.
        let ret = unsafe {
            llama_chat_apply_template(
                self.inner,
                template_ptr,
                messages.as_ptr(),
                messages.len(),
                add_ass,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
            )
        };

        // This is actually undocumented in the C++ docs, but this is what
        // happens when a tempate is unsupported by llama.cpp.
        if ret == -1 {
            return None;
        }

        // If the return is positive, it is the required length.
        let required_len: usize = ret.try_into().unwrap();

        if required_len > buf_len {
            buf.resize(required_len, 0);

            let check: usize = unsafe {
                llama_chat_apply_template(
                    self.inner,
                    template_ptr,
                    messages.as_ptr(),
                    messages.len(),
                    add_ass,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            }
            .try_into()
            .unwrap();

            assert!(check == required_len)
        }

        // Free the messages.
        for message in messages {
            // Safety: we just created these pointers above with
            // `CString::into_raw`. We are taking ownership to free the strings.
            unsafe {
                _ = CString::from_raw(message.role as *mut i8);
                _ = CString::from_raw(message.content as *mut i8);
            }
        }

        Some(String::from_utf8(buf).unwrap_or("[Invalid UTF-8]".to_string()))
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
        let ptr = unsafe { llama_token_get_text(self.inner, token) };
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
        unsafe { llama_token_get_score(self.inner, token) }
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

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { llama_free_model(self.inner) };
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_sys_3::llama_vocab_type_LLAMA_VOCAB_TYPE_BPE;

    use crate::Message;

    use super::*;

    #[test]
    fn test_model() {
        use std::path::PathBuf;

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // This should be a properly converted llama 3 model or this test will
        // fail.
        path.push("models/model.gguf");

        let model = Model::from_file(path, None).unwrap();

        assert_eq!(model.bos(), 128000);
        assert_eq!(model.eos(), 128001);
        assert_eq!(model.next_line(), 128);
        // These are for CodeLLama. These are included in the conversion of the
        // llama 3 model I am using but these are wrong. They are for llama 2.
        // TODO: Wait for a proper conversion and adjust these values.
        assert_eq!(model.infill_prefix(), 32007);
        assert_eq!(model.infill_suffix(), 32008);
        assert_eq!(model.infill_middle(), 32009);
        assert_eq!(model.eot(), 32010);
        assert_eq!(model.add_bos(), None);
        assert_eq!(model.add_eos(), None);
        assert_eq!(model.vocab_type(), llama_vocab_type_LLAMA_VOCAB_TYPE_BPE);
        assert_eq!(model.n_vocab(), 128256);
        assert_eq!(model.context_size(), 8192);
        assert_eq!(model.embedding_size(), 8192);
        assert_eq!(model.rope_type(), 0);
        assert_eq!(model.rope_freq_scale(), 1.0);
        let desc = model.desc().to_lowercase();
        assert!(desc.starts_with("llama"));
        assert_eq!(model.meta_count(), 18);
        assert_eq!(
            model.get_meta("tokenizer.chat_template").unwrap(),
            "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }".to_owned()
        );
        // The model size will be different on different systems, but the range
        // should be reasonable. Not zero and not over 200GB.
        assert!(model.size() > 8192 && model.size() < 200_000_000_000);
        // LLama v3 variants come between 7 and 70 billion parameters give or
        // take.
        assert!(
            model.n_params() > 6_000_000_000
                && model.n_params() < 80_000_000_000
        );

        // let meta = model.meta();
        // assert_eq!(meta.len(), 16);

        // test tokenization
        const EXPECTED: &str = "Hello, world!";
        let tokens = model.tokenize(EXPECTED, false);
        assert_eq!(tokens, &[9906, 11, 1917, 0]);
        assert_eq!(model.tokens_to_string(tokens), EXPECTED);

        // test get token text
        assert_eq!(model.token_to_text(9906), "Hello");
        assert_eq!(
            model
                .tokens_to_text(&[9906, 11, 1917, 0])
                .collect::<Vec<_>>(),
            &["Hello", ",", "Ġworld", "!"]
        );

        // test template application
        let messages = vec![
            Message {
                role: crate::Role::Human,
                text: "Hello, world!".to_string(),
            },
            Message {
                role: crate::Role::Agent,
                text: "Hi!".to_string(),
            },
            Message {
                role: crate::Role::Human,
                text: "So, how's it going?".to_string(),
            },
        ];

        let prompt = crate::Prompt {
            human: "user".to_string(),
            agent: "assistant".to_string(),
            system: None,
            transcript: messages,
            setting: Some(
                "A conversation between a user and an assistant.".to_string(),
            ),
        };

        // FIXME: The model currently testing with has a jinja2 template that
        // is not supported by llama.cpp. The code in llama.cpp is not actually
        // a jinja parser. It relies on heuristics and will fail if the template
        // is not recognized. This will fail until LLama3 support is released,
        // however, the code is believed to be correct.
        //
        // In the meantime we might want to fall back to our own formatting
        // methods in cases where the template is not supported.
        let template = model.get_meta("tokenizer.chat_template").unwrap();
        let result = model
            .apply_chat_template(Some(&template), &prompt, true)
            .unwrap();
        assert_eq!(
            result,
            "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>\n<|im_start|>user\nSo, how's it going?<|im_end|>\n<|im_start|>assistant\n",
        );
    }
}
