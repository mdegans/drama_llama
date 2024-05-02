//! Vocabulary constraints.

use llama_cpp_sys_3::llama_token;
use regex::Regex;

use crate::{data::banned::Banned, model::token_to_piece_ref, Model, NGram};

/// A very imperfect regex for safe tokens. This could use some improvement.
pub const SAFE_REGEX: &str = r#"^[▁ a-zA-Z]{2,32}|[ ▁\(\)\.\?!\"\'\-_]{1,32}|[aAI]{1}|\n{1,3}|\t{1,3}| {1,16}$"#;
pub const LETTERS_REGEX: &str = r#"^[a-zA-Z]{1}$"#;
pub const CODE_REGEX: &str = r#"^[ \d\\(\){\}\[\]\;\:\"\'\<\>\,\.\\\/\?\.\!\@\#\$\%\^\&\=\`\~]{1,32}|\w{2,32}$"#;

// This is temporary until we can get the regex working for llama. It works in
// regex101, but not here. With these tokens banned, weird things happen.
const LLAMA_2_ALLOW_LIST: &[llama_token] = &[
    0,     // unknown
    1,     // bos
    2,     // eos
    0x0D,  // \n
    0x20,  // space
    0x49,  // I
    0x3D,  // =
    0x61,  // a
    0x75,  // u
    29871, // ▁ (word boundary)
    29874, // a
    29889, // .
    29892, // ,
    29897, // )
    29898, // (
    29899, // -
    29901, // :
    29902, // I
    29909, // A
    29912, // {
    29913, // }
    29915, // '
    29918, // _
    29922, // =
    29930, // *
    29936, // ;
    29937, // #
    29938, // $
    29944, // л
    29961, // [
    29962, // ]
    29973, // ?
    29974, // +
    29985, // ^
    29989, // |
    29991, // !
    29992, // @
    29995, // %
    30022, // ~
    30098, // …
    30142, // λ
];

const LLAMA_2_ALLOW_RANGES: &[std::ops::RangeInclusive<llama_token>] = &[
    0x20..=0x3C, // !"#$%&'()*+,-./0123456789:;
    0x3F..=0x41, // ?@A
    0x5B..=0x60, // [\]^_`
    0x7B..=0x7E, // {|}~
];

#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VocabKind {
    /// All tokens and control characters are allowed. This is not recommended,
    /// especially if the output is going to be used in a web context. Banned
    /// n-grams are still enforced.
    Unsafe,
    /// Words, word fragments, punctuation, and the letter "a" are allowed. This
    /// is the default vocabulary. The idea is to prohibit generation of
    /// arbitrary sequences which could bypass filters, as well as code which
    /// could cause security issues.
    ///
    /// That being said *this is not yet validated* to be very safe, so care
    /// should be taken especially for web contexts.
    Safe,
    /// Letters only. Allowing this will allow generation of any sequence, but
    /// only one letter at a time. This is unsafe and should not be used unless
    /// it's absolutely necessary.
    ///
    /// Using it to generate bigotry is a violation of the license under which
    /// this software is distributed. See `LICENSE.md` for details.
    Letters,
    /// Code. This will allow generation of words, digits, and common symbols
    /// used in code. Letters are not enabled.
    // Because 4chan got GPT-4 to generate the N word by getting it to "run"
    // code concatenating individual letters, we have to ban this.
    Code,
}

// derive_more::Display is failing, so we're implementing it manually.
#[cfg(feature = "cli")]
impl std::fmt::Display for VocabKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VocabKind::Unsafe => write!(f, "unsafe"),
            VocabKind::Safe => write!(f, "safe"),
            VocabKind::Letters => write!(f, "letters"),
            VocabKind::Code => write!(f, "code"),
        }
    }
}

impl Into<Regex> for VocabKind {
    fn into(self) -> Regex {
        match self {
            VocabKind::Unsafe => Regex::new("*").unwrap(),
            VocabKind::Safe => Regex::new(SAFE_REGEX).unwrap(),
            VocabKind::Letters => Regex::new(LETTERS_REGEX).unwrap(),
            VocabKind::Code => Regex::new(CODE_REGEX).unwrap(),
        }
    }
}

#[derive(Debug)]
pub struct Vocab {
    /// Allowed tokens. This is a Vec of bool rather than a vec of ranges so
    /// that lookup is O(1). This will happen in a fairly tight loop, so it's
    /// probably worth it.
    allowed_tokens: Vec<bool>,
    /// Banned ngrams. These are at least all possible pairs of tokens that
    /// would generate a banned word. Letters are not included since the number
    /// of permutations is too high.
    banned: Option<Banned>,
    /// Longest token length. This is used to optimize search for stop strings.
    longest_token: usize,
}

impl Vocab {
    pub fn new(
        enabled: impl IntoIterator<Item = VocabKind>,
        model: &Model,
    ) -> Self {
        let enabled: Vec<VocabKind> = enabled.into_iter().collect();
        let banned = if model.desc().to_lowercase().starts_with("llama v2") {
            Some(Banned::LlamaEnglish)
        } else {
            None
        };
        if enabled.contains(&VocabKind::Unsafe) {
            return Self {
                allowed_tokens: vec![true; model.n_vocab() as usize],
                longest_token: model.max_token_len(),
                banned,
            };
        }
        let enabled: Vec<Regex> = enabled.into_iter().map(Into::into).collect();

        let n_tokens = model.n_vocab();

        let mut buf = Vec::new();

        let mut allowed_tokens: Vec<bool> = (0..n_tokens)
            .map(|token| {
                token_to_piece_ref(token, model, &mut buf);
                enabled
                    .iter()
                    .any(|re| re.is_match(&String::from_utf8_lossy(&buf)))
            })
            .collect();

        if model.desc().to_lowercase().starts_with("llama v2") {
            for &token in LLAMA_2_ALLOW_LIST {
                allowed_tokens[token as usize] = true;
            }

            for range in LLAMA_2_ALLOW_RANGES {
                for token in range.clone() {
                    allowed_tokens[token as usize] = true;
                }
            }
        }

        // TODO: Fix regex, or add LLAMA_3 allow list. As it is now, generation
        // is potato without "Unsafe" vocab because the regex is too strict.

        Self {
            allowed_tokens,
            longest_token: model.max_token_len(),
            banned,
        }
    }

    /// Returns true if an ngram is forbidden. Forbidden [`NGram`]s are those
    /// that contain a token that is not allowed, or that are in the banned
    /// ngrams set.
    pub fn is_forbidden(&self, ngram: &NGram) -> bool {
        if ngram
            .iter()
            .any(|&token| !self.allowed_tokens[token as usize])
        {
            return true;
        }
        if let Some(banned) = &self.banned {
            banned
                .as_slice()
                .binary_search(&[ngram[0], ngram[1]])
                .is_ok()
        } else {
            false
        }
    }

    /// Piece length of the longest token.
    ///
    /// Time complexity: O(1).
    pub fn max_token_len(&self) -> usize {
        self.longest_token
    }

    /// Allowed tokens.
    pub fn allowed_tokens(&self) -> &Vec<bool> {
        &self.allowed_tokens
    }

    /// Banned ngrams.
    pub fn banned(&self) -> Option<&Banned> {
        self.banned.as_ref()
    }

    /// Returns the number of allowed tokens.
    ///
    /// O(n) where n is the number of tokens.
    pub fn n_allowed(&self) -> usize {
        self.allowed_tokens.iter().filter(|&&b| b).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_cpp_sys_3::llama_token;
    use rayon::prelude::*;
    use std::{
        collections::{BTreeSet, HashSet},
        path::PathBuf,
    };

    /// Generate banned ngrams for a model. This is very slow and can take a few
    /// minutes even on a fast machine. It is only used for testing and
    /// generating the banned ngrams for the various models.
    fn generate_banned_ngrams(model: &Model) -> BTreeSet<NGram> {
        // Safety: this is only called from test code and we don't use any
        // methods that mutate the model, so it is safe to share between
        // threads. In the future we might make model actually thread safe.
        unsafe impl Sync for Model {}

        let mut banned_ngrams = BTreeSet::new();

        let n_vocab = model.n_vocab();
        let banned_regex: Vec<Regex> = crate::data::banned::ENGLISH_WORDS
            .iter()
            .map(|s| Regex::new(s).unwrap())
            .collect();

        let (tx, rx) = std::sync::mpsc::channel();
        (0..n_vocab).into_par_iter().for_each_with(tx, |tx, first| {
            let mut first_buf = Vec::new();
            let mut second_buf = Vec::new();
            let mut joined_buf = String::new();

            let mut banned_chunk: HashSet<NGram> = HashSet::new();

            for second in 0..n_vocab {
                first_buf.clear();
                second_buf.clear();
                joined_buf.clear();

                token_to_piece_ref(first, &model, &mut first_buf);
                token_to_piece_ref(second, &model, &mut second_buf);

                joined_buf.push_str(
                    String::from_utf8_lossy(&first_buf).to_lowercase().as_ref(),
                );
                joined_buf.push_str(
                    String::from_utf8_lossy(&second_buf)
                        .to_lowercase()
                        .as_ref(),
                );

                for regex in &banned_regex {
                    if regex.is_match(&joined_buf.to_lowercase()) {
                        let ngram =
                            NGram::try_from_tokens(&[first, second]).unwrap();
                        banned_chunk.insert(ngram);
                        break;
                    }
                }
            }

            tx.send(banned_chunk).unwrap();
        });

        banned_ngrams.extend(rx.into_iter().flatten());

        banned_ngrams
    }

    #[test]
    fn test_vocab() {
        // This is a llama model
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("models/model.gguf");

        let model = Model::from_file(path, None).unwrap();
        let vocab = Vocab::new(vec![VocabKind::Safe], &model);

        // Check that the ngrams are forbidden
        for forbidden in crate::data::banned::ENGLISH_BIGRAMS {
            let ngram = NGram::try_from_tokens(forbidden).unwrap();
            assert!(vocab.is_forbidden(&ngram));
        }
    }

    #[test]
    #[ignore = "very long running"]
    /// This is a very long running test that generates the banned n-grams for
    /// the Llama model. This can take a few minutes even on a fast machine.
    fn test_banned_ngrams_llama() {
        // This is a llama model
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut model_path = root.clone();
        model_path.push("models/model.gguf");
        let model = Model::from_file(model_path, None).unwrap();
        let mut out_path = root.clone();
        out_path
            .push(format!("tests/data/banned_ngrams/ngrams-english-llama.txt"));

        let expected = generate_banned_ngrams(&model);
        let actual: BTreeSet<NGram> = crate::data::banned::ENGLISH_BIGRAMS
            .iter()
            .filter_map(|slice| NGram::try_from_tokens(slice).ok())
            .collect();

        let v: Vec<Vec<llama_token>> =
            expected.iter().map(|n| n.as_slice().to_vec()).collect();

        // This representation should be easy to copy and paste into the
        // BANNED_LLAMA_NGRAMS array. We could automate this, but I don't want
        // to automate generation of code.
        std::fs::write(out_path, format!("{:#?}", v)).unwrap();

        assert_eq!(expected, actual);
    }
}
