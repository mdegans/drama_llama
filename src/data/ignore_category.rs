use llama_cpp_sys_3::llama_token;

use crate::Model;

/// Common sequences to ignore (for repetition penalty, etc.)
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub enum IgnoreCategory {
    // NOTE: If you add a new variant here, add it to ALL and keep this list
    // and ALL in alphabetical order.
    // TODO: static assert all this.
    /// English stopwords (very common words that should not be penalized).
    English,
    /// JSON structural tokens (`{`, `}`, `[`, `]`, `,`, `:`, `"`). Useful
    /// when generating JSON without grammar constraints, or when the
    /// grammar allows multiple valid closures and repetition penalty on
    /// `}` would bias the walker toward extending rather than closing.
    Json,
    /// Prose punctuation (`.`, `,`, `;`, `:`, `!`, `?`). These tokens have
    /// no lexical variety, so the "avoid repetition" signal is
    /// structurally wrong for them — penalty accumulating on `.` biases
    /// the model toward longer sentences and run-on prose.
    Punctuation,
}

/// Use [`IgnoreCategory`] instead.
#[deprecated(since = "0.7.0", note = "renamed to `IgnoreCategory`")]
pub type StopWords = IgnoreCategory;

impl IgnoreCategory {
    pub const ALL: [IgnoreCategory; 3] = [
        IgnoreCategory::English,
        IgnoreCategory::Json,
        IgnoreCategory::Punctuation,
    ];

    pub const fn as_str(&self) -> &'static str {
        match self {
            IgnoreCategory::English => "English",
            IgnoreCategory::Json => "JSON",
            IgnoreCategory::Punctuation => "Punctuation",
        }
    }

    pub const fn words(&self) -> &'static [&'static str] {
        match self {
            IgnoreCategory::English => ENGLISH,
            IgnoreCategory::Json => JSON_SYNTAX,
            IgnoreCategory::Punctuation => PUNCTUATION,
        }
    }

    /// Tokenizes `self` using the given `model``.
    pub fn into_tokens(
        self,
        model: &Model,
    ) -> impl Iterator<Item = llama_token> + '_ {
        self.words()
            .iter()
            // TODO: there is allocation here that can be avoided by turning the
            // tokenize function into a method returning an iterator, however
            // it's not a big deal since this is only done once.
            .map(|word| model.tokenize(word, false).into_iter())
            .flatten()
    }
}

/// JSON structural tokens. `"` is included so string-delimiter repetition
/// across keys and values doesn't accumulate penalty.
pub const JSON_SYNTAX: &[&str] = &["{", "}", "[", "]", ",", ":", "\""];

/// Prose punctuation. Narrow set of sentence terminators and separators
/// that shouldn't be penalized for repetition — they lack the lexical
/// variety that makes repetition penalty meaningful.
pub const PUNCTUATION: &[&str] = &[".", ",", ";", ":", "!", "?"];

/// A list of common English stopwords from NLTK.
pub const ENGLISH: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_contains_every_variant() {
        assert_eq!(IgnoreCategory::ALL.len(), 3);
        assert!(IgnoreCategory::ALL.contains(&IgnoreCategory::English));
        assert!(IgnoreCategory::ALL.contains(&IgnoreCategory::Json));
        assert!(IgnoreCategory::ALL.contains(&IgnoreCategory::Punctuation));
    }

    #[test]
    fn as_str_matches_variant() {
        assert_eq!(IgnoreCategory::English.as_str(), "English");
        assert_eq!(IgnoreCategory::Json.as_str(), "JSON");
        assert_eq!(IgnoreCategory::Punctuation.as_str(), "Punctuation");
    }

    #[test]
    fn words_spot_check() {
        assert!(IgnoreCategory::English.words().contains(&"the"));
        assert!(IgnoreCategory::Json.words().contains(&"}"));
        assert!(IgnoreCategory::Json.words().contains(&"]"));
        assert!(IgnoreCategory::Punctuation.words().contains(&"."));
        assert!(IgnoreCategory::Punctuation.words().contains(&"?"));
    }

    #[test]
    fn json_and_punctuation_overlap_on_comma_and_colon() {
        // Intentional overlap — `,` and `:` appear in both JSON structure
        // and prose. Users enabling both get the union (no harm since
        // ignoring the same token twice is a no-op in the drain loop).
        assert!(IgnoreCategory::Json.words().contains(&","));
        assert!(IgnoreCategory::Punctuation.words().contains(&","));
        assert!(IgnoreCategory::Json.words().contains(&":"));
        assert!(IgnoreCategory::Punctuation.words().contains(&":"));
    }
}
