//! Known-identifier exemption for the repetition penalty.
//!
//! An identifier (a post's UUID, a `GOV-2026.7`) has no lexical variety
//! and must be re-emitted *verbatim*; stacked n-gram penalties push a
//! model off the exact string after a few sightings, and it starts
//! distrusting its own memory. The shape of an id is not the fix — an
//! agent mostly cites by bare 8-hex prefix, which is indistinguishable
//! from any hex word — so the fix is *faithful copies of ids the context
//! already contains*:
//!
//! 1. the consumer names what an identifier looks like
//!    ([`IdPattern`]s on `RepetitionOptions`);
//! 2. per call, `Session` runs the patterns over the prompt's text
//!    (tool results, user turns, tool-call arguments — not the model's
//!    prior thoughts, which is where its own wrong ids live) into the
//!    call's *known ids*;
//! 3. at apply time, [`IdGuard`] exempts a penalized token iff its
//!    bytes, appended to the current partial word, are a prefix of a
//!    known id.
//!
//! Only faithful copies are exempt: a string that is a prefix of no
//! known id keeps its full penalty, so the exemption steers toward the
//! ids the context actually holds. A bare prefix needs no pattern of
//! its own — it is a prefix of the full id by construction. Nothing
//! here is sampler *state*: the partial word is derived from the token
//! history at every step, so snapshot and restore are unaffected.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    ops::Bound,
};

use crate::{backend::Model, Token};

use super::region::RegionGuard;

/// A compiled identifier pattern for `RepetitionOptions::id_patterns`.
///
/// A newtype over [`regex::Regex`] so the options struct keeps its
/// derived `PartialEq` (regexes compare by source) and so an invalid
/// pattern is rejected when the sidecar is read, not silently skipped
/// at generation time.
#[derive(Clone, Debug)]
pub struct IdPattern(regex::Regex);

impl IdPattern {
    /// Compile a pattern.
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        regex::Regex::new(pattern).map(Self)
    }

    /// The pattern source.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// The compiled regex.
    pub fn regex(&self) -> &regex::Regex {
        &self.0
    }
}

impl PartialEq for IdPattern {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl std::str::FromStr for IdPattern {
    type Err = regex::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl From<regex::Regex> for IdPattern {
    fn from(regex: regex::Regex) -> Self {
        Self(regex)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for IdPattern {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for IdPattern {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let s = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Self::new(&s).map_err(serde::de::Error::custom)
    }
}

/// Collect every match of every pattern in `text` into `out`.
pub(crate) fn collect_ids(
    patterns: &[IdPattern],
    text: &str,
    out: &mut BTreeSet<Vec<u8>>,
) {
    for pattern in patterns {
        for m in pattern.regex().find_iter(text) {
            out.insert(m.as_str().as_bytes().to_vec());
        }
    }
}

/// Bytes an identifier may consist of. Everything else — including
/// every non-ASCII byte — is a word boundary.
const fn is_id_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')
}

/// Longer than any identifier worth protecting. A partial word past
/// this can match nothing, so the tail walk stops here.
const TAIL_CAP: usize = 64;

/// True iff some id in `ids` starts with `prefix`. A range scan, no
/// allocation: the candidate is the first id at or after `prefix` in
/// byte order.
fn has_prefix(ids: &BTreeSet<Vec<u8>>, prefix: &[u8]) -> bool {
    !prefix.is_empty()
        && ids
            .range::<[u8], _>((Bound::Included(prefix), Bound::Unbounded))
            .next()
            .is_some_and(|id| id.starts_with(prefix))
}

/// The partial word at the end of `tokens`: bytes after the last
/// boundary byte, walking pieces back from the end. `None` when the
/// word is longer than [`TAIL_CAP`] (it can match no id). Typically one
/// to three pieces.
fn trailing_word<M: Model>(
    tokens: &[Token],
    model: &M,
    scratch: &mut Vec<u8>,
) -> Option<Vec<u8>> {
    // Collected reversed, then flipped once.
    let mut rev: Vec<u8> = Vec::with_capacity(16);
    for &token in tokens.iter().rev() {
        piece_into(model, token, scratch);
        for &b in scratch.iter().rev() {
            if !is_id_byte(b) {
                rev.reverse();
                return Some(rev);
            }
            if rev.len() == TAIL_CAP {
                return None;
            }
            rev.push(b);
        }
    }
    rev.reverse();
    Some(rev)
}

/// `token_to_piece_ref` with the buffer pre-sized to the model's
/// longest piece, which is llama.cpp's single-FFI-call path.
fn piece_into<M: Model>(model: &M, token: Token, buf: &mut Vec<u8>) {
    buf.resize(model.max_token_len(), 0);
    model.token_to_piece_ref(token, buf);
}

/// The exemption, on the same hook the constrained pass uses to protect
/// region-exit tokens ([`RegionGuard`]). Built once per sampling step
/// from the token history and the call's known ids; consulted only for
/// tokens the pass is about to penalize.
///
/// Documented edges: at an empty partial word, *any* token whose piece
/// starts some known id is exempt — with dozens of UUIDs in context
/// that is every space-prefixed hex digit and short hex word (` 1`,
/// ` be`, ` bad`) as the *first* token of a word; the next token is
/// judged against the now-longer word and re-penalized, and surgical
/// mode already spares single digits. Bounded, not zero. A merged
/// piece that completes an id and crosses a boundary (`9d]`) is exempt
/// only if the id is completed exactly.
pub(crate) struct IdGuard<'a, M: Model> {
    /// The partial word, or `None` past the cap.
    tail: Option<Vec<u8>>,
    /// Whether the current word could still become an id: empty (a
    /// word boundary) or a prefix of some id. When it cannot, only a
    /// piece that starts a new word can be exempt — the per-step
    /// early-out.
    tail_live: bool,
    ids: &'a BTreeSet<Vec<u8>>,
    model: &'a M,
    memo: RefCell<BTreeMap<Token, bool>>,
    scratch: RefCell<Vec<u8>>,
}

impl<'a, M: Model> IdGuard<'a, M> {
    /// `None` when there is nothing to exempt.
    pub(crate) fn build(
        tokens: &[Token],
        ids: &'a BTreeSet<Vec<u8>>,
        model: &'a M,
    ) -> Option<Self> {
        if ids.is_empty() {
            return None;
        }
        let mut scratch = Vec::with_capacity(model.max_token_len());
        let tail = trailing_word(tokens, model, &mut scratch);
        // An empty word is a word boundary: any id may begin here.
        let tail_live = tail
            .as_deref()
            .is_some_and(|t| t.is_empty() || has_prefix(ids, t));
        Some(Self {
            tail,
            tail_live,
            ids,
            model,
            memo: RefCell::new(BTreeMap::new()),
            scratch: RefCell::new(scratch),
        })
    }

    /// Would `piece`, emitted next, be a faithful copy of a known id?
    fn faithful(&self, piece: &[u8]) -> bool {
        match piece.iter().rposition(|&b| !is_id_byte(b)) {
            // The piece starts a new word: judge the part after its last
            // boundary on its own — this is how ` 05` after `post_id:`
            // resolves to `05`. Before that boundary it may also have
            // *completed* the current id exactly (`9d]`).
            Some(boundary) => {
                let post = &piece[boundary + 1..];
                if !post.is_empty() {
                    return has_prefix(self.ids, post);
                }
                let Some(tail) = self.tail.as_deref() else {
                    return false;
                };
                let pre = &piece[..boundary];
                // The piece itself must contribute id bytes: a bare `]`
                // after a complete id completes nothing.
                if pre.is_empty() || !pre.iter().all(|&b| is_id_byte(b)) {
                    return false;
                }
                let mut word = Vec::with_capacity(tail.len() + pre.len());
                word.extend_from_slice(tail);
                word.extend_from_slice(pre);
                self.ids.contains(&word)
            }
            // The piece extends the current word.
            None => {
                if !self.tail_live {
                    return false;
                }
                let tail = self.tail.as_deref().unwrap_or_default();
                if tail.len() + piece.len() > TAIL_CAP {
                    return false;
                }
                let mut word = Vec::with_capacity(tail.len() + piece.len());
                word.extend_from_slice(tail);
                word.extend_from_slice(piece);
                has_prefix(self.ids, &word)
            }
        }
    }
}

impl<M: Model> RegionGuard for IdGuard<'_, M> {
    fn is_protected(&self, token: Token) -> bool {
        if let Some(&hit) = self.memo.borrow().get(&token) {
            return hit;
        }
        let mut scratch = self.scratch.borrow_mut();
        piece_into(self.model, token, &mut scratch);
        let exempt = !scratch.is_empty() && self.faithful(&scratch);
        self.memo.borrow_mut().insert(token, exempt);
        exempt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pieces indexed by token id; ids past the table are empty.
    struct Pieces(&'static [&'static str]);

    impl Model for Pieces {
        type Error = std::convert::Infallible;
        fn n_vocab(&self) -> i32 {
            self.0.len() as i32
        }
        fn bos(&self) -> Token {
            0
        }
        fn eos(&self) -> Token {
            0
        }
        fn eot(&self) -> Token {
            0
        }
        fn special_tokens(&self) -> Vec<Token> {
            vec![]
        }
        fn eog_tokens(&self) -> Vec<Token> {
            vec![]
        }
        fn max_token_len(&self) -> usize {
            self.0.iter().map(|p| p.len()).max().unwrap_or(0)
        }
        fn tokenize(&self, _: &str, _: bool) -> Vec<Token> {
            unimplemented!()
        }
        fn token_to_piece(&self, token: Token) -> String {
            self.0
                .get(token as usize)
                .copied()
                .unwrap_or("")
                .to_string()
        }
        fn token_to_piece_ref(&self, token: Token, buf: &mut Vec<u8>) {
            buf.clear();
            buf.extend_from_slice(self.token_to_piece(token).as_bytes());
        }
        fn context_size(&self) -> i32 {
            0
        }
        fn chat_template_source(&self) -> Option<String> {
            None
        }
        fn recommended_sampling(&self) -> crate::SamplingParams {
            crate::SamplingParams::default()
        }
    }

    const P: &[&str] = &[
        "",     // 0
        " ",    // 1
        "[",    // 2
        "05",   // 3
        "676",  // 4
        "b9d",  // 5
        "-",    // 6
        "8aa7", // 7
        "c9",   // 8
        " 05",  // 9  space-prefixed BPE token
        "9d]",  // 10 completes-and-crosses
        "9e]",  // 11 wrong completion
        "x",    // 12 an id byte that is not an id
        "é",    // 13 non-ASCII: a boundary
    ];
    const SP: Token = 1;
    const LB: Token = 2;
    const T05: Token = 3;
    const T676: Token = 4;
    const TB9D: Token = 5;
    const DASH: Token = 6;
    const T8AA7: Token = 7;
    const TC9: Token = 8;
    const SP05: Token = 9;
    const T9D_RB: Token = 10;
    const T9E_RB: Token = 11;
    const TX: Token = 12;
    const EACUTE: Token = 13;

    fn ids(list: &[&str]) -> BTreeSet<Vec<u8>> {
        list.iter().map(|s| s.as_bytes().to_vec()).collect()
    }

    const FULL: &str = "05676b9d-8aa7-430e-9138-444080e34065";

    #[test]
    fn has_prefix_is_a_prefix_query() {
        let set = ids(&[FULL, "GOV-2026.7"]);
        assert!(has_prefix(&set, b"05"));
        assert!(has_prefix(&set, b"05676b9d"));
        assert!(has_prefix(&set, b"05676b9d-"));
        assert!(has_prefix(&set, FULL.as_bytes()));
        assert!(has_prefix(&set, b"GOV-2"));
        assert!(!has_prefix(&set, b""), "the empty word matches nothing");
        assert!(!has_prefix(&set, b"c9"));
        assert!(!has_prefix(&set, b"05676b9e"));
        let mut longer = FULL.as_bytes().to_vec();
        longer.push(b'0');
        assert!(!has_prefix(&set, &longer), "longer than the id");
    }

    #[test]
    fn trailing_word_walks_back_to_the_boundary() {
        let m = Pieces(P);
        let mut s = Vec::new();
        let w = |toks: &[Token], s: &mut Vec<u8>| trailing_word(toks, &m, s);
        assert_eq!(w(&[], &mut s).unwrap(), b"");
        assert_eq!(w(&[LB], &mut s).unwrap(), b"");
        assert_eq!(w(&[LB, T05], &mut s).unwrap(), b"05");
        assert_eq!(w(&[LB, T05, T676, TB9D], &mut s).unwrap(), b"05676b9d");
        assert_eq!(
            w(&[LB, T05, T676, TB9D, DASH, T8AA7], &mut s).unwrap(),
            b"05676b9d-8aa7"
        );
        // A piece containing a boundary starts the word after it.
        assert_eq!(w(&[T05, SP05, T676], &mut s).unwrap(), b"05676");
        // Non-ASCII is a boundary.
        assert_eq!(w(&[T05, EACUTE, T676], &mut s).unwrap(), b"676");
        // Past the cap: nothing can match.
        let long: Vec<Token> = std::iter::repeat_n(T8AA7, 20).collect();
        assert!(w(&long, &mut s).is_none());
        assert_eq!(w(&[SP], &mut s).unwrap(), b"");
    }

    #[test]
    fn faithful_copies_are_exempt_from_the_first_byte() {
        let m = Pieces(P);
        let set = ids(&[FULL]);
        // At `[`: the first token of the id is exempt, a wrong start is not.
        let g = IdGuard::build(&[LB], &set, &m).unwrap();
        assert!(g.is_protected(T05));
        assert!(!g.is_protected(TC9));
        assert!(!g.is_protected(TX));
        // Mid-word: the faithful continuation only.
        let g = IdGuard::build(&[LB, T05, T676], &set, &m).unwrap();
        assert!(g.is_protected(TB9D));
        assert!(!g.is_protected(T676));
        assert!(!g.is_protected(TC9));
        // The dash and the next group are part of the id.
        let g = IdGuard::build(&[LB, T05, T676, TB9D], &set, &m).unwrap();
        assert!(g.is_protected(DASH));
        let g = IdGuard::build(&[LB, T05, T676, TB9D, DASH], &set, &m).unwrap();
        assert!(g.is_protected(T8AA7));
        assert!(!g.is_protected(T05), "8aa7 is due, not 05");
    }

    #[test]
    fn space_prefixed_piece_resolves_past_the_space() {
        let m = Pieces(P);
        let set = ids(&[FULL]);
        // After `x` (a dead word), ` 05` starts a new, faithful word.
        let g = IdGuard::build(&[TX], &set, &m).unwrap();
        assert!(!g.tail_live);
        assert!(g.is_protected(SP05));
        assert!(!g.is_protected(T05), "`x05` is no id");
        // A bare space is a boundary with nothing after it: not exempt.
        assert!(!g.is_protected(SP));
    }

    /// A merged piece that finishes the id and crosses a boundary
    /// (`9d]`) is exempt iff the id is completed exactly; a bare
    /// boundary after an already-complete id completes nothing.
    #[test]
    fn completing_then_crossing_is_exempt_iff_exact() {
        let m = Pieces(P);
        let set = ids(&["056769d"]);
        let g = IdGuard::build(&[LB, T05, T676], &set, &m).unwrap();
        assert!(g.is_protected(T9D_RB), "05676 + 9d completes exactly");
        assert!(!g.is_protected(T9E_RB), "05676 + 9e does not");

        let set = ids(&["05676b9d"]);
        let g = IdGuard::build(&[LB, T05, T676, TB9D], &set, &m).unwrap();
        assert!(!g.is_protected(T9D_RB), "05676b9d9d is not the id");
        assert!(!g.is_protected(LB), "`[` alone contributes no id bytes");
    }

    #[test]
    fn memo_and_empty_piece() {
        let m = Pieces(P);
        let set = ids(&[FULL]);
        let g = IdGuard::build(&[LB], &set, &m).unwrap();
        assert!(!g.is_protected(0), "empty piece is never exempt");
        assert!(g.is_protected(T05));
        assert!(g.is_protected(T05), "memoized answer is the same");
        assert_eq!(g.memo.borrow().len(), 2);
        assert!(IdGuard::build(&[LB], &BTreeSet::new(), &m).is_none());
    }

    #[test]
    fn collect_ids_finds_every_match_of_every_pattern() {
        let patterns = [
            IdPattern::new(
                "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            )
            .unwrap(),
            IdPattern::new(r"\b(GOV|APP)-[0-9]{4}\.[0-9]+\b").unwrap(),
        ];
        let text =
            format!("see {FULL} and GOV-2026.7 (not APP-1.2) and {FULL}");
        let mut out = BTreeSet::new();
        collect_ids(&patterns, &text, &mut out);
        assert_eq!(out, ids(&[FULL, "GOV-2026.7"]));
    }

    #[test]
    fn id_pattern_compares_by_source() {
        let a: IdPattern = "a+".parse().unwrap();
        let b = IdPattern::new("a+").unwrap();
        assert_eq!(a, b);
        assert_ne!(a, IdPattern::new("b+").unwrap());
        assert!(IdPattern::new("[").is_err());
    }
}
