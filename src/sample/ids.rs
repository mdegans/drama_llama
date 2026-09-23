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
//!    bytes, appended to a copy in progress (a suffix of the history
//!    that begins at a word start), are a prefix of a known id.
//!
//! Only faithful copies are exempt: a string that is a prefix of no
//! known id keeps its full penalty, so the exemption steers toward the
//! ids the context actually holds. A bare prefix needs no pattern of
//! its own — it is a prefix of the full id by construction. An id may
//! contain spaces (`September 22, 2026`): a copy is judged against the
//! ids, not cut at the first boundary. Nothing here is sampler *state*:
//! the copies in progress are derived from the token history at every
//! step, so snapshot and restore are unaffected.

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

/// Bytes a *word* consists of. An id copy may begin only at a word
/// start — after a byte outside this class, or at the start of the
/// history — so `cafe` inside `deadbeef-cafe` never starts a copy of
/// an id that begins `cafe`. Ids themselves may contain any byte
/// (`Sept. 7`, `September 22, 2026`): the class decides only where a
/// copy can begin, never where it must end.
const fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')
}

/// Longer than any identifier worth protecting: the history walk never
/// looks further back than this, nor further than the longest known id.
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

/// The last `cap` bytes of `tokens`' pieces, walking back from the end,
/// and whether that is the *whole* history (so its first byte is a word
/// start rather than an arbitrary cut). Typically a handful of pieces.
fn trailing_bytes<M: Model>(
    tokens: &[Token],
    model: &M,
    cap: usize,
    scratch: &mut Vec<u8>,
) -> (Vec<u8>, bool) {
    // Collected reversed, then flipped once.
    let mut rev: Vec<u8> = Vec::with_capacity(cap);
    for &token in tokens.iter().rev() {
        piece_into(model, token, scratch);
        for &b in scratch.iter().rev() {
            if rev.len() == cap {
                rev.reverse();
                return (rev, false);
            }
            rev.push(b);
        }
    }
    rev.reverse();
    (rev, true)
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
/// A copy in progress is a *suffix* of the history that begins at a
/// word start and is a prefix of some known id. There may be several at
/// once — after `Sept. 7`, both a known `Sept. 7` and a known `7 days`
/// are live. A token is exempt
/// iff its piece extends one of them, or starts a fresh copy at a word
/// start inside the piece (` 05` after `post_id:`). Because a copy is
/// judged against the ids rather than cut at the first space, a
/// multi-word id (`Sept. 7`, `September 22, 2026`) is exempt across its
/// spaces (#113).
///
/// Documented edges: at a word start, *any* token whose piece starts
/// some known id is exempt — with dozens of UUIDs in context that is
/// every space-prefixed hex digit and short hex word (` 1`, ` be`,
/// ` bad`) as the *first* token of a word; the next token is judged
/// against the now-longer copy and re-penalized. Bounded, not zero. A
/// merged piece that completes an id and crosses into non-word bytes
/// (`9d]`, `7,`) is exempt only if the id is completed exactly.
pub(crate) struct IdGuard<'a, M: Model> {
    /// The tail of the history, at most as long as the longest id.
    tail: Vec<u8>,
    /// Offsets into `tail` where a live copy begins: `tail[s..]` is a
    /// word start and a prefix of some id, or empty at a word start
    /// (any id may begin here). Empty ⇒ only a piece containing a word
    /// start can be exempt — the per-step early-out.
    starts: Vec<usize>,
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
        let longest = ids.iter().map(Vec::len).max()?;
        let mut scratch = Vec::with_capacity(model.max_token_len());
        let (tail, whole) =
            trailing_bytes(tokens, model, longest.min(TAIL_CAP), &mut scratch);
        let starts = (0..=tail.len())
            .filter(|&s| match s {
                0 => whole,
                s => !is_word_byte(tail[s - 1]),
            })
            .filter(|&s| s == tail.len() || has_prefix(ids, &tail[s..]))
            .collect();
        Some(Self {
            tail,
            starts,
            ids,
            model,
            memo: RefCell::new(BTreeMap::new()),
            scratch: RefCell::new(scratch),
        })
    }

    /// Would `piece`, emitted next, be a faithful copy of a known id?
    fn faithful(&self, piece: &[u8]) -> bool {
        let mut word = Vec::with_capacity(TAIL_CAP + piece.len());
        // Extend a copy already in progress (or begin one at the
        // history's trailing word start).
        self.starts
            .iter()
            .any(|&s| self.extends(&self.tail[s..], piece, &mut word))
            // Begin a copy at a word start inside the piece.
            || (1..piece.len())
                .filter(|&j| !is_word_byte(piece[j - 1]))
                .any(|j| self.extends(&[], &piece[j..], &mut word))
    }

    /// `head ++ piece` is a prefix of a known id, or completes one
    /// exactly and then crosses only non-word bytes (`9d]`, `7,`). The
    /// piece must contribute to the id itself: a bare `]` after a
    /// complete id completes nothing.
    fn extends(&self, head: &[u8], piece: &[u8], word: &mut Vec<u8>) -> bool {
        word.clear();
        word.extend_from_slice(head);
        word.extend_from_slice(piece);
        if has_prefix(self.ids, word) {
            return true;
        }
        let core = word
            .iter()
            .rposition(|&b| is_word_byte(b))
            .map_or(0, |i| i + 1);
        core < word.len()
            && core > head.len()
            && self.ids.contains(&word[..core])
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
        "",      // 0
        " ",     // 1
        "[",     // 2
        "05",    // 3
        "676",   // 4
        "b9d",   // 5
        "-",     // 6
        "8aa7",  // 7
        "c9",    // 8
        " 05",   // 9  space-prefixed BPE token
        "9d]",   // 10 completes-and-crosses
        "9e]",   // 11 wrong completion
        "x",     // 12 an id byte that is not an id
        "é",     // 13 non-ASCII: a boundary
        "Sept",  // 14
        ".",     // 15
        " 7",    // 16 space-prefixed digit (merging tokenizers)
        " 8",    // 17
        "7",     // 18 bare digit (digit-splitting tokenizers)
        "7,",    // 19 completes-and-crosses
        "hello", // 20
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
    const SEPT: Token = 14;
    const DOT: Token = 15;
    const SP7: Token = 16;
    const SP8: Token = 17;
    const T7: Token = 18;
    const T7_COMMA: Token = 19;
    const HELLO: Token = 20;

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

    /// The copies in progress for `toks`, as strings.
    fn live(toks: &[Token], set: &BTreeSet<Vec<u8>>) -> Vec<String> {
        let m = Pieces(P);
        let g = IdGuard::build(toks, set, &m).unwrap();
        g.starts
            .iter()
            .map(|&s| String::from_utf8_lossy(&g.tail[s..]).into_owned())
            .collect()
    }

    #[test]
    fn trailing_bytes_stop_at_the_cap() {
        let m = Pieces(P);
        let mut s = Vec::new();
        let t = |toks: &[Token], cap, s: &mut Vec<u8>| {
            trailing_bytes(toks, &m, cap, s)
        };
        assert_eq!(t(&[], 8, &mut s), (vec![], true));
        assert_eq!(t(&[LB, T05], 8, &mut s), (b"[05".to_vec(), true));
        assert_eq!(t(&[LB, T05], 3, &mut s), (b"[05".to_vec(), true));
        // A cut mid-piece keeps the last `cap` bytes and is not whole.
        assert_eq!(t(&[LB, T05, T676], 4, &mut s), (b"5676".to_vec(), false));
    }

    #[test]
    fn copies_in_progress_begin_at_word_starts() {
        let set = ids(&[FULL]);
        // Empty history: a word start, nothing else.
        assert_eq!(live(&[], &set), [""]);
        assert_eq!(live(&[LB], &set), [""]);
        assert_eq!(live(&[LB, T05], &set), ["05"]);
        assert_eq!(live(&[LB, T05, T676, TB9D, DASH], &set), ["05676b9d-"]);
        // A word that is no id's prefix: nothing live.
        assert!(live(&[TX], &set).is_empty());
        assert!(live(&[LB, T05, T676, TC9], &set).is_empty());
        // Non-ASCII is a boundary: `676` begins a word, and is no prefix.
        assert!(live(&[T05, EACUTE, T676], &set).is_empty());
        // A multi-word id stays live across its space.
        let set = ids(&["Sept. 7"]);
        assert_eq!(live(&[SP, SEPT, DOT, SP], &set), ["Sept. ", ""]);
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
        assert!(g.starts.is_empty());
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

    /// #113: a known id containing a space is exempt across it — in
    /// both tokenizer shapes (` 7` merged; ` ` then `7` split) — and only
    /// as a copy: the wrong digit, or the right digit after unrelated
    /// prose, keeps its penalty.
    #[test]
    fn multi_word_id_is_exempt_across_its_space() {
        let m = Pieces(P);
        let set = ids(&["Sept. 7"]);
        let g = IdGuard::build(&[SP, SEPT, DOT], &set, &m).unwrap();
        assert!(g.is_protected(SP7), "Sept. + ` 7`");
        assert!(!g.is_protected(SP8), "Sept. 8 is not the id");
        let g = IdGuard::build(&[SP, SEPT, DOT, SP], &set, &m).unwrap();
        assert!(g.is_protected(T7), "Sept. ` ` + `7`");
        assert!(g.is_protected(T7_COMMA), "completes exactly, then `,`");
        // A fresh copy may begin at any word start — the documented
        // edge — so `Sept` is exempt here as the start of a new one.
        assert!(g.is_protected(SEPT));
        // The start of the id is exempt at a word start, as before.
        let g = IdGuard::build(&[SP], &set, &m).unwrap();
        assert!(g.is_protected(SEPT));
        // The digit alone is not: the word before it is no id's prefix.
        let g = IdGuard::build(&[HELLO], &set, &m).unwrap();
        assert!(!g.is_protected(SP7));
        let g = IdGuard::build(&[HELLO, SP], &set, &m).unwrap();
        assert!(!g.is_protected(T7));
    }

    /// Two copies live at once: the id-aware walk tries each.
    #[test]
    fn overlapping_copies_are_each_tried() {
        let m = Pieces(P);
        let set = ids(&["Sept. 7", "7 Sept"]);
        // After `Sept. 7`, a fresh `7 Sept` is live from the `7`.
        let g = IdGuard::build(&[SP, SEPT, DOT, SP7, SP], &set, &m).unwrap();
        assert!(g.is_protected(SEPT), "7 + ` ` + Sept");
        assert!(!g.is_protected(SP8));
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

    /// The Agora sidecar's patterns (`models/*.sampling.toml`, shapes
    /// from agora-agentkit's renderer) compile and pick out what an
    /// agent must copy — and the multi-word ones survive intact.
    #[test]
    fn agora_sidecar_patterns() {
        let patterns: Vec<IdPattern> = [
            "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            r"\b(?:GOV|APP|AMD|KEY|REC)-[0-9]{4}[-.][0-9]+\b",
            r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}(?:T[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]+)?)?(?:Z|[+-][0-9]{2}:[0-9]{2})?)?",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? [0-9]{1,2}(?:, [0-9]{4})?\b",
            r"\b[a-z]+(?:-[a-z]+)+\b",
        ]
        .iter()
        .map(|p| p.parse().unwrap())
        .collect();
        let text = format!(
            "**Today's date: 2026-09-23.** ion-alphawave posted {FULL} \
             at 2026-09-20T10:00:00Z in meta-governance; GOV-2026-0006 \
             and AMD-2026-0001 passed. The council sat on September 22, \
             2026 and again Sept. 7."
        );
        let mut out = BTreeSet::new();
        collect_ids(&patterns, &text, &mut out);
        assert_eq!(
            out,
            ids(&[
                FULL,
                "2026-09-23",
                "2026-09-20T10:00:00Z",
                "ion-alphawave",
                "meta-governance",
                "GOV-2026-0006",
                "AMD-2026-0001",
                "September 22, 2026",
                "Sept. 7",
            ])
        );
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
