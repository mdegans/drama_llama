//! GBNF-constrained sampling.
//!
//! Provides [`Grammar`] (a compiled GBNF grammar) and [`GrammarState`] (a
//! speculative matcher that validates a byte stream against the grammar one
//! UTF-8 codepoint at a time). Used by [`SamplingMode::Grammar`] to reject
//! model tokens whose bytes would violate the grammar.
//!
//! # GBNF surface syntax
//!
//! The dialect matches `llama.cpp`'s GBNF: `name ::= alt1 | alt2`, postfix
//! `*` / `+` / `?`, grouping with `( ... )`, string literals (`"foo"`), and
//! character classes (`[a-zA-Z_]`, with `^` at the start for negation).
//! Escape sequences: `\n`, `\t`, `\r`, `\\`, `\"`, `\'`, `\[`, `\]`, `\xNN`,
//! `\uNNNN`, `\UNNNNNNNN`. `.` matches any UTF-8 codepoint except newline.
//! Comments run from `#` to end-of-line. The start rule must be named
//! `root`.
//!
//! # Lifecycle
//!
//! Matches the [`crate::sample::json`] module exactly:
//!
//! * [`GrammarState::accepts_bytes`] — non-mutating; clones state and reports
//!   whether the bytes can extend the current match.
//! * [`GrammarState::advance_bytes`] — mutating; commits the bytes. Call
//!   exactly once per accepted token.
//!
//! # Grammar-violation fallback
//!
//! [`grammar_filter`] forces a single EOS candidate when zero tokens extend
//! the grammar. On success (the grammar reached an accept state), the parser
//! auto-resets so the next generation starts fresh. On violation, state is
//! preserved for inspection via [`GrammarState::stack_depth`].
//!
//! [`SamplingMode::Grammar`]: crate::SamplingMode::Grammar

use dashmap::DashMap;
use llama_cpp_sys_3::llama_token;
use rayon::prelude::*;

use crate::TokenData;
use rustc_hash::FxHashMap;
use tinyvec::{ArrayVec, TinyVec};

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

use crate::{model::token_to_piece_ref, Candidates, LlamaCppModel};

/// Inline-capacity of a single stack in the NFA simulation. Most grammars
/// keep call stacks under 4 deep; 8 covers nested alternation / repetition
/// without spilling to the heap.
const STACK_INLINE: usize = 8;

type Stack = TinyVec<[Position; STACK_INLINE]>;

// ===========================================================================
// Compiled grammar
// ===========================================================================

/// A compiled GBNF grammar.
///
/// Parse via [`Grammar::parse`] or [`Grammar::from_file`]. The compiled form
/// is immutable and cheap to share across [`GrammarState`] clones via `Arc`.
#[derive(Clone, Debug, PartialEq)]
pub struct Grammar {
    rules: Vec<Rule>,
    root: usize,
    /// Original source, retained so that serde round-trips can re-parse.
    source: String,
}

#[derive(Clone, Debug, PartialEq)]
struct Rule {
    name: String,
    alts: Vec<Vec<Atom>>,
}

#[derive(Clone, Debug, PartialEq)]
enum Atom {
    RuleRef(usize),
    CharSet(CharSet),
}

/// A predicate over a single Unicode codepoint.
///
/// `ranges` holds inclusive `(lo, hi)` codepoint pairs. `negated` inverts
/// the match: a codepoint is accepted iff it is NOT in any range.
#[derive(Clone, Debug, PartialEq)]
struct CharSet {
    negated: bool,
    ranges: Vec<(u32, u32)>,
}

impl CharSet {
    fn contains(&self, cp: u32) -> bool {
        let hit = self.ranges.iter().any(|&(lo, hi)| cp >= lo && cp <= hi);
        hit ^ self.negated
    }
}

impl Grammar {
    /// Parse GBNF source text into a compiled grammar.
    pub fn parse(source: &str) -> Result<Self, GrammarError> {
        let mut builder = GrammarBuilder::new(source);
        builder.parse_document()?;
        builder.finish()
    }

    /// Parse GBNF from a file. The file contents are read as UTF-8.
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, GrammarError> {
        let source = std::fs::read_to_string(path.as_ref()).map_err(|err| {
            GrammarError::Io {
                path: path.as_ref().to_path_buf(),
                err: err.to_string(),
            }
        })?;
        Self::parse(&source)
    }

    /// Original GBNF source text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Number of rules in the compiled grammar, including anonymous rules
    /// introduced for grouping and repetition. Useful for debugging.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ===========================================================================
// GBNF parser
// ===========================================================================

struct GrammarBuilder<'a> {
    src: &'a str,
    cursor: usize,
    /// Map from rule name to index in `rules`. Anonymous rules use names
    /// like `_anon_3`.
    name_to_idx: std::collections::HashMap<String, usize>,
    rules: Vec<Rule>,
    anon_counter: usize,
}

impl<'a> GrammarBuilder<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            cursor: 0,
            name_to_idx: std::collections::HashMap::new(),
            rules: Vec::new(),
            anon_counter: 0,
        }
    }

    fn finish(self) -> Result<Grammar, GrammarError> {
        // Check for any declared-but-empty rules: parse creates them on
        // reference before definition. If one is still empty, the name was
        // referenced but never defined.
        for rule in &self.rules {
            if rule.alts.is_empty() {
                return Err(GrammarError::UndefinedRule(rule.name.clone()));
            }
        }
        let root = self
            .name_to_idx
            .get("root")
            .copied()
            .ok_or(GrammarError::MissingRoot)?;
        Ok(Grammar {
            rules: self.rules,
            root,
            source: self.src.to_owned(),
        })
    }

    /// Look up a rule by name, creating an empty placeholder on first
    /// reference. The placeholder is filled in when the rule is defined.
    fn lookup_or_declare(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.name_to_idx.get(name) {
            return idx;
        }
        let idx = self.rules.len();
        self.rules.push(Rule {
            name: name.to_owned(),
            alts: Vec::new(),
        });
        self.name_to_idx.insert(name.to_owned(), idx);
        idx
    }

    fn next_anon_name(&mut self) -> String {
        let n = self.anon_counter;
        self.anon_counter += 1;
        format!("_anon_{n}")
    }

    // --- Character-level scanning ---

    fn peek(&self) -> Option<char> {
        self.src[self.cursor..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.cursor += c.len_utf8();
        Some(c)
    }

    fn eat(&mut self, ch: char) -> bool {
        if self.peek() == Some(ch) {
            self.cursor += ch.len_utf8();
            true
        } else {
            false
        }
    }

    /// Skip whitespace and `#` comments. Newlines are whitespace.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.bump();
                }
                Some('#') => {
                    while let Some(c) = self.peek() {
                        self.bump();
                        if c == '\n' {
                            break;
                        }
                    }
                }
                _ => return,
            }
        }
    }

    fn at_end(&self) -> bool {
        self.cursor >= self.src.len()
    }

    // --- Productions ---

    fn parse_document(&mut self) -> Result<(), GrammarError> {
        loop {
            self.skip_trivia();
            if self.at_end() {
                return Ok(());
            }
            self.parse_rule_definition()?;
        }
    }

    fn parse_rule_definition(&mut self) -> Result<(), GrammarError> {
        let name = self.parse_name()?;
        self.skip_trivia();
        if !(self.eat(':') && self.eat(':') && self.eat('=')) {
            return Err(GrammarError::Syntax {
                pos: self.cursor,
                msg: format!("expected `::=` after rule name `{name}`"),
            });
        }
        self.skip_trivia();
        let alts = self.parse_alternates()?;
        let idx = self.lookup_or_declare(&name);
        if !self.rules[idx].alts.is_empty() {
            return Err(GrammarError::Syntax {
                pos: self.cursor,
                msg: format!("rule `{name}` is defined more than once"),
            });
        }
        self.rules[idx].alts = alts;
        Ok(())
    }

    /// Parse one or more `|`-separated sequences.
    fn parse_alternates(&mut self) -> Result<Vec<Vec<Atom>>, GrammarError> {
        let mut alts = vec![self.parse_sequence()?];
        loop {
            self.skip_trivia();
            if !self.eat('|') {
                break;
            }
            self.skip_trivia();
            alts.push(self.parse_sequence()?);
        }
        Ok(alts)
    }

    /// Parse a sequence of atoms (terminated by `|`, `)`, or end-of-rule).
    fn parse_sequence(&mut self) -> Result<Vec<Atom>, GrammarError> {
        let mut seq = Vec::new();
        loop {
            self.skip_trivia_inline();
            match self.peek() {
                None => break,
                Some('|') | Some(')') => break,
                // A newline followed by a new rule ends the current rule.
                Some(c) if c == '\n' || c == '\r' => {
                    // Peek past trivia to see if we hit a new rule or EOF.
                    let save = self.cursor;
                    self.skip_trivia();
                    if self.at_end() || self.looks_like_new_rule() {
                        self.cursor = save;
                        break;
                    }
                    // Otherwise it was continued whitespace; keep going.
                }
                _ => {}
            }
            let atom = self.parse_atom()?;
            seq.push(atom);
        }
        Ok(seq)
    }

    /// Skip horizontal whitespace and line-continuation comments, but NOT
    /// newlines — newlines may end a rule.
    fn skip_trivia_inline(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c == ' ' || c == '\t' => {
                    self.bump();
                }
                Some('#') => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                _ => return,
            }
        }
    }

    /// Lookahead check: does the cursor currently point at `name ::=`?
    fn looks_like_new_rule(&self) -> bool {
        let rest = &self.src[self.cursor..];
        let mut chars = rest.char_indices();
        // Skip any whitespace/comments first.
        let mut i = 0;
        while let Some((idx, c)) = chars.clone().next() {
            if c.is_whitespace() {
                chars.next();
                i = idx + c.len_utf8();
            } else if c == '#' {
                while let Some((_, c2)) = chars.next() {
                    if c2 == '\n' {
                        break;
                    }
                }
                // After consuming the comment, re-enter the loop.
                i = rest.len() - chars.clone().as_str().len();
                continue;
            } else {
                break;
            }
        }
        let rest = &rest[i..];
        let mut j = 0;
        for c in rest.chars() {
            if is_name_char(c) {
                j += c.len_utf8();
            } else {
                break;
            }
        }
        if j == 0 {
            return false;
        }
        let after =
            &rest[j..].trim_start_matches(|c: char| c == ' ' || c == '\t');
        after.starts_with("::=")
    }

    fn parse_atom(&mut self) -> Result<Atom, GrammarError> {
        self.skip_trivia_inline();
        let start = self.cursor;
        let base = match self.peek() {
            Some('"') => self.parse_string_atom()?,
            Some('[') => Atom::CharSet(self.parse_char_class()?),
            Some('(') => {
                self.bump();
                self.skip_trivia();
                let alts = self.parse_alternates()?;
                self.skip_trivia();
                if !self.eat(')') {
                    return Err(GrammarError::Syntax {
                        pos: self.cursor,
                        msg: "expected `)` to close group".into(),
                    });
                }
                let anon = self.next_anon_name();
                let idx = self.lookup_or_declare(&anon);
                self.rules[idx].alts = alts;
                Atom::RuleRef(idx)
            }
            Some('.') => {
                self.bump();
                // `.` = any codepoint except newline.
                Atom::CharSet(CharSet {
                    negated: true,
                    ranges: vec![(b'\n' as u32, b'\n' as u32)],
                })
            }
            Some(c) if is_name_start(c) => {
                let name = self.parse_name()?;
                let idx = self.lookup_or_declare(&name);
                Atom::RuleRef(idx)
            }
            Some(c) => {
                return Err(GrammarError::Syntax {
                    pos: start,
                    msg: format!("unexpected character `{c}` in atom"),
                });
            }
            None => {
                return Err(GrammarError::Syntax {
                    pos: start,
                    msg: "unexpected end of input in atom".into(),
                });
            }
        };

        // Postfix operators `*`, `+`, `?` desugar into anonymous rules.
        self.skip_trivia_inline();
        match self.peek() {
            Some('*') => {
                self.bump();
                Ok(self.make_star(base))
            }
            Some('+') => {
                self.bump();
                Ok(self.make_plus(base))
            }
            Some('?') => {
                self.bump();
                Ok(self.make_opt(base))
            }
            _ => Ok(base),
        }
    }

    /// `X*` → anonymous rule `_: ::= | X _`
    fn make_star(&mut self, inner: Atom) -> Atom {
        let name = self.next_anon_name();
        let idx = self.lookup_or_declare(&name);
        let self_ref = Atom::RuleRef(idx);
        self.rules[idx].alts = vec![vec![], vec![inner, self_ref]];
        Atom::RuleRef(idx)
    }

    /// `X+` → anonymous rule `_: ::= X | X _`
    fn make_plus(&mut self, inner: Atom) -> Atom {
        let name = self.next_anon_name();
        let idx = self.lookup_or_declare(&name);
        let self_ref = Atom::RuleRef(idx);
        self.rules[idx].alts = vec![vec![inner.clone()], vec![inner, self_ref]];
        Atom::RuleRef(idx)
    }

    /// `X?` → anonymous rule `_: ::= | X`
    fn make_opt(&mut self, inner: Atom) -> Atom {
        let name = self.next_anon_name();
        let idx = self.lookup_or_declare(&name);
        self.rules[idx].alts = vec![vec![], vec![inner]];
        Atom::RuleRef(idx)
    }

    /// Parse a string literal like `"foo"` into a sequence of single-char
    /// CharSets wrapped in an anonymous concatenation rule. Returns a
    /// RuleRef. Empty strings produce a rule with a single empty alt.
    fn parse_string_atom(&mut self) -> Result<Atom, GrammarError> {
        if !self.eat('"') {
            return Err(GrammarError::Syntax {
                pos: self.cursor,
                msg: "expected `\"` to start string literal".into(),
            });
        }
        let mut chars: Vec<u32> = Vec::new();
        loop {
            match self.peek() {
                None => {
                    return Err(GrammarError::Syntax {
                        pos: self.cursor,
                        msg: "unterminated string literal".into(),
                    })
                }
                Some('"') => {
                    self.bump();
                    break;
                }
                Some('\\') => {
                    self.bump();
                    chars.push(self.parse_escape()?);
                }
                Some(c) => {
                    self.bump();
                    chars.push(c as u32);
                }
            }
        }
        let atoms: Vec<Atom> = chars
            .into_iter()
            .map(|cp| {
                Atom::CharSet(CharSet {
                    negated: false,
                    ranges: vec![(cp, cp)],
                })
            })
            .collect();
        // Wrap in anonymous rule for uniform RuleRef handling. An empty
        // string literal compiles to an always-matching rule with one empty
        // alternative.
        let name = self.next_anon_name();
        let idx = self.lookup_or_declare(&name);
        self.rules[idx].alts = vec![atoms];
        Ok(Atom::RuleRef(idx))
    }

    fn parse_char_class(&mut self) -> Result<CharSet, GrammarError> {
        if !self.eat('[') {
            return Err(GrammarError::Syntax {
                pos: self.cursor,
                msg: "expected `[` to start char class".into(),
            });
        }
        let negated = self.eat('^');
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        while self.peek() != Some(']') {
            let lo = self.parse_class_char()?;
            let hi = if self.peek() == Some('-') {
                // Peek ahead to distinguish `a-z` from a trailing `-`.
                let save = self.cursor;
                self.bump();
                if self.peek() == Some(']') {
                    // Trailing `-`: treat as a literal dash.
                    self.cursor = save;
                    lo
                } else {
                    self.parse_class_char()?
                }
            } else {
                lo
            };
            if lo > hi {
                return Err(GrammarError::Syntax {
                    pos: self.cursor,
                    msg: format!(
                        "char class range {lo:#x}-{hi:#x} is inverted"
                    ),
                });
            }
            ranges.push((lo, hi));
            if self.peek().is_none() {
                return Err(GrammarError::Syntax {
                    pos: self.cursor,
                    msg: "unterminated char class".into(),
                });
            }
        }
        if !self.eat(']') {
            return Err(GrammarError::Syntax {
                pos: self.cursor,
                msg: "expected `]` to close char class".into(),
            });
        }
        Ok(CharSet { negated, ranges })
    }

    fn parse_class_char(&mut self) -> Result<u32, GrammarError> {
        match self.peek() {
            Some('\\') => {
                self.bump();
                self.parse_escape()
            }
            Some(']') | None => Err(GrammarError::Syntax {
                pos: self.cursor,
                msg: "expected character inside char class".into(),
            }),
            Some(c) => {
                self.bump();
                Ok(c as u32)
            }
        }
    }

    fn parse_escape(&mut self) -> Result<u32, GrammarError> {
        match self.bump() {
            Some('n') => Ok(b'\n' as u32),
            Some('t') => Ok(b'\t' as u32),
            Some('r') => Ok(b'\r' as u32),
            Some('\\') => Ok(b'\\' as u32),
            Some('"') => Ok(b'"' as u32),
            Some('\'') => Ok(b'\'' as u32),
            Some('[') => Ok(b'[' as u32),
            Some(']') => Ok(b']' as u32),
            Some('-') => Ok(b'-' as u32),
            Some('x') => self.parse_hex(2),
            Some('u') => self.parse_hex(4),
            Some('U') => self.parse_hex(8),
            Some(c) => Err(GrammarError::Syntax {
                pos: self.cursor,
                msg: format!("unknown escape `\\{c}`"),
            }),
            None => Err(GrammarError::Syntax {
                pos: self.cursor,
                msg: "unterminated escape".into(),
            }),
        }
    }

    fn parse_hex(&mut self, n: usize) -> Result<u32, GrammarError> {
        let mut value: u32 = 0;
        for _ in 0..n {
            match self.bump() {
                Some(c) if c.is_ascii_hexdigit() => {
                    value = (value << 4) | c.to_digit(16).unwrap();
                }
                _ => {
                    return Err(GrammarError::Syntax {
                        pos: self.cursor,
                        msg: format!("expected {n}-digit hex escape"),
                    })
                }
            }
        }
        Ok(value)
    }

    fn parse_name(&mut self) -> Result<String, GrammarError> {
        self.skip_trivia_inline();
        let start = self.cursor;
        match self.peek() {
            Some(c) if is_name_start(c) => {
                self.bump();
            }
            _ => {
                return Err(GrammarError::Syntax {
                    pos: start,
                    msg: "expected rule name".into(),
                })
            }
        }
        while let Some(c) = self.peek() {
            if is_name_char(c) {
                self.bump();
            } else {
                break;
            }
        }
        Ok(self.src[start..self.cursor].to_owned())
    }
}

fn is_name_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

// ===========================================================================
// Matcher
// ===========================================================================

/// Position within the element stream: which alt of which rule, and how
/// far through it.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Position {
    rule_idx: u32,
    alt_idx: u32,
    atom_idx: u32,
}

/// Matcher state *without* the grammar reference.
///
/// Holds only the mutable simulation bits (active stacks + pending UTF-8
/// buffer). Splitting this out of [`GrammarState`] lets the hot filter
/// loop clone matcher state without bumping the `Arc<Grammar>`
/// refcount per candidate — 150k atomic ops per decode step was a real
/// cost in profiles.
///
/// Each element of `stacks` is a call stack: the innermost frame is the
/// rule currently being walked; popping returns to the caller. Multiple
/// stacks coexist because GBNF rules branch on alternation. `Stack` is a
/// [`TinyVec`] so typical-depth stacks stay inline (no per-clone heap
/// allocation).
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StackState {
    stacks: Vec<Stack>,
    pending: ArrayVec<[u8; 4]>,
}

/// Active matching state for a [`Grammar`].
///
/// Thin wrapper: owns the `Arc<Grammar>` plus the mutable [`StackState`].
/// All matcher methods delegate into [`StackState`] with a borrowed
/// `&Grammar`. See [`StackState`] for why.
///
/// Clone cost is proportional to `stacks.len() * avg_stack_depth`, which is
/// small for practical grammars. `accepts_bytes` relies on cloning for
/// speculative simulation.
///
/// Each `GrammarState` also owns an `Arc<DfaCache>` — a lazily-populated
/// memoization layer over NFA states and one-byte transitions. `Clone` on
/// `GrammarState` shares the same cache; independently constructed
/// `GrammarState`s build independent caches.
#[derive(Clone, Debug)]
pub struct GrammarState {
    grammar: Arc<Grammar>,
    inner: StackState,
    cache: Arc<DfaCache>,
}

/// Equality ignores the lazy-DFA cache: it is a pure acceleration structure
/// and two states with identical matcher contents are semantically equal
/// regardless of which cache instance happens to be attached.
impl PartialEq for GrammarState {
    fn eq(&self, other: &Self) -> bool {
        self.grammar == other.grammar && self.inner == other.inner
    }
}

impl GrammarState {
    /// Construct a fresh matcher rooted at the grammar's `root` rule.
    pub fn new(grammar: Arc<Grammar>) -> Self {
        let inner = StackState::new_rooted(&grammar);
        Self {
            grammar,
            inner,
            cache: Arc::new(DfaCache::new()),
        }
    }

    /// Reset to the fresh starting state.
    pub fn reset(&mut self) {
        self.inner.reset(&self.grammar);
    }

    /// True iff the matcher has reached an accepting state AND no partial
    /// UTF-8 codepoint is buffered.
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Current number of active stacks. Useful for UI status / debugging.
    pub fn stack_depth(&self) -> usize {
        self.inner.stacks.len()
    }

    /// Borrow the underlying grammar.
    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    /// True iff feeding `bytes` would succeed from the current state. Does
    /// not mutate `self`.
    pub fn accepts_bytes(&self, bytes: &[u8]) -> bool {
        self.inner.accepts_bytes(&self.grammar, bytes)
    }

    /// 256-bit bitmap indexed by byte value: bit `b` is set iff feeding
    /// byte `b` next could plausibly extend the match into a codepoint
    /// accepted by at least one active stack. See
    /// [`StackState::first_byte_bitmap`] for details.
    pub(crate) fn first_byte_bitmap(&self) -> [u64; 4] {
        self.inner.first_byte_bitmap(&self.grammar)
    }

    /// Commit `bytes` to the matcher. Call after sampling selects a token.
    pub fn advance_bytes(&mut self, bytes: &[u8]) -> Result<(), GrammarError> {
        self.inner.advance_bytes(&self.grammar, bytes)
    }
}

impl StackState {
    /// Fresh state rooted at `grammar.root`.
    fn new_rooted(grammar: &Grammar) -> Self {
        let mut state = Self {
            stacks: Vec::new(),
            pending: ArrayVec::new(),
        };
        state.reset(grammar);
        state
    }

    fn reset(&mut self, grammar: &Grammar) {
        self.pending.clear();
        let root = grammar.root as u32;
        let root_rule = &grammar.rules[grammar.root];
        self.stacks = (0..root_rule.alts.len())
            .map(|alt_idx| {
                let mut s: Stack = TinyVec::new();
                s.push(Position {
                    rule_idx: root,
                    alt_idx: alt_idx as u32,
                    atom_idx: 0,
                });
                s
            })
            .collect();
        self.expand(grammar);
    }

    fn is_complete(&self) -> bool {
        self.pending.is_empty() && self.stacks.iter().any(|s| s.is_empty())
    }

    /// True iff feeding `bytes` would succeed from the current state.
    /// Clones only the matcher state — the `Arc<Grammar>` is not touched.
    fn accepts_bytes(&self, grammar: &Grammar, bytes: &[u8]) -> bool {
        let mut scratch = self.clone();
        scratch.advance_bytes(grammar, bytes).is_ok()
    }

    /// Conservative 256-bit bitmap of which first byte values could plausibly
    /// extend the match from the current state. Set bit ⇒ "maybe accepted"
    /// (still needs full `accepts_bytes` confirmation); cleared bit ⇒
    /// "definitely rejected". See [`grammar_filter`] for how it's used.
    pub(crate) fn first_byte_bitmap(&self, grammar: &Grammar) -> [u64; 4] {
        let mut bitmap = [0u64; 4];
        let pending_len = self.pending.len();
        if pending_len >= 4 {
            return bitmap;
        }
        let mut hyp: [u8; 4] = [0; 4];
        hyp[..pending_len].copy_from_slice(self.pending.as_slice());
        for b in 0u8..=0xFFu8 {
            hyp[pending_len] = b;
            let Some((lo, hi)) =
                pending_codepoint_range(&hyp[..pending_len + 1])
            else {
                continue;
            };
            if self.any_stack_top_intersects(grammar, lo, hi) {
                bitmap[(b as usize) >> 6] |= 1u64 << (b & 63);
            }
        }
        bitmap
    }

    fn any_stack_top_intersects(
        &self,
        grammar: &Grammar,
        lo: u32,
        hi: u32,
    ) -> bool {
        for stack in &self.stacks {
            let Some(pos) = stack.last() else {
                continue;
            };
            let atoms = &grammar.rules[pos.rule_idx as usize].alts
                [pos.alt_idx as usize];
            let Some(Atom::CharSet(cs)) = atoms.get(pos.atom_idx as usize)
            else {
                continue;
            };
            if charset_intersects(cs, lo, hi) {
                return true;
            }
        }
        false
    }

    fn advance_bytes(
        &mut self,
        grammar: &Grammar,
        bytes: &[u8],
    ) -> Result<(), GrammarError> {
        for &b in bytes {
            self.feed_byte(grammar, b)?;
        }
        if !self.pending.is_empty()
            && !self.pending_can_still_match(grammar)
        {
            return Err(GrammarError::InvalidUtf8);
        }
        Ok(())
    }

    fn pending_can_still_match(&self, grammar: &Grammar) -> bool {
        let (lo, hi) = match pending_codepoint_range(self.pending.as_slice()) {
            Some(range) => range,
            None => return false,
        };
        self.any_stack_top_intersects(grammar, lo, hi)
    }

    fn feed_byte(
        &mut self,
        grammar: &Grammar,
        b: u8,
    ) -> Result<(), GrammarError> {
        self.pending.push(b);
        match decode_utf8(self.pending.as_slice()) {
            Utf8Decode::Complete(cp) => {
                self.pending.clear();
                self.consume(grammar, cp)
            }
            Utf8Decode::Incomplete => Ok(()),
            Utf8Decode::Invalid => Err(GrammarError::InvalidUtf8),
        }
    }

    fn consume(
        &mut self,
        grammar: &Grammar,
        cp: u32,
    ) -> Result<(), GrammarError> {
        let mut next: Vec<Stack> = Vec::with_capacity(self.stacks.len());
        for stack in self.stacks.drain(..) {
            let Some(pos) = stack.last() else {
                continue;
            };
            let atoms = &grammar.rules[pos.rule_idx as usize].alts
                [pos.alt_idx as usize];
            let Some(atom) = atoms.get(pos.atom_idx as usize) else {
                continue;
            };
            let Atom::CharSet(cs) = atom else {
                continue;
            };
            if cs.contains(cp) {
                let mut advanced = stack;
                let top = advanced.last_mut().unwrap();
                top.atom_idx += 1;
                next.push(advanced);
            }
        }
        self.stacks = next;
        self.expand(grammar);
        if self.stacks.is_empty() {
            return Err(GrammarError::NoMatch(cp));
        }
        Ok(())
    }

    /// Walk all epsilon transitions until every stack is either empty
    /// (accepting) or has a CharSet at the top.
    fn expand(&mut self, grammar: &Grammar) {
        // Fast path: every stack is already at a CharSet yield point (no
        // alt-complete pops to resolve, no RuleRef to open). No allocation,
        // no dedup needed — the incoming stacks were already deduped by
        // the prior expand call.
        let all_yield = self.stacks.iter().all(|stack| {
            let Some(pos) = stack.last() else {
                // Accepted stacks count as already at a yield point.
                return true;
            };
            let atoms = &grammar.rules[pos.rule_idx as usize].alts
                [pos.alt_idx as usize];
            matches!(atoms.get(pos.atom_idx as usize), Some(Atom::CharSet(_)))
        });
        if all_yield {
            return;
        }

        let mut queue: Vec<Stack> = std::mem::take(&mut self.stacks);
        let mut result: Vec<Stack> = Vec::with_capacity(queue.len());
        // Bound iterations to guard against pathological left-recursive
        // grammars. 4096 * initial stack count is ample for sane GBNF.
        let budget = 4096usize.saturating_mul(queue.len().max(1));
        for _ in 0..budget {
            let Some(mut stack) = queue.pop() else {
                break;
            };
            let Some(pos) = stack.last().copied() else {
                result.push(stack);
                continue;
            };
            let alts = &grammar.rules[pos.rule_idx as usize].alts;
            let alt = &alts[pos.alt_idx as usize];
            if pos.atom_idx as usize == alt.len() {
                stack.pop();
                if let Some(caller) = stack.last_mut() {
                    caller.atom_idx += 1;
                }
                queue.push(stack);
                continue;
            }
            match &alt[pos.atom_idx as usize] {
                Atom::CharSet(_) => {
                    result.push(stack);
                }
                Atom::RuleRef(r) => {
                    // Tail-call optimization: if this RuleRef is the
                    // last atom of the enclosing alt, replace the
                    // current frame with the sub-rule's frame instead
                    // of pushing. Semantically identical — when the
                    // sub-rule completes, it pops back to the original
                    // caller either way — but keeps right-recursive
                    // rules like `.+ ::= . | . _anon` bounded in depth.
                    // Without TCO, every consumed codepoint grows a
                    // stack by one frame.
                    let is_tail = pos.atom_idx as usize + 1 == alt.len();
                    let sub_alts = &grammar.rules[*r].alts;
                    for (a_idx, _) in sub_alts.iter().enumerate() {
                        let new_pos = Position {
                            rule_idx: *r as u32,
                            alt_idx: a_idx as u32,
                            atom_idx: 0,
                        };
                        let mut branched = stack.clone();
                        if is_tail {
                            *branched.last_mut().unwrap() = new_pos;
                        } else {
                            branched.push(new_pos);
                        }
                        queue.push(branched);
                    }
                }
            }
        }
        // Dedupe: identical stacks are redundant work. The NFA simulation
        // can otherwise explode on deeply nested alternations. stdlib's
        // sort short-circuits on len < 2.
        result.sort();
        result.dedup();
        self.stacks = result;
    }
}

// ===========================================================================
// Lazy-DFA cache
// ===========================================================================

/// Interned identifier for a canonical [`StackState`]. Returned by
/// [`DfaCache::intern`] and used as the key into the byte-transition table.
pub(crate) type StateId = u32;

/// Sentinel returned by [`DfaCache::transition`] when feeding the byte leaves
/// the matcher with no surviving stacks (i.e. the byte is rejected).
pub(crate) const REJECT_STATE: StateId = u32::MAX;

struct DfaInterned {
    /// Canonical `StackState` → `StateId`. Canonical = post-`expand`, so
    /// stacks are sorted + deduped.
    intern: FxHashMap<StackState, StateId>,
    /// Id → canonical `StackState`. Needed on transition misses to
    /// reconstitute the matcher, feed a byte, and re-intern the result.
    states: Vec<StackState>,
}

/// Lazy-DFA memoization layer over the NFA matcher.
///
/// The underlying matcher is still an NFA (multiple concurrent call stacks).
/// The cache interns canonical `StackState`s into compact `StateId`s and
/// memoizes one-byte transitions, so revisits of the same matcher state hit
/// a table lookup instead of rerunning the full `feed_byte` + `expand`
/// pipeline. First visits pay the normal walk cost plus a canonicalize +
/// insert.
///
/// Shared across clones of [`GrammarState`] via `Arc`.
pub(crate) struct DfaCache {
    interned: RwLock<DfaInterned>,
    /// `(state, byte)` → next state. `DashMap` for lock-striped access under
    /// the rayon fold in `grammar_filter`.
    transitions: DashMap<(StateId, u8), StateId>,
    /// Per-state first-byte acceptance bitmap. Lazily filled.
    bitmaps: DashMap<StateId, [u64; 4]>,
    /// Per-state "is this an accepting / complete state" cache.
    complete: DashMap<StateId, bool>,
    /// Per-state "would this state be valid at end-of-stream" cache — mirrors
    /// the trailing [`StackState::pending_can_still_match`] check done at the
    /// end of [`StackState::advance_bytes`].
    terminal_valid: DashMap<StateId, bool>,
    transition_hits: AtomicU64,
    transition_misses: AtomicU64,
    bitmap_hits: AtomicU64,
    bitmap_misses: AtomicU64,
}

impl DfaCache {
    pub(crate) fn new() -> Self {
        Self {
            interned: RwLock::new(DfaInterned {
                intern: FxHashMap::default(),
                states: Vec::new(),
            }),
            transitions: DashMap::new(),
            bitmaps: DashMap::new(),
            complete: DashMap::new(),
            terminal_valid: DashMap::new(),
            transition_hits: AtomicU64::new(0),
            transition_misses: AtomicU64::new(0),
            bitmap_hits: AtomicU64::new(0),
            bitmap_misses: AtomicU64::new(0),
        }
    }

    /// Intern a canonical `StackState`, returning its `StateId`. Reads fast-
    /// path under a read lock; inserts on miss under a write lock with a
    /// double-check to tolerate racing inserters under rayon.
    fn intern(&self, state: &StackState) -> StateId {
        if let Some(&id) = self.interned.read().unwrap().intern.get(state) {
            return id;
        }
        let mut g = self.interned.write().unwrap();
        if let Some(&id) = g.intern.get(state) {
            return id;
        }
        let id = g.states.len() as StateId;
        debug_assert!(id != REJECT_STATE, "state id overflow");
        g.states.push(state.clone());
        g.intern.insert(state.clone(), id);
        id
    }

    /// Reconstitute the `StackState` for a given id. Used only on cache
    /// misses; the hot path never calls this.
    fn state_of(&self, id: StateId) -> StackState {
        self.interned.read().unwrap().states[id as usize].clone()
    }

    /// Feed a byte from a state, returning the next state id (or
    /// `REJECT_STATE`). Hit path is a single `DashMap::get`.
    pub(crate) fn transition(
        &self,
        grammar: &Grammar,
        sid: StateId,
        byte: u8,
    ) -> StateId {
        if sid == REJECT_STATE {
            return REJECT_STATE;
        }
        if let Some(entry) = self.transitions.get(&(sid, byte)) {
            self.transition_hits.fetch_add(1, Ordering::Relaxed);
            return *entry;
        }
        self.transition_misses.fetch_add(1, Ordering::Relaxed);
        let mut scratch = self.state_of(sid);
        let next_id = match scratch.feed_byte(grammar, byte) {
            Ok(()) => self.intern(&scratch),
            Err(_) => REJECT_STATE,
        };
        self.transitions.insert((sid, byte), next_id);
        next_id
    }

    /// True iff `sid` would satisfy the trailing check at end-of-input — i.e.
    /// any buffered partial UTF-8 codepoint can still extend into a matching
    /// codepoint. Mirrors the trailing check inside
    /// [`StackState::advance_bytes`].
    pub(crate) fn terminal_valid(
        &self,
        grammar: &Grammar,
        sid: StateId,
    ) -> bool {
        if sid == REJECT_STATE {
            return false;
        }
        if let Some(entry) = self.terminal_valid.get(&sid) {
            return *entry;
        }
        let state = self.state_of(sid);
        let v = state.pending.is_empty()
            || state.pending_can_still_match(grammar);
        self.terminal_valid.insert(sid, v);
        v
    }

    /// First-byte acceptance bitmap for a state. Lazily populated; subsequent
    /// calls hit the `DashMap`.
    pub(crate) fn first_byte_bitmap(
        &self,
        grammar: &Grammar,
        sid: StateId,
    ) -> [u64; 4] {
        if sid == REJECT_STATE {
            return [0u64; 4];
        }
        if let Some(entry) = self.bitmaps.get(&sid) {
            self.bitmap_hits.fetch_add(1, Ordering::Relaxed);
            return *entry;
        }
        self.bitmap_misses.fetch_add(1, Ordering::Relaxed);
        let state = self.state_of(sid);
        let bm = state.first_byte_bitmap(grammar);
        self.bitmaps.insert(sid, bm);
        bm
    }

    /// True iff the state is an accepting state (empty pending + at least one
    /// empty stack).
    pub(crate) fn is_complete(&self, sid: StateId) -> bool {
        if sid == REJECT_STATE {
            return false;
        }
        if let Some(entry) = self.complete.get(&sid) {
            return *entry;
        }
        let state = self.state_of(sid);
        let c = state.is_complete();
        self.complete.insert(sid, c);
        c
    }

    /// Number of distinct canonical states seen so far.
    pub(crate) fn state_count(&self) -> usize {
        self.interned.read().unwrap().states.len()
    }

    pub(crate) fn transition_hits(&self) -> u64 {
        self.transition_hits.load(Ordering::Relaxed)
    }

    pub(crate) fn transition_misses(&self) -> u64 {
        self.transition_misses.load(Ordering::Relaxed)
    }

    pub(crate) fn bitmap_hits(&self) -> u64 {
        self.bitmap_hits.load(Ordering::Relaxed)
    }

    pub(crate) fn bitmap_misses(&self) -> u64 {
        self.bitmap_misses.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for DfaCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DfaCache")
            .field("states", &self.state_count())
            .field("transitions", &self.transitions.len())
            .field("transition_hits", &self.transition_hits())
            .field("transition_misses", &self.transition_misses())
            .finish()
    }
}

/// Env-gated toggle: set `DRAMA_LLAMA_DFA_CACHE=0` to disable the lazy-DFA
/// cache and fall back to the per-candidate clone-and-walk path. Cached at
/// first access.
fn dfa_cache_enabled() -> bool {
    static DFA_ENABLED: OnceLock<bool> = OnceLock::new();
    *DFA_ENABLED.get_or_init(|| {
        std::env::var_os("DRAMA_LLAMA_DFA_CACHE")
            .map(|v| !(v == "0" || v.is_empty()))
            .unwrap_or(true)
    })
}

// ===========================================================================
// UTF-8 decoding
// ===========================================================================

enum Utf8Decode {
    Complete(u32),
    Incomplete,
    Invalid,
}

/// Compute the inclusive codepoint range that a partial UTF-8 prefix
/// could still complete into, respecting UTF-8 validity constraints
/// (no overlong encodings, no surrogates, no codepoints > U+10FFFF).
/// Returns `None` when no valid completion exists.
fn pending_codepoint_range(pending: &[u8]) -> Option<(u32, u32)> {
    let b0 = *pending.first()?;
    if b0 & 0x80 == 0 {
        return Some((b0 as u32, b0 as u32));
    }
    if b0 & 0xE0 == 0xC0 {
        if b0 < 0xC2 {
            return None; // overlong 2-byte
        }
        let hi5 = (b0 & 0x1F) as u32;
        if pending.len() == 1 {
            let lo = hi5 << 6;
            return Some((lo.max(0x80), lo | 0x3F));
        }
        let c1 = pending[1];
        if c1 & 0xC0 != 0x80 {
            return None;
        }
        let cp = (hi5 << 6) | (c1 & 0x3F) as u32;
        Some((cp, cp))
    } else if b0 & 0xF0 == 0xE0 {
        let hi4 = (b0 & 0x0F) as u32;
        let base = hi4 << 12;
        // E0 requires second byte >= 0xA0 to avoid overlong (U+0080..U+07FF
        // are 2-byte). ED requires second byte <= 0x9F to avoid surrogates
        // (U+D800..U+DFFF).
        let (c1_min, c1_max): (u8, u8) = match b0 {
            0xE0 => (0xA0, 0xBF),
            0xED => (0x80, 0x9F),
            _ => (0x80, 0xBF),
        };
        match pending.len() {
            1 => {
                let lo = base | ((c1_min & 0x3F) as u32) << 6;
                let hi = base | ((c1_max & 0x3F) as u32) << 6 | 0x3F;
                Some((lo, hi))
            }
            2 => {
                let c1 = pending[1];
                if c1 < c1_min || c1 > c1_max {
                    return None;
                }
                let mid = base | ((c1 & 0x3F) as u32) << 6;
                Some((mid, mid | 0x3F))
            }
            _ => {
                let c1 = pending[1];
                let c2 = pending[2];
                if c1 < c1_min || c1 > c1_max || c2 & 0xC0 != 0x80 {
                    return None;
                }
                let cp = base | ((c1 & 0x3F) as u32) << 6 | (c2 & 0x3F) as u32;
                Some((cp, cp))
            }
        }
    } else if b0 & 0xF8 == 0xF0 {
        if b0 > 0xF4 {
            return None;
        }
        let hi3 = (b0 & 0x07) as u32;
        let base = hi3 << 18;
        // F0 requires second byte >= 0x90 (U+10000 minimum). F4 requires
        // second byte <= 0x8F (U+10FFFF maximum).
        let (c1_min, c1_max): (u8, u8) = match b0 {
            0xF0 => (0x90, 0xBF),
            0xF4 => (0x80, 0x8F),
            _ => (0x80, 0xBF),
        };
        match pending.len() {
            1 => {
                let lo = base | ((c1_min & 0x3F) as u32) << 12;
                let hi = base | ((c1_max & 0x3F) as u32) << 12 | 0x0FFF;
                Some((lo, hi))
            }
            2 => {
                let c1 = pending[1];
                if c1 < c1_min || c1 > c1_max {
                    return None;
                }
                let mid = base | ((c1 & 0x3F) as u32) << 12;
                Some((mid, mid | 0x0FFF))
            }
            3 => {
                let c1 = pending[1];
                let c2 = pending[2];
                if c1 < c1_min || c1 > c1_max || c2 & 0xC0 != 0x80 {
                    return None;
                }
                let inner = base
                    | ((c1 & 0x3F) as u32) << 12
                    | ((c2 & 0x3F) as u32) << 6;
                Some((inner, inner | 0x3F))
            }
            _ => None,
        }
    } else {
        None
    }
}

/// True if `cs` accepts at least one codepoint in `[lo, hi]`.
fn charset_intersects(cs: &CharSet, lo: u32, hi: u32) -> bool {
    if cs.negated {
        // Negated accepts codepoints NOT in any range. If the query
        // [lo, hi] is not fully covered by union(ranges), there's a gap
        // that the negated set accepts.
        let mut cursor = lo;
        for &(r_lo, r_hi) in &cs.ranges {
            if r_hi < cursor {
                continue;
            }
            if r_lo > hi {
                break;
            }
            if r_lo > cursor {
                return true;
            }
            cursor = r_hi.saturating_add(1);
            if cursor > hi {
                return false;
            }
        }
        cursor <= hi
    } else {
        cs.ranges
            .iter()
            .any(|&(r_lo, r_hi)| r_hi >= lo && r_lo <= hi)
    }
}

/// Decode a buffer as UTF-8. Returns:
/// * `Complete(cp)` if `buf` is exactly one codepoint.
/// * `Incomplete` if `buf` is a valid prefix of a multi-byte codepoint.
/// * `Invalid` otherwise — including lead bytes that are NEVER valid
///   (`0xC0`, `0xC1`, `0xF5..=0xFF`) or always-overlong encodings. This
///   matters in practice: llama tokenizers include byte-fallback tokens
///   for every individual byte (0x00..=0xFF), so the model can emit e.g.
///   the single byte `0xC1` even while generating ASCII text. Without
///   this check, we'd buffer that byte as an "incomplete" 2-byte lead
///   forever and every subsequent candidate would fail to decode.
fn decode_utf8(buf: &[u8]) -> Utf8Decode {
    if buf.is_empty() {
        return Utf8Decode::Incomplete;
    }
    let b0 = buf[0];
    let expected = if b0 & 0x80 == 0 {
        1
    } else if b0 & 0xE0 == 0xC0 {
        // 0xC0 and 0xC1 are always overlong (they encode ASCII with an
        // extra byte), so they're illegal UTF-8 leads.
        if b0 < 0xC2 {
            return Utf8Decode::Invalid;
        }
        2
    } else if b0 & 0xF0 == 0xE0 {
        3
    } else if b0 & 0xF8 == 0xF0 {
        // 4-byte leads only run up to 0xF4 (codepoint 0x10FFFF);
        // 0xF5..=0xF7 would encode beyond the Unicode space.
        if b0 > 0xF4 {
            return Utf8Decode::Invalid;
        }
        4
    } else {
        // 10xxxxxx (continuation byte at lead position) or 0xF8..=0xFF.
        return Utf8Decode::Invalid;
    };
    if buf.len() < expected {
        for &b in &buf[1..] {
            if b & 0xC0 != 0x80 {
                return Utf8Decode::Invalid;
            }
        }
        return Utf8Decode::Incomplete;
    }
    if buf.len() > expected {
        return Utf8Decode::Invalid;
    }
    match std::str::from_utf8(buf) {
        Ok(s) => Utf8Decode::Complete(s.chars().next().unwrap() as u32),
        Err(_) => Utf8Decode::Invalid,
    }
}

// ===========================================================================
// Errors
// ===========================================================================

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum GrammarError {
    #[error("GBNF syntax error at byte {pos}: {msg}")]
    Syntax { pos: usize, msg: String },
    #[error("grammar references undefined rule `{0}`")]
    UndefinedRule(String),
    #[error("grammar has no `root` rule")]
    MissingRoot,
    #[error("input bytes are not valid UTF-8")]
    InvalidUtf8,
    #[error("codepoint U+{0:04X} does not extend the grammar")]
    NoMatch(u32),
    #[error("I/O error reading `{path}`: {err}")]
    Io {
        path: std::path::PathBuf,
        err: String,
    },
    #[error("internal matcher inconsistency")]
    Internal,
}

static_assertions::assert_impl_all!(GrammarError: Send, Sync);
static_assertions::assert_impl_all!(Grammar: Send, Sync);
static_assertions::assert_impl_all!(GrammarState: Send, Sync);

// ===========================================================================
// Runtime-gated filter-call statistics (opt in via env var)
// ===========================================================================

/// Cumulative statistics about [`grammar_filter`] calls since process start
/// (or since the last [`grammar_stats_reset`]).
///
/// Collection is gated on the `DRAMA_LLAMA_GRAMMAR_STATS` environment
/// variable (set to any non-empty, non-`0` value). When disabled — the
/// default — the filter's hot path adds no atomics and pays zero cost.
///
/// Sums (e.g. `stacks_in_sum`) are provided so callers can compute averages
/// without the static holding floats; divide by `calls` for the mean.
#[derive(Clone, Debug, Default)]
pub struct GrammarStats {
    /// Number of [`grammar_filter`] calls recorded.
    pub calls: u64,
    /// Sum of the input `Candidates` length across calls.
    pub candidates_in: u64,
    /// Count of candidates that survived the first-byte bitmap prefilter.
    pub candidates_bitmap_pass: u64,
    /// Count of candidates that also survived the full `accepts_bytes`
    /// check — i.e. the kept set.
    pub candidates_final_pass: u64,
    /// Sum over calls of `state.inner.stacks.len()` at filter entry.
    pub stacks_in_sum: u64,
    /// Maximum across calls of `state.inner.stacks.len()` at filter entry.
    pub stacks_in_max: u64,
    /// Sum over calls of `max(stack.len())` at filter entry.
    pub depth_max_sum: u64,
    /// Maximum across calls of `max(stack.len())` at filter entry.
    pub depth_max_max: u64,
    /// Sum over calls of wall-clock time spent in the filter, microseconds.
    pub filter_us_sum: u64,
    /// Maximum across calls of wall-clock time, microseconds.
    pub filter_us_max: u64,
    /// Latest observed size of the lazy-DFA intern table (distinct canonical
    /// matcher states seen). Monotonic during a run; reset by
    /// [`grammar_stats_reset`]. Populated from the most-recently-filtered
    /// [`GrammarState`]; with multiple concurrent grammars the value reflects
    /// whichever state most recently hit the filter.
    pub dfa_states: u64,
    /// Cumulative cache hits on byte transitions. Same last-observed caveat.
    pub dfa_transition_hits: u64,
    /// Cumulative cache misses on byte transitions.
    pub dfa_transition_misses: u64,
    /// Cumulative cache hits on per-state first-byte bitmap.
    pub dfa_bitmap_hits: u64,
    /// Cumulative cache misses on per-state first-byte bitmap.
    pub dfa_bitmap_misses: u64,
}

struct StatsInner {
    calls: AtomicU64,
    candidates_in: AtomicU64,
    candidates_bitmap_pass: AtomicU64,
    candidates_final_pass: AtomicU64,
    stacks_in_sum: AtomicU64,
    stacks_in_max: AtomicU64,
    depth_max_sum: AtomicU64,
    depth_max_max: AtomicU64,
    filter_us_sum: AtomicU64,
    filter_us_max: AtomicU64,
    dfa_states: AtomicU64,
    dfa_transition_hits: AtomicU64,
    dfa_transition_misses: AtomicU64,
    dfa_bitmap_hits: AtomicU64,
    dfa_bitmap_misses: AtomicU64,
}

static STATS: StatsInner = StatsInner {
    calls: AtomicU64::new(0),
    candidates_in: AtomicU64::new(0),
    candidates_bitmap_pass: AtomicU64::new(0),
    candidates_final_pass: AtomicU64::new(0),
    stacks_in_sum: AtomicU64::new(0),
    stacks_in_max: AtomicU64::new(0),
    depth_max_sum: AtomicU64::new(0),
    depth_max_max: AtomicU64::new(0),
    filter_us_sum: AtomicU64::new(0),
    filter_us_max: AtomicU64::new(0),
    dfa_states: AtomicU64::new(0),
    dfa_transition_hits: AtomicU64::new(0),
    dfa_transition_misses: AtomicU64::new(0),
    dfa_bitmap_hits: AtomicU64::new(0),
    dfa_bitmap_misses: AtomicU64::new(0),
};

static STATS_ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether `DRAMA_LLAMA_GRAMMAR_STATS` was set to a truthy value when first
/// checked. Cached — subsequent env var changes are ignored.
pub fn grammar_stats_enabled() -> bool {
    *STATS_ENABLED.get_or_init(|| {
        std::env::var_os("DRAMA_LLAMA_GRAMMAR_STATS")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

fn atomic_fetch_max(target: &AtomicU64, val: u64) {
    let mut cur = target.load(Ordering::Relaxed);
    while val > cur {
        match target.compare_exchange_weak(
            cur,
            val,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => cur = observed,
        }
    }
}

/// Snapshot cumulative [`grammar_filter`] statistics. Returns zeros when
/// collection is disabled.
pub fn grammar_stats_snapshot() -> GrammarStats {
    GrammarStats {
        calls: STATS.calls.load(Ordering::Relaxed),
        candidates_in: STATS.candidates_in.load(Ordering::Relaxed),
        candidates_bitmap_pass: STATS
            .candidates_bitmap_pass
            .load(Ordering::Relaxed),
        candidates_final_pass: STATS
            .candidates_final_pass
            .load(Ordering::Relaxed),
        stacks_in_sum: STATS.stacks_in_sum.load(Ordering::Relaxed),
        stacks_in_max: STATS.stacks_in_max.load(Ordering::Relaxed),
        depth_max_sum: STATS.depth_max_sum.load(Ordering::Relaxed),
        depth_max_max: STATS.depth_max_max.load(Ordering::Relaxed),
        filter_us_sum: STATS.filter_us_sum.load(Ordering::Relaxed),
        filter_us_max: STATS.filter_us_max.load(Ordering::Relaxed),
        dfa_states: STATS.dfa_states.load(Ordering::Relaxed),
        dfa_transition_hits: STATS
            .dfa_transition_hits
            .load(Ordering::Relaxed),
        dfa_transition_misses: STATS
            .dfa_transition_misses
            .load(Ordering::Relaxed),
        dfa_bitmap_hits: STATS.dfa_bitmap_hits.load(Ordering::Relaxed),
        dfa_bitmap_misses: STATS.dfa_bitmap_misses.load(Ordering::Relaxed),
    }
}

/// Reset cumulative statistics. Useful to measure a single phase of
/// generation in isolation.
pub fn grammar_stats_reset() {
    STATS.calls.store(0, Ordering::Relaxed);
    STATS.candidates_in.store(0, Ordering::Relaxed);
    STATS.candidates_bitmap_pass.store(0, Ordering::Relaxed);
    STATS.candidates_final_pass.store(0, Ordering::Relaxed);
    STATS.stacks_in_sum.store(0, Ordering::Relaxed);
    STATS.stacks_in_max.store(0, Ordering::Relaxed);
    STATS.depth_max_sum.store(0, Ordering::Relaxed);
    STATS.depth_max_max.store(0, Ordering::Relaxed);
    STATS.filter_us_sum.store(0, Ordering::Relaxed);
    STATS.filter_us_max.store(0, Ordering::Relaxed);
    STATS.dfa_states.store(0, Ordering::Relaxed);
    STATS.dfa_transition_hits.store(0, Ordering::Relaxed);
    STATS.dfa_transition_misses.store(0, Ordering::Relaxed);
    STATS.dfa_bitmap_hits.store(0, Ordering::Relaxed);
    STATS.dfa_bitmap_misses.store(0, Ordering::Relaxed);
}

fn record_stats(
    candidates_in: u64,
    bitmap_pass: u64,
    final_pass: u64,
    stacks_in: u64,
    depth_max: u64,
    elapsed_us: u64,
    cache: Option<&DfaCache>,
) {
    STATS.calls.fetch_add(1, Ordering::Relaxed);
    STATS.candidates_in.fetch_add(candidates_in, Ordering::Relaxed);
    STATS
        .candidates_bitmap_pass
        .fetch_add(bitmap_pass, Ordering::Relaxed);
    STATS
        .candidates_final_pass
        .fetch_add(final_pass, Ordering::Relaxed);
    STATS.stacks_in_sum.fetch_add(stacks_in, Ordering::Relaxed);
    atomic_fetch_max(&STATS.stacks_in_max, stacks_in);
    STATS.depth_max_sum.fetch_add(depth_max, Ordering::Relaxed);
    atomic_fetch_max(&STATS.depth_max_max, depth_max);
    STATS.filter_us_sum.fetch_add(elapsed_us, Ordering::Relaxed);
    atomic_fetch_max(&STATS.filter_us_max, elapsed_us);
    if let Some(cache) = cache {
        // Cache counters are already cumulative inside the cache itself, so
        // we overwrite rather than add. Multiple concurrent grammars would
        // race here; last-writer-wins is acceptable for the single-grammar
        // common case.
        STATS
            .dfa_states
            .store(cache.state_count() as u64, Ordering::Relaxed);
        STATS
            .dfa_transition_hits
            .store(cache.transition_hits(), Ordering::Relaxed);
        STATS
            .dfa_transition_misses
            .store(cache.transition_misses(), Ordering::Relaxed);
        STATS
            .dfa_bitmap_hits
            .store(cache.bitmap_hits(), Ordering::Relaxed);
        STATS
            .dfa_bitmap_misses
            .store(cache.bitmap_misses(), Ordering::Relaxed);
    }
}

// ===========================================================================
// Filter + advance plumbing (mirror of json::json_filter / advance_all)
// ===========================================================================

/// Filter candidates to those whose token bytes extend the grammar.
///
/// On zero valid candidates, returns a single-token [`Candidates`] holding
/// EOS. Two cases trigger this:
///
/// * **Success termination**: the grammar reached an accept state; all
///   further tokens are rejected. The matcher is auto-reset so the next
///   generation starts fresh.
/// * **Grammar violation**: no candidate token extends the match. State is
///   preserved for inspection via [`GrammarState::stack_depth`].
pub(crate) fn grammar_filter(
    candidates: Candidates,
    state: &mut GrammarState,
    model: &LlamaCppModel,
) -> Candidates {
    // Each candidate check is independent: clone the matcher state,
    // try to advance it by the token's bytes, keep the token iff the
    // clone survives. Fan out across rayon's global pool so 150k-vocab
    // models don't bottleneck on a single core. `GrammarState` is
    // auto-Sync (pure data behind an Arc), `LlamaCppModel` is Sync by manual
    // impl (post-load data is immutable — see src/model.rs).
    //
    // The first-byte bitmap is a cheap O(1)/candidate prefilter that
    // rejects candidates whose first byte can't extend the match. See
    // [`StackState::first_byte_bitmap`] for the invariant.
    //
    // We borrow `&state.grammar` once outside the parallel loop so
    // per-candidate scratch clones stay inside `StackState` and never
    // bump the `Arc<Grammar>` refcount (profiles showed that atomic
    // traffic was a real cost).
    let stats_on = grammar_stats_enabled();
    let t0 = stats_on.then(Instant::now);
    let candidates_in = candidates.as_slice().len() as u64;

    let grammar: &Grammar = &state.grammar;
    let inner: &StackState = &state.inner;
    let cache_on = dfa_cache_enabled();
    let cache: &Arc<DfaCache> = &state.cache;

    // Fast path: the lazy-DFA cache memoizes one-byte transitions and the
    // first-byte bitmap per canonical matcher state. Intern the base state
    // up-front so every candidate walks the same transition table from the
    // same state id.
    let base_id = if cache_on { cache.intern(inner) } else { 0 };
    let bitmap = if cache_on {
        cache.first_byte_bitmap(grammar, base_id)
    } else {
        state.first_byte_bitmap()
    };

    #[derive(Default)]
    struct Acc {
        kept: Vec<TokenData>,
        bitmap_pass: u64,
    }

    let acc = candidates
        .as_slice()
        .par_iter()
        .fold(Acc::default, |mut a, cand| {
            let mut buf: Vec<u8> = Vec::with_capacity(32);
            token_to_piece_ref(cand.id, model, &mut buf);
            if let Some(&first) = buf.first() {
                if bitmap[(first as usize) >> 6] & (1u64 << (first & 63)) == 0
                {
                    return a;
                }
            }
            a.bitmap_pass += 1;
            let accepts = if cache_on {
                let mut sid = base_id;
                let mut rejected = false;
                for &b in &buf {
                    sid = cache.transition(grammar, sid, b);
                    if sid == REJECT_STATE {
                        rejected = true;
                        break;
                    }
                }
                !rejected && cache.terminal_valid(grammar, sid)
            } else {
                inner.accepts_bytes(grammar, &buf)
            };
            if accepts {
                a.kept.push(*cand);
            }
            a
        })
        .reduce(Acc::default, |mut a, b| {
            a.kept.extend(b.kept);
            a.bitmap_pass += b.bitmap_pass;
            a
        });

    if let Some(t0) = t0 {
        let elapsed_us = t0.elapsed().as_micros() as u64;
        let stacks_in = inner.stacks.len() as u64;
        let depth_max =
            inner.stacks.iter().map(|s| s.len()).max().unwrap_or(0) as u64;
        record_stats(
            candidates_in,
            acc.bitmap_pass,
            acc.kept.len() as u64,
            stacks_in,
            depth_max,
            elapsed_us,
            if cache_on { Some(cache.as_ref()) } else { None },
        );
    }

    let kept = acc.kept;
    if kept.is_empty() {
        if state.is_complete() {
            state.reset();
        }
        let eos = TokenData {
            id: model.eos(),
            logit: 0.0,
            p: 1.0,
        };
        return Candidates::from_vec(vec![eos]);
    }

    Candidates::from_vec_unchecked(kept)
}

/// Advance every `SamplingMode::Grammar` state in `modes` by the bytes of
/// `token`.
///
/// # Panics
///
/// Panics if any grammar mutex is poisoned. A poisoned mutex means a
/// previous panic left the matcher in an undefined state, and silently
/// continuing would produce output that violates the grammar.
pub(crate) fn advance_all(
    modes: &[crate::SamplingMode],
    token: llama_token,
    model: &LlamaCppModel,
) {
    use crate::SamplingMode;
    let mut buf: Vec<u8> = Vec::new();
    let mut computed = false;
    for mode in modes {
        if let SamplingMode::Grammar(state) = mode {
            if !computed {
                token_to_piece_ref(token, model, &mut buf);
                computed = true;
            }
            let mut locked = state.lock().expect(
                "SamplingMode::Grammar mutex poisoned during advance; \
                 matcher state is unrecoverable. Rebuild the mode and \
                 retry.",
            );
            // An advance error means the grammar was violated on the prior
            // step (EOS fallback chose an out-of-grammar token). Generation
            // terminates on the next step via the EOS stop_sequence, so
            // silently no-op here.
            let _ = locked.advance_bytes(&buf);
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts_complete(grammar_src: &str, input: &str) -> bool {
        let grammar = match Grammar::parse(grammar_src) {
            Ok(g) => Arc::new(g),
            Err(e) => panic!("grammar failed to parse: {e}\n{grammar_src}"),
        };
        let mut state = GrammarState::new(grammar);
        if state.advance_bytes(input.as_bytes()).is_err() {
            return false;
        }
        state.is_complete()
    }

    fn parse_ok(src: &str) -> Grammar {
        Grammar::parse(src).expect("grammar should parse")
    }

    #[test]
    fn parse_simple_literal() {
        parse_ok(r#"root ::= "hello""#);
    }

    #[test]
    fn accepts_literal() {
        assert!(accepts_complete(r#"root ::= "hello""#, "hello"));
        assert!(!accepts_complete(r#"root ::= "hello""#, "hell"));
        assert!(!accepts_complete(r#"root ::= "hello""#, "hellox"));
    }

    #[test]
    fn accepts_alternation() {
        let g = r#"root ::= "yes" | "no""#;
        assert!(accepts_complete(g, "yes"));
        assert!(accepts_complete(g, "no"));
        assert!(!accepts_complete(g, "maybe"));
    }

    #[test]
    fn accepts_sequence() {
        let g = r#"
            root ::= greeting " " name
            greeting ::= "hi" | "hello"
            name ::= "world" | "claude"
        "#;
        assert!(accepts_complete(g, "hi world"));
        assert!(accepts_complete(g, "hello claude"));
        assert!(!accepts_complete(g, "hey claude"));
    }

    #[test]
    fn accepts_star() {
        let g = r#"root ::= "a"*"#;
        assert!(accepts_complete(g, ""));
        assert!(accepts_complete(g, "a"));
        assert!(accepts_complete(g, "aaaa"));
        assert!(!accepts_complete(g, "ab"));
    }

    #[test]
    fn accepts_plus() {
        let g = r#"root ::= "a"+"#;
        assert!(!accepts_complete(g, ""));
        assert!(accepts_complete(g, "a"));
        assert!(accepts_complete(g, "aaa"));
    }

    #[test]
    fn accepts_optional() {
        let g = r#"root ::= "a" "b"?"#;
        assert!(accepts_complete(g, "a"));
        assert!(accepts_complete(g, "ab"));
        assert!(!accepts_complete(g, "b"));
    }

    #[test]
    fn accepts_char_class() {
        let g = r#"root ::= [a-z]+"#;
        assert!(accepts_complete(g, "abc"));
        assert!(accepts_complete(g, "z"));
        assert!(!accepts_complete(g, ""));
        assert!(!accepts_complete(g, "Abc"));
    }

    #[test]
    fn accepts_negated_char_class() {
        let g = r#"root ::= [^0-9]+"#;
        assert!(accepts_complete(g, "abc"));
        assert!(!accepts_complete(g, "a1b"));
    }

    #[test]
    fn char_class_with_escapes() {
        let g = r#"root ::= [\n\t]+"#;
        assert!(accepts_complete(g, "\n\t\n"));
        assert!(!accepts_complete(g, " "));
    }

    #[test]
    fn char_class_hex_escape() {
        let g = r#"root ::= [\x41-\x43]+"#;
        assert!(accepts_complete(g, "ABC"));
        assert!(!accepts_complete(g, "D"));
    }

    #[test]
    fn accepts_group() {
        let g = r#"root ::= ("ab" | "cd")+"#;
        assert!(accepts_complete(g, "ab"));
        assert!(accepts_complete(g, "abcd"));
        assert!(accepts_complete(g, "cdab"));
        assert!(!accepts_complete(g, "a"));
        assert!(!accepts_complete(g, "abc"));
    }

    #[test]
    fn any_codepoint_with_dot() {
        let g = r#"root ::= .+"#;
        assert!(accepts_complete(g, "anything goes"));
        assert!(!accepts_complete(g, ""));
        // `.` excludes newlines.
        assert!(!accepts_complete(g, "a\nb"));
    }

    #[test]
    fn utf8_multi_byte() {
        let g = r#"root ::= [ア-ン]+"#;
        assert!(accepts_complete(g, "アイウエオ"));
        assert!(!accepts_complete(g, "abc"));
    }

    /// Regression: byte-fallback tokens whose single byte is an invalid
    /// or grammar-incompatible UTF-8 lead used to wedge pending forever,
    /// rejecting every subsequent candidate.
    #[test]
    fn rejects_lead_byte_incompatible_with_ascii_grammar() {
        let grammar = Arc::new(parse_ok(r#"root ::= "abc""#));
        let state = GrammarState::new(grammar);
        // 0xC1 is ALWAYS invalid UTF-8 (overlong lead) — must reject.
        assert!(
            !state.accepts_bytes(&[0xC1]),
            "overlong lead 0xC1 must be rejected"
        );
        // 0xC3 is a valid 2-byte lead but encodes U+00C0..U+00FF, none
        // of which match an ASCII-only grammar. Must reject.
        assert!(
            !state.accepts_bytes(&[0xC3]),
            "non-ASCII lead must be rejected when grammar is ASCII-only"
        );
        // 0xF0 as a lone byte could start a 4-byte codepoint U+10000+,
        // still no ASCII match.
        assert!(
            !state.accepts_bytes(&[0xF0]),
            "4-byte lead must be rejected when grammar is ASCII-only"
        );
    }

    /// Non-ASCII grammars still accept split multi-byte tokens.
    #[test]
    fn accepts_split_multibyte_for_matching_grammar() {
        let grammar = Arc::new(parse_ok(r#"root ::= "é""#));
        let mut state = GrammarState::new(grammar);
        let bytes = "é".as_bytes();
        assert_eq!(bytes, &[0xC3, 0xA9]);
        // Feed byte-by-byte (simulating a split-token scenario).
        assert!(state.accepts_bytes(&bytes[..1]));
        state.advance_bytes(&bytes[..1]).unwrap();
        state.advance_bytes(&bytes[1..]).unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn utf8_split_across_tokens() {
        // Simulate a model that emits the bytes of a multi-byte codepoint
        // across multiple advance_bytes calls. The matcher must buffer the
        // partial UTF-8.
        let grammar = Arc::new(parse_ok(r#"root ::= [ぁ-ん]+"#));
        let mut state = GrammarState::new(grammar);
        let bytes = "あ".as_bytes();
        assert_eq!(bytes.len(), 3);
        state.advance_bytes(&bytes[..1]).unwrap();
        assert!(!state.is_complete(), "partial utf8 must not complete");
        state.advance_bytes(&bytes[1..2]).unwrap();
        state.advance_bytes(&bytes[2..3]).unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn comments_and_whitespace() {
        let g = r#"
            # This is a comment.
            root ::= greeting  # trailing comment
            greeting ::= "hi"  # another
            # footer comment
        "#;
        assert!(accepts_complete(g, "hi"));
    }

    #[test]
    fn reject_missing_root() {
        let err = Grammar::parse(r#"foo ::= "x""#).unwrap_err();
        assert!(matches!(err, GrammarError::MissingRoot));
    }

    #[test]
    fn reject_undefined_rule() {
        let err = Grammar::parse(r#"root ::= undefined_rule"#).unwrap_err();
        assert!(matches!(err, GrammarError::UndefinedRule(_)));
    }

    #[test]
    fn reject_syntax_error() {
        let err = Grammar::parse(r#"root := "x""#).unwrap_err();
        assert!(matches!(err, GrammarError::Syntax { .. }));
    }

    #[test]
    fn reset_clears_state() {
        let grammar = Arc::new(parse_ok(r#"root ::= "ab""#));
        let mut state = GrammarState::new(grammar);
        state.advance_bytes(b"a").unwrap();
        assert!(!state.is_complete());
        state.reset();
        // After reset, "ab" should work again.
        state.advance_bytes(b"ab").unwrap();
        assert!(state.is_complete());
    }

    #[test]
    fn accepts_bytes_no_mutation() {
        let grammar = Arc::new(parse_ok(r#"root ::= "hello""#));
        let state = GrammarState::new(grammar);
        assert!(state.accepts_bytes(b"he"));
        assert!(state.accepts_bytes(b"hello"));
        assert!(!state.accepts_bytes(b"x"));
        // Original state unchanged.
        assert_eq!(state.stack_depth(), 1);
    }

    #[test]
    fn deeply_nested_alternation() {
        let g = r#"
            root ::= a
            a ::= b | c
            b ::= d | e
            c ::= f | g
            d ::= "d"
            e ::= "e"
            f ::= "f"
            g ::= "g"
        "#;
        for &s in &["d", "e", "f", "g"] {
            assert!(accepts_complete(g, s), "should accept {s}");
        }
        assert!(!accepts_complete(g, "x"));
    }

    #[test]
    fn tool_call_shape() {
        // The motivating use case: force a specific JSON tool call.
        let g = r#"
            root ::= "{\"name\":\"" name "\",\"arguments\":" obj "}"
            name ::= "get_weather"
            obj ::= "{" pair ("," pair)* "}"
            pair ::= string ":" value
            string ::= "\"" [a-zA-Z_]+ "\""
            value ::= string | number
            number ::= [0-9]+
        "#;
        let sample =
            r#"{"name":"get_weather","arguments":{"city":"Paris","days":3}}"#;
        assert!(accepts_complete(g, sample), "should accept: {sample}");
    }

    #[test]
    fn escape_in_string_literal() {
        let g = r#"root ::= "a\nb""#;
        assert!(accepts_complete(g, "a\nb"));
        assert!(!accepts_complete(g, "anb"));
    }

    // ======================================================================
    // UTF-8 boundary regression tests (Phase 0.5.2 gap-fill)
    // ======================================================================

    /// Surrogate range U+D800..=U+DFFF encodes as `0xED 0xA0..=0xBF
    /// 0x80..=0xBF` in strict UTF-8. Those codepoints are invalid UTF-8
    /// and must be rejected even when the grammar permits any
    /// codepoint.
    #[test]
    fn rejects_surrogate_byte_sequence() {
        let g = r#"root ::= char+
char ::= [\x00-\x7F] | [\x80-\xFF]"#;
        let grammar = Arc::new(Grammar::parse(g).unwrap());
        let mut state = GrammarState::new(grammar);
        // 0xED 0xA0 0x80 = U+D800 (high surrogate)
        assert!(state.advance_bytes(&[0xED, 0xA0, 0x80]).is_err());
    }

    /// Codepoints above U+10FFFF are not valid Unicode and must be
    /// rejected. `0xF4 0x90 0x80 0x80` would decode as U+110000.
    #[test]
    fn rejects_codepoint_above_max_unicode() {
        let g = r#"root ::= char+
char ::= [\x00-\x7F] | [\x80-\xFF]"#;
        let grammar = Arc::new(Grammar::parse(g).unwrap());
        let mut state = GrammarState::new(grammar);
        assert!(state.advance_bytes(&[0xF4, 0x90, 0x80, 0x80]).is_err());
    }

    /// A lone continuation byte (`0x80..=0xBF` without a lead) is not
    /// valid UTF-8.
    #[test]
    fn rejects_lone_continuation_byte() {
        let g = r#"root ::= char+
char ::= [\x00-\x7F] | [\x80-\xFF]"#;
        let grammar = Arc::new(Grammar::parse(g).unwrap());
        let mut state = GrammarState::new(grammar);
        assert!(state.advance_bytes(&[0x80]).is_err());
    }

    /// Legacy 5/6-byte leads (`0xF8..=0xFF`) were valid in early UTF-8
    /// drafts but aren't part of the 2003+ spec. Reject them.
    #[test]
    fn rejects_legacy_long_utf8_leads() {
        let g = r#"root ::= char+
char ::= [\x00-\x7F] | [\x80-\xFF]"#;
        for lead in [0xF8u8, 0xFC, 0xFE, 0xFF] {
            let grammar = Arc::new(Grammar::parse(g).unwrap());
            let mut state = GrammarState::new(grammar);
            assert!(
                state.advance_bytes(&[lead]).is_err(),
                "lead 0x{lead:02X} should be rejected"
            );
        }
    }

    // ======================================================================
    // GBNF parser error paths (Phase 0.5.2 gap-fill)
    // ======================================================================

    #[test]
    fn duplicate_rule_definition_rejected() {
        // Two definitions of `root` — parser must flag.
        let g = "root ::= \"a\"\nroot ::= \"b\"";
        assert!(Grammar::parse(g).is_err());
    }

    #[test]
    fn char_class_unterminated_rejected() {
        assert!(Grammar::parse("root ::= [abc").is_err());
    }

    #[test]
    fn char_class_inverted_range_rejected() {
        // `[z-a]` is an inverted range, parser must error.
        assert!(Grammar::parse("root ::= [z-a]").is_err());
    }

    #[test]
    fn char_class_trailing_dash_accepted_as_literal() {
        // `[a-zA-Z0-9-]` — the trailing `-` is a literal dash, not a
        // partial range.
        let g = r#"root ::= [a-zA-Z0-9-]+"#;
        assert!(accepts_complete(g, "abc-123"));
        assert!(accepts_complete(g, "-"));
    }

    #[test]
    fn right_recursion_accepts() {
        // Right-recursive: root ::= "a" root | "b".
        let g = r#"root ::= "a" root | "b""#;
        assert!(accepts_complete(g, "b"));
        assert!(accepts_complete(g, "ab"));
        assert!(accepts_complete(g, "aaab"));
    }

    #[test]
    fn is_complete_false_mid_match() {
        // Parsing "a" against `root ::= "abc"` must leave is_complete
        // false — the grammar expects more.
        let g = r#"root ::= "abc""#;
        let grammar = Arc::new(Grammar::parse(g).unwrap());
        let mut state = GrammarState::new(grammar);
        state.advance_bytes(b"a").unwrap();
        assert!(!state.is_complete(), "mid-match should not be complete");
        state.advance_bytes(b"bc").unwrap();
        assert!(state.is_complete(), "full match should be complete");
    }

    /// End-to-end integration test: run a real model with a GBNF that
    /// forces a specific tool-call shape.
    #[cfg(feature = "serde")]
    #[test]
    #[ignore = "requires model"]
    fn grammar_integration_tool_call() {
        use crate::{Engine, PredictOptions, SampleOptions, SamplingMode};
        use std::{num::NonZeroUsize, path::PathBuf};

        let model_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let mut engine = Engine::from_path(model_path).unwrap();

        // A grammar that forces the output to look like a tool call for
        // `get_weather`. The constraint fixes the function name but lets
        // the model fill the argument value freely (as ASCII text).
        const GBNF: &str = r#"
            root ::= "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"" city "\"}}"
            city ::= [A-Za-z][A-Za-z ]*
        "#;

        const PROMPT: &str = "You have a tool called get_weather(city). \
            Call it for Paris. Output only the JSON tool call. JSON: ";

        let tokens = engine.model.tokenize(PROMPT, false);

        let mut opts = PredictOptions::default().add_model_stops(&engine.model);
        opts.n = NonZeroUsize::new(256).unwrap();
        let grammar_mode =
            SamplingMode::grammar(GBNF).expect("test grammar should parse");
        opts.sample_options = SampleOptions {
            modes: vec![grammar_mode, SamplingMode::locally_typical()],
            ..SampleOptions::default()
        };

        let eos_piece = engine.model.token_to_piece(engine.model.eos());
        let predictor = engine.predict_pieces(tokens, opts);
        let output: String = predictor.collect();

        println!(
            "=== Generated tool call ===\n{output}\n=========================="
        );
        let trimmed = output.trim_end_matches(eos_piece.as_str()).trim_end();

        // The grammar forces this prefix; the test fails loudly if not.
        assert!(
            trimmed.starts_with(r#"{"name":"get_weather","arguments":{"city":""#),
            "output must start with the forced tool-call prefix. output: {output:?}"
        );
    }

    // ======================================================================
    // First-byte bitmap prefilter
    // ======================================================================

    fn bit_is_set(bitmap: &[u64; 4], b: u8) -> bool {
        bitmap[(b as usize) >> 6] & (1u64 << (b & 63)) != 0
    }

    fn bitmap_popcount(bitmap: &[u64; 4]) -> u32 {
        bitmap.iter().map(|w| w.count_ones()).sum()
    }

    /// A literal grammar admits exactly one first byte at the start.
    #[test]
    fn bitmap_literal_single_byte() {
        let grammar = Arc::new(parse_ok(r#"root ::= "hello""#));
        let state = GrammarState::new(grammar);
        let bm = state.first_byte_bitmap();
        assert_eq!(bitmap_popcount(&bm), 1);
        assert!(bit_is_set(&bm, b'h'));
        assert!(!bit_is_set(&bm, b'H'));
        assert!(!bit_is_set(&bm, b'\0'));
    }

    /// A char class admits every byte in its range and nothing else.
    #[test]
    fn bitmap_ascii_char_class() {
        let grammar = Arc::new(parse_ok(r#"root ::= [a-c]+"#));
        let state = GrammarState::new(grammar);
        let bm = state.first_byte_bitmap();
        assert_eq!(bitmap_popcount(&bm), 3);
        for b in b'a'..=b'c' {
            assert!(bit_is_set(&bm, b), "byte {b:#x} should be set");
        }
        assert!(!bit_is_set(&bm, b'd'));
        assert!(!bit_is_set(&bm, b'A'));
    }

    /// Alternation unions the per-branch bitmaps.
    #[test]
    fn bitmap_alternation_union() {
        let grammar = Arc::new(parse_ok(r#"root ::= "yes" | "no""#));
        let state = GrammarState::new(grammar);
        let bm = state.first_byte_bitmap();
        assert!(bit_is_set(&bm, b'y'));
        assert!(bit_is_set(&bm, b'n'));
        assert_eq!(bitmap_popcount(&bm), 2);
    }

    /// Never-valid UTF-8 leads (overlong 2-byte leads, out-of-range 4-byte
    /// leads, continuation bytes in lead position) must never be set.
    #[test]
    fn bitmap_excludes_invalid_utf8_leads() {
        let grammar = Arc::new(parse_ok(r#"root ::= .+"#));
        let state = GrammarState::new(grammar);
        let bm = state.first_byte_bitmap();
        for b in [0xC0u8, 0xC1, 0xF5, 0xF6, 0xF7, 0xF8, 0xFE, 0xFF] {
            assert!(!bit_is_set(&bm, b), "invalid lead {b:#x} was set");
        }
        for b in 0x80u8..=0xBFu8 {
            assert!(!bit_is_set(&bm, b), "continuation {b:#x} was set");
        }
    }

    /// An accepting state (literal already consumed) rejects all further
    /// codepoints — the bitmap must be entirely zero.
    #[test]
    fn bitmap_accepting_state_is_empty() {
        let grammar = Arc::new(parse_ok(r#"root ::= "hi""#));
        let mut state = GrammarState::new(grammar);
        state.advance_bytes(b"hi").unwrap();
        assert!(state.is_complete());
        let bm = state.first_byte_bitmap();
        assert_eq!(bitmap_popcount(&bm), 0);
    }

    /// With a pending UTF-8 lead buffered, only the continuation bytes
    /// that would complete the codepoint into an accepted range are set.
    #[test]
    fn bitmap_with_pending_restricts_to_valid_continuations() {
        // "é" = 0xC3 0xA9. Feeding just 0xC3 leaves pending = [0xC3].
        let grammar = Arc::new(parse_ok(r#"root ::= "é""#));
        let mut state = GrammarState::new(grammar);
        state.advance_bytes(&[0xC3]).unwrap();
        let bm = state.first_byte_bitmap();
        // Exactly one continuation byte completes the codepoint.
        assert!(bit_is_set(&bm, 0xA9));
        // Anything else — including other continuations — must be clear.
        for b in 0x80u8..=0xBFu8 {
            if b != 0xA9 {
                assert!(!bit_is_set(&bm, b), "{b:#x} wrongly set");
            }
        }
        // And ASCII bytes certainly can't extend a 2-byte lead.
        for b in 0u8..=0x7Fu8 {
            assert!(!bit_is_set(&bm, b));
        }
    }

    /// Every byte whose bit is cleared must cause `accepts_bytes` to
    /// return false — that's the prefilter's soundness invariant.
    #[test]
    fn bitmap_soundness_matches_accepts_bytes() {
        for src in [
            r#"root ::= "foo""#,
            r#"root ::= [a-zA-Z]+"#,
            r#"root ::= [^ \n]+"#,
            r#"root ::= "{" [a-z]+ ":" [0-9]+ "}""#,
        ] {
            let grammar = Arc::new(parse_ok(src));
            let state = GrammarState::new(grammar);
            let bm = state.first_byte_bitmap();
            for b in 0u8..=0xFFu8 {
                if !bit_is_set(&bm, b) {
                    assert!(
                        !state.accepts_bytes(&[b]),
                        "grammar {src:?} rejected byte {b:#x} via bitmap but \
                         accepts_bytes permitted it"
                    );
                }
            }
        }
    }

    /// A Japanese-range grammar expects multi-byte leads only — the ASCII
    /// half of the bitmap must be empty, and the Hiragana leads must be
    /// set.
    #[test]
    fn bitmap_multibyte_grammar() {
        let grammar = Arc::new(parse_ok(r#"root ::= [ぁ-ん]+"#));
        let state = GrammarState::new(grammar);
        let bm = state.first_byte_bitmap();
        for b in 0u8..=0x7Fu8 {
            assert!(!bit_is_set(&bm, b));
        }
        // Hiragana U+3041..U+3093 all start with 3-byte lead 0xE3.
        assert!(bit_is_set(&bm, 0xE3));
    }
}
