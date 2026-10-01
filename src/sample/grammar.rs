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
use rayon::prelude::*;

use crate::TokenData;
use rustc_hash::FxHashMap;
use tinyvec::{ArrayVec, TinyVec};

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

use crate::{backend::Model, Candidates};

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

/// Most bytes of GBNF [`Grammar::parse`] takes, and what the schema
/// compiler stops at: a schema-derived grammar grows with the schema,
/// and a client's schema must not be able to make one large enough
/// to exhaust memory or stall compilation (the hostile-schema recheck).
/// Real grammars are far smaller — a large tool set compiles to a few
/// hundred KiB.
pub(crate) const MAX_GRAMMAR_BYTES: usize = 8 << 20;

/// Most rules [`Grammar::parse`] builds, anonymous ones (groups,
/// repetitions) included. Bounds the rule table where
/// [`MAX_GRAMMAR_BYTES`] bounds the source.
pub(crate) const MAX_GRAMMAR_RULES: usize = 1 << 18;

/// How many rules [`Grammar::parse`] builds from `source`: one per
/// definition, plus an anonymous one per string literal, group and
/// `*` / `+` / `?`. A scan, not a parse — what the schema compiler
/// checks its output against [`MAX_GRAMMAR_RULES`] with as it writes.
/// Never more than `source.len()`: each rule costs at least a byte.
pub(crate) fn rule_count(source: &str) -> usize {
    let bytes = source.as_bytes();
    // Skip a `"…"` literal or `[…]` class from its opener at `i`.
    let skip = |mut i: usize, close: u8| {
        i += 1;
        while i < bytes.len() && bytes[i] != close {
            i += 1 + usize::from(bytes[i] == b'\\');
        }
        i + 1
    };
    let (mut i, mut rules) = (0, 0);
    while i < bytes.len() {
        i = match bytes[i] {
            b'"' => {
                rules += 1;
                skip(i, b'"')
            }
            b'[' => skip(i, b']'),
            b'#' => bytes[i..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(bytes.len(), |n| i + n + 1),
            b'(' | b'*' | b'+' | b'?' => {
                rules += 1;
                i + 1
            }
            b':' if bytes[i..].starts_with(b"::=") => {
                rules += 1;
                i + 3
            }
            _ => i + 1,
        };
    }
    rules
}

impl Grammar {
    /// Parse GBNF source text into a compiled grammar.
    ///
    /// A source past 8 MiB (`MAX_GRAMMAR_BYTES`), or one that builds
    /// more than 2^18 rules (`MAX_GRAMMAR_RULES`), is
    /// [`GrammarError::TooLarge`].
    pub fn parse(source: &str) -> Result<Self, GrammarError> {
        if source.len() > MAX_GRAMMAR_BYTES {
            return Err(GrammarError::TooLarge {
                what: "bytes",
                limit: MAX_GRAMMAR_BYTES,
            });
        }
        let mut builder = GrammarBuilder::new(source);
        builder.parse_document()?;
        if builder.rules.len() > MAX_GRAMMAR_RULES {
            return Err(GrammarError::TooLarge {
                what: "rules",
                limit: MAX_GRAMMAR_RULES,
            });
        }
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
    /// Current depth of recursive `parse_alternates` / `parse_atom`
    /// calls. Bounded by [`PARSER_RECURSION_LIMIT`] to prevent
    /// stack-overflow crashes on pathological inputs (deeply-nested
    /// `(((...)))` groups). Surfaced by the in-tree fuzzer.
    parse_depth: usize,
}

/// Hard cap on `parse_atom` ↔ `parse_alternates` recursion. Generous
/// enough that any realistic grammar (incl. JSON-Schema-derived ones
/// from the deepest objects we'd see in practice) parses fine, but
/// small enough that 8 MB default stacks never approach the guard
/// page on a clean compile path.
const PARSER_RECURSION_LIMIT: usize = 256;

impl<'a> GrammarBuilder<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            cursor: 0,
            name_to_idx: std::collections::HashMap::new(),
            rules: Vec::new(),
            anon_counter: 0,
            parse_depth: 0,
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
        // Recursion guard: `parse_alternates` calls `parse_sequence`
        // which calls `parse_atom`, and `parse_atom` recurses back here
        // for `(...)` groups. Bound the chain so a pathological
        // `(((...)))` source can't blow the stack.
        if self.parse_depth >= PARSER_RECURSION_LIMIT {
            return Err(GrammarError::RecursionLimit {
                pos: self.cursor,
                limit: PARSER_RECURSION_LIMIT,
            });
        }
        self.parse_depth += 1;
        let result = (|| {
            let mut alts = vec![self.parse_sequence()?];
            loop {
                self.skip_trivia();
                if !self.eat('|') {
                    break;
                }
                self.skip_trivia();
                alts.push(self.parse_sequence()?);
            }
            Ok::<_, GrammarError>(alts)
        })();
        self.parse_depth -= 1;
        result
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
                for (_, c2) in chars.by_ref() {
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
        let after = &rest[j..].trim_start_matches([' ', '\t']);
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
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
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
// Serializable so a sampler-state snapshot can carry the matcher's
// exact position. NOTE for the deserialize door (config/state split
// Phase 3): positions index into a *specific* compiled grammar; a
// deserialized StackState is only meaningful against the same source,
// and indices must be bounds-checked on restore.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Eq, Hash, PartialEq)]
pub(crate) struct StackState {
    stacks: Vec<Stack>,
    pending: ArrayVec<[u8; 4]>,
}

impl Clone for StackState {
    fn clone(&self) -> Self {
        Self {
            stacks: self.stacks.iter().map(|s| clone_stack(s, 0)).collect(),
            pending: self.pending,
        }
    }
}

/// `stack`, with room for `extra` more frames. A spilled copy gets a
/// power-of-two capacity, so the matcher's heap blocks come in a few
/// sizes the allocator can reuse. Exact-length copies asked for a
/// slightly larger block at every level of nesting, and the freed ones,
/// too small to reuse, stayed resident: the recheck's ambiguous schema,
/// filtered every level 1,000 deep, held ~1.45 GB resident over ~50 MB
/// live; with these, ~580 MB, in the same time.
fn clone_stack(stack: &Stack, extra: usize) -> Stack {
    let len = stack.len() + extra;
    if len <= STACK_INLINE {
        return stack.clone();
    }
    let mut heap = Vec::with_capacity(len.next_power_of_two());
    heap.extend_from_slice(stack);
    TinyVec::Heap(heap)
}

/// A compiled grammar plus its lazy-DFA cache — the *config* half of a
/// grammar constraint. Immutable: matching position lives in the
/// per-call sampler state (`StackState`), never here.
///
/// `Clone` shares both the compiled rules and the cache (`Arc`), so a
/// Session-owned config keeps one warm cache across calls. The cache is
/// a pure memoization of the grammar: equality and serialization both
/// ignore it (equality compares source; serde round-trips source only —
/// `DfaCache` state ids are process-local interning and must never
/// cross a serialization boundary).
///
/// This is the sampler config's single carve-out from full derive
/// purity — one manual `PartialEq`, one source-only serde impl. Don't
/// add more.
#[derive(Clone, Debug)]
pub struct CompiledGrammar {
    pub(crate) grammar: Arc<Grammar>,
    pub(crate) dfa: Arc<DfaCache>,
}

impl CompiledGrammar {
    /// Compile from GBNF source. Returns the parse error if the grammar
    /// is malformed.
    pub fn parse(source: &str) -> Result<Self, GrammarError> {
        Ok(Self::from_grammar(Arc::new(Grammar::parse(source)?)))
    }

    /// Compile by loading a `.gbnf` file from disk.
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, GrammarError> {
        Ok(Self::from_grammar(Arc::new(Grammar::from_file(path)?)))
    }

    /// Wrap an already-compiled grammar with a fresh cache.
    pub fn from_grammar(grammar: Arc<Grammar>) -> Self {
        Self {
            grammar,
            dfa: Arc::new(DfaCache::new()),
        }
    }

    /// The original GBNF source.
    pub fn source(&self) -> &str {
        self.grammar.source()
    }

    /// SHA-256 of the GBNF source — the grammar's identity. Matcher
    /// positions in a [`SamplerState`](crate::SamplerState) index into
    /// a *specific* compiled grammar, so each grammar matcher carries
    /// this hash; `SamplerState::resumed_from` carries a cached
    /// position forward iff the identities agree (same source ⇒ same
    /// deterministic compile ⇒ same indices). The (Phase 3)
    /// deserialize door's grammar-identity gate uses the same value.
    pub(crate) fn source_hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.source().as_bytes());
        hasher.finalize().into()
    }

    /// A fresh matcher at this grammar's root rule.
    pub(crate) fn root_state(&self) -> StackState {
        StackState::new_rooted(&self.grammar)
    }
}

/// Source-identity equality; the DFA cache is ignored (pure
/// acceleration). Same rationale as [`GrammarState`]'s manual impl.
impl PartialEq for CompiledGrammar {
    fn eq(&self, other: &Self) -> bool {
        self.grammar == other.grammar
    }
}

/// Serializes as the GBNF source string only; deserialization re-parses
/// and starts a cold cache. Same compile is deterministic, so matcher
/// positions serialized alongside remain index-consistent.
#[cfg(feature = "serde")]
impl serde::Serialize for CompiledGrammar {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.source())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledGrammar {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let source = String::deserialize(deserializer)?;
        Self::parse(&source).map_err(serde::de::Error::custom)
    }
}

/// Active matching state for a [`Grammar`].
///
/// Thin wrapper: owns the `Arc<Grammar>` plus the mutable `StackState`.
/// All matcher methods delegate into `StackState` with a borrowed
/// `&Grammar`. See `StackState` for why.
///
/// Clone cost is proportional to `stacks.len() * avg_stack_depth`, which is
/// small for practical grammars. `accepts_bytes` relies on cloning for
/// speculative simulation.
///
/// The sampling chain does not use this type — `SamplingMode::Grammar`
/// holds a [`CompiledGrammar`] (config, with the lazy-DFA cache) and
/// the matcher `StackState` lives in `SamplerState`. `GrammarState` is
/// the standalone convenience for callers driving a matcher by hand.
#[derive(Clone, Debug, PartialEq)]
pub struct GrammarState {
    grammar: Arc<Grammar>,
    inner: StackState,
}

impl GrammarState {
    /// Construct a fresh matcher rooted at the grammar's `root` rule.
    pub fn new(grammar: Arc<Grammar>) -> Self {
        let inner = StackState::new_rooted(&grammar);
        Self { grammar, inner }
    }

    /// Construct a fresh matcher directly from GBNF source. Returns the
    /// parse error if the grammar is malformed.
    pub fn from_source(source: &str) -> Result<Self, GrammarError> {
        Ok(Self::new(Arc::new(Grammar::parse(source)?)))
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

    /// True iff feeding `bytes` would succeed AND leave the matcher in
    /// an accepting state — the EOG-doubles-as-exit test (see
    /// `grammar_filter`'s EOG policy). Does not mutate `self`.
    pub fn completes_with(&self, bytes: &[u8]) -> bool {
        self.inner.completes_with(&self.grammar, bytes)
    }

    /// 256-bit bitmap indexed by byte value: bit `b` is set iff feeding
    /// byte `b` next could plausibly extend the match into a codepoint
    /// accepted by at least one active stack. See
    /// `StackState::first_byte_bitmap` for details.
    ///
    /// Conservative: a set bit means "maybe accepted" and still needs
    /// [`Self::accepts_bytes`] confirmation; a cleared bit is a definite
    /// rejection. Fuzzers use this to enumerate plausible next bytes
    /// without paying for 256 full clone-and-advance probes.
    pub fn first_byte_bitmap(&self) -> [u64; 4] {
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

    pub(crate) fn is_complete(&self) -> bool {
        self.pending.is_empty() && self.stacks.iter().any(|s| s.is_empty())
    }

    /// Approximate heap bytes this state holds, as the DFA cache
    /// interns it twice (key and table): each stack inline, plus the
    /// capacity of a stack spilled past [`STACK_INLINE`].
    pub(crate) fn weight(&self) -> usize {
        let spilled: usize = self
            .stacks
            .iter()
            .filter(|s| s.len() > STACK_INLINE)
            .map(|s| s.capacity() * std::mem::size_of::<Position>())
            .sum();
        2 * (self.stacks.len() * std::mem::size_of::<Stack>() + spilled)
    }

    /// [`Self::is_complete`] and nothing can extend the match: every
    /// live stack is spent. `"ab"+` after one "ab" is complete but not
    /// exhausted — another "ab" is still legal; `"ab"` after "ab" is
    /// both.
    pub(crate) fn is_exhausted(&self) -> bool {
        self.pending.is_empty()
            && !self.stacks.is_empty()
            && self.stacks.iter().all(|s| s.is_empty())
    }

    /// True iff feeding `bytes` would succeed from the current state.
    /// Clones only the matcher state — the `Arc<Grammar>` is not touched.
    pub(crate) fn accepts_bytes(
        &self,
        grammar: &Grammar,
        bytes: &[u8],
    ) -> bool {
        let mut scratch = self.clone();
        scratch.advance_bytes(grammar, bytes).is_ok()
    }

    /// [`Self::accepts_bytes`] + the scratch state ends accepting.
    pub(crate) fn completes_with(
        &self,
        grammar: &Grammar,
        bytes: &[u8],
    ) -> bool {
        let mut scratch = self.clone();
        scratch.advance_bytes(grammar, bytes).is_ok() && scratch.is_complete()
    }

    /// True iff the current state is a *permissive* free region — a span
    /// like a JSON string body or an `until()` raw value where the grammar
    /// accepts nearly any next byte and the model, not the grammar, owns
    /// the content. Proxy: popcount of the first-byte bitmap against
    /// [`PERMISSIVE_MIN_POPCOUNT`]. Misreads are safe in both directions:
    /// structural→permissive still exempts every region-exit token via the
    /// protected walk, and permissive→structural is exactly the pre-feature
    /// behavior (penalty suspended). See `sample::region`.
    pub(crate) fn is_permissive(&self, grammar: &Grammar) -> bool {
        let bm = self.first_byte_bitmap(grammar);
        bm.iter().map(|w| w.count_ones()).sum::<u32>()
            >= PERMISSIVE_MIN_POPCOUNT
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

    pub(crate) fn advance_bytes(
        &mut self,
        grammar: &Grammar,
        bytes: &[u8],
    ) -> Result<(), GrammarError> {
        for &b in bytes {
            self.feed_byte(grammar, b)?;
        }
        if !self.pending.is_empty() && !self.pending_can_still_match(grammar) {
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

    // `pub(crate)` for the region guard's uncached walk (see
    // `sample::region`); everything else goes through `advance_bytes`.
    pub(crate) fn feed_byte(
        &mut self,
        grammar: &Grammar,
        b: u8,
    ) -> Result<(), GrammarError> {
        // `pending` is a fixed-capacity 4-byte buffer. If the previous
        // call left it full (e.g. an `Invalid` result on the 4th byte
        // returned without clearing), pushing here would panic. Defend
        // against stale state by clearing on a full buffer — a 5th
        // pending byte is structurally impossible in any valid UTF-8
        // codepoint, so resetting and treating `b` as a fresh start is
        // semantically equivalent to "previous incomplete sequence was
        // garbage." Surfaced by the in-tree fuzzer.
        if self.pending.len() == self.pending.capacity() {
            self.pending.clear();
        }
        self.pending.push(b);
        match decode_utf8(self.pending.as_slice()) {
            Utf8Decode::Complete(cp) => {
                self.pending.clear();
                self.consume(grammar, cp)
            }
            Utf8Decode::Incomplete => Ok(()),
            Utf8Decode::Invalid => {
                // Clear so the next call starts fresh — see comment
                // above; without this, a follow-up `feed_byte` on the
                // same state panics.
                self.pending.clear();
                Err(GrammarError::InvalidUtf8)
            }
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
        // Bound the walk: a left-recursive grammar (`a ::= a "x"`) never
        // runs out of epsilon steps, each one a stack a frame deeper.
        // The bound is on frames copied, the walk's real cost, not on
        // steps: a step-per-starting-stack budget (4096 each) cut a
        // wide alternation's expansion short — a 4000-property
        // all-optional object lost the stack that closes `{}`.
        let mut work = 0usize;
        for _ in 0..EXPAND_MAX_STEPS {
            // Far past the cap: the rest could only be truncated away.
            if result.len() >= 4 * MAX_STACKS || work > EXPAND_MAX_WORK {
                break;
            }
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
                        work += stack.len();
                        let mut branched = clone_stack(&stack, 1);
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
        // Over a cap: keep a deterministic subset (the sort's prefix),
        // so the DFA cache's interned states stay canonical. A stack
        // deeper than the frame cap on its own is dropped too, so a
        // state's size has a bound however deep the output nests (see
        // `MAX_STATE_FRAMES`); with nothing left, the byte is refused.
        let mut frames = 0usize;
        let keep = result
            .iter()
            .take_while(|stack| {
                frames += stack.len();
                frames <= MAX_STATE_FRAMES
            })
            .count()
            .min(MAX_STACKS);
        result.truncate(keep);
        self.stacks = result;
    }
}

/// Most stacks a matcher state keeps. Ambiguity multiplies stacks:
/// two interchangeable recursive rules (`N1 = N2 = {"c": N1 | N2}`)
/// double them at every nesting level, which no dedup can merge — the
/// stacks differ in which rule each frame is in — so 18 levels of a
/// client's schema held 393,216 stacks and took over a second a
/// byte (the hostile-schema recheck). Past the cap the extra stacks are
/// dropped: each stack is one way the input so far can continue, so a
/// subset only ever admits *fewer* continuations — never a byte the
/// full set would reject. At worst it over-restricts, which surfaces
/// as the existing grammar-violation path.
///
/// Real grammars stay far below it (the test suite peaks under 40);
/// the one legitimate shape near it is a single `enum` or all-optional
/// object of thousands of members, each member one stack until its
/// first distinguishing byte.
pub(crate) const MAX_STACKS: usize = 4096;

/// Most frames, summed over its stacks, a matcher state keeps — the
/// same cap as [`MAX_STACKS`] (and the same over-restriction) on the
/// other axis. Capped stacks still deepen with the output's nesting:
/// 4096 of them 200 levels into an ambiguous recursive schema are
/// ~20 MB, copied on every byte, and a long enough generation would
/// reach gigabytes.
///
/// 2^16 is [`MAX_STACKS`] stacks 16 frames deep. The widest states a
/// real grammar makes — an `enum`, or optional properties, of
/// thousands, a stack a member — are under 9 frames a stack in every
/// dialect's tool call nested three objects deep (measured), so this
/// cap does not bind before [`MAX_STACKS`] does. It is ~32,000 levels
/// of `[` (two frames a level), far past what `serde_json` reads back
/// (128 by default). Output nested deeper than the cap allows, a lone
/// stack included, is refused: the existing violation path.
///
/// At 2^18, with a lone stack exempt, the recheck's ambiguous schema
/// filtered every 50 levels to 400 deep took ~10 s over a 75k-token
/// vocabulary and held 1.3 GB resident. At 2^16 (with
/// [`DFA_CACHE_MAX_WEIGHT`] and `clone_stack`'s block sizes) it holds
/// ~800 MB resident, under 300 MB of physical footprint, filtering
/// *every* level to 6,000 deep at 10–25 ms a step; and `[` nested
/// 20,000 deep no longer copies a 40,000-frame stack per byte. 2^15
/// was cheaper still, but over-restricted a 2,000-member `enum` three
/// objects deep in a Hermes tool call (34,000 frames).
pub(crate) const MAX_STATE_FRAMES: usize = 1 << 16;

/// Upper bound on [`StackState::weight`] under the caps: every stack
/// inline, plus every frame spilled, at up to twice its length
/// (`clone_stack`).
const MAX_STATE_WEIGHT: usize = 2
    * (MAX_STACKS * std::mem::size_of::<Stack>()
        + 2 * MAX_STATE_FRAMES * std::mem::size_of::<Position>());

/// Most queue steps one [`StackState::expand`] takes.
const EXPAND_MAX_STEPS: usize = 1 << 20;

/// Most stack frames one [`StackState::expand`] copies branching
/// stacks — the walk's cost in time and memory. A left-recursive
/// grammar spends it in a few thousand steps (each copy a frame
/// deeper); a real grammar's widest expansion (thousands of
/// alternatives a few frames deep) spends well under 1%.
const EXPAND_MAX_WORK: usize = 1 << 22;

// ===========================================================================
// Lazy-DFA cache
// ===========================================================================

/// Interned identifier for a canonical [`StackState`]. Returned by
/// [`DfaCache::intern`] and used as the key into the byte-transition table.
pub(crate) type StateId = u32;

/// Sentinel returned by [`DfaCache::transition`] when feeding the byte leaves
/// the matcher with no surviving stacks (i.e. the byte is rejected).
pub(crate) const REJECT_STATE: StateId = u32::MAX;

/// Sentinel returned by [`DfaCache::intern_base`] and
/// [`DfaCache::transition`] when the state the byte leads to was not
/// interned — the cache is at its hard weight cap for this step, or
/// the state alone is past the per-state threshold
/// ([`DFA_CACHE_STATE_SHARE`]). Not a rejection: the caller walks that
/// input on the matcher itself (the clone-walk path), which costs
/// time, not cache memory. Never pass it to the cache's per-state
/// queries.
pub(crate) const UNCACHED_STATE: StateId = u32::MAX - 1;

/// Minimum set bits in a state's first-byte bitmap for the state to count
/// as a permissive "free region" (see [`StackState::is_permissive`]).
/// Measured margins on the built-in grammars: JSON string body ≈ 147 set
/// bits, `until()` raw-value states ≈ 179; structural states (awaiting
/// key/colon/comma, numbers, literals, whitespace) ≤ ~25. The threshold is
/// deliberately non-critical — see the safety-asymmetry note on
/// `is_permissive`. Mid-UTF-8 states (pending continuation bytes, exactly
/// 64 legal next bytes) land permissive, which is the safe side.
pub(crate) const PERMISSIVE_MIN_POPCOUNT: u32 = 64;

struct DfaInterned {
    /// Canonical `StackState` → `StateId`. Canonical = post-`expand`, so
    /// stacks are sorted + deduped.
    intern: FxHashMap<StackState, StateId>,
    /// Id → canonical `StackState`. Needed on transition misses to
    /// reconstitute the matcher, feed a byte, and re-intern the result.
    states: Vec<StackState>,
    /// Approximate bytes the interned states hold ([`StackState::weight`]).
    weight: usize,
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
    /// [`DFA_CACHE_MAX_WEIGHT`], lowered by tests.
    max_weight: usize,
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

/// Interned-state cap for a config-homed (Session-lifetime) cache.
/// Non-recursive grammars plateau at a few hundred states, but a
/// recursive grammar (arbitrarily nested JSON) mints a fresh state per
/// nesting depth, so an uncapped cache grows without bound across
/// turns. Clearing is always safe — the cache is pure memoization —
/// so on exceed we restart cold. Soft cap: checked at the base-state
/// intern (single-threaded point, once per sampled token); one token
/// step may overshoot by its own transitions, which is noise.
const DFA_CACHE_MAX_STATES: usize = 65_536;

/// Interned-state *weight* cap ([`StackState::weight`], ~bytes) for
/// the same cache. The state cap alone assumed small states, but a
/// state can hold up to [`MAX_STACKS`] stacks of up to
/// [`MAX_STATE_FRAMES`] frames between them, a few MB: 65,536 of
/// those is hundreds of GiB.
///
/// Soft at the base intern, like [`DFA_CACHE_MAX_STATES`]: past it the
/// next step restarts cold. Hard at twice that inside a step
/// ([`DfaCache::intern`]): a step that would intern more gets
/// [`UNCACHED_STATE`] instead, so no step grows the cache past it,
/// however many heavy states the vocabulary's prefixes reach. 32 MiB
/// holds every real grammar's whole working set (a few hundred states
/// of a few hundred bytes) and tens of the heaviest states; the
/// recheck's ambiguous schema, filtered every level 2,400 deep, peaks
/// at ~50 MB live where 256 MiB let it reach ~270 MB.
const DFA_CACHE_MAX_WEIGHT: usize = 32 << 20;

/// A state heavier than `1 / DFA_CACHE_STATE_SHARE` of the cache's
/// weight cap is never interned ([`UNCACHED_STATE`]): one state must
/// not be able to take the cache by itself. Under the matcher's caps
/// no state gets there ([`MAX_STATE_WEIGHT`] is ~4 MB, the threshold
/// 8 MiB) — it bites only if those caps are raised, and then a heavy
/// state costs clone-walk time rather than cache memory.
const DFA_CACHE_STATE_SHARE: usize = 4;

static_assertions::const_assert!(
    MAX_STATE_WEIGHT <= DFA_CACHE_MAX_WEIGHT / DFA_CACHE_STATE_SHARE
);

impl DfaCache {
    pub(crate) fn new() -> Self {
        Self {
            interned: RwLock::new(DfaInterned {
                intern: FxHashMap::default(),
                states: Vec::new(),
                weight: 0,
            }),
            max_weight: DFA_CACHE_MAX_WEIGHT,
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

    /// Intern the per-token-step base state, enforcing the growth cap
    /// first. MUST only be called from the single-threaded point of the
    /// sampling step (before the rayon fold): the clear invalidates
    /// every outstanding `StateId`, which is only safe when none are
    /// live. Do NOT call concurrently with `transition`.
    ///
    /// [`UNCACHED_STATE`] when even a cold cache won't take the state
    /// (it is past the per-state threshold): the step runs uncached.
    pub(crate) fn intern_base(&self, state: &StackState) -> StateId {
        let over = {
            let g = self.interned.read().unwrap();
            g.states.len() > DFA_CACHE_MAX_STATES || g.weight > self.max_weight
        };
        if over {
            let mut g = self.interned.write().unwrap();
            g.intern.clear();
            g.states.clear();
            g.weight = 0;
            self.transitions.clear();
            self.bitmaps.clear();
            self.complete.clear();
            self.terminal_valid.clear();
        }
        self.intern(state)
    }

    /// Intern a canonical `StackState`, returning its `StateId`. Reads fast-
    /// path under a read lock; inserts on miss under a write lock with a
    /// double-check to tolerate racing inserters under rayon.
    ///
    /// [`UNCACHED_STATE`] instead of growing the cache past twice its
    /// weight cap, or for a state past the per-state threshold (see
    /// [`DFA_CACHE_MAX_WEIGHT`]). Only the base intern may clear the
    /// cache, so this is where a step's growth stops.
    fn intern(&self, state: &StackState) -> StateId {
        if let Some(&id) = self.interned.read().unwrap().intern.get(state) {
            return id;
        }
        let weight = state.weight();
        if weight > self.max_weight / DFA_CACHE_STATE_SHARE {
            return UNCACHED_STATE;
        }
        let mut g = self.interned.write().unwrap();
        if let Some(&id) = g.intern.get(state) {
            return id;
        }
        if g.weight + weight > 2 * self.max_weight {
            return UNCACHED_STATE;
        }
        let id = g.states.len() as StateId;
        debug_assert!(id < UNCACHED_STATE, "state id overflow");
        g.weight += weight;
        g.states.push(state.clone());
        g.intern.insert(state.clone(), id);
        id
    }

    /// A cache whose weight cap is `max_weight`, not
    /// [`DFA_CACHE_MAX_WEIGHT`].
    #[cfg(test)]
    pub(crate) fn with_max_weight(max_weight: usize) -> Self {
        Self {
            max_weight,
            ..Self::new()
        }
    }

    /// Fill the cache to its hard cap without interning anything, so
    /// every state not yet interned comes back [`UNCACHED_STATE`] until
    /// the next base intern clears it.
    #[cfg(test)]
    pub(crate) fn saturate(&self) {
        self.interned.write().unwrap().weight = 2 * self.max_weight;
    }

    /// The interned states' weight ([`StackState::weight`]).
    #[cfg(test)]
    pub(crate) fn weight(&self) -> usize {
        self.interned.read().unwrap().weight
    }

    /// Reconstitute the `StackState` for a given id. Used only on cache
    /// misses; the hot path never calls this.
    pub(crate) fn state_of(&self, id: StateId) -> StackState {
        self.interned.read().unwrap().states[id as usize].clone()
    }

    /// Feed a byte from a state, returning the next state id (or
    /// `REJECT_STATE`, or [`UNCACHED_STATE`] when the next state was not
    /// interned). Hit path is a single `DashMap::get`.
    pub(crate) fn transition(
        &self,
        grammar: &Grammar,
        sid: StateId,
        byte: u8,
    ) -> StateId {
        if sid == REJECT_STATE || sid == UNCACHED_STATE {
            return sid;
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
        if sid == REJECT_STATE || sid == UNCACHED_STATE {
            debug_assert_ne!(sid, UNCACHED_STATE, "an uncached state");
            return false;
        }
        if let Some(entry) = self.terminal_valid.get(&sid) {
            return *entry;
        }
        let state = self.state_of(sid);
        let v =
            state.pending.is_empty() || state.pending_can_still_match(grammar);
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
        if sid == REJECT_STATE || sid == UNCACHED_STATE {
            debug_assert_ne!(sid, UNCACHED_STATE, "an uncached state");
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

    /// Memoized [`StackState::is_permissive`] for an interned state. Rides
    /// the `bitmaps` memo (the popcount on a hit is noise) — deliberately
    /// NOT a separate map, so there is no extra entry to keep in
    /// `intern_base`'s cold-clear block. `REJECT_STATE` → `false`.
    pub(crate) fn is_permissive(
        &self,
        grammar: &Grammar,
        sid: StateId,
    ) -> bool {
        if sid == REJECT_STATE || sid == UNCACHED_STATE {
            debug_assert_ne!(sid, UNCACHED_STATE, "an uncached state");
            return false;
        }
        let bm = self.first_byte_bitmap(grammar, sid);
        bm.iter().map(|w| w.count_ones()).sum::<u32>()
            >= PERMISSIVE_MIN_POPCOUNT
    }

    /// True iff the state is an accepting state (empty pending + at least one
    /// empty stack).
    ///
    /// Used by `grammar_filter`'s EOG exemption: an end-of-generation
    /// token survives mid-parse only when the DFA state after its
    /// piece bytes is accepting (a dialect exit marker doubling as a
    /// stop token).
    pub(crate) fn is_complete(&self, sid: StateId) -> bool {
        if sid == REJECT_STATE || sid == UNCACHED_STATE {
            debug_assert_ne!(sid, UNCACHED_STATE, "an uncached state");
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
/// first access. `pub(crate)` so `sample::region`'s guard walks take the
/// same path as `grammar_filter` — the flag must never change streams.
pub(crate) fn dfa_cache_enabled() -> bool {
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
#[non_exhaustive]
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
    #[error(
        "GBNF parser recursion exceeded {limit} levels at byte {pos} \
         (deeply-nested groups / alternations); likely a malicious or \
         pathological input"
    )]
    RecursionLimit { pos: usize, limit: usize },
    #[error("I/O error reading `{path}`: {err}")]
    Io {
        path: std::path::PathBuf,
        err: String,
    },
    #[error("internal matcher inconsistency")]
    Internal,
    /// The grammar is past [`Grammar::parse`]'s size limit. From a
    /// compiled JSON Schema, the schema is too complex to constrain.
    #[error(
        "grammar is too large (over {limit} {what}); the schema it was \
         compiled from is too complex"
    )]
    TooLarge { what: &'static str, limit: usize },
}

static_assertions::assert_impl_all!(GrammarError: Send, Sync);
static_assertions::assert_impl_all!(Grammar: Send, Sync);
static_assertions::assert_impl_all!(GrammarState: Send, Sync);

// ===========================================================================
// Runtime-gated filter-call statistics (opt in via env var)
// ===========================================================================

/// Cumulative statistics about `grammar_filter` calls since process start
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
    /// Number of `grammar_filter` calls recorded.
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
    /// Number of lazy sample-then-check verifications performed
    /// (`SamplerConfig::lazy_grammar`). In lazy mode this counts emitted
    /// constrained tokens; `lazy_hits / lazy_checks` is the fast-path
    /// acceptance rate.
    pub lazy_checks: u64,
    /// Lazy checks where the unconstrained pick was grammar-legal (no
    /// fallback filter needed).
    pub lazy_hits: u64,
    /// Lazy checks that rejected the pick and re-ran the full masked
    /// path. In lazy mode the filter counters above measure these
    /// fallback invocations only.
    pub lazy_fallbacks: u64,
    /// Sum over lazy checks of wall-clock verification time, µs.
    pub check_us_sum: u64,
    /// Maximum across lazy checks of wall-clock verification time, µs.
    pub check_us_max: u64,
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
    lazy_checks: AtomicU64,
    lazy_hits: AtomicU64,
    lazy_fallbacks: AtomicU64,
    check_us_sum: AtomicU64,
    check_us_max: AtomicU64,
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
    lazy_checks: AtomicU64::new(0),
    lazy_hits: AtomicU64::new(0),
    lazy_fallbacks: AtomicU64::new(0),
    check_us_sum: AtomicU64::new(0),
    check_us_max: AtomicU64::new(0),
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

/// Snapshot cumulative `grammar_filter` statistics. Returns zeros when
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
        dfa_transition_hits: STATS.dfa_transition_hits.load(Ordering::Relaxed),
        dfa_transition_misses: STATS
            .dfa_transition_misses
            .load(Ordering::Relaxed),
        dfa_bitmap_hits: STATS.dfa_bitmap_hits.load(Ordering::Relaxed),
        dfa_bitmap_misses: STATS.dfa_bitmap_misses.load(Ordering::Relaxed),
        lazy_checks: STATS.lazy_checks.load(Ordering::Relaxed),
        lazy_hits: STATS.lazy_hits.load(Ordering::Relaxed),
        lazy_fallbacks: STATS.lazy_fallbacks.load(Ordering::Relaxed),
        check_us_sum: STATS.check_us_sum.load(Ordering::Relaxed),
        check_us_max: STATS.check_us_max.load(Ordering::Relaxed),
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
    STATS.lazy_checks.store(0, Ordering::Relaxed);
    STATS.lazy_hits.store(0, Ordering::Relaxed);
    STATS.lazy_fallbacks.store(0, Ordering::Relaxed);
    STATS.check_us_sum.store(0, Ordering::Relaxed);
    STATS.check_us_max.store(0, Ordering::Relaxed);
}

/// Record one lazy sample-then-check verification
/// (`SamplerConfig::lazy_grammar`). No-op unless
/// [`grammar_stats_enabled`]; callers pass the elapsed time only when
/// they measured it (i.e. stats were enabled at check time).
pub(crate) fn record_lazy(hit: bool, elapsed_us: u64) {
    STATS.lazy_checks.fetch_add(1, Ordering::Relaxed);
    if hit {
        STATS.lazy_hits.fetch_add(1, Ordering::Relaxed);
    } else {
        STATS.lazy_fallbacks.fetch_add(1, Ordering::Relaxed);
    }
    STATS.check_us_sum.fetch_add(elapsed_us, Ordering::Relaxed);
    atomic_fetch_max(&STATS.check_us_max, elapsed_us);
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
    STATS
        .candidates_in
        .fetch_add(candidates_in, Ordering::Relaxed);
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
pub(crate) fn grammar_filter<M: Model + Sync>(
    candidates: Candidates,
    compiled: &CompiledGrammar,
    matcher: &StackState,
    model: &M,
) -> Candidates {
    // Each candidate check is independent: clone the matcher state,
    // try to advance it by the token's bytes, keep the token iff the
    // clone survives. Fan out across rayon's global pool so 150k-vocab
    // models don't bottleneck on a single core. `GrammarState` is
    // auto-Sync (pure data behind an Arc); the `M: Sync` bound on this
    // function ensures the model can be borrowed across the parallel
    // fold.
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

    let grammar: &Grammar = &compiled.grammar;
    let inner: &StackState = matcher;
    let cache_on = dfa_cache_enabled();
    let cache: &Arc<DfaCache> = &compiled.dfa;

    // Fast path: the lazy-DFA cache memoizes one-byte transitions and the
    // first-byte bitmap per canonical matcher state. Intern the base state
    // up-front (growth-capped: the config-homed cache lives for the
    // Session, not the call) so every candidate walks the same
    // transition table from the same state id.
    let base_id = if cache_on {
        cache.intern_base(inner)
    } else {
        0
    };
    // A base the cache won't hold runs this step uncached.
    let cache_on = cache_on && base_id != UNCACHED_STATE;
    let bitmap = if cache_on {
        cache.first_byte_bitmap(grammar, base_id)
    } else {
        inner.first_byte_bitmap(grammar)
    };

    #[derive(Default)]
    struct Acc {
        kept: Vec<TokenData>,
        bitmap_pass: u64,
    }

    // Drop empty-piece tokens unconditionally: zero bytes can never
    // advance the matcher, so they trivially pass it in ANY state.
    // Post-complete they kept the force-EOS branch below from firing
    // (invisible reserved-token budget burn); mid-parse they are
    // livelock fuel — an unpenalized zero-byte token can dominate
    // once the repetition penalty crushes the structural tokens the
    // grammar keeps forcing.
    //
    // Mid-parse, additionally drop end-of-generation tokens BY ID —
    // UNLESS the token's own piece bytes finish the constraint.
    // Byte-acceptance alone is the wrong test for EOG: inside a raw
    // `until()` region every byte sequence is legal value content —
    // including the literal piece bytes of `<|im_end|>` — so the
    // matcher can never reject EOG and the predictor's stop fires
    // mid-structure (observed on Qwen3.6: every sampled tool call
    // died at the second parameter when the model bailed with
    // `<|im_end|>`; greedy was immune). An active constraint owns
    // termination: EOG becomes legal again once the grammar reaches
    // an accept state — the same rule llama.cpp's grammar sampler
    // applies. The completes-with exemption covers dialect exit
    // markers that libllama also marks EOG (Gemma 4's
    // `<|tool_response>`, plan Phase G postmortem): the grammar
    // REQUIRES the marker as its final bytes, so rejecting it by id
    // left the model with no legal path and generation burned to the
    // context limit. Harmony's `<|return|>`/`<|call|>` need no
    // exemption — the grammar is already complete when they appear.
    //
    // The set is [`Model::eog_tokens`] and nothing else. Unioning in
    // `eot()` was the same class of bug from the other side: gpt-oss's
    // EOT is `<|end|>`, which is NOT EOG, and masking it mid-grammar
    // left a Harmony model unable to close its analysis channel — it
    // rambled ("Let's do. We'll call. We'll output.") to `max_tokens`.
    //
    // At an accept state EOG is legal BY ID, whatever its bytes — and
    // that has to hold when the accept is *extensible*, not just
    // terminal. After the first call of `call+` the grammar is
    // accepting yet a next opener is still legal, so `kept` is never
    // empty and the force-EOS branch below never fires; judged by its
    // bytes there, EOG (an empty piece, or `</s>` against an opener's
    // first byte) was unreachable and the model was forced into call
    // after call until the budget — 26 of them on Mistral Small 4,
    // read for a night as a model preference. EOG-or-next-opener is
    // the model's decision; this is where it gets to make it.
    let complete = inner.is_complete();
    let eog = model.eog_tokens();

    let acc = candidates
        .as_slice()
        .par_iter()
        .fold(Acc::default, |mut a, cand| {
            let cand_is_eog = eog.contains(&cand.id);
            if cand_is_eog && complete {
                a.kept.push(*cand);
                return a;
            }
            // Past this point an EOG candidate is mid-parse: kept
            // below only when its own bytes finish the constraint
            // (exit-marker case).
            let mut buf: Vec<u8> = Vec::with_capacity(32);
            model.token_to_piece_ref(cand.id, &mut buf);
            if buf.is_empty() {
                return a;
            }
            if let Some(&first) = buf.first() {
                if bitmap[(first as usize) >> 6] & (1u64 << (first & 63)) == 0 {
                    return a;
                }
            }
            a.bitmap_pass += 1;
            let uncached = || match cand_is_eog {
                true => inner.completes_with(grammar, &buf),
                false => inner.accepts_bytes(grammar, &buf),
            };
            let accepts = if cache_on {
                let sid = buf.iter().try_fold(base_id, |sid, &b| {
                    match cache.transition(grammar, sid, b) {
                        REJECT_STATE => None,
                        next => Some(next),
                    }
                });
                match sid {
                    None => false,
                    // The cache is full this step: walk the matcher.
                    Some(UNCACHED_STATE) => uncached(),
                    Some(sid) if cand_is_eog => cache.is_complete(sid),
                    Some(sid) => cache.terminal_valid(grammar, sid),
                }
            } else {
                uncached()
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
    #[cfg(feature = "axum")]
    tracing::debug!(
        target: "drama_llama::sample::grammar",
        candidates_in = candidates_in,
        bitmap_pass = acc.bitmap_pass,
        kept = kept.len(),
        "grammar_filter",
    );
    if kept.is_empty() {
        // Success termination and grammar violation converge on the
        // same force-EOS shape; matcher state is per-call now (fresh
        // via `SamplerConfig::init_state`), so there is no
        // reset-for-the-next-generation step — that contract died with
        // the config/state split.
        let eos = TokenData {
            id: model.eos(),
            logit: 0.0,
            p: 1.0,
        };
        return Candidates::from_vec(vec![eos]);
    }

    Candidates::from_vec_unchecked(kept)
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

    /// Two interchangeable recursive defs (`N1 = N2 = {"c": N1 | N2}`):
    /// every `{"c":` level doubles the ways the input so far can
    /// continue. 18 levels used to hold 393,216 stacks (over a second a
    /// byte); 40 would never finish. Capped, the matcher stays within
    /// [`MAX_STACKS`] and [`MAX_STATE_FRAMES`] and still accepts the
    /// whole document — every stack it dropped was interchangeable with
    /// one it kept.
    fn ambiguous_recursive_grammar() -> Arc<Grammar> {
        let node = serde_json::json!({
            "type": "object",
            "properties": {"c": {"anyOf": [
                {"$ref": "#/$defs/N1"},
                {"$ref": "#/$defs/N2"},
            ]}},
        });
        let schema = serde_json::json!({
            "$ref": "#/$defs/N1",
            "$defs": {"N1": node.clone(), "N2": node},
        });
        let mut src = String::from("root ::= s\n");
        crate::schema_to_gbnf(&schema, "s", &mut src).unwrap();
        src.push_str(crate::JSON_GRAMMAR);
        Arc::new(parse_ok(&src))
    }

    #[test]
    fn ambiguous_recursion_keeps_stacks_capped() {
        let mut state = GrammarState::new(ambiguous_recursive_grammar());
        let depth = 120;
        let start = Instant::now();
        for level in 1..=depth {
            state.advance_bytes(br#"{"c":"#).unwrap();
            assert!(
                state.stack_depth() <= MAX_STACKS,
                "level {level}: {} stacks",
                state.stack_depth()
            );
            let frames: usize =
                state.inner.stacks.iter().map(|s| s.len()).sum();
            assert!(frames <= MAX_STATE_FRAMES, "level {level}: {frames}");
        }
        state.advance_bytes(b"{}").unwrap();
        for _ in 0..depth {
            state.advance_bytes(b"}").unwrap();
        }
        assert!(state.is_complete());
        // Generous: capped, this takes milliseconds; uncapped it does
        // not finish.
        assert!(start.elapsed().as_secs() < 30, "{:?}", start.elapsed());
    }

    /// Truncation only removes continuations. An `enum` wider than
    /// [`MAX_STACKS`] keeps some of its members' stacks (the old expand
    /// budget, 4096 steps a starting stack, already cut one past ~2000
    /// members short, less predictably): those members still match,
    /// the rest are over-restricted away, and nothing outside the enum
    /// is ever admitted.
    #[test]
    fn capped_state_admits_only_valid_bytes() {
        let n = MAX_STACKS + 1000;
        let members: Vec<String> = (0..n).map(|i| format!("m{i:05}")).collect();
        let schema = serde_json::json!({"enum": members});
        let mut src = String::from("root ::= s\n");
        crate::schema_to_gbnf(&schema, "s", &mut src).unwrap();
        src.push_str(crate::JSON_GRAMMAR);
        let grammar = Arc::new(parse_ok(&src));
        let root = GrammarState::new(grammar);
        assert!(root.stack_depth() <= MAX_STACKS);
        let accepted = |text: &str| {
            let mut state = root.clone();
            state.advance_bytes(text.as_bytes()).is_ok() && state.is_complete()
        };
        let kept = members
            .iter()
            .filter(|m| accepted(&format!("\"{m}\"")))
            .count();
        assert_eq!(kept, MAX_STACKS);
        let outsiders = (0..n).flat_map(|i| {
            [
                format!("\"m{i:05}x\""),
                format!("\"m{i:04}\""),
                format!("\"n{i:05}\""),
            ]
        });
        for outsider in outsiders.chain([r#""m""#.into(), r#""m99999""#.into()])
        {
            assert!(!accepted(&outsider), "{outsider}");
        }
    }

    /// The DFA cache's weight cap: interning one heavy state after
    /// another, the cache restarts cold at the base intern instead of
    /// holding them all (65,536 capped states of a deep ambiguous
    /// grammar would be tens of GiB).
    #[test]
    fn dfa_cache_weight_capped() {
        let grammar = ambiguous_recursive_grammar();
        let max = 4 << 20;
        let cache = DfaCache::with_max_weight(max);
        let mut state = StackState::new_rooted(&grammar);
        let mut heaviest = 0;
        for _ in 0..120 {
            state.advance_bytes(&grammar, br#"{"c":"#).unwrap();
            heaviest = heaviest.max(state.weight());
            cache.intern_base(&state);
            assert!(cache.weight() <= max + heaviest, "{}", cache.weight());
        }
        assert!(heaviest > 0);
    }

    /// The cap holds inside a step too, where the base intern cannot
    /// clear: transitions to fresh heavy states stop interning at twice
    /// the weight cap and answer [`UNCACHED_STATE`] instead.
    #[test]
    fn dfa_cache_hard_cap_within_a_step() {
        let grammar = ambiguous_recursive_grammar();
        let mut state = StackState::new_rooted(&grammar);
        for _ in 0..40 {
            state.advance_bytes(&grammar, br#"{"c":"#).unwrap();
        }
        let max = 4 * state.weight();
        let cache = DfaCache::with_max_weight(max);
        let base = cache.intern_base(&state);
        assert_ne!(base, UNCACHED_STATE);
        // Every `{"c":` from here is a fresh, heavier state.
        let (mut sid, mut uncached) = (base, false);
        for _ in 0..40 {
            for &b in br#"{"c":"# {
                sid = cache.transition(&grammar, sid, b);
            }
            assert_ne!(sid, REJECT_STATE);
            assert!(cache.weight() <= 2 * max, "{}", cache.weight());
            if sid == UNCACHED_STATE {
                uncached = true;
                break;
            }
        }
        assert!(uncached, "the cache never filled");
        // Uncached stays uncached; the next step's base clears it.
        assert_eq!(cache.transition(&grammar, sid, b'{'), UNCACHED_STATE);
        assert_ne!(cache.intern_base(&state), UNCACHED_STATE);
        assert!(cache.weight() <= max);
    }

    /// A model whose pieces are `PIECES`, for driving [`grammar_filter`].
    struct Pieces(&'static [&'static str]);

    impl crate::backend::Model for Pieces {
        type Error = std::convert::Infallible;
        fn n_vocab(&self) -> i32 {
            self.0.len() as i32
        }
        fn bos(&self) -> crate::Token {
            0
        }
        fn eos(&self) -> crate::Token {
            0
        }
        fn eot(&self) -> crate::Token {
            0
        }
        fn special_tokens(&self) -> Vec<crate::Token> {
            vec![0]
        }
        fn eog_tokens(&self) -> Vec<crate::Token> {
            vec![0]
        }
        fn max_token_len(&self) -> usize {
            16
        }
        fn tokenize(&self, _: &str, _: bool) -> Vec<crate::Token> {
            unimplemented!("not needed by the filter")
        }
        fn token_to_piece(&self, token: crate::Token) -> String {
            self.0[token as usize].to_string()
        }
        fn token_to_piece_ref(&self, token: crate::Token, buf: &mut Vec<u8>) {
            buf.clear();
            buf.extend_from_slice(self.0[token as usize].as_bytes());
        }
        fn context_size(&self) -> i32 {
            4096
        }
        fn chat_template_source(&self) -> Option<String> {
            None
        }
        fn recommended_sampling(&self) -> crate::SamplingParams {
            crate::SamplingParams::default()
        }
    }

    /// A state past the cache's per-state share is never interned: the
    /// filter walks the matcher for it, and keeps exactly the tokens a
    /// cache that holds it keeps.
    #[test]
    fn filter_falls_back_to_the_matcher_past_the_cache() {
        let members: Vec<String> =
            (0..MAX_STACKS).map(|i| format!("m{i:04}")).collect();
        // One stack before the `<`, thousands after it.
        let mut src = String::from("root ::= \"<\" s\n");
        crate::schema_to_gbnf(
            &serde_json::json!({"enum": members}),
            "s",
            &mut src,
        )
        .unwrap();
        src.push_str(crate::JSON_GRAMMAR);
        let grammar = Arc::new(parse_ok(&src));
        let root = StackState::new_rooted(&grammar);
        let mut inside = root.clone();
        inside.advance_bytes(&grammar, b"<").unwrap();
        // Room for the root, not for any state inside the quote.
        let max =
            DFA_CACHE_STATE_SHARE * root.weight().max(inside.weight() / 2);
        assert!(inside.weight() > max / DFA_CACHE_STATE_SHARE);
        let capped = CompiledGrammar {
            grammar: grammar.clone(),
            dfa: Arc::new(DfaCache::with_max_weight(max)),
        };
        let base = capped.dfa.intern_base(&root);
        assert_ne!(base, UNCACHED_STATE);
        assert_eq!(capped.dfa.transition(&grammar, base, b'<'), UNCACHED_STATE);
        let uncapped = CompiledGrammar::from_grammar(grammar);
        let model = Pieces(&[
            "",
            "<",
            "<\"m",
            "<\"m0001\"",
            "<\"m4095\"",
            "<\"m9999\"",
            "<\"x",
            "m",
            "\"",
            "<\"m00",
        ]);
        let all = || {
            crate::Candidates::from_vec(
                (0..model.0.len() as crate::Token)
                    .map(|id| crate::TokenData {
                        id,
                        logit: 0.0,
                        p: 0.0,
                    })
                    .collect(),
            )
        };
        let kept = |compiled: &CompiledGrammar| {
            let mut ids: Vec<crate::Token> =
                grammar_filter(all(), compiled, &root, &model)
                    .as_slice()
                    .iter()
                    .map(|t| t.id)
                    .collect();
            ids.sort();
            ids
        };
        assert_eq!(kept(&capped), kept(&uncapped));
        assert_eq!(kept(&capped), vec![1, 2, 3, 4, 9]);
    }

    /// Nesting deeper than [`MAX_STATE_FRAMES`] allows is refused — the
    /// last stack too — rather than copied a frame deeper on every
    /// byte without end. Two stacks (`[` or `x` next), 33 frames a
    /// level each: refused at the level that would pass the cap.
    #[test]
    fn nesting_past_the_frame_cap_is_refused() {
        let chain: String = (1..32)
            .map(|i| format!("r{i} ::= r{} \"!\"\n", i + 1))
            .collect();
        let src = format!(
            "root ::= \"[\" r1 \"]\" | \"x\"\n{chain}r32 ::= root \"!\"\n"
        );
        let mut state = GrammarState::new(Arc::new(parse_ok(&src)));
        let depth = (0..=MAX_STATE_FRAMES)
            .find(|_| {
                let refused = state.advance_bytes(b"[").is_err();
                let frames: usize =
                    state.inner.stacks.iter().map(|s| s.len()).sum();
                assert!(frames <= MAX_STATE_FRAMES, "{frames}");
                refused
            })
            .expect("refused before the cap's depth");
        let cap = MAX_STATE_FRAMES / (2 * 33);
        assert!((cap - 2..=cap + 2).contains(&depth), "{depth} vs {cap}");
    }

    /// [`Grammar::parse`]'s size guard.
    #[test]
    fn oversized_grammar_is_rejected() {
        let mut src = String::from("root ::= \"a\"\n");
        while src.len() <= MAX_GRAMMAR_BYTES {
            src.push_str("# padding padding padding padding padding\n");
        }
        assert_eq!(
            Grammar::parse(&src),
            Err(GrammarError::TooLarge {
                what: "bytes",
                limit: MAX_GRAMMAR_BYTES
            })
        );
        let rules: String = (0..=MAX_GRAMMAR_RULES)
            .map(|i| format!("r{i} ::= \"a\"\n"))
            .collect();
        let src = format!("root ::= r0\n{rules}");
        assert!(src.len() <= MAX_GRAMMAR_BYTES);
        assert_eq!(
            Grammar::parse(&src),
            Err(GrammarError::TooLarge {
                what: "rules",
                limit: MAX_GRAMMAR_RULES
            })
        );
    }

    /// Complete-but-extensible vs exhausted: the distinction the
    /// Session's early halt turns on. A repeating root (the parallel
    /// call section's shape, with and without a separator) is complete
    /// after one repetition yet never exhausted; a fixed root is both at
    /// once; a required terminator after the repetition (Gemma 4's exit
    /// marker) is neither until the terminator lands.
    #[test]
    fn exhausted_is_complete_and_inextensible() {
        let at = |src: &str, input: &str| {
            let mut state = GrammarState::new(Arc::new(parse_ok(src)));
            state.advance_bytes(input.as_bytes()).expect("input legal");
            (state.inner.is_complete(), state.inner.is_exhausted())
        };
        assert_eq!(at(r#"root ::= "ab""#, "a"), (false, false));
        assert_eq!(at(r#"root ::= "ab""#, "ab"), (true, true));
        assert_eq!(at(r#"root ::= "ab"+"#, "ab"), (true, false));
        assert_eq!(at(r#"root ::= "ab"+"#, "abab"), (true, false));
        assert_eq!(at(r#"root ::= "ab" ( "\n" "ab" )*"#, "ab"), (true, false));
        assert_eq!(at(r#"root ::= "ab"+ "!""#, "ab"), (false, false));
        assert_eq!(at(r#"root ::= "ab"+ "!""#, "abab!"), (true, true));
    }

    /// A mid-parse `StackState` must serde round-trip exactly (derived
    /// `Eq`) and, restored against the same grammar, continue matching
    /// from the same position — including a buffered partial UTF-8
    /// codepoint in `pending`.
    #[cfg(feature = "serde")]
    #[test]
    fn stack_state_serde_round_trip() {
        let grammar = Arc::new(parse_ok(r#"root ::= "héllo" | "hénlo""#));
        let mut state = GrammarState::new(grammar.clone());
        // Stop mid-codepoint: feed "h" + the first byte of "é" (2-byte
        // UTF-8) so `pending` is non-empty and both alternation stacks
        // are still alive.
        let bytes = "héllo".as_bytes();
        state.advance_bytes(&bytes[..2]).unwrap();

        let json = serde_json::to_string(&state.inner).unwrap();
        let restored: StackState = serde_json::from_str(&json).unwrap();
        assert_eq!(state.inner, restored);

        // The restored matcher finishes the parse identically.
        let mut resumed = GrammarState {
            grammar,
            inner: restored,
        };
        resumed.advance_bytes(&bytes[2..]).unwrap();
        assert!(resumed.is_complete());
        state.advance_bytes(&bytes[2..]).unwrap();
        assert_eq!(state.inner, resumed.inner);
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

    /// `feed_byte` used to leave `pending` at full capacity (4) on the
    /// `Invalid` UTF-8 path; the next call would then push to a full
    /// `ArrayVec<[u8; 4]>` and panic. Surfaced by the in-tree fuzzer
    /// (`examples/grammar_fuzz.rs`) within seconds. Regression target:
    /// after an `InvalidUtf8` Err from a 4-byte overlong sequence, a
    /// follow-up `advance_bytes` must return cleanly (Err or Ok), not
    /// panic.
    #[test]
    fn feed_byte_recovers_after_invalid_utf8_at_capacity() {
        let grammar = Arc::new(parse_ok(r#"root ::= [\x80-\xFF]"#));
        let mut state = GrammarState::new(grammar);
        // 0xF0 0x80 0x80 0x80 is overlong U+0000 (the smallest 4-byte
        // codepoint should be U+10000); from_utf8 rejects it. The 4th
        // byte triggers Invalid.
        let _ = state.advance_bytes(&[0xF0, 0x80, 0x80, 0x80]);
        // What we care about: the next call doesn't panic. Result
        // can be either Ok or Err — pre-fix it was an unwrappable
        // capacity-overflow panic.
        let _ = state.advance_bytes(&[0x41]);
    }

    /// The fuzzer found that array-of-integers grammars crashed the
    /// matcher with an uncatchable stack overflow. The trigger isn't
    /// long input alone — it's repeated `first_byte_bitmap` calls on
    /// states with many active stacks (the bitmap interrogates every
    /// stack top against every byte 0..256, and certain inner states
    /// of the array+int grammar have stack counts that grow without
    /// bound across walker iterations).
    ///
    /// Regression target: 4096 bitmap-driven walker iterations on the
    /// array-of-integers grammar must complete cleanly.
    #[test]
    fn array_of_integers_walker_does_not_overflow() {
        let src = r#"root ::= args
args__item_1 ::= int
args ::= "[" ws ( args__item_1 ( ws "," ws args__item_1 )* )? ws "]"

value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
member ::= string ws ":" ws value
array ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string ::= "\"" char* "\""
char ::= unescaped | escape
unescaped ::= [^"\\\x00-\x1F]
escape ::= "\\" ( ["\\/bfnrt] | "u" hex hex hex hex )
hex ::= [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? ( "0" | [1-9] [0-9]* )
frac ::= "." [0-9]+
exp ::= [eE] [+\-]? [0-9]+
ws ::= [ \t\n\r]?
"#;
        let g = Arc::new(parse_ok(src));
        let mut state = GrammarState::new(g);
        // Walker pattern: at each step, ask for the bitmap, pick any
        // accepted byte, advance. The fuzzer's loop, distilled.
        for _ in 0..4096 {
            let bitmap = state.first_byte_bitmap();
            let mut chosen: Option<u8> = None;
            'outer: for (w, &word) in bitmap.iter().enumerate() {
                for b in 0..64u8 {
                    if (word >> b) & 1 == 1 {
                        let byte = (w as u8) * 64 + b;
                        if state.accepts_bytes(&[byte]) {
                            chosen = Some(byte);
                            break 'outer;
                        }
                    }
                }
            }
            let Some(byte) = chosen else { break };
            state.advance_bytes(&[byte]).unwrap();
        }
    }

    /// `parse_atom` recurses into `parse_alternates` for `(...)`
    /// groups; deeply-nested `((((...))))` sources used to blow the
    /// thread stack with no possible recovery. Surfaced by the in-tree
    /// fuzzer. Regression target: such sources now bail with
    /// `RecursionLimit`, not a `fatal runtime error: stack overflow`.
    #[test]
    fn parser_caps_recursion_at_deep_nested_groups() {
        let mut src = String::from("root ::= ");
        for _ in 0..2048 {
            src.push('(');
        }
        src.push('"');
        src.push('x');
        src.push('"');
        for _ in 0..2048 {
            src.push(')');
        }
        let err = Grammar::parse(&src).unwrap_err();
        assert!(
            matches!(err, GrammarError::RecursionLimit { .. }),
            "expected RecursionLimit, got {err:?}"
        );
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
    #[cfg(feature = "llama-cpp")]
    #[test]
    #[ignore = "requires model"]
    fn grammar_integration_tool_call() {
        use crate::{PredictOptions, SamplerConfig, SamplingMode};
        use std::{num::NonZeroUsize, path::PathBuf};

        let model_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let mut engine = crate::LlamaCppEngine::from_path(model_path).unwrap();

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
        opts.sample_options = SamplerConfig {
            modes: vec![grammar_mode, SamplingMode::locally_typical()],
            ..SamplerConfig::default()
        };

        let eos_piece = engine.model.token_to_piece(engine.model.eos());
        let predictor = engine.predict_pieces(tokens, opts, None);
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

    fn popcount(bm: &[u64; 4]) -> u32 {
        bm.iter().map(|w| w.count_ones()).sum()
    }

    /// Walk `input` from the root of `src` and return the live state.
    fn walked(src: &str, input: &str) -> GrammarState {
        let grammar = Arc::new(parse_ok(src));
        let mut state = GrammarState::new(grammar);
        state
            .advance_bytes(input.as_bytes())
            .expect("prefix should be legal for the grammar");
        state
    }

    /// String bodies in the built-in JSON grammar are permissive; every
    /// structural state is not. Popcount margins are asserted well away
    /// from `PERMISSIVE_MIN_POPCOUNT` on both sides so threshold drift
    /// (or a charset change in `JSON_GRAMMAR`) is caught here, not in a
    /// looping council seat.
    #[test]
    fn permissive_json_string_body_vs_structural() {
        let src = format!("root ::= value\n{}", crate::JSON_GRAMMAR);
        // Free regions: key and value string bodies, empty or mid-content.
        for p in ["\"", "{\"", "{\"a\":\"x", "[\"y"] {
            let st = walked(&src, p);
            let pc = popcount(&st.first_byte_bitmap());
            assert!(
                st.inner.is_permissive(&st.grammar),
                "expected permissive at {p:?} (popcount {pc})"
            );
            assert!(pc >= 100, "margin eroded at {p:?}: popcount {pc}");
        }
        // Structural: root, awaiting key/colon/value/comma, numbers,
        // mid-literal.
        for p in ["", "{", "{\"a\"", "{\"a\":", "{\"a\":1", "[tr"] {
            let st = walked(&src, p);
            let pc = popcount(&st.first_byte_bitmap());
            assert!(
                !st.inner.is_permissive(&st.grammar),
                "expected structural at {p:?} (popcount {pc})"
            );
            assert!(pc <= 32, "margin eroded at {p:?}: popcount {pc}");
        }
    }

    /// `until()` KMP states are permissive even mid-delimiter-prefix.
    /// This pins the documented v1 limitation: multi-token exit
    /// delimiters pass through permissive states, so mid-delimiter
    /// tokens remain penalizable (bounded by windowed decay) — see the
    /// follow-up note in `sample::region`.
    #[test]
    fn permissive_until_states() {
        let mut src = String::from("root ::= body\n");
        crate::emit_until_rules("body", "</arg>", &mut src);
        for p in ["", "<", "</"] {
            let st = walked(&src, p);
            assert!(
                st.inner.is_permissive(&st.grammar),
                "until state after {p:?} should be permissive"
            );
        }
    }

    /// Production-shape tool-call grammar (schema-derived): the value
    /// string is permissive; the literal key is structural.
    #[test]
    fn permissive_schema_tool_call_grammar() {
        let mut src = String::new();
        crate::schema_to_gbnf(
            &serde_json::json!({
                "type": "object",
                "properties": { "msg": { "type": "string" } },
                "required": ["msg"],
            }),
            "root",
            &mut src,
        )
        .unwrap();
        src.push_str(crate::JSON_GRAMMAR);

        let in_value = walked(&src, "{\"msg\":\"h");
        assert!(in_value.inner.is_permissive(&in_value.grammar));

        let mid_key = walked(&src, "{\"m");
        assert!(!mid_key.inner.is_permissive(&mid_key.grammar));
    }

    /// The DFA-cached predicate agrees with the uncached one at every
    /// state along a walk that visits strings, escapes, numbers,
    /// literals, and container boundaries — the parity that keeps
    /// `DRAMA_LLAMA_DFA_CACHE=0` from ever changing sampled streams.
    #[test]
    fn dfa_permissive_matches_uncached() {
        let src = format!("root ::= value\n{}", crate::JSON_GRAMMAR);
        let cg = CompiledGrammar::parse(&src).unwrap();
        let input = "{\"a\":\"x\\u00e9y\",\"b\":[1,true,\"z\"]}";

        let mut state = cg.root_state();
        let mut sid = cg.dfa.intern_base(&state);
        assert_eq!(
            cg.dfa.is_permissive(&cg.grammar, sid),
            state.is_permissive(&cg.grammar),
            "divergence at root"
        );
        for &b in input.as_bytes() {
            sid = cg.dfa.transition(&cg.grammar, sid, b);
            assert_ne!(sid, REJECT_STATE, "walk rejected byte {b:#x}");
            state.feed_byte(&cg.grammar, b).unwrap();
            assert_eq!(
                cg.dfa.is_permissive(&cg.grammar, sid),
                state.is_permissive(&cg.grammar),
                "divergence after byte {b:#x}"
            );
        }
        // Reject sentinel is never permissive.
        assert!(!cg.dfa.is_permissive(&cg.grammar, REJECT_STATE));
    }
}
