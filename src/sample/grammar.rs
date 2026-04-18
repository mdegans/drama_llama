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

use llama_cpp_sys_3::{llama_token, llama_token_data};
use tinyvec::ArrayVec;

use std::sync::Arc;

use crate::{model::token_to_piece_ref, Candidates, Model};

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
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Position {
    rule_idx: u32,
    alt_idx: u32,
    atom_idx: u32,
}

/// Active matching state for a [`Grammar`].
///
/// The matcher is an NFA-like simulation over the compiled rules. Each
/// element of `stacks` is a call stack: the innermost frame is the rule
/// currently being walked; popping returns to the caller. Multiple stacks
/// coexist because GBNF rules branch on alternation.
///
/// Clone cost is proportional to `stacks.len() * avg_stack_depth`, which is
/// small for practical grammars. `accepts_bytes` relies on cloning for
/// speculative simulation.
#[derive(Clone, Debug, PartialEq)]
pub struct GrammarState {
    grammar: Arc<Grammar>,
    stacks: Vec<Vec<Position>>,
    pending: ArrayVec<[u8; 4]>,
}

impl GrammarState {
    /// Construct a fresh matcher rooted at the grammar's `root` rule.
    pub fn new(grammar: Arc<Grammar>) -> Self {
        let mut state = Self {
            grammar,
            stacks: Vec::new(),
            pending: ArrayVec::new(),
        };
        state.reset();
        state
    }

    /// Reset to the fresh starting state.
    pub fn reset(&mut self) {
        self.pending.clear();
        let root = self.grammar.root as u32;
        let root_rule = &self.grammar.rules[self.grammar.root];
        self.stacks = (0..root_rule.alts.len())
            .map(|alt_idx| {
                vec![Position {
                    rule_idx: root,
                    alt_idx: alt_idx as u32,
                    atom_idx: 0,
                }]
            })
            .collect();
        self.expand();
    }

    /// True iff the matcher has reached an accepting state AND no partial
    /// UTF-8 codepoint is buffered.
    pub fn is_complete(&self) -> bool {
        self.pending.is_empty() && self.stacks.iter().any(|s| s.is_empty())
    }

    /// Current number of active stacks. Useful for UI status / debugging.
    pub fn stack_depth(&self) -> usize {
        self.stacks.len()
    }

    /// Borrow the underlying grammar.
    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    /// True iff feeding `bytes` would succeed from the current state. Does
    /// not mutate `self`.
    pub fn accepts_bytes(&self, bytes: &[u8]) -> bool {
        let mut clone = self.clone();
        clone.advance_bytes(bytes).is_ok()
    }

    /// Commit `bytes` to the matcher. Call after sampling selects a token.
    ///
    /// Partial UTF-8 at the end of `bytes` is buffered for the next call —
    /// llama tokens can split multi-byte codepoints. If the pending
    /// prefix cannot possibly complete into any codepoint the current
    /// grammar state would accept, this returns an error so the filter
    /// rejects the token. Without this check, byte-fallback tokens
    /// (e.g. `<0xC3>`) can wedge an ASCII-only grammar into a dead
    /// state where every subsequent candidate fails UTF-8 validation.
    pub fn advance_bytes(&mut self, bytes: &[u8]) -> Result<(), GrammarError> {
        for &b in bytes {
            self.feed_byte(b)?;
        }
        if !self.pending.is_empty() && !self.pending_can_still_match() {
            return Err(GrammarError::InvalidUtf8);
        }
        Ok(())
    }

    /// True if the pending UTF-8 prefix could still complete into a
    /// codepoint that at least one active stack would accept. Returns
    /// true conservatively (over-approximates) — a `true` result does
    /// not guarantee a match, only that one is plausible.
    fn pending_can_still_match(&self) -> bool {
        let (lo, hi) = match pending_codepoint_range(self.pending.as_slice()) {
            Some(range) => range,
            // Pending already invalid; advance_bytes would have erred.
            None => return false,
        };
        for stack in &self.stacks {
            let Some(pos) = stack.last() else {
                continue;
            };
            let atoms = &self.grammar.rules[pos.rule_idx as usize].alts
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

    fn feed_byte(&mut self, b: u8) -> Result<(), GrammarError> {
        self.pending.push(b);
        match decode_utf8(self.pending.as_slice()) {
            Utf8Decode::Complete(cp) => {
                self.pending.clear();
                self.consume(cp)
            }
            Utf8Decode::Incomplete => Ok(()),
            Utf8Decode::Invalid => Err(GrammarError::InvalidUtf8),
        }
    }

    /// Consume a single codepoint: keep only stacks whose top CharSet
    /// accepts it, then re-expand.
    fn consume(&mut self, cp: u32) -> Result<(), GrammarError> {
        let mut next: Vec<Vec<Position>> = Vec::new();
        for stack in self.stacks.drain(..) {
            let Some(pos) = stack.last() else {
                // Already accepting; a fresh codepoint here would require
                // re-starting at root. Grammar-completion is strict: reject.
                continue;
            };
            let atoms = &self.grammar.rules[pos.rule_idx as usize].alts
                [pos.alt_idx as usize];
            let Some(atom) = atoms.get(pos.atom_idx as usize) else {
                continue;
            };
            let Atom::CharSet(cs) = atom else {
                // Expansion should leave only CharSet tops. If a
                // RuleRef slips through (e.g. budget exhaustion on a
                // deep grammar), skip that stack — aborting the whole
                // consume would poison the entire candidate set.
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
        self.expand();
        if self.stacks.is_empty() {
            return Err(GrammarError::NoMatch(cp));
        }
        Ok(())
    }

    /// Walk all epsilon transitions until every stack is either empty
    /// (accepting) or has a CharSet at the top.
    fn expand(&mut self) {
        let mut queue: Vec<Vec<Position>> = std::mem::take(&mut self.stacks);
        let mut result: Vec<Vec<Position>> = Vec::new();
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
            let alts = &self.grammar.rules[pos.rule_idx as usize].alts;
            let alt = &alts[pos.alt_idx as usize];
            if pos.atom_idx as usize == alt.len() {
                // Alt complete: pop the frame. If the caller exists,
                // advance its atom pointer past the just-completed RuleRef.
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
                    let sub_alts = &self.grammar.rules[*r].alts;
                    for (a_idx, _) in sub_alts.iter().enumerate() {
                        let mut branched = stack.clone();
                        branched.push(Position {
                            rule_idx: *r as u32,
                            alt_idx: a_idx as u32,
                            atom_idx: 0,
                        });
                        queue.push(branched);
                    }
                }
            }
        }
        // Dedupe: identical stacks are redundant work. The NFA simulation
        // can otherwise explode on deeply nested alternations.
        result.sort();
        result.dedup();
        self.stacks = result;
    }
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
    model: &Model,
) -> Candidates {
    let mut buf: Vec<u8> = Vec::with_capacity(32);
    let mut kept: Vec<llama_token_data> =
        Vec::with_capacity(candidates.len().get());
    for cand in candidates.as_slice() {
        buf.clear();
        token_to_piece_ref(cand.id, model, &mut buf);
        if state.accepts_bytes(&buf) {
            kept.push(*cand);
        }
    }

    if kept.is_empty() {
        if state.is_complete() {
            state.reset();
        }
        let eos = llama_token_data {
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
    model: &Model,
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
}
