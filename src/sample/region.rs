//! Free-region guard for the constrained repetition penalty.
//!
//! When a grammar-constrained generation sits inside a *permissive* free
//! region (a JSON string body, an `until()` raw value — states where the
//! grammar accepts nearly any byte and the model owns the content), the
//! repetition penalty runs to break degenerate loops. The one guarantee
//! that keeps the grammar completable is enforced here: **no token whose
//! bytes advance the grammar out of the region ever has its logit
//! reduced.** That covers merged exit tokens (`",`, `"}`, `y"`) with zero
//! tokenizer knowledge — the walk sees their bytes leave the region, which
//! the bare-char `IgnoreCategory::Json` list structurally cannot (it
//! tokenizes each delimiter in isolation and never produces merged ids).
//!
//! Walk semantics, shared byte-for-byte by both engines (GBNF and the JSON
//! pushdown) so the engine choice never changes streams:
//! - a byte-prefix leaves the free region or completes a constraint ⇒
//!   **protected** (early return — later bytes are not consulted, so a
//!   token that exits and then violates is still protected; harmless,
//!   since the grammar filter masks it anyway);
//! - a byte rejects while still inside the region ⇒ not protected (the
//!   token is filter-masked; penalizing it is harmless, and on the lazy
//!   path it reduces wasteful fallback re-samples);
//! - the walk ends inside the region but *off its content loop* ⇒
//!   **protected** (GBNF engine; see below);
//! - an empty piece ⇒ not protected.
//!
//! Progress toward a multi-token exit. An `until("\n</parameter>")`
//! delimiter spans several tokens (`\n`, `</`, `parameter`, `>`) whose
//! intermediate KMP states are themselves permissive, so "leaves the
//! region" protects only the last of them. The rest were penalizable, and
//! since every tool call ends with the same delimiter, its n-grams repeat
//! once per call: by the third call of a turn Qwen3.8 could no longer
//! close a parameter and wrote `】`, `</target>` or `</invoke>` into the
//! value instead, then kept going (2026-09-22 Agora trial, #113). So each
//! grammar region also finds its *home*: the state a plain content byte
//! returns to (a fixed point — `q` from `q`). A token whose walk ends
//! anywhere else has made progress toward the exit (or is mid-escape,
//! or mid-codepoint) and is protected. Content tokens end at home and
//! stay penalizable. The cost is that a content token ending in a
//! delimiter prefix (`foo\n`, `a<`) is spared too — bounded, and the
//! penalty is for prose loops, which such tokens do not sustain.

use super::grammar::{
    dfa_cache_enabled, DfaCache, Grammar, StackState, StateId, REJECT_STATE,
};
use super::json::JsonState;
use super::state::{DeferredMatcher, MatcherState};
use crate::backend::Model;
use crate::{SamplingMode, Token};

/// Object-safe hook consumed by the penalty pass (`repetition.rs` stays
/// model-free; a handful of virtual calls per step is noise).
pub(crate) trait RegionGuard {
    /// True iff penalizing `token` could impede exiting the current free
    /// region of any active constraint. See the module docs for the walk
    /// semantics.
    fn is_protected(&self, token: Token) -> bool;
}

/// Two guards, either of which protects: the region guard and the
/// known-id guard (`sample::ids`) both apply inside a free region.
pub(crate) struct Either<'a>(
    pub(crate) &'a dyn RegionGuard,
    pub(crate) &'a dyn RegionGuard,
);

impl RegionGuard for Either<'_> {
    fn is_protected(&self, token: Token) -> bool {
        self.0.is_protected(token) || self.1.is_protected(token)
    }
}

/// One active, incomplete, currently-permissive constraint.
enum GuardEntry<'a> {
    Grammar {
        grammar: &'a Grammar,
        dfa: &'a DfaCache,
        /// Interned current state — `None` when the DFA cache is disabled
        /// (`DRAMA_LLAMA_DFA_CACHE=0`); the walk then clone-steps the
        /// `StackState` so the flag never changes sampled streams.
        base: Option<StateId>,
        matcher: &'a StackState,
        /// The region's content loop (module docs), in whichever
        /// representation the walk uses. `None` when no probe byte
        /// finds one: only exits are protected, as before.
        home: Option<Home>,
    },
    Json {
        state: &'a JsonState,
    },
}

/// A grammar region's content loop.
enum Home {
    Interned(StateId),
    Stack(StackState),
}

/// Bytes tried, in order, as "plain content" when looking for a region's
/// home. Rare in delimiters and legal in every free region we emit; the
/// first whose step from the base is a permissive fixed point wins.
const HOME_PROBES: &[u8] = b"qZ~%xQ";

impl Home {
    /// Find the home of the region `base` sits in (DFA path).
    fn find_interned(
        grammar: &Grammar,
        dfa: &DfaCache,
        base: StateId,
    ) -> Option<Self> {
        HOME_PROBES.iter().find_map(|&c| {
            let s = dfa.transition(grammar, base, c);
            (s != REJECT_STATE
                && !dfa.is_complete(s)
                && dfa.is_permissive(grammar, s)
                && dfa.transition(grammar, s, c) == s)
                .then_some(Home::Interned(s))
        })
    }

    /// Find the home of the region `matcher` sits in (clone-walk path).
    fn find_stack(grammar: &Grammar, matcher: &StackState) -> Option<Self> {
        HOME_PROBES.iter().find_map(|&c| {
            let mut s = matcher.clone();
            s.feed_byte(grammar, c).ok()?;
            if s.is_complete() || !s.is_permissive(grammar) {
                return None;
            }
            let mut again = s.clone();
            again.feed_byte(grammar, c).ok()?;
            (again == s).then_some(Home::Stack(s))
        })
    }
}

impl GuardEntry<'_> {
    /// The region-exit walk for this constraint. See module docs.
    fn protects(&self, piece: &[u8]) -> bool {
        match self {
            GuardEntry::Grammar {
                grammar,
                dfa,
                base: Some(base),
                home,
                ..
            } => {
                let mut sid = *base;
                for &b in piece {
                    sid = dfa.transition(grammar, sid, b);
                    if sid == REJECT_STATE {
                        return false;
                    }
                    if dfa.is_complete(sid) || !dfa.is_permissive(grammar, sid)
                    {
                        return true;
                    }
                }
                matches!(home, Some(Home::Interned(h)) if sid != *h)
            }
            GuardEntry::Grammar {
                grammar,
                base: None,
                matcher,
                home,
                ..
            } => {
                let mut scratch = (*matcher).clone();
                for &b in piece {
                    if scratch.feed_byte(grammar, b).is_err() {
                        return false;
                    }
                    if scratch.is_complete() || !scratch.is_permissive(grammar)
                    {
                        return true;
                    }
                }
                matches!(home, Some(Home::Stack(h)) if scratch != *h)
            }
            GuardEntry::Json { state } => state.exit_protects(piece),
        }
    }
}

/// The live guard for one sampling step: every active incomplete
/// constraint, pre-checked permissive.
pub(crate) struct ConstraintGuard<'a, M: Model> {
    entries: Vec<GuardEntry<'a>>,
    model: &'a M,
}

impl<'a, M: Model> ConstraintGuard<'a, M> {
    /// `Some(guard)` iff at least one byte-constraint is active and
    /// incomplete AND **every** such constraint is currently in a
    /// permissive state — the gate's condition (b). `None` means the
    /// caller must skip the penalty entirely (condition (c), a structural
    /// state — exactly the pre-feature suspension).
    ///
    /// Takes the split fields rather than `&SamplerState` so
    /// `sample_token` can borrow the n-gram accumulators mutably
    /// alongside. Must be called from the single-threaded point of the
    /// sampling step (`intern_base` contract).
    pub(crate) fn build(
        modes: &'a [SamplingMode],
        matchers: &'a [MatcherState],
        deferred: Option<&'a DeferredMatcher>,
        deferred_spec: Option<&'a crate::DeferredGrammar>,
        model: &'a M,
    ) -> Option<Self> {
        let cache_on = dfa_cache_enabled();
        let mut entries = Vec::new();

        let push_grammar = |entries: &mut Vec<GuardEntry<'a>>,
                            compiled: &'a crate::CompiledGrammar,
                            stack: &'a StackState|
         -> bool {
            if stack.is_complete() {
                return true; // not incomplete — no entry, no veto
            }
            let (permissive, base) = if cache_on {
                let sid = compiled.dfa.intern_base(stack);
                (
                    compiled.dfa.is_permissive(&compiled.grammar, sid),
                    Some(sid),
                )
            } else {
                (stack.is_permissive(&compiled.grammar), None)
            };
            if !permissive {
                return false; // structural state vetoes the whole pass
            }
            let home = match base {
                Some(sid) => {
                    Home::find_interned(&compiled.grammar, &compiled.dfa, sid)
                }
                None => Home::find_stack(&compiled.grammar, stack),
            };
            entries.push(GuardEntry::Grammar {
                grammar: &compiled.grammar,
                dfa: &compiled.dfa,
                base,
                matcher: stack,
                home,
            });
            true
        };

        for (mode, matcher) in modes.iter().zip(matchers.iter()) {
            match (mode, matcher) {
                (
                    SamplingMode::Grammar(compiled),
                    MatcherState::Grammar { stack, .. },
                ) => {
                    if !push_grammar(&mut entries, compiled, stack) {
                        return None;
                    }
                }
                (SamplingMode::Json, MatcherState::Json(s))
                    if !s.is_complete() =>
                {
                    if !s.in_free_region() {
                        return None;
                    }
                    entries.push(GuardEntry::Json { state: s });
                }
                _ => {}
            }
        }
        if let (Some(d), Some(spec)) = (deferred, deferred_spec) {
            if d.active
                && !push_grammar(&mut entries, &spec.grammar, &d.matcher)
            {
                return None;
            }
        }

        if entries.is_empty() {
            // No incomplete constraint — the caller's gate shouldn't have
            // asked, but the honest answer is "no guarded pass".
            return None;
        }
        Some(Self { entries, model })
    }
}

impl<M: Model> RegionGuard for ConstraintGuard<'_, M> {
    fn is_protected(&self, token: Token) -> bool {
        let mut piece: Vec<u8> = Vec::with_capacity(16);
        self.model.token_to_piece_ref(token, &mut piece);
        if piece.is_empty() {
            return false;
        }
        self.entries.iter().any(|e| e.protects(&piece))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompiledGrammar;

    /// Minimal model whose pieces cover content, bare-close, merged-close,
    /// escape-intro, and empty tokens. Ids index into `PIECES`.
    struct StrMock;

    const PIECES: &[&str] =
        &["", "y", " the", "\"", "\",", "\"}", "y\"", "\\", "\u{1}"];
    const EMPTY: Token = 0;
    const CONTENT: Token = 1;
    const CONTENT_WORD: Token = 2;
    const CLOSE: Token = 3;
    const CLOSE_COMMA: Token = 4;
    const CLOSE_BRACE: Token = 5;
    const CONTENT_CLOSE: Token = 6;
    const BACKSLASH: Token = 7;
    const CTRL: Token = 8;

    impl Model for StrMock {
        type Error = std::convert::Infallible;

        fn n_vocab(&self) -> i32 {
            PIECES.len() as i32
        }
        fn bos(&self) -> Token {
            EMPTY
        }
        fn eos(&self) -> Token {
            EMPTY
        }
        fn eot(&self) -> Token {
            EMPTY
        }
        fn special_tokens(&self) -> Vec<Token> {
            vec![EMPTY]
        }
        fn eog_tokens(&self) -> Vec<Token> {
            vec![EMPTY]
        }
        fn max_token_len(&self) -> usize {
            2
        }
        fn tokenize(&self, _input: &str, _special: bool) -> Vec<Token> {
            unimplemented!("not needed by the guard")
        }
        fn token_to_piece(&self, token: Token) -> String {
            PIECES[token as usize].to_string()
        }
        fn token_to_piece_ref(&self, token: Token, buf: &mut Vec<u8>) {
            buf.clear();
            buf.extend_from_slice(PIECES[token as usize].as_bytes());
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

    /// A quoted string whose close is the merged token `",` — the island
    /// shape with a multi-byte structural exit.
    const STR_GRAMMAR: &str = r#"root ::= "\"" [^"]* "\",""#;

    /// (modes, matchers) for `STR_GRAMMAR` with the matcher walked past
    /// `prefix` bytes.
    fn grammar_fixture(
        prefix: &[u8],
    ) -> (Vec<SamplingMode>, Vec<MatcherState>) {
        let compiled =
            CompiledGrammar::parse(STR_GRAMMAR).expect("grammar parses");
        let mut stack = compiled.root_state();
        stack
            .advance_bytes(&compiled.grammar, prefix)
            .expect("prefix is legal");
        let matcher = MatcherState::Grammar {
            grammar: compiled.source_hash(),
            stack,
        };
        (vec![SamplingMode::Grammar(compiled)], vec![matcher])
    }

    fn json_matcher(prefix: &str) -> MatcherState {
        let mut s = JsonState::new();
        s.advance_bytes(prefix.as_bytes()).expect("prefix is legal");
        MatcherState::Json(s)
    }

    /// Mid-string-body: content tokens penalizable, every exit-crossing
    /// token protected — including merged pieces the bare-char ignore
    /// list can never cover.
    #[test]
    fn grammar_guard_protects_exits_only() {
        let (modes, matchers) = grammar_fixture(b"\"x");
        let guard =
            ConstraintGuard::build(&modes, &matchers, None, None, &StrMock)
                .expect("mid-string state is permissive");

        assert!(!guard.is_protected(CONTENT));
        assert!(!guard.is_protected(CONTENT_WORD));
        // Empty pieces are never protected (they advance nothing).
        assert!(!guard.is_protected(EMPTY));
        // Rejected while still in-region (control byte): masked anyway.
        assert!(!guard.is_protected(CTRL));

        // The close and every merged/embedded form: protected.
        assert!(guard.is_protected(CLOSE));
        assert!(guard.is_protected(CLOSE_COMMA));
        assert!(guard.is_protected(CONTENT_CLOSE));
        // Early-return beats later rejection: `"}`'s close-byte leaves
        // the region before the `}` byte would reject.
        assert!(guard.is_protected(CLOSE_BRACE));
    }

    /// The uncached (clone-walk) path must agree with the DFA-cached
    /// path token-for-token. `dfa_cache_enabled` is a process-global
    /// OnceLock, so rather than fork a process we drive the uncached
    /// walk directly through a `base: None` entry.
    #[test]
    fn uncached_walk_matches_cached() {
        let compiled =
            CompiledGrammar::parse(STR_GRAMMAR).expect("grammar parses");
        let mut stack = compiled.root_state();
        stack.advance_bytes(&compiled.grammar, b"\"x").unwrap();
        let sid = compiled.dfa.intern_base(&stack);

        let cached = GuardEntry::Grammar {
            grammar: &compiled.grammar,
            dfa: &compiled.dfa,
            base: Some(sid),
            matcher: &stack,
            home: Home::find_interned(&compiled.grammar, &compiled.dfa, sid),
        };
        let uncached = GuardEntry::Grammar {
            grammar: &compiled.grammar,
            dfa: &compiled.dfa,
            base: None,
            matcher: &stack,
            home: Home::find_stack(&compiled.grammar, &stack),
        };
        for token in 0..PIECES.len() as Token {
            let piece = PIECES[token as usize].as_bytes();
            if piece.is_empty() {
                continue;
            }
            assert_eq!(
                cached.protects(piece),
                uncached.protects(piece),
                "cached/uncached divergence on token {token} ({piece:?})"
            );
        }
    }

    /// Both walks over a region, from `prefix`: (cached, uncached).
    fn entries_after<'a>(
        compiled: &'a CompiledGrammar,
        stack: &'a StackState,
    ) -> [GuardEntry<'a>; 2] {
        let sid = compiled.dfa.intern_base(stack);
        [
            GuardEntry::Grammar {
                grammar: &compiled.grammar,
                dfa: &compiled.dfa,
                base: Some(sid),
                matcher: stack,
                home: Home::find_interned(
                    &compiled.grammar,
                    &compiled.dfa,
                    sid,
                ),
            },
            GuardEntry::Grammar {
                grammar: &compiled.grammar,
                dfa: &compiled.dfa,
                base: None,
                matcher: stack,
                home: Home::find_stack(&compiled.grammar, stack),
            },
        ]
    }

    /// The Qwen3.8 XML parameter value: `until("\n</parameter>")`. Every
    /// token of the closing delimiter is protected at the point it is
    /// due, not only the one that finally leaves the region — else the
    /// delimiter's n-grams, repeated once per call, suppress the close
    /// (#113). Content stays penalizable, including from mid-delimiter.
    #[test]
    fn multi_token_exit_protected_at_every_step() {
        let mut src = String::from("root ::= \"<v>\" body \"<end>\"\n");
        crate::emit_until_rules("body", "\n</parameter>", &mut src);
        let compiled = CompiledGrammar::parse(&src).expect("grammar parses");
        let at = |prefix: &str| {
            let mut stack = compiled.root_state();
            stack
                .advance_bytes(&compiled.grammar, prefix.as_bytes())
                .expect("prefix is legal");
            stack
        };
        let steps: [(&str, &[&str], &[&str]); 4] = [
            ("<v>abc", &["\n", "\n</"], &["abc", " the"]),
            ("<v>abc\n", &["</", "</param"], &["abc", "q"]),
            ("<v>abc\n</", &["parameter", "param"], &["abc"]),
            ("<v>abc\n</parameter", &[">"], &["abc"]),
        ];
        for (prefix, protected, content) in steps {
            let stack = at(prefix);
            for entry in entries_after(&compiled, &stack) {
                for p in protected {
                    assert!(
                        entry.protects(p.as_bytes()),
                        "{p:?} after {prefix:?} is progress toward the exit"
                    );
                }
                for p in content {
                    assert!(
                        !entry.protects(p.as_bytes()),
                        "{p:?} after {prefix:?} is content"
                    );
                }
            }
        }
    }

    /// A single-byte exit region (a JSON-ish string) is unchanged:
    /// content tokens stay penalizable, the close is protected.
    #[test]
    fn single_byte_exit_region_keeps_content_penalizable() {
        let compiled =
            CompiledGrammar::parse(STR_GRAMMAR).expect("grammar parses");
        let mut stack = compiled.root_state();
        stack.advance_bytes(&compiled.grammar, b"\"x").unwrap();
        for entry in entries_after(&compiled, &stack) {
            assert!(!entry.protects(b"y"));
            assert!(!entry.protects(b" the"));
            assert!(entry.protects(b"\","));
        }
    }

    /// Structural states veto the whole pass: at the grammar root (only
    /// `"` is legal) `build` returns `None` — exactly the pre-feature
    /// suspension.
    #[test]
    fn structural_state_vetoes_build() {
        let (modes, matchers) = grammar_fixture(b"");
        assert!(ConstraintGuard::build(
            &modes, &matchers, None, None, &StrMock
        )
        .is_none());
    }

    /// A completed constraint contributes no entry and no veto; with no
    /// incomplete constraint left, `build` honestly returns `None`.
    #[test]
    fn complete_constraint_yields_none() {
        let (modes, matchers) = grammar_fixture(b"\"x\",");
        assert!(ConstraintGuard::build(
            &modes, &matchers, None, None, &StrMock
        )
        .is_none());
    }

    /// JSON-engine entries: same exemptions, and the escape-intro `\`
    /// is protected (AfterEscape is structural — accepted quirk).
    #[test]
    fn json_guard_protects_exits_only() {
        let modes = vec![SamplingMode::Json];
        let matchers = vec![json_matcher("{\"a\":\"x")];
        let guard =
            ConstraintGuard::build(&modes, &matchers, None, None, &StrMock)
                .expect("mid-string state is a free region");

        assert!(!guard.is_protected(CONTENT));
        assert!(!guard.is_protected(CONTENT_WORD));
        assert!(!guard.is_protected(CTRL));
        assert!(guard.is_protected(CLOSE));
        assert!(guard.is_protected(CLOSE_COMMA));
        assert!(guard.is_protected(CLOSE_BRACE));
        assert!(guard.is_protected(CONTENT_CLOSE));
        assert!(guard.is_protected(BACKSLASH));
    }

    /// Multi-constraint AND-gate: one structural constraint vetoes even
    /// when the other is permissive; protection is the OR of entries.
    #[test]
    fn multi_constraint_all_must_be_permissive() {
        let (mut modes, mut matchers) = grammar_fixture(b"\"x");
        // Add a JSON constraint sitting at a structural point.
        modes.push(SamplingMode::Json);
        matchers.push(json_matcher("{"));
        assert!(ConstraintGuard::build(
            &modes, &matchers, None, None, &StrMock
        )
        .is_none());

        // Both permissive → Some; a token exiting EITHER region is
        // protected.
        matchers[1] = json_matcher("{\"a\":\"x");
        let guard =
            ConstraintGuard::build(&modes, &matchers, None, None, &StrMock)
                .expect("both regions permissive");
        assert!(guard.is_protected(CLOSE));
        assert!(!guard.is_protected(CONTENT));
    }
}
