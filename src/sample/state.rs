//! Per-call sampler run-state — the mutable half of the config/state
//! split.
//!
//! [`SamplerState`] gathers every accumulator a generation mutates:
//! constraint matcher positions, the mirostat learning value, the
//! working RNG, and the repetition n-gram stats. It is a plain owned
//! value — no `Arc`, no interior mutability — so it clones, compares
//! (derived `PartialEq`), and serde round-trips exactly. That purity is
//! the contract that makes bit-exact snapshot/resume testable:
//! serialize → restore → continue must produce an identical stream.
//!
//! Construction is deliberately narrow: [`SamplerConfig::init_state`]
//! is the only constructor (config is the authority), with validated
//! deserialization as the single other door (the validation gates land
//! with the snapshot work — see the design memo).
//!
//! [`SamplerConfig::init_state`]: crate::SamplerConfig::init_state

use super::grammar::StackState;
use super::json::JsonState;
use crate::backend::Model;
use crate::ngram::NGramStats;
use crate::{SamplerConfig, SamplingMode, Token};

/// Matcher position for one entry of `SamplerConfig::modes`,
/// index-aligned with it. Stateless modes (truncation samplers,
/// mirostat — whose `mu` lives on [`SamplerState`] directly) hold
/// [`MatcherState::Stateless`].
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MatcherState {
    Stateless,
    Grammar {
        /// Identity of the compiled grammar this position indexes into
        /// ([`crate::CompiledGrammar::source_hash`]). Positions are
        /// only meaningful against the same source; reconciliation and
        /// the (Phase 3) deserialize door both gate on this.
        grammar: [u8; 32],
        stack: StackState,
    },
    Json(JsonState),
}

/// Run-state of a deferred grammar: the promotion flag plus the
/// matcher that starts advancing once the trigger fires.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeferredMatcher {
    /// Set by the predictor when a trigger byte sequence appears in
    /// the generated text. Replaces the old take-and-push promotion
    /// dance — the spec stays in config, only this flag and the
    /// matcher live here.
    pub(crate) active: bool,
    /// Identity of the deferred spec's compiled grammar — same
    /// contract as [`MatcherState::Grammar::grammar`].
    pub(crate) grammar: [u8; 32],
    pub(crate) matcher: StackState,
}

/// Everything a generation call mutates while sampling. See the module
/// docs for the purity contract.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerState {
    /// Index-aligned with `SamplerConfig::modes`.
    pub(crate) matchers: Vec<MatcherState>,
    /// Present iff the config has a `deferred_grammar`.
    pub(crate) deferred: Option<DeferredMatcher>,
    /// Mirostat learning value (`None` until the first mirostat step).
    pub(crate) mu: Option<f32>,
    /// Working RNG. Serialized as its full state — a restored snapshot
    /// continues the exact stream (resume ≠ restart; the *restart*
    /// seed lives in config/per-call options).
    pub(crate) rng: rand_pcg::Pcg64Mcg,
    /// Repetition-penalty accumulator.
    pub(crate) ngram_stats: NGramStats,
    /// Resolved repetition ignore set (`RepetitionOptions::ignored` ∪
    /// tokenized `ignored_categories`). A per-call memo of config ×
    /// model, computed once at `init_state` — NOT an accumulator, but
    /// homed here so the penalty pass doesn't re-tokenize category
    /// word lists per sampled token. Deterministic content, so it
    /// rides the bit-exact serialize/restore path harmlessly.
    pub(crate) resolved_ignored: std::collections::BTreeSet<crate::NGram>,
}

impl SamplerState {
    /// True while any byte-constraint (grammar/JSON mode, or an
    /// *activated* deferred grammar) is mid-parse. Gates the
    /// repetition penalty: constrained output is format-bound, and
    /// penalizing it steers generation away from exit delimiters.
    pub(crate) fn constrained_incomplete(&self) -> bool {
        self.matchers.iter().any(|m| match m {
            MatcherState::Grammar { stack, .. } => !stack.is_complete(),
            MatcherState::Json(s) => !s.is_complete(),
            MatcherState::Stateless => false,
        }) || self
            .deferred
            .as_ref()
            .is_some_and(|d| d.active && !d.matcher.is_complete())
    }

    /// True iff any byte-constraint is currently in play (an eager
    /// grammar/JSON mode, or an activated deferred grammar). Gates the
    /// lazy sample-then-check path.
    pub(crate) fn has_active_constraint(&self) -> bool {
        self.matchers
            .iter()
            .any(|m| !matches!(m, MatcherState::Stateless))
            || self.deferred.as_ref().is_some_and(|d| d.active)
    }

    /// True iff any constraint matcher — including an activated
    /// deferred grammar — has reached its accept state. The Session
    /// uses this to break out of the piece loop as soon as structured
    /// output is complete.
    pub fn grammar_complete(&self) -> bool {
        self.matchers.iter().any(|m| match m {
            MatcherState::Grammar { stack, .. } => stack.is_complete(),
            MatcherState::Json(s) => s.is_complete(),
            MatcherState::Stateless => false,
        }) || self
            .deferred
            .as_ref()
            .is_some_and(|d| d.active && d.matcher.is_complete())
    }

    /// True iff the config carried *eager* constraints (active from
    /// token 0 — not the deferred grammar, which may legitimately
    /// never trigger) and none of them reached accept. The Session's
    /// incomplete-at-end violation check.
    pub fn eager_constraint_incomplete(&self) -> bool {
        let mut any = false;
        for m in &self.matchers {
            match m {
                MatcherState::Grammar { stack, .. } => {
                    any = true;
                    if stack.is_complete() {
                        return false;
                    }
                }
                MatcherState::Json(s) => {
                    any = true;
                    if s.is_complete() {
                        return false;
                    }
                }
                MatcherState::Stateless => {}
            }
        }
        any
    }

    /// `Some(true)` iff a deferred matcher exists and has not yet been
    /// activated (i.e. the predictor should still scan for triggers).
    /// `None` when the config carries no deferred grammar.
    pub(crate) fn deferred_inactive(&self) -> Option<bool> {
        self.deferred.as_ref().map(|d| !d.active)
    }

    /// Activate the deferred matcher, feeding `tail` (the bytes at/after
    /// the trigger, per `DeferredGrammar::feed_trigger`) so the matcher
    /// lines up with the model's byte position. An `Err` means the tail
    /// violated the grammar — the caller ends generation.
    pub(crate) fn activate_deferred(
        &mut self,
        spec: &crate::DeferredGrammar,
        tail: &[u8],
    ) -> Result<(), crate::GrammarError> {
        let d = self
            .deferred
            .as_mut()
            .expect("activate_deferred: state has no deferred matcher");
        debug_assert!(!d.active, "deferred grammar activated twice");
        d.active = true;
        if tail.is_empty() {
            return Ok(());
        }
        d.matcher.advance_bytes(&spec.grammar.grammar, tail)
    }

    /// Record a prompt n-gram occurrence into the stats accumulator
    /// (predictor prompt-seeding).
    pub(crate) fn seed_prompt_ngram(
        &mut self,
        ngram: crate::NGram,
        trailing_pos: u64,
    ) -> &mut crate::ngram::NGramData {
        self.ngram_stats.add(ngram, trailing_pos)
    }

    /// Advance every active matcher (and the activated deferred
    /// grammar) by the chosen token's piece bytes. Advance errors are
    /// deliberately swallowed: they mean the constraint was violated on
    /// the prior step (EOS fallback chose an out-of-grammar token) and
    /// generation terminates on the next step via the EOS stop
    /// sequence.
    ///
    /// [`Candidates::sample_token`](crate::Candidates::sample_token)
    /// deliberately does not call this: whether the chosen token
    /// continues generation is the caller's call, and a token that
    /// terminates it must never mutate the state (tip invariant). Call
    /// this only for tokens that keep generating.
    pub fn advance<M: Model>(
        &mut self,
        config: &SamplerConfig,
        token: Token,
        model: &M,
    ) {
        debug_assert_eq!(self.matchers.len(), config.modes.len());
        let mut buf: Vec<u8> = Vec::new();
        let mut computed = false;
        let piece = |buf: &mut Vec<u8>, computed: &mut bool| {
            if !*computed {
                model.token_to_piece_ref(token, buf);
                *computed = true;
            }
        };
        for (mode, matcher) in config.modes.iter().zip(self.matchers.iter_mut())
        {
            match (mode, matcher) {
                (
                    SamplingMode::Grammar(compiled),
                    MatcherState::Grammar { stack, .. },
                ) => {
                    piece(&mut buf, &mut computed);
                    let _ = stack.advance_bytes(&compiled.grammar, &buf);
                }
                (SamplingMode::Json, MatcherState::Json(s)) => {
                    piece(&mut buf, &mut computed);
                    let _ = s.advance_bytes(&buf);
                }
                _ => {}
            }
        }
        if let (Some(d), Some(spec)) =
            (self.deferred.as_mut(), config.deferred_grammar.as_ref())
        {
            if d.active {
                piece(&mut buf, &mut computed);
                let _ = d.matcher.advance_bytes(&spec.grammar.grammar, &buf);
            }
        }
    }

    /// Build the working state for a call that resumes `cached` under
    /// `config` — the reconcile-by-grammar-identity load rule (design
    /// memo, Phase 2 round):
    ///
    /// - The matchers vec is built fresh from `config.modes` (never
    ///   cloned wholesale — the cached vec was aligned to a *different*
    ///   effective config's modes, so verbatim reuse risks
    ///   out-of-bounds rule indices). Each new grammar mode carries the
    ///   cached position forward **iff** a cached matcher walked the
    ///   identical compiled grammar ([`crate::CompiledGrammar::source_hash`];
    ///   same source ⇒ same deterministic compile ⇒ same indices);
    ///   otherwise it starts at root. This is what makes
    ///   assistant-prefill / partial-completion resume work while a
    ///   changed grammar resets only the matcher.
    /// - The JSON matcher carries unconditionally (fixed built-in
    ///   grammar, no identity to mismatch).
    /// - The deferred matcher carries (flag + position) iff the spec's
    ///   grammar identity matches; else fresh inactive root.
    /// - `mu`, the working rng, and `ngram_stats` carry
    ///   unconditionally — they are the stream being resumed.
    /// - `resolved_ignored` is recomputed from `config` × `model`: it
    ///   is a config memo riding the state, and repetition knobs are a
    ///   free per-call override.
    pub fn resumed_from<M: Model>(
        cached: &SamplerState,
        config: &SamplerConfig,
        model: &M,
    ) -> SamplerState {
        let cached_grammar = |hash: &[u8; 32]| {
            cached.matchers.iter().find_map(|m| match m {
                MatcherState::Grammar { grammar, stack } if grammar == hash => {
                    Some(stack.clone())
                }
                _ => None,
            })
        };
        let cached_json = || {
            cached.matchers.iter().find_map(|m| match m {
                MatcherState::Json(s) => Some(s.clone()),
                _ => None,
            })
        };
        SamplerState {
            matchers: config
                .modes
                .iter()
                .map(|mode| match mode {
                    SamplingMode::Grammar(compiled) => {
                        let grammar = compiled.source_hash();
                        let stack = cached_grammar(&grammar)
                            .unwrap_or_else(|| compiled.root_state());
                        MatcherState::Grammar { grammar, stack }
                    }
                    SamplingMode::Json => MatcherState::Json(
                        cached_json().unwrap_or_else(JsonState::new),
                    ),
                    _ => MatcherState::Stateless,
                })
                .collect(),
            deferred: config.deferred_grammar.as_ref().map(|spec| {
                let grammar = spec.grammar.source_hash();
                match cached.deferred.as_ref() {
                    Some(d) if d.grammar == grammar => d.clone(),
                    _ => DeferredMatcher {
                        active: false,
                        grammar,
                        matcher: spec.grammar.root_state(),
                    },
                }
            }),
            mu: cached.mu,
            rng: cached.rng.clone(),
            ngram_stats: cached.ngram_stats.clone(),
            resolved_ignored: config
                .repetition
                .as_ref()
                .map(|r| r.resolved_ignored(model))
                .unwrap_or_default(),
        }
    }

    /// Lazy-path legality of the chosen token: its piece bytes must
    /// extend (or, for a mid-parse EOG token, *finish*) every active
    /// constraint. Mirrors the masked filters' policy — empty pieces
    /// are illegal, EOG is judged by id while incomplete.
    pub(crate) fn accepts_chosen<M: Model>(
        &self,
        config: &SamplerConfig,
        chosen: Token,
        model: &M,
    ) -> bool {
        let mut buf: Vec<u8> = Vec::with_capacity(32);
        model.token_to_piece_ref(chosen, &mut buf);
        let chosen_is_eog = model.eog_tokens().contains(&chosen);

        let grammar_ok = |g: &crate::Grammar, s: &StackState| {
            !buf.is_empty()
                && if chosen_is_eog && !s.is_complete() {
                    s.completes_with(g, &buf)
                } else {
                    s.accepts_bytes(g, &buf)
                }
        };

        let modes_ok = config.modes.iter().zip(self.matchers.iter()).all(
            |(mode, matcher)| match (mode, matcher) {
                (
                    SamplingMode::Grammar(compiled),
                    MatcherState::Grammar { stack, .. },
                ) => grammar_ok(&compiled.grammar, stack),
                (SamplingMode::Json, MatcherState::Json(s)) => {
                    !buf.is_empty()
                        && if chosen_is_eog && !s.is_complete() {
                            s.completes_with(&buf)
                        } else {
                            s.accepts_bytes(&buf)
                        }
                }
                _ => true,
            },
        );
        modes_ok
            && match (&self.deferred, &config.deferred_grammar) {
                (Some(d), Some(spec)) if d.active => {
                    grammar_ok(&spec.grammar.grammar, &d.matcher)
                }
                _ => true,
            }
    }
}
