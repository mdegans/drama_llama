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
    /// Prose-corpus step counter — the `current_step` basis for
    /// windowed eviction/decay in the penalty pass. Advances +1 per
    /// prose token seeded and +1 per *executed* penalty pass
    /// (constrained spans skip the pass, so structured output consumes
    /// no window). Cross-call stable, unlike `tokens.len()`, which is
    /// suffix-relative on a cache resume.
    pub(crate) step: u64,
    /// Resolved repetition ignore set (`RepetitionOptions::ignored` ∪
    /// tokenized `ignored_categories`). A per-call memo of config ×
    /// model, computed once at `init_state` — NOT an accumulator, but
    /// homed here so the penalty pass doesn't re-tokenize category
    /// word lists per sampled token. Deterministic content, so it
    /// rides the bit-exact serialize/restore path harmlessly.
    pub(crate) resolved_ignored: std::collections::BTreeSet<crate::NGram>,
    /// **Call-local** repetition accumulator for constrained free
    /// regions (grammar string bodies, `until()` values — see
    /// `sample::region`). Serialized and compared like every other
    /// field: anything that influences logits influences RNG
    /// consumption downstream, so it must ride a mid-call snapshot for
    /// restore-and-continue to replay the identical stream (same
    /// rationale as serializing `rng`). It is NEVER carried across
    /// call boundaries — `init_state` and `resumed_from` both start it
    /// empty, which is what keeps the incremental-vs-cold fold
    /// equivalence intact (a cold fold cannot re-derive another call's
    /// sampled stream). Since #106 it is instead **reborn seeded**:
    /// after the doors zero it, `Session`'s `fold_and_snapshot` clones
    /// the folded prompt corpus into it (post-last-snapshot, step
    /// rebased — see [`RepetitionOptions::seed_constrained_regions`]),
    /// re-derived identically on the cold and resume paths. So
    /// cross-call repetition pressure flows through *prompt content*,
    /// never through carried accumulator state. Cache-resident
    /// snapshots (the tip, hash-inherited breakpoints) may therefore
    /// legitimately hold populated constrained fields — the resume
    /// door zeroes them before the re-seed; that is the mechanism, not
    /// a leak. `serde(default)` so pre-feature blobs deserialize to
    /// the empty accumulator.
    ///
    /// [`RepetitionOptions::seed_constrained_regions`]:
    ///     crate::RepetitionOptions::seed_constrained_regions
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) constrained_ngram_stats: NGramStats,
    /// Step counter for the constrained-region window/decay math —
    /// advances once per executed *guarded* pass, exactly as `step`
    /// does for the prose corpus. Same call-boundary reset and same
    /// serialization rationale as [`Self::constrained_ngram_stats`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) constrained_step: u64,
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

    /// True iff generation ended while a byte-constraint was still
    /// mid-structure — the Session's incomplete-at-end violation
    /// check. Two shapes count:
    ///
    /// * an **eager** constraint (active from token 0) where none of
    ///   the matchers reached accept;
    /// * a **deferred** grammar that *activated* — its trigger fired —
    ///   and is incomplete. A deferred grammar that never triggered is
    ///   NOT a violation: never calling a tool is legal on the Auto
    ///   path. This clause is the inverse of the deferred arm in
    ///   [`Self::grammar_complete`], and its absence was issue #38's
    ///   defect 1 — a truncated Auto call parsed as (and seated as)
    ///   plain text, poisoning the transcript for the next ingest.
    ///
    /// A terminal token whose bytes *complete* the constraint is not
    /// visible here: it never advances the matchers (tip invariant), so
    /// the exemption lives one level up in
    /// [`crate::TokenPredictor::constraint_incomplete_at_end`].
    pub fn constraint_incomplete_at_end(&self) -> bool {
        // Note the shape: `any_eager && !eager_complete`, not an early
        // `return false` on the first completion. The deferred clause
        // below has to stay reachable no matter what the eager
        // matchers did.
        let mut any_eager = false;
        let mut eager_complete = false;
        for m in &self.matchers {
            match m {
                MatcherState::Grammar { stack, .. } => {
                    any_eager = true;
                    eager_complete |= stack.is_complete();
                }
                MatcherState::Json(s) => {
                    any_eager = true;
                    eager_complete |= s.is_complete();
                }
                MatcherState::Stateless => {}
            }
        }
        let deferred_incomplete = self
            .deferred
            .as_ref()
            .is_some_and(|d| d.active && !d.matcher.is_complete());

        (any_eager && !eager_complete) || deferred_incomplete
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

    /// The repetition-penalty n-gram accumulator (read-only
    /// observability — probes and tests compare fold results;
    /// [`crate::NGramStats`] derives `PartialEq` for exactly that).
    pub fn ngram_stats(&self) -> &crate::NGramStats {
        &self.ngram_stats
    }

    /// The prose-corpus step counter (see the field docs).
    pub fn step(&self) -> u64 {
        self.step
    }

    /// The call-local constrained-region accumulator (read-only
    /// observability, mirroring [`Self::ngram_stats`]).
    pub fn constrained_ngram_stats(&self) -> &crate::NGramStats {
        &self.constrained_ngram_stats
    }

    /// The constrained-region step counter (see the field docs).
    pub fn constrained_step(&self) -> u64 {
        self.constrained_step
    }

    /// True iff `ngram` is in the resolved repetition ignore set —
    /// the seeding fold skips these, matching the live pass.
    pub(crate) fn resolved_ignored_contains(
        &self,
        ngram: &crate::NGram,
    ) -> bool {
        self.resolved_ignored.contains(ngram)
    }

    /// Record a prompt n-gram occurrence into the stats accumulator
    /// (the Session's prose-seeding fold).
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
    ///   identical compiled grammar (`CompiledGrammar::source_hash`;
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
            step: cached.step,
            resolved_ignored: config
                .repetition
                .as_ref()
                .map(|r| r.resolved_ignored(model))
                .unwrap_or_default(),
            // The determinism linchpin: the constrained-region
            // accumulator is call-local and starts EMPTY on resume —
            // deliberately NOT cloned from `cached`. A resumed call and
            // a cold-prefill call must derive identical states at every
            // breakpoint, and cold fold cannot (and must not) recover
            // mid-call ephemera. Session re-seeds it from the folded
            // corpus AFTER the snapshots (#106) — history pressure
            // flows through prompt content, never through this door.
            // See the field docs.
            constrained_ngram_stats: NGramStats::default(),
            constrained_step: 0,
        }
    }

    /// Reset every constraint matcher (eager, JSON, deferred) to its
    /// grammar's root, keeping the stream fields (`mu`, rng, n-gram
    /// stats, `step`) intact.
    ///
    /// [`resumed_from`](Self::resumed_from) carries matcher positions
    /// whenever the grammar identity matches, which is correct only
    /// when generation resumes the snapshotted assistant turn itself
    /// (assistant-prefill / partial-completion). When the new call
    /// folds prompt content *past* the snapshot — a seated tool
    /// result, a new user turn — the snapshotted turn has closed and
    /// the call opens a fresh assistant turn, so a carried position
    /// (typically parked at tool-call-complete, where only the turn
    /// terminator is legal) would mask every real candidate and force
    /// an immediate EOS: the 0-output-token second-round bug. The
    /// session decides continuity (`matcher_carry_valid` in
    /// `session::build_initial_state`); this is the reset half.
    pub(crate) fn reset_constraints(&mut self, config: &SamplerConfig) {
        self.matchers = config
            .modes
            .iter()
            .map(|mode| match mode {
                SamplingMode::Grammar(compiled) => MatcherState::Grammar {
                    grammar: compiled.source_hash(),
                    stack: compiled.root_state(),
                },
                SamplingMode::Json => MatcherState::Json(JsonState::new()),
                _ => MatcherState::Stateless,
            })
            .collect();
        self.deferred =
            config
                .deferred_grammar
                .as_ref()
                .map(|spec| DeferredMatcher {
                    active: false,
                    grammar: spec.grammar.source_hash(),
                    matcher: spec.grammar.root_state(),
                });
    }

    /// Read-only: would `token`'s piece bring any incomplete
    /// constraint (eager matcher, or the activated deferred grammar)
    /// to its accept state — or is one already there? The predictor
    /// evaluates this for the TERMINAL token, which must never mutate
    /// the state (tip invariant) but still decides generation
    /// validity: in the dialect-exit-marker shape (Gemma 4's
    /// `<|tool_response>`) the grammar's required exit bytes are also
    /// a vocab EOG/stop, so the token that ends generation is the
    /// same token that completes the grammar. Mirrors the byte rules
    /// of the masked filters (`completes_with`; empty pieces complete
    /// nothing).
    pub(crate) fn completes_with_terminal<M: Model>(
        &self,
        config: &SamplerConfig,
        token: Token,
        model: &M,
    ) -> bool {
        let mut buf: Vec<u8> = Vec::with_capacity(32);
        model.token_to_piece_ref(token, &mut buf);
        let modes = config.modes.iter().zip(self.matchers.iter()).any(
            |(mode, matcher)| match (mode, matcher) {
                (
                    SamplingMode::Grammar(compiled),
                    MatcherState::Grammar { stack, .. },
                ) => {
                    stack.is_complete()
                        || (!buf.is_empty()
                            && stack.completes_with(&compiled.grammar, &buf))
                }
                (SamplingMode::Json, MatcherState::Json(s)) => {
                    s.is_complete()
                        || (!buf.is_empty() && s.completes_with(&buf))
                }
                _ => false,
            },
        );
        modes
            || match (&self.deferred, &config.deferred_grammar) {
                (Some(d), Some(spec)) if d.active => {
                    d.matcher.is_complete()
                        || (!buf.is_empty()
                            && d.matcher
                                .completes_with(&spec.grammar.grammar, &buf))
                }
                _ => false,
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
