//! Shared helpers for the integration test suite.
//!
//! Cargo compiles files directly under `tests/` as separate test
//! binaries but leaves subdirectories (`tests/common/`) alone, so this
//! module is included with `mod common;` from each test file that wants
//! it — it never becomes its own empty test target.

#![allow(dead_code)] // not every consumer uses every helper

use std::num::NonZeroU128;

/// A seed for a sampling test that is *reproducible on failure*.
///
/// A model-backed test left on a random seed can detect a defect but
/// can't replay it — you never learn which trajectory broke. This
/// resolves a seed from `DRAMA_LLAMA_TEST_SEED` if set, otherwise picks
/// a fresh random one, and always prints it to stderr. `cargo test`
/// captures stderr and shows it only for failing tests, so a green run
/// stays quiet and a red run hands you the exact replay command:
///
/// ```text
/// DRAMA_LLAMA_TEST_SEED=<n> cargo test <name> -- --ignored
/// ```
///
/// This is the proptest/quickcheck pattern: keep the fuzzing (fresh
/// entropy by default) but make every failure reproducible. Use it in
/// place of `.with_seed(None)` (or an unset seed) wherever a test
/// samples and could flake — see `tests/session.rs`'s round-trip test
/// and issues #53 / #57.
///
/// Note: a `#[cfg(test)]`-gated print inside the library would *not*
/// fire here — integration tests link the crate built without
/// `cfg(test)` — which is exactly why this lives in test code.
pub fn test_seed() -> NonZeroU128 {
    let seed = std::env::var("DRAMA_LLAMA_TEST_SEED")
        .ok()
        .and_then(|v| v.parse::<u128>().ok())
        .and_then(NonZeroU128::new)
        // `.max(1)` keeps the value non-zero (`init_state` rejects 0).
        .unwrap_or_else(|| {
            NonZeroU128::new(rand::random::<u128>().max(1)).unwrap()
        });
    eprintln!("test seed = {seed}  (DRAMA_LLAMA_TEST_SEED={seed} to replay)");
    seed
}
