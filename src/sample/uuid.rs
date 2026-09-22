//! UUID detection for the repetition penalty
//! ([`IgnoreCategory::Uuids`](crate::data::ignore_category::IgnoreCategory::Uuids)).
//!
//! The penalty pass sees token ids; a UUID is a *byte* shape. So the
//! tracker rides [`SamplerState`](crate::SamplerState), is fed each
//! accepted token's piece from `SamplerState::advance`, and the pass
//! consults [`UuidTracker::suspended`] before doing anything. Homing it
//! in the state — rather than deriving it from the predictor's text —
//! is what keeps a mid-call serialize/restore replaying the identical
//! stream: anything that influences logits rides the state.

/// Total length of the canonical `8-4-4-4-12` form.
const LEN: u8 = 36;
/// Positions that must hold a dash; every other position is hex.
const DASHES: [u8; 4] = [8, 13, 18, 23];

/// Trailing-run scanner for the canonical `8-4-4-4-12` UUID shape.
///
/// `run` is the length of the longest text-tail that is a prefix of that
/// shape (`0..=36`), case-insensitive on the hex. The scan is a pure
/// function of the byte stream, so it serializes as one byte and
/// compares by value like the rest of the sampler state.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UuidTracker {
    run: u8,
}

const fn is_hex(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

const fn is_dash_pos(pos: u8) -> bool {
    pos == DASHES[0] || pos == DASHES[1] || pos == DASHES[2] || pos == DASHES[3]
}

impl UuidTracker {
    /// Feed the bytes of one accepted piece, in order. Equivalent to
    /// feeding them one at a time.
    pub(crate) fn feed(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.push(b);
        }
    }

    /// True while the tail is an *unfinished* UUID that has passed its
    /// first dash: `9 <= run < 36`. The pass resumes at the 36th
    /// character, so the step that records the UUID's last token also
    /// penalizes the exit token's candidates normally.
    pub(crate) fn suspended(&self) -> bool {
        self.run > DASHES[0] && self.run < LEN
    }

    /// Forget the tail. Turn-structure state, reset alongside the
    /// constraint matchers when a resume opens a fresh assistant turn.
    pub(crate) fn reset(&mut self) {
        self.run = 0;
    }

    /// Trailing hex count implied by `run`: everything since the last
    /// dash (or the whole run before the first one).
    fn trailing_hex(&self) -> u8 {
        let last_dash = DASHES.iter().rev().find(|&&d| d < self.run);
        match last_dash {
            Some(d) => self.run - (d + 1),
            None => self.run,
        }
    }

    fn push(&mut self, b: u8) {
        let pos = self.run;
        let at_dash = is_dash_pos(pos);
        self.run = match (b == b'-', is_hex(b)) {
            // Fits the shape at this position.
            (true, _) if at_dash => pos + 1,
            (_, true) if !at_dash && pos < LEN => pos + 1,
            // Hex where a dash (or the end) was due: the shape is broken,
            // but the tail's last <= 8 hex bytes may still open a UUID —
            // `abc3fa85f64-…` must detect, and a 37th hex byte leaves a
            // clean 8-hex tail behind it.
            (_, true) => (self.trailing_hex() + 1).min(DASHES[0]),
            // Dash where hex was due: only a full 8-hex tail restarts
            // (`…-f00d-12345678-`); anything shorter is not a UUID.
            (true, _) if self.trailing_hex() == DASHES[0] => DASHES[0] + 1,
            _ => 0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UUID: &[u8] = b"3fa85f64-5717-4562-b3fc-2c963f66afa6";

    fn fed(bytes: &[u8]) -> UuidTracker {
        let mut t = UuidTracker::default();
        t.feed(bytes);
        t
    }

    /// Suspended from the first dash, released exactly at the 36th
    /// character — for both cases of hex.
    #[test]
    fn suspended_from_first_dash_until_complete() {
        for uuid in [UUID.to_vec(), UUID.to_ascii_uppercase()] {
            let mut t = UuidTracker::default();
            for (i, &b) in uuid.iter().enumerate() {
                t.push(b);
                let n = i + 1;
                assert_eq!(t.run as usize, n, "run tracks the prefix");
                assert_eq!(
                    t.suspended(),
                    (9..36).contains(&n),
                    "byte {n}: suspended iff past the first dash and unfinished"
                );
            }
            assert_eq!(t.run, 36);
            assert!(!t.suspended(), "complete: pass resumes");
        }
    }

    /// Hex words and ISO dates never suspend.
    #[test]
    fn hex_word_and_iso_date_never_suspend() {
        for text in [
            &b"deadbeef"[..],
            b"deadbeefdeadbeef",
            b"2026-09-22",
            b"cafe-babe",
            b"0123456789abcdef0123456789abcdef0123456789abcdef",
        ] {
            let mut t = UuidTracker::default();
            for &b in text {
                t.push(b);
                assert!(!t.suspended(), "{:?}", std::str::from_utf8(text));
            }
        }
    }

    /// A UUID glued to preceding hex still detects — the trailing-hex
    /// rule, not a naive reset-to-one.
    #[test]
    fn detects_uuid_after_leading_hex() {
        let mut text = b"abc".to_vec();
        text.extend_from_slice(UUID);
        let t = fed(&text[..3 + 9]);
        assert_eq!(t.run, 9);
        assert!(t.suspended());
        assert_eq!(fed(&text).run, 36);
    }

    /// A dash after a full eight-hex tail restarts the shape mid-run.
    #[test]
    fn restarts_on_dash_after_eight_hex() {
        let t = fed(b"deadbeef-f00d-12345678-");
        assert_eq!(t.run, 9);
        assert!(t.suspended());
        // ...and a dash after fewer than eight does not.
        assert_eq!(fed(b"deadbeef-f00d-1234567-").run, 0);
    }

    /// A 37th hex byte breaks the UUID but leaves an 8-hex tail.
    #[test]
    fn thirty_seventh_hex_resets_to_eight() {
        let mut text = UUID.to_vec();
        text.push(b'a');
        let t = fed(&text);
        assert_eq!(t.run, 8);
        assert!(!t.suspended());
    }

    /// Any non-shape byte forgets the tail; the compact-timestamp false
    /// positive releases at its first non-fitting byte.
    #[test]
    fn non_shape_byte_resets() {
        let mut t = fed(b"20260922-1430");
        assert!(t.suspended(), "documented false positive");
        t.push(b':');
        assert_eq!(t.run, 0);
        let mut t = fed(&UUID[..20]);
        assert!(t.suspended());
        t.push(b' ');
        assert!(!t.suspended());
        assert_eq!(t.run, 0);
    }

    /// Feeding a piece at once is feeding it byte by byte.
    #[test]
    fn feed_is_per_byte() {
        let mut whole = UuidTracker::default();
        whole.feed(UUID);
        let mut split = UuidTracker::default();
        for chunk in UUID.chunks(5) {
            split.feed(chunk);
        }
        assert_eq!(whole, split);
    }

    #[test]
    fn reset_forgets() {
        let mut t = fed(&UUID[..20]);
        assert!(t.suspended());
        t.reset();
        assert_eq!(t, UuidTracker::default());
    }
}
