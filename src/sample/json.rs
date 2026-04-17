//! JSON-constrained sampling.
//!
//! Provides [`JsonState`], a pushdown automaton that validates a byte stream
//! against RFC 8259 JSON grammar one byte at a time. Used by
//! [`SamplingMode::Json`] to reject model tokens whose bytes would produce
//! invalid JSON.
//!
//! # Lifecycle
//!
//! Two operations, both at the byte level:
//!
//! * [`JsonState::accepts_bytes`] — non-mutating; clones state internally and
//!   reports whether the bytes can extend the current parse.
//! * [`JsonState::advance_bytes`] — mutating; commits the bytes to parse
//!   state. Call exactly once per accepted token.
//!
//! # Grammar-violation fallback
//!
//! If the filter would leave zero valid candidates, [`json_filter`] forces a
//! single EOS candidate. Callers should treat this as a hard stop.
//!
//! [`SamplingMode::Json`]: crate::SamplingMode::Json

use llama_cpp_sys_3::{llama_token, llama_token_data};

use crate::{model::token_to_piece_ref, Candidates, Model};

/// Pushdown-automaton state for JSON parsing at the byte level.
///
/// Clone cost is proportional to stack depth (single-digit for typical JSON).
/// `accepts_bytes` relies on cloning for speculative simulation.
#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Clone, Debug, PartialEq)]
pub struct JsonState {
    stack: Vec<Frame>,
}

impl Default for JsonState {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonState {
    /// Construct a fresh state at the root of a JSON document.
    pub fn new() -> Self {
        Self {
            stack: vec![Frame::Root { seen_value: false }],
        }
    }

    /// Reset to a fresh state.
    pub fn reset(&mut self) {
        self.stack.clear();
        self.stack.push(Frame::Root { seen_value: false });
    }

    /// True when a complete JSON document has been consumed. A pending number
    /// at the root (e.g. after `"123"`) reports true only after a terminator
    /// byte has been seen — see [`finalize`].
    ///
    /// [`finalize`]: Self::finalize
    pub fn is_complete(&self) -> bool {
        matches!(
            self.stack.as_slice(),
            [Frame::Root { seen_value: true }]
        )
    }

    /// Current parse stack depth. Useful for UI status.
    pub fn stack_depth(&self) -> usize {
        self.stack.len()
    }

    /// Close any pending number at the top of the stack, as if a terminator
    /// byte had been observed. Call this at end-of-input to determine whether
    /// a trailing number was valid.
    ///
    /// Returns `Err` if the number is in an incomplete state (e.g. trailing
    /// `.`, `e`, `e-`).
    pub fn finalize(&mut self) -> Result<(), JsonError> {
        if let Some(Frame::Number(s)) = self.stack.last() {
            if !s.is_complete() {
                return Err(JsonError::IncompleteNumber);
            }
            self.stack.pop();
            self.on_value_complete()?;
        }
        Ok(())
    }

    /// True iff feeding `bytes` would succeed from the current state. Does
    /// not mutate `self`.
    pub fn accepts_bytes(&self, bytes: &[u8]) -> bool {
        let mut clone = self.clone();
        for &b in bytes {
            if clone.feed(b).is_err() {
                return false;
            }
        }
        true
    }

    /// Commit `bytes` to parse state. Call after sampling selects a token.
    pub fn advance_bytes(&mut self, bytes: &[u8]) -> Result<(), JsonError> {
        for &b in bytes {
            self.feed(b)?;
        }
        Ok(())
    }

    /// Feed a single byte. May loop internally when a number or literal frame
    /// must be popped on lookahead.
    fn feed(&mut self, b: u8) -> Result<(), JsonError> {
        // At most one pop-and-retry per number frame, and literals consume
        // their bytes without retries — the loop is bounded by stack depth.
        for _ in 0..64 {
            let top = self.stack.last_mut().ok_or(JsonError::Internal)?;
            match top {
                Frame::Root { seen_value } => {
                    if *seen_value {
                        if is_ws(b) {
                            return Ok(());
                        }
                        return Err(JsonError::UnexpectedByte(b));
                    }
                    if is_ws(b) {
                        return Ok(());
                    }
                    return self.start_value(b);
                }
                Frame::Object(state) => {
                    match state {
                        ObjState::EmptyOrAwaitingKey => {
                            if is_ws(b) {
                                return Ok(());
                            }
                            if b == b'}' {
                                self.stack.pop();
                                return self.on_value_complete();
                            }
                            if b == b'"' {
                                *state = ObjState::AwaitingColon;
                                self.stack.push(Frame::String(StringState::Normal));
                                return Ok(());
                            }
                        }
                        ObjState::AwaitingKey => {
                            if is_ws(b) {
                                return Ok(());
                            }
                            if b == b'"' {
                                *state = ObjState::AwaitingColon;
                                self.stack.push(Frame::String(StringState::Normal));
                                return Ok(());
                            }
                        }
                        ObjState::AwaitingColon => {
                            if is_ws(b) {
                                return Ok(());
                            }
                            if b == b':' {
                                *state = ObjState::AwaitingValue;
                                return Ok(());
                            }
                        }
                        ObjState::AwaitingValue => {
                            if is_ws(b) {
                                return Ok(());
                            }
                            return self.start_value(b);
                        }
                        ObjState::AwaitingCommaOrClose => {
                            if is_ws(b) {
                                return Ok(());
                            }
                            if b == b',' {
                                *state = ObjState::AwaitingKey;
                                return Ok(());
                            }
                            if b == b'}' {
                                self.stack.pop();
                                return self.on_value_complete();
                            }
                        }
                    }
                    return Err(JsonError::UnexpectedByte(b));
                }
                Frame::Array(state) => {
                    match state {
                        ArrState::EmptyOrAwaitingValue => {
                            if is_ws(b) {
                                return Ok(());
                            }
                            if b == b']' {
                                self.stack.pop();
                                return self.on_value_complete();
                            }
                            return self.start_value(b);
                        }
                        ArrState::AwaitingValue => {
                            if is_ws(b) {
                                return Ok(());
                            }
                            return self.start_value(b);
                        }
                        ArrState::AwaitingCommaOrClose => {
                            if is_ws(b) {
                                return Ok(());
                            }
                            if b == b',' {
                                *state = ArrState::AwaitingValue;
                                return Ok(());
                            }
                            if b == b']' {
                                self.stack.pop();
                                return self.on_value_complete();
                            }
                            return Err(JsonError::UnexpectedByte(b));
                        }
                    }
                }
                Frame::String(state) => {
                    let next = match state {
                        StringState::Normal => match b {
                            b'"' => {
                                self.stack.pop();
                                return self.on_value_complete();
                            }
                            b'\\' => Some(StringState::AfterEscape),
                            0x00..=0x1F => {
                                return Err(JsonError::UnexpectedByte(b));
                            }
                            _ => Some(StringState::Normal),
                        },
                        StringState::AfterEscape => match b {
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n'
                            | b'r' | b't' => Some(StringState::Normal),
                            b'u' => Some(StringState::Hex(0)),
                            _ => return Err(JsonError::UnexpectedByte(b)),
                        },
                        StringState::Hex(idx) => {
                            if !is_hex(b) {
                                return Err(JsonError::UnexpectedByte(b));
                            }
                            if *idx == 3 {
                                Some(StringState::Normal)
                            } else {
                                Some(StringState::Hex(*idx + 1))
                            }
                        }
                    };
                    if let Some(new_state) = next {
                        *state = new_state;
                    }
                    return Ok(());
                }
                Frame::Number(state) => {
                    if let Some(new_state) = state.feed(b) {
                        *state = new_state;
                        return Ok(());
                    }
                    if !state.is_complete() {
                        return Err(JsonError::UnexpectedByte(b));
                    }
                    self.stack.pop();
                    self.on_value_complete()?;
                    // Retry this byte against the new top frame.
                    continue;
                }
                Frame::Literal { kind, idx } => {
                    let bytes = kind.bytes();
                    let i = *idx as usize;
                    if i >= bytes.len() {
                        return Err(JsonError::Internal);
                    }
                    if b != bytes[i] {
                        return Err(JsonError::UnexpectedByte(b));
                    }
                    *idx += 1;
                    if (*idx as usize) == bytes.len() {
                        self.stack.pop();
                        return self.on_value_complete();
                    }
                    return Ok(());
                }
            }
        }
        Err(JsonError::Internal)
    }

    fn start_value(&mut self, b: u8) -> Result<(), JsonError> {
        let frame = match b {
            b'{' => Frame::Object(ObjState::EmptyOrAwaitingKey),
            b'[' => Frame::Array(ArrState::EmptyOrAwaitingValue),
            b'"' => Frame::String(StringState::Normal),
            b't' => Frame::Literal { kind: LitKind::True, idx: 1 },
            b'f' => Frame::Literal { kind: LitKind::False, idx: 1 },
            b'n' => Frame::Literal { kind: LitKind::Null, idx: 1 },
            b'-' => Frame::Number(NumState::SeenMinus),
            b'0' => Frame::Number(NumState::SeenZero),
            b'1'..=b'9' => Frame::Number(NumState::SeenInt),
            _ => return Err(JsonError::UnexpectedByte(b)),
        };
        self.stack.push(frame);
        Ok(())
    }

    /// Called after a value-producing frame (String/Number/Literal/Object/
    /// Array close) has been popped. Transitions the new top frame.
    fn on_value_complete(&mut self) -> Result<(), JsonError> {
        let top = self.stack.last_mut().ok_or(JsonError::Internal)?;
        match top {
            Frame::Root { seen_value } => {
                if *seen_value {
                    return Err(JsonError::Internal);
                }
                *seen_value = true;
            }
            Frame::Object(state) => match state {
                ObjState::AwaitingColon => {
                    // Key string just completed.
                }
                ObjState::AwaitingValue => {
                    *state = ObjState::AwaitingCommaOrClose;
                }
                _ => return Err(JsonError::Internal),
            },
            Frame::Array(state) => match state {
                ArrState::EmptyOrAwaitingValue
                | ArrState::AwaitingValue => {
                    *state = ArrState::AwaitingCommaOrClose;
                }
                _ => return Err(JsonError::Internal),
            },
            Frame::String(_) | Frame::Number(_) | Frame::Literal { .. } => {
                return Err(JsonError::Internal);
            }
        }
        Ok(())
    }
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Clone, Debug, PartialEq)]
enum Frame {
    Root { seen_value: bool },
    Object(ObjState),
    Array(ArrState),
    String(StringState),
    Number(NumState),
    /// Matching a fixed literal (`true`, `false`, `null`). `idx` is the
    /// byte offset of the next expected byte within the literal.
    Literal { kind: LitKind, idx: u8 },
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum LitKind {
    True,
    False,
    Null,
}

impl LitKind {
    fn bytes(self) -> &'static [u8] {
        match self {
            LitKind::True => b"true",
            LitKind::False => b"false",
            LitKind::Null => b"null",
        }
    }
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum ObjState {
    EmptyOrAwaitingKey,
    AwaitingKey,
    AwaitingColon,
    AwaitingValue,
    AwaitingCommaOrClose,
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum ArrState {
    EmptyOrAwaitingValue,
    AwaitingValue,
    AwaitingCommaOrClose,
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum StringState {
    Normal,
    AfterEscape,
    /// Collecting \uXXXX; the u8 tracks the index of the next hex digit (0..=3).
    Hex(u8),
}

#[cfg_attr(
    feature = "serde",
    derive(rocket::serde::Deserialize, rocket::serde::Serialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "rocket::serde"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum NumState {
    SeenMinus,
    SeenZero,
    SeenInt,
    SeenDot,
    SeenFrac,
    SeenE,
    SeenExpSign,
    SeenExp,
}

impl NumState {
    fn is_complete(self) -> bool {
        matches!(
            self,
            NumState::SeenZero
                | NumState::SeenInt
                | NumState::SeenFrac
                | NumState::SeenExp
        )
    }

    /// Feed a byte to a number frame. Returns `Some(new_state)` if the byte
    /// continues the number, `None` if it does not (caller decides whether
    /// this is a valid end-of-number or an error).
    fn feed(self, b: u8) -> Option<Self> {
        match self {
            NumState::SeenMinus => match b {
                b'0' => Some(NumState::SeenZero),
                b'1'..=b'9' => Some(NumState::SeenInt),
                _ => None,
            },
            NumState::SeenZero => match b {
                b'.' => Some(NumState::SeenDot),
                b'e' | b'E' => Some(NumState::SeenE),
                _ => None,
            },
            NumState::SeenInt => match b {
                b'0'..=b'9' => Some(NumState::SeenInt),
                b'.' => Some(NumState::SeenDot),
                b'e' | b'E' => Some(NumState::SeenE),
                _ => None,
            },
            NumState::SeenDot => match b {
                b'0'..=b'9' => Some(NumState::SeenFrac),
                _ => None,
            },
            NumState::SeenFrac => match b {
                b'0'..=b'9' => Some(NumState::SeenFrac),
                b'e' | b'E' => Some(NumState::SeenE),
                _ => None,
            },
            NumState::SeenE => match b {
                b'+' | b'-' => Some(NumState::SeenExpSign),
                b'0'..=b'9' => Some(NumState::SeenExp),
                _ => None,
            },
            NumState::SeenExpSign => match b {
                b'0'..=b'9' => Some(NumState::SeenExp),
                _ => None,
            },
            NumState::SeenExp => match b {
                b'0'..=b'9' => Some(NumState::SeenExp),
                _ => None,
            },
        }
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_hex(b: u8) -> bool {
    matches!(b, b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F')
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum JsonError {
    #[error("unexpected byte {0:#04x} for current JSON state")]
    UnexpectedByte(u8),
    #[error("input ended with incomplete number")]
    IncompleteNumber,
    #[error("internal state inconsistency")]
    Internal,
}

static_assertions::assert_impl_all!(JsonError: Send, Sync);
static_assertions::assert_impl_all!(JsonState: Send, Sync);

/// Filter candidates to only those whose token bytes extend the current JSON
/// parse state.
///
/// On grammar violation (zero valid candidates), returns a single-token
/// [`Candidates`] containing EOS. The caller's stop-sequence machinery will
/// then terminate generation.
pub(crate) fn json_filter(
    candidates: Candidates,
    state: &JsonState,
    model: &Model,
) -> Candidates {
    let mut buf: Vec<u8> = Vec::with_capacity(32);
    let mut kept: Vec<llama_token_data> = Vec::with_capacity(candidates.len().get());
    for cand in candidates.as_slice() {
        buf.clear();
        token_to_piece_ref(cand.id, model, &mut buf);
        if state.accepts_bytes(&buf) {
            kept.push(*cand);
        }
    }

    if kept.is_empty() {
        let eos = llama_token_data {
            id: model.eos(),
            logit: 0.0,
            p: 1.0,
        };
        return Candidates::from_vec(vec![eos]);
    }

    Candidates::from_vec_unchecked(kept)
}

/// Advance every `SamplingMode::Json` state in `modes` by the bytes of
/// `token`.
///
/// # Panics
///
/// Panics if any `SamplingMode::Json` mutex is poisoned. A poisoned mutex
/// means a previous panic left the parser state undefined, and silently
/// continuing would produce output that violates the grammar — contrary to
/// the [`SamplingMode::Json`] contract. Rebuild the mode and retry.
///
/// [`SamplingMode::Json`]: crate::SamplingMode::Json
pub(crate) fn advance_all(
    modes: &[crate::SamplingMode],
    token: llama_token,
    model: &Model,
) {
    use crate::SamplingMode;
    let mut buf: Vec<u8> = Vec::new();
    let mut computed = false;
    for mode in modes {
        if let SamplingMode::Json(state) = mode {
            if !computed {
                token_to_piece_ref(token, model, &mut buf);
                computed = true;
            }
            let mut locked = state.lock().expect(
                "SamplingMode::Json mutex poisoned during advance; parser \
                 state is unrecoverable. Rebuild the mode with \
                 SamplingMode::json() and retry.",
            );
            // An advance error means the grammar was violated on the prior
            // step (EOS fallback chose an out-of-grammar token). Generation
            // will terminate on the next step via the stop_sequence for
            // EOS, so we silently no-op on this specific failure mode.
            let _ = locked.advance_bytes(&buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(s: &mut JsonState, input: &str) -> Result<(), JsonError> {
        s.advance_bytes(input.as_bytes())
    }

    fn accepts(state: &JsonState, input: &str) -> bool {
        state.accepts_bytes(input.as_bytes())
    }

    fn accepts_complete(input: &str) -> bool {
        let mut s = JsonState::new();
        if feed_all(&mut s, input).is_err() {
            return false;
        }
        // Close any trailing number.
        if s.finalize().is_err() {
            return false;
        }
        s.is_complete()
    }

    #[test]
    fn literals_accept() {
        for s in &["true", "false", "null"] {
            assert!(accepts_complete(s), "should accept {s}");
        }
    }

    #[test]
    fn literal_case_sensitive() {
        for s in &["True", "FALSE", "Null", "tru", "falsey", "nullx"] {
            assert!(!accepts_complete(s), "should reject {s}");
        }
    }

    #[test]
    fn numbers_accept() {
        for s in &[
            "0", "1", "-1", "123", "-123", "0.5", "-0.5", "1.5", "1e10",
            "1E10", "1e+10", "1e-10", "-1.5e+10", "0.0", "1234567890",
        ] {
            assert!(accepts_complete(s), "should accept {s}");
        }
    }

    #[test]
    fn numbers_reject_malformed() {
        // `01` is invalid per RFC 8259 (leading zero only allowed alone).
        for s in &["01", "--1", "1.", "1e", "1e+", ".5", "1..5", "1e1.5"] {
            assert!(!accepts_complete(s), "should reject {s}");
        }
    }

    #[test]
    fn strings_accept() {
        for s in &[
            r#""""#,
            r#""hello""#,
            r#""with spaces""#,
            r#""\n""#,
            r#""\"\\\/\b\f\n\r\t""#,
            r#""\u00AB""#,
            r#""\uD83D\uDE00""#,
            r#""unicode: café""#,
        ] {
            assert!(accepts_complete(s), "should accept {s}");
        }
    }

    #[test]
    fn strings_reject_malformed() {
        // Unterminated, bad escape, control char inside string, bad hex.
        for s in &[
            r#"""#,
            r#""unterminated"#,
            "\"\x01\"",
            "\"\n\"",
            r#""\x""#,
            r#""\u00""#,
            r#""\uZZZZ""#,
        ] {
            assert!(!accepts_complete(s), "should reject {s}");
        }
    }

    #[test]
    fn objects_accept() {
        for s in &[
            "{}",
            r#"{"a":1}"#,
            r#"{"a": 1}"#,
            r#"{"a": 1, "b": 2}"#,
            r#"{ "a" : 1 }"#,
            r#"{"nested":{"inner":true}}"#,
            r#"{"arr":[1,2,3]}"#,
        ] {
            assert!(accepts_complete(s), "should accept {s}");
        }
    }

    #[test]
    fn objects_reject_malformed() {
        for s in &[
            "{",
            "}",
            "{,}",
            r#"{"a"}"#,
            r#"{"a":}"#,
            r#"{:1}"#,
            r#"{"a":1,}"#,
            r#"{"a":1 "b":2}"#,
        ] {
            assert!(!accepts_complete(s), "should reject {s}");
        }
    }

    #[test]
    fn arrays_accept() {
        for s in &[
            "[]",
            "[1]",
            "[1,2,3]",
            "[1, 2, 3]",
            "[ 1 , 2 , 3 ]",
            r#"["a","b","c"]"#,
            "[[1,2],[3,4]]",
            r#"[{"a":1},{"b":2}]"#,
        ] {
            assert!(accepts_complete(s), "should accept {s}");
        }
    }

    #[test]
    fn arrays_reject_malformed() {
        for s in &["[", "]", "[,]", "[1,]", "[1 2]"] {
            assert!(!accepts_complete(s), "should reject {s}");
        }
    }

    #[test]
    fn whitespace_permitted() {
        for s in &[
            "  true  ",
            "\t\nfalse\r\n",
            r#"  { "a" : 1 }  "#,
            "\n[ 1 , 2 ]\n",
        ] {
            assert!(accepts_complete(s), "should accept {s}");
        }
    }

    #[test]
    fn partial_acceptance_mid_string() {
        // Build up a state: opened an object, started a key string.
        let mut s = JsonState::new();
        feed_all(&mut s, "{\"hel").unwrap();
        // Should accept a continuation of the key (including `]` — that's a
        // valid literal character inside a JSON string).
        assert!(accepts(&s, "lo\""));
        assert!(accepts(&s, "lo]\""));
        // Should reject an unescaped control char inside a string.
        assert!(!accepts(&s, "\n"));
        assert!(!accepts(&s, "\t"));
        // Bad escape is rejected.
        assert!(!accepts(&s, "\\x"));
    }

    #[test]
    fn partial_acceptance_awaiting_value() {
        // After `{"a":` we should accept any value start but not `,` or `}`.
        let mut s = JsonState::new();
        feed_all(&mut s, "{\"a\":").unwrap();
        assert!(accepts(&s, "1"));
        assert!(accepts(&s, "\"v\""));
        assert!(accepts(&s, "true"));
        assert!(accepts(&s, "[]"));
        assert!(accepts(&s, "{}"));
        assert!(!accepts(&s, ","));
        assert!(!accepts(&s, "}"));
    }

    #[test]
    fn no_trailing_garbage() {
        // Complete doc, then attempt another value.
        let mut s = JsonState::new();
        feed_all(&mut s, "true").unwrap();
        // Whitespace OK.
        assert!(accepts(&s, "   "));
        // Another value not OK.
        assert!(!accepts(&s, "false"));
    }

    #[test]
    fn is_complete_timing() {
        let mut s = JsonState::new();
        feed_all(&mut s, "{").unwrap();
        assert!(!s.is_complete());
        feed_all(&mut s, "}").unwrap();
        assert!(s.is_complete());
    }

    #[test]
    fn finalize_closes_pending_number() {
        let mut s = JsonState::new();
        feed_all(&mut s, "42").unwrap();
        assert!(!s.is_complete());
        s.finalize().unwrap();
        assert!(s.is_complete());

        // Incomplete number: "1e" is SeenE, not a complete state.
        let mut s = JsonState::new();
        feed_all(&mut s, "1e").unwrap();
        assert_eq!(s.finalize(), Err(JsonError::IncompleteNumber));
    }

    #[test]
    fn number_lookahead_terminates_at_comma() {
        // Parse [1,2] — number 1 ends at comma.
        assert!(accepts_complete("[1,2]"));
    }

    #[test]
    fn reset_clears_state() {
        let mut s = JsonState::new();
        feed_all(&mut s, "{\"a\":1").unwrap();
        s.reset();
        feed_all(&mut s, "[]").unwrap();
        assert!(s.is_complete());
    }

    #[test]
    fn deep_nesting() {
        let nested_open: String = "[".repeat(64);
        let nested_close: String = "]".repeat(64);
        let full = format!("{nested_open}{nested_close}");
        assert!(accepts_complete(&full));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_idle() {
        use rocket::serde::json::to_string;
        let s = JsonState::new();
        let encoded = to_string(&s).unwrap();
        let decoded: JsonState =
            rocket::serde::json::from_str(&encoded).unwrap();
        assert_eq!(s, decoded);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_mid_parse() {
        use rocket::serde::json::{from_str, to_string};
        let mut s = JsonState::new();
        feed_all(&mut s, "{\"a\":[1,2").unwrap();
        let encoded = to_string(&s).unwrap();
        let decoded: JsonState = from_str(&encoded).unwrap();
        assert_eq!(s, decoded);
        // Decoded state should accept continued input that the original would.
        let mut a = s.clone();
        let mut b = decoded;
        feed_all(&mut a, ",3]}").unwrap();
        feed_all(&mut b, ",3]}").unwrap();
        assert!(a.is_complete());
        assert!(b.is_complete());
    }

    /// End-to-end integration test: prompt a real model to emit a JSON
    /// character sheet with `SamplingMode::Json` in the chain, then assert
    /// the output parses as valid JSON.
    ///
    /// # Note on what this verifies
    ///
    /// * **Asserts**: the generated text parses via `serde_json::from_str`.
    /// * **Does NOT assert**: schema conformance. The constraint only
    ///   guarantees valid JSON syntax, not that fields match the prompt's
    ///   schema. A model that emits `{}` or `null` passes this test — which
    ///   is the correct semantic, since those are valid documents.
    ///
    /// The constraint is advisory: it rejects tokens that would produce
    /// invalid JSON, but it can't force a specific shape.
    #[cfg(feature = "serde")]
    #[test]
    #[ignore = "requires model"]
    fn json_integration_character_sheet() {
        use crate::{
            Engine, PredictOptions, SampleOptions, SamplingMode,
        };
        use std::{num::NonZeroUsize, path::PathBuf};

        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/model.gguf");
        let mut engine = Engine::from_path(model_path).unwrap();

        // Ask for a JSON character sheet. We use a minimalist system-style
        // prompt rather than a chat template so the test works regardless
        // of which model is symlinked.
        const PROMPT: &str = "Output only valid JSON for a fantasy RPG \
            character with these fields: name (string), class (string), \
            level (integer), hp (integer), alive (bool), inventory (array \
            of strings). Do not include any prose. JSON: ";

        let tokens = engine.model.tokenize(PROMPT, false);

        let mut opts = PredictOptions::default();
        opts.n = NonZeroUsize::new(256).unwrap();
        // JSON first so invalid tokens are pruned before locally-typical
        // sampling picks among what remains.
        opts.sample_options = SampleOptions {
            modes: vec![
                SamplingMode::json(),
                SamplingMode::locally_typical(),
            ],
            ..SampleOptions::default()
        };

        let predictor = engine.predict_pieces(tokens, opts);
        let output: String = predictor.collect();

        println!("=== Generated JSON ===\n{output}\n======================");

        // Must parse as valid JSON of any shape.
        let parsed: rocket::serde::json::Value =
            rocket::serde::json::from_str(output.trim())
                .unwrap_or_else(|e| {
                    panic!(
                        "output must be valid JSON (constraint guarantees \
                         this). parse error: {e}\noutput: {output:?}"
                    )
                });

        // Log what we got. We intentionally do not assert schema: the
        // constraint is syntactic, not semantic.
        println!("parsed as: {parsed:#}");
    }
}
