//! A tool call's input read up to where its generation was cut short
//! (#121, #122), returned the way Anthropic returns one.
//!
//! Captured on claude-haiku-4-5 (2026-09-30, raw bytes; misanthropic's
//! `misanthropic/test/data/stop/`):
//!
//! - **`max_tokens` mid-input** keeps only the members that *completed*;
//!   the member being generated is dropped whole, however far it got
//!   (`clip_tool.*` gave `{"path":"hello.py"}`; `clip_long_tool.*`, 140
//!   output tokens into a 200-word `contents` string, still
//!   `{"path":"story.txt"}`). Streamed, the `input_json_delta` chunks
//!   stop at the last completed member (`{"path": "story.txt"`) and the
//!   block never gets `content_block_stop`.
//! - **A stop sequence inside a string value** keeps that string, cut
//!   right before the match, and closes the JSON
//!   (`stop_sequence_tool.*`: `{"path":"hello.py","contents":"import
//!   datetime\n"}` for `stop_sequences: ["print("]`).
//!
//! Top-level members are captured. That nested containers keep their
//! completed members at every depth, the one in progress dropped, is
//! inferred: it is the top-level rule applied at each level.
//!
//! Everything here is pure: the dialect parser hands over the text of
//! the value in flight, and gets back the value a cut leaves of it.

use serde_json::{Map, Value};

/// What becomes of a string value the input ended inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpenStrings {
    /// Dropped, with its member — what a clip returns.
    Drop,
    /// Kept as far as it is known to be text: a tail that could still
    /// turn out to be the start of the string's close marker is held
    /// back. What a stop sequence is matched against.
    Held,
    /// Kept as far as it got, every byte. What locates a cut in the raw
    /// generation.
    Raw,
}

/// How strings and keys are spelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Flavor<'q> {
    /// JSON: `"`-quoted strings with escapes, quoted keys.
    Json,
    /// Gemma 4's dict encoding: bare keys, strings between two `quote`
    /// markers with no escaping, `none`/`None` for null.
    Dict { quote: &'q str },
}

/// A container read up to where its input ended.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Truncated {
    /// The container, per [`OpenStrings`]: its completed members at
    /// every depth, and a container in progress kept with *its*
    /// completed members.
    pub value: Value,
    /// Containers still open where the input ended, this one included;
    /// 0 when it closed.
    pub open: usize,
    /// The keys (array positions, spelled as numbers) from this
    /// container down to the innermost one still open: `open - 1` of
    /// them.
    pub path: Vec<String>,
}

/// Read the JSON object or array at the start of `text` (leading
/// whitespace allowed) up to where `text` ends. `None` when `text`
/// holds no container, or a malformed one.
///
/// A member is complete once its value is: a string at its closing
/// quote, a container at its closer, a literal once spelled, a number
/// once something follows it (at the end of input it could still grow
/// — and on a clip, would have).
pub(crate) fn read_partial(
    text: &str,
    flavor: Flavor<'_>,
    strings: OpenStrings,
) -> Option<Truncated> {
    let mut reader = Reader {
        text,
        pos: 0,
        flavor,
        strings,
    };
    reader.skip_ws();
    if !reader.rest().starts_with(['{', '[']) {
        return None;
    }
    match reader.value() {
        Read::Done(value) => Some(Truncated {
            value,
            open: 0,
            path: Vec::new(),
        }),
        Read::Open(Open {
            kept: Some(value),
            depth,
            path,
        }) => Some(Truncated {
            value,
            open: depth,
            path,
        }),
        Read::Open(_) | Read::Malformed => None,
    }
}

/// The object a clip leaves of the JSON text `text` begins with: its
/// completed members only (see `read_partial`); `{}` when none
/// completed. `None` when `text` is not the start of a JSON object.
pub fn truncate_partial_object(text: &str) -> Option<Value> {
    read_partial(text, Flavor::Json, OpenStrings::Drop)
        .map(|t| t.value)
        .filter(Value::is_object)
}

/// `value` as compact JSON left open where its input ended: the last
/// `open` closers dropped. What Anthropic streams for a call a clip cut
/// (`{"path":"story.txt"`); `open` is [`Truncated::open`].
pub(crate) fn unclosed_json(value: &Value, open: usize) -> String {
    let mut json = value.to_string();
    // The container in progress is the last member at every level
    // (maps keep insertion order), so its closers are the trailing ones.
    for _ in 0..open {
        if json.ends_with(['}', ']']) {
            json.pop();
        }
    }
    json
}

/// Bytes at the end of `text` that could be the start of `marker` —
/// the tail an open string holds back under [`OpenStrings::Held`].
pub(crate) fn marker_holdback(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .filter(|&k| marker.is_char_boundary(k))
        .find(|&k| text.ends_with(&marker[..k]))
        .unwrap_or(0)
}

/// Cut `value` at the first stop sequence in its string values, in the
/// order they were generated (maps keep insertion order): the members
/// before that string stand, the string is cut right before the match,
/// and what came after it is gone. Keys are never matched. Returns the
/// cut value and the index of the stop that matched.
pub(crate) fn cut_value<S: AsRef<str>>(
    value: &Value,
    stops: &[S],
) -> Option<(Value, usize)> {
    match value {
        Value::String(s) => crate::predictor::first_stop_string(s, stops)
            .map(|(at, i)| (Value::String(s[..at].to_owned()), i)),
        Value::Array(items) => {
            items.iter().enumerate().find_map(|(n, item)| {
                cut_value(item, stops).map(|(cut, i)| {
                    let kept = items[..n].iter().cloned().chain([cut]);
                    (Value::Array(kept.collect()), i)
                })
            })
        }
        Value::Object(map) => {
            map.iter().enumerate().find_map(|(n, (key, item))| {
                cut_value(item, stops).map(|(cut, i)| {
                    let kept: Map<String, Value> = map
                        .iter()
                        .take(n)
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .chain([(key.clone(), cut)])
                        .collect();
                    (Value::Object(kept), i)
                })
            })
        }
        _ => None,
    }
}

/// One value read up to where the input ended.
enum Read {
    Done(Value),
    Open(Open),
    Malformed,
}

/// A value the input ended inside.
#[derive(Default)]
struct Open {
    /// What [`OpenStrings`] keeps of it: `None` for a scalar, a value
    /// not begun, or a dropped string.
    kept: Option<Value>,
    /// Containers open, it included (0 unless it is a container).
    depth: usize,
    /// Keys from it down to the innermost open container.
    path: Vec<String>,
}

impl Open {
    /// A container the input ended in, its member `key` (if any) in
    /// flight as `child`.
    fn container(value: Value, member: Option<(String, Open)>) -> Self {
        match member {
            Some((key, child)) if child.depth > 0 => Self {
                kept: Some(value),
                depth: 1 + child.depth,
                path: std::iter::once(key).chain(child.path).collect(),
            },
            _ => Self {
                kept: Some(value),
                depth: 1,
                path: Vec::new(),
            },
        }
    }
}

/// A string read up to where the input ended.
enum Str {
    Done(String),
    Open(Option<String>),
    Malformed,
}

struct Reader<'t, 'q> {
    text: &'t str,
    pos: usize,
    flavor: Flavor<'q>,
    strings: OpenStrings,
}

impl<'t> Reader<'t, '_> {
    fn rest(&self) -> &'t str {
        &self.text[self.pos..]
    }

    fn at_end(&self) -> bool {
        self.pos >= self.text.len()
    }

    fn skip_ws(&mut self) {
        let rest = self.rest();
        self.pos += rest.len() - rest.trim_start().len();
    }

    fn eat(&mut self, literal: &str) -> bool {
        let ate = self.rest().starts_with(literal);
        if ate {
            self.pos += literal.len();
        }
        ate
    }

    fn value(&mut self) -> Read {
        self.skip_ws();
        let rest = self.rest();
        if rest.is_empty() {
            return Read::Open(Open::default());
        }
        let string = match self.flavor {
            Flavor::Json => rest.starts_with('"'),
            Flavor::Dict { quote } => {
                rest.starts_with(quote) || quote.starts_with(rest)
            }
        };
        if string {
            return match self.string() {
                Str::Done(s) => Read::Done(Value::String(s)),
                Str::Open(s) => Read::Open(Open {
                    kept: s.map(Value::String),
                    ..Open::default()
                }),
                Str::Malformed => Read::Malformed,
            };
        }
        match rest.as_bytes()[0] {
            b'{' => self.object(),
            b'[' => self.array(),
            _ => self.scalar(),
        }
    }

    fn object(&mut self) -> Read {
        self.eat("{");
        let mut map = Map::new();
        loop {
            self.skip_ws();
            if self.at_end() {
                return Read::Open(Open::container(Value::Object(map), None));
            }
            if self.eat("}") {
                return Read::Done(Value::Object(map));
            }
            let key = match self.key() {
                Some(Some(key)) => key,
                // The key, or the colon after it, is still coming.
                Some(None) => {
                    return Read::Open(Open::container(
                        Value::Object(map),
                        None,
                    ));
                }
                None => return Read::Malformed,
            };
            match self.value() {
                Read::Done(value) => {
                    map.insert(key, value);
                }
                Read::Open(mut child) => {
                    if let Some(value) = child.kept.take() {
                        map.insert(key.clone(), value);
                    }
                    return Read::Open(Open::container(
                        Value::Object(map),
                        Some((key, child)),
                    ));
                }
                Read::Malformed => return Read::Malformed,
            }
            match self.after_member('}') {
                Some(true) => continue,
                Some(false) => return Read::Done(Value::Object(map)),
                None if self.at_end() => {
                    return Read::Open(Open::container(
                        Value::Object(map),
                        None,
                    ));
                }
                None => return Read::Malformed,
            }
        }
    }

    fn array(&mut self) -> Read {
        self.eat("[");
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.at_end() {
                return Read::Open(Open::container(Value::Array(items), None));
            }
            if self.eat("]") {
                return Read::Done(Value::Array(items));
            }
            match self.value() {
                Read::Done(value) => items.push(value),
                Read::Open(mut child) => {
                    let index = items.len().to_string();
                    items.extend(child.kept.take());
                    return Read::Open(Open::container(
                        Value::Array(items),
                        Some((index, child)),
                    ));
                }
                Read::Malformed => return Read::Malformed,
            }
            match self.after_member(']') {
                Some(true) => continue,
                Some(false) => return Read::Done(Value::Array(items)),
                None if self.at_end() => {
                    return Read::Open(Open::container(
                        Value::Array(items),
                        None,
                    ));
                }
                None => return Read::Malformed,
            }
        }
    }

    /// After a member: `Some(true)` on a separator, `Some(false)` on
    /// `close`, `None` on anything else (the end of input included).
    fn after_member(&mut self, close: char) -> Option<bool> {
        self.skip_ws();
        if self.eat(",") {
            Some(true)
        } else if self.eat(close.encode_utf8(&mut [0; 4])) {
            Some(false)
        } else {
            None
        }
    }

    /// A key and its colon: `Some(None)` when either is still coming,
    /// `None` when malformed.
    fn key(&mut self) -> Option<Option<String>> {
        match self.flavor {
            Flavor::Json => {
                if !self.rest().starts_with('"') {
                    return None;
                }
                let key = match self.string_with(OpenStrings::Drop) {
                    Str::Done(key) => key,
                    Str::Open(_) => return Some(None),
                    Str::Malformed => return None,
                };
                self.skip_ws();
                if self.at_end() {
                    return Some(None);
                }
                self.eat(":").then_some(Some(key))
            }
            // Bare key up to `:` (the dict grammar's `[^:}]+`).
            Flavor::Dict { .. } => {
                let Some(colon) = self.rest().find([':', '}']) else {
                    return Some(None);
                };
                let key = self.rest()[..colon].trim();
                if self.rest().as_bytes()[colon] == b'}' || key.is_empty() {
                    return None;
                }
                let key = key.to_owned();
                self.pos += colon + 1;
                Some(Some(key))
            }
        }
    }

    fn string(&mut self) -> Str {
        self.string_with(self.strings)
    }

    fn string_with(&mut self, strings: OpenStrings) -> Str {
        match self.flavor {
            Flavor::Json => self.json_string(strings),
            Flavor::Dict { quote } => self.dict_string(quote, strings),
        }
    }

    /// A `"`-quoted JSON string. Open, it is decoded up to its last
    /// complete character — an escape still coming is not text yet.
    fn json_string(&mut self, strings: OpenStrings) -> Str {
        let start = self.pos;
        let bytes = self.text.as_bytes();
        let mut i = start + 1;
        while i < bytes.len() {
            match bytes[i] {
                b'"' => {
                    self.pos = i + 1;
                    return serde_json::from_str(&self.text[start..=i])
                        .map_or(Str::Malformed, Str::Done);
                }
                b'\\' if bytes.get(i + 1) == Some(&b'u') => {
                    if i + 6 > bytes.len() {
                        break;
                    }
                    i += 6;
                }
                b'\\' => {
                    if i + 2 > bytes.len() {
                        break;
                    }
                    i += 2;
                }
                _ => i += 1,
            }
        }
        // The input ended inside the string.
        self.pos = self.text.len();
        if strings == OpenStrings::Drop {
            return Str::Open(None);
        }
        let body = &self.text[start + 1..i.min(self.text.len())];
        Str::Open(decode_open_json(body))
    }

    /// A string between two `quote` markers, verbatim.
    fn dict_string(&mut self, quote: &str, strings: OpenStrings) -> Str {
        if !self.eat(quote) {
            // A partial opening marker: the string has not begun.
            self.pos = self.text.len();
            return Str::Open(None);
        }
        if let Some(end) = self.rest().find(quote) {
            let s = self.rest()[..end].to_owned();
            self.pos += end + quote.len();
            return Str::Done(s);
        }
        let body = self.rest();
        self.pos = self.text.len();
        Str::Open(match strings {
            OpenStrings::Drop => None,
            OpenStrings::Held => {
                Some(body[..body.len() - marker_holdback(body, quote)].into())
            }
            OpenStrings::Raw => Some(body.into()),
        })
    }

    fn scalar(&mut self) -> Read {
        let rest = self.rest();
        let literals: &[(&str, Value)] = match self.flavor {
            Flavor::Json => &[
                ("true", Value::Bool(true)),
                ("false", Value::Bool(false)),
                ("null", Value::Null),
            ],
            Flavor::Dict { .. } => &[
                ("true", Value::Bool(true)),
                ("false", Value::Bool(false)),
                ("null", Value::Null),
                ("none", Value::Null),
                ("None", Value::Null),
            ],
        };
        for (literal, value) in literals {
            if rest.starts_with(literal) {
                self.pos += literal.len();
                return Read::Done(value.clone());
            }
            if literal.starts_with(rest) {
                self.pos = self.text.len();
                return Read::Open(Open::default());
            }
        }
        let len = rest
            .find(|c: char| {
                !matches!(c, '0'..='9' | '-' | '+' | '.' | 'e' | 'E')
            })
            .unwrap_or(rest.len());
        if len == 0 {
            return Read::Malformed;
        }
        if len == rest.len() {
            // A number at the end of input could still grow.
            self.pos = self.text.len();
            return Read::Open(Open::default());
        }
        match serde_json::from_str::<Value>(&rest[..len]) {
            Ok(number @ Value::Number(_)) => {
                self.pos += len;
                Read::Done(number)
            }
            _ => Read::Malformed,
        }
    }
}

/// Decode the body of an unterminated JSON string, up to its last
/// complete character. A `\u` escape whose surrogate pair is still
/// coming is not a character yet.
fn decode_open_json(body: &str) -> Option<String> {
    let decode =
        |body: &str| serde_json::from_str::<String>(&format!("\"{body}\""));
    decode(body).ok().or_else(|| {
        let lone = body.rfind("\\u")?;
        decode(&body[..lone]).ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn json_read(text: &str, strings: OpenStrings) -> Option<Truncated> {
        read_partial(text, Flavor::Json, strings)
    }

    /// The captured rule, top level: a clip keeps the members that
    /// completed and drops the one in flight, however far it got; with
    /// none complete, `{}`.
    #[test]
    fn a_clip_keeps_only_completed_members() {
        let cases = [
            ("{", json!({})),
            (r#"{"pa"#, json!({})),
            (r#"{"path""#, json!({})),
            (r#"{"path": "#, json!({})),
            (r#"{"path": "hel"#, json!({})),
            (r#"{"path": "hello.py""#, json!({"path": "hello.py"})),
            (r#"{"path": "hello.py", "#, json!({"path": "hello.py"})),
            (
                r#"{"path": "hello.py", "contents": "import da"#,
                json!({"path": "hello.py"}),
            ),
            (r#"{"a": 1, "b": 2"#, json!({"a": 1})),
            (r#"{"a": 1, "b": 2}"#, json!({"a": 1, "b": 2})),
            (r#"{"a": tr"#, json!({})),
            (r#"{"a": true"#, json!({"a": true})),
            (
                r#"{"a": null, "b": -1.5e3,"#,
                json!({"a": null, "b": -1500.0}),
            ),
        ];
        for (text, want) in cases {
            assert_eq!(truncate_partial_object(text), Some(want), "{text:?}");
        }
        assert_eq!(truncate_partial_object(""), None);
        assert_eq!(truncate_partial_object("[1, 2"), None, "not an object");
        assert_eq!(truncate_partial_object(r#"{"a" 1"#), None, "malformed");
        assert_eq!(truncate_partial_object(r#"{"a": @"#), None, "malformed");
    }

    /// Inferred for nested containers: completed members kept at every
    /// depth, the one in progress dropped; a container in progress is
    /// kept with the members it completed.
    #[test]
    fn nested_containers_keep_completed_members_at_every_depth() {
        let cases = [
            (r#"{"a": 1, "b": {"#, json!({"a": 1, "b": {}}), 2),
            (
                r#"{"a": 1, "b": {"x": 1, "y": "pa"#,
                json!({"a": 1, "b": {"x": 1}}),
                2,
            ),
            (r#"{"a": {"x": [1, 2, "#, json!({"a": {"x": [1, 2]}}), 3),
            (r#"{"a": {"x": [1, 2"#, json!({"a": {"x": [1]}}), 3),
            (r#"{"a": ["s", "t"#, json!({"a": ["s"]}), 2),
            (r#"{"a": {"x": 1}, "b"#, json!({"a": {"x": 1}}), 1),
        ];
        for (text, want, open) in cases {
            let t = json_read(text, OpenStrings::Drop).unwrap();
            assert_eq!((t.value, t.open), (want, open), "{text:?}");
            assert_eq!(t.path.len(), open - 1, "{text:?}");
        }
        // The path names the open containers below the top.
        let t = json_read(r#"{"a": 1, "b": [{"x": {"#, OpenStrings::Drop);
        assert_eq!(t.unwrap().path, ["b", "0", "x"]);
        let t = json_read(r#"{"a": {"x": 1}, "b": "o"#, OpenStrings::Held);
        assert!(t.unwrap().path.is_empty(), "a string opens no container");
    }

    /// `Held` and `Raw` keep the string in flight — at any depth — as
    /// far as it got; a closed string is the same in every mode.
    #[test]
    fn open_strings_keep_the_string_in_flight() {
        let text = r#"{"path": "a.py", "body": {"lines": ["x", "import da"#;
        let want =
            json!({"path": "a.py", "body": {"lines": ["x", "import da"]}});
        for strings in [OpenStrings::Held, OpenStrings::Raw] {
            let t = json_read(text, strings).unwrap();
            assert_eq!(t.value, want, "{strings:?}");
            assert_eq!(t.open, 3, "{strings:?}");
        }
        // Just opened: an empty string.
        let t = json_read(r#"{"a": ""#, OpenStrings::Held).unwrap();
        assert_eq!(t.value, json!({"a": ""}));
        // A key in flight is never kept.
        let t = json_read(r#"{"a": 1, "ke"#, OpenStrings::Raw).unwrap();
        assert_eq!(t.value, json!({"a": 1}));
    }

    /// An escape still coming is not a character yet; one complete is.
    #[test]
    fn open_json_strings_decode_to_their_last_complete_character() {
        let cases = [
            (r#"{"s": "a\"#, "a"),
            (r#"{"s": "a\n"#, "a\n"),
            (r#"{"s": "a\u00"#, "a"),
            (r#"{"s": "aé"#, "a\u{e9}"),
            (r#"{"s": "a\ud83d"#, "a"),
            (r#"{"s": "a😀"#, "a\u{1f600}"),
            (r#"{"s": "café \"q\" "#, "caf\u{e9} \"q\" "),
            ("{\"s\": \"na\u{ef}ve", "na\u{ef}ve"),
        ];
        for (text, want) in cases {
            let t = json_read(text, OpenStrings::Held).unwrap();
            assert_eq!(t.value, json!({"s": want}), "{text:?}");
        }
    }

    /// Gemma 4's dict: bare keys, `<|"|>`-quoted strings. `Held` holds
    /// back a tail that could be the closing quote's start; `Raw` does
    /// not.
    #[test]
    fn dict_flavor_reads_gemma_values() {
        let dict = Flavor::Dict { quote: "<|\"|>" };
        let read = |text, strings| read_partial(text, dict, strings).unwrap();

        let t = read(
            "{city:<|\"|>Paris<|\"|>,days:3,note:<|\"|>a<|",
            OpenStrings::Drop,
        );
        assert_eq!((t.value, t.open), (json!({"city": "Paris", "days": 3}), 1));
        let t = read(
            "{city:<|\"|>Paris<|\"|>,days:3,note:<|\"|>a<|",
            OpenStrings::Held,
        );
        assert_eq!(t.value, json!({"city": "Paris", "days": 3, "note": "a"}));
        let t = read(
            "{city:<|\"|>Paris<|\"|>,days:3,note:<|\"|>a<|",
            OpenStrings::Raw,
        );
        assert_eq!(t.value, json!({"city": "Paris", "days": 3, "note": "a<|"}));

        let t = read("{a:none,b:{x:[1,<|\"|>y<|\"|>],z:tr", OpenStrings::Drop);
        assert_eq!(
            (t.value, t.open),
            (json!({"a": null, "b": {"x": [1, "y"]}}), 2)
        );
        let t = read("{a:3", OpenStrings::Drop);
        assert_eq!(t.value, json!({}), "a number at the end could grow");
        let t = read("{a:<|", OpenStrings::Raw);
        assert_eq!(t.value, json!({}), "a partial quote begins no string");
        assert!(read_partial("{:1}", dict, OpenStrings::Drop).is_none());
    }

    #[test]
    fn unclosed_json_drops_the_open_closers() {
        let t = json_read(r#"{"a": 1, "b": {"x": [1, "#, OpenStrings::Drop)
            .unwrap();
        assert_eq!(unclosed_json(&t.value, t.open), r#"{"a":1,"b":{"x":[1"#);
        let t = json_read(
            r#"{"path": "story.txt", "contents": "Once"#,
            OpenStrings::Drop,
        )
        .unwrap();
        assert_eq!(unclosed_json(&t.value, t.open), r#"{"path":"story.txt""#);
        assert_eq!(unclosed_json(&json!({}), 1), "{");
        assert_eq!(unclosed_json(&json!({"a": {}}), 0), r#"{"a":{}}"#);
    }

    #[test]
    fn marker_holdback_is_the_longest_marker_prefix() {
        assert_eq!(marker_holdback("abc", "\n</parameter>\n"), 0);
        assert_eq!(marker_holdback("abc\n", "\n</parameter>\n"), 1);
        assert_eq!(marker_holdback("abc\n</par", "\n</parameter>\n"), 6);
        // A whole marker is not a prefix of it: that one closed.
        assert_eq!(marker_holdback("<|\"|>", "<|\"|>"), 0);
        assert_eq!(marker_holdback("x<|", "<|\"|>"), 2);
    }

    /// The captured stop rule: the string is cut before the match and
    /// what followed it is gone; earlier members stand, in order.
    #[test]
    fn cut_value_cuts_the_first_match_in_generation_order() {
        let input = json!({
            "path": "hello.py",
            "contents": "import datetime\nprint(datetime.now())",
            "mode": "w",
        });
        let (cut, i) = cut_value(&input, &["zzz", "print("]).unwrap();
        assert_eq!(i, 1);
        assert_eq!(
            cut,
            json!({"path": "hello.py", "contents": "import datetime\n"})
        );
        assert_eq!(
            serde_json::to_string(&cut).unwrap(),
            r#"{"path":"hello.py","contents":"import datetime\n"}"#,
            "members keep their generation order",
        );

        // Nested and in arrays; keys never match.
        let input = json!({
            "a": 1,
            "b": {"x": ["no", "yes STOP", "later"]},
            "c": "STOP",
        });
        let (cut, _) = cut_value(&input, &["STOP"]).unwrap();
        assert_eq!(cut, json!({"a": 1, "b": {"x": ["no", "yes "]}}));
        assert_eq!(cut_value(&json!({"STOP": 1}), &["STOP"]), None);

        // A match at the start of a value leaves it empty, not gone.
        let (cut, _) =
            cut_value(&json!({"a": "x", "b": "STOP"}), &["STOP"]).unwrap();
        assert_eq!(cut, json!({"a": "x", "b": ""}));
    }
}
