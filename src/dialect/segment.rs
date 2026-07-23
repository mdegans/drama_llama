//! Marker segmentation + marker-aware diffing for template analysis.
//!
//! Rust port of llama.cpp's `chat-auto-parser-helpers.{h,cpp}`
//! (b9754 / 52b3df00, MIT). The analyzer never parses Jinja — it
//! renders sentinel payloads and diffs the outputs, so everything
//! here operates on rendered text. A *marker* is a `<...>` or
//! `[...]` span: the shape chat templates use for control tokens
//! (`<|im_start|>`, `<tool_call>`, `[TOOL_CALLS]`, …).

/// One fragment of rendered text: either a marker (`<...>` / `[...]`)
/// or plain text between markers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Segment {
    Marker(String),
    Text(String),
}

impl Segment {
    pub(crate) fn value(&self) -> &str {
        match self {
            Segment::Marker(s) | Segment::Text(s) => s,
        }
    }

    pub(crate) fn is_marker(&self) -> bool {
        matches!(self, Segment::Marker(_))
    }
}

/// Split `text` into marker and text segments.
///
/// A marker opens at `<` or `[` and closes at the matching `>` / `]`.
/// An unterminated opener falls through as text. Nested openers do
/// not nest — the first closer wins (`<a<b>` is one marker).
///
/// Known limitation inherited from upstream: a JSON array inside
/// tags segments as markers (`[{ "f": 1 }]` → `[{ "f": 1 }]` is
/// bracketed…) — hasn't bitten in practice; noted in upstream too.
pub(crate) fn segmentize_markers(text: &str) -> Vec<Segment> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut in_marker = false;
    let mut opener = 0u8;
    let mut last_border = 0usize;

    for (i, &b) in bytes.iter().enumerate() {
        if !in_marker && (b == b'<' || b == b'[') {
            if last_border < i {
                out.push(Segment::Text(text[last_border..i].to_string()));
            }
            last_border = i;
            in_marker = true;
            opener = b;
        } else if in_marker
            && ((opener == b'<' && b == b'>') || (opener == b'[' && b == b']'))
        {
            out.push(Segment::Marker(text[last_border..=i].to_string()));
            last_border = i + 1;
            in_marker = false;
        }
    }
    if last_border < text.len() {
        out.push(Segment::Text(text[last_border..].to_string()));
    }
    out
}

/// Drop whitespace-only segments.
pub(crate) fn prune_whitespace_segments(segments: &[Segment]) -> Vec<Segment> {
    segments
        .iter()
        .filter(|s| !s.value().trim().is_empty())
        .cloned()
        .collect()
}

/// Marker-aware diff of two renders: the longest common prefix and
/// suffix (aligned to marker boundaries where possible) and the two
/// differing middles.
///
/// ```text
/// left:  <html><body><div></div></body></html>
/// right: <html><body><p>Something</p></body></html>
/// → prefix "<html><body>", suffix "</body></html>",
///   left "<div></div>", right "<p>Something</p>"
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DiffSplit {
    pub prefix: String,
    pub suffix: String,
    pub left: String,
    pub right: String,
}

/// Port of upstream `calculate_diff_split`: consume equal segments
/// from both ends simultaneously, then refine the raw-text remainder
/// with plain common prefix/suffix — but only across text segments,
/// so marker boundaries stay intact.
pub(crate) fn calculate_diff_split(left: &str, right: &str) -> DiffSplit {
    let left_seg = segmentize_markers(left);
    let right_seg = segmentize_markers(right);

    let mut result = DiffSplit::default();

    if left_seg.is_empty() {
        result.right = right.to_string();
        return result;
    }
    if right_seg.is_empty() {
        result.left = left.to_string();
        return result;
    }

    // Two-ended simultaneous consumption over segment ranges
    // [ls, le] and [rs, re] (inclusive).
    let mut ls = 0usize;
    let mut le = left_seg.len() - 1;
    let mut rs = 0usize;
    let mut re = right_seg.len() - 1;
    let mut left_fully_consumed = false;
    let mut right_fully_consumed = false;

    while ls != le && rs != re {
        let mut advanced = false;
        if left_seg[ls] == right_seg[rs] {
            result.prefix.push_str(left_seg[ls].value());
            ls += 1;
            rs += 1;
            advanced = true;
        }
        if left_seg[le] == right_seg[re] {
            result.suffix =
                format!("{}{}", left_seg[le].value(), result.suffix);
            if ls != le {
                le -= 1;
            } else {
                left_fully_consumed = true;
            }
            if rs != re {
                re -= 1;
            } else {
                right_fully_consumed = true;
            }
            advanced = true;
        }
        if !advanced {
            break;
        }
    }

    if ls == le && rs != re {
        if left_seg[ls] == right_seg[re] {
            result.suffix =
                format!("{}{}", right_seg[re].value(), result.suffix);
            re -= 1;
            left_fully_consumed = true;
        } else if left_seg[ls] == right_seg[rs] {
            result.prefix.push_str(right_seg[rs].value());
            rs += 1;
            left_fully_consumed = true;
        }
    } else if rs == re && ls != le {
        if left_seg[le] == right_seg[rs] {
            result.suffix =
                format!("{}{}", left_seg[le].value(), result.suffix);
            le -= 1;
            right_fully_consumed = true;
        } else if left_seg[ls] == right_seg[rs] {
            result.prefix.push_str(left_seg[ls].value());
            ls += 1;
            right_fully_consumed = true;
        }
    } else if ls == le
        && rs == re
        && left_seg[ls] == right_seg[rs]
        && left_seg[ls].is_marker()
    {
        result.prefix.push_str(right_seg[rs].value());
        left_fully_consumed = true;
        right_fully_consumed = true;
    }

    let join = |seg: &[Segment], start: usize, end: usize, consumed: bool| {
        if consumed {
            seg[start..end]
                .iter()
                .map(Segment::value)
                .collect::<String>()
        } else {
            seg[start..=end]
                .iter()
                .map(Segment::value)
                .collect::<String>()
        }
    };
    let remainder_left = join(&left_seg, ls, le, left_fully_consumed);
    let remainder_right = join(&right_seg, rs, re, right_fully_consumed);

    let can_have_text_suffix =
        !left_seg[le].is_marker() && !right_seg[re].is_marker();
    let can_have_text_prefix =
        !left_seg[ls].is_marker() && !right_seg[rs].is_marker();

    let suffix_len = if can_have_text_suffix {
        common_suffix_len(&remainder_left, &remainder_right)
    } else {
        0
    };
    let prefix_len = if can_have_text_prefix {
        common_prefix_len(
            &remainder_left[..remainder_left.len() - suffix_len],
            &remainder_right[..remainder_right.len() - suffix_len],
        )
    } else {
        0
    };

    result.prefix.push_str(&remainder_left[..prefix_len]);
    result.suffix = format!(
        "{}{}",
        &remainder_left[remainder_left.len() - suffix_len..],
        result.suffix
    );
    result.left = remainder_left[prefix_len..remainder_left.len() - suffix_len]
        .to_string();
    result.right = remainder_right
        [prefix_len..remainder_right.len() - suffix_len]
        .to_string();

    if result.left.is_empty() && result.right.is_empty() {
        // Degenerate: no diff. Represent as prefix = everything.
        result.prefix = left.to_string();
        result.suffix.clear();
    }

    // When left is a strict prefix of right, the two-ended consumption
    // can rotate shared trailing segments into `suffix`. Enforce the
    // direct interpretation instead (upstream fix, same rationale).
    if result.left.is_empty()
        && !result.right.is_empty()
        && left.len() <= right.len()
        && right.as_bytes()[..left.len()] == *left.as_bytes()
    {
        result.prefix = left.to_string();
        result.suffix.clear();
        result.right = right[left.len()..].to_string();
    }

    result
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    // Byte-wise like upstream, then rounded down to a char boundary —
    // two renders can diverge *inside* a multi-byte character that
    // shares lead bytes (é `C3 A9` vs è `C3 A8`), and callers slice
    // `&str`s at this length. The boundary must hold in both strings:
    // at the divergence point the bytes differ, so boundary-ness can
    // differ too.
    let mut n = a
        .as_bytes()
        .iter()
        .zip(b.as_bytes())
        .take_while(|(x, y)| x == y)
        .count();
    while n > 0 && !(a.is_char_boundary(n) && b.is_char_boundary(n)) {
        n -= 1;
    }
    n
}

fn common_suffix_len(a: &str, b: &str) -> usize {
    // Char-boundary rounding as in `common_prefix_len`, measured from
    // the tails.
    let mut n = a
        .as_bytes()
        .iter()
        .rev()
        .zip(b.as_bytes().iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    while n > 0
        && !(a.is_char_boundary(a.len() - n) && b.is_char_boundary(b.len() - n))
    {
        n -= 1;
    }
    n
}

/// The prefix of `full` up to the first occurrence of the common
/// prefix of `left` and `right`. Empty when there is no common
/// prefix or it doesn't occur in `full`.
///
/// `until_common_prefix("really want a FUNCTION call", "FUNCTION a",
/// "FUNCTION b")` → `"really want a "`.
pub(crate) fn until_common_prefix(
    full: &str,
    left: &str,
    right: &str,
) -> String {
    let n = common_prefix_len(left, right);
    if n == 0 {
        return String::new();
    }
    let common = &left[..n];
    match full.find(common) {
        Some(pos) => full[..pos].to_string(),
        None => String::new(),
    }
}

/// The suffix of `full` after the last occurrence of the common
/// suffix of `left` and `right`. Mirror of [`until_common_prefix`].
pub(crate) fn after_common_suffix(
    full: &str,
    left: &str,
    right: &str,
) -> String {
    let n = common_suffix_len(left, right);
    if n == 0 {
        return String::new();
    }
    let common = &left[left.len() - n..];
    match full.rfind(common) {
        Some(pos) => full[pos + n..].to_string(),
        None => String::new(),
    }
}

/// The marker immediately preceding byte offset `pos` in `text`, with
/// only whitespace allowed between the marker's end and `pos`.
/// Returns the span from the marker's first byte through `pos`
/// (marker + intervening whitespace), so
/// `marker_before(...) + needle` reconstructs the original bytes.
pub(crate) fn marker_before(text: &str, pos: usize) -> Option<String> {
    let head = &text[..pos];
    let segs = segmentize_markers(head);
    // Walk back over trailing whitespace-only text segments to the
    // nearest marker.
    let mut end = segs.len();
    while end > 0 {
        match &segs[end - 1] {
            Segment::Text(t) if t.trim().is_empty() => end -= 1,
            _ => break,
        }
    }
    if end == 0 || !segs[end - 1].is_marker() {
        return None;
    }
    // Byte offset of that marker's start = length of everything before.
    let start: usize = segs[..end - 1].iter().map(|s| s.value().len()).sum();
    Some(head[start..].to_string())
}

/// The marker immediately following byte offset `pos` in `text`, with
/// only whitespace allowed between `pos` and the marker's start.
/// Returns the span from `pos` through the marker's last byte
/// (intervening whitespace + marker), so
/// `needle + marker_after(...)` reconstructs the original bytes.
pub(crate) fn marker_after(text: &str, pos: usize) -> Option<String> {
    let tail = &text[pos..];
    let segs = segmentize_markers(tail);
    let mut consumed = 0usize;
    for seg in &segs {
        match seg {
            Segment::Text(t) if t.trim().is_empty() => {
                consumed += t.len();
            }
            Segment::Marker(m) => {
                return Some(tail[..consumed + m.len()].to_string());
            }
            _ => return None,
        }
    }
    None
}

/// Last marker segment anywhere in `text` (just the marker).
pub(crate) fn find_last_marker(text: &str) -> Option<String> {
    segmentize_markers(text)
        .iter()
        .rev()
        .find(|s| s.is_marker())
        .map(|s| s.value().to_string())
}

/// First marker segment anywhere in `text` (just the marker).
pub(crate) fn find_first_marker(text: &str) -> Option<String> {
    segmentize_markers(text)
        .iter()
        .find(|s| s.is_marker())
        .map(|s| s.value().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmentize_upstream_examples() {
        let s = segmentize_markers(
            "<|tool_call|>[args]{ are here }[/args]<|tool_call_end|>",
        );
        let expected = vec![
            Segment::Marker("<|tool_call|>".into()),
            Segment::Marker("[args]".into()),
            Segment::Text("{ are here }".into()),
            Segment::Marker("[/args]".into()),
            Segment::Marker("<|tool_call_end|>".into()),
        ];
        assert_eq!(s, expected);

        let s = segmentize_markers("plain text only");
        assert_eq!(s, vec![Segment::Text("plain text only".into())]);

        // Unterminated opener falls through as text.
        let s = segmentize_markers("a < b");
        assert_eq!(
            s,
            vec![Segment::Text("a ".into()), Segment::Text("< b".into())]
        );
    }

    #[test]
    fn prune_whitespace() {
        let s = segmentize_markers("<a>\n<b>\n   \n</b>");
        let p = prune_whitespace_segments(&s);
        assert!(p.iter().all(Segment::is_marker));
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn diff_split_upstream_examples() {
        let d = calculate_diff_split(
            "<html><body><div></div></body></html>",
            "<html><body><p>Something</p></body></html>",
        );
        assert_eq!(d.prefix, "<html><body>");
        assert_eq!(d.suffix, "</body></html>");
        assert_eq!(d.left, "<div></div>");
        assert_eq!(d.right, "<p>Something</p>");

        let d = calculate_diff_split(
            "<html><body>Something</body></html>",
            "<html><body></body></html>",
        );
        assert_eq!(d.prefix, "<html><body>");
        assert_eq!(d.suffix, "</body></html>");
        assert_eq!(d.left, "Something");
        assert_eq!(d.right, "");
    }

    #[test]
    fn diff_split_left_prefix_of_right() {
        // The rotation guard: shared trailing "\n" must not migrate
        // into suffix when left is a strict prefix of right.
        let d = calculate_diff_split(
            "<|user|>hi\n",
            "<|user|>hi\n<|assistant|>yo\n",
        );
        assert_eq!(d.prefix, "<|user|>hi\n");
        assert_eq!(d.suffix, "");
        assert_eq!(d.left, "");
        assert_eq!(d.right, "<|assistant|>yo\n");
    }

    #[test]
    fn diff_split_multibyte_divergence() {
        // é (C3 A9) and è (C3 A8) share a lead byte; the byte-wise
        // common prefix ends mid-char and must be rounded back to the
        // boundary rather than panicking (any non-English template).
        let d = calculate_diff_split("aé", "aè");
        assert_eq!(d.prefix, "a");
        assert_eq!(d.suffix, "");
        assert_eq!(d.left, "é");
        assert_eq!(d.right, "è");

        // Same class from the suffix side: shared continuation byte.
        let d = calculate_diff_split("éa", "èa");
        assert_eq!(d.prefix, "");
        assert_eq!(d.suffix, "a");
        assert_eq!(d.left, "é");
        assert_eq!(d.right, "è");

        // And through the marker-segmented path.
        let d =
            calculate_diff_split("<|user|>café<|end|>", "<|user|>cafè<|end|>");
        assert_eq!(d.prefix, "<|user|>caf");
        assert_eq!(d.left, "é");
        assert_eq!(d.right, "è");
    }

    #[test]
    fn common_helpers_multibyte_divergence() {
        // `until_common_prefix` slices `left[..n]` — n must land on a
        // char boundary when the divergence is mid-char.
        assert_eq!(until_common_prefix("x défg", "dé1", "dè2"), "x ");
        assert_eq!(after_common_suffix("x défg y", "1éfg", "2èfg"), " y");
    }

    #[test]
    fn diff_split_identical() {
        let d = calculate_diff_split("same", "same");
        assert_eq!(d.prefix, "same");
        assert!(d.left.is_empty() && d.right.is_empty());
    }

    #[test]
    fn common_prefix_suffix_helpers() {
        assert_eq!(
            until_common_prefix(
                "really want a FUNCTION call",
                "FUNCTION alpha",
                "FUNCTION beta"
            ),
            "really want a "
        );
        assert_eq!(until_common_prefix("some text", "1234", "abcd"), "");
        assert_eq!(
            after_common_suffix(
                "really want a FUNCTION call",
                "first FUNCTION",
                "second FUNCTION"
            ),
            " call"
        );
    }

    #[test]
    fn marker_adjacency_helpers() {
        let text = "<tool_call>\nNEEDLE\n</tool_call>";
        let pos = text.find("NEEDLE").unwrap();
        assert_eq!(marker_before(text, pos).as_deref(), Some("<tool_call>\n"));
        let end = pos + "NEEDLE".len();
        assert_eq!(marker_after(text, end).as_deref(), Some("\n</tool_call>"));

        // Non-whitespace between needle and marker: no adjacency.
        let text = "<a>xx NEEDLE yy<b>";
        let pos = text.find("NEEDLE").unwrap();
        assert_eq!(marker_before(text, pos), None);
        assert_eq!(marker_after(text, pos + "NEEDLE".len()), None);
    }

    #[test]
    fn first_last_marker() {
        let text = "junk <a> mid [b] tail";
        assert_eq!(find_first_marker(text).as_deref(), Some("<a>"));
        assert_eq!(find_last_marker(text).as_deref(), Some("[b]"));
        assert_eq!(find_first_marker("no markers"), None);
    }
}
