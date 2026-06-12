# Future work: parse Qwen3.6 XML-parameter tool calls (unforced path)

Found 2026-06-12 during v0.8.0 validation (misanthropic-examples-vs-
blallama harness). Candidate GitHub issue — not yet filed.

When tools are advertised but `tool_choice` is unset (no grammar
forcing), Qwen3.6-35B-A3B emits its template-native call format:

```
<tool_call>
<function=count_letters>
<parameter=letter>
r
</parameter>
<parameter=string>
strawberry
</parameter>
</function>
</tool_call>
```

`Session::parse` only recognizes the JSON dialect
(`<tool_call>{"name": …, "arguments": {…}}</tool_call>`), so the call
ships as a plain `Text` block with `stop_reason: end_turn` instead of
a `ToolUse` block + `tool_use` stop. SDK clients driving tool loops
via `response.tool_use()` get `None`.

Forced flows are unaffected — the tool-choice grammar emits the JSON
shape (which the model follows and parse handles); that's why every
grammar-path test passes and the in-repo strawberry example works.

Fix direction: teach `session/parse.rs` the XML-parameter dialect,
keyed off the template family — or normalize the dialect at the
grammar/template layer. Repro: POST /v1/messages to blallama with a
tool advertised, no tool_choice, Qwen3.6 loaded.
