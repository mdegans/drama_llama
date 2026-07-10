# Qwen3.6 renders tool calls in an XML-ish shape (not Qwen3's JSON)

**Date:** 2026-07-10 (session: alpha.7 migration + snapshot landing)
**Model:** Qwen3.6-35B-A3B (UD-IQ4_XS), `models/model.gguf` on the
Linux/CUDA box.

## Observation

Qwen3.6's embedded `tokenizer.chat_template` renders assistant
`ToolUse` blocks as:

```
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

Qwen3 (and Hermes/cogito family) rendered JSON inside the same
`<tool_call>` tags: `{"name": "get_weather", "arguments": {...}}`.

Caught by `chat_template_renders_assistant_tool_call_against_real_model`,
which asserted the JSON keys; the assertion is now shape-agnostic
(name/key/value must survive rendering, envelope unasserted).

## Implication for tool-choice grammar (open, not urgent)

`ToolChoiceOptions::default()` grammar-forces the JSON envelope inside
`<tool_call>` wrap tags. On Qwen3.6 that is *off-distribution*: the
model was presumably trained to emit the XML-ish shape above. Grammar
keeps forced calls well-formed regardless, so nothing breaks — but:

- forced-JSON may cost call quality (model fighting its training);
- `ToolChoice::Auto` (no grammar) will emit the XML shape, which the
  session-side tool-call *parser* (`session/parse.rs`) presumably does
  not recognize — check before relying on Auto with Qwen3.6.

Mike (2026-07-10): this may be Qwen3.5 behavior too, not new in 3.6.
Downstream (at least one consumer) asks for JSON tool calls in the
system prompt *and* grammar-forces them; grammar is not on by default
there. The model is "flexible enough" to comply when forced — the open
question is whether forcing off-distribution costs measurable call
quality, which nobody has quantified. A/B-ing forced-JSON vs
forced-XML call accuracy on the same tool set would answer it (the
strawberry/count_letters harness is nearly sufficient).

## Follow-up when the model-support session happens

1. Add an XML-ish call shape to `ToolChoiceOptions` (grammar emitting
   `<function=NAME><parameter=KEY>VALUE</parameter>...</function>`,
   schema-pinnable per tool like the JSON path).
2. Teach `parse.rs` to recognize the shape on re-ingest.
3. Template-family detection from the GGUF template source could pick
   the default (JSON vs XML) automatically.
