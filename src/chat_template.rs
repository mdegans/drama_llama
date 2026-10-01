//! Jinja-based chat templating.
//!
//! Most modern GGUF models embed their chat template under the
//! `tokenizer.chat_template` metadata key. [`ChatTemplate`] compiles that
//! template with [`minijinja`] and renders a [`Prompt`] into the exact byte
//! sequence the model was trained on — Llama 3.1, Qwen, Mistral,
//! Gemma, etc., without per-model Rust code.
//!
//! # Scope
//!
//! drama_llama renders [`Block::Text`], [`Block::Thought`],
//! [`Block::ToolUse`], and [`Block::ToolResult`] blocks. Tool calls emit
//! as `{role, content, tool_calls: [{id, function: {name, arguments}}]}`
//! messages; tool results emit as standalone `{role: "tool", ...}`
//! messages between user turns. Images and redacted-thought blocks are
//! skipped until the surrounding infrastructure needs them.
//!
//! Tool definitions come from [`Prompt::tools`] and surface in the
//! template as the `tools` variable. Llama 3.1 / Mistral / Qwen all
//! JSON-serialize tools via the `tojson` filter, which is enabled here
//! via `minijinja`'s `json` feature.
//!
//! # Extras
//!
//! [`RenderOptions`] lets callers push additional template variables
//! (e.g. `tools_in_user_message`, `builtin_tools`, `custom_tools`). It
//! also carries an optional `date_string`; if omitted, drama_llama
//! defaults to today's UTC date in HF's `%d %b %Y` format so templates
//! that unconditionally concatenate `"Today Date: " + date_string` do
//! not blow up.
//!
//! # Dialect compatibility
//!
//! HuggingFace chat templates are written in Jinja2 but use a narrow
//! subset. [`minijinja`] covers that subset. The templater exposes a few
//! functions that HF templates commonly call:
//!
//! * `raise_exception(msg)` — surfaces as a render error
//! * `strftime_now(fmt)` — current UTC time, formatted via chrono-like
//!   `%Y-%m-%d` etc. (minimal subset)
//!
//! [`Prompt`]: crate::Prompt
//! [`Block::Text`]: crate::Block
//! [`Block::Thought`]: crate::Block

use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

use minijinja::{
    value::{Kwargs, Value as JinjaValue},
    Environment, Error as JinjaError, UndefinedBehavior,
};
use serde::Serialize;

use misanthropic::prompt::{
    message::{CacheControl, CacheTtl},
    Effort, OutputConfig, Thinking,
};

use crate::{
    backend::Model, prompt::Tool, Block, Content, Prompt, Role, Token,
};

/// Render a [`Tool`] as the OpenAI wire envelope cogito / Qwen /
/// Hermes-family chat templates expect.
///
/// These templates do `{{ tool | tojson }}` and feed the resulting
/// JSON straight to the model. The training data — produced by
/// ollama's Go runtime — always takes the shape
/// `{"type": "function", "function": {"name": ..., "description":
/// ..., "parameters": ...}}`. misanthropic's [`tool::CustomMethodDef`]
/// serializes in Anthropic shape (`input_schema` rather than
/// `parameters`, no wire envelope), so we adapt through this
/// function before handing off to minijinja.
///
/// Key order in this `json!` literal is the wire order: with
/// `preserve_order` on (serde_json + minijinja), maps render in
/// insertion order, so the envelope goes out `type, function` /
/// `name, description, parameters` — the shape ollama's Go runtime
/// produced in the training data — and `parameters` keeps the
/// schema's field declaration order (#60).
///
/// [`tool::CustomMethodDef`]: misanthropic::tool::CustomMethodDef
fn tool_wire_value(tool: &Tool) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name.as_ref(),
            "description": tool.description.as_ref(),
            "parameters": &tool.schema,
        }
    })
}

/// A compiled chat template tied to a specific model's tokens.
///
/// Load via [`ChatTemplate::from_model`] (reads the template string plus
/// BOS/EOS tokens from GGUF metadata) or [`ChatTemplate::from_source`] for
/// manual control.
#[derive(Clone)]
pub struct ChatTemplate {
    /// Shared `Environment` so clones are cheap. Environments are
    /// thread-safe but clone-by-Arc so we don't reparse on each clone.
    env: Arc<Environment<'static>>,
    /// Sibling of [`Self::env`] with `raise_exception` rebound to a
    /// no-op (returns the empty string) so partial renders for
    /// cache-breakpoint hashing don't fail on templates that gate the
    /// full render on invariants the truncated prompt doesn't satisfy.
    ///
    /// The canonical case is Qwen3's `No user query found in messages.`
    /// raise, which fires whenever the messages list contains no
    /// non-tool-response user-role message — exactly the state
    /// [`render_partial`] produces for [`PromptBreakpoint::AfterTools`] and
    /// [`PromptBreakpoint::AfterSystem`] (both truncate `messages` to empty).
    /// Without permissive rendering those partials get dropped silently
    /// from `partial_texts`, taking the front-of-prompt cache anchor
    /// with them.
    ///
    /// Used exclusively by [`render_partial`]. Full renders go through
    /// [`Self::env`] and continue to surface raises as errors.
    env_permissive: Arc<Environment<'static>>,
    /// Name under which `env` registered the template. Always "chat".
    template_name: &'static str,
    /// BOS piece (e.g. `<|begin_of_text|>` for Llama 3.1). Passed to the
    /// template as `bos_token`.
    bos_token: String,
    /// EOS piece (e.g. `<|end_of_text|>`). Passed as `eos_token`.
    eos_token: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Load the chat template from any [`Model`].
    ///
    /// Reads the template via [`Model::chat_template_source`]
    /// (`tokenizer.chat_template` GGUF metadata for llama.cpp; the
    /// bundled `chat_template.jinja` for moeflux) and renders BOS/EOS
    /// pieces from the model's token ids.
    pub fn from_model<M: Model>(model: &M) -> Result<Self, ChatTemplateError> {
        let source = model
            .chat_template_source()
            .ok_or(ChatTemplateError::NoTemplate)?;
        let bos = model.token_to_piece(model.bos());
        let eos = model.token_to_piece(model.eos());
        Self::from_source(source, bos, eos)
    }

    /// Compile a chat template from a raw Jinja source string plus
    /// BOS/EOS piece strings.
    pub fn from_source(
        source: String,
        bos_token: String,
        eos_token: String,
    ) -> Result<Self, ChatTemplateError> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(
            minijinja_contrib::pycompat::unknown_method_callback,
        );
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        register_template_filters(&mut env);
        env.add_template_owned("chat", source.clone())
            .map_err(ChatTemplateError::from_jinja)?;

        let mut env_permissive = Environment::new();
        env_permissive.set_unknown_method_callback(
            minijinja_contrib::pycompat::unknown_method_callback,
        );
        // Chainable: `messages[0].role` on `messages = []` returns
        // undefined instead of erroring. Required because suppressing
        // `raise_exception` alone leaves Qwen3's tools branch to access
        // `messages[0]` unconditionally after the suppressed raise — the
        // template assumes the raise halts rendering, which the strict
        // env honors but the permissive env can't without dropping
        // breakpoint coverage we explicitly want to keep.
        env_permissive.set_undefined_behavior(UndefinedBehavior::Chainable);
        env_permissive.add_function("raise_exception", raise_exception_noop);
        env_permissive.add_function("strftime_now", strftime_now);
        // Must match the strict env exactly: a partial render that
        // escapes differently from the full render is not a prefix of
        // it, and `Session` silently drops such partials — costing the
        // very breakpoints this is meant to preserve.
        register_template_filters(&mut env_permissive);
        env_permissive
            .add_template_owned("chat", source)
            .map_err(ChatTemplateError::from_jinja)?;

        Ok(Self {
            env: Arc::new(env),
            env_permissive: Arc::new(env_permissive),
            template_name: "chat",
            bos_token,
            eos_token,
        })
    }

    /// BOS piece the template will prepend if its logic calls for it.
    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    /// EOS piece.
    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }

    /// Render a [`Prompt`] with default options.
    ///
    /// `add_generation_prompt` controls whether the template appends an
    /// empty assistant header so the model generates the next turn. Set
    /// this to `true` for live chat, `false` when tokenizing a stored
    /// transcript. Shorthand for
    /// [`ChatTemplate::render_with`] with defaults.
    pub fn render(
        &self,
        prompt: &Prompt,
        add_generation_prompt: bool,
    ) -> Result<String, ChatTemplateError> {
        self.render_with(
            prompt,
            &RenderOptions {
                add_generation_prompt,
                ..RenderOptions::default()
            },
        )
    }

    /// Render a [`Prompt`] with explicit [`RenderOptions`].
    ///
    /// Use this when the template needs template-specific variables
    /// beyond `messages` / `tools` / `bos_token` — for example Llama 3.1's
    /// `date_string`, `tools_in_user_message`, or `builtin_tools`.
    pub fn render_with(
        &self,
        prompt: &Prompt,
        opts: &RenderOptions,
    ) -> Result<String, ChatTemplateError> {
        self.render_with_env(&self.env, prompt, opts)
            .map(|(text, _)| text)
    }

    /// [`Self::render_with`], plus how many times each reserved piece
    /// was neutralized in the prompt's content (empty without
    /// [`RenderOptions::literals`]).
    pub(crate) fn render_counted(
        &self,
        prompt: &Prompt,
        opts: &RenderOptions,
    ) -> Result<(String, LiteralCounts), ChatTemplateError> {
        self.render_with_env(&self.env, prompt, opts)
    }

    /// Shared render path. `env` selects strict vs. permissive raise
    /// handling — full renders pass [`Self::env`]; partial renders for
    /// cache-breakpoint hashing pass [`Self::env_permissive`].
    fn render_with_env(
        &self,
        env: &Environment<'static>,
        prompt: &Prompt,
        opts: &RenderOptions,
    ) -> Result<(String, LiteralCounts), ChatTemplateError> {
        // Images require a media sentinel to render into — anything
        // else is the silent drop this check exists to kill.
        if opts.media_sentinel.is_none() && prompt_has_images(prompt) {
            return Err(ChatTemplateError::MediaUnsupported);
        }
        // An open thought at the tail is *withheld* from the template
        // and appended to the finished render instead — see
        // `open_thought_tail`. Its bytes must never pass through Jinja.
        let open_tail = open_thought_tail(prompt);
        let reasoning_start = match (open_tail, &opts.reasoning_start) {
            (Some(_), None) => {
                return Err(ChatTemplateError::OpenThoughtUnsupported)
            }
            (_, start) => start.as_deref().unwrap_or_default(),
        };
        let surfaces = Surfaces::new(opts);
        let messages = build_messages(
            prompt,
            opts.thought_reingest,
            &surfaces,
            open_tail.is_some(),
        )?;
        // Only custom (client-executed) tool defs render into the
        // template; server tools execute on Anthropic's side and their
        // schemas aren't even visible to us.
        let custom_tools: Vec<&crate::Tool> = prompt
            .tools
            .iter()
            .flatten()
            .filter_map(|def| def.as_method())
            .collect();
        let tools_value = if custom_tools.is_empty() {
            JinjaValue::from(()) // renders as None / null
        } else {
            for (i, tool) in custom_tools.iter().enumerate() {
                surfaces.identifier(&tool.name, is_tool_name, || {
                    format!("tool definition {i}: name")
                })?;
            }
            let wire = serde_json::Value::Array(
                custom_tools.iter().map(|t| tool_wire_value(t)).collect(),
            );
            // Descriptions and schemas are content — a third-party
            // tool's description is as untrusted as its results. Not
            // counted: `Session`'s scan walks messages, not tools.
            surfaces.value(&wire, false)
        };
        // Default `date_string` to today in HF's "%d %b %Y" format when
        // the caller didn't supply one. The template unconditionally
        // concatenates `"Today Date: " + date_string + ...`, so passing
        // `none` would blow up with a string-plus-none type error.
        let date_string = opts.date_string.clone().unwrap_or_else(|| {
            format_strftime_subset("%d %b %Y", current_unix_secs())
        });
        // Derive `enable_thinking` from `prompt.thinking` so templates
        // that gate their `<think>` block on this variable (Qwen3 family,
        // among others) honour Anthropic's semantics: `thinking: None`
        // means thinking disabled, `Some(_)` means enabled. Caller-set
        // `extras.with_extra("enable_thinking", _)` always wins, so we
        // only add the derived value when the caller hasn't.
        let has_extra = |key: &str| opts.extras.iter().any(|(k, _)| k == key);
        // An open trailing thought IS a generation prompt: the model
        // resumes *inside* the reasoning block, so the render must end
        // at the assistant header (plus the withheld body) and never
        // close the turn. This overrides the caller's setting, which
        // is what keeps the partial-render and canonicalization paths
        // — both of which pass `false` — consistent with the full one.
        let add_generation_prompt =
            opts.add_generation_prompt || open_tail.is_some();
        // Values derived from the prompt, each added only when the
        // caller didn't set it: context merges are left-wins, so a
        // derived key would otherwise shadow the caller's extra.
        // `reasoning_effort` follows `output_config.effort` (see
        // `derive_reasoning_effort`).
        let mut derived: BTreeMap<&str, JinjaValue> = BTreeMap::new();
        if !has_extra("enable_thinking") {
            derived.insert(
                "enable_thinking",
                JinjaValue::from(thinking_enabled(prompt)),
            );
        }
        if !has_extra("reasoning_effort") {
            if let Some(effort) = derive_reasoning_effort(prompt, &opts.efforts)
            {
                derived.insert("reasoning_effort", JinjaValue::from(effort));
            }
        }
        let base_ctx = minijinja::context! {
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            messages => messages,
            tools => tools_value,
            add_generation_prompt => add_generation_prompt,
            date_string => date_string,
            ..JinjaValue::from_serialize(&derived)
        };
        // Merge caller-supplied extras on top of the base context.
        let ctx = if opts.extras.is_empty() {
            base_ctx
        } else {
            let mut extras: BTreeMap<String, JinjaValue> = BTreeMap::new();
            for (k, v) in &opts.extras {
                extras.insert(k.clone(), v.clone());
            }
            minijinja::context!(
                ..base_ctx,
                ..JinjaValue::from_serialize(&extras)
            )
        };
        let tmpl = env
            .get_template(self.template_name)
            .map_err(ChatTemplateError::from_jinja)?;
        let mut out =
            tmpl.render(ctx).map_err(ChatTemplateError::from_jinja)?;
        // Append the withheld open thought, raw. Byte-exactness is the
        // whole point: the KV cache holds these bytes verbatim, and
        // anything routed through Jinja would come back normalized
        // (Qwen3.6's stock template `|trim`s message content and
        // lstrip/rstrip's the halves it splits on `</think>`; the baked
        // replacement renders a *closed* turn verbatim, but has no
        // spelling for an unclosed one), making `\n` and `\n\n\n`
        // indistinguishable.
        if let Some(body) = open_tail {
            // The open marker comes from whichever side actually emits
            // it: a thinking-enabled template ends its generation
            // prompt with `<think>` already, so appending another would
            // duplicate it. Otherwise (thinking off — Qwen scaffolds a
            // *closed* `<think>\n\n</think>\n\n` — or a template with
            // no reasoning scaffold) we supply it, which reproduces the
            // exact bytes a spontaneous unclosed `<think>` emitted.
            if !out.trim_end().ends_with(reasoning_start) {
                out.push_str(reasoning_start);
            }
            out.push_str(&surfaces.text(body, true));
        }
        Ok((out, surfaces.counts.into_inner()))
    }

    /// Render the prompt plus one partial render per `cache_control`
    /// breakpoint.
    ///
    /// Use this when a caller — e.g. `Session` — needs to know where
    /// in the tokenized output the caller's cache breakpoints land so a
    /// later call can compute cache-reuse boundaries in tokens. See
    /// `RenderedWithBreakpoints` for the returned shape.
    ///
    /// Partial renders force `add_generation_prompt = false` regardless
    /// of what the caller set on `opts` — only the full render honors
    /// the caller's preference. If `prompt` carries no cache markers,
    /// the returned `partial_texts` is empty and the full render is
    /// equivalent to [`render_with`](Self::render_with).
    ///
    /// **Fail-open on partial errors.** If the full prompt renders
    /// cleanly but a partial (e.g. the `AfterSystem` breakpoint's
    /// "system-only, no user message" prompt) hits a template
    /// `raise_exception` — Qwen3's "No user query found in messages."
    /// is the canonical case — that breakpoint is silently dropped
    /// from `partial_texts` rather than failing the whole call. The
    /// caller loses cache reuse at that boundary; correctness is
    /// preserved.
    ///
    /// [`Session`]: crate::Session
    pub fn render_with_breakpoints(
        &self,
        prompt: &Prompt,
        opts: &RenderOptions,
    ) -> Result<RenderedWithBreakpoints, ChatTemplateError> {
        self.render_with_breakpoints_counted(prompt, opts)
            .map(|(rendered, _)| rendered)
    }

    /// [`Self::render_with_breakpoints`], plus the full render's
    /// neutralization counts (see [`Self::render_counted`]).
    pub(crate) fn render_with_breakpoints_counted(
        &self,
        prompt: &Prompt,
        opts: &RenderOptions,
    ) -> Result<(RenderedWithBreakpoints, LiteralCounts), ChatTemplateError>
    {
        let (text, counts) = self.render_counted(prompt, opts)?;
        let breakpoints = collect_breakpoints(prompt);
        let mut partials = Vec::with_capacity(breakpoints.len());
        for (bp, ttl) in breakpoints {
            match render_partial(self, prompt, opts, bp) {
                Ok(s) => partials.push((bp, ttl, s)),
                Err(e) => {
                    // Drop this breakpoint — same fail-open posture
                    // tokenize_with_breakpoints uses for non-prefix-
                    // safe partials. A breakpoint we can't render
                    // can't be tokenized either, so the
                    // partial_texts slot is unrecoverable. Warn at
                    // default level — losing a breakpoint silently
                    // is a cache-correctness signal, not a debug
                    // nicety.
                    tracing::warn!(
                        target: "drama_llama::chat_template",
                        event = "cache_degrade",
                        reason = "partial_render_failed",
                        breakpoint = ?bp,
                        error = %e,
                        "partial render failed; breakpoint dropped from \
                         partial_texts (cache reuse lost at this position)",
                    );
                }
            }
        }
        Ok((RenderedWithBreakpoints { text, partials }, counts))
    }
}

/// Rendered prompt plus one partial render per `cache_control`
/// breakpoint.
///
/// Used by [`Session`] (Phase 3 of prompt caching) to compute which
/// prefix of the new prompt's token stream matches a previously-cached
/// state. Each partial is rendered with `add_generation_prompt=false`;
/// each should tokenize to a strict prefix of [`text`](Self::text) for
/// any well-behaved chat template.
///
/// `partials` is ordered canonically by how much of the prompt is
/// included: [`AfterTools`], then [`AfterSystem`], then
/// [`AfterMessage(0)`], [`AfterMessage(1)`], … Only breakpoints that
/// actually exist in the prompt appear. Each partial carries the
/// [`PromptBreakpoint`] it truncates at, so cache machinery can map a
/// matched breakpoint back to its position in Prompt *structure*
/// (which blocks precede it) — the Session's block-gated seeding fold
/// resumes from exactly that cursor.
///
/// [`Session`]: crate::Session
/// [`AfterTools`]: PromptBreakpoint::AfterTools
/// [`AfterSystem`]: PromptBreakpoint::AfterSystem
/// [`AfterMessage(0)`]: PromptBreakpoint::AfterMessage
/// [`AfterMessage(1)`]: PromptBreakpoint::AfterMessage
#[derive(Clone, Debug)]
pub struct RenderedWithBreakpoints {
    /// Full render — equivalent to
    /// [`ChatTemplate::render_with`]'s output.
    pub text: String,
    /// One `(breakpoint, ttl, partial render)` triple per breakpoint,
    /// in canonical order. The TTL is the marker's `cache_control`
    /// ephemeral lifetime (5m default, 1h opt-in) — see
    /// `collect_breakpoints` for how a section with several cached
    /// blocks resolves to one TTL.
    pub partials: Vec<(PromptBreakpoint, CacheTtl, String)>,
}

/// Options passed to [`ChatTemplate::render_with`].
///
/// Variables that are universal to HuggingFace-style templates
/// (`messages`, `tools`, `bos_token`, `eos_token`,
/// `add_generation_prompt`) are sourced from [`Prompt`] and the
/// [`ChatTemplate`] itself. Everything else — template-specific flags
/// like Llama 3.1's `tools_in_user_message`, date overrides, or
/// `builtin_tools` — goes through [`extras`](Self::extras).
#[derive(Clone, Debug, Default)]
pub struct RenderOptions {
    /// Ask the template to append an empty assistant header so the model
    /// generates the next turn. True for live chat, false for
    /// tokenizing a stored transcript.
    pub add_generation_prompt: bool,
    /// Current date string (e.g. `"17 Apr 2026"`). Llama 3.1's template
    /// reads `date_string` when stamping a system-message header. If
    /// `None`, the template's default (static fallback) is used.
    pub date_string: Option<String>,
    /// Template-specific extra variables. Keys become top-level names in
    /// the Jinja context. Values are arbitrary Serialize-able data.
    pub extras: Vec<(String, JinjaValue)>,
    /// How assistant [`Block::Thought`]s re-ingest: inline
    /// `<think>…</think>` text in `content` (default, legacy Qwen
    /// convention) or as the message's `reasoning` /
    /// `reasoning_content` fields (Gemma 4, DeepSeek-style templates
    /// that own the reasoning markers themselves).
    /// `Session` sets this from the model's analyzed dialect
    /// ([`CallSyntax::reasoning`](crate::CallSyntax)).
    ///
    /// [`Block::Thought`]: crate::Block
    pub thought_reingest: crate::dialect::ReasoningReingest,
    /// Per-call random media sentinel. When set, each
    /// [`Block::Image`] renders as `<{sentinel}:{source_hash_hex}>`
    /// — an out-of-band marker the caller (`Session`) later splits
    /// the render on, mapping each occurrence back to its image by
    /// source hash (sha256 of the block's base64 payload — see
    /// `image_source_hash`). When unset, a prompt
    /// containing images fails to render with
    /// [`ChatTemplateError::MediaUnsupported`] — never a silent
    /// drop.
    ///
    /// The sentinel being random per call (and never surfaced
    /// anywhere) is what makes marker injection structurally
    /// impossible: no content — user text, tool results, source
    /// files under discussion, even a literal mtmd `<__media__>` —
    /// can collide with it, so downstream media splitting never
    /// interprets content bytes.
    ///
    /// [`Block::Image`]: crate::Block
    pub media_sentinel: Option<String>,
    /// The dialect's reasoning **open** marker, trimmed (`"<think>"`).
    /// Required only to render a prompt whose tail is an *open* thought
    /// ([`OPEN_THOUGHT_SIGNATURE`]); such a prompt fails with
    /// [`ChatTemplateError::OpenThoughtUnsupported`] when this is
    /// unset, never a silently dropped body. `Session` sets it from the
    /// analyzed dialect, beside [`Self::thought_reingest`].
    ///
    /// Trimmed on purpose: the parser matches the trimmed form and the
    /// model emits the trimmed form; any canonical whitespace around
    /// the marker belongs to the template, not to us.
    ///
    /// [`OPEN_THOUGHT_SIGNATURE`]: crate::prompt::OPEN_THOUGHT_SIGNATURE
    pub reasoning_start: Option<String>,
    /// The `reasoning_effort` values the template accepts, lowest
    /// first ([`ReasoningSyntax::efforts`]). When a thinking-enabled
    /// prompt carries `output_config.effort`, the render sets
    /// `reasoning_effort` to that level if accepted, else to the
    /// nearest accepted one (the lower on a tie): Qwen3.8 has no `max`,
    /// so `Max` renders `xhigh`. Empty (the default) = no knob; the
    /// effort is ignored. A caller's `reasoning_effort` extra always
    /// wins. `Session` sets this from the analyzed dialect, beside
    /// [`Self::thought_reingest`].
    ///
    /// [`ReasoningSyntax::efforts`]: crate::dialect::ReasoningSyntax::efforts
    pub efforts: Vec<String>,
    /// Content-literal neutralization (see [`LiteralNeutralizer`]).
    /// When set, every content string the template sees has its
    /// reserved pieces replaced by markers, and tool names and
    /// tool-use ids must match Anthropic's patterns
    /// ([`ChatTemplateError::InvalidIdentifier`]). `Session` sets this
    /// on every render itself, whatever
    /// [`Session::with_render_opts`](crate::Session::with_render_opts)
    /// was given.
    pub literals: Option<Literals>,
}

impl RenderOptions {
    /// Builder: set `add_generation_prompt`.
    pub fn with_generation_prompt(mut self, yes: bool) -> Self {
        self.add_generation_prompt = yes;
        self
    }

    /// Builder: set `date_string`.
    pub fn with_date<S>(mut self, date: S) -> Self
    where
        S: Into<String>,
    {
        self.date_string = Some(date.into());
        self
    }

    /// Builder: add an arbitrary `(key, value)` pair to the Jinja
    /// context. Useful for `tools_in_user_message`, `builtin_tools`,
    /// etc. Any [`serde::Serialize`] value works — numbers, strings,
    /// booleans, structs, `serde_json::Value`, etc.
    pub fn with_extra<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Serialize,
    {
        self.extras
            .push((key.into(), JinjaValue::from_serialize(&value)));
        self
    }

    /// Builder: set the thought re-ingest convention.
    pub fn with_thought_reingest(
        mut self,
        reingest: crate::dialect::ReasoningReingest,
    ) -> Self {
        self.thought_reingest = reingest;
        self
    }

    /// Builder: set the per-call media sentinel (see
    /// [`RenderOptions::media_sentinel`]).
    pub fn with_media_sentinel<S>(mut self, sentinel: S) -> Self
    where
        S: Into<String>,
    {
        self.media_sentinel = Some(sentinel.into());
        self
    }

    /// Builder: set the dialect's reasoning open marker (see
    /// [`RenderOptions::reasoning_start`]). Stored trimmed; an
    /// all-whitespace or empty marker is treated as absent, since a
    /// dialect with no open marker cannot express an open thought.
    pub fn with_reasoning_start<S>(mut self, start: S) -> Self
    where
        S: AsRef<str>,
    {
        let start = start.as_ref().trim();
        self.reasoning_start = (!start.is_empty()).then(|| start.to_string());
        self
    }

    /// Builder: set the template's accepted effort levels (see
    /// [`RenderOptions::efforts`]).
    pub fn with_efforts<I, S>(mut self, efforts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.efforts = efforts.into_iter().map(Into::into).collect();
        self
    }

    /// Builder: neutralize content literals (see
    /// [`RenderOptions::literals`]).
    pub fn with_literals(mut self, literals: Literals) -> Self {
        self.literals = Some(literals);
        self
    }
}

/// The ordered effort scale, lowest first — Anthropic's
/// [`Effort`] levels, which are also the chat-template spellings.
/// Nearest-level mapping and the analyzer's effort probe both walk it.
pub(crate) const EFFORT_SCALE: [&str; 5] =
    ["low", "medium", "high", "xhigh", "max"];

/// Map a requested [`Effort`] onto the levels a template `accepted`:
/// the level itself if accepted, else the nearest accepted one on
/// [`EFFORT_SCALE`], the lower on a tie. `None` for an unknown
/// ([`Effort::Custom`]) level or an empty set.
pub(crate) fn resolve_effort<'a>(
    requested: &Effort,
    accepted: &'a [String],
) -> Option<&'a str> {
    let rank = |level: &str| EFFORT_SCALE.iter().position(|&l| l == level);
    let want = rank(requested.as_str())?;
    accepted
        .iter()
        .filter_map(|level| Some((rank(level)?, level.as_str())))
        .min_by_key(|&(r, _)| (want.abs_diff(r), r))
        .map(|(_, level)| level)
}

/// Whether `prompt` asks for thinking. `Some(Thinking::Disabled)` is
/// an explicit *off*: checking `thinking.is_some()` instead rendered it
/// as `enable_thinking = true`.
pub(crate) fn thinking_enabled(prompt: &Prompt) -> bool {
    !matches!(prompt.thinking, None | Some(Thinking::Disabled))
}

/// The `reasoning_effort` a render passes the template, if any: only
/// for a thinking-enabled prompt that requests an effort, mapped onto
/// the template's accepted levels by [`resolve_effort`]. Thinking off
/// never sets it, so a thinking-off render is unchanged.
fn derive_reasoning_effort<'a>(
    prompt: &Prompt,
    accepted: &'a [String],
) -> Option<&'a str> {
    if !thinking_enabled(prompt) {
        return None;
    }
    let requested = prompt.output_config.as_ref()?.effort.as_ref()?;
    if accepted.is_empty() {
        debug_once(format!(
            "effort `{requested}` requested, but the chat template has \
             no reasoning_effort knob; ignoring it"
        ));
        return None;
    }
    let chosen = resolve_effort(requested, accepted);
    match chosen {
        Some(level) if level != requested.as_str() => debug_once(format!(
            "effort `{requested}` isn't accepted by the chat template \
             ({accepted:?}); rendering reasoning_effort `{level}`"
        )),
        None => debug_once(format!(
            "effort `{requested}` is not a known level; leaving \
             reasoning_effort unset"
        )),
        Some(_) => {}
    }
    chosen
}

/// Log `message` at debug level the first time it occurs. Every call
/// renders the prompt several times (full plus one partial per cache
/// breakpoint), so a per-render log would repeat itself.
fn debug_once(message: String) {
    use std::{
        collections::BTreeSet,
        sync::{Mutex, OnceLock},
    };
    static SEEN: OnceLock<Mutex<BTreeSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    let first = seen
        .lock()
        .map(|mut seen| seen.insert(message.clone()))
        .unwrap_or(true);
    if first {
        tracing::debug!("{message}");
    }
}

// ===========================================================================
// Cache-breakpoint discovery
// ===========================================================================

/// A position in a [`Prompt`] where the caller placed a
/// `cache_control` marker.
///
/// Ordered chronologically in the rendered output: [`AfterTools`] (if
/// any tool is cached) comes before [`AfterSystem`] (if any system
/// block is cached), which comes before [`AfterMessage(i)`], and
/// later messages come after earlier ones. This matches the order
/// of the partial-render truncations — each covers strictly more of
/// the prompt than the previous — which is what gives us the
/// tokens-are-a-prefix property downstream.
///
/// Which variant a particular template renders "first" in bytes is
/// template-specific (Llama 3.1 renders tools inside the system
/// header; cogito renders them between system and user). We don't
/// care: we always truncate by prompt structure (tools, system,
/// messages), and the partial is a prefix of the full if and only if
/// the template is well-behaved. For templates where a coarse-
/// grained tools-only render isn't a byte-prefix (because the tool
/// list's closing `]` lands differently), we silently drop that
/// breakpoint at tokenization time.
///
/// [`AfterTools`]: PromptBreakpoint::AfterTools
/// [`AfterSystem`]: PromptBreakpoint::AfterSystem
/// [`AfterMessage(i)`]: PromptBreakpoint::AfterMessage
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptBreakpoint {
    /// After the tools section — any cache marker on any
    /// [`tool::CustomMethodDef`](misanthropic::tool::CustomMethodDef) in
    /// [`Prompt::tools`] produces this, regardless of which
    /// specific method was marked. Coarse by design: per-tool
    /// partial rendering would not produce a byte-prefix of the full
    /// render (the closing `]` of the tools JSON array lands
    /// differently).
    AfterTools,
    /// After the system section — any cache marker on any block in
    /// [`Prompt::system`] produces this. Coarse by design for the
    /// same reason as [`AfterTools`](Self::AfterTools).
    AfterSystem,
    /// After message index `i`, inclusive. Emitted iff
    /// `prompt.messages[i]`'s content has at least one block with
    /// `cache_control` set (via [`Block::is_cached`]). This is the
    /// fine-grained case that matters for the Agora reactor workload
    /// — N agents sharing a system + tools + early messages,
    /// diverging only in the last turn.
    ///
    /// [`Block::is_cached`]: misanthropic::prompt::message::Block::is_cached
    AfterMessage(usize),
}

/// Walk `prompt` and return the ordered list of cache breakpoints it
/// declares, each with its `cache_control` TTL. See [`PromptBreakpoint`]
/// for the ordering rule and granularity. Since a breakpoint is
/// per-*section* (tools / system / message) while markers are
/// per-*block*, a section with several cached blocks resolves to the
/// **max** TTL among them — the generous reading: any block asking for
/// an hour keeps the whole section's prefix alive for an hour.
///
/// The request-level [`Prompt::cache_control`] (Anthropic's automatic
/// caching) contributes one more breakpoint, on the section holding
/// the last cacheable block, with its own TTL. Explicit markers only
/// sit on cacheable blocks, so that section is never before the last
/// explicit one: the automatic breakpoint either extends the list or
/// merges into its last entry (max TTL again).
fn collect_breakpoints(prompt: &Prompt) -> Vec<(PromptBreakpoint, CacheTtl)> {
    let mut out = explicit_breakpoints(prompt);
    let auto = prompt
        .cache_control
        .as_ref()
        .zip(auto_cache_target(prompt))
        .map(|(control, target)| (target.at, control_ttl_of(control)));
    match (auto, out.last_mut()) {
        (Some((at, ttl)), Some((last, last_ttl))) if *last == at => {
            *last_ttl = max_ttl(last_ttl.clone(), ttl);
        }
        (Some(auto), _) => out.push(auto),
        (None, _) => {}
    }
    out
}

/// [`collect_breakpoints`] without the automatic breakpoint: the
/// sections the markers on tools and blocks declare.
fn explicit_breakpoints(prompt: &Prompt) -> Vec<(PromptBreakpoint, CacheTtl)> {
    let mut out = Vec::new();

    let tools_ttl = prompt.tools.as_ref().and_then(|defs| {
        defs.iter().filter_map(method_cache_ttl).reduce(max_ttl)
    });
    if let Some(ttl) = tools_ttl {
        out.push((PromptBreakpoint::AfterTools, ttl));
    }

    let system_ttl = prompt.system.as_ref().and_then(content_cache_ttl);
    if let Some(ttl) = system_ttl {
        out.push((PromptBreakpoint::AfterSystem, ttl));
    }

    for (i, m) in prompt.messages.iter().enumerate() {
        if let Some(ttl) = content_cache_ttl(&m.content) {
            out.push((PromptBreakpoint::AfterMessage(i), ttl));
        }
    }

    out
}

/// The TTL a `cache_control` marker asks for: the explicit `ttl`, or
/// five minutes when the marker omits it (Anthropic semantics —
/// `{type: "ephemeral"}` alone means 5m).
fn control_ttl(control: &Option<CacheControl>) -> Option<CacheTtl> {
    control.as_ref().map(control_ttl_of)
}

/// [`control_ttl`] for a marker known to be present.
fn control_ttl_of(control: &CacheControl) -> CacheTtl {
    match control {
        CacheControl::Ephemeral { ttl } => {
            ttl.clone().unwrap_or(CacheTtl::FiveMinutes)
        }
    }
}

/// Where Anthropic's automatic caching ([`Prompt::cache_control`])
/// places its breakpoint: the section holding the prompt's last
/// cacheable block, and that block's own explicit marker, if any.
#[derive(Clone, Debug)]
struct AutoCacheTarget {
    /// The section the breakpoint lands after.
    at: PromptBreakpoint,
    /// The TTL of an explicit marker already on the target block;
    /// `None` too for a cached server-tool definition, whose TTL is
    /// not readable upstream.
    marker: Option<CacheTtl>,
}

/// The last cacheable block of `prompt` in processing order (`tools`,
/// `system`, `messages`), walking backward past blocks that cannot
/// carry a marker (thoughts, server-tool results) as Anthropic does.
/// `None` when nothing can carry one; Anthropic then caches nothing.
fn auto_cache_target(prompt: &Prompt) -> Option<AutoCacheTarget> {
    fn last_cacheable(content: &Content) -> Option<&Block> {
        content
            .0
            .iter()
            .rev()
            .find(|block| block_is_cacheable(block))
    }
    let in_messages =
        prompt.messages.iter().enumerate().rev().find_map(|(i, m)| {
            last_cacheable(&m.content).map(|block| AutoCacheTarget {
                at: PromptBreakpoint::AfterMessage(i),
                marker: block_cache_ttl(block),
            })
        });
    let in_system = || {
        let block = last_cacheable(prompt.system.as_ref()?)?;
        Some(AutoCacheTarget {
            at: PromptBreakpoint::AfterSystem,
            marker: block_cache_ttl(block),
        })
    };
    let in_tools = || {
        use misanthropic::tool::MethodDef;
        let tool = prompt.tools.as_ref()?.last()?;
        // A server definition's TTL is not readable upstream: no marker
        // to compare, as in [`check_cache_controls`].
        let marker = match tool {
            MethodDef::Custom(c) => control_ttl(&c.cache_control),
            MethodDef::Server(_) => None,
        };
        Some(AutoCacheTarget {
            at: PromptBreakpoint::AfterTools,
            marker,
        })
    };
    in_messages.or_else(in_system).or_else(in_tools)
}

/// `block`'s `cache_control` field — `Some` for the kinds that can
/// carry a marker (holding the marker, if set), `None` for those that
/// cannot (thoughts, server-tool results, tool references). The one
/// place that knows which is which: [`block_is_cacheable`] and
/// [`block_cache_ttl`] both read it.
///
/// Every variant is named, as in misanthropic's `Block::is_cached`.
/// `Block` is `#[non_exhaustive]`, so a downstream match needs the
/// trailing wildcard and a new upstream variant still compiles — but
/// the `clippy::wildcard_enum_match_arm` denial below fires the moment
/// the wildcard would match a variant not named here, so `just check`
/// fails until someone decides which side it belongs on.
#[deny(clippy::wildcard_enum_match_arm)]
fn block_cache_control(block: &Block) -> Option<&Option<CacheControl>> {
    use misanthropic::tool;
    match block {
        Block::Text { cache_control, .. }
        | Block::Image { cache_control, .. }
        | Block::Document { cache_control, .. }
        | Block::ToolUse {
            call: tool::Use { cache_control, .. },
        }
        | Block::ToolResult {
            result: tool::Result { cache_control, .. },
        }
        | Block::ServerToolUse {
            call: tool::Use { cache_control, .. },
        } => Some(cache_control),
        Block::Thought { .. }
        | Block::RedactedThought { .. }
        | Block::WebSearchToolResult { .. }
        | Block::WebFetchToolResult { .. }
        | Block::ToolSearchToolResult { .. }
        | Block::CodeExecutionToolResult { .. }
        | Block::BashCodeExecutionToolResult { .. }
        | Block::TextEditorCodeExecutionToolResult { .. }
        | Block::ToolReference { .. } => None,
        // Unreachable today; see the lint note above.
        _ => None,
    }
}

/// Whether `block` can carry a `cache_control` marker
/// ([`block_cache_control`]).
fn block_is_cacheable(block: &Block) -> bool {
    block_cache_control(block).is_some()
}

/// One explicit `cache_control` marker, as Anthropic addresses it in
/// an error: its JSON path (`tools.1`, `system.0`,
/// `messages.2.content.0`) and its TTL. `ttl` is `None` for a server
/// tool definition, whose marker misanthropic does not expose — it
/// counts toward the limit but takes no part in the TTL rules, rather
/// than guessing and answering a valid request with a 400.
#[derive(Debug)]
struct Marker {
    path: String,
    ttl: Option<CacheTtl>,
}

/// Every explicit `cache_control` marker in `prompt`, in processing
/// order (`tools`, `system`, `messages`).
fn explicit_markers(prompt: &Prompt) -> Vec<Marker> {
    use misanthropic::tool::MethodDef;
    let tools =
        prompt
            .tools
            .iter()
            .flatten()
            .enumerate()
            .filter_map(|(i, def)| {
                let ttl = match def {
                    MethodDef::Custom(c) => {
                        Some(control_ttl(&c.cache_control)?)
                    }
                    MethodDef::Server(s) => s.is_cached().then_some(None)?,
                };
                Some(Marker {
                    path: format!("tools.{i}"),
                    ttl,
                })
            });
    let blocks_of = |prefix: String, content: &Content| {
        content
            .0
            .iter()
            .enumerate()
            .filter_map(|(i, block)| {
                Some(Marker {
                    path: format!("{prefix}{i}"),
                    ttl: Some(block_cache_ttl(block)?),
                })
            })
            .collect::<Vec<_>>()
    };
    let system = prompt
        .system
        .iter()
        .flat_map(|system| blocks_of("system.".into(), system));
    let messages = prompt.messages.iter().enumerate().flat_map(|(m, msg)| {
        blocks_of(format!("messages.{m}.content."), &msg.content)
    });
    tools.chain(system).chain(messages).collect()
}

/// Anthropic's per-request limit on `cache_control` markers, the
/// automatic one ([`Prompt::cache_control`]) included.
pub const MAX_CACHE_CONTROLS: usize = 4;

/// Anthropic's message for a 1-hour marker after a 5-minute one, at
/// `path` — the offending explicit marker's `….cache_control.ttl`, or
/// `cache_control` for the request-level automatic one.
fn ttl_order_error(path: &str) -> String {
    format!(
        "{path}: a ttl='1h' cache_control block must not come after a \
         ttl='5m' cache_control block. Note that blocks are processed in \
         the following order: `tools`, `system`, `messages`."
    )
}

/// The checks Anthropic makes on a request's `cache_control` markers,
/// each an `invalid_request_error` whose message this returns in
/// Anthropic's exact wording (captured 2026-09-30 on claude-haiku-4-5
/// via `count_tokens`; rules 1, 3 and 4 on `/v1/messages` too):
///
/// 1. At most [`MAX_CACHE_CONTROLS`] markers, and the automatic one
///    always counts — even on a block that already carries an explicit
///    marker with the same TTL, which the docs call a no-op but the
///    wire counts: `A maximum of 4 blocks with cache_control may be
///    provided. Found 5.`
/// 2. Longer TTLs come first: a 1-hour explicit marker must not follow
///    a 5-minute one. Anthropic names the first offender by path —
///    `messages.0.content.1.cache_control.ttl: a ttl='1h' …`.
/// 3. The automatic marker's TTL must match an explicit marker on the
///    block it lands on (not checked when that block is a cached
///    server-tool definition).
/// 4. The same ordering rule for the automatic marker, which comes
///    last: a 1-hour automatic marker after any 5-minute explicit one,
///    reported at the path `cache_control`.
///
/// Checked in that order, which is Anthropic's: for each rule, a
/// request breaking it and every later one was captured answering
/// with that rule's message. A cached server-tool definition counts
/// toward rule 1, but its TTL is not readable upstream, so it takes
/// no part in rules 2–4 rather than guessing a TTL and answering a
/// valid request with a 400.
pub fn check_cache_controls(prompt: &Prompt) -> Result<(), String> {
    let explicit = explicit_markers(prompt);
    let found = explicit.len() + usize::from(prompt.cache_control.is_some());
    if found > MAX_CACHE_CONTROLS {
        return Err(format!(
            "A maximum of {MAX_CACHE_CONTROLS} blocks with cache_control \
             may be provided. Found {found}."
        ));
    }
    let is_hour =
        |ttl: &CacheTtl| ttl_duration(ttl) == ttl_duration(&CacheTtl::OneHour);
    let first_five = explicit
        .iter()
        .position(|m| m.ttl.as_ref().is_some_and(|ttl| !is_hour(ttl)));
    let hour_after_five = first_five.and_then(|at| {
        explicit[at..]
            .iter()
            .find(|m| m.ttl.as_ref().is_some_and(is_hour))
    });
    if let Some(marker) = hour_after_five {
        return Err(ttl_order_error(&format!(
            "{}.cache_control.ttl",
            marker.path
        )));
    }
    let Some(auto) = prompt.cache_control.as_ref().map(control_ttl_of) else {
        return Ok(());
    };
    let marker = auto_cache_target(prompt).and_then(|target| target.marker);
    if let Some(marker) = marker {
        if ttl_duration(&marker) != ttl_duration(&auto) {
            return Err(format!(
                "Top-level cache_control has ttl='{auto}' but the target \
                 block already has cache_control with ttl='{marker}'. When \
                 both are specified on the same block, they must have \
                 matching TTLs."
            ));
        }
    }
    if is_hour(&auto) && first_five.is_some() {
        return Err(ttl_order_error("cache_control"));
    }
    Ok(())
}

/// The TTL of `block`'s cache marker, if it carries one
/// ([`block_cache_control`]).
fn block_cache_ttl(block: &Block) -> Option<CacheTtl> {
    block_cache_control(block).and_then(control_ttl)
}

/// The TTL of a tool definition's cache marker, if any, for placing
/// breakpoints. Server-side definitions don't expose their
/// `cache_control` upstream (private accessor), so a marked server def
/// keeps its breakpoint for the conservative 5-minute default; the
/// 400 checks ([`check_cache_controls`]) give it no TTL at all.
fn method_cache_ttl(def: &misanthropic::tool::MethodDef) -> Option<CacheTtl> {
    use misanthropic::tool::MethodDef;
    match def {
        MethodDef::Custom(c) => control_ttl(&c.cache_control),
        MethodDef::Server(s) => s.is_cached().then_some(CacheTtl::FiveMinutes),
    }
}

/// The section-level TTL of `content`: max TTL over its cached blocks,
/// `None` when no block carries a marker.
fn content_cache_ttl(content: &Content) -> Option<CacheTtl> {
    content.0.iter().filter_map(block_cache_ttl).reduce(max_ttl)
}

/// Wall-clock lifetime of a cache TTL.
pub(crate) fn ttl_duration(ttl: &CacheTtl) -> std::time::Duration {
    match ttl {
        CacheTtl::FiveMinutes => std::time::Duration::from_secs(5 * 60),
        CacheTtl::OneHour => std::time::Duration::from_secs(60 * 60),
    }
}

/// The longer-lived of two TTLs (by [`ttl_duration`] — `CacheTtl`
/// deliberately has no `Ord` upstream).
pub(crate) fn max_ttl(a: CacheTtl, b: CacheTtl) -> CacheTtl {
    if ttl_duration(&b) > ttl_duration(&a) {
        b
    } else {
        a
    }
}

/// Render `prompt` truncated at `up_to` with
/// `add_generation_prompt=false`. Used as the breakpoint-discovery
/// partial render; does not mutate `prompt` (clones a truncated view).
fn render_partial(
    template: &ChatTemplate,
    prompt: &Prompt,
    opts: &RenderOptions,
    up_to: PromptBreakpoint,
) -> Result<String, ChatTemplateError> {
    // Everything the template can see besides `messages` must carry
    // over, or the partial is not a prefix of the full render and
    // gets dropped. `thinking` reaches the template as
    // `enable_thinking`; Mistral Small 4 writes it into the prompt
    // PREFIX (`[MODEL_SETTINGS]{"reasoning_effort": ...}`), so a
    // partial rendered with it unset diverged from every thinking-on
    // full render and the model lost every breakpoint (#93 follow-up,
    // 2026-09-12). Qwen only reads it at the generation tail, which
    // partials never render, so it never showed there.
    //
    // `output_config.effort` reaches the template as
    // `reasoning_effort`, which Qwen3.8, Mistral and gpt-oss all write
    // into the system PREFIX — same failure, same fix. Only the effort
    // is carried: the render never reads `format`, and a partial has no
    // business depending on it.
    let output_config = prompt
        .output_config
        .as_ref()
        .and_then(|c| c.effort.clone())
        .map(OutputConfig::effort);
    let truncated = match up_to {
        PromptBreakpoint::AfterTools => Prompt {
            tools: prompt.tools.clone(),
            thinking: prompt.thinking,
            output_config: output_config.clone(),
            // Carry the system content too. Every modern template
            // (Qwen3, Llama 3.1, Hermes, Cogito) coalesces tools into
            // the system block, so a "tools-only, no system" truncation
            // has no byte-level analogue in the full render — and
            // worse, it leaves Qwen3's permissive-env render walking
            // `messages[::-1]` over an empty list, which minijinja
            // panics on. Including the system content keeps the
            // truncation a real prefix of the full render on those
            // templates and dodges the panic.
            system: prompt.system.clone(),
            messages: Vec::new(),
            ..Prompt::default()
        },
        PromptBreakpoint::AfterSystem => Prompt {
            tools: prompt.tools.clone(),
            system: prompt.system.clone(),
            messages: Vec::new(),
            thinking: prompt.thinking,
            output_config: output_config.clone(),
            ..Prompt::default()
        },
        PromptBreakpoint::AfterMessage(i) => Prompt {
            tools: prompt.tools.clone(),
            system: prompt.system.clone(),
            messages: prompt.messages[..=i].to_vec(),
            thinking: prompt.thinking,
            output_config,
            ..Prompt::default()
        },
    };
    let partial_opts = opts.clone().with_generation_prompt(false);
    template
        .render_with_env(&template.env_permissive, &truncated, &partial_opts)
        .map(|(text, _)| text)
}

/// Tokenize the full render and each partial in `rendered`, returning
/// the full token stream plus the sorted, deduplicated breakpoint
/// token indices.
///
/// Each partial's tokens MUST be a prefix of the full's tokens —
/// that's what makes a breakpoint useful for KV-cache reuse. If a
/// partial's tokens are not a prefix (unexpected template weirdness,
/// e.g. a non-prefix-safe truncation point), the breakpoint is
/// silently dropped from the returned indices. We fail open to
/// uncached behavior for that call rather than erroring, so cache
/// oddities degrade performance rather than correctness.
///
/// Tokenizes the whole render with `parse_special=true` so chat
/// markers (`<|im_start|>`, `<|eot_id|>`, …) resolve to their single
/// special-token IDs. A diagnostic helper, not what [`Session`]
/// feeds the model: this knows nothing of images or content literals
/// ([`RenderOptions::literals`]), so a special piece spelled by
/// content tokenizes here as the real special, and a render made with
/// markers is not split. `Session` tokenizes marker-aware.
///
/// [`Session`]: crate::Session
pub fn tokenize_with_breakpoints<M: Model>(
    model: &M,
    rendered: &RenderedWithBreakpoints,
) -> (Vec<Token>, Vec<usize>) {
    let full_tokens = model.tokenize(&rendered.text, true);
    let mut indices: Vec<usize> = Vec::with_capacity(rendered.partials.len());
    for (_, _, partial) in &rendered.partials {
        let partial_tokens = model.tokenize(partial, true);
        if partial_tokens.len() <= full_tokens.len()
            && full_tokens[..partial_tokens.len()] == partial_tokens[..]
        {
            indices.push(partial_tokens.len());
        } else {
            // Fail-open: drop the breakpoint. Logging is deliberately
            // omitted — drama_llama doesn't wire a tracing crate, and
            // a silent drop is the documented behavior.
        }
    }
    indices.sort_unstable();
    indices.dedup();
    (full_tokens, indices)
}

// ===========================================================================
// Prompt -> Jinja context conversion
// ===========================================================================

/// Build the `messages` sequence the template will iterate.
///
/// If `prompt.system` is set we synthesize a leading `system` message —
/// that matches the HF/Llama convention of carrying the system prompt as
/// the first message in the transcript.
///
/// Tool-calling messages are emitted in the shape HF templates expect:
///
/// * An assistant message with [`Block::ToolUse`]s becomes
///   `{role: "assistant", tool_calls: [{function: {name, arguments}},
///   …]}` — one message carrying *every* call, the shape template
///   `tool_calls` loops iterate (parallel calls included).
///   Accompanying text stays as `content`; thoughts route per
///   `reingest`.
/// * A user message containing [`Block::ToolResult`] blocks is split:
///   each tool result emits a separate `{role: "tool", content: ...}`
///   message. Any remaining text in the same user turn follows as a
///   normal user message.
///
/// Every content string passes through `surfaces` on its way in (see
/// [`LiteralNeutralizer`]); tool names and tool-use ids are validated
/// there instead.
fn build_messages(
    prompt: &Prompt,
    reingest: crate::dialect::ReasoningReingest,
    surfaces: &Surfaces<'_>,
    withhold_tail: bool,
) -> Result<Vec<JinjaValue>, ChatTemplateError> {
    let mut out: Vec<JinjaValue> =
        Vec::with_capacity(prompt.messages.len() + 1);
    if let Some(system) = prompt.system.as_ref() {
        let system = flatten_text(system, surfaces.media_sentinel);
        out.push(text_message("system", system.render(surfaces, true)));
    }
    let messages = match withhold_tail {
        // The trailing open-thought message is rendered by the caller,
        // after the generation prompt — see `open_thought_tail`.
        true => &prompt.messages[..prompt.messages.len() - 1],
        false => &prompt.messages[..],
    };
    for (index, m) in messages.iter().enumerate() {
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            // Mid-conversation system turns (misanthropic ≥alpha.2).
            // HF templates broadly accept repeated system messages.
            Role::System => "system",
        };
        append_message(&mut out, role, index, &m.content, reingest, surfaces)?;
    }
    Ok(out)
}

/// Emit one or more Jinja messages for a single misanthropic Message
/// (the prompt's `index`th).
fn append_message(
    out: &mut Vec<JinjaValue>,
    role: &str,
    index: usize,
    content: &Content,
    reingest: crate::dialect::ReasoningReingest,
    surfaces: &Surfaces<'_>,
) -> Result<(), ChatTemplateError> {
    let blocks: Vec<&Block> = content.0.iter().collect();
    let media_sentinel = surfaces.media_sentinel;

    // User turn: split ToolResult blocks into their own "tool" messages,
    // collect remaining text/thought into a trailing user message.
    if role == "user" {
        let mut residual = Flat::default();
        for (b, block) in blocks.iter().enumerate() {
            match block {
                Block::ToolResult { result } => {
                    surfaces.identifier(
                        &result.tool_use_id,
                        is_identifier,
                        || format!("message {index} block {b}: tool_use_id"),
                    )?;
                    let content = flatten_text(&result.content, media_sentinel)
                        .render(surfaces, true);
                    out.push(tool_result_message(&result.tool_use_id, content));
                }
                other => {
                    append_block_text(&mut residual, other, media_sentinel)
                }
            }
        }
        if !residual.is_empty() {
            out.push(text_message(role, residual.render(surfaces, true)));
        }
        return Ok(());
    }

    // Assistant turn. Thoughts route by convention: inline
    // `<think>…</think>` in content (legacy Qwen templates
    // reconstruct from content), or concatenated into the
    // `reasoning`/`reasoning_content` fields (Gemma 4/DeepSeek-style
    // templates own the markers; inlining would pollute content).
    use crate::dialect::ReasoningReingest;
    let calls: Vec<(usize, &crate::prompt::ToolUse)> = blocks
        .iter()
        .enumerate()
        .filter_map(|(b, block)| match block {
            Block::ToolUse { call } => Some((b, call)),
            _ => None,
        })
        .collect();
    // Text splits by block order around the first ToolUse: prose the
    // model emitted BEFORE calling (announce-then-call) vs after.
    // Reordering them would invert causality in the transcript, so
    // causality-aware templates get both halves; stock templates read
    // the merged `content` and keep their own layout.
    let mut reasoning = String::new();
    let mut content_pre = Flat::default();
    let mut content_post = Flat::default();
    let mut seen_call = false;
    for b in &blocks {
        match b {
            Block::ToolUse { .. } => seen_call = true,
            Block::Thought { thought, .. }
                if matches!(
                    reingest,
                    ReasoningReingest::Field | ReasoningReingest::Thinking
                ) =>
            {
                reasoning.push_str(thought);
            }
            other => append_block_text(
                if seen_call {
                    &mut content_post
                } else {
                    &mut content_pre
                },
                other,
                media_sentinel,
            ),
        }
    }
    // Consecutive thoughts concatenate with nothing between them, so
    // the joined string is what gets neutralized.
    let reasoning = surfaces.text(&reasoning, true);
    let chunks = assistant_chunks(&blocks, surfaces);

    // One message carrying every call: the shape template
    // `tool_calls` loops iterate, so parallel calls re-render intact.
    if !calls.is_empty() {
        let tool_calls = calls
            .iter()
            .map(|&(b, call)| {
                let at = || format!("message {index} block {b}: tool_use");
                surfaces.identifier(&call.id, is_identifier, || {
                    format!("{} id", at())
                })?;
                surfaces.identifier(&call.name, is_tool_name, || {
                    format!("{} name", at())
                })?;
                Ok(minijinja::context! {
                    id => call.id.as_ref(),
                    function => minijinja::context! {
                        name => call.name.as_ref(),
                        arguments => surfaces.value(&call.input, true),
                    },
                })
            })
            .collect::<Result<Vec<JinjaValue>, ChatTemplateError>>()?;
        // The merged `content` joins the halves with nothing between
        // them, so it is the counted form; the halves are what
        // causality-aware templates render around the calls.
        let content = content_pre.join(&content_post).render(surfaces, true);
        let message = tool_call_message(
            role,
            content,
            (
                &content_pre.render(surfaces, false),
                &content_post.render(surfaces, false),
            ),
            tool_calls,
            &reasoning,
            reingest,
        );
        out.push(minijinja::context! { chunks => chunks, ..message });
        return Ok(());
    }
    let message = assistant_text_message(
        role,
        content_pre.render(surfaces, true),
        &reasoning,
    );
    out.push(minijinja::context! { chunks => chunks, ..message });
    Ok(())
}

/// An assistant message's blocks in emission order, as the `chunks`
/// list templates may read in place of the merged fields: `{type:
/// "text", text}` for a run of prose, `{type: "thinking", thinking}`
/// for each thought, and one `{type: "tool_calls"}` where the first
/// call sits (the calls themselves are the message's `tool_calls`).
///
/// The merged `content` and `reasoning` fields lose two things the
/// model wrote, and a template that renders the turn from them cannot
/// reproduce it: where each thought sat relative to the prose, and
/// that two back-to-back thoughts were two (Mistral 4's
/// `…[/THINK][THINK]…`, live 2026-10-01). Stock Mistral templates read
/// exactly this shape (`content` as a list of text and thinking
/// chunks), and the baked Mistral 4 and gpt-oss templates render from
/// it. Neutralized uncounted: the merged fields carry the count.
fn assistant_chunks(blocks: &[&Block], surfaces: &Surfaces<'_>) -> JinjaValue {
    let media_sentinel = surfaces.media_sentinel;
    let mut chunks: Vec<JinjaValue> = Vec::new();
    let mut prose = Flat::default();
    let mut seen_call = false;
    let flush = |prose: &mut Flat, chunks: &mut Vec<JinjaValue>| {
        if !prose.is_empty() {
            let text = std::mem::take(prose).render(surfaces, false);
            chunks.push(minijinja::context! { type => "text", text => text });
        }
    };
    for block in blocks {
        match block {
            Block::Thought { thought, .. } => {
                flush(&mut prose, &mut chunks);
                chunks.push(minijinja::context! {
                    type => "thinking",
                    thinking => surfaces.text(thought, false),
                });
            }
            Block::ToolUse { .. } => {
                flush(&mut prose, &mut chunks);
                if !std::mem::replace(&mut seen_call, true) {
                    chunks.push(minijinja::context! { type => "tool_calls" });
                }
            }
            other => append_block_text(&mut prose, other, media_sentinel),
        }
    }
    flush(&mut prose, &mut chunks);
    JinjaValue::from(chunks)
}

fn text_message(role: &str, content: String) -> JinjaValue {
    minijinja::context! {
        role => role,
        content => content,
    }
}

/// Assistant message with one `tool_calls` entry per call. Shape:
/// `{role, content, content_pre, content_post,
/// tool_calls: [{id, function: {name, arguments}}, …]}`.
///
/// Llama 3.1's template reads `.function.name` and `.function.arguments`
/// off each entry. OpenAI-style templates also look at `.id`, so we
/// include it. We intentionally omit the `type` field — templates that
/// need it default to `"function"`. A non-empty `reasoning` renders as
/// both `reasoning` and `reasoning_content` (templates read one or the
/// other).
///
/// `content` is the merged text (pre ++ post) — what stock templates
/// read, laid out however they lay it out. `content_pre` /
/// `content_post` carry the block-order split around the first call
/// so causality-aware templates (our Gemma 4 cache-stable patch) can
/// render announce-then-call in emission order.
fn tool_call_message(
    role: &str,
    content: String,
    (content_pre, content_post): (&str, &str),
    tool_calls: Vec<JinjaValue>,
    reasoning: &str,
    reingest: crate::dialect::ReasoningReingest,
) -> JinjaValue {
    if reasoning.is_empty() {
        minijinja::context! {
            role => role,
            content => content,
            content_pre => content_pre,
            content_post => content_post,
            tool_calls => tool_calls,
        }
    } else if reingest == crate::dialect::ReasoningReingest::Thinking {
        // gpt-oss convention (upstream parity): the template reads
        // `thinking`, and raises when merged `content` accompanies it
        // on a tool-call message — withhold `content`; causality-
        // aware sidecars read the pre/post split instead.
        minijinja::context! {
            role => role,
            content_pre => content_pre,
            content_post => content_post,
            tool_calls => tool_calls,
            thinking => reasoning,
            reasoning => reasoning,
            reasoning_content => reasoning,
        }
    } else {
        minijinja::context! {
            role => role,
            content => content,
            content_pre => content_pre,
            content_post => content_post,
            tool_calls => tool_calls,
            reasoning => reasoning,
            reasoning_content => reasoning,
        }
    }
}

/// Assistant message without tool calls; carries `reasoning` /
/// `reasoning_content` fields when the caller routed thoughts there
/// ([`ReasoningReingest::Field`](crate::dialect::ReasoningReingest)).
fn assistant_text_message(
    role: &str,
    content: String,
    reasoning: &str,
) -> JinjaValue {
    if reasoning.is_empty() {
        return text_message(role, content);
    }
    minijinja::context! {
        role => role,
        content => content,
        reasoning => reasoning,
        reasoning_content => reasoning,
    }
}

/// Tool-result message. Shape: `{role: "tool", tool_call_id, content}`.
/// `tool_call_id` matches what HF / OpenAI templates read; Llama 3.1's
/// template ignores it, but it's cheap to include.
fn tool_result_message(tool_use_id: &str, content: String) -> JinjaValue {
    minijinja::context! {
        role => "tool",
        tool_call_id => tool_use_id,
        content => content,
    }
}

/// Flatten any [`Content`] using [`append_block_text`] for each part.
fn flatten_text(content: &Content, media_sentinel: Option<&str>) -> Flat {
    let mut out = Flat::default();
    for b in &content.0 {
        append_block_text(&mut out, b, media_sentinel);
    }
    out
}

/// Append a single block's user-visible text to `out`. Tool-use and
/// tool-result blocks are handled at the message level (see
/// [`append_message`]); here they contribute nothing.
fn append_block_text(
    out: &mut Flat,
    block: &Block,
    media_sentinel: Option<&str>,
) {
    match block {
        Block::Text { text, .. } => out.content(text),
        // The wrappers are ours — framing, never neutralized; the body
        // is content.
        Block::Thought { thought, .. } => {
            out.framing("<think>");
            out.content(thought);
            out.framing("</think>");
        }
        // Sentinel emission: `<{R}:{source_hash_hex}>`. The caller
        // splits the render on this and resolves each occurrence
        // back to its image by source hash — no ordering contract
        // between renderer and collector, robust even to templates
        // that reorder messages. `render_with_env` guarantees the
        // sentinel is present whenever images are (the
        // `MediaUnsupported` check), so the `None` arm here is
        // unreachable rather than a silent drop.
        Block::Image { image, .. } => {
            if let Some(sentinel) = media_sentinel {
                out.framing(&media_marker(sentinel, &image_source_hash(image)));
            }
        }
        // Tool-use / tool-result blocks are handled at the message
        // level; redacted thoughts, documents, and the server-tool
        // block family contribute no template-visible text.
        _ => {}
    }
}

/// The body of a renderable *open* thought at the prompt's tail, if
/// there is one.
///
/// An open thought ([`OPEN_THOUGHT_SIGNATURE`]) is a reasoning block the
/// model never closed, or one a caller prefilled to steer the reasoning.
/// It renders by being withheld from the template and appended to the
/// finished generation prompt, so the only expressible position is the
/// **sole block of the trailing assistant message** — anything else
/// would need the template to lay out bytes around it, and the template
/// normalizes whitespace irreversibly.
///
/// Returning `None` for every other position is not a silent pass:
/// `Session` rejects those at ingest
/// ([`SessionError::UnrenderableOpenThought`]) before rendering is
/// reached. This function only decides *renderable or not*.
///
/// [`OPEN_THOUGHT_SIGNATURE`]: crate::prompt::OPEN_THOUGHT_SIGNATURE
/// [`SessionError::UnrenderableOpenThought`]: crate::SessionError::UnrenderableOpenThought
pub(crate) fn open_thought_tail(prompt: &Prompt) -> Option<&str> {
    let last = prompt.messages.last()?;
    if last.role != Role::Assistant {
        return None;
    }
    match last.content.0.as_slice() {
        [Block::Thought { thought, signature }]
            if signature.as_ref() == crate::prompt::OPEN_THOUGHT_SIGNATURE =>
        {
            Some(thought.as_ref())
        }
        _ => None,
    }
}

/// Does any block anywhere in `prompt` (system, messages, nested
/// tool-result content) carry an image?
pub(crate) fn prompt_has_images(prompt: &Prompt) -> bool {
    fn block_has_image(block: &Block) -> bool {
        match block {
            Block::Image { .. } => true,
            Block::ToolResult { result } => {
                result.content.0.iter().any(block_has_image)
            }
            _ => false,
        }
    }
    prompt
        .system
        .iter()
        .flat_map(|c| c.0.iter())
        .chain(prompt.messages.iter().flat_map(|m| m.content.0.iter()))
        .any(block_has_image)
}

// ===========================================================================
// Media sentinels
// ===========================================================================

/// SHA-256 identifying a [`Block::Image`]'s *source* bytes — the
/// base64 payload (or URL string), domain-separated by kind. This is
/// the correlation key between a sentinel occurrence in the render
/// and the block it came from; it is NOT the cache identity (that is
/// the RGB8 pixel hash, [`crate::Image::id`], computed after decode).
/// Two different encodings of the same pixels get different source
/// hashes but the same cache id — correct on both axes.
///
/// [`Block::Image`]: crate::Block
pub(crate) fn image_source_hash(
    image: &misanthropic::prompt::message::Image,
) -> [u8; 32] {
    use misanthropic::prompt::message::Image as ApiImage;
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    match image {
        ApiImage::Base64 { data, .. } => {
            hasher.update(b"b64:");
            hasher.update(data.as_bytes());
        }
        ApiImage::Url { url } => {
            // URLs are never fetched — this errors later at decode —
            // but the sentinel still needs a unique key to keep the
            // render structurally valid up to that typed error.
            hasher.update(b"url:");
            hasher.update(url.as_bytes());
        }
    }
    hasher.finalize().into()
}

/// The rendered form of one image: `<{sentinel}:{hex64}>`.
pub(crate) fn media_marker(sentinel: &str, source_hash: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(sentinel.len() + 68);
    s.push('<');
    s.push_str(sentinel);
    s.push(':');
    for b in source_hash {
        write!(s, "{b:02x}").expect("writing to String cannot fail");
    }
    s.push('>');
    s
}

/// One out-of-band marker in a render: an image, or a reserved piece
/// that content spelled (see [`LiteralNeutralizer`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderMarker {
    /// `<{sentinel}:{hex64}>` — the image's source hash (see
    /// [`image_source_hash`]).
    Media([u8; 32]),
    /// `<{sentinel}:t{id}>` — content that spelled the piece of
    /// special token `id`.
    Literal(Token),
}

/// A render split on its sentinel: `n + 1` text segments interleaved
/// with `n` markers, in render order. A render without markers comes
/// back as one segment.
///
/// Consumed by `Session` (the only splitter); dead-code-allowed for
/// builds without a session backend, where emission still exists but
/// nothing splits.
#[cfg_attr(
    not(any(
        feature = "llama-cpp",
        all(feature = "moeflux", target_os = "macos")
    )),
    allow(dead_code)
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SplitRender<'a> {
    /// Text between (around) markers;
    /// `segments.len() == markers.len() + 1`.
    pub segments: Vec<&'a str>,
    /// The marker after each segment but the last.
    pub markers: Vec<RenderMarker>,
}

/// Split `text` on `<{sentinel}:…>` markers: `{hex64}` for an image,
/// `t{id}` for a content literal.
///
/// The sentinel is per-call random and never surfaced, so content
/// cannot contain it — every occurrence is one of our own emissions.
/// A sentinel occurrence that does not parse as a full marker means
/// the template mangled it (truncation mid-marker, etc.): that is an
/// internal invariant violation, returned as `Err` with the byte
/// offset, never silently treated as content.
#[cfg_attr(
    not(any(
        feature = "llama-cpp",
        all(feature = "moeflux", target_os = "macos")
    )),
    allow(dead_code)
)]
pub(crate) fn split_render<'a>(
    text: &'a str,
    sentinel: &str,
) -> Result<SplitRender<'a>, usize> {
    let pattern = format!("<{sentinel}:");
    let mut segments = Vec::new();
    let mut markers = Vec::new();
    let mut rest = text;
    let mut base = 0usize;
    while let Some(at) = rest.find(&pattern) {
        let body_start = at + pattern.len();
        let (marker, end) = parse_marker(&rest[body_start..])
            .ok_or(base + at)
            .map(|(marker, len)| (marker, body_start + len))?;
        segments.push(&rest[..at]);
        markers.push(marker);
        rest = &rest[end..];
        base += end;
    }
    segments.push(rest);
    Ok(SplitRender { segments, markers })
}

/// Whether `split`'s text still holds `sentinel` in any letter case:
/// a marker a template filter transformed so [`split_render`] could not
/// see it (Gemma 4's cache-stable template applies `| upper` to schema
/// `type` values, turning a marker there into `<HEX:T123>`; `| e` would
/// escape its `<`). Safety holds — no special is emitted, the guard
/// still passes — but the marker reaches the model as text, and since
/// the sentinel is per call, that prefix misses the cache every call.
#[cfg_attr(
    not(any(
        feature = "llama-cpp",
        all(feature = "moeflux", target_os = "macos")
    )),
    allow(dead_code)
)]
pub(crate) fn has_transformed_marker(
    split: &SplitRender<'_>,
    sentinel: &str,
) -> bool {
    let needle = sentinel.as_bytes();
    let Some(&first) = needle.first() else {
        return false;
    };
    split.segments.iter().any(|segment| {
        let hay = segment.as_bytes();
        (0..hay.len().saturating_sub(needle.len() - 1)).any(|at| {
            hay[at].eq_ignore_ascii_case(&first)
                && hay[at..at + needle.len()].eq_ignore_ascii_case(needle)
        })
    })
}

/// Parse one marker body (what follows `<{sentinel}:`), returning the
/// marker and the byte length consumed including the closing `>`.
fn parse_marker(body: &str) -> Option<(RenderMarker, usize)> {
    let close = body.find('>')?;
    let inner = &body[..close];
    let marker = match inner.strip_prefix('t') {
        Some(id) => {
            // Canonical decimal only: `t007` is not a marker we wrote.
            let canonical = !id.is_empty()
                && id.bytes().all(|b| b.is_ascii_digit())
                && (id == "0" || !id.starts_with('0'));
            RenderMarker::Literal(canonical.then(|| id.parse().ok()).flatten()?)
        }
        None if inner.len() == 64
            && inner.bytes().all(|b| b.is_ascii_hexdigit()) =>
        {
            let mut hash = [0u8; 32];
            for (i, byte) in hash.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&inner[i * 2..i * 2 + 2], 16)
                    .expect("checked hexdigit above");
            }
            RenderMarker::Media(hash)
        }
        None => return None,
    };
    Some((marker, close + 1))
}

// ===========================================================================
// Content literals
// ===========================================================================

/// How many times each reserved piece was neutralized, by token id.
pub(crate) type LiteralCounts = BTreeMap<Token, usize>;

/// The reserved special-token pieces of a vocabulary, matched in prompt
/// *content* so they reach the model as spelled text instead of as the
/// control tokens they spell.
///
/// Every prepare path tokenizes the render with special-token parsing
/// on, which the chat framing needs: `<|im_start|>` in the template
/// must become one control token. Without this, the same piece in a
/// tool result, a user message or a tool description would become one
/// too, and the content could restructure the conversation. So each
/// piece found in a content surface is replaced before the template
/// sees it with an out-of-band marker, `<{sentinel}:t{id}>`, which
/// `Session` splits back out and tokenizes as text. The template's own
/// framing is never touched.
///
/// Matching is leftmost-longest (Aho-Corasick), so overlapping pieces
/// resolve the way the longer one reads and no piece survives
/// neutralization. `Session` builds one per model at construction and
/// injects it into every render through [`RenderOptions::literals`];
/// a caller rendering with a [`ChatTemplate`] directly can do the same
/// with [`RenderOptions::with_literals`].
#[derive(Clone)]
pub struct LiteralNeutralizer {
    /// `None` when there are no pieces.
    matcher: Option<aho_corasick::AhoCorasick>,
    /// Pattern index → `(token id, piece)`.
    pieces: Vec<(Token, String)>,
    /// Token id → pattern index.
    by_id: std::collections::HashMap<Token, usize>,
    /// Pattern indices in piece order, for [`Self::could_grow`].
    sorted: Vec<usize>,
    /// Special id → pattern index, for a special whose piece is
    /// reserved under another id; see [`Self::emitted_piece`].
    aliases: std::collections::HashMap<Token, usize>,
}

impl std::fmt::Debug for LiteralNeutralizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Counts, not pieces: the pieces are reserved bytes.
        f.debug_struct("LiteralNeutralizer")
            .field("pieces", &self.pieces.len())
            .finish()
    }
}

impl LiteralNeutralizer {
    /// Build from `(token id, piece)` pairs. Empty pieces are skipped,
    /// and a duplicate piece or id keeps its first entry.
    pub fn new<I, S>(pieces: I) -> Self
    where
        I: IntoIterator<Item = (Token, S)>,
        S: Into<String>,
    {
        let mut seen = std::collections::HashSet::new();
        let mut by_id = std::collections::HashMap::new();
        let pieces: Vec<(Token, String)> = pieces
            .into_iter()
            .map(|(id, piece)| (id, piece.into()))
            .filter(|(id, piece)| {
                !piece.is_empty()
                    && !by_id.contains_key(id)
                    && seen.insert(piece.clone())
                    && by_id.insert(*id, by_id.len()).is_none()
            })
            .collect();
        let matcher = (!pieces.is_empty()).then(|| {
            aho_corasick::AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(pieces.iter().map(|(_, piece)| piece))
                // Only a pattern set past the automaton's size limits
                // fails to build; a vocabulary's specials are a few
                // thousand short strings at most.
                .expect("reserved pieces fit an Aho-Corasick automaton")
        });
        let mut sorted: Vec<usize> = (0..pieces.len()).collect();
        sorted.sort_by(|&a, &b| pieces[a].1.cmp(&pieces[b].1));
        Self {
            matcher,
            pieces,
            by_id,
            sorted,
            aliases: std::collections::HashMap::new(),
        }
    }

    /// Also read each special in `specials` whose piece is reserved
    /// under another id as that piece when the model emits it
    /// ([`Self::emitted_piece`]). A vocabulary can hold two specials
    /// with one text; content spelling it tokenizes to one of them, the
    /// one reserved, but the model can emit either as framing.
    pub(crate) fn with_aliases<I, S>(mut self, specials: I) -> Self
    where
        I: IntoIterator<Item = (Token, S)>,
        S: AsRef<str>,
    {
        let by_piece: std::collections::HashMap<&str, usize> = self
            .pieces
            .iter()
            .enumerate()
            .map(|(i, (_, piece))| (piece.as_str(), i))
            .collect();
        let aliases = specials
            .into_iter()
            .filter(|(id, _)| !self.by_id.contains_key(id))
            .filter_map(|(id, piece)| {
                by_piece.get(piece.as_ref()).map(|&i| (id, i))
            })
            .collect();
        self.aliases = aliases;
        self
    }

    /// The piece a real emission of `id` reads as, when that piece is
    /// reserved: [`Self::piece`], or the piece of a special sharing its
    /// text (see [`Self::with_aliases`]). Emission provenance asks
    /// this, content neutralization never does.
    pub(crate) fn emitted_piece(&self, id: Token) -> Option<&str> {
        self.by_id
            .get(&id)
            .or_else(|| self.aliases.get(&id))
            .map(|&i| self.pieces[i].1.as_str())
    }

    /// Whether there is nothing to neutralize.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Number of reserved pieces.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether `id` is one of the reserved tokens.
    pub fn contains(&self, id: Token) -> bool {
        self.by_id.contains_key(&id)
    }

    /// The piece of reserved token `id`.
    pub fn piece(&self, id: Token) -> Option<&str> {
        self.by_id.get(&id).map(|&i| self.pieces[i].1.as_str())
    }

    /// The reserved token ids.
    pub fn ids(&self) -> impl Iterator<Item = Token> + '_ {
        self.pieces.iter().map(|(id, _)| *id)
    }

    /// Every reserved piece in `text`, leftmost-longest, as
    /// `(byte range, token id)`.
    pub fn find_iter<'t>(
        &'t self,
        text: &'t str,
    ) -> impl Iterator<Item = (std::ops::Range<usize>, Token)> + 't {
        self.matcher.iter().flat_map(move |ac| {
            ac.find_iter(text).map(move |m| {
                (m.start()..m.end(), self.pieces[m.pattern().as_usize()].0)
            })
        })
    }

    /// Whether `tail` is a proper prefix of a reserved piece: text that
    /// more bytes could still turn into one. The pieces starting with
    /// `tail` sort contiguously from where `tail` itself would, and at
    /// most the first of them equals it.
    pub(crate) fn could_grow(&self, tail: &str) -> bool {
        let piece = |i: usize| self.pieces[i].1.as_str();
        let at = self.sorted.partition_point(|&i| piece(i) < tail);
        self.sorted[at..]
            .iter()
            .take(2)
            .any(|&i| piece(i).len() > tail.len() && piece(i).starts_with(tail))
    }

    /// Replace every reserved piece in `text` with its marker under
    /// `sentinel`, adding each replacement to `counts`. Borrows when
    /// there is nothing to replace, so clean text is untouched.
    fn neutralize<'t>(
        &self,
        text: &'t str,
        sentinel: &str,
        mut counts: Option<&mut LiteralCounts>,
    ) -> Cow<'t, str> {
        let mut out: Option<String> = None;
        let mut last = 0;
        for (range, id) in self.find_iter(text) {
            let buf =
                out.get_or_insert_with(|| String::with_capacity(text.len()));
            buf.push_str(&text[last..range.start]);
            buf.push_str(&literal_marker(sentinel, id));
            last = range.end;
            if let Some(counts) = counts.as_deref_mut() {
                *counts.entry(id).or_default() += 1;
            }
        }
        match out {
            None => Cow::Borrowed(text),
            Some(mut buf) => {
                buf.push_str(&text[last..]);
                Cow::Owned(buf)
            }
        }
    }
}

/// The rendered form of one content literal: `<{sentinel}:t{id}>`.
pub(crate) fn literal_marker(sentinel: &str, id: Token) -> String {
    format!("<{sentinel}:t{id}>")
}

/// Per-render content-literal configuration: the vocabulary's
/// [`LiteralNeutralizer`] and the sentinel its markers render under.
///
/// The sentinel must be something no content can contain — `Session`
/// draws a fresh random one per call, the same way it does for images
/// (see [`RenderOptions::media_sentinel`]), and shares it with the
/// image markers when both are present.
#[derive(Clone, Debug)]
pub struct Literals {
    sentinel: String,
    neutralizer: Arc<LiteralNeutralizer>,
}

impl Literals {
    /// Neutralize with `neutralizer`, marking under `sentinel`.
    pub fn new<S>(sentinel: S, neutralizer: Arc<LiteralNeutralizer>) -> Self
    where
        S: Into<String>,
    {
        Self {
            sentinel: sentinel.into(),
            neutralizer,
        }
    }

    /// The marker sentinel.
    pub fn sentinel(&self) -> &str {
        &self.sentinel
    }

    /// The neutralizer.
    pub fn neutralizer(&self) -> &LiteralNeutralizer {
        &self.neutralizer
    }
}

/// Anthropic's pattern for a tool name, `^[a-zA-Z0-9_-]{1,64}$`. The
/// dialect parser holds model-emitted names to it too, so a call the
/// model names badly degrades to text instead of seating a name the
/// next ingest would reject. (It cannot see the vocabulary, so a name
/// of these characters that spells a reserved piece would still seat;
/// no fleet vocabulary has such a piece — theirs are bracketed tags.)
pub(crate) fn is_tool_name(s: &str) -> bool {
    (1..=64).contains(&s.len()) && is_identifier(s)
}

/// Anthropic's pattern for a tool-use id, `^[a-zA-Z0-9_-]+$` — no
/// length cap, since our own ids (`call_{n}_{name}`) outgrow 64 bytes
/// for long tool names.
fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// The content surfaces of one render: where every string the template
/// can see is neutralized, counted and, for identifiers, validated.
struct Surfaces<'a> {
    media_sentinel: Option<&'a str>,
    literals: Option<&'a Literals>,
    /// Replacements in the counted surfaces — one count per content
    /// string, so `Session` can check it against its own scan.
    counts: std::cell::RefCell<LiteralCounts>,
}

impl<'a> Surfaces<'a> {
    fn new(opts: &'a RenderOptions) -> Self {
        Self {
            media_sentinel: opts.media_sentinel.as_deref(),
            literals: opts.literals.as_ref(),
            counts: Default::default(),
        }
    }

    /// Neutralize content `text`; `counted` adds its replacements to
    /// [`Self::counts`]. A string the template sees twice (the merged
    /// assistant `content` and its `content_pre`/`content_post`
    /// halves) is counted once.
    fn text<'t>(&self, text: &'t str, counted: bool) -> Cow<'t, str> {
        let Some(lit) = self.literals else {
            return Cow::Borrowed(text);
        };
        let mut counts = self.counts.borrow_mut();
        lit.neutralizer.neutralize(
            text,
            &lit.sentinel,
            counted.then_some(&mut *counts),
        )
    }

    /// Neutralize every key and string leaf of `value`.
    fn value(&self, value: &serde_json::Value, counted: bool) -> JinjaValue {
        use serde_json::Value;
        fn walk(s: &Surfaces<'_>, v: &Value, counted: bool) -> Value {
            match v {
                Value::String(t) => {
                    Value::String(s.text(t, counted).into_owned())
                }
                Value::Array(items) => Value::Array(
                    items.iter().map(|i| walk(s, i, counted)).collect(),
                ),
                Value::Object(map) => Value::Object(
                    map.iter()
                        .map(|(k, v)| {
                            (
                                s.text(k, counted).into_owned(),
                                walk(s, v, counted),
                            )
                        })
                        .collect(),
                ),
                other => other.clone(),
            }
        }
        match self.literals {
            None => JinjaValue::from_serialize(value),
            Some(_) => JinjaValue::from_serialize(walk(self, value, counted)),
        }
    }

    /// Reject an identifier the model would see verbatim: tool names
    /// and tool-use ids are not neutralized (the grammar and parser key
    /// on them), so they must hold Anthropic's character set and no
    /// reserved piece. Only checked when neutralizing.
    fn identifier(
        &self,
        value: &str,
        valid: fn(&str) -> bool,
        what: impl FnOnce() -> String,
    ) -> Result<(), ChatTemplateError> {
        let Some(lit) = self.literals else {
            return Ok(());
        };
        if valid(value) && lit.neutralizer.find_iter(value).next().is_none() {
            Ok(())
        } else {
            Err(ChatTemplateError::InvalidIdentifier { what: what() })
        }
    }
}

/// Template-visible text assembled from content and our own framing
/// (thought wrappers, image markers). Adjacent content joins into one
/// run before it is neutralized, so a piece split across two blocks
/// the template sees concatenated is still caught; framing is never
/// neutralized.
#[derive(Default, Clone)]
struct Flat {
    /// `(is_content, text)`, adjacent content merged.
    spans: Vec<(bool, String)>,
}

impl Flat {
    fn content(&mut self, text: &str) {
        match self.spans.last_mut() {
            Some((true, run)) => run.push_str(text),
            _ => self.spans.push((true, text.to_string())),
        }
    }

    fn framing(&mut self, text: &str) {
        self.spans.push((false, text.to_string()));
    }

    /// `self` followed by `other`, content runs joined at the seam.
    fn join(&self, other: &Flat) -> Flat {
        let mut out = self.clone();
        for (is_content, text) in &other.spans {
            match is_content {
                true => out.content(text),
                false => out.framing(text),
            }
        }
        out
    }

    fn is_empty(&self) -> bool {
        self.spans.iter().all(|(_, text)| text.is_empty())
    }

    fn render(&self, surfaces: &Surfaces<'_>, counted: bool) -> String {
        self.spans
            .iter()
            .map(|(is_content, text)| match is_content {
                true => surfaces.text(text, counted),
                false => Cow::Borrowed(text.as_str()),
            })
            .collect()
    }
}

// ===========================================================================
// Jinja-side helpers
// ===========================================================================

/// `tojson` without Jinja's HTML-safety escaping.
///
/// Jinja2's `tojson` is `htmlsafe_json_dumps`, which escapes `'`, `&`,
/// `<`, and `>` to `'`, `&`, `<`, `>` — a defense
/// against JSON embedded in `<script>` blocks, inherited from Jinja's
/// web origins. minijinja matches that faithfully (verified byte-for-byte
/// against the Python reference renderer in
/// `tests/fixtures/render_jinja.py`), so this is *correct* Jinja
/// behavior, not drift.
///
/// It is nonetheless wrong for us, in two ways:
///
/// 1. **Round-trip.** A model emits a literal `'`; the template renders
///    it back as `'`. The re-render is then not byte-identical to
///    what the KV holds, the auto-tip is discarded, and prefix reuse
///    collapses to the last `cache_control` breakpoint — measured at
///    4705 tokens lost per turn against cogito-32b (#85). Byte-stable
///    re-rendering is the invariant the whole prefix cache rests on.
/// 2. **Fidelity.** The escaped form is what the model *reads back* as
///    its own prior turn, so its history diverges from what it wrote.
///    The same applies to tool descriptions in the `<tools>` block,
///    which also route through this filter.
///
/// Constraining generation to emit the escaped form instead was measured
/// and rejected: `'` and friends tokenize to exactly 5 tokens with
/// no merges, taking a realistic prose argument from 47 to 107 tokens
/// (+128%) — generated tokens, on the most common punctuation in
/// English.
///
/// There is no HTML anywhere in a chat prompt, so nothing is lost. The
/// cost is that our rendered bytes differ from other Jinja-based stacks
/// in exactly these four characters.
/// The `indent=N` kwarg is honored because real templates pass it —
/// Llama 3.1 renders its tool listing with `t | tojson(indent=4)`, and
/// dropping the kwarg is a render-time "too many arguments" error, not
/// a silent formatting change.
fn tojson_unescaped(
    value: JinjaValue,
    kwargs: Kwargs,
) -> Result<JinjaValue, JinjaError> {
    let indent: Option<usize> = kwargs.get("indent").ok();
    // Rejects any kwarg we don't model rather than ignoring it: a
    // silently-dropped formatting argument would change rendered bytes
    // without anyone noticing, which is the class of bug this whole
    // filter exists to close.
    kwargs.assert_all_used()?;

    let json = match indent {
        Some(width) => {
            let pad = vec![b' '; width];
            let mut buf = Vec::new();
            let mut ser = serde_json::Serializer::with_formatter(
                &mut buf,
                serde_json::ser::PrettyFormatter::with_indent(&pad),
            );
            value.serialize(&mut ser).map_err(|e| json_err(&e))?;
            String::from_utf8(buf).map_err(|e| {
                JinjaError::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("tojson: {e}"),
                )
            })?
        }
        None => serde_json::to_string(&value).map_err(|e| json_err(&e))?,
    };

    // Safe-string so an autoescaping template can't re-escape the JSON
    // we just deliberately left unescaped.
    Ok(JinjaValue::from_safe_string(json))
}

fn json_err(e: &serde_json::Error) -> JinjaError {
    JinjaError::new(
        minijinja::ErrorKind::InvalidOperation,
        format!("tojson: {e}"),
    )
}

/// Python `json.dumps` default spacing (`": "`, `", "`), unescaped —
/// the serializer owned chat templates render tool-call arguments
/// through.
///
/// Stock templates use `tojson` (compact); tool-tuned models emit
/// `json.dumps` spacing (cogito measured greedy-unforced,
/// `tests/probe_unforced_habit.rs`), so a stock re-render never
/// byte-matches the emission and #85's canonical form had to pin the
/// model *off* its habit. An owned template renders through this
/// filter instead; the dialect analyzer measures the resulting
/// spacing ([`crate::JsonSpacing::Spaced`]) and the grammar and
/// `render_reference` follow — three views, one byte string (#88).
///
/// Escaping matches [`tojson_unescaped`] (serde_json's), so output is
/// byte-identical to `json.dumps(value, ensure_ascii=False)`. No
/// kwargs — an owned template has no business asking for indent.
fn json_dumps_filter(
    value: JinjaValue,
    kwargs: Kwargs,
) -> Result<JinjaValue, JinjaError> {
    kwargs.assert_all_used()?;
    let json = crate::json_canon::to_spaced_string(&value)
        .map_err(|e| json_err(&e))?;
    // Safe-string so an autoescaping template can't re-escape the JSON
    // we just deliberately left unescaped.
    Ok(JinjaValue::from_safe_string(json))
}

/// Register drama_llama's template filters on `env`.
///
/// Every environment that renders chat templates — [`ChatTemplate`]'s
/// strict and permissive envs and the dialect analyzer's probe env —
/// registers the same set, or a template could render under one and
/// fail (or render *different bytes*) under another. The analyzer
/// measures renders that the real path must then reproduce
/// byte-for-byte, so the environments may not diverge.
pub(crate) fn register_template_filters(env: &mut Environment<'_>) {
    // Overrides minijinja's builtin. See `tojson_unescaped`.
    env.add_filter("tojson", tojson_unescaped);
    env.add_filter("json_dumps", json_dumps_filter);
}

/// HF templates commonly call `raise_exception("msg")` to reject invalid
/// input. Surface that as a render-time error instead of panicking.
fn raise_exception(msg: Cow<'_, str>) -> Result<JinjaValue, JinjaError> {
    Err(JinjaError::new(
        minijinja::ErrorKind::InvalidOperation,
        format!("chat template raised: {msg}"),
    ))
}

/// Permissive counterpart to [`raise_exception`] for partial renders.
///
/// Drops the raise on the floor (returns an empty string) so a template
/// that gates the full render on invariants violated by a truncated
/// prompt still produces a usable byte sequence for cache-breakpoint
/// hashing. Wired into [`ChatTemplate::env_permissive`] in place of
/// [`raise_exception`]; full renders continue to surface raises as
/// errors via [`ChatTemplate::env`].
///
/// Suppressed messages are logged at `debug` so they remain auditable
/// when chasing template-specific cache regressions (the partial-render
/// warn at the [`ChatTemplate::render_with_breakpoints`] call site
/// fires on render errors, not on suppressed raises — those are by
/// construction invisible there).
fn raise_exception_noop(_msg: Cow<'_, str>) -> String {
    #[cfg(feature = "axum")]
    tracing::debug!(
        target: "drama_llama::chat_template",
        suppressed = %_msg,
        "raise_exception suppressed in permissive env (partial render)",
    );
    String::new()
}

/// Minimal `strftime_now` that returns current UTC time formatted via the
/// `time` crate's strftime-style format string. Enough for templates that
/// stamp `"%d %b %Y"` or `"%Y-%m-%d"`.
fn strftime_now(fmt: Cow<'_, str>) -> String {
    format_strftime_subset(&fmt, current_unix_secs())
}

fn current_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Handle the specifiers HF chat templates actually use: `%Y`, `%m`, `%d`,
/// `%b`, `%B`, `%H`, `%M`, `%S`. Passes other characters through.
fn format_strftime_subset(fmt: &str, unix_secs: i64) -> String {
    let (y, mo, d, h, mi, s) = civil_from_unix(unix_secs);
    let mut out = String::with_capacity(fmt.len() + 8);
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&format!("{y:04}")),
            Some('m') => out.push_str(&format!("{mo:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('H') => out.push_str(&format!("{h:02}")),
            Some('M') => out.push_str(&format!("{mi:02}")),
            Some('S') => out.push_str(&format!("{s:02}")),
            Some('b') => {
                out.push_str(MONTH_ABBR[(mo - 1).clamp(0, 11) as usize])
            }
            Some('B') => {
                out.push_str(MONTH_FULL[(mo - 1).clamp(0, 11) as usize])
            }
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

const MONTH_ABBR: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
    "Nov", "Dec",
];
const MONTH_FULL: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Convert Unix seconds (UTC) into (year, month, day, hour, min, sec).
/// Proleptic Gregorian calendar; handles post-1970 dates.
fn civil_from_unix(secs: i64) -> (i32, i32, i32, i32, i32, i32) {
    // Split into days and time-of-day.
    let days = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400);
    let h = time_of_day / 3600;
    let mi = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    // Howard Hinnant's civil_from_days algorithm, epoch 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y as i32, mo as i32, d as i32, h as i32, mi as i32, s as i32)
}

// ===========================================================================
// Errors
// ===========================================================================

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ChatTemplateError {
    #[error("model has no `tokenizer.chat_template` metadata")]
    NoTemplate,
    #[error("chat template error: {0}")]
    Jinja(String),
    /// The prompt contains [`Block::Image`]s but the caller provided
    /// no [`RenderOptions::media_sentinel`] — there is nowhere for
    /// the images to go. `Session` sets the sentinel when (and only
    /// when) it can actually consume images; anything else erroring
    /// here is what kills the historical silent image drop.
    #[error(
        "prompt contains image blocks but no media sentinel is \
         configured; this renderer cannot represent images"
    )]
    MediaUnsupported,
    /// The prompt's tail is an *open* thought
    /// ([`OPEN_THOUGHT_SIGNATURE`]) but no reasoning open marker was
    /// configured ([`RenderOptions::reasoning_start`]), so there is
    /// nothing to resume the reasoning block with. Rendering the body
    /// without its marker would hand the model prose where it expects
    /// to be mid-thought; dropping the body would lose it silently.
    /// Neither is acceptable, so this is typed.
    ///
    /// `Session` always configures the marker from the analyzed
    /// dialect, so this only reaches direct [`ChatTemplate`] callers.
    ///
    /// [`OPEN_THOUGHT_SIGNATURE`]: crate::prompt::OPEN_THOUGHT_SIGNATURE
    #[error(
        "prompt ends with an open thought but no reasoning open marker \
         is configured; set `RenderOptions::with_reasoning_start`"
    )]
    OpenThoughtUnsupported,
    /// A tool name or tool-use id the model would read verbatim does
    /// not match Anthropic's pattern (`^[a-zA-Z0-9_-]{1,64}$` for a
    /// name, `^[a-zA-Z0-9_-]+$` for an id), or contains a reserved
    /// special-token piece. Content is neutralized (see
    /// [`LiteralNeutralizer`]); identifiers are rejected instead,
    /// because the tool-call grammar and parser key on them. Only
    /// checked when [`RenderOptions::literals`] is set. The value is
    /// withheld from the message, which is relayed to clients.
    #[error(
        "{what} is not a valid tool identifier (letters, digits, `_` \
         and `-` only; names at most 64 bytes)"
    )]
    InvalidIdentifier {
        /// Where the identifier sits in the prompt.
        what: String,
    },
}

impl ChatTemplateError {
    fn from_jinja(err: JinjaError) -> Self {
        Self::Jinja(format!("{err:#}"))
    }
}

static_assertions::assert_impl_all!(ChatTemplateError: Send, Sync);

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    /// A Llama-3-style template, simplified. Verifies the basic control
    /// flow: BOS, system message, role headers, EOT markers, and the
    /// optional generation prompt.
    const LLAMA3_LIKE: &str = r#"{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>

{{ m['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}"#;

    /// Minimal Qwen3-shape template that reproduces the two raises which
    /// motivate [`ChatTemplate::env_permissive`]:
    ///
    /// 1. `{% if not messages %}{{ raise_exception('No messages provided.') }}{% endif %}`
    ///    — fires on [`PromptBreakpoint::AfterTools`] where `messages` is empty.
    /// 2. The `multi_step_tool` walk: scans `messages[::-1]` for a
    ///    user-role message whose content isn't a `<tool_response>...`
    ///    wrapper, raises `No user query found in messages.` if none
    ///    found. Fires on [`PromptBreakpoint::AfterSystem`] where the only
    ///    message present is the synthesized system message.
    ///
    /// Trimmed from the real Qwen3.6-35B-A3B `tokenizer.chat_template`
    /// to keep the test data readable; the two raise sites are
    /// byte-for-byte the same Jinja as upstream so the permissive-env
    /// behavior is exercised by the same code paths the production
    /// template hits.
    const QWEN3_LIKE: &str = r#"{%- if not messages -%}
{{- raise_exception('No messages provided.') -}}
{%- endif -%}
{%- if tools and tools is iterable and tools is not mapping -%}
<|im_start|>system
# Tools
{% for tool in tools %}{{ tool | tojson }}
{% endfor -%}
{%- if messages[0].role == 'system' -%}

{{ messages[0].content }}
{%- endif -%}
<|im_end|>
{%- else -%}
{%- if messages[0].role == 'system' -%}
<|im_start|>system
{{ messages[0].content }}<|im_end|>
{%- endif -%}
{%- endif -%}
{%- set ns = namespace(multi_step_tool=true) -%}
{%- for message in messages[::-1] -%}
{%- if ns.multi_step_tool and message.role == "user" -%}
{%- set content = message.content | trim -%}
{%- if not (content.startswith('<tool_response>') and content.endswith('</tool_response>')) -%}
{%- set ns.multi_step_tool = false -%}
{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- if ns.multi_step_tool -%}
{{- raise_exception('No user query found in messages.') -}}
{%- endif -%}
{%- for message in messages -%}
{%- if message.role == "user" -%}
<|im_start|>user
{{ message.content }}<|im_end|>
{%- elif message.role == "assistant" -%}
<|im_start|>assistant
{{ message.content }}<|im_end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- endif -%}"#;

    fn qwen3_like_tmpl() -> ChatTemplate {
        ChatTemplate::from_source(
            QWEN3_LIKE.to_owned(),
            "".to_owned(),
            "".to_owned(),
        )
        .expect("Qwen3-like template should compile")
    }

    fn simple_prompt() -> Prompt {
        Prompt::default()
            .system("You are helpful.")
            .add_message((Role::User, "Hi!"))
            .unwrap()
            .add_message((Role::Assistant, "Hello!"))
            .unwrap()
            .add_message((Role::User, "What is 2+2?"))
            .unwrap()
    }

    fn tmpl() -> ChatTemplate {
        ChatTemplate::from_source(
            LLAMA3_LIKE.to_owned(),
            "<|begin_of_text|>".to_owned(),
            "<|end_of_text|>".to_owned(),
        )
        .expect("template should compile")
    }

    /// #60: the wire envelope serializes in `json!`-literal order —
    /// `type` before `function`; inside, `name`, `description`,
    /// `parameters` — the shape ollama's Go runtime produced in the
    /// training data. Rides on `serde_json/preserve_order`.
    #[test]
    fn tool_wire_value_is_insertion_ordered() {
        let tool = crate::Tool::builder("t")
            .description("d")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {},
            }))
            .build()
            .expect("valid tool");
        let wire =
            serde_json::to_string(&tool_wire_value(&tool)).expect("serialize");
        assert!(
            wire.starts_with(r#"{"type":"function","function":{"name":"t","description":"d","parameters":"#),
            "wire envelope order drifted: {wire}"
        );
    }

    #[test]
    fn renders_full_turn() {
        let out = tmpl().render(&simple_prompt(), true).unwrap();
        assert!(out.starts_with("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"));
        assert!(out.contains(
            "<|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|>"
        ));
        assert!(out.contains(
            "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>"
        ));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|>"));
        assert!(
            out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n")
        );
    }

    #[test]
    fn skips_generation_prompt_when_false() {
        let out = tmpl().render(&simple_prompt(), false).unwrap();
        assert!(
            !out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n")
        );
    }

    #[test]
    fn omits_system_when_none() {
        let p = Prompt::default().add_message((Role::User, "hi")).unwrap();
        let out = tmpl().render(&p, false).unwrap();
        assert!(!out.contains("<|start_header_id|>system"));
        assert!(out.contains(
            "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"
        ));
    }

    fn literals(sentinel: &str) -> Literals {
        Literals::new(
            sentinel,
            Arc::new(LiteralNeutralizer::new([
                (1, "<|eot_id|>"),
                (2, "<think>"),
                (3, "</think>"),
                // Duplicates keep the first entry.
                (4, "<think>"),
                (2, "<other>"),
            ])),
        )
    }

    /// Content literals: clean renders are byte-identical, a thought's
    /// body is neutralized inside our own (kept) wrappers, a piece
    /// split across blocks is caught once joined, and the counts cover
    /// what was replaced.
    #[test]
    fn content_literals_neutralize_content_not_framing() {
        let sentinel = "0123456789abcdef0123456789abcdef";
        let opts = RenderOptions::default().with_literals(literals(sentinel));
        assert_eq!(opts.literals.as_ref().unwrap().neutralizer().len(), 3);

        let clean = simple_prompt();
        assert_eq!(
            tmpl().render_counted(&clean, &opts).unwrap(),
            (tmpl().render(&clean, false).unwrap(), LiteralCounts::new()),
            "a clean prompt renders byte-identically",
        );

        let p = Prompt {
            messages: vec![Message {
                role: Role::Assistant,
                content: Content(vec![
                    Block::Thought {
                        thought: "no <think> here".into(),
                        signature: "sig".into(),
                    },
                    Block::text("split <|eot"),
                    Block::text("_id|> joined"),
                ]),
            }],
            ..Default::default()
        };
        let (out, counts) = tmpl().render_counted(&p, &opts).unwrap();
        let think = literal_marker(sentinel, 2);
        let eot = literal_marker(sentinel, 1);
        assert!(
            out.contains(&format!(
                "<think>no {think} here</think>split {eot} joined"
            )),
            "{out}",
        );
        assert_eq!(counts, [(1, 1), (2, 1)].into_iter().collect());
        // Every marker splits back out.
        let split = split_render(&out, sentinel).unwrap();
        assert_eq!(
            split.markers,
            [RenderMarker::Literal(2), RenderMarker::Literal(1)]
        );
    }

    #[test]
    fn thought_block_wraps_with_think_tags() {
        let mut content = Content(vec![
            Block::Thought {
                thought: "I should be concise.".into(),
                signature: "sig".into(),
            },
            Block::text("Hello!"),
        ]);
        let _ = &mut content; // appease borrow lint if any
        let p = Prompt {
            system: None,
            messages: vec![Message {
                role: Role::Assistant,
                content,
            }],
            tools: None,
            ..Default::default()
        };
        let out = tmpl().render(&p, false).unwrap();
        assert!(out.contains("<think>I should be concise.</think>Hello!"));
    }

    /// An assistant message's `chunks` are its blocks in emission order:
    /// each thought its own chunk (back-to-back ones included), prose
    /// runs between them, and one `tool_calls` chunk where the first
    /// call sat, whatever the reasoning re-ingest convention.
    #[test]
    fn assistant_chunks_keep_emission_order() {
        use crate::prompt::ToolUse;
        let src = "{% for m in messages %}{% for c in m.chunks or [] %}\
                   {{ c.type }}:{{ c.text or c.thinking or '' }}|\
                   {% endfor %}{% endfor %}"
            .to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let thought = |t: &'static str| Block::Thought {
            thought: t.into(),
            signature: "".into(),
        };
        let call = |id: &'static str| Block::ToolUse {
            call: ToolUse::new("get_weather", serde_json::json!({}))
                .with_id(id),
        };
        let p = Prompt {
            messages: vec![
                Message {
                    role: Role::User,
                    content: Content::text("Hi"),
                },
                Message {
                    role: Role::Assistant,
                    content: Content(vec![
                        thought("A."),
                        thought("B."),
                        "Checking.".into(),
                        thought("C."),
                        call("call1"),
                        call("call2"),
                        "Done.".into(),
                    ]),
                },
            ],
            ..Default::default()
        };
        for reingest in [
            crate::dialect::ReasoningReingest::Field,
            crate::dialect::ReasoningReingest::Thinking,
            crate::dialect::ReasoningReingest::InlineThink,
        ] {
            let opts = RenderOptions::default().with_thought_reingest(reingest);
            assert_eq!(
                t.render_with(&p, &opts).unwrap(),
                "thinking:A.|thinking:B.|text:Checking.|thinking:C.|\
                 tool_calls:|text:Done.|",
                "{reingest:?}"
            );
        }
    }

    #[test]
    fn raise_exception_surfaces_as_error() {
        let src = r#"{{ raise_exception("boom") }}"#.to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let err = t.render(&Prompt::default(), false).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("boom"), "error must surface message: {msg}");
    }

    #[test]
    fn strftime_now_renders_current_year() {
        let src = r#"{{ strftime_now("%Y-%m-%d") }}"#.to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let out = t.render(&Prompt::default(), false).unwrap();
        // Loose assertion: must look like YYYY-MM-DD with current millennium.
        assert!(out.starts_with("20"), "expected 20YY-MM-DD, got {out}");
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn multi_part_text_flattens() {
        let p = Prompt {
            system: None,
            messages: vec![Message {
                role: Role::User,
                content: Content(vec![
                    Block::text("Hello "),
                    Block::text("world"),
                ]),
            }],
            tools: None,
            ..Default::default()
        };
        let out = tmpl().render(&p, false).unwrap();
        assert!(out.contains("Hello world"));
    }

    #[cfg(feature = "llama-cpp")]
    /// End-to-end: render with the real chat template embedded in
    /// `models/model.gguf` and sanity-check the output. Assertions are
    /// template-agnostic: templates differ on whether BOS appears in
    /// the rendered text (Llama 3.1 emits it; Qwen's ChatML does not),
    /// but every chat template must render the message contents and an
    /// assistant generation header.
    #[test]
    #[ignore = "requires model"]
    fn chat_template_from_real_model() {
        use std::path::PathBuf;

        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = crate::LlamaCppEngine::from_path(path).unwrap();
        let tmpl = ChatTemplate::from_model(&engine.model)
            .expect("model should have a chat template");
        let out = tmpl
            .render(&simple_prompt(), true)
            .expect("real template should render");
        assert!(
            out.contains("You are helpful."),
            "rendered prompt should contain the system text: {out}"
        );
        assert!(
            out.contains("Hi!"),
            "rendered prompt should contain the user text: {out}"
        );
        // And it should end with the assistant generation header.
        assert!(
            out.contains("assistant"),
            "rendered prompt missing assistant header"
        );
    }

    #[cfg(feature = "llama-cpp")]
    /// Exercise the tools branch of the real template in
    /// `models/model.gguf`: pass a single function definition, render,
    /// and assert the rendered prompt includes the function name and
    /// schema. Date and ipython-header checks only apply when the
    /// template itself supports them (Llama 3.1 does; Qwen's ChatML
    /// has neither).
    #[test]
    #[ignore = "requires model"]
    fn chat_template_renders_tools_against_real_model() {
        use crate::Tool;
        use serde_json::json;
        use std::path::PathBuf;

        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = crate::LlamaCppEngine::from_path(path).unwrap();
        let tmpl = ChatTemplate::from_model(&engine.model).unwrap();

        let tool = Tool::builder("get_weather")
            .description("Look up the current weather in a city.")
            .schema(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }))
            .build()
            .expect("valid test tool");

        let prompt = Prompt {
            system: Some(Content::text("You are helpful.")),
            messages: vec![Message {
                role: Role::User,
                content: Content::text("What's the weather in Paris?"),
            }],
            tools: Some(vec![tool.into()]),
            ..Default::default()
        };

        let opts = RenderOptions::default()
            .with_generation_prompt(true)
            .with_date("17 Apr 2026");
        let out = tmpl.render_with(&prompt, &opts).unwrap();

        assert!(
            out.contains("get_weather"),
            "tools branch must mention function name. output:\n{out}"
        );
        assert!(
            out.contains("\"city\""),
            "schema should appear in rendered output. output:\n{out}"
        );
        // Only templates that consume these variables can be expected
        // to render them.
        let source = engine
            .model
            .get_meta("tokenizer.chat_template")
            .expect("model has no tokenizer.chat_template");
        if source.contains("date_string") {
            assert!(
                out.contains("17 Apr 2026"),
                "date_string should appear when provided. output:\n{out}"
            );
        }
        if source.contains("ipython") {
            assert!(
                out.contains("Environment: ipython"),
                "system header should include ipython env when tools present"
            );
        }
    }

    #[cfg(feature = "llama-cpp")]
    /// Render an assistant tool-call turn, confirming the tool-call
    /// branch renders the call at all — name, argument key, and
    /// argument value must survive into the transcript. The envelope
    /// is deliberately unasserted: it varies by template family
    /// (Llama 3.1 emits JSON with `"parameters"`, Qwen3 JSON with
    /// `"arguments"`, Qwen3.6 an XML-ish
    /// `<function=name><parameter=key>value` shape).
    #[test]
    #[ignore = "requires model"]
    fn chat_template_renders_assistant_tool_call_against_real_model() {
        use crate::prompt::ToolUse;

        use serde_json::json;
        use std::{borrow::Cow, path::PathBuf};

        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = crate::LlamaCppEngine::from_path(path).unwrap();
        let tmpl = ChatTemplate::from_model(&engine.model).unwrap();

        let call = ToolUse {
            id: Cow::Borrowed("call_1"),
            name: Cow::Borrowed("get_weather"),
            input: json!({"city": "Paris"}),
            cache_control: None,
            caller: None,
        };

        let prompt = Prompt {
            system: None,
            messages: vec![
                Message {
                    role: Role::User,
                    content: Content::text("Call get_weather for Paris."),
                },
                Message {
                    role: Role::Assistant,
                    content: Content(vec![Block::ToolUse { call }]),
                },
            ],
            tools: None,
            ..Default::default()
        };

        let out = tmpl.render(&prompt, false).unwrap();
        assert!(
            out.contains("get_weather"),
            "tool-call branch must include the function name. output:\n{out}"
        );
        assert!(
            out.contains("city"),
            "tool-call branch must include the argument key. output:\n{out}"
        );
        assert!(
            out.contains("Paris"),
            "tool-call branch must include the argument value. output:\n{out}"
        );
    }

    #[cfg(feature = "llama-cpp")]
    /// Diagnostic: read a file from
    /// `DRAMA_LLAMA_TOKENIZE_FILE` and print (count, first 20 ids, last 20
    /// ids) of drama_llama's tokenization. Lets us compare against
    /// ollama's token count on the same bytes to confirm tokenizer
    /// parity. Run with
    /// `DRAMA_LLAMA_TOKENIZE_FILE=/tmp/dl_turn2.txt cargo test
    /// dump_tokenize -- --ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic helper"]
    fn dump_tokenize() {
        use std::path::PathBuf;
        let Some(file) = std::env::var_os("DRAMA_LLAMA_TOKENIZE_FILE") else {
            // Helper, not a test: skip cleanly in `--include-ignored`
            // sweeps instead of polluting them with a failure.
            eprintln!("skipped: set DRAMA_LLAMA_TOKENIZE_FILE to the input path to use this helper");
            return;
        };
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("read {file:?}: {e}"));
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = crate::LlamaCppEngine::from_path(path).unwrap();
        let tokens_nospec = engine.model.tokenize(&text, false);
        let tokens_spec = engine.model.tokenize(&text, true);
        println!("parse_special=false: {} tokens", tokens_nospec.len());
        println!("parse_special=true:  {} tokens", tokens_spec.len());
        let head_spec: Vec<_> = tokens_spec.iter().take(20).collect();
        println!("(spec) first 20: {head_spec:?}");
    }

    #[cfg(feature = "llama-cpp")]
    /// One-off dump: render the strawberry turn-1 Prompt through our
    /// ChatTemplate and write the bytes to
    /// `DRAMA_LLAMA_DUMP_OUTPUT`. For diffing against the Python
    /// jinja2 cross-check renderer.
    #[test]
    #[ignore = "fixture helper"]
    fn dump_strawberry_turn_1_output() {
        use crate::Tool;
        use serde_json::json;
        use std::path::PathBuf;
        let Some(dest) = std::env::var_os("DRAMA_LLAMA_DUMP_OUTPUT") else {
            // Helper, not a test: skip cleanly in `--include-ignored`
            // sweeps instead of polluting them with a failure.
            eprintln!("skipped: set DRAMA_LLAMA_DUMP_OUTPUT to the output path to use this helper");
            return;
        };
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = crate::LlamaCppEngine::from_path(path).unwrap();
        let tmpl = ChatTemplate::from_model(&engine.model).unwrap();

        let tool = Tool::builder("count_letters")
            .description(
                "Count the number of times a letter appears in a string.",
            )
            .schema(json!({
                "type": "object",
                "properties": {
                    "letter": {"type": "string", "description": "the letter to count"},
                    "string": {"type": "string", "description": "the string to search"}
                },
                "required": ["letter", "string"]
            }))
            .build()
            .expect("valid test tool");
        let prompt = Prompt {
            system: Some(Content::text(
                "You are a helpful assistant. You cannot count letters in a \
                 word reliably on your own because you see in tokens, not \
                 letters. Use the `count_letters` tool when asked to count \
                 characters.",
            )),
            messages: vec![Message {
                role: Role::User,
                content: Content::text(
                    "Count the number of r's in 'strawberry'",
                ),
            }],
            tools: Some(vec![tool.into()]),
            ..Default::default()
        };
        let opts = RenderOptions::default()
            .with_generation_prompt(true)
            .with_date("17 Apr 2026")
            .with_extra("enable_thinking", true);
        let out = tmpl.render_with(&prompt, &opts).unwrap();
        std::fs::write(&dest, &out)
            .unwrap_or_else(|e| panic!("write {dest:?}: {e}"));
        println!("wrote {} bytes to {:?}", out.len(), dest);
    }

    #[cfg(feature = "llama-cpp")]
    /// One-off dump helper: write the GGUF's embedded
    /// `tokenizer.chat_template` out to the path in
    /// `DRAMA_LLAMA_DUMP_TEMPLATE` so we can commit it as a pinned
    /// fixture. Run with
    /// `DRAMA_LLAMA_DUMP_TEMPLATE=tests/fixtures/cogito_14b_template.jinja \
    /// cargo test dump_template_fixture -- --ignored --nocapture`.
    #[test]
    #[ignore = "fixture helper"]
    fn dump_template_fixture() {
        use std::path::PathBuf;
        let Some(dest) = std::env::var_os("DRAMA_LLAMA_DUMP_TEMPLATE") else {
            // Helper, not a test: skip cleanly in `--include-ignored`
            // sweeps instead of polluting them with a failure.
            eprintln!("skipped: set DRAMA_LLAMA_DUMP_TEMPLATE to the output path to use this helper");
            return;
        };
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = crate::LlamaCppEngine::from_path(path).unwrap();
        let source = engine
            .model
            .get_meta("tokenizer.chat_template")
            .expect("model has no tokenizer.chat_template");
        std::fs::write(&dest, &source)
            .unwrap_or_else(|e| panic!("write {dest:?}: {e}"));
        println!("wrote {} bytes to {:?}", source.len(), dest);
    }

    /// Pin the shape of the rendered `tools` variable: OpenAI wire
    /// envelope (`type: "function"`, nested `function` object) with
    /// `parameters` rather than Anthropic's `input_schema`.
    ///
    /// Regression lock for the tool-shape bug: cogito / Qwen / Hermes
    /// templates do `{{ tool | tojson }}` and the model was trained on
    /// ollama's runtime output. Emitting bare `{name, description,
    /// input_schema}` caused three cogito sizes to consistently swap
    /// tool-call arguments.
    #[test]
    fn tools_rendered_as_openai_wire_envelope() {
        use serde_json::json;
        let tool = crate::Tool::builder("count_letters")
            .description("Count letters in a string.")
            .schema(json!({
                "type": "object",
                "properties": {
                    "letter": {"type": "string"},
                    "string": {"type": "string"}
                },
                "required": ["letter", "string"]
            }))
            .build()
            .expect("valid test tool");
        let src =
            r#"{%- for t in tools %}{{ t | tojson }}{% endfor %}"#.to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let prompt = Prompt {
            tools: Some(vec![tool.into()]),
            ..Default::default()
        };
        let out = t.render(&prompt, false).unwrap();

        // Must contain the wire envelope.
        assert!(
            out.contains("\"type\":\"function\""),
            "expected wire envelope `type: function`. got:\n{out}"
        );
        assert!(
            out.contains("\"function\":{"),
            "expected nested `function` object. got:\n{out}"
        );
        // Field name must be `parameters`, NOT Anthropic's `input_schema`.
        assert!(
            out.contains("\"parameters\":"),
            "expected `parameters` field. got:\n{out}"
        );
        assert!(
            !out.contains("\"input_schema\""),
            "Anthropic-shape `input_schema` must not leak. got:\n{out}"
        );
        // Tool name and schema content must survive.
        assert!(out.contains("\"name\":\"count_letters\""));
        assert!(out.contains("\"letter\""));
        assert!(out.contains("\"string\""));
    }

    // ----------------------------------------------------------------
    // Phase 2: cache_control breakpoint discovery + partial rendering
    // ----------------------------------------------------------------

    use misanthropic::prompt::message::{CacheControl, CacheTtl};
    use serde_json::json;
    use std::borrow::Cow;

    /// A tool with no cache marker.
    fn tool_plain(name: &'static str) -> Tool {
        Tool::builder(name)
            .description("tool")
            .schema(json!({"type": "object", "properties": {}}))
            .build()
            .expect("valid test tool")
    }

    /// A tool marked with an ephemeral cache breakpoint.
    fn tool_cached(name: &'static str) -> Tool {
        Tool::builder(name)
            .description("tool")
            .schema(json!({"type": "object", "properties": {}}))
            .cache()
            .build()
            .expect("valid test tool")
    }

    /// Make a `Role::User` message whose content is a single Text
    /// block with an ephemeral cache breakpoint attached.
    fn cached_user_msg(text: &'static str) -> Message {
        Message {
            role: Role::User,
            content: Content(vec![Block::Text {
                text: Cow::Borrowed(text),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            }]),
        }
    }

    #[test]
    fn test_collect_breakpoints_empty() {
        let prompt = simple_prompt();
        assert_eq!(
            collect_breakpoints(&prompt),
            Vec::<(PromptBreakpoint, CacheTtl)>::new()
        );
    }

    #[test]
    fn test_collect_breakpoints_tools_only() {
        let prompt = Prompt {
            tools: Some(vec![tool_cached("cached_tool").into()]),
            ..Prompt::default()
        };
        assert_eq!(
            collect_breakpoints(&prompt),
            vec![(PromptBreakpoint::AfterTools, CacheTtl::FiveMinutes)]
        );
    }

    #[test]
    fn test_collect_breakpoints_all_levels() {
        // Tool marker + system marker + cache on messages[0] and
        // messages[2]. messages[1] is uncached — verifies the emitted
        // message indices match the actually-cached ones.
        let system = Content(vec![Block::Text {
            text: Cow::Borrowed("You are helpful."),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }]);
        let prompt = Prompt {
            tools: Some(vec![tool_plain("a").into(), tool_cached("b").into()]),
            system: Some(system),
            messages: vec![
                cached_user_msg("first"),
                Message {
                    role: Role::Assistant,
                    content: Content::text("reply"),
                },
                cached_user_msg("third"),
            ],
            ..Prompt::default()
        };
        assert_eq!(
            collect_breakpoints(&prompt),
            vec![
                (PromptBreakpoint::AfterTools, CacheTtl::FiveMinutes),
                (PromptBreakpoint::AfterSystem, CacheTtl::FiveMinutes),
                (PromptBreakpoint::AfterMessage(0), CacheTtl::FiveMinutes),
                (PromptBreakpoint::AfterMessage(2), CacheTtl::FiveMinutes),
            ]
        );
    }

    #[test]
    fn test_collect_breakpoints_section_ttl_is_max() {
        // Two cached blocks in one system section, 5m and 1h — the
        // section resolves to the max (1h). A 1h marker on a message
        // carries through unchanged.
        let system = Content(vec![
            Block::Text {
                text: Cow::Borrowed("You are helpful."),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            },
            Block::Text {
                text: Cow::Borrowed("Stay helpful."),
                cache_control: Some(CacheControl::one_hour()),
                citations: None,
            },
        ]);
        let msg = Message {
            role: Role::User,
            content: Content(vec![Block::Text {
                text: Cow::Borrowed("hi"),
                cache_control: Some(CacheControl::one_hour()),
                citations: None,
            }]),
        };
        let prompt = Prompt {
            system: Some(system),
            messages: vec![msg],
            ..Prompt::default()
        };
        assert_eq!(
            collect_breakpoints(&prompt),
            vec![
                (PromptBreakpoint::AfterSystem, CacheTtl::OneHour),
                (PromptBreakpoint::AfterMessage(0), CacheTtl::OneHour),
            ]
        );
    }

    // ----------------------------------------------------------------
    // Automatic caching: the request-level `cache_control`
    // ----------------------------------------------------------------

    /// A `Role::User` text message, cached with `control` when given.
    fn user_text(text: &'static str, control: Option<CacheControl>) -> Message {
        Message {
            role: Role::User,
            content: Content(vec![Block::Text {
                text: Cow::Borrowed(text),
                cache_control: control,
                citations: None,
            }]),
        }
    }

    /// An assistant message holding only a thought — a block that can
    /// carry no marker.
    fn thought_only() -> Message {
        Message {
            role: Role::Assistant,
            content: Content(vec![Block::Thought {
                thought: Cow::Borrowed("hmm"),
                signature: Cow::Borrowed(""),
            }]),
        }
    }

    /// The automatic breakpoint lands after the message holding the
    /// last cacheable block, with the request-level TTL — alongside,
    /// not instead of, the explicit markers.
    #[test]
    fn auto_cache_breakpoint_lands_on_the_last_message() {
        let prompt = Prompt {
            system: Some(Content(vec![Block::Text {
                text: Cow::Borrowed("You are helpful."),
                cache_control: Some(CacheControl::one_hour()),
                citations: None,
            }])),
            messages: simple_prompt().messages,
            cache_control: Some(CacheControl::ephemeral()),
            ..Prompt::default()
        };
        assert_eq!(
            collect_breakpoints(&prompt),
            vec![
                (PromptBreakpoint::AfterSystem, CacheTtl::OneHour),
                (PromptBreakpoint::AfterMessage(2), CacheTtl::FiveMinutes),
            ]
        );
        assert_eq!(
            explicit_breakpoints(&prompt),
            vec![(PromptBreakpoint::AfterSystem, CacheTtl::OneHour)],
            "the automatic breakpoint is not an explicit marker",
        );
    }

    /// Past a trailing block that cannot carry a marker (a thought),
    /// the breakpoint walks back to the nearest one that can; with no
    /// messages it falls to the system, then the tools; with nothing
    /// cacheable at all there is none, as on Anthropic.
    #[test]
    fn auto_cache_walks_back_to_a_cacheable_block() {
        let auto = |prompt: Prompt| Prompt {
            cache_control: Some(CacheControl::ephemeral()),
            ..prompt
        };
        let at = |prompt: &Prompt| auto_cache_target(prompt).map(|t| t.at);

        let past_thought = auto(Prompt {
            messages: vec![user_text("q", None), thought_only()],
            ..Prompt::default()
        });
        assert_eq!(at(&past_thought), Some(PromptBreakpoint::AfterMessage(0)));

        let system_only = auto(Prompt::default().system("You are helpful."));
        assert_eq!(at(&system_only), Some(PromptBreakpoint::AfterSystem));

        let tools_only = auto(Prompt {
            tools: Some(vec![tool_plain("a").into()]),
            messages: vec![thought_only()],
            ..Prompt::default()
        });
        assert_eq!(at(&tools_only), Some(PromptBreakpoint::AfterTools));
        assert_eq!(
            collect_breakpoints(&tools_only),
            vec![(PromptBreakpoint::AfterTools, CacheTtl::FiveMinutes)]
        );

        let nothing = auto(Prompt {
            messages: vec![thought_only()],
            ..Prompt::default()
        });
        assert_eq!(at(&nothing), None);
        assert!(collect_breakpoints(&nothing).is_empty());
    }

    /// On a section that already carries an explicit marker the
    /// automatic breakpoint merges into it (one anchor, max TTL).
    #[test]
    fn auto_cache_merges_into_an_explicit_marker_on_its_section() {
        let prompt = Prompt {
            messages: vec![user_text("q", Some(CacheControl::ephemeral()))],
            cache_control: Some(CacheControl::one_hour()),
            ..Prompt::default()
        };
        assert_eq!(
            collect_breakpoints(&prompt),
            vec![(PromptBreakpoint::AfterMessage(0), CacheTtl::OneHour)]
        );
    }

    /// The automatic breakpoint gets a partial render like any other,
    /// ending exactly where the generation prompt begins — the anchor
    /// the next request reads back.
    #[test]
    fn auto_cache_partial_ends_before_the_generation_prompt() {
        let template = tmpl();
        let prompt = Prompt {
            cache_control: Some(CacheControl::ephemeral()),
            ..simple_prompt()
        };
        let opts = RenderOptions::default().with_generation_prompt(true);
        let rendered = template
            .render_with_breakpoints(&prompt, &opts)
            .expect("render");
        let [(bp, ttl, partial)] = &rendered.partials[..] else {
            panic!("one breakpoint: {:?}", rendered.partials);
        };
        assert_eq!(
            (bp, ttl),
            (&PromptBreakpoint::AfterMessage(2), &CacheTtl::FiveMinutes)
        );
        let generation_prompt = rendered
            .text
            .strip_prefix(partial.as_str())
            .expect("the partial is a prefix of the full render");
        assert!(
            !generation_prompt.is_empty()
                && !generation_prompt.contains("What is 2+2?"),
            "only the generation prompt follows: {generation_prompt:?}"
        );
    }

    /// Anthropic's cache_control checks, with its exact messages
    /// (captured 2026-09-30, claude-haiku-4-5).
    #[test]
    fn check_cache_controls_matches_anthropic() {
        let five = || Some(CacheControl::ephemeral());
        let hour = || Some(CacheControl::one_hour());
        let msgs = |controls: Vec<Option<CacheControl>>| {
            let blocks = controls
                .into_iter()
                .map(|cache_control| Block::Text {
                    text: Cow::Borrowed("x"),
                    cache_control,
                    citations: None,
                })
                .collect();
            vec![Message {
                role: Role::User,
                content: Content(blocks),
            }]
        };
        let found_5 = "A maximum of 4 blocks with cache_control may be \
                       provided. Found 5.";

        let explicit_5 = Prompt {
            messages: msgs(vec![five(), five(), five(), five(), five()]),
            ..Prompt::default()
        };
        assert_eq!(check_cache_controls(&explicit_5), Err(found_5.into()));

        // Four explicit plus the automatic one: five, on the wire.
        let four_and_auto = Prompt {
            messages: msgs(vec![five(), five(), five(), five(), None]),
            cache_control: five(),
            ..Prompt::default()
        };
        assert_eq!(check_cache_controls(&four_and_auto), Err(found_5.into()));

        // Even when the automatic marker lands on an explicitly marked
        // block with the same TTL (the docs' "no-op"): still five.
        let four_marked_last = Prompt {
            messages: msgs(vec![five(), five(), five(), five()]),
            cache_control: five(),
            ..Prompt::default()
        };
        assert_eq!(
            check_cache_controls(&four_marked_last),
            Err(found_5.into())
        );

        let three_marked_last = Prompt {
            messages: msgs(vec![five(), five(), five()]),
            cache_control: five(),
            ..Prompt::default()
        };
        assert_eq!(check_cache_controls(&three_marked_last), Ok(()));

        let mismatch = |auto, marker| Prompt {
            messages: msgs(vec![marker]),
            cache_control: auto,
            ..Prompt::default()
        };
        assert_eq!(
            check_cache_controls(&mismatch(five(), hour())),
            Err("Top-level cache_control has ttl='5m' but the target block \
                 already has cache_control with ttl='1h'. When both are \
                 specified on the same block, they must have matching TTLs."
                .into())
        );
        assert_eq!(
            check_cache_controls(&mismatch(hour(), five())),
            Err("Top-level cache_control has ttl='1h' but the target block \
                 already has cache_control with ttl='5m'. When both are \
                 specified on the same block, they must have matching TTLs."
                .into())
        );

        let system = |control| {
            Some(Content(vec![Block::Text {
                text: Cow::Borrowed("sys"),
                cache_control: control,
                citations: None,
            }]))
        };
        let hour_after_five = Prompt {
            system: system(five()),
            messages: msgs(vec![None]),
            cache_control: hour(),
            ..Prompt::default()
        };
        assert_eq!(
            check_cache_controls(&hour_after_five),
            Err("cache_control: a ttl='1h' cache_control block must not \
                 come after a ttl='5m' cache_control block. Note that \
                 blocks are processed in the following order: `tools`, \
                 `system`, `messages`."
                .into())
        );
        let five_after_hour = Prompt {
            system: system(hour()),
            messages: msgs(vec![None]),
            cache_control: five(),
            ..Prompt::default()
        };
        assert_eq!(check_cache_controls(&five_after_hour), Ok(()));

        // Two explicit markers out of order: Anthropic names the first
        // 1-hour marker after a 5-minute one by its path, and checks
        // this before everything but the count (captured 2026-09-30,
        // count_tokens, claude-haiku-4-5).
        let order_at = |path: &str| {
            Err(format!(
                "{path}.cache_control.ttl: a ttl='1h' cache_control block \
                 must not come after a ttl='5m' cache_control block. Note \
                 that blocks are processed in the following order: \
                 `tools`, `system`, `messages`."
            ))
        };
        let explicit = |controls, auto| Prompt {
            messages: msgs(controls),
            cache_control: auto,
            ..Prompt::default()
        };
        assert_eq!(
            check_cache_controls(&explicit(vec![five(), hour()], None)),
            order_at("messages.0.content.1")
        );
        // The first offender, not the last.
        assert_eq!(
            check_cache_controls(&explicit(
                vec![hour(), five(), hour(), hour()],
                None
            )),
            order_at("messages.0.content.2")
        );
        // Before the automatic marker's own TTL checks: a 1h automatic
        // marker on a target already out of order, and a 5m one whose
        // target's 1h marker it disagrees with.
        assert_eq!(
            check_cache_controls(&explicit(vec![five(), hour()], hour())),
            order_at("messages.0.content.1")
        );
        assert_eq!(
            check_cache_controls(&explicit(vec![five(), hour()], five())),
            order_at("messages.0.content.1")
        );
        // After the count: five markers, one out of order, answer
        // with the count — five explicit, or four and the automatic
        // one (both captured 2026-09-30, count_tokens).
        assert_eq!(
            check_cache_controls(&explicit(
                vec![five(), hour(), five(), five(), five()],
                None
            )),
            Err(found_5.into())
        );
        assert_eq!(
            check_cache_controls(&explicit(
                vec![five(), hour(), five(), five(), None],
                five()
            )),
            Err(found_5.into())
        );
        // Across sections, in processing order.
        let across = Prompt {
            system: system(five()),
            messages: msgs(vec![None, hour()]),
            ..Prompt::default()
        };
        assert_eq!(
            check_cache_controls(&across),
            order_at("messages.0.content.1")
        );
        let tools_first = Prompt {
            tools: Some(vec![tool_cached("t").into()]),
            system: system(hour()),
            messages: msgs(vec![None]),
            ..Prompt::default()
        };
        assert_eq!(check_cache_controls(&tools_first), order_at("system.0"));
        // The count before the automatic marker's TTL checks: five
        // markers whose automatic one also disagrees with its target,
        // or also follows a 5m marker.
        assert_eq!(
            check_cache_controls(&explicit(
                vec![hour(), hour(), hour(), five()],
                hour()
            )),
            Err(found_5.into())
        );
        assert_eq!(
            check_cache_controls(&explicit(
                vec![five(), five(), five(), five(), None],
                hour()
            )),
            Err(found_5.into())
        );
        // An explicit order that holds leaves the target mismatch as
        // the answer: [1h, 5m] under a 1h automatic marker.
        assert_eq!(
            check_cache_controls(&explicit(vec![hour(), five()], hour())),
            Err("Top-level cache_control has ttl='1h' but the target block \
                 already has cache_control with ttl='5m'. When both are \
                 specified on the same block, they must have matching TTLs."
                .into())
        );

        // A cached server-tool definition counts toward the limit
        // (captured 2026-09-30, count_tokens), but its TTL is not
        // readable upstream, so as the automatic marker's target it is
        // no mismatch here — our choice, not a capture — where a
        // custom tool's marker is.
        let server = || {
            let mut def: misanthropic::tool::MethodDef =
                misanthropic::tool::ServerMethodDef::web_search(
                    Default::default(),
                )
                .into();
            def.cache_with(CacheControl::one_hour());
            def
        };
        let tools_only = |def, auto| Prompt {
            tools: Some(vec![def]),
            cache_control: auto,
            ..Prompt::default()
        };
        assert_eq!(check_cache_controls(&tools_only(server(), five())), Ok(()));
        assert_eq!(
            check_cache_controls(&tools_only(tool_cached("t").into(), hour())),
            Err("Top-level cache_control has ttl='1h' but the target block \
                 already has cache_control with ttl='5m'. When both are \
                 specified on the same block, they must have matching TTLs."
                .into())
        );
        let server_and_four = Prompt {
            tools: Some(vec![server()]),
            messages: msgs(vec![five(), five(), five(), five()]),
            ..Prompt::default()
        };
        assert_eq!(check_cache_controls(&server_and_four), Err(found_5.into()));
    }

    #[test]
    fn test_max_ttl_and_duration() {
        use std::time::Duration;
        assert_eq!(
            ttl_duration(&CacheTtl::FiveMinutes),
            Duration::from_secs(300)
        );
        assert_eq!(ttl_duration(&CacheTtl::OneHour), Duration::from_secs(3600));
        assert_eq!(
            max_ttl(CacheTtl::FiveMinutes, CacheTtl::OneHour),
            CacheTtl::OneHour
        );
        assert_eq!(
            max_ttl(CacheTtl::OneHour, CacheTtl::FiveMinutes),
            CacheTtl::OneHour
        );
        assert_eq!(
            max_ttl(CacheTtl::FiveMinutes, CacheTtl::FiveMinutes),
            CacheTtl::FiveMinutes
        );
    }

    /// The load-bearing property of the whole open-thought feature:
    /// rendering a prompt whose tail is an open thought produces
    /// exactly `<the generation prompt> ++ <the thought's bytes>`.
    ///
    /// The body deliberately ends in `\n\n\n`. Whitespace is the entire
    /// game — the KV cache holds the model's bytes verbatim, and any
    /// path through Jinja may normalize them away (Qwen3.6's stock
    /// template `|trim`s message content), making `\n` and `\n\n\n`
    /// indistinguishable and the re-render a silent cache miss.
    #[test]
    fn open_thought_tail_appends_raw_after_generation_prompt() {
        let base = Prompt::default().add_message((Role::User, "why?")).unwrap();
        let opts = RenderOptions::default()
            .with_generation_prompt(true)
            .with_reasoning_start("<think>");

        let generation_prompt =
            qwen3_like_tmpl().render_with(&base, &opts).unwrap();

        let body = "Weighing the two options.\n\n\n";
        let mut resumed = base.clone();
        resumed.messages.push(Message {
            role: Role::Assistant,
            content: Content(vec![crate::prompt::open_thought(body)]),
        });
        let rendered = qwen3_like_tmpl().render_with(&resumed, &opts).unwrap();

        assert_eq!(
            rendered,
            format!("{generation_prompt}<think>{body}"),
            "an open tail must render as generation prompt ++ raw body",
        );
        // No close marker, and the turn is never ended: the model
        // resumes *inside* the reasoning block.
        assert!(!rendered.contains("</think>"));
        assert!(rendered.ends_with("\n\n\n"));
    }

    /// The append happens even when the caller asked for
    /// `add_generation_prompt=false` — an open trailing thought IS a
    /// generation prompt, and this is what keeps the partial-render and
    /// canonicalization paths (which both pass `false`) consistent with
    /// the full render.
    #[test]
    fn open_thought_tail_forces_the_generation_prompt() {
        let mut prompt =
            Prompt::default().add_message((Role::User, "why?")).unwrap();
        prompt.messages.push(Message {
            role: Role::Assistant,
            content: Content(vec![crate::prompt::open_thought("hmm")]),
        });
        let rendered = qwen3_like_tmpl()
            .render_with(
                &prompt,
                &RenderOptions::default()
                    .with_generation_prompt(false)
                    .with_reasoning_start("<think>"),
            )
            .unwrap();
        assert!(rendered.ends_with("<|im_start|>assistant<think>hmm"));
    }

    /// A dialect with no reasoning open marker cannot resume a thought.
    /// Dropping the body would be the silent failure this whole feature
    /// exists to eliminate, so it is typed.
    #[test]
    fn open_thought_without_marker_is_typed_error() {
        let mut prompt =
            Prompt::default().add_message((Role::User, "why?")).unwrap();
        prompt.messages.push(Message {
            role: Role::Assistant,
            content: Content(vec![crate::prompt::open_thought("hmm")]),
        });
        assert!(matches!(
            qwen3_like_tmpl().render_with(&prompt, &RenderOptions::default()),
            Err(ChatTemplateError::OpenThoughtUnsupported),
        ));
    }

    /// Only the sole-block trailing assistant shape is diverted. A
    /// leading `Text` beside it, or a non-tail position, renders the
    /// ordinary way — `Session` rejects those at ingest, so the
    /// renderer must not silently treat them as resumable.
    #[test]
    fn open_thought_tail_requires_sole_block_at_the_tail() {
        let base = Prompt::default().add_message((Role::User, "why?")).unwrap();

        let mut with_text = base.clone();
        with_text.messages.push(Message {
            role: Role::Assistant,
            content: Content(vec![
                Block::from("\n"),
                crate::prompt::open_thought("hmm"),
            ]),
        });
        assert!(open_thought_tail(&with_text).is_none());

        let mut not_tail = base.clone();
        not_tail.messages.push(Message {
            role: Role::Assistant,
            content: Content(vec![crate::prompt::open_thought("hmm")]),
        });
        not_tail
            .messages
            .push(Message::from((Role::User, "still there?")));
        assert!(open_thought_tail(&not_tail).is_none());

        // A user turn can never carry reasoning.
        let mut user_side = base.clone();
        user_side.messages.push(Message {
            role: Role::User,
            content: Content(vec![crate::prompt::open_thought("hmm")]),
        });
        assert!(open_thought_tail(&user_side).is_none());
    }

    #[test]
    fn test_render_partial_after_system() {
        // Truncated prompt at AfterSystem should render with an empty
        // messages list; the rendered bytes must match what
        // `render_with` produces on the same prompt with messages
        // cleared and `add_generation_prompt=false`.
        let prompt = Prompt {
            system: Some(Content::text("Sys.")),
            messages: vec![Message {
                role: Role::User,
                content: Content::text("q"),
            }],
            ..Prompt::default()
        };
        let opts = RenderOptions::default().with_generation_prompt(true);
        let partial = render_partial(
            &tmpl(),
            &prompt,
            &opts,
            PromptBreakpoint::AfterSystem,
        )
        .unwrap();

        let reference = Prompt {
            system: prompt.system.clone(),
            messages: vec![],
            ..Prompt::default()
        };
        let expected = tmpl()
            .render_with(
                &reference,
                &RenderOptions::default().with_generation_prompt(false),
            )
            .unwrap();
        assert_eq!(partial, expected);
    }

    /// A template that writes `enable_thinking` into the prompt
    /// prefix (Mistral Small 4's `[MODEL_SETTINGS]` shape): every
    /// partial must be a byte prefix of the full render when the
    /// request enables thinking. Before the fix the partial rendered
    /// with thinking unset and diverged at the switch.
    #[test]
    fn test_render_with_breakpoints_carries_thinking_into_partials() {
        use misanthropic::prompt::thinking::Thinking;
        let src = "[SETTINGS]{{ 'on' if enable_thinking else 'off' }}\
                   [/SETTINGS]{% for m in messages %}[{{ m['role'] }}]\
                   {{ m['content'] }}{% endfor %}"
            .to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let prompt = Prompt {
            system: Some(Content(vec![Block::Text {
                text: Cow::Borrowed("sys"),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            }])),
            messages: vec![
                cached_user_msg("hi"),
                Message {
                    role: Role::Assistant,
                    content: Content::text("hello"),
                },
                cached_user_msg("again"),
            ],
            ..Prompt::default()
        }
        .thinking(Thinking::Enabled {
            budget_tokens: std::num::NonZeroU32::new(64).unwrap(),
            display: None,
        });
        let out = t
            .render_with_breakpoints(&prompt, &RenderOptions::default())
            .unwrap();
        assert!(out.text.starts_with("[SETTINGS]on"), "{}", out.text);
        assert_eq!(out.partials.len(), 3);
        for (bp, _, partial) in &out.partials {
            assert!(
                out.text.starts_with(partial.as_str()),
                "{bp:?} is not a byte prefix of the full render:\n  \
                 full:    {:?}\n  partial: {:?}",
                out.text,
                partial
            );
        }
    }

    // ----------------------------------------------------------------
    // output_config.effort → reasoning_effort
    // ----------------------------------------------------------------

    /// Qwen3.8's effort shape, trimmed: `high` is rewritten to
    /// `xhigh`, anything outside `xhigh`/`medium`/`low` raises, and the
    /// instruction lands in the system block — the prompt *prefix*.
    const QWEN_EFFORT_SRC: &str = "\
        {%- set ri = '' %}\
        {%- if enable_thinking is undefined or enable_thinking is true %}\
        {%- set e = reasoning_effort|default('xhigh') %}\
        {%- if e == 'high' %}{%- set e = 'xhigh' %}{%- endif %}\
        {%- if e not in ('xhigh', 'medium', 'low') %}\
        {{- raise_exception('bad effort ' ~ e) }}{%- endif %}\
        {%- if e == 'xhigh' %}{%- set ri = 'THINK HARD.' %}\
        {%- elif e == 'low' %}{%- set ri = 'THINK BRIEFLY.' %}{%- endif %}\
        {%- endif %}\
        <|im_start|>system\n\
        {%- for t in tools or [] %}{{ t.function.name }};{% endfor %}\
        {{- ri }}<|im_end|>\n\
        {%- for m in messages %}<|im_start|>{{ m.role }}\n\
        {{- m.content }}<|im_end|>\n{% endfor %}\
        {%- if add_generation_prompt %}<|im_start|>assistant\n<think>\n\
        {%- endif %}";

    fn qwen_efforts() -> RenderOptions {
        RenderOptions::default()
            .with_generation_prompt(true)
            .with_efforts(["low", "medium", "high", "xhigh"])
    }

    fn thinking_on() -> Thinking {
        Thinking::Enabled {
            budget_tokens: std::num::NonZeroU32::new(1024).unwrap(),
            display: None,
        }
    }

    fn user_prompt() -> Prompt {
        Prompt::default().add_message((Role::User, "hi")).unwrap()
    }

    /// Nearest accepted level, the lower on a tie; an unknown level or
    /// an empty set maps to nothing.
    #[test]
    fn test_resolve_effort_table() {
        let qwen = ["low", "medium", "high", "xhigh"].map(String::from);
        let mistral = ["high".to_string()];
        let gptoss = ["low", "medium", "high"].map(String::from);
        let ends = ["low", "max"].map(String::from);
        let custom = Effort::Custom(Cow::Borrowed("ultra"));
        let cases: &[(&Effort, &[String], Option<&str>)] = &[
            (&Effort::Low, &qwen, Some("low")),
            (&Effort::Medium, &qwen, Some("medium")),
            (&Effort::High, &qwen, Some("high")),
            (&Effort::XHigh, &qwen, Some("xhigh")),
            (&Effort::Max, &qwen, Some("xhigh")),
            (&Effort::Low, &mistral, Some("high")),
            (&Effort::Max, &mistral, Some("high")),
            (&Effort::XHigh, &gptoss, Some("high")),
            (&Effort::Max, &gptoss, Some("high")),
            (&Effort::Medium, &gptoss, Some("medium")),
            // `high` is two from `low` and two from `max`: the lower wins.
            (&Effort::High, &ends, Some("low")),
            (&Effort::XHigh, &ends, Some("max")),
            (&custom, &qwen, None),
            (&Effort::Low, &[], None),
        ];
        for (requested, accepted, want) in cases {
            assert_eq!(
                resolve_effort(requested, accepted),
                *want,
                "{requested:?} over {accepted:?}"
            );
        }
    }

    /// `Thinking::Disabled` is an explicit *off*: it must render as
    /// `enable_thinking = false`, like an absent `thinking`, not as on.
    #[test]
    fn test_thinking_disabled_renders_off() {
        let src = "T={{ enable_thinking }}".to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let opts = RenderOptions::default();
        let render = |p: &Prompt| t.render_with(p, &opts).unwrap();
        assert_eq!(render(&user_prompt()), "T=false");
        assert_eq!(
            render(&user_prompt().thinking(Thinking::Disabled)),
            "T=false"
        );
        assert_eq!(render(&user_prompt().thinking(thinking_on())), "T=true");
    }

    /// Which `reasoning_effort` reaches the template, read back
    /// directly: set only for thinking-on prompts that request an
    /// effort, never over a caller extra, never for an unknown level
    /// or a template without the knob.
    #[test]
    fn test_render_reasoning_effort_derivation() {
        let src = "E={{ reasoning_effort|default('unset') }}".to_owned();
        let t = ChatTemplate::from_source(src, "".into(), "".into()).unwrap();
        let render = |prompt: &Prompt, opts: &RenderOptions| {
            t.render_with(prompt, opts).unwrap()
        };
        let opts = qwen_efforts();
        let on = user_prompt().thinking(thinking_on());

        assert_eq!(render(&on.clone().effort(Effort::Low), &opts), "E=low");
        assert_eq!(render(&on.clone().effort(Effort::Max), &opts), "E=xhigh");
        // No effort requested: the template's default.
        assert_eq!(render(&on, &opts), "E=unset");
        // Thinking off — absent or explicitly disabled — never sets it.
        assert_eq!(
            render(&user_prompt().effort(Effort::Low), &opts),
            "E=unset"
        );
        assert_eq!(
            render(
                &user_prompt()
                    .thinking(Thinking::Disabled)
                    .effort(Effort::Low),
                &opts
            ),
            "E=unset"
        );
        // A caller extra wins.
        let pinned = opts.clone().with_extra("reasoning_effort", "medium");
        assert_eq!(
            render(&on.clone().effort(Effort::Low), &pinned),
            "E=medium"
        );
        // No knob, or a level we can't place: left alone.
        let low = on.clone().effort(Effort::Low);
        assert_eq!(render(&low, &RenderOptions::default()), "E=unset");
        let custom = on.effort(Effort::Custom(Cow::Borrowed("ultra")));
        assert_eq!(render(&custom, &opts), "E=unset");
    }

    /// Against the Qwen3.8 shape: `Low` renders the low instruction,
    /// no effort renders the template's `xhigh` default, and `Max`
    /// (which the template would reject) renders `xhigh` rather than
    /// failing.
    #[test]
    fn test_render_effort_qwen_like() {
        let t = ChatTemplate::from_source(
            QWEN_EFFORT_SRC.to_owned(),
            "".into(),
            "".into(),
        )
        .unwrap();
        let on = user_prompt().thinking(thinking_on());
        let opts = qwen_efforts();

        let low = t.render_with(&on.clone().effort(Effort::Low), &opts);
        let low = low.unwrap();
        assert!(low.contains("THINK BRIEFLY."), "{low}");
        assert!(!low.contains("THINK HARD."), "{low}");

        let default = t.render_with(&on, &opts).unwrap();
        assert!(default.contains("THINK HARD."), "{default}");

        let max = t.render_with(&on.effort(Effort::Max), &opts).unwrap();
        assert_eq!(max, default, "Max maps to xhigh, the default");
    }

    /// The effort lands in the prompt PREFIX (system block), so every
    /// partial must carry it or none is a prefix of the full render and
    /// the cache loses every breakpoint — the #93 thinking bug again.
    #[test]
    fn test_render_with_breakpoints_carries_effort_into_partials() {
        let t = ChatTemplate::from_source(
            QWEN_EFFORT_SRC.to_owned(),
            "".into(),
            "".into(),
        )
        .unwrap();
        let prompt = Prompt {
            tools: Some(vec![tool_cached("ping").into()]),
            system: Some(Content(vec![Block::Text {
                text: Cow::Borrowed("sys"),
                cache_control: Some(CacheControl::ephemeral()),
                citations: None,
            }])),
            messages: vec![
                cached_user_msg("hi"),
                Message {
                    role: Role::Assistant,
                    content: Content::text("hello"),
                },
                cached_user_msg("again"),
            ],
            ..Prompt::default()
        }
        .thinking(thinking_on())
        .effort(Effort::Low);
        let out = t.render_with_breakpoints(&prompt, &qwen_efforts()).unwrap();
        assert!(out.text.contains("THINK BRIEFLY."), "{}", out.text);
        let kinds: Vec<_> = out.partials.iter().map(|(bp, ..)| *bp).collect();
        assert_eq!(
            kinds,
            vec![
                PromptBreakpoint::AfterTools,
                PromptBreakpoint::AfterSystem,
                PromptBreakpoint::AfterMessage(0),
                PromptBreakpoint::AfterMessage(2),
            ]
        );
        for (bp, _, partial) in &out.partials {
            assert!(
                out.text.starts_with(partial.as_str()),
                "{bp:?} is not a byte prefix of the full render:\n  \
                 full:    {:?}\n  partial: {:?}",
                out.text,
                partial
            );
        }
    }

    #[test]
    fn test_render_with_breakpoints_no_cache_control() {
        let out = tmpl()
            .render_with_breakpoints(
                &simple_prompt(),
                &RenderOptions::default().with_generation_prompt(true),
            )
            .unwrap();
        assert!(out.partials.is_empty());
        assert!(out.text.starts_with("<|begin_of_text|>"));
    }

    #[test]
    fn test_render_with_breakpoints_generation_prompt_forced_false() {
        // Prompt with a mid-conversation cache breakpoint and the
        // caller asking for add_generation_prompt=true. The full
        // render must honor it; each partial must not.
        let prompt = Prompt {
            system: Some(Content::text("sys")),
            messages: vec![
                cached_user_msg("hi"),
                Message {
                    role: Role::Assistant,
                    content: Content::text("hello"),
                },
            ],
            ..Prompt::default()
        };
        let out = tmpl()
            .render_with_breakpoints(
                &prompt,
                &RenderOptions::default().with_generation_prompt(true),
            )
            .unwrap();
        const GEN: &str = "<|start_header_id|>assistant<|end_header_id|>\n\n";
        assert!(
            out.text.ends_with(GEN),
            "full render must end with generation prompt: {:?}",
            out.text
        );
        assert_eq!(out.partials.len(), 1, "one cache marker → one partial");
        for (i, (_, _, p)) in out.partials.iter().enumerate() {
            assert!(
                !p.ends_with(GEN),
                "partial {i} must not end with generation prompt: {p:?}"
            );
        }
    }

    /// Qwen3-family templates raise `No user query found in messages.`
    /// whenever `messages` contains no non-tool-response user-role
    /// entry — exactly the state [`render_partial`] produces for
    /// [`PromptBreakpoint::AfterSystem`] (truncates `messages` to empty).
    /// Before the permissive-env split, this raise silently dropped the
    /// AfterSystem partial from `partial_texts`, killing the front-of-
    /// prompt cache anchor on every call. Permissive env rebinds
    /// `raise_exception` to return an empty string so the partial
    /// renders the system header bytes that the cache key needs.
    #[test]
    fn test_render_partial_after_system_qwen3_like_permissive_succeeds() {
        let prompt = Prompt {
            system: Some(Content::text("You are agent.")),
            messages: vec![Message {
                role: Role::User,
                content: Content::text("hi"),
            }],
            ..Prompt::default()
        };
        let opts = RenderOptions::default().with_generation_prompt(true);
        let partial = render_partial(
            &qwen3_like_tmpl(),
            &prompt,
            &opts,
            PromptBreakpoint::AfterSystem,
        )
        .expect(
            "permissive env must let AfterSystem render even when the \
             template raises on the empty-messages state",
        );
        assert!(
            partial.contains("You are agent."),
            "AfterSystem partial must carry the system bytes; got: {partial:?}"
        );
        assert!(
            partial.contains("<|im_start|>system"),
            "AfterSystem partial must contain the system header; got: \
             {partial:?}"
        );
        // Forced add_generation_prompt=false on partials.
        assert!(
            !partial.contains("<|im_start|>assistant"),
            "partial must not emit the generation prompt; got: {partial:?}"
        );
    }

    /// Same idea for `PromptBreakpoint::AfterTools`: `render_partial` truncates
    /// both `system` and `messages` to empty, hitting both Qwen3 raises
    /// (`No messages provided.` and `No user query found in messages.`).
    /// Permissive env must drop both, leaving the tools-embedded system
    /// header as the rendered bytes.
    #[test]
    fn test_render_partial_after_tools_qwen3_like_permissive_succeeds() {
        let prompt = Prompt {
            tools: Some(vec![tool_cached("ping").into()]),
            system: Some(Content::text("sys")),
            messages: vec![Message {
                role: Role::User,
                content: Content::text("q"),
            }],
            ..Prompt::default()
        };
        let opts = RenderOptions::default().with_generation_prompt(true);
        let partial = render_partial(
            &qwen3_like_tmpl(),
            &prompt,
            &opts,
            PromptBreakpoint::AfterTools,
        )
        .expect(
            "permissive env must let AfterTools render even when the \
             template raises on the empty-messages state",
        );
        assert!(
            partial.contains("# Tools"),
            "AfterTools partial must contain the tools section; got: \
             {partial:?}"
        );
        assert!(
            partial.contains("\"name\":\"ping\""),
            "AfterTools partial must carry the tool definition; got: \
             {partial:?}"
        );
    }

    /// End-to-end through `render_with_breakpoints`: a Qwen3-shape
    /// prompt with `cache_control` on the system content must surface a
    /// non-empty `partial_texts` entry. The pre-permissive behavior
    /// silently dropped this and returned `partial_texts.is_empty()`.
    #[test]
    fn test_render_with_breakpoints_after_system_survives_on_qwen3_like() {
        let system = Content(vec![Block::Text {
            text: Cow::Borrowed("You are agent."),
            cache_control: Some(CacheControl::ephemeral()),
            citations: None,
        }]);
        let prompt = Prompt {
            system: Some(system),
            messages: vec![Message {
                role: Role::User,
                content: Content::text("hi"),
            }],
            ..Prompt::default()
        };
        let out = qwen3_like_tmpl()
            .render_with_breakpoints(
                &prompt,
                &RenderOptions::default().with_generation_prompt(true),
            )
            .expect("full render must succeed on a well-formed Qwen3 prompt");
        assert_eq!(
            out.partials.len(),
            1,
            "AfterSystem breakpoint must survive partial rendering"
        );
        assert!(out.partials[0].2.contains("You are agent."));
    }

    /// Strict env (full render) must continue to surface raises as
    /// errors after the permissive split — the permissive behavior is
    /// scoped to partial renders only.
    #[test]
    fn test_full_render_strict_env_still_raises_on_qwen3_like() {
        // Bare system, no user message at all. The full render walks
        // multi_step_tool, finds no user message, raises. Strict env
        // must propagate that as an error.
        let prompt = Prompt {
            system: Some(Content::text("sys")),
            ..Prompt::default()
        };
        let err = qwen3_like_tmpl()
            .render(&prompt, true)
            .expect_err("full render must surface the raise");
        let msg = format!("{err}");
        assert!(
            msg.contains("No user query") || msg.contains("No messages"),
            "error must carry the raised message; got: {msg}"
        );
    }

    #[cfg(feature = "llama-cpp")]
    /// LlamaCppModel-backed round-trip: render a prompt with a mid-
    /// conversation cache breakpoint, tokenize full + partial, and
    /// assert the partial's tokens are a proper prefix of the full's
    /// — i.e. the breakpoint index in the returned list equals the
    /// partial's token length, and the full's first `idx` tokens
    /// equal the partial's tokens.
    #[test]
    #[ignore = "requires model"]
    fn test_tokenize_with_breakpoints_prefix_property() {
        use std::path::PathBuf;

        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
        let engine = crate::LlamaCppEngine::from_path(path).unwrap();
        let t = ChatTemplate::from_model(&engine.model).unwrap();

        let prompt = Prompt {
            system: Some(Content::text("You are helpful.")),
            messages: vec![
                cached_user_msg("Who am I?"),
                Message {
                    role: Role::Assistant,
                    content: Content::text("A human."),
                },
                Message {
                    role: Role::User,
                    content: Content::text("What next?"),
                },
            ],
            ..Prompt::default()
        };

        let opts = RenderOptions::default().with_generation_prompt(true);
        let rendered = t.render_with_breakpoints(&prompt, &opts).unwrap();
        assert_eq!(rendered.partials.len(), 1);

        let (full_tokens, indices) =
            tokenize_with_breakpoints(&engine.model, &rendered);
        assert_eq!(indices.len(), 1, "exactly one breakpoint expected");
        let idx = indices[0];
        assert!(idx <= full_tokens.len());

        // The partial's tokens equal the full's tokens up to `idx`.
        let partial_tokens =
            engine.model.tokenize(&rendered.partials[0].2, true);
        assert_eq!(partial_tokens.len(), idx);
        assert_eq!(&full_tokens[..idx], partial_tokens.as_slice());
    }

    // =======================================================================
    // Media sentinels
    // =======================================================================

    fn one_px_png_block() -> Block {
        // Any base64 payload works — the render layer only hashes the
        // source bytes; nothing decodes here.
        Block::Image {
            image: misanthropic::prompt::message::Image::Base64 {
                media_type: misanthropic::prompt::message::MediaType::Png,
                data: "aGVsbG8gcGl4ZWxz".into(),
            },
            cache_control: None,
        }
    }

    fn image_prompt() -> Prompt {
        Prompt {
            messages: vec![Message {
                role: Role::User,
                content: Content(vec![
                    Block::Text {
                        text: "What breed is ".into(),
                        cache_control: None,
                        citations: None,
                    },
                    one_px_png_block(),
                    Block::Text {
                        text: " shown here?".into(),
                        cache_control: None,
                        citations: None,
                    },
                ]),
            }],
            ..Prompt::default()
        }
    }

    #[test]
    fn image_without_sentinel_is_a_typed_error() {
        let err = tmpl()
            .render(&image_prompt(), true)
            .expect_err("image with no sentinel must not render");
        assert!(matches!(err, ChatTemplateError::MediaUnsupported));
        // Imageless prompts are unaffected.
        assert!(tmpl().render(&simple_prompt(), true).is_ok());
    }

    #[test]
    fn image_renders_as_sentinel_marker_and_splits_back() {
        let sentinel = "0123456789abcdef0123456789abcdef";
        let opts = RenderOptions::default()
            .with_generation_prompt(true)
            .with_media_sentinel(sentinel);
        let out = tmpl().render_with(&image_prompt(), &opts).unwrap();

        let src_hash = match one_px_png_block() {
            Block::Image { image, .. } => image_source_hash(&image),
            _ => unreachable!(),
        };
        let marker = media_marker(sentinel, &src_hash);
        assert!(out.contains(&marker), "render carries the marker");

        let split = split_render(&out, sentinel).unwrap();
        assert_eq!(split.markers, vec![RenderMarker::Media(src_hash)]);
        assert_eq!(split.segments.len(), 2);
        assert!(split.segments[0].ends_with("What breed is "));
        assert!(split.segments[1].starts_with(" shown here?"));
        // Reassembly is lossless.
        let reassembled =
            format!("{}{}{}", split.segments[0], marker, split.segments[1]);
        assert_eq!(reassembled, out);
    }

    #[test]
    fn literal_markers_in_content_are_inert() {
        let sentinel = "ffffffffffffffffffffffffffffffff";
        // Content containing mtmd's real marker AND a full sentinel-
        // shaped string with a different random part: all inert prose.
        let evil = format!(
            "look: <__media__> and <{}:{}>",
            "00000000000000000000000000000000",
            "ab".repeat(32)
        );
        let p = Prompt::default()
            .add_message((Role::User, evil.as_str()))
            .unwrap();
        let opts = RenderOptions::default().with_media_sentinel(sentinel);
        let out = tmpl().render_with(&p, &opts).unwrap();
        let split = split_render(&out, sentinel).unwrap();
        assert!(split.markers.is_empty());
        assert_eq!(split.segments.len(), 1);
        assert!(split.segments[0].contains(&evil), "content round-trips");
    }

    #[test]
    fn split_media_render_rejects_mangled_markers() {
        let sentinel = "0123456789abcdef0123456789abcdef";
        // Truncated mid-hash: parse must fail loudly, never fall back
        // to treating the mangled marker as content.
        let mangled = format!("text <{sentinel}:abc123 more");
        assert!(split_render(&mangled, sentinel).is_err());
        // Literal markers: truncated, non-canonical, or out of range.
        for bad in ["t12", "t", "t007", "tx>", "t99999999999>"] {
            let mangled = format!("a <{sentinel}:{bad} b");
            assert_eq!(
                split_render(&mangled, sentinel),
                Err(2),
                "{bad:?} must fail at the marker's offset",
            );
        }
        // A well-formed literal marker splits out.
        let lit = format!("a {} b", literal_marker(sentinel, 42));
        let split = split_render(&lit, sentinel).unwrap();
        assert_eq!(split.segments, vec!["a ", " b"]);
        assert_eq!(split.markers, vec![RenderMarker::Literal(42)]);
        // Sentinel-free text is one segment.
        let clean = split_render("no media here", sentinel).unwrap();
        assert_eq!(clean.segments, vec!["no media here"]);
    }

    /// A marker a template filter transformed is caught case-blind; a
    /// clean split and a real marker are not.
    #[test]
    fn transformed_markers_are_detected() {
        let sentinel = "0123456789abcdef0123456789abcdef";
        let marker = literal_marker(sentinel, 302);
        let clean = format!("{{\"type\": \"{marker}\"}}");
        let split = split_render(&clean, sentinel).unwrap();
        assert_eq!(split.markers.len(), 1);
        assert!(!has_transformed_marker(&split, sentinel));
        for render in [
            clean.to_uppercase(),
            clean.replace('<', "&lt;"),
            format!("x {}", &sentinel.to_uppercase()[..]),
        ] {
            let split = split_render(&render, sentinel).unwrap();
            assert!(split.markers.is_empty(), "{render}");
            assert!(has_transformed_marker(&split, sentinel), "{render}");
        }
        let plain = split_render("no markers", sentinel).unwrap();
        assert!(!has_transformed_marker(&plain, sentinel));
        assert!(!has_transformed_marker(&plain, ""));
    }

    #[test]
    fn image_source_hash_separates_kinds_and_payloads() {
        use misanthropic::prompt::message::{Image as ApiImage, MediaType};
        let a = ApiImage::Base64 {
            media_type: MediaType::Png,
            data: "AAAA".into(),
        };
        let b = ApiImage::Base64 {
            media_type: MediaType::Png,
            data: "BBBB".into(),
        };
        let url = ApiImage::Url {
            url: "https://example.com/x.png".into(),
        };
        assert_ne!(image_source_hash(&a), image_source_hash(&b));
        assert_ne!(image_source_hash(&a), image_source_hash(&url));
        assert_eq!(image_source_hash(&a), image_source_hash(&a));
    }

    #[test]
    fn image_in_tool_result_renders_a_marker() {
        use misanthropic::tool;
        let sentinel = "11111111111111111111111111111111";
        let result = tool::Result {
            tool_use_id: "call_1".into(),
            content: Content(vec![
                Block::Text {
                    text: "screenshot:".into(),
                    cache_control: None,
                    citations: None,
                },
                one_px_png_block(),
            ]),
            is_error: false,
            cache_control: None,
        };
        let p = Prompt {
            messages: vec![Message {
                role: Role::User,
                content: Content(vec![Block::ToolResult { result }]),
            }],
            ..Prompt::default()
        };
        let opts = RenderOptions::default().with_media_sentinel(sentinel);
        let out = tmpl().render_with(&p, &opts).unwrap();
        let split = split_render(&out, sentinel).unwrap();
        assert_eq!(
            split.markers.len(),
            1,
            "tool-result images render markers too"
        );
    }
}
