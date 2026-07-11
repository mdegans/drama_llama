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
    value::Value as JinjaValue, Environment, Error as JinjaError,
    UndefinedBehavior,
};
use serde::Serialize;

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
/// Key order within the JSON object is irrelevant for the model
/// (minijinja alphabetizes on output anyway); the structural shape
/// and field names are what matter.
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
    /// [`render_partial`] produces for [`Breakpoint::AfterTools`] and
    /// [`Breakpoint::AfterSystem`] (both truncate `messages` to empty).
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
    }

    /// Shared render path. `env` selects strict vs. permissive raise
    /// handling — full renders pass [`Self::env`]; partial renders for
    /// cache-breakpoint hashing pass [`Self::env_permissive`].
    fn render_with_env(
        &self,
        env: &Environment<'static>,
        prompt: &Prompt,
        opts: &RenderOptions,
    ) -> Result<String, ChatTemplateError> {
        // Images require a media sentinel to render into — anything
        // else is the silent drop this check exists to kill.
        if opts.media_sentinel.is_none() && prompt_has_images(prompt) {
            return Err(ChatTemplateError::MediaUnsupported);
        }
        let messages = build_messages(
            prompt,
            opts.thought_reingest,
            opts.media_sentinel.as_deref(),
        );
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
            let wire: Vec<serde_json::Value> =
                custom_tools.iter().map(|t| tool_wire_value(t)).collect();
            JinjaValue::from_serialize(&wire)
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
        let extras_has_thinking =
            opts.extras.iter().any(|(k, _)| k == "enable_thinking");
        let base_ctx = if extras_has_thinking {
            minijinja::context! {
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
                messages => messages,
                tools => tools_value,
                add_generation_prompt => opts.add_generation_prompt,
                date_string => date_string,
            }
        } else {
            minijinja::context! {
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
                messages => messages,
                tools => tools_value,
                add_generation_prompt => opts.add_generation_prompt,
                date_string => date_string,
                enable_thinking => prompt.thinking.is_some(),
            }
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
        tmpl.render(ctx).map_err(ChatTemplateError::from_jinja)
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
        let text = self.render_with(prompt, opts)?;
        let breakpoints = collect_breakpoints(prompt);
        let mut partial_texts = Vec::with_capacity(breakpoints.len());
        for bp in breakpoints {
            match render_partial(self, prompt, opts, bp) {
                Ok(s) => partial_texts.push(s),
                Err(_e) => {
                    // Drop this breakpoint — same fail-open posture
                    // tokenize_with_breakpoints uses for non-prefix-
                    // safe partials. A breakpoint we can't render
                    // can't be tokenized either, so the
                    // partial_texts slot is unrecoverable. Warn at
                    // default level — losing a breakpoint silently
                    // is a cache-correctness signal, not a debug
                    // nicety.
                    #[cfg(feature = "axum")]
                    tracing::warn!(
                        target: "drama_llama::chat_template",
                        breakpoint = ?bp,
                        error = %_e,
                        "partial render failed; breakpoint dropped from \
                         partial_texts (cache reuse lost at this position)",
                    );
                }
            }
        }
        Ok(RenderedWithBreakpoints {
            text,
            partial_texts,
        })
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
/// `partial_texts` is ordered canonically by how much of the prompt is
/// included: [`AfterTools`], then [`AfterSystem`], then
/// [`AfterMessage(0)`], [`AfterMessage(1)`], … Only breakpoints that
/// actually exist in the prompt appear.
///
/// [`Session`]: crate::Session
/// [`AfterTools`]: Breakpoint::AfterTools
/// [`AfterSystem`]: Breakpoint::AfterSystem
/// [`AfterMessage(0)`]: Breakpoint::AfterMessage
/// [`AfterMessage(1)`]: Breakpoint::AfterMessage
#[derive(Clone, Debug)]
pub struct RenderedWithBreakpoints {
    /// Full render — equivalent to
    /// [`ChatTemplate::render_with`]'s output.
    pub text: String,
    /// One partial render per breakpoint, in canonical order.
    pub partial_texts: Vec<String>,
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
    /// source hash (see [`image_source_hash`]). When unset, a prompt
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
/// [`AfterTools`]: Breakpoint::AfterTools
/// [`AfterSystem`]: Breakpoint::AfterSystem
/// [`AfterMessage(i)`]: Breakpoint::AfterMessage
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Breakpoint {
    /// After the tools section — any cache marker on any
    /// [`tool::CustomMethodDef`](misanthropic::tool::CustomMethodDef) in
    /// [`Prompt::functions`] produces this, regardless of which
    /// specific method was marked. Coarse by design: per-tool
    /// partial rendering would not produce a byte-prefix of the full
    /// render (the closing `]` of the tools JSON array lands
    /// differently).
    AfterTools,
    /// After the system section — any cache marker on any block in
    /// [`Prompt::system`] produces this. Coarse by design for the
    /// same reason as [`AfterTools`].
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
/// declares. See [`Breakpoint`] for the ordering rule and granularity.
fn collect_breakpoints(prompt: &Prompt) -> Vec<Breakpoint> {
    let mut out = Vec::new();

    let tools_cached = prompt
        .tools
        .as_ref()
        .is_some_and(|defs| defs.iter().any(|m| m.is_cached()));
    if tools_cached {
        out.push(Breakpoint::AfterTools);
    }

    let system_cached =
        prompt.system.as_ref().is_some_and(|c| content_has_cache(c));
    if system_cached {
        out.push(Breakpoint::AfterSystem);
    }

    for (i, m) in prompt.messages.iter().enumerate() {
        if content_has_cache(&m.content) {
            out.push(Breakpoint::AfterMessage(i));
        }
    }

    out
}

/// Does any block inside `content` carry a `cache_control` marker?
fn content_has_cache(content: &Content) -> bool {
    content.0.iter().any(|b| b.is_cached())
}

/// Render `prompt` truncated at `up_to` with
/// `add_generation_prompt=false`. Used as the breakpoint-discovery
/// partial render; does not mutate `prompt` (clones a truncated view).
fn render_partial(
    template: &ChatTemplate,
    prompt: &Prompt,
    opts: &RenderOptions,
    up_to: Breakpoint,
) -> Result<String, ChatTemplateError> {
    let truncated = match up_to {
        Breakpoint::AfterTools => Prompt {
            tools: prompt.tools.clone(),
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
        Breakpoint::AfterSystem => Prompt {
            tools: prompt.tools.clone(),
            system: prompt.system.clone(),
            messages: Vec::new(),
            ..Prompt::default()
        },
        Breakpoint::AfterMessage(i) => Prompt {
            tools: prompt.tools.clone(),
            system: prompt.system.clone(),
            messages: prompt.messages[..=i].to_vec(),
            ..Prompt::default()
        },
    };
    let partial_opts = opts.clone().with_generation_prompt(false);
    template.render_with_env(
        &template.env_permissive,
        &truncated,
        &partial_opts,
    )
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
/// Tokenizes with `parse_special=true` so chat markers
/// (`<|im_start|>`, `<|eot_id|>`, …) resolve to their single
/// special-token IDs — the same convention `Session::prepare_call`
/// uses.
///
/// Used internally by `Session::prepare_call_cached` for the
/// prefix-cache lookup, and exposed publicly so callers can build
/// inspection / diagnostic tools that reproduce exactly the same
/// tokenization the cache machinery sees (see
/// `examples/inspect_prompt.rs`).
///
/// [`Session::prepare_call`]: crate::Session
pub fn tokenize_with_breakpoints<M: Model>(
    model: &M,
    rendered: &RenderedWithBreakpoints,
) -> (Vec<Token>, Vec<usize>) {
    let full_tokens = model.tokenize(&rendered.text, true);
    let mut indices: Vec<usize> =
        Vec::with_capacity(rendered.partial_texts.len());
    for partial in &rendered.partial_texts {
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
fn build_messages(
    prompt: &Prompt,
    reingest: crate::dialect::ReasoningReingest,
    media_sentinel: Option<&str>,
) -> Vec<JinjaValue> {
    let mut out: Vec<JinjaValue> =
        Vec::with_capacity(prompt.messages.len() + 1);
    if let Some(system) = prompt.system.as_ref() {
        out.push(text_message(
            "system",
            flatten_text(system, media_sentinel),
        ));
    }
    for m in &prompt.messages {
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            // Mid-conversation system turns (misanthropic ≥alpha.2).
            // HF templates broadly accept repeated system messages.
            Role::System => "system",
        };
        append_message(&mut out, role, &m.content, reingest, media_sentinel);
    }
    out
}

/// Emit one or more Jinja messages for a single misanthropic Message.
fn append_message(
    out: &mut Vec<JinjaValue>,
    role: &str,
    content: &Content,
    reingest: crate::dialect::ReasoningReingest,
    media_sentinel: Option<&str>,
) {
    let blocks: Vec<&Block> = content.0.iter().collect();

    // User turn: split ToolResult blocks into their own "tool" messages,
    // collect remaining text/thought into a trailing user message.
    if role == "user" {
        let mut residual = String::new();
        for b in &blocks {
            match b {
                Block::ToolResult { result } => {
                    let content =
                        flatten_text(&result.content, media_sentinel);
                    out.push(tool_result_message(&result.tool_use_id, content));
                }
                other => {
                    append_block_text(&mut residual, other, media_sentinel)
                }
            }
        }
        if !residual.is_empty() {
            out.push(text_message(role, residual));
        }
        return;
    }

    // Assistant turn. Thoughts route by convention: inline
    // `<think>…</think>` in content (legacy Qwen templates
    // reconstruct from content), or concatenated into the
    // `reasoning`/`reasoning_content` fields (Gemma 4/DeepSeek-style
    // templates own the markers; inlining would pollute content).
    use crate::dialect::ReasoningReingest;
    let calls: Vec<&crate::prompt::ToolUse> = blocks
        .iter()
        .filter_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .collect();
    // Text splits by block order around the first ToolUse: prose the
    // model emitted BEFORE calling (announce-then-call) vs after.
    // Reordering them would invert causality in the transcript, so
    // causality-aware templates get both halves; stock templates read
    // the merged `content` and keep their own layout.
    let mut reasoning = String::new();
    let mut content_pre = String::new();
    let mut content_post = String::new();
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

    // One message carrying every call: the shape template
    // `tool_calls` loops iterate, so parallel calls re-render intact.
    if !calls.is_empty() {
        out.push(tool_call_message(
            role,
            (&content_pre, &content_post),
            &calls,
            &reasoning,
            reingest,
        ));
        return;
    }
    out.push(assistant_text_message(role, content_pre, &reasoning));
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
    (content_pre, content_post): (&str, &str),
    calls: &[&crate::prompt::ToolUse],
    reasoning: &str,
    reingest: crate::dialect::ReasoningReingest,
) -> JinjaValue {
    let tool_calls: Vec<JinjaValue> = calls
        .iter()
        .map(|call| {
            minijinja::context! {
                id => call.id.as_ref(),
                function => minijinja::context! {
                    name => call.name.as_ref(),
                    arguments => JinjaValue::from_serialize(&call.input),
                },
            }
        })
        .collect();
    let content = format!("{content_pre}{content_post}");
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

/// Flatten any [`Content`] to a single string using [`append_block_text`]
/// for each part.
fn flatten_text(content: &Content, media_sentinel: Option<&str>) -> String {
    let mut out = String::new();
    for b in &content.0 {
        append_block_text(&mut out, b, media_sentinel);
    }
    out
}

/// Append a single block's user-visible text to `out`. Tool-use and
/// tool-result blocks are handled at the message level (see
/// [`append_message`]); here they contribute nothing.
fn append_block_text(
    out: &mut String,
    block: &Block,
    media_sentinel: Option<&str>,
) {
    match block {
        Block::Text { text, .. } => out.push_str(text),
        Block::Thought { thought, .. } => {
            out.push_str("<think>");
            out.push_str(thought);
            out.push_str("</think>");
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
                out.push_str(&media_marker(
                    sentinel,
                    &image_source_hash(image),
                ));
            }
        }
        // Tool-use / tool-result blocks are handled at the message
        // level; redacted thoughts, documents, and the server-tool
        // block family contribute no template-visible text.
        _ => {}
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

/// A media-bearing render split on its sentinel: `n + 1` text
/// segments interleaved with `n` image source hashes, in render
/// order. Imageless renders come back as one segment and no hashes.
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
    /// Text between (around) media markers;
    /// `segments.len() == source_hashes.len() + 1`.
    pub segments: Vec<&'a str>,
    /// Source hash of the image at each marker (see
    /// [`image_source_hash`]).
    pub source_hashes: Vec<[u8; 32]>,
}

/// Split `text` on `<{sentinel}:{hex64}>` markers.
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
pub(crate) fn split_media_render<'a>(
    text: &'a str,
    sentinel: &str,
) -> Result<SplitRender<'a>, usize> {
    let pattern = format!("<{sentinel}:");
    let mut segments = Vec::new();
    let mut source_hashes = Vec::new();
    let mut rest = text;
    let mut base = 0usize;
    while let Some(at) = rest.find(&pattern) {
        let hex_start = at + pattern.len();
        let hex_end = hex_start + 64;
        let ok = rest.len() > hex_end
            && rest.as_bytes()[hex_end] == b'>'
            && rest[hex_start..hex_end]
                .bytes()
                .all(|b| b.is_ascii_hexdigit());
        if !ok {
            return Err(base + at);
        }
        let mut hash = [0u8; 32];
        for (i, byte) in hash.iter_mut().enumerate() {
            *byte = u8::from_str_radix(
                &rest[hex_start + i * 2..hex_start + i * 2 + 2],
                16,
            )
            .expect("checked hexdigit above");
        }
        segments.push(&rest[..at]);
        source_hashes.push(hash);
        rest = &rest[hex_end + 1..];
        base += hex_end + 1;
    }
    segments.push(rest);
    Ok(SplitRender {
        segments,
        source_hashes,
    })
}

// ===========================================================================
// Jinja-side helpers
// ===========================================================================

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
    ///    — fires on [`Breakpoint::AfterTools`] where `messages` is empty.
    /// 2. The `multi_step_tool` walk: scans `messages[::-1]` for a
    ///    user-role message whose content isn't a `<tool_response>...`
    ///    wrapper, raises `No user query found in messages.` if none
    ///    found. Fires on [`Breakpoint::AfterSystem`] where the only
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

    use misanthropic::prompt::message::CacheControl;
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
        assert_eq!(collect_breakpoints(&prompt), Vec::<Breakpoint>::new());
    }

    #[test]
    fn test_collect_breakpoints_tools_only() {
        let prompt = Prompt {
            tools: Some(vec![tool_cached("cached_tool").into()]),
            ..Prompt::default()
        };
        assert_eq!(collect_breakpoints(&prompt), vec![Breakpoint::AfterTools]);
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
                Breakpoint::AfterTools,
                Breakpoint::AfterSystem,
                Breakpoint::AfterMessage(0),
                Breakpoint::AfterMessage(2),
            ]
        );
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
        let partial =
            render_partial(&tmpl(), &prompt, &opts, Breakpoint::AfterSystem)
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

    #[test]
    fn test_render_with_breakpoints_no_cache_control() {
        let out = tmpl()
            .render_with_breakpoints(
                &simple_prompt(),
                &RenderOptions::default().with_generation_prompt(true),
            )
            .unwrap();
        assert!(out.partial_texts.is_empty());
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
        assert_eq!(
            out.partial_texts.len(),
            1,
            "one cache marker → one partial"
        );
        for (i, p) in out.partial_texts.iter().enumerate() {
            assert!(
                !p.ends_with(GEN),
                "partial {i} must not end with generation prompt: {p:?}"
            );
        }
    }

    /// Qwen3-family templates raise `No user query found in messages.`
    /// whenever `messages` contains no non-tool-response user-role
    /// entry — exactly the state [`render_partial`] produces for
    /// [`Breakpoint::AfterSystem`] (truncates `messages` to empty).
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
            Breakpoint::AfterSystem,
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

    /// Same idea for `Breakpoint::AfterTools`: `render_partial` truncates
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
            Breakpoint::AfterTools,
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
            out.partial_texts.len(),
            1,
            "AfterSystem breakpoint must survive partial rendering"
        );
        assert!(out.partial_texts[0].contains("You are agent."));
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
        assert_eq!(rendered.partial_texts.len(), 1);

        let (full_tokens, indices) =
            tokenize_with_breakpoints(&engine.model, &rendered);
        assert_eq!(indices.len(), 1, "exactly one breakpoint expected");
        let idx = indices[0];
        assert!(idx <= full_tokens.len());

        // The partial's tokens equal the full's tokens up to `idx`.
        let partial_tokens =
            engine.model.tokenize(&rendered.partial_texts[0], true);
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

        let split = split_media_render(&out, sentinel).unwrap();
        assert_eq!(split.source_hashes, vec![src_hash]);
        assert_eq!(split.segments.len(), 2);
        assert!(split.segments[0].ends_with("What breed is "));
        assert!(split.segments[1].starts_with(" shown here?"));
        // Reassembly is lossless.
        let reassembled = format!(
            "{}{}{}",
            split.segments[0], marker, split.segments[1]
        );
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
        let split = split_media_render(&out, sentinel).unwrap();
        assert!(split.source_hashes.is_empty());
        assert_eq!(split.segments.len(), 1);
        assert!(split.segments[0].contains(&evil), "content round-trips");
    }

    #[test]
    fn split_media_render_rejects_mangled_markers() {
        let sentinel = "0123456789abcdef0123456789abcdef";
        // Truncated mid-hash: parse must fail loudly, never fall back
        // to treating the mangled marker as content.
        let mangled = format!("text <{sentinel}:abc123 more");
        assert!(split_media_render(&mangled, sentinel).is_err());
        // Sentinel-free text is one segment.
        let clean = split_media_render("no media here", sentinel).unwrap();
        assert_eq!(clean.segments, vec!["no media here"]);
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
                content: Content(vec![Block::ToolResult {
                    result: result.into(),
                }]),
            }],
            ..Prompt::default()
        };
        let opts = RenderOptions::default().with_media_sentinel(sentinel);
        let out = tmpl().render_with(&p, &opts).unwrap();
        let split = split_media_render(&out, sentinel).unwrap();
        assert_eq!(
            split.source_hashes.len(),
            1,
            "tool-result images render markers too"
        );
    }
}
