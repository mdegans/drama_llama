#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["jinja2>=3.1"]
# ///
"""Reference implementation Jinja renderer — cross-check against
drama_llama's minijinja output.

Usage:
    render_jinja.py <template_path> <variables_json_path>

Stdout: the rendered prompt bytes. Errors go to stderr with non-zero
exit.

Why this exists: minijinja (Rust, used by drama_llama at runtime) and
jinja2 (Python, the reference implementation + what HuggingFace uses
for `tokenizer.apply_chat_template`) can drift on edge cases. This
script gives us a neutral ground-truth renderer for the same Jinja
template source + variables, so fixture tests can assert
byte-equality in both directions:

  our_output == hand_mocked_expected  # regression lock
  our_output == python_jinja_output   # spec-compliance cross-check

If they ever diverge, we know whether the issue is in our template
conversion layer (`build_messages` / `tool_wire_value` in
chat_template.rs) or in minijinja's Jinja implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from jinja2.exceptions import TemplateError


def render(template_path: Path, variables_path: Path) -> str:
    source = template_path.read_text(encoding="utf-8")
    variables = json.loads(variables_path.read_text(encoding="utf-8"))

    # Match minijinja-contrib's `pycompat` + drama_llama's Environment
    # setup as closely as possible: no autoescape (chat templates emit
    # raw text, not HTML), strict undefined so typos surface instead
    # of silently rendering empty strings.
    env = Environment(
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=False,
        undefined=StrictUndefined,
    )
    # Override `tojson` to emit compact JSON (`{"a":1}` — no spaces
    # after `:` or `,`). Minijinja's `tojson` is compact by default,
    # and so is Go's `encoding/json` — which is what ollama's runtime
    # template uses and therefore what cogito was trained on. Python
    # jinja2's default `tojson` is pretty-printed (via `json.dumps`'s
    # default `', '` / `': '` separators) and is the outlier here.
    env.policies["json.dumps_kwargs"] = {
        "sort_keys": True,
        "separators": (",", ":"),
        "ensure_ascii": False,
    }
    # Register stubs for the helpers drama_llama installs in its
    # Environment. We don't need real implementations — template
    # shapes that call them can be excluded from cross-check, or you
    # can add a var for the expected value.
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now

    try:
        template = env.from_string(source)
        return template.render(**variables)
    except TemplateError as e:
        print(f"jinja2 error: {e}", file=sys.stderr)
        sys.exit(2)


def _raise_exception(msg: str) -> None:
    raise RuntimeError(msg)


def _strftime_now(fmt: str) -> str:
    # drama_llama's tests pin `date_string` explicitly; if a template
    # reaches for this helper we want the test to be deterministic,
    # not the wall clock. Pass `strftime_now` as a variable instead
    # when you need it.
    raise RuntimeError(
        "strftime_now called during cross-check render — pin date_string "
        "via variables instead of relying on the clock."
    )


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    template_path = Path(sys.argv[1])
    variables_path = Path(sys.argv[2])
    sys.stdout.write(render(template_path, variables_path))


if __name__ == "__main__":
    main()
