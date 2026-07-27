#!/usr/bin/env python3
"""Read tokenizer.chat_template out of a GGUF without loading weights.

Parses only the GGUF metadata header, so it works on any size model in
milliseconds. Two uses:

  # dump the embedded template
  python3 scripts/gguf_template.py models/foo.gguf

  # compare against a baked detection key, the way baked::detect does
  # (byte equality, trailing whitespace ignored) — exit 0 on match
  python3 scripts/gguf_template.py models/foo.gguf \
      --compare templates/gptoss-gguf.jinja

The compare mode answers the load-bearing rung-2 question for a new
quant (see src/baked.rs and issue #88): will Session recognize this
GGUF's embedded template and apply the baked cache-stable replacement,
or fall through to the best-effort tier? On mismatch the embedded
template is written next to the reference as `<reference>.embedded`
for diffing, and the exit code is 1.
"""
import argparse
import struct
import sys


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def _skip_value(f, vtype):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in sizes:
        f.read(sizes[vtype])
    elif vtype == 8:
        _read_str(f)
    elif vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            _skip_value(f, etype)
    else:
        raise ValueError(f"unknown gguf value type {vtype}")


def chat_template(path):
    """The embedded template string, or None when the GGUF has none."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF")
        (version,) = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported gguf version {version}")
        f.read(8)  # tensor count
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == b"tokenizer.chat_template":
                if vtype != 8:
                    raise ValueError(f"template kv has type {vtype}, not string")
                return _read_str(f).decode("utf-8")
            _skip_value(f, vtype)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gguf", help="path to the .gguf file")
    parser.add_argument(
        "--compare",
        metavar="REFERENCE",
        help="reference .jinja to compare against instead of dumping",
    )
    args = parser.parse_args()

    embedded = chat_template(args.gguf)
    if embedded is None:
        print(f"{args.gguf}: no tokenizer.chat_template", file=sys.stderr)
        return 2

    if args.compare is None:
        sys.stdout.write(embedded)
        return 0

    with open(args.compare, encoding="utf-8") as f:
        reference = f.read()
    if embedded.rstrip() == reference.rstrip():
        print(f"MATCH: {args.gguf} == {args.compare} ({len(embedded)} bytes)")
        return 0
    dump = args.compare + ".embedded"
    with open(dump, "w", encoding="utf-8") as f:
        f.write(embedded)
    print(f"MISMATCH: {args.gguf} vs {args.compare}")
    print(f"embedded template written to {dump} for diffing")
    print("rung 2 will NOT fire for this GGUF — it lands on the")
    print("best-effort tier unless a sidecar is installed or a second")
    print("detection key is added to src/baked.rs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
