# `settings_tool`

Is a very simple tool for editing `drama_llama` options via a gui. It's mostly
to test the `egui` feature but it may be useful to generate configuration files.

Run it like:

```text
cargo run --bin settings_tool --features="egui,toml,serde,serde_json"
```

## Notes

- TOML cannot store the settings properly because it doesn't support u128, or at
  least the `toml` crate doesn't. It's there because at some point we might
  store the seed differently (like two u64s).
