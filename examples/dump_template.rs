//! Dump the chat template + key metadata for the symlinked model.
//! Run with: `cargo run --example dump_template`
use std::path::PathBuf;

use drama_llama::Model;

fn main() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf");
    let model = Model::from_file(path, None).expect("load model");
    let tmpl = model
        .get_meta("tokenizer.chat_template")
        .unwrap_or_else(|| String::from("(no tokenizer.chat_template)"));
    println!("=== tokenizer.chat_template ({} bytes) ===", tmpl.len());
    println!("{tmpl}");
    println!("=== end ===");
    println!("bos token: {:?}", model.token_to_piece(model.bos()));
    println!("eos token: {:?}", model.token_to_piece(model.eos()));
    println!("eot token: {:?}", model.token_to_piece(model.eot()));
}
