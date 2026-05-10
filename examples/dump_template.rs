//! Dump the chat template + key metadata for a GGUF model.
//!
//! ```bash
//! cargo run --example dump_template                     # models/model.gguf
//! cargo run --example dump_template -- /path/to/x.gguf  # explicit path
//! ```
use std::path::PathBuf;

use drama_llama::LlamaCppModel;

fn main() {
    let path = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
    );
    let model = LlamaCppModel::from_file(path, None).expect("load model");
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
