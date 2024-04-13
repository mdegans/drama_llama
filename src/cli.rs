use std::path::PathBuf;

use clap::Parser;

use llama_cpp_sys::{
    llama_context_default_params, llama_context_params,
    llama_model_default_params, llama_model_params,
};

#[derive(Debug, Parser)]
pub struct Args {
    /// Path to the model
    #[arg(short, long)]
    pub model: PathBuf,
    /// Context size
    #[arg(short, long, default_value_t = 1024)]
    pub context: u32,
    /// Disable on-by-default GPU acceleration
    #[arg(short, long, default_value_t = false)]
    pub no_gpu: bool,
}

impl Args {
    /// Create `llama_model_params` from `Args`. Defaults are used for fields
    /// not specified in `Args`.
    pub fn model_params(&self) -> llama_model_params {
        self.into()
    }

    /// Create `llama_context_params` from `Args`. Defaults are used for fields
    /// not specified in `Args`.
    pub fn context_params(&self) -> llama_context_params {
        self.into()
    }
}

impl From<&Args> for llama_model_params {
    fn from(args: &Args) -> Self {
        // Safety: This returns POD and makes no allocations for the pointer
        // fields, which are optional and initialized to null.
        let mut params = unsafe { llama_model_default_params() };
        params.n_gpu_layers = if args.no_gpu { 0 } else { 1000 };

        params
    }
}

impl From<&Args> for llama_context_params {
    fn from(args: &Args) -> Self {
        // Safety: same as above
        let mut params = unsafe { llama_context_default_params() };
        params.n_ctx = args.context;

        params
    }
}
