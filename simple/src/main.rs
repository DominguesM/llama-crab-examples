//! `simple` — the minimal one-shot text completion.
//!
//! By default the example points at Qwen2.5 0.5B Instruct on Hugging
//! Face Hub. The `hf-hub` feature of `llama-crab` downloads it on the
//! first run.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin simple
//! ```
//!
//! Or pass a different model:
//!
//! ```bash
//! cargo run --release --bin simple -- \
//!     TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
//!     tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
//!     "Once upon a time"
//! ```

use anyhow::Result;
use llama_crab::{Llama, LlamaParams};

const DEFAULT_HF_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const DEFAULT_HF_FILE: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const DEFAULT_PROMPT: &str = "Once upon a time";

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let hf_repo = args
        .next()
        .unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
    let hf_filename = args.next().unwrap_or_else(|| DEFAULT_HF_FILE.to_string());
    let prompt = args
        .next()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(512),
    )?;
    let resp = llama.create_completion(&prompt, 64)?;
    println!("{}", resp.text);
    Ok(())
}
