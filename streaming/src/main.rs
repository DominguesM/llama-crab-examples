//! `streaming` — high-level callback-driven streaming completion.
//!
//! Each `CompletionChunk` is written to stdout as it becomes available.
//! The callback can return `StreamControl::Stop` to abort generation
//! (the example returns `Stop` if writing to stdout fails).
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin streaming
//! ```

use std::error::Error;
use std::io::{self, Write};

use llama_crab::{CompletionOptions, Llama, LlamaParams, StreamControl};

const DEFAULT_HF_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const DEFAULT_HF_FILE: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const DEFAULT_PROMPT: &str = "Write one short sentence about Rust.";
const DEFAULT_MAX_TOKENS: usize = 64;

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let hf_repo = args
        .next()
        .unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
    let hf_filename = args.next().unwrap_or_else(|| DEFAULT_HF_FILE.to_string());
    let prompt = args
        .next()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());
    let max_tokens = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_MAX_TOKENS);

    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(512)
            .with_n_gpu_layers(99),
    )?;

    let mut stdout = io::stdout().lock();
    let mut write_error: Option<io::Error> = None;
    let _completion =
        llama.create_completion_stream(&prompt, CompletionOptions::new(max_tokens), |chunk| {
            if let Err(err) = write!(stdout, "{}", chunk.text).and_then(|_| stdout.flush()) {
                write_error = Some(err);
                return StreamControl::Stop;
            }
            StreamControl::Continue
        })?;

    if let Some(err) = write_error {
        return Err(err.into());
    }

    writeln!(stdout)?;
    Ok(())
}
