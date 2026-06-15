//! `speculative` — prompt-lookup speculative decoding demo.
//!
//! `PromptLookupDecoding` drafts candidate token continuations by
//! looking for repeated n-grams in the prompt — useful as a compact
//! illustration of the speculative API. For a small text-only model
//! it produces a few draft tokens and the example prints them.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin speculative
//! ```

use anyhow::{anyhow, Result};
use llama_crab::speculative::{DraftModel, PromptLookupDecoding};
use llama_crab::{Llama, LlamaParams};

const DEFAULT_HF_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const DEFAULT_HF_FILE: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const DEFAULT_PROMPT: &str = "Rust is fast and memory safe. Rust is fast";

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

    let llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(1024),
    )?;
    let prompt_tokens = llama.model().tokenize(&prompt, true, true)?;

    // n_gram=3, max_draft=8. Tweak to taste.
    let draft = PromptLookupDecoding::new(3, 8);
    let drafted = draft.draft(&prompt_tokens, 8);
    if drafted.is_empty() {
        return Err(anyhow!("prompt lookup produced no draft tokens"));
    }

    let drafted_text = llama.model().detokenize(&drafted, false)?;
    println!("prompt> {prompt}");
    println!("drafted token ids> {drafted:?}");
    println!("drafted text> {}", drafted_text.trim());
    Ok(())
}
