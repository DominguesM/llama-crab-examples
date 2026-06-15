//! `embeddings` — extract a single embedding vector for an input text.
//!
//! Default model is `bge-small-en-v1.5` (Q4_K_M, ~30 MB), an embedding
//! GGUF that fits in any modern CPU's RAM. The example tokenizes the
//! input, encodes it, and prints the L2-normalized vector alongside
//! its dimension, L2 norm and a 8-element preview.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin embeddings
//! ```

use anyhow::Result;
use llama_crab::context::params::PoolingType;
use llama_crab::{Llama, LlamaParams};

const DEFAULT_HF_REPO: &str = "nomic-ai/nomic-embed-text-v1.5-GGUF";
const DEFAULT_HF_FILE: &str = "nomic-embed-text-v1.5.Q4_K_M.gguf";
const DEFAULT_TEXT: &str = "Hello, world!";

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let hf_repo = args
        .next()
        .unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
    let hf_filename = args.next().unwrap_or_else(|| DEFAULT_HF_FILE.to_string());
    let text = args.next().unwrap_or_else(|| DEFAULT_TEXT.to_string());

    // nomic-embed-text-v1.5 is a BERT-style encoder; CLS pooling
    // matches its official configuration. Mean pooling also works.
    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(512)
            .with_embeddings(true)
            .with_pooling_type(PoolingType::Cls),
    )?;
    let tokens = llama.model().tokenize(&text, true, false)?;
    let embedding = llama.embed(&text, true)?;
    let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    let preview: Vec<f32> = embedding.iter().copied().take(8).collect();

    use std::io::Write;
    let mut out = std::io::stdout().lock();
    writeln!(out, "text: {text}")?;
    writeln!(out, "tokens: {tokens:?}")?;
    writeln!(out, "embedding_dim: {}", embedding.len())?;
    writeln!(out, "embedding_l2_norm: {norm:.6}")?;
    writeln!(out, "embedding_preview: {preview:.6?}")?;
    out.flush()?;
    Ok(())
}
