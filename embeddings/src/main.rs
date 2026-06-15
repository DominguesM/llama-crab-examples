//! `embeddings` — extract a single embedding vector for an input text.
//!
//! Default model is `nomic-embed-text-v1.5` (Q4_K_M, ~80 MB), an
//! embedding GGUF that fits in any modern CPU's RAM. The example
//! tokenizes the input, encodes it, and prints the L2-normalized
//! vector alongside its dimension, L2 norm and a 8-element preview.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin embeddings
//! ```

use anyhow::Result;
use llama_crab::batch::LlamaBatch;
use llama_crab::context::params::PoolingType;
use llama_crab::{Llama, LlamaParams};

const DEFAULT_HF_REPO: &str = "CompendiumLabs/bge-small-en-v1.5-gguf";
const DEFAULT_HF_FILE: &str = "bge-small-en-v1.5-q4_k_m.gguf";
const DEFAULT_TEXT: &str = "Hello, world!";

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let hf_repo = args
        .next()
        .unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
    let hf_filename = args.next().unwrap_or_else(|| DEFAULT_HF_FILE.to_string());
    let text = args.next().unwrap_or_else(|| DEFAULT_TEXT.to_string());

    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(512)
            .with_n_batch(512)
            .with_n_ubatch(512)
            .with_embeddings(true)
            .with_pooling_type(PoolingType::Mean),
    )?;
    // For Bert-style encoder models the BOS slot is already the
    // `[CLS]` token; asking llama.cpp to prepend a BOS on top would
    // duplicate the prefix and the encode loop hangs.
    let tokens = llama.model().tokenize(&text, false, false)?;
    println!("text: {text}");
    println!("tokens: {tokens:?}");

    // Manual encode loop. `Llama::embed` is the recommended shortcut
    // but several encoder-only GGUF builds in the wild hang inside
    // it on Apple Silicon — the manual flow goes through the same
    // encode + embeddings_seq path and is reliable.
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    for (i, &t) in tokens.iter().enumerate() {
        let logits = i + 1 == tokens.len();
        batch
            .add(t, i as i32, &[0], logits)
            .map_err(anyhow::Error::from)?;
    }
    llama.context().encode(&batch)?;
    // `embeddings_ith` returns the vector for the *token* at position
    // `tokens.len() - 1` of the last batch. For pooled embeddings the
    // last position is the EOS token — the natural place to read
    // Mean or CLS pooling from depending on the model.
    let mut embedding: Vec<f32> = llama.context().embeddings_ith((tokens.len() - 1) as i32)?.to_vec();

    let norm_before = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    {
        let sum: f32 = embedding.iter().map(|v| v * v).sum();
        let inv = if sum > 0.0 { 1.0 / sum.sqrt() } else { 1.0 };
        for v in &mut embedding {
            *v *= inv;
        }
    }
    let norm_after = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    let preview: Vec<f32> = embedding.iter().copied().take(8).collect();

    println!("embedding_dim: {}", embedding.len());
    println!("embedding_l2_norm_before: {norm_before:.6}");
    println!("embedding_l2_norm_after:  {norm_after:.6}");
    println!("embedding_preview: {preview:.6?}");
    Ok(())
}
