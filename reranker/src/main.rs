//! `reranker` — embedding-based cross-encoder reranking demo.
//!
//! Uses the `Llama::rerank` helper with `bge-reranker-base` (a
//! cross-encoder that scores `(query, document)` pairs) to rank a
//! short list of documents. Higher score = more relevant.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin reranker
//! ```

use anyhow::Result;
use llama_crab::context::params::PoolingType;
use llama_crab::{Llama, LlamaParams};
use std::cmp::Ordering;

const DEFAULT_HF_REPO: &str = "turingevo/bge-reranker-base-Q4_K_M-GGUF";
const DEFAULT_HF_FILE: &str = "bge-reranker-base-q4_k_m.gguf";
const QUERY: &str = "safe systems programming language";
const DOCUMENTS: &[&str] = &[
    "Rust is a memory-safe systems programming language.",
    "Paris is the capital city of France.",
    "Bananas are yellow fruit rich in potassium.",
];

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let hf_repo = args
        .next()
        .unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
    let hf_filename = args.next().unwrap_or_else(|| DEFAULT_HF_FILE.to_string());

    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(512)
            .with_embeddings(true)
            .with_pooling_type(PoolingType::Rank),
    )?;
    let scores = llama.rerank(QUERY, DOCUMENTS)?;

    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    println!("query: {QUERY}");
    for (rank, (idx, score)) in ranked.iter().enumerate() {
        println!(
            "{:>2}. score={score:+.4} document={}",
            rank + 1,
            DOCUMENTS[*idx]
        );
    }
    Ok(())
}
