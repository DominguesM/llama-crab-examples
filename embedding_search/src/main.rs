//! `embedding_search` — semantic search over a small fixed corpus.
//!
//! Embeds the query + a 4-document corpus with `bge-small-en-v1.5`
//! and ranks the documents by cosine similarity.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin embedding_search
//! ```
//!
//! Override the query as the third positional argument, or via
//! `LLAMA_CRAB_QUERY` env var.

use anyhow::{Context, Result};
use llama_crab::context::params::PoolingType;
use llama_crab::{Llama, LlamaParams};
use std::time::Instant;

const DEFAULT_HF_REPO: &str = "nomic-ai/nomic-embed-text-v1.5-GGUF";
const DEFAULT_HF_FILE: &str = "nomic-embed-text-v1.5.Q4_K_M.gguf";
const DEFAULT_QUERY: &str = "What programming language is safest?";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let hf_repo = args
        .next()
        .unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
    let hf_filename = args.next().unwrap_or_else(|| DEFAULT_HF_FILE.to_string());
    let query = args
        .next()
        .or_else(|| std::env::var("LLAMA_CRAB_QUERY").ok())
        .unwrap_or_else(|| DEFAULT_QUERY.to_string());

    eprintln!("🦀 llama-crab embeddings example");
    eprintln!("   hf_repo    : {hf_repo}");
    eprintln!("   hf_filename: {hf_filename}");
    eprintln!("   query      : {query}");
    eprintln!();

    let start = Instant::now();
    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(512)
            .with_embeddings(true)
            .with_pooling_type(PoolingType::Cls),
    )
    .with_context(|| format!("failed to load {hf_repo}/{hf_filename}"))?;
    eprintln!("✓ model loaded in {:.2}s", start.elapsed().as_secs_f64());

    // Tiny fixed corpus.
    let corpus: &[(&str, &str)] = &[
        (
            "doc-1",
            "Rust is a memory-safe systems language without a garbage collector.",
        ),
        (
            "doc-2",
            "Python is a high-level dynamic language with duck typing.",
        ),
        (
            "doc-3",
            "The Eiffel Tower is one of the most visited monuments in the world.",
        ),
        (
            "doc-4",
            "Borrow checking enforces lifetimes at compile time in Rust.",
        ),
    ];

    eprintln!("\n▶ embedding {} corpus items + 1 query", corpus.len() + 1);
    let q_vec = llama.embed(&query, true)?;
    let mut scored: Vec<(&str, f32)> = Vec::with_capacity(corpus.len());
    for (id, text) in corpus {
        let v = llama.embed(text, true)?;
        scored.push((id, cosine(&q_vec, &v)));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("\n📊 results (cosine similarity, higher = more similar):");
    for (id, score) in &scored {
        let text = corpus.iter().find(|(i, _)| *i == *id).unwrap().1;
        eprintln!("   {score:>5.3}  {id:<6}  {text}");
    }

    println!("\nQuery: {query}");
    println!("Top match: {} (cosine = {:.3})", scored[0].0, scored[0].1);
    let top = corpus.iter().find(|(i, _)| *i == scored[0].0).unwrap().1;
    println!("  > {top}");
    Ok(())
}
