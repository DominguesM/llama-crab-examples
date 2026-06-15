//! `structured` — JSON-schema grammar-constrained decoding.
//!
//! Converts a JSON Schema to a GBNF grammar, wraps it in a sampler
//! chain together with the greedy sampler, and asks the model to emit
//! a JSON object that matches the schema. The result is parsed back
//! to a `serde_json::Value` and pretty-printed.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin structured
//! ```

use anyhow::{anyhow, Result};
use llama_crab::chat::{render_builtin, BuiltinTemplate};
use llama_crab::high_level::chat_completion::ChatMessage;
use llama_crab::high_level::completion::{json_schema_grammar, CompletionOptions};
use llama_crab::sampling::LlamaSampler;
use llama_crab::{Llama, LlamaParams, Role};
use serde_json::{json, Value};

const DEFAULT_HF_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const DEFAULT_HF_FILE: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let hf_repo = args
        .next()
        .unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
    let hf_filename = args.next().unwrap_or_else(|| DEFAULT_HF_FILE.to_string());

    let schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"]
    });
    let grammar_text = json_schema_grammar(&schema)?;
    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(1024),
    )?;
    let grammar = unsafe { LlamaSampler::grammar(llama.model(), &grammar_text, "root")? };
    let greedy =
        LlamaSampler::greedy().ok_or_else(|| anyhow!("failed to create greedy sampler"))?;
    let mut sampler = LlamaSampler::chain(vec![grammar, greedy], false)
        .ok_or_else(|| anyhow!("failed to create sampler chain"))?;
    let messages = vec![
        ChatMessage::new(
            Role::System,
            "You create compact structured data. Return only JSON.",
        ),
        ChatMessage::new(
            Role::User,
            "Create one fictional person. Return only JSON with keys name and age.",
        ),
    ];
    let prompt = render_builtin(BuiltinTemplate::ChatMl, &messages, &[], true);
    let resp =
        llama.create_completion_with_sampler(&prompt, CompletionOptions::new(96), &mut sampler)?;
    let person: Value = serde_json::from_str(resp.text.trim())?;

    println!("{}", serde_json::to_string_pretty(&person)?);
    Ok(())
}
