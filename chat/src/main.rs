//! `chat` — one-shot chat completion using a built-in chat template.
//!
//! Renders a short `Vec<ChatMessage>` with the ChatML template and
//! prints the assistant response. Default model is Qwen2.5 0.5B
//! Instruct; override via CLI args.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin chat
//! ```

use anyhow::Result;
use llama_crab::chat::BuiltinTemplate;
use llama_crab::high_level::chat_completion::{create_chat_completion_with, ChatMessage};
use llama_crab::{Llama, LlamaParams, Role};

const DEFAULT_HF_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const DEFAULT_HF_FILE: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

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
            .with_n_ctx(2048)
            .with_n_gpu_layers(99),
    )?;
    let history = vec![
        ChatMessage::new(
            Role::System,
            "You are a helpful assistant. Always answer in English. Be concise.",
        ),
        ChatMessage::new(
            Role::User,
            "Introduce yourself in one short English sentence.",
        ),
    ];
    let resp =
        create_chat_completion_with(&mut llama, &history, BuiltinTemplate::ChatMl, &[], 128)?;
    println!("assistant> {}", resp.content);
    Ok(())
}
