//! `stateful_chat` — interactive multi-turn chat REPL.
//!
//! The model is loaded once; each user turn appends the previous
//! assistant reply to the history and asks for a new one. The history
//! is **not** truncated between turns (the context size is the only
//! limit).
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin stateful_chat
//! ```
//!
//! Commands while running:
//!   * `/exit` — quit
//!   * `/clear` — reset the history
//!   * `/save` — print the conversation so far
//!   * anything else — user message

use anyhow::{Context, Result};
use llama_crab::chat::BuiltinTemplate;
use llama_crab::high_level::chat_completion::{create_chat_completion_with, ChatMessage};
use llama_crab::{Llama, LlamaParams, Role};
use std::io::{self, BufRead, Write};
use std::time::Instant;

const DEFAULT_HF_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const DEFAULT_HF_FILE: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let hf_repo = args
        .next()
        .unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
    let hf_filename = args.next().unwrap_or_else(|| DEFAULT_HF_FILE.to_string());

    eprintln!("🦀 llama-crab interactive chat");
    eprintln!("   hf_repo    : {hf_repo}");
    eprintln!("   hf_filename: {hf_filename}");
    eprintln!("   commands: /exit  /clear  /save");
    eprintln!();

    let start = Instant::now();
    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(4096)
            .with_n_threads(4),
    )
    .with_context(|| format!("failed to load {hf_repo}/{hf_filename}"))?;
    eprintln!(
        "✓ model loaded in {:.2}s  ({} layers)",
        start.elapsed().as_secs_f64(),
        llama.model().n_layer()
    );

    let mut history: Vec<ChatMessage> = vec![ChatMessage::new(
        Role::System,
        "You are a helpful, concise assistant. Always reply in English, in under 2 sentences.",
    )];

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("\n> ");
        stdout.flush().ok();
        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            eprintln!("(EOF)");
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/exit" | "/quit" | "/q" => {
                eprintln!("bye.");
                break;
            }
            "/clear" => {
                history.truncate(1);
                eprintln!("(history cleared)");
                continue;
            }
            "/save" => {
                let json = serde_json::to_string_pretty(&history)?;
                eprintln!("{json}");
                continue;
            }
            _ => {}
        }

        history.push(ChatMessage::new(Role::User, input.to_string()));
        let t0 = Instant::now();
        match create_chat_completion_with(&mut llama, &history, BuiltinTemplate::ChatMl, &[], 96) {
            Ok(resp) => {
                let reply = resp.content.trim().to_string();
                let dt = t0.elapsed();
                eprintln!("  ({:.2}s)", dt.as_secs_f64());
                println!("assistant> {reply}");
                history.push(ChatMessage::new(Role::Assistant, reply));
            }
            Err(e) => {
                eprintln!("error: {e}");
                history.pop();
            }
        }
    }
    Ok(())
}
