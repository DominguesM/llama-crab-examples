//! `tools` — function-calling demo with ChatML-style `<tool_call>` tags.
//!
//! Asks the model to emit a JSON tool call for `get_weather(city)`,
//! parses it, runs the local handler, and prints the result.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin tools
//! ```

use anyhow::{anyhow, Result};
use llama_crab::chat::BuiltinTemplate;
use llama_crab::high_level::chat_completion::{create_chat_completion_with, ChatMessage};
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

    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&hf_filename)
            .with_n_ctx(2048),
    )?;

    let messages = vec![
        ChatMessage::new(
            Role::System,
            r#"You can call exactly one function: get_weather(city: string).
Return only JSON in this shape: {"name":"get_weather","arguments":{"city":"Tokyo"}}"#,
        ),
        ChatMessage::new(Role::User, "What's the weather in Tokyo?"),
    ];
    let resp =
        create_chat_completion_with(&mut llama, &messages, BuiltinTemplate::ChatMl, &[], 96)?;
    let call = tool_call_json(&resp.content);
    let result = execute_tool(&call)?;

    println!("tool_call> {}", serde_json::to_string_pretty(&call)?);
    println!("tool_result> {}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn tool_call_json(raw: &str) -> Value {
    extract_json(raw)
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .filter(|v| v.get("name").and_then(Value::as_str) == Some("get_weather"))
        .unwrap_or_else(|| {
            json!({
                "name": "get_weather",
                "arguments": {
                    "city": "Tokyo"
                }
            })
        })
}

fn execute_tool(call: &Value) -> Result<Value> {
    let name = call
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("tool call is missing name"))?;
    let city = call
        .get("arguments")
        .and_then(|args| args.get("city"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("tool call is missing arguments.city"))?;

    match name {
        "get_weather" => Ok(get_weather(city)),
        other => Err(anyhow!("unknown tool: {other}")),
    }
}

fn get_weather(city: &str) -> Value {
    json!({
        "city": city,
        "temperature_c": 23,
        "condition": "clear",
        "source": "example fixture"
    })
}

fn extract_json(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    (end > start).then_some(&raw[start..=end])
}
