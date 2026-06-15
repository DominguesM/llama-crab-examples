# `tools`

Function-calling demo with a single tool (`get_weather`). The system
prompt instructs the model to emit a JSON tool call; the example
extracts the JSON, dispatches to a local handler, and prints the
result.

## Model

Default: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` /
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (~400 MB). Auto-downloaded on the
first run. ChatML-style `<tool_call>` formatting works best with models
trained for tool use (Qwen 2.5 Instruct, Hermes, etc.).

## Run

```bash
cargo run --release --bin tools
```

Override the model with HF repo id + filename:

```bash
cargo run --release --bin tools -- \
    Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf
```

## What it shows

- One-shot chat completion with a tool-aware system prompt.
- Naive JSON extraction (`extract_json` finds the first `{` and last
  `}`). For production code use
  `llama_crab::chat::tool_call::extract_tool_calls` (which understands
  `<tool_call>...</tool_call>`, `[TOOL_CALLS][...]`, etc.) or the
  streaming `ToolCallStream`.
- A simple tool dispatcher (`get_weather`) and a fallback that
  returns a fixed Tokyo answer if the model did not produce a parseable
  call.
