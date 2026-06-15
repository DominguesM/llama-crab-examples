# `chat`

One-shot chat completion with a built-in `BuiltinTemplate::ChatMl`
template.

## Model

Default: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` /
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (~400 MB). Auto-downloaded on the
first run.

## Run

```bash
cargo run --release --bin chat
```

Override the model with HF repo id + filename:

```bash
cargo run --release --bin chat -- \
    TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## What it shows

- One-shot chat completion through `create_chat_completion_with`.
- Built-in `ChatMl` template (Qwen, OpenHermes, etc.).
- `n_gpu_layers(99)` to offload as many layers as possible to Metal on
  Apple Silicon (harmless on CPU-only systems).
