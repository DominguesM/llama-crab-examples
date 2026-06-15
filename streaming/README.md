# `streaming`

High-level streaming completion. Each `CompletionChunk` is written to
stdout as it becomes available. The callback can return
`StreamControl::Stop` to abort generation.

## Model

Default: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` /
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (~400 MB). Auto-downloaded on the
first run.

## Run

```bash
cargo run --release --bin streaming
```

CLI signature: `<hf_repo> <hf_filename> [prompt] [max_tokens]`.

```bash
cargo run --release --bin streaming -- \
    Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf \
    "Write a haiku about Rust" 128
```

## What it shows

- `create_completion_stream` with `CompletionOptions::new(max_tokens)`.
- Returning `StreamControl::Stop` from the callback to abort early on
  I/O failure.
- The final `Completion` is discarded (`let _completion = ...`) — the
  real output was already streamed to stdout.
