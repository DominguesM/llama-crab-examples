# `simple`

The minimum possible end-to-end example: load a GGUF, run one
`create_completion` call, print the result.

## Model

Default: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` /
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (~400 MB). Downloaded to the
Hugging Face cache on the first run.

## Run

```bash
cargo run --release
```

Pass the HF repo id, GGUF filename and an optional prompt to override:

```bash
cargo run --release -- \
    TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    "Once upon a time"
```

Pass a local path instead of an HF repo id if you already have the
file on disk; `Llama::load` will detect that and skip the HF download.
