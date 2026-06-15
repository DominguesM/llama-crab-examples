# `quickstart`

The smallest end-to-end example: load a model, tokenize a prompt, run
a single text completion, run one chat completion, try fill-in-the-middle.

## Model

The default model is `Qwen/Qwen2.5-0.5B-Instruct-GGUF` /
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (~400 MB). It is downloaded to the
Hugging Face cache (`~/.cache/huggingface/hub`) on the first run and
reused afterwards.

## Run

```bash
cargo run --release
```

The first run downloads the GGUF and compiles `llama-crab-sys` (≈ 3 min
on a 16-core machine). Subsequent runs are cached and only rebuild the
example crate.

## Use a different model

Pass the HF repo id and the GGUF filename as positional arguments:

```bash
cargo run --release -- \
    TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

You can also point at a local file by passing a path — `Llama::load`
auto-detects whether the argument is an HF repo id or a local path.

## What it does

1. Loads the GGUF with `n_ctx=2048`, `n_threads=4`.
2. Tokenizes `"The capital of France is"` and prints the token ids.
3. Runs `create_completion` for 16 tokens.
4. Runs `create_chat_completion` for one round of Q&A.
5. Calls `complete_infill` to fill in between `"fn main() {"` and `"}"`.
