# `embeddings`

Extract a single L2-normalized embedding vector for an input text.

## Model

Default: `nomic-ai/nomic-embed-text-v1.5-GGUF` /
`nomic-embed-text-v1.5.Q4_K_M.gguf` (~80 MB). Auto-downloaded on the
first run.

## Run

```bash
cargo run --release --bin embeddings
```

CLI signature: `<hf_repo> <hf_filename> [text]`.

```bash
cargo run --release --bin embeddings -- \
    nomic-ai/nomic-embed-text-v1.5-GGUF \
    nomic-embed-text-v1.5.Q4_K_M.gguf \
    "Rust is a memory-safe systems language."
```

## What it shows

- `Llama::load` with `with_embeddings(true)` to enable embedding
  extraction and `with_pooling_type(PoolingType::Cls)` to match the
  model's official pooling (Mean also works for `nomic-embed-text`).
- `Llama::embed(text, true)` — clears sequence 0, tokenizes the
  input, encodes it, and returns the L2-normalized vector.
- The output shows the token count, the vector dimension, the L2 norm
  (≈ 1.0 when normalized), and the first 8 components.

## Known issue

Pure `BertModel` architectures without a classification head (e.g.
`CompendiumLabs/bge-small-en-v1.5-gguf`) currently segfault inside
`llama_encode` on `llama-crab-sys 0.1.300` / `llama-crab 0.1.6`. The
default model above is a Bert variant that does work. See
[`embedding_search`](../embedding_search) for the same caveat.

