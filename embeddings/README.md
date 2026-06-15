# `embeddings`

Extract a single L2-normalized embedding vector for an input text.

## Model

Default: `CompendiumLabs/bge-small-en-v1.5-gguf` /
`bge-small-en-v1.5-q4_k_m.gguf` (~30 MB). Auto-downloaded on the first
run.

## Run

```bash
cargo run --release --bin embeddings
```

CLI signature: `<hf_repo> <hf_filename> [text]`.

```bash
cargo run --release --bin embeddings -- \
    CompendiumLabs/bge-small-en-v1.5-gguf \
    bge-small-en-v1.5-q4_k_m.gguf \
    "Rust is a memory-safe systems language."
```

## What it shows

- `Llama::load` with `with_embeddings(true)` to enable embedding
  extraction.
- `Llama::embed(text, normalize=true)` — clears sequence 0, tokenizes
  the input, encodes it, and returns the L2-normalized vector.
- The output shows the token count, the vector dimension, the L2 norm
  (≈ 1.0 when normalized), and the first 8 components.
