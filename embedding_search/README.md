# `embedding_search`

Semantic search over a small fixed corpus. Embeds a query and four
documents with `bge-small-en-v1.5` and ranks the documents by cosine
similarity.

## Model

Default: `CompendiumLabs/bge-small-en-v1.5-gguf` /
`bge-small-en-v1.5-q4_k_m.gguf` (~30 MB). Auto-downloaded on the first
run. The example uses CLS pooling — the same as the upstream BGE
configuration for English short-text retrieval.

## Run

```bash
cargo run --release --bin embedding_search
```

CLI signature: `<hf_repo> <hf_filename> [query]`.

```bash
cargo run --release --bin embedding_search -- \
    CompendiumLabs/bge-small-en-v1.5-gguf \
    bge-small-en-v1.5-q4_k_m.gguf \
    "What language prevents memory bugs?"
```

Override the query with the `LLAMA_CRAB_QUERY` environment variable:

```bash
LLAMA_CRAB_QUERY="What is borrow checking?" \
    cargo run --release --bin embedding_search
```

## What it shows

- `with_embeddings(true)` + `with_pooling_type(PoolingType::Cls)`.
- `Llama::embed(text, true)` returns an L2-normalized vector; cosine
  similarity is then a plain dot product, but the example still
  divides by both norms to make the math explicit.
- Sort by descending score and print the top hit.
