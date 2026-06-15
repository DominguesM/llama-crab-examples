# `embedding_search`

Semantic search over a small fixed corpus. Embeds a query and four
documents with `nomic-embed-text-v1.5` and ranks the documents by
cosine similarity.

## Model

Default: `nomic-ai/nomic-embed-text-v1.5-GGUF` /
`nomic-embed-text-v1.5.Q4_K_M.gguf` (~80 MB). Auto-downloaded on the
first run. The example uses CLS pooling, which matches the upstream
`nomic-embed-text` configuration for English short-text retrieval.

## Run

```bash
cargo run --release --bin embedding_search
```

CLI signature: `<hf_repo> <hf_filename> [query]`.

```bash
cargo run --release --bin embedding_search -- \
    nomic-ai/nomic-embed-text-v1.5-GGUF \
    nomic-embed-text-v1.5.Q4_K_M.gguf \
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

## Known issue

Pure `BertModel` architectures without a classification head (e.g.
`CompendiumLabs/bge-small-en-v1.5-gguf`) currently segfault inside
`llama_encode` on `llama-crab-sys 0.1.300` / `llama-crab 0.1.6`. The
default model above is a Bert variant that does work.
