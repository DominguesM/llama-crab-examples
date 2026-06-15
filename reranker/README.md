# `reranker`

Embedding-based cross-encoder reranking demo. The example uses
`Llama::rerank` with a `bge-reranker-base` GGUF to score a short list
of `(query, document)` pairs and prints them in descending order.

## Model

Default: `turingevo/bge-reranker-base-Q4_K_M-GGUF` /
`bge-reranker-base-q4_k_m.gguf` (~220 MB). Auto-downloaded on the
first run.

## Run

```bash
cargo run --release --bin reranker
```

Override the model with HF repo id + filename:

```bash
cargo run --release --bin reranker -- \
    turingevo/bge-reranker-base-Q4_K_M-GGUF \
    bge-reranker-base-q4_k_m.gguf
```

## What it shows

- `with_embeddings(true)` + `with_pooling_type(PoolingType::Rank)` —
  the configuration the GGUF declares.
- `Llama::rerank(query, documents)` — encodes each pair in a
  separate forward pass and returns a `Vec<f32>` of scores in input
  order.
- A simple descending sort to turn the scores into a ranked list.
