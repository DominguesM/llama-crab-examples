# `structured`

JSON-schema grammar-constrained decoding. The example converts a JSON
Schema into a GBNF grammar with `json_schema_grammar`, wraps the
grammar in a sampler chain together with the greedy sampler, and asks
the model to emit a JSON object that matches the schema.

## Model

Default: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` /
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (~400 MB). Auto-downloaded on the
first run.

## Run

```bash
cargo run --release --bin structured
```

Override the model with HF repo id + filename:

```bash
cargo run --release --bin structured -- \
    Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf
```

## What it shows

- `llama_crab::high_level::completion::json_schema_grammar` — JSON
  Schema → GBNF conversion.
- `LlamaSampler::grammar(model, grammar_text, "root")` — wires the
  GBNF grammar into the sampler.
- `LlamaSampler::chain([grammar, greedy], false)` — composes the
  grammar-constrained sampler with greedy sampling.
- `create_completion_with_sampler` — runs a single completion with
  the caller-provided sampler chain.
- The output is parsed by `serde_json::from_str` and pretty-printed.
