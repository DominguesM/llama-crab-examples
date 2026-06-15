# `speculative`

A compact demonstration of the speculative-decoding API. The example
uses `PromptLookupDecoding` (a draft strategy that finds repeated
n-grams in the prompt) to produce a small draft of candidate tokens,
then prints the draft. It is illustrative — not a benchmark harness.

## Model

Default: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` /
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (~400 MB). Auto-downloaded on the
first run.

## Run

```bash
cargo run --release --bin speculative
```

CLI signature: `<hf_repo> <hf_filename> [prompt]`. The default prompt
has a repeated phrase so the look-up decoder can find a match.

## What it shows

- `llama_crab::speculative::PromptLookupDecoding` and the `DraftModel`
  trait.
- `draft.draft(&prompt_tokens, max_draft)` returns a `Vec<LlamaToken>`
  to be validated by the target model.
