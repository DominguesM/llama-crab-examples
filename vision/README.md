# `vision`

High-level multimodal inference: load a vision-language GGUF + its
`mmproj` projector and ask a question about an image.

## Model

Default: `unsloth/LFM2.5-VL-1.6B-GGUF` (the LFM2.5-VL 1.6B GGUF +
`mmproj-BF16.gguf` projector, ~1.4 GB total). Auto-downloaded on the
first run.

The example also pulls a synthetic 256×256 PNG fixture from
`DominguesM/llama-crab-examples` / `test_image.png` when the local
`tests/fixtures/test_image.png` is missing.

## Run

```bash
cargo run --release --bin vision
```

CLI signature:

```
cargo run --release --bin vision -- \
    <hf_repo> <text_gguf> <mmproj_gguf> [image.png] [prompt]
```

For example, to point at Gemma 4 instead of LFM:

```bash
cargo run --release --bin vision -- \
    lmstudio-community/gemma-4-E4B-it-GGUF \
    gemma-4-E4B-it-Q4_K_M.gguf \
    mmproj-gemma-4-E4B-it-BF16.gguf
```

The model family is detected from the HF repo id (`lfm` in the path
=> ChatML framing, anything else => raw marker + prompt).

## What it shows

- `LlamaParams::new("<hf_repo>")` + `with_hf_filename("<text>")` to
  download the text GGUF through the `hf-hub` feature.
- The text side is a plain `Llama::load`. The `mmproj` side is
  fetched directly via `hf-hub` (the example has a small
  `hf_helper::ensure_hf_file` wrapper).
- `MtmdContext::init_from_file(mmproj, llama.model())` then
  `mtmd.tokenize(MtmdInputText::new(...), &[&bitmap])` to interleave
  text and image tokens.
- A greedy decode loop over `chunks.eval` (multimodal) +
  `LlamaContext::decode` (token-by-token).
