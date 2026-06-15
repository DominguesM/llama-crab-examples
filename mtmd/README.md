# `mtmd`

Minimal multimodal example that drives the `mtmd` API directly. The
text model is loaded through `Llama::load` (which can pull the GGUF
from Hugging Face Hub), the `mmproj` projector is fetched separately
via `hf-hub`, and the multimodal chunks are evaluated with
`chunks.eval(...)`. The example then runs a greedy decode loop
token-by-token.

## Model

Default: `unsloth/LFM2.5-VL-1.6B-GGUF` (text + `mmproj-BF16.gguf`
projector, ~1.4 GB total). Auto-downloaded on the first run. The
synthetic PNG fixture is pulled from
`DominguesM/llama-crab-examples` / `test_image.png` if missing.

## Run

```bash
cargo run --release --bin mtmd
```

CLI signature:

```
cargo run --release --bin mtmd -- \
    <hf_repo> <text_gguf> <mmproj_gguf> [image.png] [prompt]
```

For example, with Gemma 4:

```bash
cargo run --release --bin mtmd -- \
    lmstudio-community/gemma-4-E4B-it-GGUF \
    gemma-4-E4B-it-Q4_K_M.gguf \
    mmproj-gemma-4-E4B-it-BF16.gguf
```

## What it shows

- `MtmdContext::init_from_file` to bind a projector to the text model.
- `MtmdBitmap::from_file` for the image side.
- `MtmdInputText::new` + `mtmd.tokenize` to produce multimodal chunks.
- `chunks.eval(&mtmd, ctx_ptr, n_past, seq_id, n_batch, logits_last=true)`
  to push the chunks into the KV cache.
- A small `LlamaBatch::one` loop to feed back each sampled token.
