# `server_lfm`

A launcher for the published `llama-crab-server` HTTP binary,
pre-wired for the **Liquid AI LFM2.5-VL 1.6B** vision-language model.
The example resolves the text GGUF and the multimodal projector
through Hugging Face Hub on startup (cached after the first run) and
then spawns `llama-crab-server` with the resolved on-disk paths and
any extra CLI arguments you pass.

## Prerequisites

Install the server binary once (it is not pulled in by this Cargo
crate because the `llama-crab-server` Rust crate only exposes the
binary, not a library):

```bash
cargo install llama-crab-server --version 0.1.6 --features mtmd --force
```

## Run

```bash
cargo run --release --bin server_lfm
```

The example fetches `LFM2.5-VL-1.6B-Q4_K_M.gguf` and
`mmproj-BF16.gguf` from `unsloth/LFM2.5-VL-1.6B-GGUF` on the first
run, then spawns:

```text
llama-crab-server \
    --model <path to LFM2.5-VL-1.6B-Q4_K_M.gguf> \
    --mmproj <path to mmproj-BF16.gguf>
```

## Override the model

Pass an HF repo id, text filename and mmproj filename as the first
three positional arguments:

```bash
cargo run --release --bin server_lfm -- \
    lmstudio-community/gemma-4-E4B-it-GGUF \
    gemma-4-E4B-it-Q4_K_M.gguf \
    mmproj-gemma-4-E4B-it-BF16.gguf \
    -- --host 127.0.0.1 --port 8080 --n-ctx 2048
```

Anything after the third positional argument is forwarded to
`llama-crab-server`. The `--` separator used by the previous shell
wrapper is also accepted and dropped.

## Endpoints

See [the server docs](https://llama-crab.nlp.rocks/server/running) for
the full route list. Quick smoke test:

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}

curl http://127.0.0.1:8080/v1/models
# {"object":"list","data":[{"id":"llama-crab",...}]}

curl http://127.0.0.1:8080/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{
        "messages": [
            {"role":"user","content":[
                {"type":"text","text":"What is in this image?"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
            ]}
        ]
    }'
```
