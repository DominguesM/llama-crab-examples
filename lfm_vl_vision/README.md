# `lfm_vl_vision`

Interactive multi-question REPL using **Liquid AI LFM2.5-VL 1.6B**.
LFM2.5-VL 1.6B is the smallest current VLM in the LFM2.5 family and
runs comfortably on CPU and on older Apple Silicon (~1 GB on disk).

The example keeps the chat history in memory and re-renders the whole
prompt for each turn (the same approach used by `stateful_chat`),
with the media marker inserted into the latest user turn.

## Model

Default: `unsloth/LFM2.5-VL-1.6B-GGUF` (text +
`mmproj-BF16.gguf` projector). Auto-downloaded on the first run. The
synthetic PNG fixture is pulled from
`DominguesM/llama-crab-examples` / `test_image.png` if missing.

## Run

```bash
cargo run --release --bin lfm_vl_vision
```

CLI signature (all positional arguments optional):

```
cargo run --release --bin lfm_vl_vision -- \
    <hf_repo> <text_gguf> <mmproj_gguf> [image.png] [prompt]
```

If you pass a fifth positional argument, the example runs in
one-shot mode and prints a single reply.

## Commands

| Command | Effect |
| --- | --- |
| `/image <path>` | Change the active image (downloaded from HF if missing). |
| `/clear` | Reset the chat history to the system prompt. |
| `/exit` (also `/quit`, `/q`) | Quit the REPL. |
| anything else | Send a user message. |
