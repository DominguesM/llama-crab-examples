# `stateful_chat`

Interactive multi-turn chat REPL. The model is loaded once and the
conversation history grows turn by turn — the example keeps the
`Vec<ChatMessage>` in memory and re-renders the whole prompt for each
turn.

## Model

Default: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` /
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (~400 MB). Auto-downloaded on the
first run.

## Run

```bash
cargo run --release --bin stateful_chat
```

Override the model with HF repo id + filename:

```bash
cargo run --release --bin stateful_chat -- \
    TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## Commands

| Command | Effect |
| --- | --- |
| `/exit` (also `/quit`, `/q`) | Quit the REPL. |
| `/clear` | Reset the history to just the system prompt. |
| `/save` | Print the current history as JSON. |
| anything else | Send a user message. |

## Notes

- The context size is the only limit on history length. With
  `n_ctx = 4096` you can run a long conversation before the prompt
  starts overflowing.
- The example uses `BuiltinTemplate::ChatMl` (Qwen / OpenHermes).
  Switch to `Llama3`, `Gemma`, `MistralInstruct`, etc. for other
  model families.
