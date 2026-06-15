# `llama-crab` examples

A collection of self-contained programs that demonstrate every public
feature of the [`llama-crab`](https://crates.io/crates/llama-crab)
library. Each example lives in its own Cargo crate (one `[[bin]]` per
crate) so you can copy-paste the parts you need into your own project
without dragging in the rest.

The examples target **`llama-crab` 0.1.6** and rely on the
`hf-hub` cargo feature, which is now part of the default build
(see the [CHANGELOG](https://github.com/DominguesM/llama-crab/blob/main/CHANGELOG.md)).
This means each example points at a Hugging Face repository directly
in its source code — `Llama::load` resolves the repo id, downloads
the GGUF on first run and caches it under
`~/.cache/huggingface/hub` (or `$HF_HOME/hub`).

## TL;DR — run an example

```bash
cargo run --release --bin quickstart
```

The first run downloads `qwen2.5-0.5b-instruct-q4_k_m.gguf`
(~400 MB) and compiles `llama-crab-sys` (≈ 3 min on a 16-core
machine). Subsequent runs only rebuild the example crate and re-use
the cached GGUF.

## Directory layout

```
.
├── README.md                    # this file
├── Cargo.toml                   # workspace (llama-crab = 0.1.6)
│
├── quickstart/                  # smallest end-to-end demo (text only)
├── streaming/                   # high-level token-by-token output
├── stateful_chat/               # interactive multi-turn chat REPL
├── chat/                        # one-shot chat completion
├── simple/                      # one-shot text completion
├── structured/                  # JSON-schema grammar-constrained decoding
├── speculative/                 # prompt-lookup speculative drafting
├── tools/                       # function-calling + tool parser
├── embeddings/                  # raw embedding extraction
├── embedding_search/            # BGE-small + cosine ranking
├── reranker/                    # embedding-based cross-encoder reranking
├── mtmd/                        # multimodal (vision) via raw mtmd.h
├── vision/                      # vision via the high-level API
├── lfm_vl_vision/               # LFM2.5-VL multimodal REPL
├── server_lfm/                  # launcher for llama-crab-server w/ LFM2.5-VL
└── tauri-chat-lfm/              # Tauri 2 chat app for LFM2.5 350M
```

Each example is a self-contained crate: it has its own `README.md`,
its own `Cargo.toml` and its own `src/main.rs`. The README inside
each folder is the source of truth for that example's CLI, model
choice and behaviour.

## Per-example guide

| Example | Default model (Hugging Face repo / file) | Size | What it shows |
| --- | --- | --- | --- |
| `quickstart` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` / `qwen2.5-0.5b-instruct-q4_k_m.gguf` | ~400 MB | Load → tokenize → complete → chat → FIM |
| `streaming` | same as `quickstart` | ~400 MB | High-level `create_completion_stream` |
| `stateful_chat` | same as `quickstart` | ~400 MB | REPL with growing history, `/clear`, `/save` |
| `chat` | same as `quickstart` | ~400 MB | One-shot chat + `BuiltinTemplate::ChatMl` |
| `simple` | same as `quickstart` | ~400 MB | One-shot `create_completion` |
| `structured` | same as `quickstart` | ~400 MB | `json_schema_grammar()` + grammar sampler |
| `speculative` | same as `quickstart` | ~400 MB | `PromptLookupDecoding` drafting |
| `tools` | same as `quickstart` | ~400 MB | Function-calling + JSON extraction |
| `embeddings` | `CompendiumLabs/bge-small-en-v1.5-gguf` / `bge-small-en-v1.5-q4_k_m.gguf` | ~30 MB | L2-normalized embedding + raw vector preview |
| `embedding_search` | same as `embeddings` | ~30 MB | Embed + cosine ranking over a fixed corpus |
| `reranker` | `turingevo/bge-reranker-base-Q4_K_M-GGUF` / `bge-reranker-base-q4_k_m.gguf` | ~220 MB | `Llama::rerank` over a short list |
| `mtmd` | `unsloth/LFM2.5-VL-1.6B-GGUF` / `LFM2.5-VL-1.6B-Q4_K_M.gguf` + `mmproj-BF16.gguf` | ~1.4 GB | Raw `mtmd.h` API: bitmap → chunks → eval |
| `vision` | same as `mtmd` | ~1.4 GB | High-level `MtmdContext` API |
| `lfm_vl_vision` | same as `mtmd` | ~1.4 GB | LFM2.5-VL multimodal REPL |
| `server_lfm` | same as `mtmd` | ~1.4 GB | Spawns `llama-crab-server` with the resolved model |
| `tauri_chat_lfm` | `LiquidAI/LFM2.5-350M-GGUF` / `LFM2.5-350M-Q4_K_M.gguf` | ~229 MB | Tauri 2 chat app with download progress |

The two vision example families both need a vision-language GGUF
**and** its `mmproj` projector. The text side is downloaded through
`llama-crab`'s `hf-hub` integration; the `mmproj` side is fetched
through a small `hf-hub` helper in each example.

## Running an example by hand

Every example exposes a `[[bin]]` named after the example folder. The
default arguments are baked into the source — running
`cargo run --release --bin <name>` works out of the box. CLI
arguments, when present, follow this convention:

```
cargo run --release --bin <name> -- \
    <hf_repo> <hf_filename> [extra args...]
```

For vision examples, the second positional argument is the text
filename and the third is the `mmproj` filename:

```
cargo run --release --bin vision -- \
    unsloth/LFM2.5-VL-1.6B-GGUF \
    LFM2.5-VL-1.6B-Q4_K_M.gguf \
    mmproj-BF16.gguf
```

## Using a different model

Any HF repo id that contains a `.gguf` can be used. If the repo
contains multiple `.gguf` files you must pass the filename with the
second positional argument; the auto-pick path refuses to guess
(there is a clear error in that case).

For local files, pass the path instead of the repo id — `Llama::load`
auto-detects that the path exists on disk and skips the HF download.

## Server example

`server_lfm` is the only example that does not compile a self-contained
Rust binary. It resolves the text and `mmproj` GGUFs through HF Hub
and then spawns the published `llama-crab-server` HTTP binary. Install
the server once with:

```bash
cargo install llama-crab-server --version 0.1.6 --features mtmd --force
```

Then `cargo run --release --bin server_lfm` brings it up against
LFM2.5-VL 1.6B with the resolved paths.

## Tauri example

`tauri-chat-lfm` is a Tauri 2 desktop chat app. It uses `pnpm` and
the `tauri` CLI:

```bash
cd tauri-chat-lfm
pnpm install
pnpm tauri dev
```

The Rust side resolves the LFM 350M GGUF on first launch and streams
download progress to the renderer through an IPC channel. The
`@llama-crab/tauri` client then loads the resolved model into the
plugin and streams chat completions.

## Hugging Face authentication

Set `HF_TOKEN` (or use `hf auth login` before running the example)
to access private or gated repos. The `hf-hub` feature of `llama-crab`
and the `hf-hub` crate used directly in the vision examples both
read `HF_TOKEN` from the environment via `ApiBuilder::from_env()`.

To point at an HF mirror, set `HF_ENDPOINT=https://hf-mirror.com`
(or use the `--hf-endpoint` builder on `RealHfDownloader` when
embedding the loader into your own code).

## Troubleshooting

* **`hf-hub feature is disabled`** — you are building with
  `--no-default-features` or a custom feature set that drops
  `hf-hub`. The workspace's `Cargo.toml` enables it by default;
  use `cargo build` without feature overrides.
* **`ambiguous: N gguf files in repo ...`** — the HF repo has more
  than one `.gguf` and the example was started without
  `with_hf_filename`. Pass the filename as the second positional
  argument.
* **`failed to allocate context`** — the model needs more memory than
  is available. Try a smaller quant (`Q4_K_M` → `Q3_K_M` → `Q2_K`)
  or reduce `n_ctx` / `n_gpu_layers`.
* **First build is slow** — `llama-crab-sys` compiles all 17
  llama.cpp backends (~3 min on a 16-core machine). Subsequent
  builds are cached.
* **First vision run is slow** — `mmproj` for LFM2.5-VL is ~340 MB.
  Subsequent runs reuse the HF cache.
* **`embeddings` and `embedding_search` crash on a pure BertModel**
  (e.g. `CompendiumLabs/bge-small-en-v1.5-gguf`) — known segfault
  inside `llama_encode` on `llama-crab-sys 0.1.300` /
  `llama-crab 0.1.6`. The default models are `nomic-embed-text-v1.5`
  variants that do not hit the bug. Override with a
  `BertForSequenceClassification` GGUF (e.g. the one used by
  `rerank`) to confirm `Llama::embed` itself works on your machine.

## Adding a new example

The boilerplate for a new example crate is ~15 lines:

```toml
# my_example/Cargo.toml
[package]
name = "my_example"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish = false

[[bin]]
name = "my_example"
path = "src/main.rs"

[dependencies]
llama-crab.workspace = true
anyhow = "1"
```

```rust
// my_example/src/main.rs
use anyhow::Result;
use llama_crab::{Llama, LlamaParams};

const HF_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const HF_FILE: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn main() -> Result<()> {
    let mut llama = Llama::load(
        LlamaParams::new(HF_REPO)
            .with_hf_filename(HF_FILE)
            .with_n_ctx(2048),
    )?;
    let resp = llama.create_completion("Hello!", 32)?;
    print!("{}", resp.text);
    Ok(())
}
```

Add `my_example` to the `members = [...]` list in the root
`Cargo.toml` and a row to the table above.
