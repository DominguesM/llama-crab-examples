# `tauri-chat-lfm`

A minimal Tauri 2 desktop chat app that runs **Liquid AI LFM2.5 350M**
locally through `tauri-plugin-llama-crab`. The Rust side resolves the
GGUF through Hugging Face Hub on first use and streams a download
progress to the renderer through an IPC channel. The renderer feeds
the resolved path into the plugin's `load_model`.

## Model

Default: `LiquidAI/LFM2.5-350M-GGUF` / `LFM2.5-350M-Q4_K_M.gguf`
(~229 MB). Auto-downloaded on the first run.

## Run

```bash
cd tauri-chat-lfm
pnpm install
pnpm tauri dev
```

Build a release bundle with `pnpm tauri build`.

## What it shows

- Tauri 2 + Vite + TypeScript skeleton.
- A `tauri::command` that resolves a HF repo file and streams
  progress through an IPC `Channel<DownloadProgress>`.
- `tauri-plugin-llama-crab`'s `load_model` (the plugin enables the
  `hf-hub` feature of `llama-crab` by default, so future versions
  can pass a HF repo id directly).
- A streaming chat completion with the `LlamaCrabTauri` client and
  a one-page HTML/CSS/TS UI.
