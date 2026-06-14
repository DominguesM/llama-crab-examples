#!/usr/bin/env bash
# Download the model required by an example (if missing) and run it.
#
# Usage:
#   ./run.sh quickstart      # downloads Qwen2.5 0.5B if needed
#   ./run.sh chat            # same model, interactive REPL
#   ./run.sh streaming       # same model, token streaming
#   ./run.sh vision gemma4   # downloads Gemma 4 + mmproj
#   ./run.sh vision lfm-vl   # downloads LFM2.5-VL
#   ./run.sh lfm_vl          # REPL against the LFM VL model
#   ./run.sh server_lfm      # boots llama-crab-server w/ LFM
#   ./run.sh tauri_chat_lfm  # opens the Tauri chat example
#   ./run.sh embeddings      # downloads BGE-small
#   ./run.sh rerank          # boots server with a reranker
#   ./run.sh multimodal_http # boots mtmd-enabled server w/ LFM
#   ./run.sh tools           # function-calling example
#
# Without any arguments, lists the available examples.
#
# Requirements: cargo (Rust 1.88+) and `hf` from `huggingface_hub` for
# the first-time download. Set `HF_TOKEN` to use gated models.
#
# Written for bash 3.2, the version Apple ships in /bin/bash.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

LLAMA_CRAB_VERSION="${LLAMA_CRAB_VERSION:-0.1.300}"

SMOL_MODEL="models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
BGE_MODEL="models/bge-small-en-v1.5-q4_k_m.gguf"
RERANK_MODEL="models/bge-reranker-base-q4_k_m.gguf"
GEMMA4_MODEL="models/gemma-4-E4B-it-Q4_K_M.gguf"
GEMMA4_MMPROJ="models/mmproj-gemma-4-E4B-it-BF16.gguf"
LFM_MODEL="models/LFM2.5-VL-1.6B-Q4_K_M.gguf"
LFM_MMPROJ="models/LFM2.5-VL-1.6B-mmproj-BF16.gguf"
TEST_IMAGE="tests/fixtures/test_image.png"

ensure_server() {
  if command -v llama-crab-server >/dev/null 2>&1; then
    return 0
  fi
  echo "==> installing llama-crab-server $LLAMA_CRAB_VERSION from crates.io"
  cargo install llama-crab-server --version "$LLAMA_CRAB_VERSION" --features mtmd
}

# Resolve the example name → "download-target|binary-name".
# Implemented as a `case` because bash 3.2 has no associative arrays.
example_target_bin() {
  case "$1" in
    quickstart)        echo "smol|run_quickstart" ;;
    streaming)         echo "smol|run_streaming" ;;
    chat)              echo "smol|chat" ;;
    stateful_chat)     echo "smol|run_chat" ;;
    simple)            echo "smol|simple" ;;
    structured)        echo "smol|structured" ;;
    tools)             echo "smol|tools" ;;
    tool_calls_qwen)   echo "smol|tools" ;;
    embeddings)        echo "bge|embeddings" ;;
    embedding_search)  echo "bge|run_embeddings" ;;
    rerank)            echo "bge-reranker|__server_rerank" ;;
    reranker)          echo "bge|reranker" ;;
    speculative)       echo "smol|speculative" ;;
    vision)            echo "vision_model|vision" ;;
    mtmd)              echo "vision_model|mtmd" ;;
    lfm_vl)            echo "lfm-vl|run_lfm_vl" ;;
    server_lfm)        echo "lfm-text|run_server_lfm" ;;
    tauri_chat_lfm)    echo "none|__tauri_chat_lfm" ;;
    multimodal_http)   echo "lfm-vl|__server_multimodal" ;;
    *) return 1 ;;
  esac
}

vision_paths_for() {
  case "$1" in
    gemma4) echo "$GEMMA4_MODEL|$GEMMA4_MMPROJ" ;;
    lfm-vl) echo "$LFM_MODEL|$LFM_MMPROJ" ;;
    *) return 1 ;;
  esac
}

vision_model_for() {
  case "${1:-}" in
    gemma4)  echo gemma4 ;;
    lfm-vl)  echo lfm-vl ;;
    lfmvl)   echo lfm-vl ;;
    "") echo "missing vision model argument (use 'gemma4' or 'lfm-vl')" >&2; exit 2 ;;
    *)  echo "unknown vision model: $1 (use 'gemma4' or 'lfm-vl')" >&2; exit 2 ;;
  esac
}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <example> [extra args...]" >&2
  echo >&2
  echo "Available examples:" >&2
  for ex in quickstart streaming chat stateful_chat simple structured tools \
            tool_calls_qwen embeddings embedding_search rerank reranker \
            speculative vision mtmd lfm_vl server_lfm tauri_chat_lfm \
            multimodal_http; do
    if [[ "$ex" == "tauri_chat_lfm" ]]; then
      printf "  %-18s  (opens the Tauri app; app downloads the model)\n" "$ex" >&2
    else
      printf "  %-18s  (downloads + runs the binary)\n" "$ex" >&2
    fi
  done
  echo >&2
  echo "Vision examples also need a model choice: $0 vision gemma4" >&2
  exit 1
fi

example="$1"
shift

if ! mapping="$(example_target_bin "$example")"; then
  echo "unknown example: $example" >&2
  echo "available: quickstart, streaming, chat, stateful_chat, simple, structured, tools," >&2
  echo "           tool_calls_qwen, embeddings, embedding_search, rerank, reranker," >&2
  echo "           speculative, vision, mtmd, lfm_vl, server_lfm, tauri_chat_lfm," >&2
  echo "           multimodal_http" >&2
  exit 2
fi

target="${mapping%|*}"
bin="${mapping#*|}"
model_args=()

if [[ "$target" == "vision_model" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "the 'vision' and 'mtmd' examples need a vision model argument." >&2
    echo "try:  $0 $example gemma4" >&2
    exit 2
  fi
  vision_arg="$1"; shift
  download_target="$(vision_model_for "$vision_arg")"
  vision_paths="$(vision_paths_for "$download_target")"
  model_args=("${vision_paths%|*}" "${vision_paths#*|}" "$TEST_IMAGE")
else
  download_target="$target"
  case "$target" in
    smol) model_args=("$SMOL_MODEL") ;;
    bge)  model_args=("$BGE_MODEL") ;;
    bge-reranker) model_args=("$RERANK_MODEL") ;;
    lfm-vl) model_args=("$LFM_MODEL" "$LFM_MMPROJ" "$TEST_IMAGE") ;;
    lfm-text) model_args=("$LFM_MODEL") ;;
    none) model_args=() ;;
  esac
fi

echo "==> ensuring model is available (target=$download_target)"
if [[ "${LLAMA_CRAB_SKIP_DOWNLOAD:-0}" == "1" ]]; then
  echo "==> skipped download (LLAMA_CRAB_SKIP_DOWNLOAD=1)"
elif [[ "$download_target" == "none" ]]; then
  echo "==> skipped download (example handles downloads at runtime)"
else
  ./scripts/download_models.sh "$download_target"
fi

echo
case "$bin" in
  __server_rerank)
    if [[ "${1:-}" == "--" ]]; then shift; fi
    [[ "${LLAMA_CRAB_DRY_RUN:-0}" == "1" ]] || ensure_server
    cmd=(llama-crab-server --model "$RERANK_MODEL" --reranking --pooling rank "$@")
    ;;
  __server_multimodal)
    if [[ "${1:-}" == "--" ]]; then shift; fi
    [[ "${LLAMA_CRAB_DRY_RUN:-0}" == "1" ]] || ensure_server
    cmd=(llama-crab-server --model "$LFM_MODEL" --mmproj "$LFM_MMPROJ" "$@")
    ;;
  __tauri_chat_lfm)
    if [[ "${1:-}" == "--" ]]; then shift; fi
    cmd=(pnpm --filter tauri-chat-lfm tauri dev "$@")
    ;;
  *)
    cmd=(cargo run --release --bin "$bin" -- "${model_args[@]}" "$@")
    ;;
esac
echo "==> running: ${cmd[*]}"
if [[ "${LLAMA_CRAB_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
exec "${cmd[@]}"
