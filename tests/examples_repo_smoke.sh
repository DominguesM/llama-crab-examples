#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -f Cargo.toml ]] || fail "missing standalone Cargo workspace"
[[ -x run.sh ]] || fail "missing executable root run.sh"
[[ -x scripts/download_models.sh ]] || fail "missing executable scripts/download_models.sh"
[[ -f tests/fixtures/test_image.png ]] || fail "missing shared test image fixture"

if rg -n 'path = ".*/crates/|workspace:\*|\.\./crates|cargo run --release -p llama-crab-server|cargo run --release --package llama-crab-server' \
  -g 'Cargo.toml' -g 'package.json' -g '*.sh' -g '!tests/examples_repo_smoke.sh' -g '!target/**' -g '!node_modules/**' .; then
  fail "examples must depend on published crates/npm packages, not the old monorepo workspace"
fi

grep -Fq 'llama-crab = { version = "0.1.300"' Cargo.toml \
  || fail "root Cargo.toml must pin the published llama-crab crate"

grep -Fq '"@llama-crab/tauri": "0.1.300"' tauri-chat-lfm/package.json \
  || fail "Tauri example must depend on the published @llama-crab/tauri npm package"

grep -Fq 'tauri-plugin-llama-crab = "0.1.300"' tauri-chat-lfm/src-tauri/Cargo.toml \
  || fail "Tauri Rust app must depend on the published tauri-plugin-llama-crab crate"

run_dry() {
  LLAMA_CRAB_SKIP_DOWNLOAD=1 LLAMA_CRAB_DRY_RUN=1 ./run.sh "$@"
}

out="$(run_dry chat)"
grep -Fq "cargo run --release --bin chat -- models/qwen2.5-0.5b-instruct-q4_k_m.gguf" <<<"$out" \
  || fail "chat dry-run did not include the smol model path"

out="$(run_dry vision lfm-vl)"
grep -Fq "cargo run --release --bin vision -- models/LFM2.5-VL-1.6B-Q4_K_M.gguf models/LFM2.5-VL-1.6B-mmproj-BF16.gguf tests/fixtures/test_image.png" <<<"$out" \
  || fail "vision dry-run did not include model, mmproj and image paths"

out="$(run_dry rerank)"
grep -Fq "llama-crab-server --model models/bge-reranker-base-q4_k_m.gguf --reranking --pooling rank" <<<"$out" \
  || fail "rerank dry-run must run the published llama-crab-server binary"

out="$(run_dry tauri_chat_lfm)"
grep -Fq "pnpm --filter tauri-chat-lfm tauri dev" <<<"$out" \
  || fail "tauri_chat_lfm dry-run did not run the Tauri app command"

out="$(LLAMA_CRAB_DRY_RUN=1 ./scripts/download_models.sh smol)"
grep -Fq "hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf" <<<"$out" \
  || fail "download dry-run did not use hf download for smol"

echo "examples repository smoke tests passed"
