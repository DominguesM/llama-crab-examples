#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -f Cargo.toml ]] || fail "missing standalone Cargo workspace"
[[ -f tests/fixtures/test_image.png ]] || fail "missing shared test image fixture"

if rg -n 'path = ".*/crates/|workspace:\*|\.\./crates|cargo run --release -p llama-crab-server|cargo run --release --package llama-crab-server' \
  -g 'Cargo.toml' -g 'package.json' -g '*.sh' -g '!tests/examples_repo_smoke.sh' -g '!target/**' -g '!node_modules/**' .; then
  fail "examples must depend on published crates/npm packages, not the old monorepo workspace"
fi

grep -Fq 'llama-crab = { version = "0.1.8"' Cargo.toml \
  || fail "root Cargo.toml must pin the published llama-crab crate"

grep -Fq '"@llama-crab/tauri": "0.1.8"' tauri-chat-lfm/package.json \
  || fail "Tauri example must depend on the published @llama-crab/tauri npm package"

grep -Fq 'tauri-plugin-llama-crab = "0.1.8"' tauri-chat-lfm/src-tauri/Cargo.toml \
  || fail "Tauri Rust app must depend on the published tauri-plugin-llama-crab crate"

echo "examples repository smoke tests passed"
