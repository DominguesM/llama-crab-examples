//! `server_lfm` — a launcher for the published `llama-crab-server`
//! HTTP binary, pre-wired for the Liquid AI LFM2.5-VL 1.6B vision-language
//! model.
//!
//! On startup the example resolves the text GGUF and the multimodal
//! projector through the Hugging Face Hub (cached on the first run) and
//! then spawns `llama-crab-server` with the same arguments you would
//! pass on the command line.
//!
//! Run with:
//!
//! ```bash
//! cargo install llama-crab-server --version 0.1.6 --features mtmd --force
//! cargo run --release --bin server_lfm
//! ```

use std::io::ErrorKind;
use std::process::{Command, ExitCode};

const DEFAULT_HF_REPO: &str = "unsloth/LFM2.5-VL-1.6B-GGUF";
const DEFAULT_TEXT_FILE: &str = "LFM2.5-VL-1.6B-Q4_K_M.gguf";
const DEFAULT_MMPROJ_FILE: &str = "mmproj-BF16.gguf";

fn main() -> ExitCode {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let hf_repo = args
        .next()
        .unwrap_or_else(|| DEFAULT_HF_REPO.to_string());
    let text_filename = args
        .next()
        .unwrap_or_else(|| DEFAULT_TEXT_FILE.to_string());
    let mmproj_filename = args
        .next()
        .unwrap_or_else(|| DEFAULT_MMPROJ_FILE.to_string());
    let forwarded: Vec<String> = args.collect();

    // Pre-resolve the model + mmproj so the server starts fast. The
    // downloads happen on the user's machine — no extra network
    // traffic after this point.
    eprintln!("→ resolving {hf_repo}/{text_filename} via HF Hub");
    let model_path = match ensure_hf_file(&hf_repo, &text_filename) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("failed to fetch model: {err}");
            return ExitCode::from(1);
        }
    };
    eprintln!("✓ model at {}", model_path.display());

    eprintln!("→ resolving {hf_repo}/{mmproj_filename} via HF Hub");
    let mmproj_path = match ensure_hf_file(&hf_repo, &mmproj_filename) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("failed to fetch mmproj: {err}");
            return ExitCode::from(1);
        }
    };
    eprintln!("✓ mmproj at {}", mmproj_path.display());

    let mut cmd = Command::new("llama-crab-server");
    cmd.arg("--model")
        .arg(&model_path)
        .arg("--mmproj")
        .arg(&mmproj_path);
    for arg in &forwarded {
        cmd.arg(arg);
    }

    eprintln!("→ launching llama-crab-server with the resolved model paths");
    let status = match cmd.status() {
        Ok(s) => s,
        Err(err) if err.kind() == ErrorKind::NotFound => {
            eprintln!(
                "llama-crab-server was not found. Install it with: \
                 cargo install llama-crab-server --version 0.1.6 --features mtmd"
            );
            return ExitCode::from(127);
        }
        Err(err) => {
            eprintln!("failed to spawn llama-crab-server: {err}");
            return ExitCode::from(1);
        }
    };

    ExitCode::from(status.code().unwrap_or(1) as u8)
}

fn ensure_hf_file(repo: &str, filename: &str) -> anyhow::Result<std::path::PathBuf> {
    use anyhow::Context;
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .build()
        .context("failed to build hf-hub Api")?;
    let api_repo = api.repo(hf_hub::Repo::new(repo.to_string(), hf_hub::RepoType::Model));
    let path = api_repo
        .get(filename)
        .with_context(|| format!("failed to fetch {repo}/{filename}"))?;
    Ok(path)
}
