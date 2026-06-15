//! Tiny helper for downloading a single file from a Hugging Face repo
//! to a local path. Used by the vision examples to fetch the
//! multimodal projector (mmproj) file, since `llama-crab`'s HF Hub
//! integration only resolves the text GGUF through `Llama::load`.
//!
//! The crate `hf-hub` (pulled in via the `hf-hub` cargo feature of
//! `llama-crab`) handles caching, tokens and mirror endpoints for us.

use std::path::PathBuf;

/// Download (or read from cache) a single file from a Hugging Face repo.
///
/// Honours `HF_TOKEN`, `HF_HOME` and `HF_ENDPOINT` via `hf-hub`'s
/// environment-based builder.
///
/// # Errors
/// Returns an error if the network round-trip fails, the file is
/// missing, or the destination cannot be written.
pub fn ensure_hf_file(repo: &str, filename: &str) -> anyhow::Result<PathBuf> {
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
