//! `mtmd` — minimal multimodal example that drives the `mtmd` API
//! directly (tokenize + eval + sample loop).
//!
//! Default model: Liquid AI LFM2.5-VL 1.6B (smallest current VLM).
//!
//! Run with:
//!
//! ```bash
//! cargo run --release --bin mtmd
//! ```

mod hf_helper;

use anyhow::{Context, Result};
use llama_crab::batch::LlamaBatch;
use llama_crab::multimodal::{default_media_marker, MtmdBitmap, MtmdContext, MtmdInputText};
use llama_crab::sampling::LlamaSampler;
use llama_crab::token::LlamaToken;
use llama_crab::{Llama, LlamaParams};
use std::path::{Path, PathBuf};

// Same defaults as the `vision` example.
const DEFAULT_HF_REPO: &str = "unsloth/LFM2.5-VL-1.6B-GGUF";
const DEFAULT_TEXT_FILE: &str = "LFM2.5-VL-1.6B-Q4_K_M.gguf";
const DEFAULT_MMPROJ_FILE: &str = "mmproj-BF16.gguf";
const DEFAULT_IMAGE: &str = "tests/fixtures/test_image.png";
const DEFAULT_PROMPT: &str = "Describe this image in one short sentence.";

const TEST_IMAGE_HF_REPO: &str = "DominguesM/llama-crab-examples";
const TEST_IMAGE_HF_FILE: &str = "test_image.png";

fn main() -> Result<()> {
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
    let image = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_IMAGE.to_string()));
    let prompt = args
        .next()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    let image = ensure_test_image(&image)?;

    let mut llama = Llama::load(
        LlamaParams::new(&hf_repo)
            .with_hf_filename(&text_filename)
            .with_n_ctx(4096),
    )
    .with_context(|| format!("failed to load {hf_repo}/{text_filename}"))?;

    let mmproj_path = hf_helper::ensure_hf_file(&hf_repo, &mmproj_filename)
        .with_context(|| format!("failed to fetch {hf_repo}/{mmproj_filename}"))?;
    let mtmd = MtmdContext::init_from_file(&mmproj_path, llama.model())?;
    if !mtmd.support_vision() {
        anyhow::bail!("this projector does not support vision");
    }

    let bitmap = MtmdBitmap::from_file(&image)
        .with_context(|| format!("failed to decode image at {}", image.display()))?;
    let marker = default_media_marker();
    let media_prompt = if hf_repo.to_ascii_lowercase().contains("lfm") {
        format!("<|im_start|>user\n{marker}\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        format!("{marker}\n{prompt}")
    };
    let chunks = mtmd.tokenize(MtmdInputText::new(&media_prompt), &[&bitmap])?;

    let _ = llama.context().seq_rm(0, -1, -1);
    let ctx_ptr = llama.context().raw_handle();
    let new_n_past = unsafe {
        chunks.eval(
            &mtmd,
            ctx_ptr,
            0,
            0,
            llama.context().n_batch() as i32,
            true,
        )
    }?;

    let mut sampler = LlamaSampler::greedy().expect("greedy");
    let eos = llama.model().token_eos();
    let mut out = String::new();
    for n_generated in 0_usize..96 {
        let idx = if n_generated == 0 { -1 } else { 0 };
        let tok: LlamaToken = unsafe { sampler.sample(ctx_ptr, idx) };
        sampler.accept(tok);
        if tok == eos {
            break;
        }
        out.push_str(&llama.model().detokenize(&[tok], false)?);

        let single = LlamaBatch::one(tok, new_n_past + n_generated as i32, 0, true);
        llama.context().decode(&single)?;
    }

    println!("assistant> {}", out.trim());
    Ok(())
}

fn ensure_test_image(image: &Path) -> Result<PathBuf> {
    use anyhow::Context;
    if image.exists() {
        return Ok(image.to_path_buf());
    }
    eprintln!(
        "image not found at {}; downloading synthetic fixture from HF",
        image.display()
    );
    let cached = hf_helper::ensure_hf_file(TEST_IMAGE_HF_REPO, TEST_IMAGE_HF_FILE)
        .with_context(|| format!("failed to fetch {TEST_IMAGE_HF_REPO}/{TEST_IMAGE_HF_FILE}"))?;
    if let Some(parent) = image.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create parent directory {}", parent.display())
            })?;
        }
    }
    std::fs::copy(&cached, image)
        .with_context(|| format!("failed to copy {} -> {}", cached.display(), image.display()))?;
    Ok(image.to_path_buf())
}
