"""Model loading and cache checks for the Bina OCR model."""

import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

from inspector import inspector_to_markdown
from ocr import write_outputs

MODEL_ID = "Reza2kn/Bina-0.1"
ENGINES = ("bina", "inspector", "oneocr", "chrome")


def model_cache_info():
    """Returns (is_cached, local_size_gb) for the model repo on disk."""
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(HF_HUB_CACHE) / f"models--{MODEL_ID.replace('/', '--')}"
    if not cache_dir.is_dir():
        return False, None
    total = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    return True, total / 1e9


def repo_size_gb():
    """Total download size (GB) of the model repo per the Hugging Face API."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_ID, files_metadata=True)
    return sum(s.size or 0 for s in info.siblings) / 1e9


def load_model(force_cpu=False, log=print):
    if force_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log(f"[INFO] Device: {device}, dtype: {dtype}")

    cached, size = model_cache_info()
    if cached:
        log(f"[INFO] Model found in local cache ({size:.1f} GB)")
    else:
        log("[INFO] Model not cached - will download on first load")

    log(f"[INFO] Loading processor for {MODEL_ID} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    log(f"[INFO] Processor loaded in {time.time() - t0:.1f}s")

    # Bina's config declares eos_token_id=248044, which is outside the vocab
    # (65425). The tokenizer's <|im_end|> (=2) is the real EOS; patch the
    # config so transformers doesn't warn and generation stops correctly.
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    if cfg.text_config.eos_token_id is not None and cfg.text_config.eos_token_id >= cfg.text_config.vocab_size:
        cfg.text_config.eos_token_id = processor.tokenizer.eos_token_id
        log(f"[INFO] Fixed invalid eos_token_id ({cfg.text_config.eos_token_id}) in model config")

    log(f"[INFO] Loading model weights (dtype={dtype}, device_map={'auto' if device == 'cuda' else 'None'}) ...")
    t0 = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        config=cfg,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    log(f"[INFO] Model weights loaded in {time.time() - t0:.1f}s")

    if device == "cpu":
        log(f"[INFO] Moving model to {device} ...")
        t0 = time.time()
        model = model.to(device)
        log(f"[INFO] Model moved in {time.time() - t0:.1f}s")

    n_params = sum(p.numel() for p in model.parameters())
    log(f"[INFO] Model ready: {n_params / 1e9:.2f}B parameters on {device}")
    if cached:
        log(f"[INFO] Model files on disk: {size:.1f} GB")
    model.eval()
    return processor, model, device


def write_inspector_transcript(pdf_path: Path, output_base: Path, formats, direction="rtl", log=print):
    """Fast text-based PDF extraction via pdf-inspector, exported in the requested formats."""
    t0 = time.time()
    log(f"[INFO] pdf-inspector: extracting text from {pdf_path.name} ...")
    markdown = inspector_to_markdown(pdf_path)
    elapsed = time.time() - t0

    write_outputs([markdown], output_base, formats, log, direction)

    log(f"[INFO] pdf-inspector finished in {elapsed:.2f}s")
    log("\n--- Summary ---")
    log(f"Pages processed: 1 (whole PDF)")
    log(f"Total time: {elapsed:.2f}s")
