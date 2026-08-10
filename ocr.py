"""OCR transcription: single page inference, the batch loop, and output export."""

import subprocess
import time
from pathlib import Path

import torch
from PIL import Image

PROMPT_TEXT = "Transcribe all text in this image exactly as it appears."
FORMATS = ("md", "pdf", "azw3", "epub", "txt")
PDF_SERIF_FAMILY = "Segoe UI"  # covers Persian glyphs for calibre PDF export


def transcribe_page(processor, model, image_path: Path, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text = processor.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def run_ocr_pages(transcribe, pages, output_base, formats,
                  log=print, progress=None, should_stop=lambda: False):
    """OCR each page and export the transcript in the requested formats.

    transcribe(page_path) -> str     engine-specific page transcription
    output_base            path without extension
    formats                subset of FORMATS
    log(msg) -> None       called for page results and the summary
    progress(i, total, elapsed, name) -> None   called after each page
    should_stop() -> bool checked before each page
    """
    texts = []
    timings = []
    total_start = time.time()
    total = len(pages)

    for i, page_path in enumerate(pages, start=1):
        if should_stop():
            log("Stopped by user.")
            break

        page_start = time.time()

        try:
            text = transcribe(page_path)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            text = "[ERROR: out of memory, page skipped]"
            log(f"  [OOM] {page_path.name}")
        except Exception as e:
            text = f"[ERROR: {e}]"
            log(f"  [ERROR] {page_path.name}: {e}")

        texts.append(text)
        elapsed = time.time() - page_start
        timings.append((page_path.name, round(elapsed, 2)))

        if i % 10 == 0:
            torch.cuda.empty_cache()

        log(f"[{i}/{total}] {page_path.name} - {elapsed:.2f}s")
        if progress:
            progress(i, total, elapsed, page_path.name)

    total_elapsed = time.time() - total_start

    write_outputs(texts, output_base, formats, log)

    log("\n--- Summary ---")
    log(f"Pages processed: {len(timings)}")
    log(f"Total time: {total_elapsed:.2f}s")
    log(f"Average time/page: {sum(t for _, t in timings) / len(timings):.2f}s")


def write_outputs(texts, output_base, formats, log=print):
    """Write the transcript body in each requested format.

    md/txt are written directly; epub/pdf/azw3 go through calibre
    (ebook-convert), chaining md -> epub -> pdf/azw3. pdf/azw3 need the
    epub, and epub needs the md, so intermediates are written as needed.
    """
    formats = [f for f in formats if f in FORMATS]
    body = "\n\n".join(texts).strip() + "\n"
    output_base = Path(output_base)
    requested = set(formats)

    need_md = "md" in requested or {"epub", "pdf", "azw3"} & requested
    if need_md:
        md_path = output_base.with_suffix(".md")
        md_path.write_text(body, encoding="utf-8")
        log(f"Wrote {md_path.name}")

    if "txt" in requested:
        txt_path = output_base.with_suffix(".txt")
        txt_path.write_text(body, encoding="utf-8")
        log(f"Wrote {txt_path.name}")

    if {"epub", "pdf", "azw3"} & requested:
        epub_path = output_base.with_suffix(".epub")
        if "epub" not in requested or not epub_path.exists():
            _convert(md_path, epub_path, log)

    if "pdf" in requested:
        _convert(epub_path, output_base.with_suffix(".pdf"), log,
                 "--pdf-serif-family", PDF_SERIF_FAMILY)

    if "azw3" in requested:
        _convert(epub_path, output_base.with_suffix(".azw3"), log)


def _convert(src, dst, log, *extra_args):
    log(f"Converting {src.name} -> {dst.name} ...")
    subprocess.run(
        ["ebook-convert", str(src), str(dst), *extra_args],
        check=True,
        capture_output=True,
        text=True,
    )
    log(f"Wrote {dst.name}")
