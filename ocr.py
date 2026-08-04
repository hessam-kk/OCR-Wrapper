"""OCR transcription: single page inference and the batch loop."""

import time
from pathlib import Path

import torch
from PIL import Image

PROMPT_TEXT = "Transcribe all text in this image exactly as it appears."


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


def run_ocr_pages(processor, model, pages, output_path, max_new_tokens,
                  log=print, progress=None, should_stop=lambda: False):
    """OCR each page and write the transcript.

    log(msg) -> None      called for page results and the summary
    progress(i, total, elapsed, name) -> None   called after each page
    should_stop() -> bool checked before each page
    """
    timings = []
    total_start = time.time()
    total = len(pages)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, page_path in enumerate(pages, start=1):
            if should_stop():
                log("Stopped by user.")
                break

            page_start = time.time()

            try:
                text = transcribe_page(processor, model, page_path, max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                text = "[ERROR: out of memory, page skipped]"
                log(f"  [OOM] {page_path.name}")
            except Exception as e:
                text = f"[ERROR: {e}]"
                log(f"  [ERROR] {page_path.name}: {e}")

            elapsed = time.time() - page_start
            timings.append((page_path.name, round(elapsed, 2)))

            out_f.write(f"## Page {i}: {page_path.name}\n\n{text}\n\n")
            out_f.flush()

            if i % 10 == 0:
                torch.cuda.empty_cache()

            log(f"[{i}/{total}] {page_path.name} - {elapsed:.2f}s")
            if progress:
                progress(i, total, elapsed, page_path.name)

    total_elapsed = time.time() - total_start

    log("\n--- Summary ---")
    log(f"Pages processed: {len(timings)}")
    log(f"Total time: {total_elapsed:.2f}s")
    log(f"Average time/page: {sum(t for _, t in timings) / len(timings):.2f}s")
    log(f"Transcript saved to: {output_path}")
