"""OCR transcription: single page inference, the batch loop, and output export."""

import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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


def _ocr_worker(transcribe, page_path):
    """Transcribe one page, mapping errors to placeholder text."""
    try:
        return transcribe(page_path)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "[ERROR: out of memory, page skipped]"
    except Exception as e:
        return f"[ERROR: {e}]"


def run_ocr_pages(transcribe, pages, output_base, formats,
                  direction="rtl", total=None, workers=1, parallel_mode="thread",
                  parallel_engine=None, log=print, progress=None,
                  should_stop=lambda: False):
    """OCR each page and export the transcript in the requested formats.

    transcribe(page_path) -> str     engine-specific page transcription
    output_base            path without extension
    formats                subset of FORMATS
    total                  known page count (None if unknown; derived from
                           len(pages) when possible)
    workers                parallel page workers (1 = sequential)
    parallel_mode          "thread" or "process" (chrome needs "process":
                           its DLL races on shared state across threads but
                           is safe across processes)
    parallel_engine        engine name for process workers (each child
                           rebuilds its own engine)
    log(msg) -> None       called for page results and the summary
    progress(i, total, elapsed, name) -> None   called after each page
    should_stop() -> bool checked before each page
    """
    total_start = time.time()
    if total is None:
        try:
            total = len(pages)
        except TypeError:
            total = None  # lazy iterator (PDF pages render on demand)

    if workers > 1:
        if parallel_mode == "process":
            texts = _run_process_parallel(parallel_engine, pages, total, workers, log, progress, should_stop)
        else:
            texts = _run_parallel(transcribe, pages, total, workers, log, progress, should_stop)
    else:
        texts = _run_sequential(transcribe, pages, total, log, progress, should_stop)

    total_elapsed = time.time() - total_start

    write_outputs(texts, output_base, formats, log, direction)

    log("\n--- Summary ---")
    log(f"Pages processed: {len(texts)}")
    log(f"Total time: {total_elapsed:.2f}s")
    log(f"Average time/page: {total_elapsed / max(len(texts), 1):.2f}s")


def _run_sequential(transcribe, pages, total, log, progress, should_stop):
    texts = []
    for i, page_path in enumerate(pages, start=1):
        if should_stop():
            log("Stopped by user.")
            break

        page_start = time.time()
        text = _ocr_worker(transcribe, page_path)
        if text.startswith("[ERROR"):
            log(f"  {text} ({page_path.name})")
        texts.append(text)

        elapsed = time.time() - page_start
        if i % 10 == 0:
            torch.cuda.empty_cache()

        log(f"[{i}/{total or '?'}] {page_path.name} - {elapsed:.2f}s")
        if progress:
            progress(i, total or 0, elapsed, page_path.name)
    return texts


def _run_parallel(transcribe, pages, total, workers, log, progress, should_stop):
    # Parallel workers need random access; materialize the lazy PDF iterator.
    try:
        pages = list(pages)
        if total is None:
            total = len(pages)
    except TypeError:
        pass  # already a list

    texts = [None] * len(pages)
    stop = False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for idx, page_path in enumerate(pages):
            if should_stop() or stop:
                stop = True
                break
            page_start = time.time()
            fut = pool.submit(_ocr_worker, transcribe, page_path)
            pending[fut] = (idx, page_path, page_start)

        for fut in as_completed(pending):
            idx, page_path, page_start = pending[fut]
            text = fut.result()
            if text.startswith("[ERROR"):
                log(f"  {text} ({page_path.name})")
            texts[idx] = text
            elapsed = time.time() - page_start
            log(f"[{idx + 1}/{total or '?'}] {page_path.name} - {elapsed:.2f}s")
            if progress:
                progress(idx + 1, total or 0, elapsed, page_path.name)

    if stop:
        log("Stopped by user.")
    return [t for t in texts if t is not None]


# Process-pool worker: the transcribe closure can't be pickled and module
# globals don't survive Windows spawn, so each child rebuilds its own engine
# from the engine name.
_ENGINE_BUILDERS = {}


def _register_engine_builder(name, builder):
    _ENGINE_BUILDERS[name] = builder


def _process_worker(args):
    engine_name, page_path = args
    try:
        transcribe = _ENGINE_BUILDERS[engine_name](page_path)
    except KeyError:
        return f"[ERROR: unknown engine {engine_name}]"
    return _ocr_worker(transcribe, page_path)


def _run_process_parallel(engine_name, pages, total, workers, log, progress, should_stop):
    try:
        pages = list(pages)
        if total is None:
            total = len(pages)
    except TypeError:
        pass

    texts = [None] * len(pages)
    stop = False
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for idx, page_path in enumerate(pages):
            if should_stop() or stop:
                stop = True
                break
            page_start = time.time()
            fut = pool.submit(_process_worker, (engine_name, str(page_path)))
            pending[fut] = (idx, page_path, page_start)

        for fut in as_completed(pending):
            idx, page_path, page_start = pending[fut]
            text = fut.result()
            if text.startswith("[ERROR"):
                log(f"  {text} ({page_path.name})")
            texts[idx] = text
            elapsed = time.time() - page_start
            log(f"[{idx + 1}/{total or '?'}] {page_path.name} - {elapsed:.2f}s")
            if progress:
                progress(idx + 1, total or 0, elapsed, page_path.name)

    if stop:
        log("Stopped by user.")
    return [t for t in texts if t is not None]


def write_outputs(texts, output_base, formats, log=print, direction="rtl", title=None):
    """Write the transcript body in each requested format.

    md/txt are written directly; epub/pdf/azw3 go through calibre
    (ebook-convert), chaining md -> epub -> pdf/azw3. pdf/azw3 need the
    epub, and epub needs the md, so intermediates are written as needed.
    direction ("rtl"/"ltr") marks the md body direction for renderers.
    """
    formats = [f for f in formats if f in FORMATS]
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if title is None:
        title = output_base.name
        if title.endswith("_transcript"):
            title = title[: -len("_transcript")]
    requested = set(formats)

    body = "\n\n".join(texts).strip() + "\n"
    md_body = body
    if direction == "rtl":
        md_body = '<div dir="rtl">\n\n' + body + "</div>\n"

    # The ebook body must NOT be wrapped in a single raw-HTML div: markdown
    # inside a raw HTML block is not parsed, so "## Page" headings and
    # paragraph breaks would flatten to literal text and calibre's splitter
    # would find no legal split points (SplitError on large books). Use
    # per-paragraph HTML tags instead; markdown stays parseable outside them.
    ebook_parts = []
    for i, text in enumerate(texts):
        if text.startswith("[ERROR"):
            ebook_parts.append(text)
            continue
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paras:
            continue
        joined = "\n\n".join(f'<p dir="{direction}">{p}</p>' for p in paras)
        ebook_parts.append(f"## Page {i + 1}\n\n{joined}")
    ebook_body = "\n\n".join(ebook_parts) + "\n"

    if "md" in requested or {"epub", "pdf", "azw3"} & requested:
        md_path = output_base.with_suffix(".md")
        md_path.write_text(md_body, encoding="utf-8")
        log(f"Wrote {md_path.name}")

    if "txt" in requested:
        txt_path = output_base.with_suffix(".txt")
        txt_path.write_text(body, encoding="utf-8")
        log(f"Wrote {txt_path.name}")

    if {"epub", "pdf", "azw3"} & requested:
        ebook_md_path = output_base.with_suffix(".ebook.md")
        ebook_md_path.write_text(ebook_body, encoding="utf-8")
        epub_path = output_base.with_suffix(".epub")
        # Headings + paragraph tags give calibre legal split points; the
        # default 260KB flow-size now splits properly instead of SplitError.
        _convert(ebook_md_path, epub_path, log, "--title", title)

    if "pdf" in requested:
        _convert(epub_path, output_base.with_suffix(".pdf"), log,
                 "--title", title, "--pdf-serif-family", PDF_SERIF_FAMILY)

    if "azw3" in requested:
        _convert(epub_path, output_base.with_suffix(".azw3"), log,
                 "--title", title)


def _convert(src, dst, log, *extra_args):
    log(f"Converting {src.name} -> {dst.name} ...")
    subprocess.run(
        ["ebook-convert", str(src), str(dst), *extra_args],
        check=True,
        capture_output=True,
        text=True,
    )
    log(f"Wrote {dst.name}")
