"""OCR transcription: single page inference, the batch loop, and output export."""

import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image

PROMPT_TEXT = "Transcribe all text in this image exactly as it appears."
FORMATS = ("md", "pdf", "azw3", "epub", "txt")
PDF_SERIF_FAMILY = "Segoe UI"  # covers Persian glyphs for calibre PDF export

# Sentence-ending punctuation (after stripping trailing closing quotes etc.).
# Used to decide whether a page boundary cut a paragraph mid-sentence.
TERMINAL_PUNCT = (".", "!", "؟", "…")
TRAILING_WRAPPERS = ('"', "'", "»", ")", "”", "’")


def _looks_like_paragraph_end(text: str) -> bool:
    text = text.rstrip()
    while text and text[-1] in TRAILING_WRAPPERS:
        text = text[:-1]
    return bool(text) and text[-1] in TERMINAL_PUNCT


def _merge_pages(texts):
    """Turn per-page OCR text into a flat paragraph list, merging any
    paragraph that a page boundary cut mid-sentence. Returns
    (paragraphs, page_breaks) where page_breaks is the set of paragraph
    indices at which a new physical page genuinely begins."""
    paragraphs, page_breaks = [], set()
    for text in texts:
        if text.startswith("[ERROR"):
            paragraphs.append(text)
            page_breaks.add(len(paragraphs) - 1)
            continue
        page_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not page_paras:
            continue
        if paragraphs and not paragraphs[-1].startswith("[ERROR") and not _looks_like_paragraph_end(paragraphs[-1]):
            paragraphs[-1] = paragraphs[-1].rstrip() + " " + page_paras[0]
            page_paras = page_paras[1:]
        else:
            page_breaks.add(len(paragraphs))
        paragraphs.extend(page_paras)
    return paragraphs, page_breaks


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
    # Keep pages lazy: prefetch only `workers` ahead so PDF rendering stays
    # incremental (materializing the whole iterator defeats lazy rendering).
    # Use a while-pending wait loop so newly submitted futures are awaited.
    from concurrent.futures import wait, FIRST_COMPLETED

    texts = {}
    stop = False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        page_iter = enumerate(pages, start=1)
        for _ in range(workers):
            try:
                idx, page_path = next(page_iter)
            except StopIteration:
                break
            page_start = time.time()
            fut = pool.submit(_ocr_worker, transcribe, page_path)
            pending[fut] = (idx, page_path, page_start)

        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                idx, page_path, page_start = pending.pop(fut)
                text = fut.result()
                if text.startswith("[ERROR"):
                    log(f"  {text} ({page_path.name})")
                texts[idx] = text
                elapsed = time.time() - page_start
                log(f"[{idx}/{total or '?'}] {page_path.name} - {elapsed:.2f}s")
                if progress:
                    progress(idx, total or 0, elapsed, page_path.name)

                if not (should_stop() or stop):
                    try:
                        nidx, npath = next(page_iter)
                    except StopIteration:
                        continue
                    page_start = time.time()
                    fut = pool.submit(_ocr_worker, transcribe, npath)
                    pending[fut] = (nidx, npath, page_start)
                else:
                    stop = True

    if stop:
        log("Stopped by user.")
    return [texts[i] for i in sorted(texts)]


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
    # Same lazy prefetch pattern as threads; each child builds its own engine.
    from concurrent.futures import wait, FIRST_COMPLETED

    texts = {}
    stop = False

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        page_iter = enumerate(pages, start=1)
        for _ in range(workers):
            try:
                idx, page_path = next(page_iter)
            except StopIteration:
                break
            page_start = time.time()
            fut = pool.submit(_process_worker, (engine_name, str(page_path)))
            pending[fut] = (idx, page_path, page_start)

        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                idx, page_path, page_start = pending.pop(fut)
                text = fut.result()
                if text.startswith("[ERROR"):
                    log(f"  {text} ({page_path.name})")
                texts[idx] = text
                elapsed = time.time() - page_start
                log(f"[{idx}/{total or '?'}] {page_path.name} - {elapsed:.2f}s")
                if progress:
                    progress(idx, total or 0, elapsed, page_path.name)

                if not (should_stop() or stop):
                    try:
                        nidx, npath = next(page_iter)
                    except StopIteration:
                        continue
                    page_start = time.time()
                    fut = pool.submit(_process_worker, (engine_name, str(npath)))
                    pending[fut] = (nidx, npath, page_start)
                else:
                    stop = True

    if stop:
        log("Stopped by user.")
    return [texts[i] for i in sorted(texts)]


def write_outputs(texts, output_base, formats, log=print, direction="rtl", title=None):
    """Write the transcript body in each requested format.

    md/txt are written directly; epub/pdf/azw3 go through calibre
    (ebook-convert), chaining md -> epub -> pdf/azw3. pdf/azw3 need the
    epub, and epub needs the md, so intermediates are written as needed.
    direction ("rtl"/"ltr") marks the md body direction for renderers.
    If texts is None and the .md exists, re-export from it (skip OCR).
    """
    formats = [f for f in formats if f in FORMATS]
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    if title is None:
        title = output_base.name
        if title.endswith("_transcript"):
            title = title[: -len("_transcript")]
    requested = set(formats)

    md_path = output_base.with_suffix(".md")
    pagemap_path = output_base.with_suffix(".pagemap.json")
    if texts is None:
        # Re-export from an existing markdown (skip OCR). Paragraph boundaries
        # are recovered from the file's blank lines; physical page boundaries
        # come from the sidecar written on the original OCR run. Without it
        # (old .md files), no page breaks are inserted.
        if not md_path.exists():
            raise FileNotFoundError(f"Skip-OCR requested but {md_path.name} does not exist")
        md_body = md_path.read_text(encoding="utf-8")
        body = md_body
        if md_body.startswith('<div dir="rtl">'):
            body = md_body.replace('<div dir="rtl">', "").replace("</div>", "").strip()
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [body]
        if pagemap_path.exists():
            page_breaks = set(json.loads(pagemap_path.read_text(encoding="utf-8")))
            pagemap_path.unlink()
        else:
            page_breaks = set()
    else:
        paragraphs, page_breaks = _merge_pages(texts)
        body = "\n\n".join(paragraphs).strip() + "\n"
        md_body = body
        pagemap_path.write_text(json.dumps(sorted(page_breaks)), encoding="utf-8")
    if direction == "rtl" and not md_body.startswith("<div"):
        md_body = '<div dir="rtl">\n\n' + md_body + "</div>\n"

    # The ebook body must NOT be wrapped in a single raw-HTML div: markdown
    # inside a raw HTML block is not parsed, so headings and paragraph breaks
    # would flatten to literal text and calibre's splitter would find no legal
    # split points (SplitError on large books). Use per-paragraph HTML tags;
    # markdown stays parseable outside them. No "## Page N" markers — the
    # output is a clean continuous document. Invisible page breaks at genuine
    # page starts give calibre split points (so azw3 doesn't hit "Could not
    # find chunk for aid" on single-file epubs) without visible page labels.
    #
    # Kindle e-ink doesn't do Arabic contextual shaping, so Persian text is
    # pre-shaped into joined presentation forms for the ebook outputs only.
    # The .md/txt stay canonical (searchable); this reshaping is expected to
    # make the ebook variant non-searchable — accepted trade-off.
    reshaped = direction == "rtl" and _reshape_persian

    PAGE_BREAK = '<div style="page-break-before:always"></div>'
    ebook_parts = []
    for i, p in enumerate(paragraphs):
        if i in page_breaks and i > 0:
            ebook_parts.append(PAGE_BREAK)
        tag = _reshape_persian(p) if reshaped else p
        ebook_parts.append(f'<p dir="{direction}">{tag}</p>')
    ebook_body = "\n\n".join(ebook_parts) + "\n"

    if "md" in requested or {"epub", "pdf", "azw3"} & requested:
        md_path.write_text(md_body, encoding="utf-8")
        log(f"Wrote {md_path.name}")

    if "txt" in requested:
        txt_path = output_base.with_suffix(".txt")
        txt_path.write_text(body if not body.startswith("<div") else
                            body.replace('<div dir="rtl">', "").replace("</div>", "").strip(),
                            encoding="utf-8")
        log(f"Wrote {txt_path.name}")

    if {"epub", "pdf", "azw3"} & requested:
        # Feed calibre a minimal HTML shell with document-level RTL + fa
        # so Kindle/readers shape Persian correctly (letters join instead
        # of appearing separated). The per-paragraph <p dir> tags stay.
        ebook_md_path = output_base.with_suffix(".ebook.md")
        ebook_html = (
            '<!DOCTYPE html>\n<html dir="rtl" lang="fa">\n<body>\n'
            + ebook_body
            + "</body>\n</html>\n"
        )
        ebook_md_path.write_text(ebook_html, encoding="utf-8")
        epub_path = output_base.with_suffix(".epub")
        _convert(ebook_md_path, epub_path, log, "--title", title)

    if "pdf" in requested:
        _convert(epub_path, output_base.with_suffix(".pdf"), log,
                 "--title", title, "--pdf-serif-family", PDF_SERIF_FAMILY)

    if "azw3" in requested:
        _convert(epub_path, output_base.with_suffix(".azw3"), log,
                 "--title", title)


def _reshape_persian(text: str) -> str:
    """Convert Persian text to pre-joined presentation forms for Kindle.

    Only Arabic-script runs are reshaped; Latin/digits/markup pass through.
    """
    import arabic_reshaper

    ARABIC = ("؀", "ۿ")

    def reshape_run(run):
        return arabic_reshaper.reshape(run) if any(ARABIC[0] <= c <= ARABIC[1] for c in run) else run

    out = []
    buf = []
    for ch in text:
        if ARABIC[0] <= ch <= ARABIC[1]:
            buf.append(ch)
        else:
            if buf:
                out.append(reshape_run("".join(buf)))
                buf = []
            out.append(ch)
    if buf:
        out.append(reshape_run("".join(buf)))
    return "".join(out)


def _convert(src, dst, log, *extra_args):
    log(f"Converting {src.name} -> {dst.name} ...")
    subprocess.run(
        ["ebook-convert", str(src), str(dst), *extra_args],
        check=True,
        capture_output=True,
        text=True,
    )
    log(f"Wrote {dst.name}")


if __name__ == "__main__":
    # Page boundary cuts a sentence in two -> merged into one paragraph.
    p, pb = _merge_pages([
        "می‌خواستم به تو چیزی",
        "بگویم که مهم بود.",
        "[ERROR: page skipped]",
        "این پاراگراف کامل است.",
    ])
    assert p == ["می‌خواستم به تو چیزی بگویم که مهم بود.", "[ERROR: page skipped]", "این پاراگراف کامل است."], p
    # paragraph 0 starts page 1 and absorbs page 2 (cut mid-sentence, no
    # break marker); the ERROR marker starts page 3; the last paragraph
    # starts page 4.
    assert pb == {0, 1, 2}, pb
    # Page boundary falls at a paragraph end -> kept, marked as a break.
    p, pb = _merge_pages(["جمله تمام شده است.", "جمله بعدی."])
    assert p == ["جمله تمام شده است.", "جمله بعدی."] and pb == {0, 1}, (p, pb)
    assert not _looks_like_paragraph_end("می‌خواستم به تو چیزی")
    assert _looks_like_paragraph_end("تمام شد.")
    assert _looks_like_paragraph_end('گفت: "تمام شد".')
    print("merge_pages self-check OK")
