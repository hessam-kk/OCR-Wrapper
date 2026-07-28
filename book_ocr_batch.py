"""
Batch OCR extraction using Reza2kn/Bina-0.1 (Persian OCR vision-language model).

Usage:
    python book_ocr_batch.py --input_dir ./book_pages --output_file book_transcript.md
    python book_ocr_batch.py --pdf book.pdf --output_file book_transcript.md
    python book_ocr_batch.py --gui

Requirements:
    pip install torch transformers accelerate safetensors pillow pymupdf

Notes:
- Designed for low-VRAM GPUs (e.g. 4GB laptop GPUs). Model is ~0.7B params.
- Processes images in sorted filename order, so name pages like
  page_001.jpg, page_002.jpg, ... to keep book order correct.
- PDF files are rendered page-by-page at 300 DPI for OCR.
- Writes a running transcript + per-page timing log so you can gauge
  whether local inference is fast enough for a full book.
"""

import argparse
import csv
import os
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import fitz  # PyMuPDF
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "Reza2kn/Bina-0.1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROMPT_TEXT = "Transcribe all text in this image exactly as it appears."
PDF_DPI = 300


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


def get_page_images(input_dir: Path):
    pages = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not pages:
        raise FileNotFoundError(f"No image files found in {input_dir}")
    return pages


def get_pdf_images(pdf_path: Path, tmp_dir: Path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=PDF_DPI)
        out = tmp_dir / f"page_{i + 1:04d}.png"
        pix.save(out)
        pages.append(out)
    doc.close()
    return pages


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


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bina OCR Batch")
        self.root.geometry("720x620")
        self.root.resizable(True, True)

        self.running = False

        # --- Input source ---
        src_frame = ttk.LabelFrame(root, text="Input Source", padding=8)
        src_frame.pack(fill="x", padx=10, pady=(10, 4))

        self.input_type = tk.StringVar(value="dir")
        ttk.Radiobutton(src_frame, text="Image Folder", variable=self.input_type, value="dir").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(src_frame, text="PDF File", variable=self.input_type, value="pdf").grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.input_path = tk.StringVar()
        path_frame = ttk.Frame(src_frame)
        path_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Entry(path_frame, textvariable=self.input_path, width=60).pack(side="left", fill="x", expand=True)
        ttk.Button(path_frame, text="Browse", command=self._browse_input).pack(side="left", padx=(4, 0))

        src_frame.columnconfigure(1, weight=1)

        # --- Output ---
        out_frame = ttk.LabelFrame(root, text="Output", padding=8)
        out_frame.pack(fill="x", padx=10, pady=4)

        ttk.Label(out_frame, text="Transcript:").grid(row=0, column=0, sticky="w")
        self.output_file = tk.StringVar(value="book_transcript.md")
        ttk.Entry(out_frame, textvariable=self.output_file, width=50).grid(row=0, column=1, sticky="ew", padx=(4, 4))
        ttk.Button(out_frame, text="Browse", command=self._browse_output).grid(row=0, column=2)

        ttk.Label(out_frame, text="Timing log:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.timing_log = tk.StringVar(value="page_timings.csv")
        ttk.Entry(out_frame, textvariable=self.timing_log, width=50).grid(row=1, column=1, sticky="ew", padx=(4, 4), pady=(4, 0))
        ttk.Button(out_frame, text="Browse", command=self._browse_timing).grid(row=1, column=2, pady=(4, 0))

        out_frame.columnconfigure(1, weight=1)

        # --- Options ---
        opt_frame = ttk.LabelFrame(root, text="Options", padding=8)
        opt_frame.pack(fill="x", padx=10, pady=4)

        ttk.Label(opt_frame, text="Max new tokens:").grid(row=0, column=0, sticky="w")
        self.max_tokens = tk.IntVar(value=1024)
        ttk.Spinbox(opt_frame, from_=64, to=4096, textvariable=self.max_tokens, width=8).grid(row=0, column=1, sticky="w", padx=(4, 16))

        ttk.Label(opt_frame, text="Page limit (0=all):").grid(row=0, column=2, sticky="w")
        self.page_limit = tk.IntVar(value=0)
        ttk.Spinbox(opt_frame, from_=0, to=99999, textvariable=self.page_limit, width=8).grid(row=0, column=3, sticky="w", padx=(4, 0))

        # --- Progress ---
        prog_frame = ttk.Frame(root, padding=8)
        prog_frame.pack(fill="x", padx=10)

        self.progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress.pack(fill="x")
        self.status_label = ttk.Label(prog_frame, text="Ready")
        self.status_label.pack(anchor="w", pady=(2, 0))

        # --- Log ---
        log_frame = ttk.LabelFrame(root, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.log = scrolledtext.ScrolledText(log_frame, height=12, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True)

        # --- Buttons ---
        btn_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        btn_frame.pack(fill="x")

        self.start_btn = ttk.Button(btn_frame, text="Start OCR", command=self._start)
        self.start_btn.pack(side="right")
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="right", padx=(0, 8))

    # --- File dialogs ---

    def _browse_input(self):
        if self.input_type.get() == "dir":
            path = filedialog.askdirectory(title="Select image folder")
        else:
            path = filedialog.askopenfilename(
                title="Select PDF file",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            )
        if path:
            self.input_path.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save transcript as",
            defaultextension=".md",
            initialfile=self.output_file.get(),
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.output_file.set(path)

    def _browse_timing(self):
        path = filedialog.asksaveasfilename(
            title="Save timing log as",
            defaultextension=".csv",
            initialfile=self.timing_log.get(),
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.timing_log.set(path)

    # --- Logging ---

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _ask_download(self, size_gb):
        return messagebox.askyesno(
            "Model not downloaded",
            f"Model {MODEL_ID} is not cached locally.\n\n"
            f"Download size: ~{size_gb:.1f} GB\n\n"
            "Download it now?",
        )

    # --- OCR worker ---

    def _start(self):
        input_path = self.input_path.get().strip()
        if not input_path:
            messagebox.showwarning("Missing input", "Please select an input folder or PDF file.")
            return

        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress["value"] = 0
        self.status_label.configure(text="Loading model...")
        self._log("Starting OCR...")

        threading.Thread(target=self._run_ocr, daemon=True).start()

    def _stop(self):
        self.running = False
        self._log("Stopping after current page...")
        self.stop_btn.configure(state="disabled")

    def _run_ocr(self):
        try:
            input_path = Path(self.input_path.get().strip())
            input_type = self.input_type.get()

            # Collect pages
            pdf_tmp_dir = None
            if input_type == "pdf":
                if not input_path.is_file():
                    self.root.after(0, lambda: messagebox.showerror("Error", f"PDF not found: {input_path}"))
                    return
                pdf_tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
                self.root.after(0, lambda: self._log(f"Rendering PDF pages at {PDF_DPI} DPI..."))
                pages = get_pdf_images(input_path, pdf_tmp_dir)
            else:
                if not input_path.is_dir():
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Folder not found: {input_path}"))
                    return
                pages = get_page_images(input_path)

            limit = self.page_limit.get()
            if limit > 0:
                pages = pages[:limit]

            total = len(pages)
            self.root.after(0, lambda: self._log(f"Found {total} pages to process."))

            # Load model
            cached, cache_size = model_cache_info()
            if not cached:
                size_gb = repo_size_gb()
                self.root.after(0, lambda s=size_gb: self._log(f"[INFO] Model {MODEL_ID} is not downloaded yet (~{s:.1f} GB)."))
                ask = self._ask_download(size_gb)
                if not ask:
                    self.root.after(0, lambda: self._log("Aborted - model not downloaded."))
                    return

            self.root.after(0, lambda: self.status_label.configure(text="Loading model..."))
            processor, model, device = load_model(log=lambda m: self.root.after(0, lambda s=m: self._log(s)))

            output_path = Path(self.output_file.get())
            timing_path = Path(self.timing_log.get())
            max_tokens = self.max_tokens.get()

            timings = []
            total_start = time.time()

            self.root.after(0, lambda: self.status_label.configure(text=f"Processing 0/{total} pages..."))
            self.root.after(0, lambda: self.progress.configure(maximum=total))

            with open(output_path, "w", encoding="utf-8") as out_f:
                for i, page_path in enumerate(pages, start=1):
                    if not self.running:
                        self.root.after(0, lambda: self._log("Stopped by user."))
                        break

                    page_start = time.time()

                    try:
                        text = transcribe_page(processor, model, page_path, max_tokens)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        text = "[ERROR: out of memory, page skipped]"
                        self.root.after(0, lambda p=page_path: self._log(f"  [OOM] {p.name}"))
                    except Exception as e:
                        text = f"[ERROR: {e}]"
                        self.root.after(0, lambda p=page_path, err=e: self._log(f"  [ERROR] {p.name}: {err}"))

                    elapsed = time.time() - page_start
                    timings.append((page_path.name, round(elapsed, 2)))

                    out_f.write(f"## Page {i}: {page_path.name}\n\n{text}\n\n")
                    out_f.flush()

                    if i % 10 == 0:
                        torch.cuda.empty_cache()

                    cur_i, cur_total, cur_elapsed = i, total, elapsed
                    self.root.after(0, lambda idx=cur_i, tot=cur_total, el=cur_elapsed, name=page_path.name: (
                        self.progress.configure(value=idx),
                        self.status_label.configure(text=f"Processing {idx}/{tot} pages... ({el:.1f}s)"),
                        self._log(f"[{idx}/{tot}] {name} - {el:.2f}s"),
                    ))

            total_elapsed = time.time() - total_start

            if pdf_tmp_dir:
                for f in pdf_tmp_dir.iterdir():
                    f.unlink()
                pdf_tmp_dir.rmdir()

            if timings:
                with open(timing_path, "w", newline="", encoding="utf-8") as csv_f:
                    writer = csv.writer(csv_f)
                    writer.writerow(["page_file", "seconds"])
                    writer.writerows(timings)

                avg_time = sum(t for _, t in timings) / len(timings)
                summary = (
                    f"\n--- Summary ---\n"
                    f"Pages processed: {len(timings)}\n"
                    f"Total time: {total_elapsed:.2f}s\n"
                    f"Average time/page: {avg_time:.2f}s\n"
                    f"Transcript saved to: {output_path}\n"
                    f"Timing log saved to: {timing_path}"
                )
                self.root.after(0, lambda s=summary: self._log(s))

            self.root.after(0, lambda: self.status_label.configure(text="Done"))
            self.root.after(0, lambda: self.progress.configure(value=total))
            self.root.after(0, lambda: self._log("Complete."))

        except Exception as e:
            self.root.after(0, lambda err=e: (
                self._log(f"FATAL: {err}"),
                self.status_label.configure(text="Error"),
                messagebox.showerror("Error", str(err)),
            ))
        finally:
            self.running = False
            self.root.after(0, lambda: (
                self.start_btn.configure(state="normal"),
                self.stop_btn.configure(state="disabled"),
            ))


def main():
    parser = argparse.ArgumentParser(description="Batch OCR a folder of book page images or a PDF file.")
    parser.add_argument("--input_dir", help="Folder containing page images")
    parser.add_argument("--pdf", help="Path to a PDF file")
    parser.add_argument("--output_file", default="book_transcript.md", help="Combined transcript output path")
    parser.add_argument("--timing_log", default="page_timings.csv", help="Per-page timing CSV path")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens generated per page")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pages")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI instead of CLI")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    args = parser.parse_args()

    # Launch GUI if --gui or no CLI args provided
    if args.gui or (not args.input_dir and not args.pdf):
        root = tk.Tk()
        OCRApp(root)
        root.mainloop()
        return

    if not args.input_dir and not args.pdf:
        parser.error("Either --input_dir or --pdf is required.")
    if args.input_dir and args.pdf:
        parser.error("Use either --input_dir or --pdf, not both.")

    pdf_tmp_dir = None

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.is_file():
            parser.error(f"PDF file not found: {pdf_path}")
        pdf_tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
        pages = get_pdf_images(pdf_path, pdf_tmp_dir)
    else:
        input_dir = Path(args.input_dir)
        pages = get_page_images(input_dir)

    output_path = Path(args.output_file)
    timing_path = Path(args.timing_log)

    if args.limit:
        pages = pages[: args.limit]

    cached, _ = model_cache_info()
    if not cached:
        print(f"[INFO] Model {MODEL_ID} is not downloaded yet (~{repo_size_gb():.1f} GB).")
        try:
            confirm = input("Download it now? [y/N] ")
        except EOFError:
            confirm = ""
        if confirm.strip().lower() not in ("y", "yes"):
            print("Aborted - model not downloaded.")
            sys.exit(0)

    processor, model, device = load_model(force_cpu=args.cpu)

    timings = []
    total_start = time.time()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, page_path in enumerate(tqdm(pages, desc="OCR", unit="page"), start=1):
            page_start = time.time()

            try:
                text = transcribe_page(processor, model, page_path, args.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                text = "[ERROR: out of memory, page skipped]"
            except Exception as e:
                text = f"[ERROR: {e}]"

            elapsed = time.time() - page_start
            timings.append((page_path.name, round(elapsed, 2)))

            out_f.write(f"## Page {i}: {page_path.name}\n\n{text}\n\n")
            out_f.flush()

            if i % 10 == 0:
                torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start

    if pdf_tmp_dir:
        for f in pdf_tmp_dir.iterdir():
            f.unlink()
        pdf_tmp_dir.rmdir()

    with open(timing_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["page_file", "seconds"])
        writer.writerows(timings)

    avg_time = sum(t for _, t in timings) / len(timings)
    print("\n--- Summary ---")
    print(f"Pages processed: {len(pages)}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Average time/page: {avg_time:.2f}s")
    print(f"Transcript saved to: {output_path}")
    print(f"Timing log saved to: {timing_path}")


if __name__ == "__main__":
    main()
