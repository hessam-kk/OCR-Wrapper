"""Tkinter GUI for batch OCR."""

import itertools
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import torch

from model import (
    MODEL_ID,
    load_model,
    model_cache_info,
    repo_size_gb,
    write_inspector_transcript,
)
from normalize import get_normalizer, normalize_transcribe
from ocr import FORMATS, run_ocr_pages, transcribe_page, write_outputs
from pages import PDF_DPI, get_page_images, get_pdf_images
from chrome_ocr_engine import chrome_transcribe_page, get_screenai_engine
from windows_ocr import get_ocr_engine, oneocr_transcribe_page
from pdf_batch import create_pdf_jobs, pdfs_in_folder, process_pdf_jobs


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR-Wrapper")
        self.root.geometry("820x700")
        self.root.resizable(True, True)

        self.running = False
        self.pdf_paths = []

        # --- Input source ---
        src_frame = ttk.LabelFrame(root, text="Input Source", padding=8)
        src_frame.pack(fill="x", padx=10, pady=(10, 4))

        self.input_type = tk.StringVar(value="pdf")
        input_choices = (
            ("PDF File", "pdf"),
            ("Multiple PDFs", "pdfs"),
            ("PDF Folder", "pdf_dir"),
            ("Image Folder", "dir"),
        )
        for column, (label, value) in enumerate(input_choices):
            ttk.Radiobutton(
                src_frame,
                text=label,
                variable=self.input_type,
                value=value,
                command=self._input_type_changed,
            ).grid(row=0, column=column, sticky="w", padx=(0 if column == 0 else 10, 0))

        self.input_path = tk.StringVar()
        path_frame = ttk.Frame(src_frame)
        path_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        ttk.Entry(path_frame, textvariable=self.input_path, width=60).pack(side="left", fill="x", expand=True)
        ttk.Button(path_frame, text="Browse", command=self._browse_input).pack(side="left", padx=(4, 0))

        src_frame.columnconfigure(1, weight=1)

        # --- Output ---
        out_frame = ttk.LabelFrame(root, text="Output", padding=8)
        out_frame.pack(fill="x", padx=10, pady=4)

        self.output_label = ttk.Label(out_frame, text="Transcript:")
        self.output_label.grid(row=0, column=0, sticky="w")
        self.output_file = tk.StringVar(value="book_transcript")
        ttk.Entry(out_frame, textvariable=self.output_file, width=50).grid(row=0, column=1, sticky="ew", padx=(4, 4))
        ttk.Button(out_frame, text="Browse", command=self._browse_output).grid(row=0, column=2)

        self.folder_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(out_frame, text="Save in folder", variable=self.folder_var).grid(row=0, column=3, sticky="w", padx=(12, 0))

        self.skip_ocr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            out_frame,
            text="Skip OCR (re-export from existing md)",
            variable=self.skip_ocr_var,
        ).grid(row=0, column=4, sticky="w", padx=(12, 0))

        ttk.Label(out_frame, text="Formats:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.format_vars = {}
        fmt_frame = ttk.Frame(out_frame)
        fmt_frame.grid(row=1, column=1, columnspan=2, sticky="w", padx=(4, 0), pady=(4, 0))
        for col, fmt in enumerate(FORMATS):
            var = tk.BooleanVar(value=fmt == "md")
            self.format_vars[fmt] = var
            ttk.Checkbutton(fmt_frame, text=fmt, variable=var).grid(row=0, column=col, sticky="w", padx=(0, 12))

        out_frame.columnconfigure(1, weight=1)

        # --- Options ---
        opt_frame = ttk.LabelFrame(root, text="Options", padding=8)
        opt_frame.pack(fill="x", padx=10, pady=4)

        ttk.Label(opt_frame, text="Max tokens:").grid(row=0, column=0, sticky="w")
        self.max_tokens = tk.IntVar(value=1024)
        ttk.Spinbox(opt_frame, from_=64, to=4096, textvariable=self.max_tokens, width=8).grid(row=0, column=1, sticky="w", padx=(4, 0))

        ttk.Label(opt_frame, text="Page limit:").grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.page_limit = tk.IntVar(value=0)
        ttk.Spinbox(opt_frame, from_=0, to=99999, textvariable=self.page_limit, width=8).grid(row=0, column=3, sticky="w", padx=(4, 0))

        ttk.Label(opt_frame, text="DPI:").grid(row=0, column=4, sticky="w", padx=(8, 0))
        self.dpi_var = tk.StringVar(value="300")
        ttk.Combobox(
            opt_frame,
            textvariable=self.dpi_var,
            values=["150", "200", "300", "400"],
            width=5,
            state="readonly",
        ).grid(row=0, column=5, sticky="w", padx=(4, 0))

        ttk.Label(opt_frame, text="Engine:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.engine_var = tk.StringVar(value="chrome")
        ttk.Radiobutton(
            opt_frame,
            text="Chrome (Screen AI)",
            variable=self.engine_var,
            value="chrome",
        ).grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(6, 0))
        ttk.Radiobutton(opt_frame, text="Windows (oneocr)", variable=self.engine_var, value="oneocr").grid(row=1, column=2, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Radiobutton(opt_frame, text="bina (OCR)", variable=self.engine_var, value="bina").grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Radiobutton(opt_frame, text="pdf-inspector", variable=self.engine_var, value="inspector").grid(row=1, column=4, sticky="w", padx=(6, 0), pady=(6, 0))

        ttk.Label(opt_frame, text="Device:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        ttk.Radiobutton(opt_frame, text="GPU", variable=self.device_var, value="cuda").grid(row=2, column=1, sticky="w", padx=(4, 0), pady=(6, 0))
        ttk.Radiobutton(opt_frame, text="CPU", variable=self.device_var, value="cpu").grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(6, 0))

        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opt_frame,
            text="Normalize Persian (half-space)",
            variable=self.normalize_var,
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))

        worker_frame = ttk.Frame(opt_frame)
        worker_frame.grid(row=3, column=1, sticky="w", padx=(30, 0), pady=(6, 0))
        self.workers_var = tk.IntVar(value=1)
        ttk.Label(worker_frame, text="Workers:").pack(side="left")
        ttk.Spinbox(worker_frame, from_=1, to=8, textvariable=self.workers_var, width=3).pack(side="left")

        dir_frame = ttk.Frame(opt_frame)
        dir_frame.grid(row=3, column=2, sticky="w", pady=(6, 0))
        self.direction_var = tk.StringVar(value="rtl")
        ttk.Label(dir_frame, text="Dir:").pack(side="left")
        ttk.Radiobutton(dir_frame, text="RTL", variable=self.direction_var, value="rtl").pack(side="left")
        ttk.Radiobutton(dir_frame, text="LTR", variable=self.direction_var, value="ltr").pack(side="left")

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

        self.timer_label = ttk.Label(btn_frame, text="0s")
        self.timer_label.pack(side="left")
        self.start_btn = ttk.Button(btn_frame, text="Start OCR", command=self._start)
        self.start_btn.pack(side="right")
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="right", padx=(0, 8))

    # --- File dialogs ---

    def _input_type_changed(self):
        self.pdf_paths = []
        self.input_path.set("")
        is_batch = self.input_type.get() in ("pdfs", "pdf_dir")
        self.output_label.configure(text="Output folder:" if is_batch else "Transcript:")
        if is_batch:
            self.output_file.set("transcripts")

    def _browse_input(self):
        input_type = self.input_type.get()

        if input_type in ("dir", "pdf_dir"):
            title = "Select image folder" if input_type == "dir" else "Select folder containing PDFs"
            path = filedialog.askdirectory(title=title)
        elif input_type == "pdfs":
            paths = filedialog.askopenfilenames(
                title="Select PDF files",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            )
            if not paths:
                return
            self.pdf_paths = [Path(path) for path in paths]
            self.input_path.set(f"{len(paths)} PDF files selected")
            self.output_file.set(str(self.pdf_paths[0].parent / "transcripts"))
            return
        else:
            path = filedialog.askopenfilename(
                title="Select PDF file",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            )
        if path:
            self.pdf_paths = []
            self.input_path.set(path)
            if input_type == "pdf":
                self.output_file.set(Path(path).stem + "_transcript")
            elif input_type == "pdf_dir":
                self.output_file.set(str(Path(path) / "transcripts"))

    def _browse_output(self):
        if self.input_type.get() in ("pdfs", "pdf_dir"):
            path = filedialog.askdirectory(title="Select output folder")
            if path:
                self.output_file.set(path)
            return

        path = filedialog.asksaveasfilename(
            title="Save transcript as",
            defaultextension=".md",
            initialfile=self.output_file.get(),
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.output_file.set(Path(path).stem)

    def _batch_jobs(self):
        if self.input_type.get() == "pdfs":
            pdf_paths = self.pdf_paths
        else:
            pdf_paths = pdfs_in_folder(Path(self.input_path.get().strip()))
        return create_pdf_jobs(
            pdf_paths,
            Path(self.output_file.get().strip()),
            separate_folders=self.folder_var.get(),
        )

    # --- Logging ---

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _ask_download(self, size_gb):
        return messagebox.askyesno(
            "Model not downloaded",
            f"Model {MODEL_ID} is not cached locally.\n\n" f"Download size: ~{size_gb:.1f} GB\n\n" "Download it now?",
        )

    def _output_base(self):
        base = Path(self.output_file.get())
        if self.folder_var.get():
            base = base / base.name
        return base

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

        self._start_time = time.time()
        self._tick_timer()

        threading.Thread(target=self._run_ocr, daemon=True).start()

    def _tick_timer(self):
        if not self.running:
            return
        self.timer_label.configure(text=f"{int(time.time() - self._start_time)}s")
        self.root.after(1000, self._tick_timer)

    def _stop(self):
        self.running = False
        self._finalize_timer()
        self._log("Stopping after current page...")
        self.stop_btn.configure(state="disabled")

    def _finalize_timer(self):
        if hasattr(self, "_start_time"):
            self.timer_label.configure(text=f"{int(time.time() - self._start_time)}s")

    def _run_ocr(self):
        try:
            input_path = Path(self.input_path.get().strip())
            input_type = self.input_type.get()
            engine = self.engine_var.get()

            if input_type in ("pdfs", "pdf_dir"):
                self._run_pdf_batch(engine)
                return

            if self.skip_ocr_var.get():
                output_base = self._output_base()
                formats = [f for f, v in self.format_vars.items() if v.get()]
                self.root.after(0, lambda: self.status_label.configure(text="Re-exporting..."))
                write_outputs(
                    None,
                    output_base,
                    formats,
                    log=lambda m: self.root.after(0, lambda s=m: self._log(s)),
                    direction=self.direction_var.get(),
                )
                self.root.after(0, lambda: self.status_label.configure(text="Done"))
                return

            if engine == "inspector":
                if input_type != "pdf":
                    self.root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Error",
                            "pdf-inspector only processes PDF files. Pick a PDF or switch to bina.",
                        ),
                    )
                    return
                if not input_path.is_file():
                    self.root.after(
                        0,
                        lambda: messagebox.showerror("Error", f"PDF not found: {input_path}"),
                    )
                    return
                output_base = self._output_base()
                formats = [f for f, v in self.format_vars.items() if v.get()]
                self.root.after(0, lambda: self.status_label.configure(text="Extracting text..."))
                write_inspector_transcript(
                    input_path,
                    output_base,
                    formats,
                    self.direction_var.get(),
                    log=lambda m: self.root.after(0, lambda s=m: self._log(s)),
                )
                self.root.after(0, lambda: self.status_label.configure(text="Done"))
                return

            # Collect pages
            pdf_tmp_dir = None
            limit = self.page_limit.get()
            if input_type == "pdf":
                if not input_path.is_file():
                    self.root.after(
                        0,
                        lambda: messagebox.showerror("Error", f"PDF not found: {input_path}"),
                    )
                    return
                pdf_tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
                dpi = int(self.dpi_var.get())
                self.root.after(
                    0,
                    lambda: self._log(f"Rendering PDF pages at {dpi} DPI (lazy, page by page)..."),
                )
                pages, total = get_pdf_images(input_path, pdf_tmp_dir, dpi)
                if limit > 0:
                    pages = itertools.islice(pages, limit)
                    total = min(total, limit)
            else:
                if not input_path.is_dir():
                    self.root.after(
                        0,
                        lambda: messagebox.showerror("Error", f"Folder not found: {input_path}"),
                    )
                    return
                pages = get_page_images(input_path)
                total = len(pages)
                if limit > 0:
                    pages = pages[:limit]
                    total = len(pages)

            self.root.after(0, lambda: self._log(f"Found {total} pages to process."))

            output_base = self._output_base()
            formats = [f for f, v in self.format_vars.items() if v.get()]

            if self.normalize_var.get():
                normalizer = get_normalizer()
                self.root.after(0, lambda: self._log("[INFO] Persian normalization enabled (hazm)"))

            if engine == "oneocr":
                self.root.after(
                    0,
                    lambda: self.status_label.configure(text="Loading Windows OCR engine..."),
                )
                ocr_engine = get_ocr_engine()
                transcribe = lambda p: oneocr_transcribe_page(ocr_engine, p)
            elif engine == "chrome":
                self.root.after(
                    0,
                    lambda: self.status_label.configure(text="Loading Chrome Screen AI..."),
                )
                ocr_engine = get_screenai_engine()
                transcribe = lambda p: chrome_transcribe_page(ocr_engine, p)
            else:
                # Load model
                cached, cache_size = model_cache_info()
                if not cached:
                    size_gb = repo_size_gb()
                    self.root.after(
                        0,
                        lambda s=size_gb: self._log(f"[INFO] Model {MODEL_ID} is not downloaded yet (~{s:.1f} GB)."),
                    )
                    ask = self._ask_download(size_gb)
                    if not ask:
                        self.root.after(0, lambda: self._log("Aborted - model not downloaded."))
                        return

                self.root.after(0, lambda: self.status_label.configure(text="Loading model..."))
                processor, model, device = load_model(
                    force_cpu=self.device_var.get() == "cpu",
                    log=lambda m: self.root.after(0, lambda s=m: self._log(s)),
                )
                max_tokens = self.max_tokens.get()
                transcribe = lambda p: transcribe_page(processor, model, p, max_tokens)

            if self.normalize_var.get():
                transcribe = normalize_transcribe(transcribe, normalizer)

            self.root.after(
                0,
                lambda: self.status_label.configure(text=f"Processing 0/{total} pages..."),
            )
            self.root.after(0, lambda: self.progress.configure(maximum=total))

            def on_progress(i, tot, elapsed, name):
                self.progress.configure(value=i)
                self.status_label.configure(text=f"Processing {i}/{tot} pages... ({elapsed:.1f}s)")

            workers = self.workers_var.get()
            if engine == "bina" and workers > 1:
                self.root.after(
                    0,
                    lambda: self._log("[INFO] bina engine is single-device - ignoring workers"),
                )
                workers = 1
            # chrome's DLL races across threads but is safe across processes
            parallel_mode = "process" if engine == "chrome" else "thread"

            run_ocr_pages(
                transcribe,
                pages,
                output_base,
                formats,
                direction=self.direction_var.get(),
                total=total,
                workers=workers,
                parallel_mode=parallel_mode,
                parallel_engine=engine,
                log=lambda m: self.root.after(0, lambda s=m: self._log(s)),
                progress=lambda i, t, e, n: self.root.after(0, lambda: on_progress(i, t, e, n)),
                should_stop=lambda: not self.running,
            )

            if pdf_tmp_dir:
                for f in pdf_tmp_dir.iterdir():
                    f.unlink()
                pdf_tmp_dir.rmdir()

            self.root.after(0, lambda: self.status_label.configure(text="Done"))
            self.root.after(0, lambda: self.progress.configure(value=total))

        except Exception as e:
            self.root.after(
                0,
                lambda err=e: (
                    self._log(f"FATAL: {err}"),
                    self.status_label.configure(text="Error"),
                    messagebox.showerror("Error", str(err)),
                ),
            )
        finally:
            self.running = False
            self.root.after(
                0,
                lambda: (
                    self._finalize_timer(),
                    self.start_btn.configure(state="normal"),
                    self.stop_btn.configure(state="disabled"),
                ),
            )

    def _run_pdf_batch(self, engine):
        """Process selected PDFs with one shared engine instance."""
        jobs = self._batch_jobs()
        formats = [name for name, enabled in self.format_vars.items() if enabled.get()]
        direction = self.direction_var.get()

        def log(message):
            self.root.after(0, lambda value=message: self._log(value))

        self.root.after(0, lambda: self.progress.configure(maximum=len(jobs), value=0))
        log(f"[INFO] Batch contains {len(jobs)} PDF files.")

        if self.skip_ocr_var.get():

            def process(job, index, total):
                self.root.after(
                    0,
                    lambda: self.status_label.configure(text=f"Re-exporting PDF {index}/{total}..."),
                )
                write_outputs(None, job.output_base, formats, log, direction)
                self.root.after(0, lambda: self.progress.configure(value=index))

        elif engine == "inspector":

            def process(job, index, total):
                self.root.after(
                    0,
                    lambda: self.status_label.configure(text=f"Extracting PDF {index}/{total}..."),
                )
                write_inspector_transcript(
                    job.input_path,
                    job.output_base,
                    formats,
                    direction,
                    log=log,
                )
                self.root.after(0, lambda: self.progress.configure(value=index))

        else:
            transcribe = self._create_batch_transcriber(engine, log)
            if transcribe is None:
                return

            def process(job, index, total_documents):
                with tempfile.TemporaryDirectory(prefix="ocr_pdf_") as temp_dir:
                    dpi = int(self.dpi_var.get())
                    pages, total_pages = get_pdf_images(job.input_path, Path(temp_dir), dpi)
                    limit = self.page_limit.get()
                    if limit > 0:
                        pages = itertools.islice(pages, limit)
                        total_pages = min(total_pages, limit)

                    log(f"[INFO] Found {total_pages} pages to process.")
                    self.root.after(
                        0,
                        lambda: self.progress.configure(maximum=total_pages, value=0),
                    )

                    workers = self.workers_var.get()
                    if engine == "bina" and workers > 1:
                        log("[INFO] bina engine is single-device - ignoring workers")
                        workers = 1

                    def on_progress(page, page_total, elapsed):
                        self.progress.configure(value=page)
                        self.status_label.configure(text=(f"PDF {index}/{total_documents}, page " f"{page}/{page_total} ({elapsed:.1f}s)"))

                    run_ocr_pages(
                        transcribe,
                        pages,
                        job.output_base,
                        formats,
                        direction=direction,
                        total=total_pages,
                        workers=workers,
                        parallel_mode=("process" if engine == "chrome" else "thread"),
                        parallel_engine=engine,
                        log=log,
                        progress=lambda page, page_total, elapsed, _name: self.root.after(
                            0,
                            lambda: on_progress(page, page_total, elapsed),
                        ),
                        should_stop=lambda: not self.running,
                    )

        summary = process_pdf_jobs(
            jobs,
            process,
            log=log,
            should_stop=lambda: not self.running,
        )
        log("\n=== Batch summary ===")
        log(f"PDFs succeeded: {summary.succeeded}/{len(jobs)}")
        log(f"Output directory: {Path(self.output_file.get()).resolve()}")

        if summary.stopped:
            self.root.after(0, lambda: self.status_label.configure(text="Stopped"))
        elif summary.failures:
            log(f"PDFs failed: {len(summary.failures)}")
            self.root.after(0, lambda: self.status_label.configure(text="Done with errors"))
        else:
            self.root.after(0, lambda: self.status_label.configure(text="Done"))

    def _create_batch_transcriber(self, engine, log):
        """Initialize an OCR engine once, then reuse it for every PDF."""
        normalizer = None
        if self.normalize_var.get():
            normalizer = get_normalizer()
            log("[INFO] Persian normalization enabled (hazm)")

        if engine == "oneocr":
            self.root.after(
                0,
                lambda: self.status_label.configure(text="Loading Windows OCR engine..."),
            )
            ocr_engine = get_ocr_engine()
            transcribe = lambda path: oneocr_transcribe_page(ocr_engine, path)
        elif engine == "chrome":
            self.root.after(
                0,
                lambda: self.status_label.configure(text="Loading Chrome Screen AI..."),
            )
            ocr_engine = get_screenai_engine()
            transcribe = lambda path: chrome_transcribe_page(ocr_engine, path)
        else:
            cached, _cache_size = model_cache_info()
            if not cached:
                size_gb = repo_size_gb()
                log(f"[INFO] Model {MODEL_ID} is not downloaded yet " f"(~{size_gb:.1f} GB).")
                if not self._ask_download(size_gb):
                    log("Aborted - model not downloaded.")
                    return None

            self.root.after(0, lambda: self.status_label.configure(text="Loading model..."))
            processor, model, _device = load_model(
                force_cpu=self.device_var.get() == "cpu",
                log=log,
            )
            max_tokens = self.max_tokens.get()
            transcribe = lambda path: transcribe_page(processor, model, path, max_tokens)

        if normalizer is not None:
            transcribe = normalize_transcribe(transcribe, normalizer)
        return transcribe
