"""Tkinter GUI for batch OCR."""

import itertools
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import torch

from model import MODEL_ID, load_model, model_cache_info, repo_size_gb, write_inspector_transcript
from normalize import get_normalizer, normalize_transcribe
from ocr import FORMATS, run_ocr_pages, transcribe_page
from pages import PDF_DPI, get_page_images, get_pdf_images
from chrome_ocr_engine import chrome_transcribe_page, get_screenai_engine
from windows_ocr import get_ocr_engine, oneocr_transcribe_page


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bina OCR Batch")
        self.root.geometry("820x700")
        self.root.resizable(True, True)

        self.running = False

        # --- Input source ---
        src_frame = ttk.LabelFrame(root, text="Input Source", padding=8)
        src_frame.pack(fill="x", padx=10, pady=(10, 4))

        self.input_type = tk.StringVar(value="pdf")
        ttk.Radiobutton(src_frame, text="PDF File", variable=self.input_type, value="pdf").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(src_frame, text="Image Folder", variable=self.input_type, value="dir").grid(row=0, column=1, sticky="w", padx=(10, 0))

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
        self.output_file = tk.StringVar(value="book_transcript")
        ttk.Entry(out_frame, textvariable=self.output_file, width=50).grid(row=0, column=1, sticky="ew", padx=(4, 4))
        ttk.Button(out_frame, text="Browse", command=self._browse_output).grid(row=0, column=2)

        self.folder_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(out_frame, text="Save in folder", variable=self.folder_var).grid(row=0, column=3, sticky="w", padx=(12, 0))

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
        self.dpi_var = tk.StringVar(value=str(PDF_DPI))
        ttk.Combobox(opt_frame, textvariable=self.dpi_var, values=["150", "200", "300", "400"], width=5, state="readonly").grid(row=0, column=5, sticky="w", padx=(4, 0))

        ttk.Label(opt_frame, text="Engine:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.engine_var = tk.StringVar(value="chrome")
        ttk.Radiobutton(opt_frame, text="Chrome OCR (Screen AI) - best", variable=self.engine_var, value="chrome").grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(6, 0))
        ttk.Radiobutton(opt_frame, text="Windows OCR (oneocr)", variable=self.engine_var, value="oneocr").grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Radiobutton(opt_frame, text="Bina OCR (OCR)", variable=self.engine_var, value="bina").grid(row=1, column=3, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Radiobutton(opt_frame, text="pdf-inspector (Parser)", variable=self.engine_var, value="inspector").grid(row=1, column=4, sticky="w", padx=(8, 0), pady=(6, 0))

        ttk.Label(opt_frame, text="Device:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        ttk.Radiobutton(opt_frame, text="GPU", variable=self.device_var, value="cuda").grid(row=2, column=1, sticky="w", padx=(4, 0), pady=(6, 0))
        ttk.Radiobutton(opt_frame, text="CPU", variable=self.device_var, value="cpu").grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(6, 0))

        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Normalize Persian (half-spaces)", variable=self.normalize_var).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Label(opt_frame, text="Direction:").grid(row=3, column=3, sticky="w", pady=(6, 0))
        self.direction_var = tk.StringVar(value="rtl")
        ttk.Radiobutton(opt_frame, text="RTL", variable=self.direction_var, value="rtl").grid(row=3, column=4, sticky="w", padx=(4, 0), pady=(6, 0))
        ttk.Radiobutton(opt_frame, text="LTR", variable=self.direction_var, value="ltr").grid(row=3, column=5, sticky="w", padx=(4, 0), pady=(6, 0))

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
            if self.input_type.get() == "pdf":
                self.output_file.set(Path(path).stem + "_transcript")

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save transcript as",
            defaultextension=".md",
            initialfile=self.output_file.get(),
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.output_file.set(Path(path).stem)

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

        threading.Thread(target=self._run_ocr, daemon=True).start()

    def _stop(self):
        self.running = False
        self._log("Stopping after current page...")
        self.stop_btn.configure(state="disabled")

    def _run_ocr(self):
        try:
            input_path = Path(self.input_path.get().strip())
            input_type = self.input_type.get()
            engine = self.engine_var.get()

            if engine == "inspector":
                if input_type != "pdf":
                    self.root.after(0, lambda: messagebox.showerror("Error", "pdf-inspector only processes PDF files. Pick a PDF or switch to Bina OCR."))
                    return
                if not input_path.is_file():
                    self.root.after(0, lambda: messagebox.showerror("Error", f"PDF not found: {input_path}"))
                    return
                output_base = self._output_base()
                formats = [f for f, v in self.format_vars.items() if v.get()]
                self.root.after(0, lambda: self.status_label.configure(text="Extracting text..."))
                write_inspector_transcript(
                    input_path, output_base, formats, self.direction_var.get(),
                    log=lambda m: self.root.after(0, lambda s=m: self._log(s)),
                )
                self.root.after(0, lambda: self.status_label.configure(text="Done"))
                return

            # Collect pages
            pdf_tmp_dir = None
            limit = self.page_limit.get()
            if input_type == "pdf":
                if not input_path.is_file():
                    self.root.after(0, lambda: messagebox.showerror("Error", f"PDF not found: {input_path}"))
                    return
                pdf_tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
                dpi = int(self.dpi_var.get())
                self.root.after(0, lambda: self._log(f"Rendering PDF pages at {dpi} DPI (lazy, page by page)..."))
                pages, total = get_pdf_images(input_path, pdf_tmp_dir, dpi)
                if limit > 0:
                    pages = itertools.islice(pages, limit)
                    total = min(total, limit)
            else:
                if not input_path.is_dir():
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Folder not found: {input_path}"))
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
                self.root.after(0, lambda: self.status_label.configure(text="Loading Windows OCR engine..."))
                ocr_engine = get_ocr_engine()
                transcribe = lambda p: oneocr_transcribe_page(ocr_engine, p)
            elif engine == "chrome":
                self.root.after(0, lambda: self.status_label.configure(text="Loading Chrome Screen AI..."))
                ocr_engine = get_screenai_engine()
                transcribe = lambda p: chrome_transcribe_page(ocr_engine, p)
            else:
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
                processor, model, device = load_model(
                    force_cpu=self.device_var.get() == "cpu",
                    log=lambda m: self.root.after(0, lambda s=m: self._log(s)),
                )
                max_tokens = self.max_tokens.get()
                transcribe = lambda p: transcribe_page(processor, model, p, max_tokens)

            if self.normalize_var.get():
                transcribe = normalize_transcribe(transcribe, normalizer)

            self.root.after(0, lambda: self.status_label.configure(text=f"Processing 0/{total} pages..."))
            self.root.after(0, lambda: self.progress.configure(maximum=total))

            def on_progress(i, tot, elapsed, name):
                self.progress.configure(value=i)
                self.status_label.configure(text=f"Processing {i}/{tot} pages... ({elapsed:.1f}s)")

            run_ocr_pages(
                transcribe, pages, output_base, formats,
                direction=self.direction_var.get(),
                total=total,
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
