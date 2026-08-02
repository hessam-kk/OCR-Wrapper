# Bina OCR

Batch OCR extraction using [Reza2kn/Bina-0.1](https://huggingface.co/Reza2kn/Bina-0.1) — a Persian OCR vision-language model (~0.7B params).

## Features

- **PDF input** — renders pages at configurable DPI via PyMuPDF
- **Image folder input** — processes sorted image files (jpg, png, webp, etc.)
- **Tkinter GUI** — file pickers, progress bar, live log, GPU/CPU selector (launches by default with no args)
- **CLI mode** — for scripting and batch runs
- **CPU fallback** — `--cpu` flag, or GPU/CPU selector in the GUI

## Requirements

```
pip install torch torchvision pymupdf transformers accelerate safetensors pillow tqdm
```

For GPU support, install PyTorch with CUDA:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> **Note:** Bina-0.1 uses the `qwen3_5` architecture, which requires a recent transformers build. If model loading fails, install from source:
> ```
> pip install --force-reinstall git+https://github.com/huggingface/transformers.git
> ```

> **Note:** MX150 (sm_61) requires PyTorch 2.6+cu126. Newer PyTorch versions only support sm_75+.

## Usage

### GUI (default)

```bash
python book_ocr_batch.py
```

### CLI — PDF

```bash
python book_ocr_batch.py --pdf book.pdf --output_file transcript.md
```

### CLI — Image folder

```bash
python book_ocr_batch.py --input_dir ./pages --output_file transcript.md
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | — | Path to PDF file |
| `--input_dir` | — | Folder of page images |
| `--output_file` | `book_transcript.md` | Transcript output path |
| `--max_new_tokens` | `1024` | Max tokens generated per page |
| `--limit` | all | Process only first N pages |
| `--cpu` | off | Force CPU even if GPU is available |
| `--gui` | — | Launch GUI explicitly |

## Output

- **Transcript** — Markdown file with `## Page N: filename` sections

## Notes

- Model is cached locally after first download (~1.3GB)
- 150 DPI is default for PDF rendering; raise for better accuracy, lower for speed
- On MX150, expect minutes/page; on a proper GPU, ~10-30s/page
- `torch.cuda.empty_cache()` runs every 10 pages for low-VRAM GPUs
