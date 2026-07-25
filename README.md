# Bina OCR

Batch OCR extraction using [Reza2kn/Bina-0.1](https://huggingface.co/Reza2kn/Bina-0.1) — a Persian OCR vision-language model (~0.7B params).

## Features

- **PDF input** — renders pages at configurable DPI via PyMuPDF
- **Image folder input** — processes sorted image files (jpg, png, webp, etc.)
- **Tkinter GUI** — file pickers, progress bar, live log (launches by default with no args)
- **CLI mode** — for scripting and batch runs
- **Kaggle notebook** — ready-to-run `.ipynb` with GPU support
- **CPU fallback** — `--cpu` flag for machines without CUDA

## Requirements

```
pip install torch torchvision pymupdf transformers accelerate safetensors pillow tqdm
```

For GPU support, install PyTorch with CUDA:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

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
| `--timing_log` | `page_timings.csv` | Per-page timing CSV |
| `--max_new_tokens` | `1024` | Max tokens generated per page |
| `--limit` | all | Process only first N pages |
| `--cpu` | off | Force CPU even if GPU is available |
| `--gui` | — | Launch GUI explicitly |

### Kaggle

1. Upload your PDF as a Kaggle Dataset
2. Open `bina_ocr_kaggle.ipynb` in Kaggle
3. Set GPU accelerator (Settings → GPU)
4. Edit the `INPUT_PATH` config cell
5. Run all cells, restart kernel after install cell

## Output

- **Transcript** — Markdown file with `## Page N: filename` sections
- **Timing log** — CSV with per-page inference times

## Notes

- Model is cached locally after first download (~1.3GB)
- 300 DPI is default for PDF rendering; lower for faster processing
- On CPU, expect ~4 min/page; on GPU (P100/T4), ~10-30s/page
- `torch.cuda.empty_cache()` runs every 10 pages for low-VRAM GPUs
