# Bina OCR

Batch OCR extraction using [Reza2kn/Bina-0.1](https://huggingface.co/Reza2kn/Bina-0.1) — a Persian OCR vision-language model (~0.7B params).

![Bina OCR GUI](ocr%20preview%201.png)

## Features

- **PDF input** — renders pages at configurable DPI via PyMuPDF
- **Image folder input** — processes sorted image files (jpg, png, webp, etc.)
- **Tkinter GUI** — file pickers, progress bar, live log, GPU/CPU selector (launches by default with no args)
- **CLI mode** — for scripting and batch runs
- **CPU fallback** — `--cpu` flag, or GPU/CPU selector in the GUI
- **Modular code** — split into `model.py`, `pages.py`, `ocr.py`, `gui.py` around the `book_ocr_batch.py` entry point
- **Model check before download** — shows cache status and repo size, asks before downloading

## Requirements

```bash
pip install -r requirements.txt
```

> **Note:** For GPU support, install PyTorch with CUDA first:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```
> then `pip install -r requirements.txt`.

> **Note:** Bina-0.1 uses the `qwen3_5` architecture, which requires a recent transformers build. If model loading fails, install from source:
> ```
> pip install --force-reinstall git+https://github.com/huggingface/transformers.git
> ```

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

### Sample result (page 1 of «۱» , RTL Persian)

![Sample OCR output](sample%20result%201.png)

> ۱
> همه غافلگیر شدند.
> در سال ۲۰۰۵، جنی مک‌کورمیک با استفاده از تلسکوپ
> بیست‌وپنج‌سانتی‌متری رصدخانهٔ فارم کاول، در اوکلند نیوزیلند
> توانست سیاره‌ای ناشناخته را در منظومه‌ای کشف کند که پانزده هزار
> سال نوری با زمین فاصله داشت. جنی چند سال بعدتر بازهم مایهٔ
> شگفتی همه شد، چون یک سیارک! تازه کشف کرد و روی حساب
> وطن‌پرستی اسمش را هم گذاشت نیوزیلند. از آن به بعد چیزی حدود
> بیست مقالهٔ علمی را با همکاری دیگران نوشته که در مجلات
> دانشگاهی و ازجمله نشریهٔ معتبر ساینس! منتشر شده و کار به جایی
> رسیده که گیتس مک‌فادن، بازیگر مجموعهٔ تلویزیونی پیشتازان فضا،
> وقتی او را در نمایشگاه قصه‌های علمی‌تخیلی دید، از او امضا گرفت.
> این‌ها به کنار، شاید مهم‌ترین دستاورد جنی همانی باشد که کمتر
> کسی می‌داند: او یکی از مهم‌ترین ستاره‌شناسان امروز دنیاست، بدون
> تحصیلات دانشگاهی.
> راستش را بخواهید حتی دبیرستان را هم تمام نکرده.
> سایهٔ پدر بر سرش نبوده و در شهر کوچک وانگانویی، بزرگ شده و

> The screenshot above shows the rendered output — [full transcript](book_transcript.md).

## Notes

- Model is cached locally after first download (~1.3GB); the tool checks the cache and asks before downloading
- 150 DPI is default for PDF rendering; raise for better accuracy, lower for speed
- Expect minutes/page on low-end GPUs; ~10-30s/page on a proper GPU
- `torch.cuda.empty_cache()` runs every 10 pages for low-VRAM GPUs
- Stop button (GUI) / Ctrl-C (CLI) stops after the current page
