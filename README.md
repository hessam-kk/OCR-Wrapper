# Bina OCR

Batch OCR extraction using [Reza2kn/Bina-0.1](https://huggingface.co/Reza2kn/Bina-0.1) — a Persian OCR vision-language model (~0.7B params).

![Bina OCR GUI](images/ocr%20preview%201.png)

## Features

- **PDF input** — renders pages at configurable DPI via PyMuPDF
- **Image folder input** — processes sorted image files (jpg, png, webp, etc.)
- **Three engines** — `bina` (vision-model OCR, handles scanned/images), `pdf-inspector` (instant text extraction for text-based PDFs), or `oneocr` (Windows Snipping Tool OCR)
- **Persian normalization** — optional hazm post-processing that reinserts half-spaces (ZWNJ) and unifies glyphs/digits, which OCR models often drop
- **Tkinter GUI** — file pickers, progress bar, live log, engine + GPU/CPU selectors (launches by default with no args)
- **CLI mode** — for scripting and batch runs
- **CPU fallback** — `--cpu` flag, or GPU/CPU selector in the GUI
- **Modular code** — split into `model.py`, `pages.py`, `ocr.py`, `inspector.py`, `windows_ocr.py`, `gui.py` around the `book_ocr_batch.py` entry point
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

Fast text extraction of a text-based PDF (no OCR, no model download):

```bash
python book_ocr_batch.py --pdf book.pdf --engine inspector --output_file transcript.md
```

Windows Snipping Tool OCR (high accuracy, fully offline — needs model files, see [oneocr setup](#oneocr-setup-windows-snipping-tool-ocr)):

```bash
python book_ocr_batch.py --pdf book.pdf --engine oneocr --output_file transcript.md
```

With Persian normalization (reinserts half-spaces/ZWNJ that OCR models often drop — recommended for Persian text):

```bash
python book_ocr_batch.py --pdf book.pdf --engine oneocr --normalize --output_file transcript.md
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
| `--engine` | `bina` | `bina` (vision OCR), `inspector` (pdf-inspector, PDF only) or `oneocr` (Windows OCR) |
| `--normalize` | off | Normalize Persian text with hazm (reinserts half-spaces/ZWNJ) |
| `--cpu` | off | Force CPU even if GPU is available |
| `--gui` | — | Launch GUI explicitly |

## Output

- **Transcript** — Markdown file with `## Page N: filename` sections

### Sample result (page 1 of «۱» , RTL Persian)

![Sample OCR output](images/sample%20result%201.png)

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

## oneocr setup (Windows Snipping Tool OCR)

The `oneocr` engine is the same high-accuracy OCR model used by Windows Snipping Tool's "Text actions" — Windows-only, fully offline, and typically faster and more accurate than classic OCR libraries. The pip package is just a wrapper; the model itself ships inside the Snipping Tool app, so it needs a one-time manual setup.

**1. Install the wrapper**

```bash
pip install oneocr
```

**2. Get the model files** (`oneocr.dll`, `oneocr.onemodel`, `onnxruntime.dll`)

The package doesn't ship them — they live inside the Snipping Tool app package. The easiest way to get them without fighting Windows' locked `WindowsApps` folder permissions:

- Go to [store.rg-adguard.net](https://store.rg-adguard.net), paste in `https://apps.microsoft.com/detail/9mz95kl8mr0l` (Snipping Tool's store link), and download the newest `Microsoft.ScreenSketch` `.msixbundle`
- Rename it to `.zip` and extract it
- Extract the inner `SnippingToolApp` `.msix` for your CPU arch (`x64` for AMD64, `ARM64` for ARM) the same way — a `.msix` is also just a zip
- The three files are in the resulting `SnippingTool` folder

**3. Drop them into place**

```bash
mkdir "%USERPROFILE%\.config\oneocr"
copy extracted\SnippingTool\oneocr.dll "%USERPROFILE%\.config\oneocr\"
copy extracted\SnippingTool\oneocr.onemodel "%USERPROFILE%\.config\oneocr\"
copy extracted\SnippingTool\onnxruntime.dll "%USERPROFILE%\.config\oneocr\"
```

That's it — `oneocr.OcrEngine()` picks the files up automatically. To verify:

```bash
python -c "from windows_ocr import get_ocr_engine; get_ocr_engine(); print('oneocr ready')"
```

**Alternative (often locked):** if Snipping Tool is installed, locate its live install folder with `Get-AppxPackage Microsoft.ScreenSketch | Select-Object -ExpandProperty InstallLocation` and copy the three files from its `SnippingTool` subfolder. `WindowsApps` is locked down by the OS, so if you hit permission errors, fall back to the extraction method above.

> **Note:** the `oneocr` engine uses a model extracted from Microsoft's proprietary Snipping Tool — check licensing before distributing.

## Notes

- Model is cached locally after first download (~1.3GB); the tool checks the cache and asks before downloading
- pdf-inspector is instant (<1s) but only handles text-based PDFs — scanned pages need the `bina` engine
- oneocr does not emit U+200C (ZWNJ) — the `--normalize` flag fixes half-spaces (`می‌رود`) and unifies digits/glyphs via [hazm](https://github.com/sobhe/hazm); works with any engine
- 150 DPI is default for PDF rendering; raise for better accuracy, lower for speed
- Expect minutes/page on low-end GPUs; ~10-30s/page on a proper GPU
- `torch.cuda.empty_cache()` runs every 10 pages for low-VRAM GPUs
- Stop button (GUI) / Ctrl-C (CLI) stops after the current page

## License

[MIT](LICENSE)
