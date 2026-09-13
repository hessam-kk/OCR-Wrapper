# OCR-Wrapper 📚

Batch OCR extraction wrapping 4 engines: [Reza2kn/Bina-0.1](https://huggingface.co/Reza2kn/Bina-0.1) (Persian OCR vision-language model, ~0.7B params), pdf-inspector, oneocr, and Chrome Screen AI.

![OCR-Wrapper GUI](images/ocr%20preview%201.png)

## Features ✨

- **PDF input** 📄 — renders pages lazily at configurable DPI via PyMuPDF (300 DPI default)
- **Image folder input** 🖼️ — processes sorted image files (jpg, png, webp, bmp, tif, etc.)
- **Four engines** ⚙️ — `bina` (vision-model OCR, handles scanned/images), `pdf-inspector` (instant text extraction for text-based PDFs), `oneocr` (Windows Snipping Tool OCR), or `chrome` (Chrome/Edge Screen AI OCR)
- **Multi-format export** 📦 — Markdown, plain text, and ebooks (`md` `txt` `epub` `pdf` `azw3`; `epub`/`pdf`/`azw3` go through calibre's `ebook-convert`)
- **Clean continuous output** 🧩 — paragraphs are stitched across page boundaries so an OCR-split sentence doesn't break the reading flow, and nothing but the transcript is left in the output folder
- **Persian normalization** 🇮🇷 — optional hazm post-processing that reinserts half-spaces (ZWNJ) and unifies glyphs/digits, which OCR models often drop
- **Kindle-ready Persian** 📖 — ebook outputs pre-shape Arabic-script runs into joined presentation forms (via arabic-reshaper) so Kindle e-ink renders the script correctly
- **Parallel workers** 🚀 — optionally OCR pages concurrently (threads; the chrome engine uses processes since its DLL isn't thread-safe)
- **Skip OCR** ⏭️ — re-export an existing transcript to other formats without re-running the model
- **Qt GUI (PySide6)** 🖥️ — file pickers, format checkboxes, progress bar, live log, engine + GPU/CPU/DPI/workers/direction selectors (launches by default with no args); OCR runs on a worker thread and reports back through Qt signals
- **CLI mode** ⌨️ — for scripting and batch runs
- **CPU fallback** 💻 — `--cpu` flag, or GPU/CPU selector in the GUI
- **Modular code** 🧱 — split into `model.py`, `pages.py`, `ocr.py`, `inspector.py`, `windows_ocr.py`, `chrome_ocr_engine.py`, `normalize.py`, `transcriber.py` (one place that wires up every engine, shared by the CLI and the GUI), `pdf_batch.py`, `pdf_pipeline.py`, `settings_store.py`, `gui.py` around the `main.py` entry point
- **Model check before download** 📥 — shows cache status and repo size, asks before downloading

## Requirements 🛠️

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

> **Note:** exporting `epub`/`pdf`/`azw3` requires [calibre](https://calibre-ebook.com/) (`ebook-convert` on PATH).

## Usage 🚀

### GUI (default) 🖥️

```bash
python main.py
```

The GUI remembers the last form — input, output name, formats, engine, device,
DPI, workers, direction, normalization, and the folder options — and restores it
on the next launch. It is saved when a run starts and when the window closes, to
`%APPDATA%\OCR-Wrapper\settings.json` (`~/Library/Application Support/` on macOS,
`$XDG_CONFIG_HOME` on Linux); set `OCR_WRAPPER_CONFIG_DIR` to keep it elsewhere.
A missing or damaged file just means the defaults. A remembered batch selection
is re-checked on restore, so PDFs you have since deleted are dropped from it.

### CLI — PDF 📄

```bash
python main.py --pdf book.pdf --output_file transcript
```

Keep the outputs next to the source PDF instead of the working directory (`--output_file` then supplies only the name):

```bash
python main.py --pdf books/paper.pdf --output_file transcript --same_dir
```

Fast text extraction of a text-based PDF (no OCR, no model download):

```bash
python main.py --pdf book.pdf --engine inspector --output_file transcript
```

Process several selected PDFs with the same settings:

```bash
python main.py --pdfs paper-one.pdf paper-two.pdf --engine inspector --direction ltr --output_dir transcripts
```

Batch mode writes one short-named folder per PDF, containing the transcript in
each requested format:

```
transcripts/paper/transcript.md
transcripts/another-book/transcript.md
```

Long PDF names are shortened with a stable hash so the generated files remain
openable by Windows applications that enforce the legacy path-length limit.

Windows Snipping Tool OCR (high accuracy, fully offline — needs model files, see [oneocr setup](#oneocr-setup-windows-snipping-tool-ocr)):

```bash
python main.py --pdf book.pdf --engine oneocr --output_file transcript
```

With Persian normalization (reinserts half-spaces/ZWNJ that OCR models often drop — recommended for Persian text):

```bash
python main.py --pdf book.pdf --engine oneocr --normalize --output_file transcript
```

Chrome/Edge Screen AI OCR (offline, layout-aware — needs setup, see Notes):

```bash
python main.py --pdf book.pdf --engine chrome --output_file transcript
```

### CLI — Image folder 🖼️

```bash
python main.py --input_dir ./pages --output_file transcript
```

### Exporting to multiple formats 📦

`--output_file` is a base name; an extension is added per selected format. `epub`/`pdf`/`azw3` need calibre and build on each other (md → epub → pdf/azw3):

```bash
python main.py --pdf book.pdf --formats md txt epub azw3 --output_file transcript
```

Spread the page load across workers (2-8; `bina` stays single-device, `chrome` uses processes):

```bash
python main.py --pdf book.pdf --engine chrome --workers 4 --output_file transcript
```

Re-export an existing markdown transcript without re-running OCR:

```bash
python main.py --skip-ocr --output_file transcript --formats epub azw3
```

### Options ⚙️

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | — | Path to PDF file |
| `--pdfs` | — | Paths to one or more selected PDF files |
| `--input_dir` | — | Folder of page images |
| `--output_file` | `book_transcript` | Output base name (extension added per format) |
| `--same_dir` | off | Write outputs next to the input (beside the PDF, or inside the image folder) |
| `--output_dir` | `transcripts` | Root output folder for `--pdfs` (one subfolder per PDF) |
| `--formats` | `md` | Output formats: `md` `txt` `epub` `pdf` `azw3` (ebook formats need calibre) |
| `--direction` | `rtl` | Text direction of the exported output (`rtl`/`ltr`) |
| `--max_new_tokens` | `1024` | Max tokens generated per page (bina) |
| `--limit` | all | Process only first N pages |
| `--engine` | `bina` | `bina` (vision OCR), `inspector` (pdf-inspector, PDF only), `oneocr` (Windows OCR) or `chrome` (Chrome Screen AI) |
| `--workers` | `1` | Parallel page workers (2-8; bina stays 1, chrome uses processes) |
| `--normalize` | off | Normalize Persian text with hazm (reinserts half-spaces/ZWNJ) |
| `--skip-ocr` | off | Skip OCR and re-export the existing `.md` to the selected formats |
| `--cpu` | off | Force CPU even if GPU is available |
| `--gui` | — | Launch GUI explicitly |

## Output 📝

- **Markdown** 📄 — a clean continuous document with paragraphs merged across page boundaries (no `## Page N` markers), wrapped in `<div dir="rtl">` for renderers
- **Formats** 🗂️ — `md`/`txt` are written directly; `epub`/`pdf`/`azw3` are produced via calibre (`ebook-convert`)
- **No sidecar files** 🧹 — the output folder holds only the transcript in each requested format; `--skip-ocr` still honours a `.pagemap.json` left by an older run (so those page breaks survive the re-export) and then deletes it
- **Next to the input** 📁 — `--same_dir` (GUI: “Save next to input”) writes the transcript beside the source PDF, or inside the image folder, instead of the working directory; with “Save in folder” the transcript folder is created there
- **Kindle** 📖 — `epub`/`azw3` output is pre-shaped Arabic-script Persian for e-ink; the `.md`/`.txt` stay canonical and searchable

### Sample result (page 1 of «۱» , RTL Persian) 🎯

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

## oneocr setup (Windows Snipping Tool OCR) ✂️

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

## Notes 📌

- Model is cached locally after first download (~1.3GB); the tool checks the cache and asks before downloading
- pdf-inspector is instant (<1s) but only handles text-based PDFs — scanned pages need the `bina` engine
- oneocr does not emit U+200C (ZWNJ) — the `--normalize` flag fixes half-spaces (`می‌رود`) and unifies digits/glyphs via [hazm](https://github.com/sobhe/hazm); works with any engine. In the GUI, normalization is on by default
- `chrome` engine needs [chrome-ocr](https://github.com/ayismas/chrome-ocr) installed from source (`git clone https://github.com/ayismas/chrome-ocr && cd chrome-ocr && pip install -e ".[pdf]"`), plus the Screen AI DLL: open Chrome → Settings → Accessibility, enable a screen-reader option, and confirm `chrome_screen_ai.dll` lands in `%LOCALAPPDATA%\Google\Chrome\User Data\screen_ai\` (run `chrome-ocr doctor` to verify). Edge's DLL lives at a different path — the wrapper only auto-detects Chrome's, but `ScreenAIEngine(dll_path=...)` accepts a custom path. Windows-only; DLL subject to Google's terms
- 300 DPI is default for PDF rendering; raise for better accuracy, lower for speed
- Expect minutes/page on low-end GPUs; ~10-30s/page on a proper GPU
- `torch.cuda.empty_cache()` runs every 10 pages for low-VRAM GPUs
- Stop button (GUI) / Ctrl-C (CLI) stops after the current page
- The GUI's saved settings live outside the project (`%APPDATA%\OCR-Wrapper\settings.json` on Windows), so they survive a re-clone; delete that file to get the defaults back
- The GUI is PySide6 (Qt), installed by `requirements.txt`. On a headless Linux box it also needs the usual Qt X11 runtime libs (`libxcb-*`); without a display, use the CLI instead

## License 📄

[MIT](LICENSE)
