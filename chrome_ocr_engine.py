"""Chrome Screen AI OCR engine via the chrome-ocr wrapper."""

from pathlib import Path


def get_screenai_engine():
    """Return a chrome_ocr.ScreenAIEngine, or raise a friendly error."""
    try:
        from chrome_ocr import ScreenAIEngine
    except ImportError:
        raise RuntimeError(
            "chrome-ocr is not installed. Run: "
            "git clone https://github.com/ayismas/chrome-ocr && cd chrome-ocr && pip install -e \".[pdf]\""
        ) from None
    try:
        return ScreenAIEngine()
    except Exception as e:
        raise RuntimeError(
            "Chrome Screen AI not found. Enable a screen-reader option in "
            "Chrome Settings -> Accessibility so chrome_screen_ai.dll downloads "
            "into %LOCALAPPDATA%\\Google\\Chrome\\User Data\\screen_ai\\"
        ) from e


def chrome_transcribe_page(engine, image_path: Path) -> str:
    from chrome_ocr import ocr_img

    return ocr_img(str(image_path), engine=engine).strip()
