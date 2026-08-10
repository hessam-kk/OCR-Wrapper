"""Windows OCR engine via oneocr (Snipping Tool OCR model)."""

from pathlib import Path


def get_ocr_engine():
    """Return a oneocr.OcrEngine, or raise a friendly error if not usable."""
    try:
        import oneocr
    except ImportError:
        raise RuntimeError(
            "oneocr is not installed. Run: pip install oneocr"
        ) from None
    try:
        return oneocr.OcrEngine()
    except RuntimeError as e:
        if "DLL initialization failed" not in str(e):
            raise
        raise RuntimeError(
            "oneocr engine files not found. Put oneocr.dll, oneocr.onemodel and "
            "onnxruntime.dll in ~/.config/oneocr/ (extract them from the Snipping "
            "Tool msixbundle)"
        ) from None


def oneocr_transcribe_page(engine, image_path: Path) -> str:
    from PIL import Image

    result = engine.recognize_pil(Image.open(image_path).convert("RGB"))
    return (result.get("text") or "").strip()
