"""
stt.py — local speech-to-text for ChatGB10, running on the GB10 (no cloud).

Keeps transcription fully on-box so the on-prem story holds. Prefers
faster-whisper (CTranslate2, GPU, decodes browser webm/opus via bundled PyAV)
and falls back to openai-whisper (PyTorch; needs the `ffmpeg` binary present).
The model is loaded once on first use and reused for the life of the process.

Everything is configurable via environment variables, all optional:

    CHATGB10_STT_BACKEND   auto | faster | openai      (default: auto)
    CHATGB10_STT_MODEL     whisper size: tiny|base|small|medium|large-v3
                                                         (default: base)
    CHATGB10_STT_DEVICE    cuda | cpu | auto            (default: auto)
    CHATGB10_STT_LANGUAGE  e.g. en  (blank = autodetect; default: blank)

Install ONE backend on the GB10 (both are optional — without either, the
microphone simply returns a clear 501 and the rest of ChatGB10 is unaffected):

    pip install faster-whisper --break-system-packages          # recommended
  or
    pip install openai-whisper --break-system-packages
    sudo apt-get install -y ffmpeg                              # for openai-whisper
"""

import os
import tempfile

BACKEND_PREF = os.environ.get("CHATGB10_STT_BACKEND", "auto").lower()
MODEL_SIZE = os.environ.get("CHATGB10_STT_MODEL", "base")
DEVICE_PREF = os.environ.get("CHATGB10_STT_DEVICE", "auto").lower()
LANGUAGE = os.environ.get("CHATGB10_STT_LANGUAGE", "").strip() or None

_backend = None      # "faster" | "openai" once loaded
_model = None        # loaded model handle
_load_err = ""       # human-readable reason load failed, if any


def _pick_device() -> str:
    if DEVICE_PREF in ("cuda", "cpu"):
        return DEVICE_PREF
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load() -> None:
    """Lazily load a Whisper backend. Safe to call repeatedly; never raises."""
    global _backend, _model, _load_err
    if _model is not None or _load_err:
        return
    device = _pick_device()

    # 1) faster-whisper (preferred) — unless the user pinned openai
    if BACKEND_PREF in ("auto", "faster"):
        try:
            from faster_whisper import WhisperModel
            compute = "float16" if device == "cuda" else "int8"
            _model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute)
            _backend = "faster"
            return
        except Exception as e:  # noqa: BLE001
            if BACKEND_PREF == "faster":
                _load_err = f"faster-whisper unavailable: {e}"
                return
            # else: fall through to openai-whisper

    # 2) openai-whisper fallback
    if BACKEND_PREF in ("auto", "openai"):
        try:
            import whisper
            _model = whisper.load_model(MODEL_SIZE, device=device)
            _backend = "openai"
            return
        except Exception as e:  # noqa: BLE001
            _load_err = ("no whisper backend available — install faster-whisper "
                         f"or openai-whisper ({e})")
            return

    _load_err = f"unknown STT backend preference: {BACKEND_PREF}"


def available() -> bool:
    _load()
    return _model is not None


def status() -> dict:
    _load()
    return {
        "available": _model is not None,
        "backend": _backend,
        "model": MODEL_SIZE,
        "device": _pick_device(),
        "language": LANGUAGE or "auto",
        "error": _load_err,
    }


def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe a recorded audio clip to text. Blocking — call from a
    threadpool so the async event loop is not stalled."""
    _load()
    if _model is None:
        raise RuntimeError(_load_err or "STT not available")

    suffix = os.path.splitext(filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(audio_bytes)
        path = tf.name
    try:
        if _backend == "faster":
            segments, _info = _model.transcribe(path, language=LANGUAGE, vad_filter=True)
            return "".join(seg.text for seg in segments).strip()
        # openai-whisper
        kw = {"fp16": _pick_device() == "cuda"}
        if LANGUAGE:
            kw["language"] = LANGUAGE
        result = _model.transcribe(path, **kw)
        return (result.get("text") or "").strip()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
