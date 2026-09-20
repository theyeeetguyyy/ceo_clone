"""
Voice API — Gemini STT transcription endpoint.
POST /api/voice/transcribe — multipart audio → transcript text

Migrated off Groq Whisper; the app now talks to Gemini only.
"""

from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from backend.utils.gemini_client import get_pool
from backend.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/voice", tags=["voice"])
limiter = Limiter(key_func=get_remote_address)

# Extension → MIME type Gemini accepts for inline audio parts.
SUPPORTED_FORMATS = {
    ".mp3":  "audio/mp3",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".m4a":  "audio/mp4",
    ".mp4":  "audio/mp4",
    ".wav":  "audio/wav",
    ".webm": "audio/webm",
    ".ogg":  "audio/ogg",
    ".aac":  "audio/aac",
    ".flac": "audio/flac",
}
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/transcribe")
@limiter.limit("5/minute")
async def transcribe_audio(request: Request, audio: UploadFile = File(...)):
    """
    Transcribe audio using the Gemini speech-to-text model.
    Returns the transcript text.
    Accepts: wav, mp3, webm, ogg, m4a, etc.
    Rate limited: 5 requests/min per IP. Max file size: 10 MB.
    """
    suffix = Path(audio.filename or "audio.webm").suffix.lower()
    mime_type = SUPPORTED_FORMATS.get(suffix)
    if mime_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(audio_bytes)/1024/1024:.1f}MB). Max: 10MB."
        )
    log.info(f"Transcribing audio | format={suffix} | size={len(audio_bytes)/1024:.1f}KB")

    pool = get_pool()

    try:
        transcript = await pool.transcribe(audio_bytes, mime_type)
        log.info(f"Transcription: '{transcript[:100]}'")
        return {"transcript": transcript, "format": suffix}

    except Exception as e:
        log.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
