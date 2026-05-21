"""
Voice API — Groq Whisper STT transcription endpoint.
POST /api/voice/transcribe — multipart audio → transcript text

BUG FIX: Now uses GroqKeyPool for rate-limit resilience instead of raw client.
"""

from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from backend.utils.groq_rotator import get_pool
from backend.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/voice", tags=["voice"])
limiter = Limiter(key_func=get_remote_address)

SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/transcribe")
@limiter.limit("5/minute")
async def transcribe_audio(req: Request, audio: UploadFile = File(...)):
    """
    Transcribe audio using Groq Whisper API.
    Returns the transcript text.
    Accepts: wav, mp3, webm, ogg, m4a, etc.
    Rate limited: 5 requests/min per IP. Max file size: 10 MB.
    """
    suffix = Path(audio.filename or "audio.webm").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}. Supported: {SUPPORTED_FORMATS}"
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(audio_bytes)/1024/1024:.1f}MB). Max: 10MB."
        )
    log.info(f"Transcribing audio | format={suffix} | size={len(audio_bytes)/1024:.1f}KB")

    # BUG FIX: use GroqKeyPool for rate-limit rotation instead of raw client
    pool = get_pool()
    client = pool.get_client()

    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=(f"audio{suffix}", audio_bytes, f"audio/{suffix.lstrip('.')}"),
            response_format="text",
            language="en",
        )
        transcript = str(transcription).strip()
        log.info(f"Transcription: '{transcript[:100]}'")
        return {"transcript": transcript, "format": suffix}

    except Exception as e:
        log.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
