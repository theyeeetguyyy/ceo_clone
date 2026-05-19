"""
Voice API — Groq Whisper STT transcription endpoint.
POST /api/voice/transcribe — multipart audio → transcript text

BUG FIX: Now uses GroqKeyPool for rate-limit resilience instead of raw client.
"""

from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.utils.groq_rotator import get_pool
from backend.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/voice", tags=["voice"])

SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}


@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio using Groq Whisper API.
    Returns the transcript text.
    Accepts: wav, mp3, webm, ogg, m4a, etc.
    """
    suffix = Path(audio.filename or "audio.webm").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}. Supported: {SUPPORTED_FORMATS}"
        )

    audio_bytes = await audio.read()
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
