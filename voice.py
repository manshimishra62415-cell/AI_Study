"""Voice assistant helpers: speech-to-text (Gemini), intent detection (NLP), text-to-speech (gTTS)."""
import io
import logging
import os
import re

log = logging.getLogger("study_ai.voice")

INTENTS = [
    ("plan_today", r"\b(plan|schedule|tasks?|to-?do)\b.*\b(today|now)\b|what (should|do) i (study|do)"),
    ("progress", r"\b(progress|scores?|weak|performance|how am i doing)\b"),
]


def detect_intent(text):
    """Rule-based intent classifier. Anything unmatched is treated as a study question."""
    t = text.lower()
    for name, pattern in INTENTS:
        if re.search(pattern, t):
            return name
    return "ask"


def transcribe(client, model, audio_bytes, mime="audio/wav"):
    from google.genai import types
    resp = client.models.generate_content(
        model=model,
        contents=["Transcribe the speech exactly. Return only the transcript.",
                  types.Part.from_bytes(data=audio_bytes, mime_type=mime)])
    return (resp.text or "").strip()


def speak(text, max_chars=900):
    """Return MP3 bytes, or None if TTS is unavailable (e.g. offline)."""
    try:
        from gtts import gTTS
        clean = re.sub(r"[*_`#>\[\]]", "", text)[:max_chars]
        buf = io.BytesIO()
        gTTS(text=clean, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        log.exception("TTS failed")
        return None
