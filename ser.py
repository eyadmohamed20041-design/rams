import os
import logging

from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from openai import OpenAI

# ====================== LOGGING ======================112
logging.basicConfig(level=logging.INFO)

# ====================== API ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_SECRET = os.getenv("API_SECRET", "SECRET123")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not found")

client = OpenAI(api_key=OPENAI_API_KEY)

# ====================== APP ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 🔊 FIXED + THINKING TTS ONLY
# =========================================================
# السيرفر ده وظيفته الوحيدة:
# ياخد النص ويحوله صوت بدون GPT
# استخدمه مرة واحدة فقط عشان يعمل cache
# =========================================================

@app.post("/tts")
async def tts(
    request: Request,
    text: str = Form(...)
):

    try:

        # ================= AUTH =================
        if request.headers.get("x-api-key") != API_SECRET:
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden"}
            )

        text = text.strip()

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty text"}
            )

        logging.info(f"TTS: {text}")

        # ================= TTS =================
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )

        audio_bytes = speech.read()

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg"
        )

    except Exception as e:

        logging.error("TTS ERROR", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# ====================== HEALTH ======================
@app.get("/")
async def health():
    return {
        "status": "running",
        "mode": "tts_only"
    }
