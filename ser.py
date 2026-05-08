import os
import re
import time
import logging

from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from openai import OpenAI

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO)

# ====================== API KEY ======================
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

# ====================== STATE ======================
current_language = "ar"
user_last_request = {}
MIN_INTERVAL = 0.5

LANGUAGE_NAMES = {
    "ar": "العربية",
    "en": "English",
    "de": "Deutsch",
    "zh": "中文"
}

# ====================== HELPERS ======================
def normalize(text: str):
    return re.sub(r"\s+", " ", text.lower().strip())


def is_greeting(text: str):
    greetings = ["hello", "hi", "ازيك", "عامل اي", "hallo", "你好"]
    return any(g in text.lower() for g in greetings)


# ====================== ASK ======================
@app.post("/ask")
async def ask(request: Request, text: str = Form(...)):

    try:
        # ================= AUTH =================
        if request.headers.get("x-api-key") != API_SECRET:
            return JSONResponse(status_code=403, content={"error": "Forbidden"})

        # ================= RATE LIMIT =================
        ip = request.client.host
        now = time.time()

        if now - user_last_request.get(ip, 0) < MIN_INTERVAL:
            return JSONResponse(status_code=429, content={"error": "Too many requests"})

        user_last_request[ip] = now

        # ================= INPUT =================
        text = normalize(text)

        if not text:
            return JSONResponse(status_code=400, content={"error": "Empty text"})

        logging.info(f"USER: {text}")

        # ================= SYSTEM PROMPT =================
        system_prompt = f"""
أنت الملك رمسيس الثاني.

- تحدث بلغة {LANGUAGE_NAMES.get(current_language, "العربية")}
- لا تخرج عن الشخصية
- لا تذكر أنك AI
"""

        if is_greeting(text):
            system_prompt += "\nاجعل الرد قصير جدًا."

        # ================= GPT =================
        gpt_response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_output_tokens=500
        )

        reply = ""

        for item in getattr(gpt_response, "output", []):
            for content in getattr(item, "content", []):
                if content.type == "output_text":
                    reply += content.text

        reply = reply.strip() or "لم أفهم السؤال."

        logging.info(f"RAMSES: {reply}")

        # ================= TTS =================
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply
        )

        audio_bytes = speech.read()

        # ================= IMPORTANT FIX =================
        # ❌ NO HEADERS WITH ARABIC TEXT (THIS WAS YOUR BUG)

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg"
        )

    except Exception as e:
        logging.error("SERVER ERROR", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ====================== LANGUAGE ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):
    global current_language
    current_language = lang.lower()

    return {
        "status": "ok",
        "language": current_language
    }


# ====================== HEALTH ======================
@app.get("/")
async def health():
    return {
        "status": "running",
        "mode": "voice_ai_fixed"
    }
