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

# ====================== RATE LIMIT ======================
user_last_request = {}
MIN_INTERVAL = 0.5

# ====================== LANG MAP ======================
LANGUAGE_NAMES = {
    "ar": "العربية",
    "en": "English",
    "de": "Deutsch",
    "zh": "中文"
}

# ====================== HELPERS ======================
def normalize(text: str):
    return re.sub(r"\s+", " ", text.lower().strip())


def detect_language(text: str):
    # Chinese
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    # Arabic
    if re.search(r'[\u0600-\u06FF]', text):
        return "ar"
    # German / English fallback (Latin)
    if re.search(r'[a-zA-Z]', text):
        return "en"
    return "ar"


# ====================== ASK ======================
@app.post("/ask")
async def ask(
    request: Request,
    text: str = Form(...),
    lang: str = Form(None),   # ✅ اللغة من Unity (اختياري)
    rtype: str = Form("long")
):

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

        # ================= LANGUAGE LOGIC =================
        # Unity language OR auto-detect OR Arabic default
        if lang:
            lang = lang.lower().strip()
        else:
            lang = detect_language(text) or "ar"

        if lang not in LANGUAGE_NAMES:
            lang = "ar"

        logging.info(f"LANG: {lang}")

        # ================= SYSTEM PROMPT =================
        system_prompt = f"""
أنت الملك رمسيس الثاني، ملك عظيم وحكيم من مصر القديمة.

مهم جداً:
- يجب الرد بالكامل باللغة: {LANGUAGE_NAMES.get(lang, "العربية")}
- إذا لم يتم تحديد لغة، الرد يكون بالعربية تلقائيًا
- ممنوع خلط اللغات نهائيًا

أسلوبك:
- حكيم
- هادئ
- واثق
- لا تذكر أنك ذكاء اصطناعي أبداً
"""

        if rtype == "short":
            system_prompt += "\nالرد قصير."
        else:
            system_prompt += "\nالرد مفصل."

        # ================= GPT =================
        gpt_response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_output_tokens=400
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

        return Response(
            content=speech.read(),
            media_type="audio/mpeg"
        )

    except Exception as e:
        logging.error("SERVER ERROR", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ====================== TTS ONLY ======================
@app.post("/tts")
async def tts(
    request: Request,
    text: str = Form(...)
):

    try:
        if request.headers.get("x-api-key") != API_SECRET:
            return JSONResponse(status_code=403, content={"error": "Forbidden"})

        text = text.strip()

        if not text:
            return JSONResponse(status_code=400, content={"error": "Empty text"})

        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )

        return Response(
            content=speech.read(),
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
        "mode": "voice_ai_fixed_multilang"
    }
