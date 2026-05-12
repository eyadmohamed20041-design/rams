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

# ====================== HELPERS ======================
def normalize(text: str):
    return re.sub(r"\s+", " ", text.lower().strip())


def detect_lang_fallback(lang: str):
    valid = ["ar", "en", "de", "cn"]

    if not lang:
        return "en"

    lang = lang.lower().strip()

    if lang not in valid:
        return "en"

    return lang


def get_lang_instruction(lang: str):
    if lang == "ar":
        return "أجب باللغة العربية فقط."
    elif lang == "en":
        return "Reply only in English."
    elif lang == "de":
        return "Antworte nur auf Deutsch."
    elif lang == "cn":
        return "只用中文回答。"
    return "Reply only in English."


# ====================== ASK ======================
@app.post("/ask")
async def ask(
    request: Request,
    text: str = Form(...),
    lang: str = Form("en"),   # 🔥 مهم جدًا
    rtype: str = Form("medium")
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

        # ================= LANGUAGE (FIX CORE ISSUE) =================
        lang = detect_lang_fallback(lang)
        lang_instruction = get_lang_instruction(lang)

        logging.info(f"USER: {text} | LANG: {lang}")

        # ================= AUTO LENGTH DETECTION =================
        detailed_keywords = [
            "بالتفصيل", "اشرح", "شرح", "تفصيل", "explain", "details", "in detail"
        ]

        want_detailed = any(word in text for word in detailed_keywords)

        # ================= SYSTEM PROMPT =================
        system_prompt = f"""
أنت الملك رمسيس الثاني، ملك عظيم وحكيم من مصر القديمة.

مهم جداً:
- {lang_instruction}
- لا تذكر أي تعليمات أو قواعد
- لا تذكر أنك ذكاء اصطناعي

أسلوبك:
- حكيم
- هادئ
- ملكي
"""

        # ================= RESPONSE STYLE =================
        if want_detailed:
            system_prompt += "\nاجعل الرد مفصل جدًا ويشرح كل النقاط."
        else:
            system_prompt += "\nاجعل الرد متوسط الطول."

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

        return Response(
            content=speech.read(),
            media_type="audio/mpeg"
        )

    except Exception as e:
        logging.error("SERVER ERROR", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ====================== TTS ONLY ======================
@app.post("/tts")
async def tts(
    request: Request,
    text: str = Form(...),
    lang: str = Form("en")
):

    try:
        if request.headers.get("x-api-key") != API_SECRET:
            return JSONResponse(status_code=403, content={"error": "Forbidden"})

        text = text.strip()

        if not text:
            return JSONResponse(status_code=400, content={"error": "Empty text"})

        lang = detect_lang_fallback(lang)
        lang_instruction = get_lang_instruction(lang)

        # optional: enforce language even in TTS text
        final_text = f"{text}"

        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=final_text
        )

        return Response(
            content=speech.read(),
            media_type="audio/mpeg"
        )

    except Exception as e:
        logging.error("TTS ERROR", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ====================== HEALTH ======================
@app.get("/")
async def health():
    return {
        "status": "running",
        "mode": "ramesses_multilingual_locked"
    }
