import io
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


# ====================== API KEYS ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_SECRET = os.getenv("API_SECRET", "SECRET123")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not found")


client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ====================== SERVER ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================== LANGUAGE ======================
current_language = "ar"

LANGUAGE_NAMES = {
    "ar": "العربية",
    "en": "English",
    "de": "Deutsch",
    "zh": "中文"
}


# ====================== RATE LIMIT ======================
user_last_request = {}

MIN_INTERVAL = 0.5


# ====================== NORMALIZE ======================
def normalize(text: str):

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ====================== RESPONSE TYPE ======================
def determine_response_type(text: str):

    greetings = [
        "ازيك",
        "عامل اي",
        "اخبارك",
        "hello",
        "hi",
        "hallo",
        "guten tag",
        "你好",
        "您好"
    ]

    text_lower = text.lower()

    for g in greetings:

        if g in text_lower:
            return "short"

    return "normal"


# ====================== CLEAN ======================
def clean_for_tts(text: str):

    text = text.strip()

    text = text.replace("*", "")
    text = text.replace("#", "")
    text = text.replace("_", "")

    return text


# ====================== ASK ======================
@app.post("/ask")
async def ask(
    request: Request,
    text: str = Form(...)
):

    global user_last_request

    try:

        # ================= AUTH =================
        if request.headers.get("x-api-key") != API_SECRET:

            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden"
                }
            )

        # ================= RATE LIMIT =================
        client_ip = request.client.host

        now = time.time()

        last_time = user_last_request.get(
            client_ip,
            0
        )

        if now - last_time < MIN_INTERVAL:

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests"
                }
            )

        user_last_request[client_ip] = now

        # ================= VALIDATE =================
        text = normalize(text)

        if not text:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Empty text"
                }
            )

        logging.info(f"USER: {text}")

        # ================= RESPONSE TYPE =================
        response_type = determine_response_type(
            text
        )

        # ================= SYSTEM PROMPT =================
        system_prompt = f"""
أنت الملك رمسيس الثاني، ملك مصر العظيم.

الأسلوب:
- حكيم.
- هادئ.
- واثق.
- تتحدث وكأنك تعيش في مصر القديمة.

القواعد:
- تحدث بلغة {LANGUAGE_NAMES.get(current_language, 'العربية')}.
- لا تذكر أنك ذكاء اصطناعي.
- لا تخرج عن شخصيتك التاريخية.
- إذا سئلت عن شيء حديث أجب بطريقة تاريخية مناسبة.
- استخدم معلومات تاريخية دقيقة.
"""

        if response_type == "short":

            system_prompt += """
- اجعل الرد قصير جدا.
"""

        else:

            system_prompt += """
- اجعل الرد مفصل وغني.
"""

        # ================= GPT =================
        gpt_response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_output_tokens=500
        )

        reply = ""

        for item in getattr(
            gpt_response,
            "output",
            []
        ):

            for content in getattr(
                item,
                "content",
                []
            ):

                if content.type == "output_text":

                    reply += content.text

        reply = reply.strip()

        if not reply:

            reply = "لم أفهم سؤالك."

        reply = clean_for_tts(reply)

        logging.info(f"RAMSES: {reply}")

        # ================= TTS MP3 =================
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply
        )

        mp3_bytes = speech.read()

        # ================= RETURN DIRECT MP3 =================
        return Response(
            content=mp3_bytes,
            media_type="audio/mpeg",
            headers={
                "X-AI-Text": reply
            }
        )

    except Exception as e:

        logging.error(
            "SERVER ERROR",
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# ====================== LANGUAGE ======================
@app.post("/set_language")
async def set_language(
    lang: str = Form(...)
):

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
        "mode": "ultra_fast_mp3"
    }
