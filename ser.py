import os
import re
import time
import logging
import numpy as np

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

# ====================== SEMANTIC CACHE ======================
semantic_cache = []

# ====================== HELPERS ======================
def normalize(text: str):
    return re.sub(r"\s+", " ", text.lower().strip())

def is_greeting(text: str):
    greetings = ["hello", "hi", "ازيك", "عامل اي", "hallo", "你好"]
    return any(g in text.lower() for g in greetings)

# ====================== EMBEDDINGS ======================
def get_embedding(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_semantic_cache(query_vec):
    best_score = 0
    best_audio = None

    for item in semantic_cache:
        score = cosine_similarity(query_vec, item["vector"])

        if score > best_score:
            best_score = score
            best_audio = item["audio"]

    if best_score >= 0.85:
        return best_audio

    return None

# ====================== ASK ======================
@app.post("/ask")
async def ask(
    request: Request,
    text: str = Form(...),
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

        # ================= EMBEDDING =================
        query_vec = get_embedding(text)

        # ================= SEMANTIC CACHE CHECK =================
        cached_audio = find_semantic_cache(query_vec)

        if cached_audio is not None:
            logging.info("🔥 SEMANTIC CACHE HIT")
            return Response(content=cached_audio, media_type="audio/mpeg")

        # ================= SYSTEM PROMPT =================
        system_prompt = f"""
أنت الملك رمسيس الثاني، ملك عظيم وحكيم من مصر القديمة.

أسلوب:
- هادئ وواثق.
- حكيم وصادق.
- لا تتحدث عن كونك ذكاء اصطناعي.

القواعد:
- الرد بلغة {LANGUAGE_NAMES.get(current_language, 'العربية')}.
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

        # ================= SAVE TO SEMANTIC CACHE =================
        semantic_cache.append({
            "text": text,
            "vector": query_vec,
            "audio": audio_bytes
        })

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

# =========================================================
# 🔊 TTS ONLY ENDPOINT
# =========================================================
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
        "mode": "voice_ai_semantic_ramses"
    }
