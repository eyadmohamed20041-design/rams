import os
import io
import json
import time
import logging
import string
import re
import math
import hashlib

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pydub import AudioSegment
from openai import OpenAI


# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)


# ======================
# API KEYS
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_SECRET = os.getenv("API_SECRET", "SECRET123")

client = OpenAI(api_key=OPENAI_API_KEY)


# ======================
# SERVER SETUP
# ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================
# ROOT
# ======================
@app.get("/")
def root():
    return {"status": "running"}


# ======================
# LANGUAGE
# ======================
current_language = "ar"

LANGUAGE_NAMES = {
    "ar": "العربية",
    "en": "English",
    "de": "Deutsch",
    "zh": "中文"
}


# ======================
# STORAGE
# ======================
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "cache.json")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

os.makedirs(AUDIO_DIR, exist_ok=True)


# ======================
# LOAD CACHE
# ======================
cache = {}

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)


def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ======================
# TEXT NORMALIZE
# ======================
def normalize(text: str):

    text = text.lower().strip()

    # إزالة التشكيل العربي
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)

    # إزالة الرموز
    text = re.sub(r'[^\w\s]', '', text)

    # مسافات
    text = re.sub(r'\s+', ' ', text)

    return text


# ======================
# HASH KEY
# ======================
def make_cache_key(text):

    norm = normalize(text)

    return hashlib.sha256(
        norm.encode("utf-8")
    ).hexdigest()


# ======================
# EMBEDDING
# ======================
def get_embedding(text):

    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return res.data[0].embedding


# ======================
# COSINE SIMILARITY
# ======================
def cosine_similarity(a, b):

    dot = sum(x*y for x, y in zip(a, b))

    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))

    return dot / (na * nb)


# ======================
# SEMANTIC CACHE
# ======================
def semantic_cache_lookup(new_emb, threshold=0.88):

    best = None
    best_score = 0

    for item in cache.values():

        old_emb = item.get("embedding")

        if not old_emb:
            continue

        score = cosine_similarity(new_emb, old_emb)

        if score > best_score:
            best_score = score
            best = item

    if best_score >= threshold:
        return best

    return None


# ======================
# FIX TEXT
# ======================
def smart_correct_text(text: str):

    prompt = f"""
النص التالي ناتج من تحويل صوت إلى نص.

المطلوب:
- تصحيح الأخطاء.
- إعادة الصياغة بالعربية الفصحى.
- بدون تشكيل.
- بدون رموز.
- مناسب للنطق.

النص:
{text}
"""

    res = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=150
    )

    fixed = ""

    for item in getattr(res, "output", []):
        for content in getattr(item, "content", []):
            if content.type == "output_text":
                fixed += content.text

    return fixed.strip() or text


# ======================
# CLEAN TTS
# ======================
def clean_for_tts(text):

    text = normalize(text)

    text = text.replace("ـ", "")

    return text


# ======================
# RESPONSE TYPE
# ======================
def determine_response_type(text):

    greetings = [
        "ازيك", "عامل اي", "اخبارك", "hello", "hi"
    ]

    for g in greetings:
        if g in text.lower():
            return "short"

    return "normal"


# ======================
# RATE LIMIT
# ======================
last_request_time = 0
MIN_INTERVAL = 2


# ======================
# MAIN ENDPOINT
# ======================
@app.post("/ask")
async def ask(request: Request, file: UploadFile = File(...)):

    global last_request_time


    # -------- AUTH --------
    if request.headers.get("x-api-key") != API_SECRET:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})


    # -------- RATE LIMIT --------
    now = time.time()

    if now - last_request_time < MIN_INTERVAL:
        return JSONResponse(status_code=429, content={"error": "Too many requests"})

    last_request_time = now


    try:

        # -------- READ AUDIO --------
        audio_bytes = await file.read()

        if len(audio_bytes) < 2000:
            return JSONResponse(status_code=400, content={"error": "Audio too small"})


        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "speech.wav"


        # -------- TRANSCRIBE --------
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

        raw_text = transcript.strip()

        if not raw_text:
            return JSONResponse(status_code=400, content={"error": "No speech"})


        # -------- FIX TEXT --------
        fixed_text = smart_correct_text(raw_text)


        logging.info(f"RAW: {raw_text}")
        logging.info(f"FIXED: {fixed_text}")


        # -------- HASH --------
        key = make_cache_key(fixed_text)


        # -------- FAST CACHE --------
        if key in cache:

            c = cache[key]

            return {
                "text": c["text"],
                "audio_url": f"/audio/{os.path.basename(c['audio_file'])}",
                "cached": True,
                "type": "hash"
            }


        # -------- EMBEDDING --------
        embedding = get_embedding(fixed_text)


        # -------- SEMANTIC CACHE --------
        semantic = semantic_cache_lookup(embedding)

        if semantic:

            return {
                "text": semantic["text"],
                "audio_url": f"/audio/{os.path.basename(semantic['audio_file'])}",
                "cached": True,
                "type": "semantic"
            }


        # -------- RESPONSE TYPE --------
        rtype = determine_response_type(fixed_text)


        # -------- SYSTEM PROMPT --------
        system_prompt = f"""
أنت الملك رمسيس الثاني.

أسلوب:
- هادئ.
- واثق.
- حكيم.

قواعد:
- الرد بالعربية.
- بدون ذكر العصر الحديث.
- لا تخرج عن الشخصية.
"""

        if rtype == "short":
            system_prompt += "\nالرد قصير."
        else:
            system_prompt += "\nالرد مفصل."


        # -------- GPT --------
        res = client.responses.create(

            model="gpt-4o-mini",

            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fixed_text}
            ],

            max_output_tokens=1200
        )


        reply = ""

        for item in getattr(res, "output", []):
            for content in getattr(item, "content", []):
                if content.type == "output_text":
                    reply += content.text


        reply = reply.strip() or "لم أفهم سؤالك."


        # -------- CLEAN --------
        reply = clean_for_tts(reply)


        # -------- TTS --------
        speech = client.audio.speech.create(

            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply
        )


        audio_full = speech.read()


        audio = AudioSegment.from_file(io.BytesIO(audio_full))

        audio = audio.set_frame_rate(44100)\
                     .set_sample_width(2)\
                     .set_channels(1)


        # -------- SAVE AUDIO --------
        filename = f"reply_{len(cache)+1}.wav"

        path = os.path.join(AUDIO_DIR, filename)

        audio.export(path, format="wav")


        # -------- SAVE CACHE --------
        cache[key] = {
            "original": fixed_text,
            "embedding": embedding,
            "text": reply,
            "audio_file": path
        }

        save_cache()


        return {
            "text": reply,
            "audio_url": f"/audio/{filename}",
            "cached": False
        }


    except Exception as e:

        logging.error("ERROR", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ======================
# AUDIO SERVE
# ======================
@app.get("/audio/{file}")
async def serve_audio(file: str):

    path = os.path.join(AUDIO_DIR, file)

    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Not found"})

    return FileResponse(path, media_type="audio/wav")


# ======================
# LANGUAGE
# ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):

    global current_language

    current_language = lang.lower()

    return {"status": "ok"}
