import os
import io
import json
import time
import logging
import re
import math
import hashlib

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydub import AudioSegment
from openai import OpenAI
import redis
import cloudinary
import cloudinary.uploader

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO)

# ====================== API KEYS ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVER_API_SECRET = os.getenv("SERVER_API_SECRET", "SECRET123")

client = OpenAI(api_key=OPENAI_API_KEY)

# ====================== CLOUDINARY ======================
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ====================== REDIS ======================
REDIS_URL = os.getenv("REDIS_URL")
r = redis.from_url(REDIS_URL, decode_responses=True)

CACHE_VERSION = "v2"
CACHE_TTL = 60 * 60 * 24  # 24 hours

# ====================== FASTAPI ======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running"}

# ====================== LANGUAGE ======================
current_language = "ar"
LANGUAGE_NAMES = {"ar": "العربية", "en": "English", "de": "Deutsch", "zh": "中文"}

# ====================== UTIL ======================
def normalize(text: str):
    text = text.lower().strip()
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def make_cache_key(text):
    base = hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
    return f"{CACHE_VERSION}:{base}"

def get_embedding(text):
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb)

def semantic_cache_lookup(new_emb, threshold=0.80):
    best = None
    best_score = 0

    for key in r.scan_iter(match=f"{CACHE_VERSION}:*"):
        item_json = r.get(key)
        if not item_json:
            continue

        item = json.loads(item_json)
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

# ====================== TTS CLEAN ======================
def clean_for_tts(text):
    text = normalize(text)
    text = text.replace("ـ", "")
    return text

# ====================== UPLOAD ======================
def upload_audio_to_cloudinary(audio_bytes, public_id):
    result = cloudinary.uploader.upload(
        io.BytesIO(audio_bytes),
        resource_type="video",
        public_id=public_id,
        overwrite=True
    )
    return result["secure_url"]

# ====================== MAIN ======================
@app.post("/ask")
async def ask(request: Request, file: UploadFile = File(...)):
    try:
        if request.headers.get("x-api-key") != SERVER_API_SECRET:
            return JSONResponse(status_code=403, content={"error": "Forbidden"})

        audio_bytes = await file.read()

        if len(audio_bytes) < 2000:
            return JSONResponse(status_code=400, content={"error": "Audio too small"})

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "speech.wav"

        # -------- TRANSCRIPTION --------
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

        raw_text = transcript.strip()
        if not raw_text:
            return JSONResponse(status_code=400, content={"error": "No speech"})

        raw_key = make_cache_key(raw_text)

        # -------- HASH CACHE --------
        cached_data = r.get(raw_key)
        if cached_data:
            c = json.loads(cached_data)

            audio_url = c.get("audio_url")
            text = c.get("text")

            if audio_url and text:
                logging.info("Returning from HASH cache")
                return {
                    "text": text,
                    "audio_url": audio_url,
                    "cached": True,
                    "type": "hash"
                }
            else:
                logging.warning("Old invalid cache entry deleted")
                r.delete(raw_key)

        # -------- GPT --------
        res = client.responses.create(
            model="gpt-4o-mini",
            input=raw_text,
            max_output_tokens=1000
        )

        reply = ""
        for item in getattr(res, "output", []):
            for content in getattr(item, "content", []):
                if content.type == "output_text":
                    reply += content.text

        reply = reply.strip() or "لم أفهم سؤالك."

        # -------- EMBEDDING --------
        embedding = get_embedding(reply)

        # -------- SEMANTIC CACHE --------
        semantic = semantic_cache_lookup(embedding)
        if semantic:
            audio_url = semantic.get("audio_url")
            text = semantic.get("text")

            if audio_url and text:
                logging.info("Returning from SEMANTIC cache")
                return {
                    "text": text,
                    "audio_url": audio_url,
                    "cached": True,
                    "type": "semantic"
                }

        # -------- TTS --------
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=clean_for_tts(reply)
        )

        audio_full = speech.read()

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_full))
        audio_segment = audio_segment.set_frame_rate(44100).set_sample_width(2).set_channels(1)

        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)

        audio_url = upload_audio_to_cloudinary(buffer.read(), raw_key)

        # -------- SAVE CACHE --------
        cache_item = {
            "text": reply,
            "audio_url": audio_url,
            "embedding": embedding
        }

        r.setex(raw_key, CACHE_TTL, json.dumps(cache_item))

        return {
            "text": reply,
            "audio_url": audio_url,
            "cached": False
        }

    except Exception as e:
        logging.error("ERROR", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ====================== LANGUAGE ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):
    global current_language
    current_language = lang.lower()
    return {"status": "ok"}
