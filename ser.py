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
from fastapi.responses import FileResponse, JSONResponse

from pydub import AudioSegment
from openai import OpenAI
import redis

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO)

# ====================== API KEYS ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_SECRET = os.getenv("API_SECRET", "SECRET123")
client = OpenAI(api_key=OPENAI_API_KEY)

# ====================== REDIS SETUP ======================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(REDIS_URL, decode_responses=True)

# ====================== SERVER SETUP ======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== ROOT ======================
@app.get("/")
def root():
    return {"status": "running"}

# ====================== LANGUAGE ======================
current_language = "ar"
LANGUAGE_NAMES = {"ar": "العربية", "en": "English", "de": "Deutsch", "zh": "中文"}

# ====================== STORAGE ======================
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ====================== TEXT NORMALIZE ======================
def normalize(text: str):
    text = text.lower().strip()
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ====================== HASH KEY ======================
def make_cache_key(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()

# ====================== EMBEDDING ======================
def get_embedding(text):
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding

# ====================== COSINE SIMILARITY ======================
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb)

# ====================== SEMANTIC CACHE ======================
def semantic_cache_lookup(new_emb, threshold=0.80):
    best = None
    best_score = 0
    for key in r.keys():
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

# ====================== FIX TEXT ======================
def smart_correct_text(text: str):
    if current_language == "ar":
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
    else:
        return text

# ====================== CLEAN TTS ======================
def clean_for_tts(text):
    text = normalize(text)
    text = text.replace("ـ", "")
    return text

# ====================== RESPONSE TYPE ======================
def determine_response_type(text):
    greetings = ["ازيك", "عامل اي", "اخبارك", "hello", "hi", "hallo", "hi there", "guten tag", "你好", "您好", "嗨", "guten morgen", "guten nacht"]

    for g in greetings:
        if g in text.lower():
            return "short"
    return "normal"

# ====================== RATE LIMIT ======================
last_request_time = 0
MIN_INTERVAL = 2

# ====================== MAIN ENDPOINT ======================
@app.post("/ask")
async def ask(request: Request, file: UploadFile = File(...)):
    global last_request_time

    if request.headers.get("x-api-key") != API_SECRET:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    now = time.time()
    if now - last_request_time < MIN_INTERVAL:
        return JSONResponse(status_code=429, content={"error": "Too many requests"})
    last_request_time = now

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) < 2000:
            return JSONResponse(status_code=400, content={"error": "Audio too small"})

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "speech.wav"

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        raw_text = transcript.strip()
        if not raw_text:
            return JSONResponse(status_code=400, content={"error": "No speech"})

        raw_key = make_cache_key(raw_text)

        # -------- FAST CACHE (REDIS) --------
        cached_data = r.get(raw_key)
        if cached_data:
            c = json.loads(cached_data)
            return {
                "text": c["text"],
                "audio_url": f"/audio/{os.path.basename(c['audio_file'])}",
                "cached": True,
                "type": "hash"
            }

        # -------- FIX TEXT --------
        fixed_text = smart_correct_text(raw_text)
        logging.info(f"RAW: {raw_text}")
        logging.info(f"FIXED: {fixed_text}")

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
أنت الملك رمسيس الثاني، ملك عظيم وحكيم من مصر القديمة.

أسلوب:
- هادئ وواثق.
- حكيم وصادق.
- لا تتحدث عن كونك ذكاء اصطناعي أو أي شيء من العصر الحديث.
- لا تخرج عن شخصيتك التاريخية.

القواعد:
- الرد بلغة {LANGUAGE_NAMES.get(current_language, 'العربية')}.
- إذا سُئلت عن أشياء لم تكن موجودة في عصر مصر القديمة، أجب بطريقة مناسبة مثل: "في عصرنا لم يكن هذا موجودًا، ولكن…".
- استخدم أمثلة وتفاصيل تاريخية دقيقة من عهد مصر القديمة.
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
        reply = clean_for_tts(reply)

        # -------- TTS (CACHE USING raw_key) --------
        filename = f"{raw_key}.wav"
        path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(path):
            speech = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=reply
            )
            audio_full = speech.read()
            audio = AudioSegment.from_file(io.BytesIO(audio_full))
            audio = audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)
            audio.export(path, format="wav")

        # -------- SAVE CACHE TO REDIS --------
        cache_item = {
            "original": fixed_text,
            "embedding": embedding,
            "text": reply,
            "audio_file": path
        }
        r.set(raw_key, json.dumps(cache_item))

        return {
            "text": reply,
            "audio_url": f"/audio/{filename}",
            "cached": False
        }

    except Exception as e:
        logging.error("ERROR", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ====================== AUDIO SERVE ======================
@app.get("/audio/{file}")
async def serve_audio(file: str):
    path = os.path.join(AUDIO_DIR, file)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return FileResponse(path, media_type="audio/wav")

# ====================== LANGUAGE ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):
    global current_language
    current_language = lang.lower()
    return {"status": "ok"}
