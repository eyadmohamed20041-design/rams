import io
import os
import json
import time
import logging
import re
import math
import hashlib

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

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
REDIS_URL = os.getenv("REDIS_URL")
r = redis.from_url(REDIS_URL, decode_responses=False)  # important: store bytes

CACHE_EXPIRE = 3600  # 1 hour

# ====================== SERVER SETUP ======================
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
LANGUAGE_NAMES = {"ar": "العربية", "en": "English", "de": "Deutsch", "zh": "中文"}

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
        # تجاهل الصوت
        if isinstance(key, bytes):
            k_str = key.decode("utf-8")
        else:
            k_str = key
        if k_str.startswith("audio:"):
            continue
        item_json = r.get(key)
        if not item_json:
            continue
        item_json_str = item_json.decode("utf-8") if isinstance(item_json, bytes) else item_json
        item = json.loads(item_json_str)
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
            c = json.loads(cached_data.decode("utf-8") if isinstance(cached_data, bytes) else cached_data)
            logging.info("Returning from cache")
            return {
                "text": c["text"],
                "audio_url": f"/audio/{raw_key}",
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
            logging.info("Returning from semantic cache")
            return {
                "text": semantic["text"],
                "audio_url": f"/audio/{semantic['key']}",
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

        # -------- GPT RESPONSE --------
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

        # -------- TTS WAV --------
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply
        )
        audio_full = speech.read()

        # تحويل الصوت لـ WAV باستخدام pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_full))
        audio = audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)

        # تخزين الصوت مباشرة في Redis
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        r.set(f"audio:{raw_key}", buf.read(), ex=CACHE_EXPIRE)

        # -------- SAVE CACHE TO REDIS (JSON) --------
        cache_item = {
            "original": raw_text,
            "embedding": embedding,
            "text": reply,
            "key": raw_key
        }
        r.set(raw_key, json.dumps(cache_item), ex=CACHE_EXPIRE)

        return {
            "text": reply,
            "audio_url": f"/audio/{raw_key}",
            "cached": False
        }

    except Exception as e:
        logging.error("ERROR", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ====================== AUDIO SERVE ======================
@app.get("/audio/{key}")
async def serve_audio(key: str):
    audio_data = r.get(f"audio:{key}")
    if not audio_data:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return Response(content=audio_data, media_type="audio/wav")

# ====================== LANGUAGE ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):
    global current_language
    current_language = lang.lower()
    return {"status": "ok"}
