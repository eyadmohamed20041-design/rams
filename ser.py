import os
import io
import json
import time
import logging
import re
import hashlib
import redis

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pydub import AudioSegment
from openai import OpenAI


# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO)

# ====================== KEYS ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_SECRET = os.getenv("API_SECRET", "SECRET123")

client = OpenAI(api_key=OPENAI_API_KEY)

# ====================== REDIS ======================
r = redis.Redis(
    host=os.getenv("REDISHOST"),
    port=os.getenv("REDISPORT"),
    password=os.getenv("REDISPASSWORD"),
    decode_responses=True
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

# ====================== ROOT ======================
@app.get("/")
def root():
    return {"status": "running"}

# ====================== STORAGE ======================
AUDIO_DIR = "./data/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ====================== NORMALIZE ======================
def normalize(text: str):
    text = text.lower().strip()
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ====================== HASH ======================
def make_cache_key(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()

# ====================== CANONICAL QUESTION ======================
def canonicalize_question(text):

    prompt = f"""
حوّل السؤال التالي إلى صيغة عربية فصحى موحدة قصيرة.
احذف الحشو.
لا تضف أي شرح.
ارجع السؤال فقط.

السؤال:
{text}
"""

    res = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=60
    )

    out = ""

    for item in getattr(res, "output", []):
        for content in getattr(item, "content", []):
            if content.type == "output_text":
                out += content.text

    return out.strip() or text

# ====================== CLEAN TTS ======================
def clean_for_tts(text):
    return normalize(text).replace("ـ", "")

# ====================== RATE LIMIT ======================
last_request_time = 0
MIN_INTERVAL = 2

# ====================== ASK ======================
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

        # ========== CANONICAL ==========
        canonical = canonicalize_question(raw_text)

        key = make_cache_key(canonical)

        cached = r.get(key)

        if cached:
            c = json.loads(cached)
            return {
                "text": c["text"],
                "audio_url": f"/audio/{os.path.basename(c['audio_file'])}",
                "cached": True
            }

        # ========== SYSTEM PROMPT ==========
        system_prompt = """
أنت الملك رمسيس الثاني، أحد أعظم ملوك مصر القديمة.

الشخصية:
- هادئ وواثق.
- حكيم وفصيح.
- تتحدث كملك تاريخي حقيقي.

القواعد الصارمة:
- الرد بالعربية الفصحى فقط.
- لا تذكر الذكاء الاصطناعي.
- لا تشير إلى العصر الحديث.
- لا تخرج عن شخصيتك إطلاقًا.

إذا سُئلت عن شيء لم يكن موجودًا في عصرك:
أجب مثل:
"في عصرنا لم يكن هذا معروفًا، ولكن يمكنني أن أحدثك عما نعرفه من حكمة القدماء."

استخدم:
- أمثلة تاريخية من مصر القديمة.
- أسلوب ملكي راقٍ.
- لغة واضحة مناسبة للنطق الصوتي.
"""

        # ========== GPT ANSWER ==========
        res = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": canonical}
            ],
            max_output_tokens=1000
        )

        reply = ""

        for item in getattr(res, "output", []):
            for content in getattr(item, "content", []):
                if content.type == "output_text":
                    reply += content.text

        reply = clean_for_tts(reply.strip() or "لم أفهم سؤالك")

        # ========== TTS ==========
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply
        )

        audio_full = speech.read()

        audio = AudioSegment.from_file(io.BytesIO(audio_full))
        audio = audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)

        filename = f"reply_{int(time.time())}.wav"
        path = os.path.join(AUDIO_DIR, filename)

        audio.export(path, format="wav")

        # ========== SAVE REDIS ==========
        r.setex(key, 86400, json.dumps({
            "text": reply,
            "audio_file": path
        }))

        return {
            "text": reply,
            "audio_url": f"/audio/{filename}",
            "cached": False
        }

    except Exception as e:
        logging.error("ERROR", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ====================== AUDIO ======================
@app.get("/audio/{file}")
async def serve_audio(file: str):

    path = os.path.join(AUDIO_DIR, file)

    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Not found"})

    return FileResponse(path, media_type="audio/wav")

# ====================== LANGUAGE ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):
    return {"status": "ok"}

