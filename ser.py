import os
import io
import json
import string
import time
import logging
import difflib

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
# FILES & CACHE
# ======================
TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

RESPONSES_FILE = os.path.join(TMP_DIR, "responses.json")
cache = {}

if os.path.exists(RESPONSES_FILE):
    with open(RESPONSES_FILE, encoding="utf-8") as f:
        cache = json.load(f)

def save_cache():
    with open(RESPONSES_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def normalize(text: str):
    text = text.lower().replace(" ", "")
    return text.translate(str.maketrans("", "", string.punctuation))

# ======================
# RATE LIMIT
# ======================
last_request_time = 0
MIN_INTERVAL = 2

# ======================
# SMART TEXT FIXER
# ======================
def smart_correct_text(text: str):
    prompt = f"""
النص التالي ناتج من تحويل صوت إلى نص وقد يحتوي على أخطاء نطق أو كتابة أو لهجة مصرية.

مهمتك:
- فهم المقصود الحقيقي للجملة حتى لو كانت عامية.
- تصحيح أي أخطاء نطق أو كتابة.
- إعادة صياغة الجملة بلغة عربية واضحة.
- لا تضف أي كلمات جديدة.

النص:
{text}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=150
    )

    fixed = ""
    for item in getattr(response, "output", []):
        contents = getattr(item, "content", None)
        if contents:
            for content in contents:
                if getattr(content, "type", "") == "output_text":
                    fixed += getattr(content, "text", "")

    return fixed.strip() if fixed else text

# ======================
# SMART CACHE MATCHING
# ======================
def smart_cache_lookup(text: str, threshold=0.85):
    norm = normalize(text)
    best_match = None
    best_score = 0

    for k in cache.keys():
        score = difflib.SequenceMatcher(None, norm, k).ratio()
        if score > best_score:
            best_score = score
            best_match = k

    if best_score >= threshold:
        return cache.get(best_match)

    return None

# ======================
# HELPER
# ======================
def determine_response_type(user_text: str):
    greetings_and_personal = [
        "ازيك", "إزيك", "كيف حالك", "مرحبا", "hello", "hi",
        "زوجتك", "اولادك", "أطفالك", "ابنك", "بنتك"
    ]
    for phrase in greetings_and_personal:
        if phrase in user_text.lower():
            return "short"
    return "normal"

# ======================
# MAIN ENDPOINT
# ======================
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
        if not audio_bytes or len(audio_bytes) < 2000:
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
            return JSONResponse(status_code=400, content={"error": "No speech detected"})

        fixed_text = smart_correct_text(raw_text)

        logging.info(f"🎤 RAW: {raw_text}")
        logging.info(f"🧠 FIXED: {fixed_text}")

        cached = smart_cache_lookup(fixed_text)
        if cached:
            audio_name = os.path.basename(cached["audio_file"])
            return {
                "text": cached["text"],
                "audio_url": f"/audio/{audio_name}",
                "cached": True
            }

        response_type = determine_response_type(fixed_text)

        system_prompt = f"""
أنت الملك رمسيس الثاني، فرعون مصر العظيم.

- الرد باللغة {LANGUAGE_NAMES.get(current_language, "العربية")}
- لا تذكر العصر الحديث أو التكنولوجيا.
- لا تخرج عن شخصيتك الفرعونية.
"""

        if response_type == "short":
            system_prompt += "\nالرد يجب أن يكون قصير جدًا."
        else:
            system_prompt += "\nالرد يجب أن يكون مفصل ودقيق تاريخيًا."

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fixed_text}
            ],
            max_output_tokens=2000
        )

        reply_text = ""
        for item in getattr(response, "output", []):
            contents = getattr(item, "content", None)
            if contents:
                for content in contents:
                    if getattr(content, "type", "") == "output_text":
                        reply_text += getattr(content, "text", "")

        reply_text = reply_text.strip() or "عذرًا لم أفهم سؤالك."

        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply_text
        )

        audio_bytes_full = speech.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes_full))
        audio = audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)

        audio_filename = os.path.join(TMP_DIR, f"reply_{len(cache)+1}.wav")
        audio.export(audio_filename, format="wav")

        cache[normalize(fixed_text)] = {
            "text": reply_text,
            "audio_file": audio_filename
        }
        save_cache()

        return {
            "text": reply_text,
            "audio_url": f"/audio/{os.path.basename(audio_filename)}",
            "cached": False
        }

    except Exception as e:
        logging.error("🔥 ERROR", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ======================
# AUDIO SERVE
# ======================
@app.get("/audio/{audio_file}")
async def serve_audio(audio_file: str):
    path = os.path.join(TMP_DIR, audio_file)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "file_not_found"})
    return FileResponse(path, media_type="audio/wav")

# ======================
# LANGUAGE
# ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):
    global current_language
    current_language = lang.lower()
    return {"status": "ok", "language": current_language}
