import os
import io
import json
import string
import time
import logging

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pydub import AudioSegment
import openai

# ======================
# LOGGING
# ======================
logging.basicConfig(level=logging.INFO)

# ======================
# API KEYS
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_SECRET = os.getenv("API_SECRET", "SECRET123")
openai.api_key = OPENAI_API_KEY

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
# HELPER: تحديد نوع الرد
# ======================
def determine_response_type(user_text: str):
    greetings = ["إزيك","ازيك","كيف حالك","مرحبا","hello","hi","hallo","你好"]
    user_lower = user_text.lower()
    for word in greetings:
        if word in user_lower:
            return "short"
    return "normal"

# ======================
# MAIN ENDPOINT
# ======================
@app.post("/ask")
async def ask(request: Request, file: UploadFile = File(...)):
    global last_request_time

    # AUTH
    if request.headers.get("x-api-key") != API_SECRET:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    # RATE LIMIT
    now = time.time()
    if now - last_request_time < MIN_INTERVAL:
        return JSONResponse(status_code=429, content={"error": "Too many requests"})
    last_request_time = now

    try:
        # READ AUDIO
        audio_bytes = await file.read()
        if not audio_bytes or len(audio_bytes) < 2000:
            return JSONResponse(status_code=400, content={"error": "Audio too small"})
        logging.info(f"📥 Audio size: {len(audio_bytes)} bytes")

        # WHISPER STT
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "speech.wav"

        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        user_text = transcript.strip()
        if not user_text or len(user_text) < 2:
            return JSONResponse(status_code=400, content={"error": "No clear speech detected"})
        logging.info(f"🎤 USER: {user_text}")

        # CACHE
        clean_question = normalize(user_text)
        if clean_question in cache:
            audio_file_name = os.path.basename(cache[clean_question]["audio_file"])
            return {
                "text": cache[clean_question]["text"],
                "audio_url": f"/audio/{audio_file_name}"
            }

        # تحديد نوع الرد
        response_type = determine_response_type(user_text)

        # ======================
        # SYSTEM PROMPT
        # ======================
        system_prompt = f"""
أنت الملك رمسيس الثاني، فرعون مصر العظيم.
الرد يجب أن يكون باللغة {LANGUAGE_NAMES.get(current_language, "العربية")}.
ممنوع ذكر أنك ذكاء اصطناعي.
"""

        if response_type == "short":
            system_prompt += """
الردود:
- إذا كان السؤال عن التحية أو الإحوال أو الأسئلة البسيطة اليومية، رد **جملة قصيرة مباشرة**.
- إذا كان السؤال تاريخي أو عن أحداث مصر أو معلومات عامة، اعطي رد **مفصل وطويل**.
"""
        else:
            system_prompt += """
الردود:
- حاول أن يكون الرد مفصل وواضح إذا كان السؤال تاريخي أو عن أحداث مصر أو معلومات عامة.
- تجنب الردود القصيرة جدًا.
"""

        # GPT
        completion = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            max_completion_tokens=250
        )

        reply_text = completion.choices[0].message.content.strip()
        if not reply_text or len(reply_text) < 3:
            # fallback آمن
            reply_text = "أهلاً! 😃" if response_type == "short" else "لم أستطع الرد الآن، حاول مرة أخرى."
        logging.info(f"🤖 AI: {reply_text}")

        # TTS
        audio_output = openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply_text
        )
        audio_bytes_full = audio_output.read()
        if not audio_bytes_full:
            raise Exception("TTS failed")

        # CONVERT TO WAV
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes_full))
        audio = audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)

        audio_filename = os.path.join(TMP_DIR, f"reply_{len(cache)+1}.wav")
        audio.export(audio_filename, format="wav")

        # SAVE CACHE
        cache[clean_question] = {"text": reply_text, "audio_file": audio_filename}
        save_cache()

        return {"text": reply_text, "audio_url": f"/audio/{os.path.basename(audio_filename)}"}

    except Exception as e:
        logging.error(f"🔥 ERROR: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# ======================
# AUDIO
# ======================
@app.get("/audio/{audio_file}")
async def serve_audio(audio_file: str):
    file_path = os.path.join(TMP_DIR, audio_file)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "file_not_found"})
    return FileResponse(file_path, media_type="audio/wav")

# ======================
# LANGUAGE
# ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):
    global current_language
    current_language = lang.lower()
    return {"status": "ok", "language": current_language}
