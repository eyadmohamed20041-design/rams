import os
import json
import string
import time
import io

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pydub import AudioSegment
import openai

# ======================
# API KEYS
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # لازم تحط هنا المفتاح
API_SECRET = os.getenv("API_SECRET", "SECRET123")  # مفتاح حماية بسيط
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
    "en": "الإنجليزية",
    "de": "الألمانية",
    "zh": "الصينية"
}

# ======================
# FILES & CACHE
# ======================
TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

RESPONSES_FILE = os.path.join(TMP_DIR, "responses.json")
MEMORY_FILE = os.path.join(TMP_DIR, "memory.json")

cache = {}
memory = []

if os.path.exists(RESPONSES_FILE):
    cache = json.load(open(RESPONSES_FILE, encoding="utf-8"))

if os.path.exists(MEMORY_FILE):
    memory = json.load(open(MEMORY_FILE, encoding="utf-8"))

def save_cache():
    json.dump(cache, open(RESPONSES_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def save_memory():
    json.dump(memory, open(MEMORY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def normalize(text: str):
    text = text.lower().replace(" ", "")
    return text.translate(str.maketrans("", "", string.punctuation))

# ======================
# RATE LIMIT
# ======================
last_request_time = 0
MIN_INTERVAL = 3  # ثواني بين كل request

# ======================
# MAIN ENDPOINT
# ======================
@app.post("/ask")
async def ask(request: Request, file: UploadFile = File(...)):
    global last_request_time

    # 🔐 Secret Header
    if request.headers.get("x-api-key") != API_SECRET:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    # ⏱ Rate Limit
    now = time.time()
    if now - last_request_time < MIN_INTERVAL:
        return JSONResponse(status_code=429, content={"error": "Too many requests"})
    last_request_time = now

    try:
        audio_bytes = await file.read()

        # ===== STT: تحويل الصوت لنص باستخدام OpenAI Whisper =====
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "speech.wav"
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        user_text = transcript.text
        if not user_text.strip():
            return JSONResponse(status_code=400, content={"error": "Empty speech"})

        clean_question = normalize(user_text)

        # ===== CACHE =====
        if clean_question in cache:
            audio_file_name = os.path.basename(cache[clean_question]["audio_file"])
            return {
                "text": cache[clean_question]["text"],
                "audio_url": f"/audio/{audio_file_name}"
            }

        # ===== LLM: الرد على النص باستخدام GPT-5 Mini =====
        system_prompt = f"""
أنت الملك رمسيس الثاني، فرعون مصر العظيم.
الرد يجب أن يكون باللغة {LANGUAGE_NAMES.get(current_language, "العربية")}.
ممنوع ذكر أنك ذكاء اصطناعي.
السياق السابق:
{chr(10).join(memory[-5:])}
"""
        completion = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.6,
            max_tokens=300
        )
        reply_text = completion.choices[0].message.content
        memory.append(f"User: {user_text}\nRamses: {reply_text}")
        save_memory()

        # ===== TTS: تحويل النص لصوت باستخدام OpenAI TTS =====
        audio_output = openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=reply_text
        )

        audio_bytes_full = audio_output.read()

        # تحويل الصوت لصيغة WAV متوافقة
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes_full))
        audio = audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)

        audio_filename = os.path.join(TMP_DIR, f"reply_{len(cache)+1}.wav")
        audio.export(audio_filename, format="wav")

        # حفظ الـ cache
        cache[clean_question] = {
            "text": reply_text,
            "audio_file": audio_filename
        }
        save_cache()

        return {
            "text": reply_text,
            "audio_url": f"/audio/{os.path.basename(audio_filename)}"
        }

    except Exception as e:
        print("🔥 ERROR:", e)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

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
