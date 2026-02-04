import os
import json
import string
import time
from difflib import get_close_matches
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from elevenlabs import ElevenLabs

# ======================
# Gemini (STT)
# ======================
import google.genai as genai
from google.genai import types

# ======================
# Groq AI (LLM)
# ======================
from groq import Groq

# ======================
# API KEYS
# ======================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_SECRET = os.getenv("API_SECRET", "SECRET123")  # 🔐 بسيط للحماية

# ======================
# GLOBAL CLIENTS
# ======================
stt_client = None
tts_client = None
llm_client = None

# ======================
# RATE LIMIT
# ======================
last_request_time = 0
MIN_INTERVAL = 3  # ثواني بين كل request

# ======================
# LIFESPAN
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_client, tts_client, llm_client

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY missing")

    if not ELEVEN_API_KEY:
        raise RuntimeError("ELEVEN_API_KEY missing")

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing")

    stt_client = genai.Client(api_key=GOOGLE_API_KEY)
    tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)
    llm_client = Groq(api_key=GROQ_API_KEY)

    print("✅ Server Started")

    yield

    print("🛑 Server Stopped")

# ======================
# SERVER SETUP
# ======================
app = FastAPI(lifespan=lifespan)

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
# FILES
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

# ======================
# HELPERS
# ======================
def normalize(text: str):
    text = text.lower().replace(" ", "")
    return text.translate(str.maketrans("", "", string.punctuation))

def find_best_match(question, cache_keys, cutoff=0.8):
    matches = get_close_matches(question, cache_keys, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def save_cache():
    json.dump(cache, open(RESPONSES_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def save_memory():
    json.dump(memory, open(MEMORY_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def violates_rules(text):
    forbidden = ["ذكاء", "اصطناعي", "ai", "assistant", "model"]
    return any(w in text.lower() for w in forbidden)

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

        # ===== STT =====
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"
        )

        stt_response = stt_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(parts=[audio_part], role="user")]
        )

        user_text = stt_response.text or ""

        if not user_text.strip():
            return JSONResponse(status_code=400, content={"error": "Empty speech"})

        clean_question = normalize(user_text)

        # ===== CACHE =====
        best_match = find_best_match(clean_question, list(cache.keys()))
        if best_match:
            audio_file = os.path.basename(cache[best_match]["audio_file"])
            return {
                "text": cache[best_match]["text"],
                "audio_url": f"/audio/{audio_file}"
            }

        # ===== LLM =====
        language_name = LANGUAGE_NAMES.get(current_language, "العربية")
        previous_memory = "\n".join(memory[-5:])

        system_prompt = f"""
أنت الملك رمسيس الثاني، فرعون مصر العظيم.
الرد يجب أن يكون باللغة {language_name}.
ممنوع ذكر أنك ذكاء اصطناعي.
{previous_memory}
"""

        completion = llm_client.chat.completions.create(
            model = "llama-3.1-8b-instant",  # ✅ أخف وأأمن
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.6,
            max_tokens=300
        )

        reply_text = completion.choices[0].message.content

        if violates_rules(reply_text):
            reply_text = "تفضل أيها الزائر الكريم، بماذا تأمر؟"

        memory.append(f"User: {user_text}\nRamses: {reply_text}")
        save_memory()

        # ===== TTS =====
        audio_stream = tts_client.text_to_speech.convert(
            text=reply_text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2"
        )

        audio_bytes_full = b"".join(audio_stream)
        audio_filename = os.path.join(TMP_DIR, f"reply_{len(cache)+1}.wav")

        with open(audio_filename, "wb") as f:
            f.write(audio_bytes_full)

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
        err = str(e)
        print("🔥 ASK ERROR:", err)

        if "detected_unusual_activity" in err:
            return JSONResponse(
                status_code=401,
                content={"error": "AI provider blocked Free Tier usage"}
            )

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

