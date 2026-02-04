import os
import json
import string
from difflib import get_close_matches
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from elevenlabs import ElevenLabs

# ======================
# Gemini AI (Google)
# ======================
import google.genai as genai
from google.genai import types


# ======================
# API KEYS
# ======================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")


# ======================
# GLOBAL CLIENTS
# ======================
client = None
tts_client = None


# ======================
# LIFESPAN
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, tts_client

    try:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is missing")

        if not ELEVEN_API_KEY:
            raise RuntimeError("ELEVEN_API_KEY is missing")

        client = genai.Client(api_key=GOOGLE_API_KEY)
        tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)

        print("✅ Server Started")

        yield

        print("🛑 Server Stopped")

    except Exception as e:
        print("🔥 STARTUP ERROR:", e)
        raise e


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
# ROOT (Health Check)
# ======================
@app.get("/")
def root():
    return {"status": "running"}


# ======================
# LANGUAGE SUPPORT
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
    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)


if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)


# ======================
# HELPERS
# ======================
def normalize(text: str):
    text = text.lower().replace(" ", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def find_best_match(question, cache_keys, cutoff=0.8):
    matches = get_close_matches(question, cache_keys, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def save_cache():
    with open(RESPONSES_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def violates_rules(text):
    forbidden = [
        "ذكاء", "اصطناعي", "ai", "assistant",
        "model", "machine", "language model", "computer"
    ]
    return any(w in text.lower() for w in forbidden)


# ======================
# AUDIO HELPERS
# ======================
def is_valid_audio(audio_bytes: bytes):
    return True


def save_audio_file(audio_bytes: bytes, output_filename: str):
    with open(output_filename, "wb") as f:
        f.write(audio_bytes)


# ======================
# MAIN ENDPOINT
# ======================
@app.post("/ask")
async def ask(request: Request, file: UploadFile = File(...)):

    try:

        audio_bytes = await file.read()

        # ===== STT =====
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"
        )

        try:
            stt_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Content(parts=[audio_part], role="user")]
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "stt_failed", "details": str(e)}
            )

        user_text = stt_response.text or ""
        clean_question = normalize(user_text)


        # ===== CACHE =====
        best_match = find_best_match(clean_question, list(cache.keys()))

        if best_match:

            audio_file = os.path.basename(cache[best_match]["audio_file"])

            return {
                "text": cache[best_match]["text"],
                "audio_url": f"/audio/{audio_file}"
            }


        # ===== AI RESPONSE =====
        language_name = LANGUAGE_NAMES.get(current_language, "العربية")

        previous_memory = "\n".join(memory[-10:])

        system_prompt = f"""
أنت الملك رمسيس الثاني، فرعون مصر العظيم.
الرد يجب أن يكون باللغة {language_name}.
ممنوع ذكر أنك ذكاء اصطناعي.
{previous_memory}
        """


        while True:

            ai_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_prompt}\n{user_text}"
            )

            reply_text = ai_response.text or ""

            if not violates_rules(reply_text):
                break


        memory.append(f"User: {user_text}\nRamses: {reply_text}")
        save_memory()


        # ===== TTS =====
        voice_id = "JBFqnCBsd6RMkjVDRZzb"

        try:
            audio_stream = tts_client.text_to_speech.convert(
                text=reply_text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2"
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "tts_failed", "details": str(e)}
            )


        audio_bytes_full = b"".join(audio_stream)

        audio_filename = os.path.join(
            TMP_DIR, f"reply_{len(cache)+1}.wav"
        )

        save_audio_file(audio_bytes_full, audio_filename)


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

        print("🔥 ASK ERROR:", e)

        return JSONResponse(
            status_code=500,
            content={"error": "server_error", "details": str(e)}
        )


# ======================
# SERVE AUDIO
# ======================
@app.get("/audio/{audio_file}")
async def serve_audio(audio_file: str):

    file_path = os.path.join(TMP_DIR, audio_file)

    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"error": "file_not_found"}
        )

    return FileResponse(file_path, media_type="audio/wav")


# ======================
# SET LANGUAGE
# ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):

    global current_language

    current_language = lang.lower()

    return {
        "status": "ok",
        "language": current_language
    }
