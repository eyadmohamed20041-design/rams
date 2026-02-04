import os
import json
import string

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from google import genai
from google.genai import types
from difflib import get_close_matches

from pydub import AudioSegment
from elevenlabs import ElevenLabs
from io import BytesIO

# ======================
# API KEYS (FROM ENV)
# ======================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)

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
# LANGUAGE SUPPORT
# ======================
current_language = "ar"

LANGUAGE_NAMES = {
    "ar": "العربية",
    "en": "الإنجليزية",
    "de": "الألمانية",
    "zh": "الصينية"
}

RESPONSES_FILE = "/tmp/responses.json"
MEMORY_FILE = "/tmp/memory.json"

# ======================
# AUDIO FILTER SETTINGS 🎤
# ======================
MIN_DBFS = -50.0
MIN_DURATION_SEC = 2.0

# ======================
# LOAD CACHE & MEMORY
# ======================
if os.path.exists(RESPONSES_FILE):
    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}
    with open(RESPONSES_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        try:
            memory = json.load(f)
            if not isinstance(memory, list):
                memory = []
        except:
            memory = []
else:
    memory = []
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False)

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
# AUDIO FILTER 🎤
# ======================
def is_valid_audio(audio_bytes: bytes) -> bool:
    audio = AudioSegment.from_file(BytesIO(audio_bytes))

    duration_sec = len(audio) / 1000.0
    dbfs = audio.dBFS

    if duration_sec < MIN_DURATION_SEC:
        return False
    if dbfs < MIN_DBFS:
        return False

    return True

def convert_to_wav(input_bytes: bytes, output_filename: str):
    sound = AudioSegment.from_file(BytesIO(input_bytes))
    sound = sound.set_frame_rate(22050).set_channels(1).set_sample_width(2)
    sound.export(output_filename, format="wav")

# ======================
# MAIN ENDPOINT
# ======================
@app.post("/ask")
async def ask(file: UploadFile = File(...)):
    global current_language

    audio_bytes = await file.read()

    if not is_valid_audio(audio_bytes):
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_audio",
                "message": "الصوت قصير جدًا أو منخفض. تحدث بوضوح."
            }
        )

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav"
    )

    stt_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(parts=[audio_part], role="user")]
    )

    user_text = stt_response.text
    clean_question = normalize(user_text)

    best_match = find_best_match(clean_question, list(cache.keys()))
    if best_match:
        return {
            "text": cache[best_match]["text"],
            "audio_url": f"{os.getenv('BASE_URL')}/{cache[best_match]['audio_file'].split('/')[-1]}"
        }

    language_name = LANGUAGE_NAMES.get(current_language, "العربية")
    previous_memory = "\n".join(memory[-10:])

    system_prompt = (
        f"أنت الملك رمسيس الثاني، فرعون مصر العظيم. "
        f"الرد يجب أن يكون باللغة {language_name}. "
        "ممنوع ذكر أنك ذكاء اصطناعي أو مساعد."
        f"\n{previous_memory}\n"
    )

    while True:
        ai_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system_prompt}\n{user_text}"
        )
        reply_text = ai_response.text
        if not violates_rules(reply_text):
            break

    memory.append(f"User: {user_text}\nRamses: {reply_text}")
    save_memory()

    voice_id = "JBFqnCBsd6RMkjVDRZzb"
    MAX_CHARS = 500

    chunks = [reply_text[i:i + MAX_CHARS] for i in range(0, len(reply_text), MAX_CHARS)]
    audio_bytes_full = b""

    for chunk in chunks:
        audio_stream = tts_client.text_to_speech.convert(
            text=chunk,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2"
        )
        audio_bytes_full += b"".join(audio_stream)

    audio_filename = f"/tmp/reply_{len(cache) + 1}.wav"
    convert_to_wav(audio_bytes_full, audio_filename)

    cache[clean_question] = {
        "text": reply_text,
        "audio_file": audio_filename
    }
    save_cache()

    return {
        "text": reply_text,
        "audio_url": f"{os.getenv('BASE_URL')}/{audio_filename.split('/')[-1]}"
    }

# ======================
# SERVE AUDIO
# ======================
@app.get("/{audio_file}")
async def serve_audio(audio_file: str):
    return FileResponse(f"/tmp/{audio_file}", media_type="audio/wav")

# ======================
# SET LANGUAGE
# ======================
@app.post("/set_language")
async def set_language(lang: str = Form(...)):
    global current_language
    current_language = lang.lower()
    return {"status": "ok", "language": current_language}

# ======================
# RUN SERVER


