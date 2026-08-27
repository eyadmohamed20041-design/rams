import os
import re
import time
import json
import logging
from typing import Generator

from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from openai import OpenAI


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO
)


# =========================================================
# API KEYS
# =========================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

API_SECRET = os.getenv(
    "API_SECRET",
    "SECRET123"
)

if not OPENAI_API_KEY:
    raise Exception(
        "OPENAI_API_KEY not found"
    )


client = OpenAI(
    api_key=OPENAI_API_KEY
)


# =========================================================
# APP
# =========================================================

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# RATE LIMIT
# =========================================================

user_last_request = {}

MIN_INTERVAL = 0.5


# =========================================================
# HELPERS
# =========================================================

def normalize(text: str):
    return re.sub(
        r"\s+",
        " ",
        text.lower().strip()
    )


def detect_lang_fallback(lang: str):

    valid = [
        "ar",
        "en",
        "de",
        "cn"
    ]

    if not lang:
        return "en"

    lang = (
        lang
        .lower()
        .strip()
    )

    if lang not in valid:
        return "en"

    return lang


def get_lang_instruction(lang: str):

    if lang == "ar":
        return "أجب باللغة العربية فقط."

    if lang == "en":
        return "Reply only in English."

    if lang == "de":
        return "Antworte nur auf Deutsch."

    if lang == "cn":
        return "只用中文回答。"

    return "Reply only in English."


# =========================================================
# SYSTEM PROMPT
# =========================================================

def build_system_prompt(
    lang: str,
    detailed: bool
):

    lang_instruction = get_lang_instruction(
        lang
    )

    prompt = f"""

أنت الملك رمسيس الثاني، فرعون مصر العظمى، ولا شيء آخر.

{lang_instruction}

قواعد الشخصية:

1. أنت رمسيس الثاني فقط.

2. تحدث دائماً بأسلوب:
- ملكي.
- هادئ.
- حكيم.
- واثق.
- رسمي.

3. معرفتك الأساسية مرتبطة بعصر رمسيس الثاني ومصر القديمة، وتشمل:
- حياتك.
- الأسرة التاسعة عشرة.
- مصر القديمة.
- المعابد.
- الحروب.
- معركة قادش.
- الآثار.
- الدين المصري القديم.
- الحياة اليومية في الدولة الحديثة.
- تاريخ مصر القديم المرتبط بعهدك أو بما سبقه.

4. إذا سألك المستخدم عن شيء حديث أو خارج نطاق عصرك:
لا تجب عنه بشكل مباشر.

قل بأسلوب ملكي إن هذا الأمر ليس من زمانك ولا من علوم عصرك، ووجّه الحديث إلى مصر القديمة.

5. لا تتحدث عن الذكاء الاصطناعي باعتبارك نظاماً.

6. لا تقل إنك ChatGPT.

7. لا تخرج من شخصية رمسيس الثاني.

8. لا تخمن.

9. لا تستخدم معلومات حديثة كأنك عشتها.

10. إذا كانت المعلومة غير معروفة لك، اعترف بذلك داخل الشخصية بدلاً من اختراع إجابة.

11. هذه محادثة صوتية مع زائر، لذلك:
- لا تستخدم عناوين كثيرة.
- لا تستخدم قوائم طويلة.
- لا تكرر السؤال.
- لا تبدأ كل إجابة بعبارة ثابتة.
- اجعل الكلام طبيعياً عند سماعه صوتياً.

"""

    if detailed:

        prompt += """

عندما يكون السؤال تاريخياً ومناسباً لعصرك:

أعطِ إجابة مفصلة، ولكن لا تجعلها محاضرة طويلة.
اشرح التفاصيل المهمة بطريقة سهلة للمحادثة الصوتية.
"""

    else:

        prompt += """

عندما يكون السؤال عادياً:

أعطِ إجابة متوسطة.

اجعل الإجابة تقريباً من 2 إلى 4 جمل صوتية واضحة.

لا تجعلها قصيرة جداً بحيث تبدو كإجابة مبتورة.

ولا تجعلها طويلة بحيث تصبح محاضرة.

الأولوية للمعلومة المهمة ثم لمسة تاريخية أو شخصية من رمسيس الثاني.
"""

    return prompt


# =========================================================
# SENTENCE SPLITTER
# =========================================================

def split_sentences(buffer: str):

    """
    يحاول إخراج جملة مكتملة بمجرد انتهاء
    علامة ترقيم مناسبة.

    يدعم:
    العربية
    الإنجليزية
    الألمانية
    الصينية
    """

    pattern = r"(.+?[.!?؟。！？])(?:\s+|$)"

    match = re.search(
        pattern,
        buffer,
        flags=re.DOTALL
    )

    if not match:
        return None, buffer

    sentence = match.group(1).strip()

    rest = buffer[
        match.end():
    ].strip()

    return sentence, rest


# =========================================================
# STREAM GPT
# =========================================================

def generate_stream(
    text: str,
    lang: str,
    detailed: bool
) -> Generator[str, None, None]:

    system_prompt = build_system_prompt(
        lang,
        detailed
    )

    logging.info(
        f"STREAM USER: {text} | LANG: {lang}"
    )

    stream = client.responses.create(

        model="gpt-4o-mini",

        input=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": text
            }

        ],

        max_output_tokens=350,

        stream=True
    )

    buffer = ""

    for event in stream:

        try:

            if (
                event.type
                == "response.output_text.delta"
            ):

                delta = event.delta

                if not delta:
                    continue

                buffer += delta

                while True:

                    sentence, rest = split_sentences(
                        buffer
                    )

                    if not sentence:
                        break

                    buffer = rest

                    yield json.dumps(
                        {
                            "type": "sentence",
                            "text": sentence
                        },
                        ensure_ascii=False
                    ) + "\n"

        except Exception:

            logging.exception(
                "STREAM EVENT ERROR"
            )

    # =====================================================
    # REMAINING TEXT
    # =====================================================

    if buffer.strip():

        yield json.dumps(
            {
                "type": "sentence",
                "text": buffer.strip()
            },
            ensure_ascii=False
        ) + "\n"


# =========================================================
# ASK STREAM
# =========================================================

@app.post("/ask/stream")
async def ask_stream(
    request: Request,
    text: str = Form(...),
    lang: str = Form("en")
):

    try:

        # =====================================================
        # AUTH
        # =====================================================

        if (
            request.headers.get("x-api-key")
            != API_SECRET
        ):

            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden"
                }
            )

        # =====================================================
        # RATE LIMIT
        # =====================================================

        ip = request.client.host

        now = time.time()

        last = user_last_request.get(
            ip,
            0
        )

        if (
            now - last
            < MIN_INTERVAL
        ):

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests"
                }
            )

        user_last_request[ip] = now

        # =====================================================
        # INPUT
        # =====================================================

        text = normalize(
            text
        )

        if not text:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Empty text"
                }
            )

        # =====================================================
        # LANGUAGE
        # =====================================================

        lang = detect_lang_fallback(
            lang
        )

        # =====================================================
        # DETAIL DETECTION
        # =====================================================

        detailed_keywords = [

            "بالتفصيل",
            "اشرح",
            "شرح",
            "تفصيل",

            "explain",
            "details",
            "in detail",

            "erkläre",

            "详细",
            "解释"
        ]

        detailed = any(
            word in text
            for word in detailed_keywords
        )

        # =====================================================
        # STREAM
        # =====================================================

        return StreamingResponse(

            generate_stream(
                text,
                lang,
                detailed
            ),

            media_type="application/x-ndjson",

            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:

        logging.error(
            "STREAM SERVER ERROR",
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# =========================================================
# OLD ASK ENDPOINT
# =========================================================

@app.post("/ask")
async def ask(
    request: Request,
    text: str = Form(...),
    lang: str = Form("en")
):

    try:

        # =====================================================
        # AUTH
        # =====================================================

        if (
            request.headers.get("x-api-key")
            != API_SECRET
        ):

            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden"
                }
            )

        # =====================================================
        # INPUT
        # =====================================================

        text = normalize(
            text
        )

        if not text:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Empty text"
                }
            )

        # =====================================================
        # LANGUAGE
        # =====================================================

        lang = detect_lang_fallback(
            lang
        )

        # =====================================================
        # DETAIL DETECTION
        # =====================================================

        detailed_keywords = [

            "بالتفصيل",
            "اشرح",
            "شرح",
            "تفصيل",

            "explain",
            "details",
            "in detail",

            "erkläre",

            "详细",
            "解释"
        ]

        detailed = any(
            word in text
            for word in detailed_keywords
        )

        # =====================================================
        # SYSTEM PROMPT
        # =====================================================

        system_prompt = build_system_prompt(
            lang,
            detailed
        )

        # =====================================================
        # OPENAI
        # =====================================================

        response = client.responses.create(

            model="gpt-4o-mini",

            input=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": text
                }

            ],

            max_output_tokens=350
        )

        # =====================================================
        # RESPONSE TEXT
        # =====================================================

        reply = response.output_text.strip()

        if not reply:

            reply = "لم أفهم سؤالك."

        logging.info(
            f"RAMSES: {reply}"
        )

        # =====================================================
        # TTS
        # =====================================================

        speech = client.audio.speech.create(

            model="gpt-4o-mini-tts",

            voice="alloy",

            input=reply
        )

        return Response(
            content=speech.read(),
            media_type="audio/mpeg"
        )

    except Exception as e:

        logging.error(
            "ASK ERROR",
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# =========================================================
# TTS
# =========================================================

@app.post("/tts")
async def tts(
    request: Request,
    text: str = Form(...),
    lang: str = Form("en")
):

    try:

        # =====================================================
        # AUTH
        # =====================================================

        if (
            request.headers.get("x-api-key")
            != API_SECRET
        ):

            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden"
                }
            )

        # =====================================================
        # INPUT
        # =====================================================

        text = text.strip()

        if not text:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Empty text"
                }
            )

        # =====================================================
        # LANGUAGE
        # =====================================================

        lang = detect_lang_fallback(
            lang
        )

        # =====================================================
        # TTS
        # =====================================================

        speech = client.audio.speech.create(

            model="gpt-4o-mini-tts",

            voice="alloy",

            input=text
        )

        return Response(
            content=speech.read(),
            media_type="audio/mpeg"
        )

    except Exception as e:

        logging.error(
            "TTS ERROR",
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
async def health():

    return {
        "status": "running",
        "mode": "ramesses_streaming_multilingual_interruptible"
    }
