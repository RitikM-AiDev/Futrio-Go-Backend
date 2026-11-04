import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from toolsss import run_url_generator_agent

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/admin/futrio/transcribe/at/live/244/futrio")
async def user_query(file: UploadFile, request: Request = None):
    groq_key = None
    gemini_key = None
    tavily_key = None
    serper_key = None

    if request:
        groq_key = request.headers.get("X-Groq-Key")
        gemini_key = request.headers.get("X-API-Key")
        tavily_key = request.headers.get("X-Tavily-Key")
        serper_key = request.headers.get("X-Serper-Key")

    groq_key = groq_key or os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    gemini_key = gemini_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("X-API-Key")
    tavily_key = tavily_key or os.getenv("TAVILY_API_KEY") or os.getenv("tavily_api_key")
    serper_key = serper_key or os.getenv("SERPER_API_KEY") or os.getenv("serper_api_key")

    contents = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(contents)
        temp_path = temp_audio.name

    try:
        groq_client = Groq(api_key=groq_key)
        with open(temp_path, "rb") as audio:
            result = groq_client.audio.transcriptions.create(
                file=audio,
                model="whisper-large-v3-turbo",
                response_format="text"
            )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    query_text = result if isinstance(result, str) else getattr(result, "text", str(result))
    query_text = query_text.strip() if query_text else ""

    if not query_text:
        return {
            "url": "None",
            "reply": "No speech detected in audio file."
        }

    url, reply = run_url_generator_agent(
        query=query_text,
        gemini_api_key=gemini_key,
        serper_key=serper_key,
        tavily_key=tavily_key
    )
    print(url,reply)

    return {
        "url": url,
        "reply": reply
    }

@app.post("/api/ask/automate/query/adhd/sjds/user")
async def text_query(request: Request):
    data = await request.json()
    query_text = data.get("query", "").strip()

    gemini_key = request.headers.get("X-API-Key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("X-API-Key")
    tavily_key = request.headers.get("X-Tavily-Key") or os.getenv("TAVILY_API_KEY") or os.getenv("tavily_api_key")
    serper_key = request.headers.get("X-Serper-Key") or os.getenv("SERPER_API_KEY") or os.getenv("serper_api_key")

    if not query_text:
        return {
            "url": "None",
            "reply": "No query provided."
        }

    url, reply = run_url_generator_agent(
        query=query_text,
        gemini_api_key=gemini_key,
        serper_key=serper_key,
        tavily_key=tavily_key
    )

    return {
        "url": url,
        "reply": reply
    }
