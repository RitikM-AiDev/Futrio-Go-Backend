from openai.resources import beta
import tempfile
from fastapi import Request
from typing import Optional
import os
import requests
from fastapi import FastAPI, Body,Header,UploadFile
import tempfile
# pyrefly: ignore [missing-import]
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware
from open import get_play_youtube_agent,get_navigate_agent,get_open_website_agent,get_web_search_agent,get_user_intent_agent,casual_message
# pyrefly: ignore [missing-import]  
from agents import Runner,Agent,OpenAIChatCompletionsModel
from openai import AsyncOpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost",      
        "http://127.0.0.1:5173",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/admin/futrio/transcribe/at/live/244/futrio")
def user_query(file : UploadFile,request : Request = None):
    print("Before loading model")
    model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    cpu_threads=1,    
    num_workers=1,   
    download_root="/tmp/whisper_models" 
)

    print("After loading model")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(file.file.read())
        temp_path = f.name
    try:
        segments,info = model.transcribe(
        temp_path,
        language="en",
        beam_size=5,             
        vad_filter=True,
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        temperature=0.0,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-0.5,
    )
        text = "".join(segment.text for segment in segments).strip()
    except Exception as e:
        print(f"Error in ASR: {e}")
        text = ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    print(text)
    if text == "":
        return {
            "url": "None",
            "reply": "I am sorry, I didn't hear you"
        }
    api_key    = request.headers.get("X-API-Key")
    tavily_key = request.headers.get("X-Tavily-Key")
    serper_key = request.headers.get("X-Serper-Key")
    client = AsyncOpenAI(
            base_url=os.getenv("BASE"),
               api_key=api_key
        )
    model = OpenAIChatCompletionsModel(
        openai_client=client,
        model="gemini-3.1-flash-lite"
    )
    prompt_modify = """
    You are an expert search query optimizer for voice-transcribed input.

The input may come from a speech-to-text system and can contain:
- Misheard words
- Misspelled words
- Incorrect punctuation
- Missing words
- Filler words
- Homophones
- Grammatical mistakes

Your job is to infer the user's intended request and rewrite it into the best possible search query.

Rules:
- Preserve the user's intent exactly.
- Remove filler words such as "can you", "please", "I want to", "could you", "show me", etc.
- Correct spelling mistakes.
- Correct likely speech-to-text transcription errors.
- Resolve homophones and misheard words using context.
- Keep proper nouns, names, brands, places, products, and technical terms accurate. If a name is likely mistranscribed, correct it to the most probable real-world entity.
- Expand abbreviations only when doing so improves search quality.
- Add relevant keywords only if they improve search results without changing the user's intent.
- Never invent facts, names, dates, or details not implied by the user's request.
- If multiple interpretations are possible, choose the one that is most likely based on context and common usage.
- For song or music requests, append "official video" unless the user explicitly requests lyrics, audio, live, remix, karaoke, instrumental, or another specific version.
- Produce a concise, high-quality search query suitable for Google or a web search engine.

Return ONLY the optimized search query.
Do not explain your reasoning.
Do not add quotation marks or formatting.
    """
    prompt_modify_agent = Agent(
        name="Futrio Go",
        model=model,
        instructions=prompt_modify,
    )
    modified_prompt = Runner.run_sync(
        starting_agent=prompt_modify_agent,
        input=text
    )
    user_prompt = modified_prompt.final_output
    print(user_prompt)
    ans=""
    type_ = get_user_intent_agent(user_prompt, model, serper_key, tavily_key)
    print(type_)
    if type_ == "PLAY_YOUTUBE":
        youtube_result = get_play_youtube_agent(user_prompt, model, serper_key, tavily_key)
        ans=youtube_result

    elif type_ == "NAVIGATE_WEBSITE":
        navigate_result = get_navigate_agent(user_prompt, model, serper_key, tavily_key)
        ans=navigate_result

    elif type_ == "OPEN_WEBSITE":
        open_result = get_open_website_agent(user_prompt, model, serper_key, tavily_key)
        ans = open_result

    elif type_ == "WEB_SEARCH":
        print("web search running")
        search_result = get_web_search_agent(user_prompt, model, serper_key, tavily_key)
        ans = search_result

    else:
        casual_agent = Agent(
        name="Futrio Go",
        model=model,
        instructions=casual_message,
        )
        casual_result = Runner.run_sync(
            starting_agent=casual_agent,
            input=user_prompt
        )
        ans = casual_result.final_output
    if ans.startswith(("https://", "http://")):
        return {
            "url": ans,
            "reply": "Your request has been completed. Let me know if you'd like me to do anything else."
        }
    else:
        return {
            "url": "None",
            "reply": ans
    }




@app.post("/api/ask/automate/query/adhd/sjds/user")
def user_query(data : dict = Body(...),request : Request = None):

    api_key    = request.headers.get("X-API-Key")
    tavily_key = request.headers.get("X-Tavily-Key")
    serper_key = request.headers.get("X-Serper-Key")
    client = AsyncOpenAI(
            base_url=os.getenv("BASE"),
               api_key=api_key
        )
    model = OpenAIChatCompletionsModel(
        openai_client=client,
        model="gemini-3.1-flash-lite"
    )
    prompt_modify = """
    You are a query optimizer. Rewrite the user's request into the best possible search query.
    Rules:
    - Preserve intent exactly
    - Remove filler words ("can you", "please", "I want to", etc.)
    - Keep names as-is
    - Correct obvious spelling mistakes
    - Expand abbreviations only if it helps
    - Add keywords only when they improve results
    - Never invent facts
    - For song/music requests, always append "official video" unless user specifies lyrics, audio, or live — then use that instead
    Return ONLY the optimized query. No explanation.
    """
    prompt_modify_agent = Agent(
        name="Futrio Go",
        model=model,
        instructions=prompt_modify,
    )
    modified_prompt = Runner.run_sync(
        starting_agent=prompt_modify_agent,
        input=data["query"]
    )
    user_prompt = modified_prompt.final_output
    print(user_prompt)
    ans=""
    type_ = get_user_intent_agent(user_prompt, model, serper_key, tavily_key)
    print(type_)
    if type_ == "PLAY_YOUTUBE":
        youtube_result = get_play_youtube_agent(user_prompt, model, serper_key, tavily_key)
        ans=youtube_result

    elif type_ == "NAVIGATE_WEBSITE":
        navigate_result = get_navigate_agent(user_prompt, model, serper_key, tavily_key)
        ans=navigate_result

    elif type_ == "OPEN_WEBSITE":
        open_result = get_open_website_agent(user_prompt, model, serper_key, tavily_key)
        ans = open_result

    elif type_ == "WEB_SEARCH":
        print("web search running")
        search_result = get_web_search_agent(user_prompt, model, serper_key, tavily_key)
        ans = search_result

    else:
        casual_agent = Agent(
        name="Futrio Go",
        model=model,
        instructions=casual_message,
        )
        casual_result = Runner.run_sync(
            starting_agent=casual_agent,
            input=user_prompt
        )
        ans = casual_result.final_output
    if ans.startswith(("https://", "http://")):
        return {
            "url": ans,
            "reply" : "I have completed task assigned to me boss"
        }
    else:
        return {
            "url": "None",
            "reply": ans
    }
