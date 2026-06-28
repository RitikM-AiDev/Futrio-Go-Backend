from fastapi import Request
from typing import Optional
import os
import requests
from fastapi import FastAPI, Body,Header
from fastapi.middleware.cors import CORSMiddleware
from open import get_play_youtube_agent,get_navigate_agent,get_open_website_agent,get_web_search_agent,get_user_intent_agent,casual_message
# pyrefly: ignore [missing-import]  
from agents import Runner,Agent,OpenAIChatCompletionsModel
from openai import AsyncOpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/query/user")
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
