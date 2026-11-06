import os
import json
import re
import urllib.parse
import warnings
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from google import genai
from google.genai import types

warnings.filterwarnings("ignore")
load_dotenv(override=True)

def tavily_search(query: str, tavily_api_key: str = None):
    key = tavily_api_key or os.getenv("TAVILY_API_KEY") or os.getenv("tavily_api_key")
    if key:
        search = TavilySearchResults(max_results=3, tavily_api_key=key)
        print(search)
    else:
        search = TavilySearchResults(max_results=3)
        print(search)
    return search.invoke(query)

def run_url_generator_agent(query: str, gemini_api_key: str = None, serper_key: str = None, tavily_key: str = None, language: str = None):
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("X-API-Key")
    client = genai.Client(api_key=api_key)

    def youtube_search(video_title: str) -> str:
        try:
            res1 = str(tavily_search(f"{video_title} site:youtube.com/watch", tavily_api_key=tavily_key))
            print(res1)
            return res1
        except Exception as err:
            return f"Error: {err}"

    def tavily_web_search(search_query: str) -> str:
        try:
            res = str(tavily_search(search_query, tavily_api_key=tavily_key))
            print(res)
            return res
        except Exception as err:
            return f"Error: {err}"

    system_instruction = """You are Futrio Go, an intelligent desktop and browser action agent.

Your primary responsibility is to understand the user's COMPLETE intent before selecting any action. Analyze the entire query, identify the desired outcome, extract relevant entities, and preserve every requested action. Never act on an isolated keyword or the first recognized intent.

RULES:

1. INTENT UNDERSTANDING
- Determine what the user actually wants to accomplish, not merely which application they mention.
- Distinguish between opening an application, performing an action within it, finding specific content, and asking a question.
- If a query contains multiple actions, prioritize the final desired outcome and fulfill all compatible actions.
- Application names are context, not necessarily the user's primary intent.
- Never discard information, constraints, names, or actions from the user's query.

2. ACTION ROUTING
- Opening an application or website alone: return its official homepage URL.
- Requesting specific content or an operation within an application: identify the relevant destination and return its exact actionable URL.
- For media playback, find the exact playable content URL, never a search-results URL.
- For searches, locate the requested information or destination rather than merely opening the search platform.
- For communication, distinguish between opening a messaging application and preparing a message for a recipient.
- Apply the same reasoning to every application; do not depend on predefined application-specific examples.
- If an action requires a tool, use the appropriate tool and verify its result before returning a URL.
- If no external action is requested, return url as "None" and answer normally.

3. WHATSAPP
- If a phone number is provided, return https://wa.me/<phone>?text=<encoded_message>.
- Otherwise, return https://web.whatsapp.com.
- Distinguish exact message content from instructions to generate a message.
- Generate a complete, natural message when the user describes an intended communication.
- Preserve explicitly supplied message content.
- Never include the user's instruction or task description in the generated message.
- Return only the final WhatsApp URL.

4. LANGUAGE
- Respond in the user's language.
- Tanglish/Tamil queries may receive English or Tamil replies.
- Never use Hindi unless the user explicitly writes in Hindi using Devanagari script.

5. ACCURACY
- Never invent URLs, search results, or completed actions.
- Do not assume that mentioning an application means the user only wants to open it.
- If essential information is missing, ask one concise clarification.
- Prefer the user's complete intended outcome over literal keyword matching.
CRITICAL: COMPLETE-INTENT PRIORITY

Always interpret the entire user query before routing.

If the user mentions an application AND requests specific content or an operation within it, the requested content or operation MUST take priority over simply opening the application.

Extract:
- Application/platform
- Requested content or target
- Desired action
- Additional constraints

The application name identifies WHERE to perform the task; the remaining words identify WHAT to do.

Never return an application homepage when the user has requested specific content within that application.

If the query contains multiple actions, preserve their order and execute the complete request. Never stop after identifying the first action.
6. OUTPUT
Return ONLY valid JSON:
{
  "url": "exact actionable URL or None",
  "reply": "concise response in the user's language"
}"""

    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash"
    ]
    if language in ("ta", "Tamil"):
        prompt_input = f"User Request (Tamil / Tanglish): {query}"
    elif language:
        prompt_input = f"Spoken Language: {language}\nUser Request: {query}"
    else:
        prompt_input = query

    response = None
    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[youtube_search, tavily_web_search],
                    temperature=0.2,
                )
            )
            if response:
                break
        except Exception as err:
            last_error = err
            err_str = str(err).lower()
            if "not found" in err_str or "404" in err_str or "503" in err_str or "unknown model" in err_str or "unsupported" in err_str:
                continue
            break

    url = "None"
    reply = ""

    if response and response.text:
        text = response.text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    url = data.get("url") or "None"
                    reply = data.get("reply") or ""
            except Exception:
                pass

        if not reply:
            urls = re.findall(r'https?://[^\s<>"\')]+', text)
            if urls:
                url = urls[0]
                reply = text.replace(url, "").strip() or f"Found URL: {url}"
            else:
                url = "None"
                reply = text

    if "results?search_query=" in url:
        url = "None"

    is_video_url = url != "None" and ("youtube.com/watch" in url or "youtu.be/" in url)
    if not is_video_url:
        video_markers = (
            "video", "watch", "song", "music", "track",
            "paatu", "paadal", "padam", "pattu", "geetham", "gana", "gaana",
            "cancion", "musica", "chanson", "lied", "uta",
            "பாட்டு", "பாடல்", "வீடியோ", "படம்",
            "गाना", "गीत", "वीडियो", "పాట", "వీడియో"
        )
        if any(k in query.lower() for k in video_markers):
            raw_text = str(response.text) if response else ""
            yt_urls = re.findall(r'https?://(?:www\.)?youtube\.com/watch\?v=[a-zA-Z0-9_-]+|https?://youtu\.be/[a-zA-Z0-9_-]+', raw_text)
            if yt_urls:
                url = yt_urls[0]
            else:
                direct_search = youtube_search(query) or ""
                found = re.findall(r'https?://(?:www\.)?youtube\.com/watch\?v=[a-zA-Z0-9_-]+|https?://youtu\.be/[a-zA-Z0-9_-]+', str(direct_search))
                if found:
                    url = found[0]
                else:
                    url = "None"

    if not reply:
        if last_error:
            err_msg = str(last_error).lower()
            if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                if language in ("ta", "Tamil"):
                    reply = "உங்கள் API quota வரம்பு முடிந்துவிட்டது (Error 429). புதிய Gmail கணக்கைத் தொடங்கி புதிய Gemini API key-ஐ மாற்றவும் அல்லது கூடுதல் வரம்புகளுக்கு சந்தா பெறவும்."
                else:
                    reply = "You have exceeded your current API quota limit (Error 429). Please update your API key by creating a new key with a new Gmail account, or upgrade your plan to get higher limits."
            else:
                reply = f"Error: {last_error}"
        else:
            reply = "I have completed the task assigned to me." if url != "None" else "No response generated."

    err_check = reply.lower()
    if "429" in err_check or "resource_exhausted" in err_check or "quota" in err_check:
        if language in ("ta", "Tamil"):
            reply = "உங்கள் API quota வரம்பு முடிந்துவிட்டது (Error 429). புதிய Gmail கணக்கைத் தொடங்கி புதிய Gemini API key-ஐ மாற்றவும் அல்லது கூடுதல் வரம்புகளுக்கு சந்தா பெறவும்."
        else:
            reply = "You have exceeded your current API quota limit (Error 429). Please update your API key by creating a new key with a new Gmail account, or upgrade your plan to get higher limits."

    if not url or url.strip().lower() in ("null", "none", "n/a", ""):
        url = "None"

    return url, reply
