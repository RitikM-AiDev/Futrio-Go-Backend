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
    q_lower = query.lower().strip()
    is_play_intent = any(w in q_lower for w in ("play", "watch", "song", "paatu", "paadal", "track", "music"))
    if not is_play_intent:
        is_open_intent = any(w in q_lower for w in ("open", "go to", "launch", "start", "visit", "pannu"))
        if is_open_intent:
            direct_sites = {
                "youtube": "https://www.youtube.com",
                "google": "https://www.google.com",
                "github": "https://github.com",
                "gmail": "https://mail.google.com",
                "instagram": "https://www.instagram.com",
                "whatsapp": "https://web.whatsapp.com",
                "twitter": "https://x.com",
                "reddit": "https://www.reddit.com",
                "chatgpt": "https://chatgpt.com",
                "spotify": "https://open.spotify.com",
                "netflix": "https://www.netflix.com",
                "amazon": "https://www.amazon.com",
                "facebook": "https://www.facebook.com",
                "linkedin": "https://www.linkedin.com"
            }
            for site_name, site_url in direct_sites.items():
                if site_name in q_lower:
                    reply_text = f"Opening {site_name.capitalize()}."
                    if language in ("ta", "Tamil"):
                        reply_text = f"{site_name.capitalize()} திறக்கிறேன்."
                    return site_url, reply_text

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

    system_instruction = """You are Futrio Go, an AI agent for desktop and browser actions.
Your task is to analyze the user query in whatever language it is presented, use the search tools when required, and return a verified URL (if an action such as playing a song/video or opening a website is requested) and an answer/reply in the user's language.

Rules:
1. Intent Classification:
- Opening Websites / Apps: When the user asks to open a website or platform (e.g. 'open youtube', 'youtube open pannu', 'open google', 'github open pannu'), return the official homepage URL (e.g. https://www.youtube.com). DO NOT search for or return video tutorials on how to open a website!
- Playing Videos or Songs: Only when the user wants to play/watch specific content, songs, or videos (e.g. 'play verappa song', 'watch trailer'), use youtube_search to find and return the exact playable YouTube watch URL (format: https://www.youtube.com/watch?v=... or https://youtu.be/...). NEVER return a search results URL (https://www.youtube.com/results?search_query=...).
- Questions or Chat: Return 'None' for url and give a clear answer in reply.
- WhatsApp Messages: Return the WhatsApp URL (https://wa.me/<phone>?text=<encoded_text>).

2. Language Consistency:
- NEVER reply in Hindi unless the user query is explicitly in Hindi and written in Devanagari script.
- If the user speaks or writes in Tanglish or Tamil (e.g. 'youtube open pannu', 'paatu podu'): reply in English or Tamil (e.g. 'Opening YouTube' or 'YouTube திறக்கிறேன்'). Do NOT reply in Hindi.
- If the user query is in English, reply in English.
- Always respect the user's language and never switch to Hindi unless Hindi was explicitly spoken.

3. Output Format:
You must respond ONLY with a valid JSON object in this format:
{
    "url": "the exact URL or None",
    "reply": "the answer or confirmation for the user"
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
        is_open_homepage = any(w in query.lower() for w in ("open pannu", "open youtube", "open google", "open whatsapp", "open github"))
        if not is_open_homepage and any(k in query.lower() for k in video_markers):
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
