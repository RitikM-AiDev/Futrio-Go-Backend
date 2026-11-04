import os
import json
import re
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from google import genai
from google.genai import types

load_dotenv(override=True)

def tavily_search(query: str, tavily_api_key: str = None):
    key = tavily_api_key or os.getenv("TAVILY_API_KEY") or os.getenv("tavily_api_key")
    if key:
        search = TavilySearchResults(max_results=3, tavily_api_key=key)
    else:
        search = TavilySearchResults(max_results=3)
    return search.invoke(query)

def serper_search(query: str, serper_api_key: str = None):
    key = serper_api_key or os.getenv("SERPER_API_KEY") or os.getenv("serper_api_key")
    if key:
        search = GoogleSerperAPIWrapper(serper_api_key=key)
    else:
        search = GoogleSerperAPIWrapper()
    return search.results(query)

def run_url_generator_agent(query: str, gemini_api_key: str = None, serper_key: str = None, tavily_key: str = None):
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("X-API-Key")
    client = genai.Client(api_key=api_key)

    def google_search(search_query: str) -> str:
        try:
            return str(serper_search(search_query, serper_api_key=serper_key))
        except Exception as err:
            return f"Error: {err}"

    def tavily_web_search(search_query: str) -> str:
        try:
            return str(tavily_search(search_query, tavily_api_key=tavily_key))
        except Exception as err:
            return f"Error: {err}"

    system_instruction = """You are Futrio Go, an AI agent for desktop and browser actions.
Your task is to analyze the user query, use Google Search and Tavily Search tools when required, and return a verified URL (if an action such as playing a song/video or opening a website is requested) and an answer/reply explaining what you are doing or answering the user request.
Rules:
1. For music or video playback requests, find and return the official YouTube watch URL (e.g. https://www.youtube.com/watch?v=...).
2. For website navigation or opening requests, return the exact official homepage or target URL.
3. For questions, factual lookups, or casual conversation, provide the direct answer in the reply and set url to "None".
4.If User ask you to open whatsapp and send a message you can generate url for that request
4. You must respond ONLY with a valid JSON object in this format:
{
    "url": "the exact URL or None",
    "reply": "the answer or confirmation for the user"
}"""

    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    response = None
    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[google_search, tavily_web_search],
                    temperature=0.2,
                )
            )
            if response:
                break
        except Exception as err:
            last_error = err
            err_str = str(err).lower()
            if "not found" in err_str or "404" in err_str or "unknown model" in err_str or "unsupported" in err_str:
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

    if not reply:
        if last_error:
            reply = f"Error: {last_error}"
        else:
            reply = "I have completed the task assigned to me." if url != "None" else "No response generated."

    if not url or url.strip().lower() in ("null", "none", "n/a", ""):
        url = "None"

    return url, reply
