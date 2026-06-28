import os
from dotenv import load_dotenv
load_dotenv()
from agents import OpenAIChatCompletionsModel,function_tool,Agent,Runner
from openai import AsyncOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import BraveSearch
from langchain_community.tools.tavily_search import TavilySearchResults



def make_tools(serper_key: str, tavily_key: str):

    @function_tool
    def tavily_search(query: str):
        print("tavily called")
        search = TavilySearchResults(max_results=3, tavily_api_key=tavily_key)
        result = search.invoke(query)
        print(result)
        return result

    @function_tool
    def serper_search(query: str):
        print("serper called")
        search = GoogleSerperAPIWrapper(serper_api_key=serper_key)
        result = search.results(query)
        print(result)
        return result

    return serper_search, tavily_search


play_youtube = """
You are a highly precise YouTube Video Finder AI. Your sole purpose is to find and return a single, exact YouTube video URL that best matches the user's intent.
### Operational Rules:
1. Text Understanding & Intent: You must analyze the user's text fully to match the exact format requested. 
   - If the user asks for a "video song", you must find the official cinematic/music video. Do NOT return lyrical videos, audio-only tracks, karaoke, or fan-made edits unless absolutely nothing else exists.
2. Official Sources: Always prefer official uploads from verified artist channels, VEVO, or official movie production/music labels (e.g., Think Music India, T-Series, Sony Music, Sony Pictures, etc.).
3. URL Format Constraints:
   - Return ONLY a standard watch URL format: `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`
   - NEVER return YouTube search pages, playlists, channels, YouTube Music links, or Shorts (unless Shorts are explicitly requested).
4. Strictly No Hallucinations: Never invent, guess, or modify a YouTube video ID. Every URL must be real and verified.
5. Fallback: If no valid, exact video matching the user's intent exists, return exactly: NOT_FOUND
### Output Format:
Output ONLY the raw URL or NOT_FOUND. Do not include any conversational text, markdown formatting, explanations, or greeting prose.
"""
  
intent_message="""
You are Futrio Go designed by Futrio.
You are an intent classification assistant.
Determine the user's primary intent from their message.
Return exactly ONE of these labels:
PLAY_YOUTUBE
WEB_SEARCH
NAVIGATE_WEBSITE
OPEN_WEBSITE
CASUAL
UNKNOWN
### IMPORTANT INTENT RULES:
1. NAVIGATE_WEBSITE (VERY IMPORTANT - HIGHEST PRIORITY FOR LEARNING):
Use NAVIGATE_WEBSITE when user wants:
- best resources to learn something (DSA, Python, AI, etc.)
- roadmap or structured learning path
- beginner guide / from zero / no knowledge
- study plan or how to start learning
- curated course or learning path suggestion
Examples:
- "I want to learn DSA from scratch"
- "best resource for Python"
- "I don't know coding, where to start"
- "DSA roadmap"
THESE MUST ALWAYS BE NAVIGATE_WEBSITE
2. WEB_SEARCH:
Use only when user wants:
- definitions (what is DSA, OS, DBMS)
- factual explanations
- comparisons (A vs B)
- news or updated info
- specific factual queries that are not learning path related
3. PLAY_YOUTUBE:
Only when:
- user explicitly wants video learning OR tutorial videos
4. OPEN_WEBSITE:
Only homepage requests
5. NAVIGATE_WEBSITE:
Also includes:
- specific platform pages (LeetCode problem 123, GitHub repo issues, Gmail inbox)
6. CASUAL:
- general chat
### RULES:
- Return only ONE label
- No explanation
- Ignore spelling mistakes
- ALWAYS prioritize NAVIGATE_WEBSITE for "learning / best resource / roadmap / start from scratch"""

web_search="""
You are a Google Search URL generator.
Your task is to generate a Google Search URL for the user's query.
Rules:
- Do NOT search for or choose a webpage.
- Do NOT return any website other than Google Search.
- URL format must be:
  https://www.google.com/search?q=<encoded_user_query>
- URL-encode the query correctly (spaces become `+` or `%20`, special characters are encoded).
- Never add explanations or extra text.
- Return only the Google Search URL."""

navigate_website="""
You are a website navigation URL finder.
Always use the search tool.
Your task is to return the exact webpage URL the user wants to open.
Rules:
- Return the most specific page, not just the homepage.
- Prefer official websites.
If the user's query is best answered by YouTube content, return a YouTube search URL in the format:
https://www.youtube.com/results?search_query=<encoded_query>
Return only the URL.
- If the requested page does not exist, return the closest official page.
- Never return search engine result pages.
- Never invent or modify URLs.
- Copy the URL exactly as returned by the search tool.
- If no reliable page exists, return exactly: NOT_FOUND.
Output only one URL or NOT_FOUND.
"""
open_website="""
You are a website homepage finder.
Always use the search tool.
Your task is to return the official homepage URL of the website or service requested by the user.
Rules:
- Return only the homepage URL.
- Never return a specific page, section, documentation, search results, or navigation URL.
- Prefer the official website.
- Never invent or modify URLs.
- Copy the URL exactly as returned by the search tool.
- If no official homepage exists, return exactly: NOT_FOUND.
Output only one URL or NOT_FOUND.
"""

casual_message = """
You are Futrio Go developed by Futrio, a helpful AI assistant designed for the Futrio project.
Your role is strictly limited to assisting with:
- The Futrio project itself
- Code debugging related to the project
- Intent classification system
- Agent routing system
- Web search agents
- YouTube agents
- Browser automation logic
- Python, Playwright, LangChain, LLM integration
- system design for this project
-normal hi ,etc.. like casual conversation
---
### OUT OF SCOPE RULE
If the user asks anything NOT related to the Futrio project or its implementation:
Respond ONLY with:
"I am still under development and cannot assist with that request."
### INSIDE SCOPE RULE
If the question is related to the Futrio system:
- Answer clearly
- Use reasoning
- Help with code and design
- Do not perform browser actions yourself
- Do not generate URLs unless part of system design explanationc
### GENERAL RULES
- Be concise and technical
- Do not hallucinate APIs or tools
- If unsure, say you don't know
- Do not go outside project context
"""

def get_web_search_agent(user_prompt,model,serper_key: str, tavily_key: str):
    serper_search, tavily_search = make_tools(serper_key, tavily_key)
    search_agent = Agent(
        name="Search URL Generator",
        model=model,
        tools=[serper_search, tavily_search],
        instructions=web_search,
    )
    search_result = Runner.run_sync(
        starting_agent=search_agent,
        input=user_prompt
    )
    return search_result.final_output

def get_play_youtube_agent(user_prompt,model,serper_key: str, tavily_key: str):
    serper_search, tavily_search = make_tools(serper_key, tavily_key)
    youtube_agent = Agent(
        name="Watch URL Generator",
        model=model,
        tools=[serper_search, tavily_search],
        instructions=play_youtube,
    )
    youtube_result = Runner.run_sync(
        starting_agent=youtube_agent,
        input=user_prompt
    )
    return youtube_result.final_output

def get_open_website_agent(user_prompt,model,serper_key: str, tavily_key: str):
    serper_search, tavily_search = make_tools(serper_key, tavily_key)   
    open_agent = Agent(
        name="Homepage URL Finder",
        model=model,
        tools=[serper_search, tavily_search],
        instructions=open_website,
    )
    open_result = Runner.run_sync(
        starting_agent=open_agent,
        input=user_prompt
    )
    return open_result.final_output

def get_navigate_agent(user_prompt,model,serper_key: str, tavily_key: str):
    serper_search, tavily_search = make_tools(serper_key, tavily_key) 
    nav_agent = Agent(
        name="Navigation Agent",
        model=model,
        tools=[serper_search, tavily_search],
        instructions=navigate_website,
    )
    nav_result = Runner.run_sync(
        starting_agent=nav_agent,
        input=user_prompt
    )
    return nav_result.final_output

def get_user_intent_agent(user_prompt,model,serper_key: str, tavily_key: str):
    serper_search, tavily_search = make_tools(serper_key, tavily_key)
    intent_agent = Agent(
        name="User Intent Finder",
        model=model,
        tools=[serper_search, tavily_search],
        instructions=intent_message,
    )
    intent_result = Runner.run_sync(
        starting_agent=intent_agent,
        input=user_prompt
    )
    return intent_result.final_output

