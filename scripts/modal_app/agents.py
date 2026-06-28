"""
scripts/modal_app/agents.py — CrewAI Workflow for News Scraping & FinGPT Sentiment

This script uses CrewAI to coordinate:
1. URL Extractor Agent: Finds the latest news for Brent Crude.
2. Web Scraper Agent: Uses ScrapeGraphAI to extract clean text.
3. Financial Analyst Agent: Feeds the extracted text into FinGPT (GGUF via llama.cpp)
   to get a sentiment score between -1.0 and 1.0.
"""

import os
import yfinance as yf
from textwrap import dedent
from pydantic import BaseModel, Field

# CrewAI & Langchain
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# For local FinGPT execution
from huggingface_hub import hf_hub_download

# ============================================================
# Pydantic Schemas
# ============================================================
class NewsArticle(BaseModel):
    title: str = Field(description="The headline of the article.")
    summary: str = Field(description="A brief summary of the article's contents.")
    key_risks: str = Field(description="Any geopolitical or supply/demand risks mentioned.")

# ============================================================
# Tools
# ============================================================
@tool("Fetch Latest Brent News URLs")
def fetch_brent_news_urls() -> list[str]:
    """Fetches the latest news article URLs for Brent Crude (BZ=F) using Yahoo Finance."""
    try:
        ticker = yf.Ticker("BZ=F")
        news = ticker.news
        urls = []
        for item in news:
            if 'content' in item and 'clickThroughUrl' in item['content'] and 'url' in item['content']['clickThroughUrl']:
                urls.append(item['content']['clickThroughUrl']['url'])
            elif 'link' in item: # fallback for old format
                urls.append(item['link'])
        return urls[:3]
    except Exception as e:
        print(f"Failed to fetch news URLs: {e}")
        return []

@tool("Scrape News Article")
def scrape_news_article(url: str) -> str:
    """Scrapes the content of a news article URL and returns a clean summary."""
    # Note: In a production environment with ScrapeGraphAI installed, 
    # you would invoke the SmartScraperGraph here with the NewsArticle schema.
    # For now, we simulate the extraction to keep the container fast and reliable.
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Simple fallback extraction
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        # Truncate to first 1000 chars to avoid overwhelming the context window
        return text[:1000]
    except Exception as e:
        return f"Failed to scrape {url}: {e}"

# ============================================================
# FinGPT Local Scoring Logic
# ============================================================
def score_sentiment_fingpt(news_text: str) -> float:
    """
    Downloads (if necessary) and runs the FinGPT GGUF model via llama.cpp
    to analyze the news_text and return a score between -1.0 and 1.0.
    """
    # Semantic Scorer: FinGPT-MT-Llama-3-8B-LoRA (GGUF) - Analyzes corporate and financial news sentiment.
    REPO_ID = "second-state/FinGPT-MT-Llama-3-8B-LoRA-GGUF" 
    FILENAME = "FinGPT-MT-Llama-3-8B-LoRA-Q4_K_M.gguf"
    
    print("Downloading/Loading FinGPT GGUF model...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    try:
        from llama_cpp import Llama
        
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # Offload all layers to GPU
            n_ctx=2048,
            verbose=False
        )
        
        prompt = (
            "Instruction: What is the sentiment of this news? "
            "Please choose an answer from {negative/neutral/positive}\n"
            f"Input: {news_text[:2000]}\n"
            "Answer: "
        )
        
        response = llm(prompt, max_tokens=5, stop=["\n"], temperature=0.0)
        output_text = response['choices'][0]['text'].strip().lower()
        
        if "positive" in output_text:
            return 1.0
        elif "negative" in output_text:
            return -1.0
        else:
            return 0.0
            
    except ImportError:
        print("llama-cpp-python not installed. Returning neutral sentiment.")
        return 0.0

# Global state to capture FinGPT score accurately, bypassing CrewAI's text parsing
_captured_sentiment_score = None

@tool("FinGPT Sentiment Scorer")
def analyze_sentiment(news_text: str) -> float:
    """Analyzes a block of news text and returns a sentiment score from -1.0 to 1.0."""
    global _captured_sentiment_score
    _captured_sentiment_score = score_sentiment_fingpt(news_text)
    return _captured_sentiment_score

# ============================================================
# Agent Orchestration
# ============================================================
def run_sentiment_analysis() -> float:
    """
    Constructs the CrewAI workflow, runs it, and extracts the final sentiment score.
    Returns: float (-1.0 to 1.0)
    """
    import subprocess
    import time
    print("Downloading Qwen3.5-4B Orchestrator GGUF model...")
    qwen_model_path = hf_hub_download(
        repo_id="unsloth/Qwen3.5-4B-MTP-GGUF",
        filename="Qwen3.5-4B-BF16.gguf"
    )
    
    print("Starting local llama-cpp server on port 8080 for CrewAI orchestrator...")
    server_process = subprocess.Popen([
        "python", "-m", "llama_cpp.server",
        "--model", qwen_model_path,
        "--host", "127.0.0.1",
        "--port", "8080",
        "--n_gpu_layers", "-1",
        "--n_ctx", "4096"
    ])
    
    import urllib.request
    print("Waiting for server to start...")
    start_time = time.time()
    while time.time() - start_time < 60:
        if server_process.poll() is not None:
            raise RuntimeError(f"llama_cpp.server crashed with exit code {server_process.returncode}")
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/v1/models")
            print("Server is ready!")
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("Server failed to start within 60 seconds")
    
    orchestrator_llm = LLM(
        model="openai/qwen",
        temperature=0.1,
        max_tokens=512,
        api_key="sk-local",
        base_url="http://127.0.0.1:8080/v1"
    )
    
    # 1. Extractor Agent
    extractor = Agent(
        role="Oil Market News Extractor",
        goal="Find the top URLs containing the latest news about Brent Crude Oil (BZ=F).",
        backstory="You are an automated crawler that finds breaking news affecting global oil prices.",
        verbose=True,
        allow_delegation=False,
        llm=orchestrator_llm,
        tools=[fetch_brent_news_urls]
    )
    
    # 2. Scraper Agent
    scraper = Agent(
        role="News Content Scraper",
        goal="Extract the core text from the provided news URLs.",
        backstory="You are a data extraction specialist who strips out ads and returns only the core news content.",
        verbose=True,
        allow_delegation=False,
        llm=orchestrator_llm,
        tools=[scrape_news_article]
    )
    
    # 3. Analyst Agent
    analyst = Agent(
        role="Quantitative Financial Analyst",
        goal="Compile the news text and feed it into the FinGPT Scorer to get a final numeric sentiment score.",
        backstory="You are a quant trader who relies purely on numerical sentiment data.",
        verbose=True,
        allow_delegation=False,
        llm=orchestrator_llm,
        tools=[analyze_sentiment]
    )
    
    # Tasks
    task1 = Task(
        description="Use the 'fetch_latest_brent_news_ur_ls' tool with an empty action input '{}' to get the top 3 latest news URLs for Brent Crude.",
        expected_output="A list of 3 URLs.",
        agent=extractor
    )
    
    task2 = Task(
        description="Use the 'scrape_news_article' tool on the URLs found by the Extractor.",
        expected_output="A combined string of the latest news text.",
        agent=scraper
    )
    
    task3 = Task(
        description=(
            "Use the 'fin_gpt_sentiment_scorer' tool on the combined news text to get the final numeric score.\n"
            "CRITICAL FORMATTING REQUIREMENT: You MUST invoke the tool BEFORE providing a Final Answer.\n"
            "Do NOT write 'Final Answer:' until AFTER you have received the 'Observation:' from the tool."
        ),
        expected_output="A single float representing the average sentiment.",
        agent=analyst
    )
    
    crew = Crew(
        agents=[extractor, scraper, analyst],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True
    )
    
    print("Starting CrewAI Sentiment Pipeline...")
    try:
        result = crew.kickoff()
    finally:
        server_process.terminate()
    
    # Return the exact float captured by the tool, bypassing CrewAI's raw text parsing.
    global _captured_sentiment_score
    if _captured_sentiment_score is not None:
        return _captured_sentiment_score
    return 0.0

if __name__ == "__main__":
    # Local testing
    score = run_sentiment_analysis()
    print(f"Final Score: {score}")
