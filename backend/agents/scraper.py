"""
scraper.py — Scraper Agent
Uses OpenRouter (Llama 3.1 free) to identify relevant URLs,
then Firecrawl to scrape each page.
"""

import os
import json
from firecrawl import FirecrawlApp
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_firecrawl = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "meta-llama/llama-3.1-8b-instruct"


def _get_urls(query: str, num_sources: int) -> list[str]:
    """Ask LLM to suggest real, scrapeable URLs for this query."""
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Given a research query, "
                    "return a JSON array of ONLY real, working, publicly accessible URLs "
                    "that definitely exist and contain relevant information. "
                    "Prefer: Wikipedia, arxiv.org, nature.com, wired.com, techcrunch.com, "
                    "medium.com, towardsdatascience.com, blogs.microsoft.com, openai.com/blog, "
                    "deepmind.google/discover/blog. "
                    "Do NOT invent or guess URLs. Only use URLs you are confident exist. "
                    "Only return the JSON array, nothing else. "
                    f"Return exactly {num_sources} URLs."
                )
            },
            {
                "role": "user",
                "content": f"Research query: {query}"
            }
        ],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    urls = json.loads(raw)
    return urls[:num_sources]


def run(query: str, num_sources: int = 5) -> list[dict]:
    """
    Get URLs for the query and scrape each one.
    Returns list of {url, title, content} dicts.
    """
    print(f"[Scraper] Finding sources for: {query}")
    urls = _get_urls(query, num_sources)
    print(f"[Scraper] Got {len(urls)} URLs: {urls}")

    sources = []
    for url in urls:
        print(f"[Scraper] Scraping: {url}")
        try:
            scraped = _firecrawl.scrape_url(
                url,
                params={"formats": ["markdown"]}
            )
            content = scraped.get("markdown", "")
            title = scraped.get("metadata", {}).get("title", url)

            if content:
                sources.append({
                    "url": url,
                    "title": title,
                    "content": content[:3000],
                })
                print(f"[Scraper] ✓ {title} ({len(content)} chars)")
        except Exception as e:
            print(f"[Scraper] ✗ Failed {url}: {e}")
            continue

    print(f"[Scraper] Done — {len(sources)} sources collected")
    return sources