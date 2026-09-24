"""
summarizer.py — Summarizer Agent
Takes raw scraped sources, condenses each one to key claims,
and stores summaries in Pinecone for later retrieval.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from vector_store import upsert_texts
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "meta-llama/llama-3.1-8b-instruct"


def _summarize_one(source: dict) -> str:
    """Summarize a single source into key claims."""
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research summarizer. Given a web page's content, "
                    "extract the 5 most important factual claims or insights. "
                    "Format as a numbered list. Be concise and specific."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Source: {source['url']}\n"
                    f"Title: {source['title']}\n\n"
                    f"Content:\n{source['content'][:2000]}"
                )
            }
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def run(sources: list[dict], query: str) -> list[dict]:
    """
    Summarize each source and store in Pinecone.
    Returns list of {url, title, summary, namespace} dicts.
    """
    print(f"[Summarizer] Summarizing {len(sources)} sources")
    summaries = []
    namespace = f"research-{abs(hash(query)) % 65536}"

    for source in sources:
        print(f"[Summarizer] Summarizing: {source['title'][:60]}")
        try:
            summary = _summarize_one(source)
            summaries.append({
                "url": source["url"],
                "title": source["title"],
                "summary": summary,
                "namespace": namespace,
            })
            print(f"[Summarizer] ✓ Done ({len(summary)} chars)")
        except Exception as e:
            print(f"[Summarizer] ✗ Failed: {e}")
            continue

    if summaries:
        texts = [s["summary"] for s in summaries]
        metadata = [
            {"source": s["url"], "agent": "summarizer", "title": s["title"]}
            for s in summaries
        ]
        upsert_texts(texts, metadata, namespace=namespace)
        print(f"[Summarizer] Stored {len(summaries)} summaries in Pinecone namespace '{namespace}'")

    print(f"[Summarizer] Done — {len(summaries)} summaries produced")
    return summaries