import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "meta-llama/llama-3.1-8b-instruct"


def run(summaries: list[dict], fact_results: list[dict], query: str) -> dict:
    print(f"[Synthesizer] Building final report for: {query}")

    summaries_text = "\n\n".join([
        f"### {s['title']}\nSource: {s['url']}\n{s['summary']}"
        for s in summaries
    ])

    supported = [r for r in fact_results if r["status"] == "SUPPORTED"]
    contradicted = [r for r in fact_results if r["status"] == "CONTRADICTED"]
    unverified = [r for r in fact_results if r["status"] == "UNVERIFIED"]

    fact_text = ""
    if supported:
        fact_text += "**Supported claims:**\n" + "\n".join(f"- {r['claim']}" for r in supported) + "\n\n"
    if contradicted:
        fact_text += "**Contradicted claims:**\n" + "\n".join(f"- {r['claim']}" for r in contradicted) + "\n\n"
    if unverified:
        fact_text += "**Unverified claims:**\n" + "\n".join(f"- {r['claim']}" for r in unverified) + "\n\n"

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research report writer. Given summaries and fact-check results, "
                    "produce a structured research report in markdown with these sections:\n"
                    "# Research Report: <topic>\n"
                    "## Executive Summary\n"
                    "## Key Findings\n"
                    "## Fact-Check Results\n"
                    "## Sources\n"
                    "## Confidence Score\n\n"
                    "End with a confidence score from 0-100 based on how well the claims were verified."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Research query: {query}\n\n"
                    f"SUMMARIES:\n{summaries_text}\n\n"
                    f"FACT-CHECK RESULTS:\n{fact_text}"
                )
            }
        ],
        temperature=0.3,
    )

    report = response.choices[0].message.content.strip()
    sources = list(set([s["url"] for s in summaries]))
    confidence = len(supported) / max(len(fact_results), 1) * 100

    print(f"[Synthesizer] Done — report {len(report)} chars, confidence {confidence:.0f}%")
    return {
        "report": report,
        "sources": sources,
        "confidence": round(confidence),
        "stats": {
            "sources_scraped": len(summaries),
            "claims_checked": len(fact_results),
            "supported": len(supported),
            "contradicted": len(contradicted),
            "unverified": len(unverified),
        }
    }
