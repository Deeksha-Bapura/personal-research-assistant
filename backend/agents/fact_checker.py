import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from vector_store import search_texts
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "meta-llama/llama-3.1-8b-instruct"


def _check_claim(claim: str, namespace: str) -> dict:
    hits = search_texts(claim, top_k=3, namespace=namespace)
    context = "\n\n".join([f"[Source: {h['source']}]\n{h['text']}" for h in hits])

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact-checker. Given a claim and supporting sources, "
                    "determine if the claim is: SUPPORTED, CONTRADICTED, or UNVERIFIED. "
                    "Reply in this exact format:\n"
                    "STATUS: <SUPPORTED|CONTRADICTED|UNVERIFIED>\n"
                    "REASON: <one sentence explanation>"
                )
            },
            {
                "role": "user",
                "content": f"CLAIM: {claim}\n\nSOURCES:\n{context}"
            }
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    status = "UNVERIFIED"
    reason = raw
    for line in raw.split("\n"):
        if line.startswith("STATUS:"):
            status = line.replace("STATUS:", "").strip()
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
    return {"claim": claim, "status": status, "reason": reason}


def run(summaries: list[dict], query: str) -> list[dict]:
    print(f"[FactChecker] Checking claims from {len(summaries)} summaries")
    namespace = summaries[0]["namespace"] if summaries else "research"
    results = []

    for summary in summaries:
        lines = [
            l.strip() for l in summary["summary"].split("\n")
            if l.strip() and l.strip()[0].isdigit()
        ]
        print(f"[FactChecker] Checking {len(lines)} claims from '{summary['title'][:40]}'")
        for line in lines[:3]:
            claim = line.lstrip("0123456789.-) ").strip()
            if claim:
                result = _check_claim(claim, namespace)
                result["source"] = summary["url"]
                results.append(result)
                print(f"[FactChecker] {result['status']} — {claim[:60]}")

    print(f"[FactChecker] Done — {len(results)} claims checked")
    return results
