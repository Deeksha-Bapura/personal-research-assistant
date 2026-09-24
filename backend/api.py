import os
import asyncio
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Multi-Agent Research API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str
    num_sources: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research/stream")
async def research_stream(req: ResearchRequest):
    async def event_generator():
        query = req.query
        num_sources = req.num_sources

        try:
            # Agent 1 — Scraper
            yield {"event": "agent_start", "data": json.dumps({"agent": "scraper", "message": f"Finding sources for: {query}"})}
            await asyncio.sleep(0)

            from agents.scraper import run as scrape
            sources = await asyncio.to_thread(scrape, query, num_sources)
            yield {"event": "agent_done", "data": json.dumps({"agent": "scraper", "message": f"Found {len(sources)} sources", "count": len(sources)})}
            await asyncio.sleep(0)

            # Agent 2 — Summarizer
            yield {"event": "agent_start", "data": json.dumps({"agent": "summarizer", "message": f"Summarizing {len(sources)} sources"})}
            await asyncio.sleep(0)

            from agents.summarizer import run as summarize
            summaries = await asyncio.to_thread(summarize, sources, query)
            yield {"event": "agent_done", "data": json.dumps({"agent": "summarizer", "message": f"Summarized {len(summaries)} sources"})}
            await asyncio.sleep(0)

            # Agent 3 — Fact Checker
            yield {"event": "agent_start", "data": json.dumps({"agent": "fact_checker", "message": "Cross-referencing claims"})}
            await asyncio.sleep(0)

            from agents.fact_checker import run as fact_check
            facts = await asyncio.to_thread(fact_check, summaries, query)
            yield {"event": "agent_done", "data": json.dumps({"agent": "fact_checker", "message": f"Checked {len(facts)} claims"})}
            await asyncio.sleep(0)

            # Agent 4 — Synthesizer
            yield {"event": "agent_start", "data": json.dumps({"agent": "synthesizer", "message": "Writing final report"})}
            await asyncio.sleep(0)

            from agents.synthesizer import run as synthesize
            result = await asyncio.to_thread(synthesize, summaries, facts, query)
            yield {"event": "agent_done", "data": json.dumps({"agent": "synthesizer", "message": "Report complete"})}
            await asyncio.sleep(0)

            # Final report
            yield {"event": "report", "data": json.dumps(result)}

        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())
