"""
agent.py — Agentic Research Assistant
======================================
Drop this file into your /backend folder alongside server.py.

Adds a POST /agent endpoint to your existing Flask app with:
  - Tool use (search_documents, web_search, summarize_topic, flag_for_review)
  - ReAct reasoning loop (Claude decides when/what to retrieve)
  - Session memory across turns

HOW TO INTEGRATE INTO server.py:
  from agent import agent_bp
  app.register_blueprint(agent_bp)

DEPENDENCIES:
  pip install anthropic flask chromadb tavily-python
"""

import json
import os
from datetime import datetime
from flask import Blueprint, request, jsonify
import anthropic
import chromadb
from tavily import TavilyClient

# ── Blueprint ─────────────────────────────────────────────────────────────────
agent_bp = Blueprint("agent", __name__)

# ── Clients ───────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# Reuse the same ChromaDB collection your server.py already creates
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = chroma_client.get_collection("documents")
except Exception:
    collection = chroma_client.create_collection("documents")

# ── In-memory session store (replace with Redis/DB for production) ─────────────
sessions: dict[str, list] = {}

# ── Tool definitions (passed to Claude) ───────────────────────────────────────
TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search the user's uploaded documents using semantic similarity. "
            "Use this when the question might be answered by internal documents, "
            "policies, contracts, or reports the user has uploaded. "
            "Returns the top matching chunks with their source filenames."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific — use key terms from the question.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 4, max: 8).",
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current information not in the uploaded documents. "
            "Use for recent news, regulations, market data, or any topic not covered "
            "by the user's documents. Returns summaries from top results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise web search query (like you'd type into Google).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "summarize_topic",
        "description": (
            "Perform a deep search across ALL uploaded documents on a broad topic and "
            "return a synthesized summary. Use when the user asks for a comprehensive "
            "overview rather than a specific fact. More thorough than search_documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to summarize across all documents.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "flag_for_review",
        "description": (
            "Flag the current question or finding for human review. Use when: "
            "(1) you find conflicting information across sources, "
            "(2) the answer has compliance or legal implications, "
            "(3) you're not confident enough to give a definitive answer. "
            "Always explain WHY you're flagging."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this needs human review.",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium"],
                    "description": "Your confidence level in the available information.",
                },
            },
            "required": ["reason", "confidence"],
        },
    },
]

# ── Tool implementations ───────────────────────────────────────────────────────

def tool_search_documents(query: str, n_results: int = 4) -> str:
    """Query ChromaDB and return formatted results."""
    n_results = min(n_results, 8)
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return "No relevant documents found for this query."

        output_parts = [f"Found {len(docs)} relevant document chunk(s):\n"]
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            source = meta.get("source", meta.get("filename", "Unknown source"))
            relevance = round((1 - dist) * 100, 1)
            output_parts.append(
                f"[{i}] Source: {source} | Relevance: {relevance}%\n{doc[:600]}\n"
            )
        return "\n".join(output_parts)

    except Exception as e:
        return f"Document search error: {str(e)}"


def tool_web_search(query: str) -> str:
    """Web search via Tavily — reliable, AI-optimized results."""
    try:
        result = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )

        output_parts = []

        # Top results
        for r in result.get("results", [])[:4]:
            output_parts.append(
                f"• [{r.get('title', 'No title')}] {r.get('url', '')}\n"
                f"  {r.get('content', '')[:400]}\n"
            )

        return "\n".join(output_parts) if output_parts else "No results found."

    except Exception as e:
        return f"Web search error: {str(e)}"


def tool_summarize_topic(topic: str) -> str:
    """Pull many chunks across all docs and ask Claude to synthesize them."""
    try:
        results = collection.query(
            query_texts=[topic],
            n_results=10,
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return f"No documents found covering '{topic}'."

        combined = "\n\n---\n\n".join(
            f"[From: {m.get('source', m.get('filename', 'Unknown'))}]\n{d[:800]}"
            for d, m in zip(docs, metas)
        )

        synthesis = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Synthesize the following document excerpts into a clear, "
                        f"structured summary about: {topic}\n\n{combined}"
                    ),
                }
            ],
        )
        return synthesis.content[0].text

    except Exception as e:
        return f"Summarization error: {str(e)}"


def tool_flag_for_review(reason: str, confidence: str) -> str:
    """Record a flag and return acknowledgment."""
    timestamp = datetime.now().isoformat()
    print(f"[FLAG FOR REVIEW] reason={reason} confidence={confidence} time={timestamp}")
    return (
        f"Flagged for human review.\n"
        f"Reason: {reason}\n"
        f"Confidence level: {confidence}\n"
        f"Timestamp: {timestamp}"
    )


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_documents":
        return tool_search_documents(
            query=tool_input["query"],
            n_results=tool_input.get("n_results", 4),
        )
    elif tool_name == "web_search":
        return tool_web_search(query=tool_input["query"])
    elif tool_name == "summarize_topic":
        return tool_summarize_topic(topic=tool_input["topic"])
    elif tool_name == "flag_for_review":
        return tool_flag_for_review(
            reason=tool_input["reason"],
            confidence=tool_input["confidence"],
        )
    else:
        return f"Unknown tool: {tool_name}"


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert research and compliance assistant for banking and enterprise use.

You have access to four tools:
1. search_documents — search the user's uploaded internal documents
2. web_search — search the web for current/external information
3. summarize_topic — deep synthesis across all uploaded documents
4. flag_for_review — escalate uncertain or compliance-sensitive findings

## Your reasoning approach:
- ALWAYS think before acting: consider what information you need and which tool is best
- For questions about internal policies, contracts, or uploaded reports → start with search_documents
- For recent regulations, market data, or external facts → use web_search
- For broad topics needing synthesis → use summarize_topic
- If you find conflicting info or low confidence on important matters → flag_for_review
- You may call multiple tools in sequence — retrieve, evaluate, retrieve more if needed
- Cite your sources in the final answer (e.g., "According to [filename]..." or "According to [URL]...")
- Be concise but complete. Structure complex answers with headers.
- Always flag if an answer has legal, compliance, or financial implications you're uncertain about.
"""

# ── ReAct agentic loop ─────────────────────────────────────────────────────────

def run_agent(user_message: str, session_id: str) -> dict:
    """Run the full agentic ReAct loop for one user turn."""
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": "user", "content": user_message})

    tools_used = []
    flagged = False
    max_iterations = 8

    for iteration in range(max_iterations):
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=history,
        )

        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            history.append({"role": "assistant", "content": response.content})
            sessions[session_id] = history
            return {
                "answer": final_text,
                "tools_used": tools_used,
                "flagged": flagged,
                "iterations": iteration + 1,
            }

        elif response.stop_reason == "tool_use":
            history.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tools_used.append(tool_name)

                    if tool_name == "flag_for_review":
                        flagged = True

                    print(f"[AGENT] Tool: {tool_name} | Input: {tool_input}")
                    result = dispatch_tool(tool_name, tool_input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            history.append({"role": "user", "content": tool_results})

        else:
            break

    return {
        "answer": "I reached the maximum number of reasoning steps. Please try a more specific question.",
        "tools_used": tools_used,
        "flagged": flagged,
        "iterations": max_iterations,
    }


# ── Flask endpoints ────────────────────────────────────────────────────────────

@agent_bp.route("/api/agent", methods=["POST"])
def agent_endpoint():
    """
    POST /api/agent
    Body: { "message": "...", "session_id": "..." }
    Returns: { "answer": "...", "tools_used": [...], "flagged": bool, "iterations": int }
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    user_message = data["message"].strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    result = run_agent(user_message, session_id)
    return jsonify(result)


@agent_bp.route("/api/agent/session/<session_id>", methods=["DELETE"])
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"message": f"Session '{session_id}' cleared."})


@agent_bp.route("/api/agent/health", methods=["GET"])
def agent_health():
    """Status check."""
    doc_count = collection.count()
    tavily_ok = bool(os.environ.get("TAVILY_API_KEY"))
    return jsonify({
        "status": "ok",
        "documents_indexed": doc_count,
        "tavily_configured": tavily_ok,
        "tools_available": [t["name"] for t in TOOLS],
    })