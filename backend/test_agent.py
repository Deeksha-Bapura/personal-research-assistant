"""
test_agent.py — Quick test script for the agentic endpoint
============================================================
Run this after starting your Flask server to verify everything works.

Usage:
  python test_agent.py
"""

import requests

BASE_URL = "http://localhost:5000"
SESSION_ID = "test-session-001"


def pretty(label: str, data: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Answer:     {data.get('answer', '')[:200]}...")
    print(f"  Tools used: {data.get('tools_used', [])}")
    print(f"  Flagged:    {data.get('flagged', False)}")
    print(f"  Iterations: {data.get('iterations', 0)}")


def ask(message: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/api/agent",
        json={"message": message, "session_id": SESSION_ID},
    )
    return resp.json()


if __name__ == "__main__":
    # 1. Health check
    print("\nChecking agent health...")
    health = requests.get(f"{BASE_URL}/api/agent/health").json()
    print(f"  Status: {health['status']}")
    print(f"  Documents indexed: {health['documents_indexed']}")
    print(f"  Tavily configured: {health['tavily_configured']}")
    print(f"  Tools: {health['tools_available']}")

    # 2. Test: should use search_documents
    result = ask("What does our compliance policy say about data retention?")
    pretty("Test 1: Internal doc search", result)

    # 3. Test: should use web_search (Tavily)
    result = ask("What are the latest Basel III capital requirements for banks in 2024?")
    pretty("Test 2: Web search for regulations", result)

    # 4. Test: multi-step — should search docs AND web
    result = ask(
        "Compare our internal risk policy against current regulatory standards. "
        "Flag anything that seems outdated."
    )
    pretty("Test 3: Multi-step with possible flag", result)

    # 5. Test: conversation memory (follow-up)
    result = ask("Can you expand on the first point you mentioned?")
    pretty("Test 4: Follow-up (uses session memory)", result)

    # Clear session
    requests.delete(f"{BASE_URL}/api/agent/session/{SESSION_ID}")
    print("\n  Session cleared. ✓")