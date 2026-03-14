#!/usr/bin/env python3
"""Minimal LLM agent that answers questions via OpenAI-compatible API."""

import json
import os
import sys
import urllib.request
import urllib.error


def main():
    # Parse CLI argument
    if len(sys.argv) < 2:
        print("Usage: agent.py <question>", file=sys.stderr)
        sys.exit(1)

    question = sys.argv[1]

    # Read config from environment
    api_key = os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_API_BASE")
    model = os.environ.get("LLM_MODEL")

    if not api_key:
        print("Error: LLM_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not api_base:
        print("Error: LLM_API_BASE not set", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("Error: LLM_MODEL not set", file=sys.stderr)
        sys.exit(1)

    # Build request
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": question},
        ],
    }).encode("utf-8")

    # Send request
    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Request failed: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract answer
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"Unexpected API response: {e}", file=sys.stderr)
        sys.exit(1)

    # Output result
    result = {"answer": answer, "tool_calls": []}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
