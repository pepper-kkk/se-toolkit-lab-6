#!/usr/bin/env python3
"""LLM agent with tool calling for read_file, list_files, and query_api."""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path


# Project root for security checks
PROJECT_ROOT = Path(__file__).parent.resolve()
MAX_TOOL_CALLS = 10


def read_file(path: str) -> str:
    """Read a file relative to project root. Reject paths outside root."""
    # Security: resolve and check for traversal
    try:
        full_path = (PROJECT_ROOT / path).resolve()
        if not str(full_path).startswith(str(PROJECT_ROOT)):
            return f"Error: Path '{path}' is outside project root"
    except Exception as e:
        return f"Error: Invalid path '{path}': {e}"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{path}' not found"
    except Exception as e:
        return f"Error reading '{path}': {e}"


def list_files(path: str) -> str:
    """List files/directories relative to project root. Reject paths outside root."""
    # Security: resolve and check for traversal
    try:
        full_path = (PROJECT_ROOT / path).resolve()
        if not str(full_path).startswith(str(PROJECT_ROOT)):
            return f"Error: Path '{path}' is outside project root"
    except Exception as e:
        return f"Error: Invalid path '{path}': {e}"

    try:
        entries = os.listdir(full_path)
        return "\n".join(sorted(entries))
    except FileNotFoundError:
        return f"Error: Directory '{path}' not found"
    except NotADirectoryError:
        return f"Error: '{path}' is not a directory"
    except Exception as e:
        return f"Error listing '{path}': {e}"


def query_api(method: str, path: str, body: str = None) -> str:
    """Query the backend API. Returns JSON string with status_code and body."""
    api_base = os.environ.get("AGENT_API_BASE_URL", "http://localhost:42002")
    lms_api_key = os.environ.get("LMS_API_KEY")

    if not lms_api_key:
        return json.dumps({"status_code": 0, "body": "Error: LMS_API_KEY not set"})

    url = f"{api_base.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {lms_api_key}",
    }

    data = None
    if body:
        headers["Content-Type"] = "application/json"
        data = body.encode("utf-8")

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_body = resp.read().decode("utf-8")
            return json.dumps({"status_code": resp.status, "body": response_body})
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        return json.dumps({"status_code": e.code, "body": error_body})
    except urllib.error.URLError as e:
        return json.dumps({"status_code": 0, "body": f"Error: {e.reason}"})
    except Exception as e:
        return json.dumps({"status_code": 0, "body": f"Error: {e}"})


# Tool schemas for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file relative to the project root",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file relative to project root (e.g., 'wiki/git.md')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a directory relative to project root",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory relative to project root (e.g., 'wiki')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_api",
            "description": "Query the backend API to get live data about database, items, or system state",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method (GET, POST, etc.)"
                    },
                    "path": {
                        "type": "string",
                        "description": "API path (e.g., /api/items)"
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional JSON body for POST/PUT requests"
                    }
                },
                "required": ["method", "path"]
            }
        }
    }
]

# Map function names to implementations
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_files": list_files,
    "query_api": query_api,
}


def call_llm(messages: list, api_key: str, api_base: str, model: str, with_tools: bool = True) -> dict:
    """Send messages to LLM and return response."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    body_data = {
        "model": model,
        "messages": messages,
    }
    if with_tools:
        body_data["tools"] = TOOLS

    body = json.dumps(body_data).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Request failed: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response: {e}", file=sys.stderr)
        sys.exit(1)


def execute_tool_call(tool_call: dict) -> str:
    """Execute a single tool call and return the result."""
    func_name = tool_call["function"]["name"]
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError:
        return "Error: Invalid arguments JSON"

    if func_name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{func_name}'"

    func = TOOL_FUNCTIONS[func_name]
    try:
        return func(**args)
    except TypeError as e:
        return f"Error: Invalid arguments for {func_name}: {e}"
    except Exception as e:
        return f"Error executing {func_name}: {e}"


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

    # System prompt
    system_prompt = (
        "You are a documentation and system assistant. "
        "Use list_files/read_file for documentation questions about wiki or source code. "
        "Use query_api for questions about live system state, database, or API endpoints. "
        "Answer based on the information you gather. Include source when available."
    )

    # Initialize conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Track tool calls for output
    all_tool_calls = []
    source = ""

    # Agentic loop
    for _ in range(MAX_TOOL_CALLS):
        response = call_llm(messages, api_key, api_base, model)

        # Extract message from response
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            print(f"Unexpected API response: {e}", file=sys.stderr)
            sys.exit(1)

        # Check for tool calls
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            # No tool calls - model returned final answer
            answer = message.get("content", "")
            if not source and answer:
                source = "llm"
            break

        # Execute tool calls
        for tool_call in tool_calls:
            # Record tool call for output
            try:
                args = json.loads(tool_call["function"]["arguments"])
                all_tool_calls.append({
                    "tool": tool_call["function"]["name"],
                    "arguments": args,
                })
                # Track source from read_file calls
                if tool_call["function"]["name"] == "read_file":
                    source = args.get("path", "")
            except (json.JSONDecodeError, KeyError):
                pass

            # Execute and get result
            result = execute_tool_call(tool_call)

            # Append tool result to conversation
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", "1"),
                "content": result,
            })
    else:
        # Max iterations reached
        answer = "Error: Max tool calls reached"
        if not source:
            source = "error"

    # Output result
    result = {
        "answer": answer,
        "source": source,
        "tool_calls": all_tool_calls,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
