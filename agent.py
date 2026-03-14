#!/usr/bin/env python3
"""LLM agent with tool calling for read_file and list_files."""

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
    }
]

# Map function names to implementations
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_files": list_files,
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
        "You are a documentation assistant. Use list_files to discover wiki files, "
        "then read_file to find specific information. "
        "Answer based on the file contents. Include source as the file path."
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
