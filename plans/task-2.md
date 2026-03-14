# Task 2 Plan: The Documentation Agent

## Overview

Extend agent.py to support tool calling with two tools: `read_file` and `list_files`. The agent will use these tools to answer questions about the project documentation in the `wiki/` directory.

## Tools

### read_file(path: str) -> str
- Reads file contents relative to project root
- Returns error message if path is outside project root (security)
- Returns error message if file doesn't exist

### list_files(path: str) -> str
- Lists files/directories relative to project root
- Returns newline-separated string
- Returns error message if path is outside project root (security)

## Agentic Loop

1. Send user question + system prompt + tool schemas to LLM
2. If LLM returns tool calls:
   - Execute each tool call
   - Append tool results to conversation
   - Repeat (max 10 iterations)
3. When LLM returns text answer (no tool calls), return final JSON

## Output Format

```json
{
  "answer": "Final answer text",
  "source": "wiki/git-workflow.md#resolving-conflicts",
  "tool_calls": [
    {"tool": "list_files", "arguments": {"path": "wiki"}},
    {"tool": "read_file", "arguments": {"path": "wiki/git-workflow.md"}}
  ]
}
```

## System Prompt

Tell the model to:
- Use `list_files` to discover wiki files
- Use `read_file` to find specific information
- Provide `source` as file path with optional section anchor

## Testing

Two regression tests with mock HTTP server:
1. Test with `list_files` + `read_file` sequence
2. Test with `list_files` only
