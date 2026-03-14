# Task 3 Plan: The System Agent

## Overview

Extend agent.py with a new tool `query_api` to query the live backend API. This enables the agent to answer questions about live system state (database, API endpoints) in addition to static documentation.

## New Tool: query_api

### Schema

```json
{
  "name": "query_api",
  "description": "Query the backend API to get live data",
  "parameters": {
    "type": "object",
    "properties": {
      "method": {"type": "string", "description": "HTTP method (GET, POST, etc.)"},
      "path": {"type": "string", "description": "API path (e.g., /api/items)"},
      "body": {"type": "string", "description": "Optional JSON body for POST/PUT requests"}
    },
    "required": ["method", "path"]
  }
}
```

### Implementation

- Base URL from env: `AGENT_API_BASE_URL` (default: `http://localhost:42002`)
- Auth header: `Authorization: Bearer {LMS_API_KEY}`
- Returns JSON string with `status_code` and `body`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | LLM provider API key |
| `LLM_API_BASE` | — | LLM API base URL |
| `LLM_MODEL` | — | Model name |
| `LMS_API_KEY` | — | Backend API auth key |
| `AGENT_API_BASE_URL` | `http://localhost:42002` | Backend API base URL |

## Tool Selection Strategy

The system prompt guides the model:
- **Wiki/source questions** → use `list_files` / `read_file`
- **Live API questions** → use `query_api`

Example prompt:
> "Use list_files/read_file for documentation questions. Use query_api for questions about live system state, database, or API endpoints."

## Initial Benchmark / Iteration Strategy

1. Test with common eval questions:
   - "What framework does the backend use?" → read_file backend/main.py
   - "How many items are in the database?" → query_api GET /api/items
2. If model chooses wrong tool, refine system prompt
3. Limit to 5 tool calls max for efficiency

## Output Format

```json
{
  "answer": "The backend uses FastAPI framework.",
  "source": "backend/main.py",
  "tool_calls": [
    {"tool": "read_file", "arguments": {"path": "backend/main.py"}}
  ]
}
```
