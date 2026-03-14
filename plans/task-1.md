# Task 1 Plan: Call an LLM from Code

## Provider Choice

- **Provider style:** OpenAI-compatible API
- **Model source:** Environment variable `LLM_MODEL`
- **API endpoint:** `{LLM_API_BASE}/chat/completions`

## Architecture

Simple linear flow:

```
CLI argument → agent.py → HTTP POST to LLM API → Parse response → JSON output
```

## Components

1. **CLI parsing** — read question from `sys.argv[1]`
2. **Environment config** — read `LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL` from environment
3. **HTTP request** — POST to `{LLM_API_BASE}/chat/completions` with standard OpenAI format
4. **Response parsing** — extract `choices[0].message.content` from API response
5. **JSON output** — print `{"answer": "...", "tool_calls": []}` to stdout

## Error Handling

- Missing CLI argument → stderr message, exit 1
- Missing env vars → stderr message, exit 1
- HTTP errors → stderr message, exit 1

## Testing

- Mock HTTP server in test to simulate LLM API
- Run `agent.py` as subprocess with test env vars
- Assert: exit code 0, valid JSON, `answer` present, `tool_calls == []`
