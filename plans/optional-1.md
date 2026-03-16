# Optional Task 1 Plan — Advanced Agent Features

## Chosen extensions

I implemented two extensions:

1. Retry logic with exponential backoff
2. In-memory caching for tool results

## Why these extensions

### Retry logic with backoff
LLM API requests may fail with temporary errors such as:
- HTTP 429 (Too Many Requests)
- HTTP 5xx (server errors)

Automatic retry improves reliability and reduces failures caused by temporary API issues.

### In-memory caching
The agent may call the same tool more than once with the same arguments during one run.
For example, it may read the same file multiple times.

Caching improves performance because repeated tool calls return instantly without doing the same work again.

## Expected improvement

- Higher reliability: fewer failed runs when the LLM API returns temporary errors
- Lower latency: repeated tool calls are faster due to cache hits
- Better efficiency: fewer repeated file reads and API requests in a single run

## Notes
These extensions do not change the external behavior of the agent. They improve robustness and speed while keeping the agent interface the same.
