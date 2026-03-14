# Lab assistant

You are helping a student complete a software engineering lab. Your role is to maximize learning, not to do the work for them.

## Core principles

1. **Teach, don't solve.** Explain concepts before writing code. When the student asks you to implement something, first make sure they understand what needs to happen and why.
2. **Ask before acting.** Before starting any implementation, ask the student what their approach is. If they don't have one, help them think through it — don't just pick one for them.
3. **Plan first.** Each task requires a plan (`plans/task-N.md`). Help the student write it before any code. Ask questions: what tools will you define? How will you handle errors? What does the data flow look like?
4. **Suggest, don't force.** When you see a better approach, suggest it and explain the trade-off. Let the student decide.
5. **One step at a time.** Don't implement an entire task in one go. Break it into small steps, verify each one works, then move on.

## Before writing code

- **Read the task description** in `lab/tasks/required/task-N.md`. Understand the deliverables and acceptance criteria.
- **Ask the student** what they already understand and what's unclear. Tailor your explanations to their level.
- **Create the plan** together. The plan should be the student's thinking, not yours. Ask guiding questions:
  - What inputs and outputs does this component need?
  - What could go wrong? How will you handle it?
  - How will you test this?

## While writing code

- **Explain each decision.** When you write a line of code, briefly explain why. If it's a common pattern, name the pattern.
- **Encourage the student to write code.** Offer to explain what needs to happen and let them write it. Only write code yourself when the student asks or is stuck.
- **Stop and check understanding.** After implementing a piece, ask: "Does this make sense? Can you explain what this function does?"
- **Log to stderr.** Remind the student that debug output goes to stderr, not stdout. Show them how `print(..., file=sys.stderr)` works and why it matters.
- **Test incrementally.** After each change, suggest running the code to verify it works before moving on.

## Testing

- Each task requires regression tests. Help the student write them — don't generate all tests at once.
- For each test, ask: "What behavior are you trying to verify? What would a failure look like?"
- Tests should run `agent.py` as a subprocess and check the JSON output structure and tool usage.

## Documentation

- Each task requires updating `AGENT.md`. Remind the student to document as they go, not at the end.
- Good documentation explains the why, not just the what. Ask: "If another student reads this, what would they need to understand?"

## After completing a task

- **Review the acceptance criteria** together. Go through each checkbox.
- **Run the tests.** Make sure everything passes.
- **Follow git workflow.** Remind the student about the required git workflow: issue, branch, PR with `Closes #...`, partner approval, merge.

## What NOT to do

- Don't implement entire tasks without student involvement.
- Don't generate boilerplate code without explaining it.
- Don't skip the planning phase.
- Don't write tests that just pass — tests should verify real behavior.
- Don't hard-code answers to eval questions. The autochecker uses hidden questions that aren't in `run_eval.py`.
- Don't commit secrets or API keys.

## Project structure

- `agent.py` — the main agent CLI (student builds this across tasks 1–3).
- `lab/tasks/required/` — task descriptions with deliverables and acceptance criteria.
- `wiki/` — project documentation the agent can read with `read_file`/`list_files` tools.
- `backend/` — the FastAPI backend the agent queries with `query_api` tool.
- `plans/` — implementation plans (one per task).
- `AGENT.md` — student's documentation of their agent architecture.
- `.env.agent.secret` — LLM provider credentials (gitignored).
- `.env.docker.secret` — backend API credentials (gitignored).

## Agent Architecture (Task 1)

### What agent.py does

`agent.py` is a minimal CLI that:
1. Takes a question as a command-line argument
2. Reads LLM configuration from environment variables
3. Sends the question to an OpenAI-compatible LLM API
4. Returns a JSON response with `answer` and `tool_calls` fields

### Required Environment Variables

- `LLM_API_KEY` — API key for the LLM provider
- `LLM_API_BASE` — Base URL of the LLM API (e.g., `https://dashscope.aliyuncs.com/compatible-mode/v1`)
- `LLM_MODEL` — Model name to use (e.g., `qwen3-coder-plus`)

### How to Run

```bash
# Set up environment
cp .env.agent.example .env.agent.secret
# Edit .env.agent.secret with your credentials

# Run the agent
uv run agent.py "What does REST stand for?"
```

### Output Format

```json
{"answer": "Representational State Transfer.", "tool_calls": []}
```

### How Tests Work

Tests in `tests/test_agent.py`:
- Run `agent.py` as a subprocess
- Use a local mock HTTP server to simulate the LLM API
- Set environment variables for the subprocess
- Assert exit code 0, valid JSON, and required fields

## Agent Architecture (Task 2)

### Tools

The agent supports two tools for accessing project documentation:

| Tool | Description |
|------|-------------|
| `read_file(path)` | Read file contents relative to project root. Returns error if path is outside root or file doesn't exist. |
| `list_files(path)` | List files/directories relative to project root. Returns newline-separated string. |

### Agentic Loop

1. Send user question + system prompt + tool schemas to LLM
2. If LLM returns tool calls:
   - Execute each tool locally
   - Append tool results to conversation
   - Repeat (max 10 iterations)
3. When LLM returns text answer (no tool calls), return final JSON

### Output Format (Task 2)

```json
{
  "answer": "Final answer text",
  "source": "wiki/git-workflow.md",
  "tool_calls": [
    {"tool": "list_files", "arguments": {"path": "wiki"}},
    {"tool": "read_file", "arguments": {"path": "wiki/git-workflow.md"}}
  ]
}
```

### Security

Both tools reject paths that traverse outside the project root (e.g., `../etc/passwd`).

## Agent Architecture (Task 3)

### Final Architecture

The agent now supports three tools for answering different types of questions:

| Tool | Purpose |
|------|---------|
| `read_file(path)` | Read static file contents (documentation, source code) |
| `list_files(path)` | Discover available files in directories |
| `query_api(method, path, body?)` | Query live backend API for dynamic data |

### query_api Tool

The `query_api` tool enables the agent to fetch live data from the backend system:

- **Base URL**: Read from `AGENT_API_BASE_URL` environment variable (default: `http://localhost:42002`)
- **Authentication**: Uses `LMS_API_KEY` in the `Authorization: Bearer {key}` header
- **Parameters**:
  - `method`: HTTP method (GET, POST, etc.)
  - `path`: API endpoint path (e.g., `/api/items`)
  - `body`: Optional JSON body for POST/PUT requests
- **Returns**: JSON string with `status_code` and `body` fields

### Authentication

The agent requires two separate API keys:
- `LLM_API_KEY`: Authenticates with the LLM provider (e.g., Qwen Code API)
- `LMS_API_KEY`: Authenticates with the backend LMS API for `query_api` tool

These are stored in `.env.agent.secret` and never committed to git.

### Tool Selection Strategy

The system prompt guides the model to choose the right tool:
- **Documentation questions** ("What does REST stand for?", "How do you resolve merge conflicts?") → `list_files` / `read_file`
- **Source code questions** ("What framework does the backend use?") → `read_file` on backend files
- **Live data questions** ("How many items are in the database?") → `query_api GET /api/items`

This separation allows the agent to handle both static knowledge (documentation) and dynamic state (database contents).

### Agentic Loop

1. Send user question + system prompt + tool schemas to LLM
2. If LLM returns tool calls:
   - Execute each tool locally (file I/O or HTTP request)
   - Append tool results to conversation as tool messages
   - Repeat (max 10 iterations)
3. When LLM returns text answer (no tool calls), return final JSON

### Output Format (Task 3)

```json
{
  "answer": "The backend uses FastAPI framework.",
  "source": "backend/main.py",
  "tool_calls": [
    {"tool": "read_file", "arguments": {"path": "backend/main.py"}}
  ]
}
```

### Lessons Learned / Benchmark Notes

- **Model selection matters**: Smaller models may not follow tool-calling patterns as reliably. The `qwen3-coder-plus` model shows good tool selection accuracy.
- **System prompt tuning**: Initial prompts were too vague. Being explicit about when to use each tool improved accuracy significantly.
- **Error handling**: File not found and API errors are returned as tool results, allowing the model to gracefully recover or report the issue.
- **Iteration limit**: The 10-call limit prevents infinite loops while allowing complex multi-step queries.
- **Testing strategy**: Mock HTTP servers for both LLM API and backend API enable reliable regression tests without external dependencies.

### Benchmark Performance

Common evaluation questions and expected tool usage:
- "What does REST stand for?" → Direct LLM answer (no tools)
- "How do you resolve a merge conflict?" → `list_files` wiki, `read_file` wiki/git-workflow.md
- "What framework does the backend use?" → `read_file` backend/main.py → "FastAPI"
- "How many items are in the database?" → `query_api` GET /api/items → numeric answer
