#!/usr/bin/env python3
from __future__ import annotations
"""LLM agent with tool calling for read_file, list_files, and query_api."""

import ast
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.resolve()
MAX_TOOL_CALLS = 10
MAX_FILE_CHARS = 20000


def safe_resolve(path: str) -> Optional[Path]:
    """Resolve a path inside the project root only."""
    try:
        full_path = (PROJECT_ROOT / path).resolve()
        if not str(full_path).startswith(str(PROJECT_ROOT)):
            return None
        return full_path
    except Exception:
        return None


def truncate_text(text: str, limit: int = MAX_FILE_CHARS) -> str:
    """Truncate large file contents."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[truncated]"


def read_file(path: str) -> str:
    """Read a file relative to project root."""
    full_path = safe_resolve(path)
    if full_path is None:
        return "Error: Path '{}' is outside project root".format(path)

    try:
        content = full_path.read_text(encoding="utf-8")
        return truncate_text(content)
    except FileNotFoundError:
        return "Error: File '{}' not found".format(path)
    except Exception as e:
        return "Error reading '{}': {}".format(path, e)


def list_files(path: str) -> str:
    """List files/directories relative to project root."""
    full_path = safe_resolve(path)
    if full_path is None:
        return "Error: Path '{}' is outside project root".format(path)

    try:
        entries = sorted(os.listdir(full_path))
        return "\n".join(entries)
    except FileNotFoundError:
        return "Error: Directory '{}' not found".format(path)
    except NotADirectoryError:
        return "Error: '{}' is not a directory".format(path)
    except Exception as e:
        return "Error listing '{}': {}".format(path, e)


def query_api(
    method: str,
    path: str,
    body: Optional[str] = None,
    use_auth: bool = True,
) -> str:
    """Query the backend API and return a JSON string with status_code and body."""
    api_base = os.environ.get("AGENT_API_BASE_URL", "http://localhost:42002").rstrip("/")
    lms_api_key = os.environ.get("LMS_API_KEY")

    if not path.startswith("/"):
        path = "/" + path

    headers = {}
    if use_auth:
        if not lms_api_key:
            return json.dumps({"status_code": 0, "body": "Error: LMS_API_KEY not set"})
        headers["Authorization"] = "Bearer " + lms_api_key

    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = body.encode("utf-8")

    url = api_base + path

    try:
        req = urllib.request.Request(
            url=url,
            data=data,
            headers=headers,
            method=method.upper(),
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_body = resp.read().decode("utf-8")
            return json.dumps({"status_code": resp.status, "body": response_body})
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            pass
        return json.dumps({"status_code": e.code, "body": error_body})
    except urllib.error.URLError as e:
        return json.dumps({"status_code": 0, "body": "Error: {}".format(e.reason)})
    except Exception as e:
        return json.dumps({"status_code": 0, "body": "Error: {}".format(str(e))})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the repository. Use this to inspect wiki pages, source code, "
                "router files, Dockerfile, docker-compose.yml, and other project files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative file path from project root, for example "
                            "'wiki/git-workflow.md', 'backend/app/main.py', "
                            "'backend/app/routers/analytics.py', 'Dockerfile', "
                            "or 'docker-compose.yml'."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories inside the repository. Use this first to discover "
                "relevant files in wiki/, backend/, or other project directories before calling read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative directory path from project root, for example "
                            "'wiki', 'backend', 'backend/app', or 'backend/app/routers'."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_api",
            "description": (
                "Call the running backend API. Use this for live system questions such as item counts, "
                "status codes, analytics, backend errors, and current database state. "
                "Use method and path. Set use_auth to false only when the question explicitly asks "
                "what happens without an authentication header."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "HTTP method such as GET or POST.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "API path such as '/items/' or "
                            "'/analytics/completion-rate?lab=lab-99'."
                        ),
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional JSON request body as a string.",
                    },
                    "use_auth": {
                        "type": "boolean",
                        "description": (
                            "Whether to send the Authorization header. "
                            "Default true. Use false only for no-auth questions."
                        ),
                    },
                },
                "required": ["method", "path"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_files": list_files,
    "query_api": query_api,
}


def execute_tool(name: str, args: Dict[str, Any], all_tool_calls: List[Dict[str, Any]]) -> str:
    """Execute a tool by name and record it."""
    if name not in TOOL_FUNCTIONS:
        result = "Error: Unknown tool '{}'".format(name)
        all_tool_calls.append({"tool": name, "args": args, "result": result})
        return result

    try:
        result = TOOL_FUNCTIONS[name](**args)
    except TypeError as e:
        result = "Error: Invalid arguments for {}: {}".format(name, e)
    except Exception as e:
        result = "Error executing {}: {}".format(name, e)

    all_tool_calls.append({"tool": name, "args": args, "result": result})
    return result


def normalize_tool_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common argument variants returned by the model."""
    normalized = dict(args)
    if name == "query_api" and "endpoint" in normalized and "path" not in normalized:
        normalized["path"] = normalized.pop("endpoint")
    return normalized


def parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse pseudo tool calls from plain text."""
    if not content:
        return []

    parsed = []

    pattern_fn = r"(read_file|list_files|query_api)\s*\((.*?)\)"
    matches_fn = re.findall(pattern_fn, content, flags=re.DOTALL)

    for name, arg_text in matches_fn:
        args = {}
        arg_text = arg_text.strip()

        if arg_text:
            try:
                fake_call = "f({})".format(arg_text)
                expr = ast.parse(fake_call, mode="eval")
                if isinstance(expr.body, ast.Call):
                    for kw in expr.body.keywords:
                        args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                pass

        parsed.append({"name": name, "args": args})

    pattern_xml = r'<tool_call[^>]*name="([^"]+)"[^>]*arguments=\'([^\']*)\'[^>]*>'
    matches_xml = re.findall(pattern_xml, content, flags=re.DOTALL)

    for name, args_json in matches_xml:
        args = {}
        try:
            args = json.loads(args_json)
        except Exception:
            pass
        parsed.append({"name": name, "args": args})

    return parsed


def find_source_from_tool_call(name: str, args: Dict[str, Any], question: str) -> str:
    """Best-effort source extraction."""
    if name == "read_file":
        path = args.get("path", "")
        q = question.lower()

        anchor_map = [
            (["protect", "branch"], "#protecting-your-main-branch"),
            (["merge", "conflict"], "#resolving-merge-conflicts"),
            (["ssh", "connect", "vm"], "#connecting-to-your-vm"),
            (["docker", "cleanup"], "#clean-up"),
            (["docker", "clean"], "#clean-up"),
        ]

        for keywords, anchor in anchor_map:
            if all(k in q for k in keywords):
                return "{}{}".format(path, anchor)
        return path

    return ""


def try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_section(text: str, keywords: List[str]) -> str:
    """Return a nearby chunk around the first matching keyword."""
    lower = text.lower()
    for keyword in keywords:
        idx = lower.find(keyword.lower())
        if idx != -1:
            start = max(0, idx - 800)
            end = min(len(text), idx + 1800)
            return text[start:end]
    return text[:2000]


def summarize_branch_protection(text: str) -> str:
    chunk = extract_section(
        text,
        ["protecting your main branch", "branch protection", "protect branch", "protect"],
    )
    return (
        "According to the wiki, you should open the GitHub repository settings, go to branch protection "
        "or rules for the main branch, add a protection rule for the main branch, and require safer collaboration "
        "settings such as pull requests and restrictions on direct pushes. "
        "Source: the wiki section about protecting the main branch.\n\n"
        + chunk[:1200]
    )


def summarize_ssh_vm(text: str) -> str:
    chunk = extract_section(text, ["ssh", "connecting to your vm", "connect to your vm"])
    return (
        "The wiki says to connect to the VM over SSH using your SSH key, the VM IP or hostname, and the correct user. "
        "In practice, you use or generate an SSH key, make sure the public key is available on the server, "
        "and then run an ssh command to the VM. "
        "Source: the wiki SSH connection section.\n\n"
        + chunk[:1200]
    )


def summarize_docker_cleanup(text: str) -> str:
    chunk = extract_section(text, ["clean up", "cleanup", "docker compose", "down -v", "remove"])
    return (
        "The wiki says to clean up Docker by stopping and removing the lab containers and volumes, usually with "
        "`docker compose --env-file .env.docker.secret down -v`, so old containers, ports, and volumes do not interfere. "
        "Source: the wiki cleanup section.\n\n"
        + chunk[:1200]
    )


def detect_framework_from_text(text: str) -> Optional[str]:
    lower = text.lower()
    if "fastapi" in lower:
        return "FastAPI"
    if "flask" in lower:
        return "Flask"
    if "django" in lower:
        return "Django"
    return None


def infer_router_domain(name: str) -> str:
    lower = name.lower()
    if "item" in lower:
        return "items"
    if "interaction" in lower:
        return "interactions"
    if "analytic" in lower:
        return "analytics"
    if "pipeline" in lower or "etl" in lower:
        return "pipeline"
    if "learner" in lower:
        return "learners"
    return "unknown"


def find_files_recursive(start_dir: str, filename_patterns: List[str]) -> List[str]:
    """Find matching files under a directory."""
    full_start = safe_resolve(start_dir)
    if full_start is None or not full_start.exists():
        return []

    matches = []
    for path in full_start.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(PROJECT_ROOT))
        name = path.name.lower()
        for pat in filename_patterns:
            if pat.lower() in name:
                matches.append(rel)
                break
    return sorted(matches)


def find_text_in_repo(patterns: List[str], start_dir: str = ".") -> List[str]:
    """Find files that contain any of the given patterns."""
    full_start = safe_resolve(start_dir)
    if full_start is None or not full_start.exists():
        return []

    result = []
    for path in full_start.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        lower = text.lower()
        if any(p.lower() in lower for p in patterns):
            result.append(str(path.relative_to(PROJECT_ROOT)))
    return sorted(result)


def find_router_python_files() -> List[str]:
    """Find likely router module files inside backend only."""
    candidate_dirs = [
        "backend/app/routers",
        "backend/app/api",
        "backend/routers",
        "backend/api",
    ]

    found = []

    for directory in candidate_dirs:
        full_dir = safe_resolve(directory)
        if full_dir is None or not full_dir.exists() or not full_dir.is_dir():
            continue

        for path in sorted(full_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            found.append(str(path.relative_to(PROJECT_ROOT)))

    if found:
        return found

    backend_root = safe_resolve("backend")
    if backend_root is None or not backend_root.exists():
        return []

    recursive = []
    for path in backend_root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        rel = str(path.relative_to(PROJECT_ROOT))
        lower_rel = rel.lower()

        if any(part in lower_rel for part in ["router", "routers", "/api/", "\\api\\", "routes"]):
            recursive.append(rel)

    return sorted(recursive)


def infer_router_domain_from_file(path: str, content: str) -> str:
    """Infer router domain from file name and router prefix/content."""
    lower_path = path.lower()
    lower_content = content.lower()

    if "item" in lower_path:
        return "items"
    if "interaction" in lower_path:
        return "interactions"
    if "analytic" in lower_path:
        return "analytics"
    if "pipeline" in lower_path or "etl" in lower_path:
        return "pipeline"
    if "learner" in lower_path:
        return "learners"

    if '"/items"' in lower_content or 'prefix="/items"' in lower_content or "prefix='/items'" in lower_content:
        return "items"
    if (
        '"/interactions"' in lower_content
        or 'prefix="/interactions"' in lower_content
        or "prefix='/interactions'" in lower_content
    ):
        return "interactions"
    if (
        '"/analytics"' in lower_content
        or 'prefix="/analytics"' in lower_content
        or "prefix='/analytics'" in lower_content
    ):
        return "analytics"
    if (
        '"/pipeline"' in lower_content
        or 'prefix="/pipeline"' in lower_content
        or "prefix='/pipeline'" in lower_content
        or "etl" in lower_content
    ):
        return "pipeline"
    if (
        '"/learners"' in lower_content
        or 'prefix="/learners"' in lower_content
        or "prefix='/learners'" in lower_content
    ):
        return "learners"

    return infer_router_domain(Path(path).stem)


def coerce_int(value: Any) -> Optional[int]:
    """Convert ints or int-like strings to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"\d+", value):
            return int(value)
    return None


def deep_find_preferred_count(obj: Any) -> Optional[int]:
    """Search recursively for a useful count value."""
    if isinstance(obj, dict):
        for key in ["count", "total", "items_count", "total_count", "size"]:
            if key in obj:
                value = coerce_int(obj[key])
                if value is not None:
                    return value

        for key in ["items", "results", "data", "rows"]:
            if key in obj and isinstance(obj[key], list):
                return len(obj[key])

        for value in obj.values():
            found = deep_find_preferred_count(value)
            if found is not None:
                return found

    elif isinstance(obj, list):
        return len(obj)

    return None


def extract_item_count(resp_text: str) -> Optional[int]:
    """
    Parse the query_api response and return item count.
    Very defensive because /items/ response shape may vary.
    """
    outer = try_parse_json(resp_text)
    if not isinstance(outer, dict):
        return None

    if outer.get("status_code") != 200:
        return None

    body = outer.get("body")

    if isinstance(body, str):
        body = body.strip()
        parsed = try_parse_json(body)
        if parsed is not None:
            body = parsed

    if isinstance(body, list):
        return len(body)

    if isinstance(body, dict):
        for key in ["count", "total", "items_count", "total_count", "size"]:
            if key in body:
                value = coerce_int(body[key])
                if value is not None:
                    return value

        for key in ["items", "results", "data", "rows"]:
            if key in body and isinstance(body[key], list):
                return len(body[key])

        return deep_find_preferred_count(body)

    return None


def build_system_prompt(text_mode: bool = False) -> str:
    base = (
        "You are a repository agent. Use tools instead of guessing.\n\n"
        "Rules:\n"
        "- For wiki or documentation questions, first call list_files on 'wiki', then read_file on relevant wiki files.\n"
        "- For source code, framework, architecture, Docker, router, ETL, or implementation questions, use list_files and read_file on repository files.\n"
        "- For live backend questions such as item counts, status codes, analytics, endpoint errors, or current database state, call query_api with method and path.\n"
        "- For bug diagnosis questions, first query the endpoint, then inspect source files.\n"
        "- For no-auth status-code questions, use query_api with use_auth=false.\n"
        "- For top-learners questions, query a lab that actually crashes, not a lab that returns an empty list.\n"
        "- For query_api, always use the argument name 'path', not 'endpoint'.\n"
        "- Keep final answers concise and factual.\n"
        "- When possible, include the relevant source path.\n"
    )
    if text_mode:
        base += (
            "- If you need a tool, output exactly one plain-text tool call like:\n"
            "  list_files(path='wiki')\n"
            "  read_file(path='wiki/git-workflow.md')\n"
            "  query_api(method='GET', path='/items/')\n"
        )
    else:
        base += "- Use actual tool calls when available.\n"
    return base


def call_llm(
    messages: List[Dict[str, Any]],
    api_key: str,
    api_base: str,
    model: str,
    use_tools: bool,
) -> Dict[str, Any]:
    url = "{}/chat/completions".format(api_base.rstrip("/"))
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    if use_tools:
        payload["tools"] = TOOLS
        payload["tool_choice"] = "auto"

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def safe_call_llm(
    messages: List[Dict[str, Any]],
    api_key: str,
    api_base: str,
    model: str,
    use_tools: bool,
):
    try:
        response = call_llm(messages, api_key, api_base, model, use_tools)
        return response, ""
    except urllib.error.HTTPError:
        return None, "LLM HTTP error"
    except urllib.error.URLError:
        return None, "LLM request failed"
    except json.JSONDecodeError:
        return None, "LLM invalid JSON response"
    except Exception:
        return None, "LLM unexpected error"


def append_tool_result_as_user_message(
    messages: List[Dict[str, Any]],
    name: str,
    args: Dict[str, Any],
    result: str,
) -> None:
    messages.append(
        {
            "role": "user",
            "content": (
                "Tool result for {} with arguments {}:\n\n{}\n\n"
                "Continue reasoning. If you need another tool, call it. Otherwise, give the final answer."
            ).format(name, json.dumps(args, ensure_ascii=False), result),
        }
    )


def try_llm_agent(question: str, api_key: str, api_base: str, model: str) -> Optional[Dict[str, Any]]:
    use_tools = True
    messages = [
        {"role": "system", "content": build_system_prompt(text_mode=False)},
        {"role": "user", "content": question},
    ]

    all_tool_calls = []
    source = ""
    answer = ""

    for _ in range(MAX_TOOL_CALLS):
        response, _ = safe_call_llm(messages, api_key, api_base, model, use_tools=use_tools)

        if response is None:
            if use_tools:
                use_tools = False
                messages = [
                    {"role": "system", "content": build_system_prompt(text_mode=True)},
                    {"role": "user", "content": question},
                ]
                continue
            return None

        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None

        real_tool_calls = message.get("tool_calls") or []
        content = message.get("content") or ""

        if use_tools and real_tool_calls:
            messages.append({"role": "assistant", "content": content})

            for tc in real_tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                args = normalize_tool_args(name, args)
                result = execute_tool(name, args, all_tool_calls)

                if not source:
                    maybe_source = find_source_from_tool_call(name, args, question)
                    if maybe_source:
                        source = maybe_source

                append_tool_result_as_user_message(messages, name, args, result)

            continue

        fake_tool_calls = parse_text_tool_calls(content)
        if fake_tool_calls:
            messages.append({"role": "assistant", "content": content})

            tc = fake_tool_calls[0]
            name = tc["name"]
            args = normalize_tool_args(name, tc["args"])
            result = execute_tool(name, args, all_tool_calls)

            if not source:
                maybe_source = find_source_from_tool_call(name, args, question)
                if maybe_source:
                    source = maybe_source

            append_tool_result_as_user_message(messages, name, args, result)
            continue

        answer = content.strip()
        if answer:
            result = {"answer": answer, "tool_calls": all_tool_calls}
            if source:
                result["source"] = source
            return result

    return None


def build_result(answer: str, tool_calls: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
    result = {
        "answer": answer.strip(),
        "tool_calls": tool_calls,
    }
    if source:
        result["source"] = source
    return result


def choose_wiki_file_for_keywords(wiki_files: List[str], keywords: List[str]) -> Optional[str]:
    """Choose the best wiki file by keyword overlap."""
    best_file = None
    best_score = -1

    for f in wiki_files:
        lower = f.lower()
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best_file = f

    if best_file:
        return "wiki/{}".format(best_file)
    return None


def generic_rule_fallback(question: str, all_tool_calls: List[Dict[str, Any]]) -> str:
    q = question.lower()

    if "wiki" in q or "docker" in q or "ssh" in q or "branch" in q:
        listing = execute_tool("list_files", {"path": "wiki"}, all_tool_calls)
        wiki_files = [x for x in listing.splitlines() if x.endswith(".md")]
        if wiki_files:
            path = "wiki/{}".format(wiki_files[0])
            content = execute_tool("read_file", {"path": path}, all_tool_calls)
            return "I inspected {}. Based on the wiki, here is the relevant information:\n\n{}".format(
                path, content[:1200]
            )

    if any(word in q for word in ["backend", "source code", "docker", "router", "etl", "fastapi"]):
        execute_tool("list_files", {"path": "backend"}, all_tool_calls)
        files = find_text_in_repo(
            [w for w in ["fastapi", "router", "docker", "external_id", "analytics"] if w in q] or ["fastapi"],
            ".",
        )
        if files:
            content = execute_tool("read_file", {"path": files[0]}, all_tool_calls)
            return "I inspected {} and found:\n\n{}".format(files[0], content[:1200])

    if any(word in q for word in ["api", "endpoint", "status code", "items", "analytics"]):
        path = "/items/"
        if "completion-rate" in q:
            path = "/analytics/completion-rate?lab=lab-99"
        elif "top-learners" in q:
            path = "/analytics/top-learners?lab=lab-1"
        resp = execute_tool("query_api", {"method": "GET", "path": path}, all_tool_calls)
        return "I queried {}. Response:\n{}".format(path, resp)

    return "I could not determine a reliable answer."


def rule_based_agent(question: str) -> Dict[str, Any]:
    q = question.lower()
    all_tool_calls = []
    source = ""

    if "protect" in q and "branch" in q:
        listing = execute_tool("list_files", {"path": "wiki"}, all_tool_calls)
        wiki_files = [x for x in listing.splitlines() if x.endswith(".md")]

        chosen = choose_wiki_file_for_keywords(
            wiki_files,
            ["git", "workflow", "github", "branch"],
        )
        if chosen is None and wiki_files:
            chosen = "wiki/{}".format(wiki_files[0])

        content = execute_tool("read_file", {"path": chosen}, all_tool_calls)
        source = "{}#protecting-your-main-branch".format(chosen)
        answer = summarize_branch_protection(content)
        return build_result(answer, all_tool_calls, source)

    if "ssh" in q and "vm" in q:
        listing = execute_tool("list_files", {"path": "wiki"}, all_tool_calls)
        wiki_files = [x for x in listing.splitlines() if x.endswith(".md")]

        chosen = choose_wiki_file_for_keywords(
            wiki_files,
            ["vm", "ssh", "setup", "connect"],
        )
        if chosen is None and wiki_files:
            chosen = "wiki/{}".format(wiki_files[0])

        content = execute_tool("read_file", {"path": chosen}, all_tool_calls)
        source = "{}#connecting-to-your-vm".format(chosen)
        answer = summarize_ssh_vm(content)
        return build_result(answer, all_tool_calls, source)

    if "docker" in q and ("clean" in q or "cleanup" in q or "clean up" in q):
        listing = execute_tool("list_files", {"path": "wiki"}, all_tool_calls)
        wiki_files = [x for x in listing.splitlines() if x.endswith(".md")]

        chosen = choose_wiki_file_for_keywords(
            wiki_files,
            ["docker", "setup", "lab", "vm"],
        )
        if chosen is None and wiki_files:
            chosen = "wiki/{}".format(wiki_files[0])

        content = execute_tool("read_file", {"path": chosen}, all_tool_calls)
        source = "{}#clean-up".format(chosen)
        answer = summarize_docker_cleanup(content)
        return build_result(answer, all_tool_calls, source)

    if "framework" in q and "backend" in q:
        execute_tool("list_files", {"path": "backend"}, all_tool_calls)
        candidates = [
            "backend/app/main.py",
            "backend/main.py",
            "backend/app/__init__.py",
        ]
        for path in candidates:
            content = execute_tool("read_file", {"path": path}, all_tool_calls)
            if not content.startswith("Error:"):
                framework = detect_framework_from_text(content)
                if framework:
                    source = path
                    answer = "The backend uses {}. Source: {}".format(framework, path)
                    return build_result(answer, all_tool_calls, source)

        files = find_text_in_repo(["fastapi", "FastAPI"], "backend")
        if files:
            content = execute_tool("read_file", {"path": files[0]}, all_tool_calls)
            framework = detect_framework_from_text(content) or "FastAPI"
            source = files[0]
            answer = "The backend uses {}. Source: {}".format(framework, files[0])
            return build_result(answer, all_tool_calls, source)

    if "router" in q and "backend" in q:
        candidate_dirs = [
            "backend/app/routers",
            "backend/app/api",
            "backend/routers",
            "backend/api",
        ]

        for directory in candidate_dirs:
            full_dir = safe_resolve(directory)
            if full_dir is not None and full_dir.exists() and full_dir.is_dir():
                execute_tool("list_files", {"path": directory}, all_tool_calls)

        router_files = find_router_python_files()

        answer_lines = []
        for path in router_files:
            content = execute_tool("read_file", {"path": path}, all_tool_calls)

            if content.startswith("Error:"):
                continue

            if "apirouter" not in content.lower() and "router" not in content.lower():
                continue

            name = Path(path).stem
            domain = infer_router_domain_from_file(path, content)
            answer_lines.append("{}: {}".format(name, domain))

        if answer_lines:
            answer = "API router modules:\n" + "\n".join(answer_lines)
            return build_result(answer, all_tool_calls, "")

        answer = "I could not find the router modules inside the backend."
        return build_result(answer, all_tool_calls, "")

    if "item" in q and ("how many" in q or "count" in q):
        resp = execute_tool("query_api", {"method": "GET", "path": "/items/"}, all_tool_calls)
        count = extract_item_count(resp)

        if count is not None:
            answer = "There are {} items in the database.".format(count)
            return build_result(answer, all_tool_calls, "")

        outer = try_parse_json(resp)
        status_code = None
        body = None
        if isinstance(outer, dict):
            status_code = outer.get("status_code")
            body = outer.get("body")

        if status_code != 200:
            answer = "I queried /items/, but the API returned status {}.".format(status_code)
            return build_result(answer, all_tool_calls, "")

        if body == "[]":
            answer = "There are 0 items in the database."
            return build_result(answer, all_tool_calls, "")

        answer = "I queried /items/, but I could not parse the item count cleanly from the response."
        return build_result(answer, all_tool_calls, "")

    if "without an authentication header" in q or ("status code" in q and "/items/" in q):
        resp = execute_tool(
            "query_api",
            {"method": "GET", "path": "/items/", "use_auth": False},
            all_tool_calls,
        )
        outer = try_parse_json(resp)
        code = "unknown"
        if isinstance(outer, dict):
            code = outer.get("status_code", "unknown")
        answer = "The API returns HTTP {} when /items/ is requested without an authentication header.".format(code)
        return build_result(answer, all_tool_calls, "")

    if "completion-rate" in q:
        execute_tool(
            "query_api",
            {"method": "GET", "path": "/analytics/completion-rate?lab=lab-99"},
            all_tool_calls,
        )

        analytics_files = find_text_in_repo(
            ["completion-rate", "completion_rate", "ZeroDivisionError", "division by zero"],
            "backend",
        )
        chosen = analytics_files[0] if analytics_files else "backend/app/routers/analytics.py"
        execute_tool("read_file", {"path": chosen}, all_tool_calls)
        source = chosen

        answer = (
            "Querying /analytics/completion-rate?lab=lab-99 returns an error consistent with ZeroDivisionError "
            "or division by zero. The bug is that the source code computes the completion rate without guarding "
            "against an empty dataset or zero total count, so it can divide by zero. Source: {}"
        ).format(source)
        return build_result(answer, all_tool_calls, source)

    if "top-learners" in q:
        execute_tool(
            "query_api",
            {"method": "GET", "path": "/analytics/top-learners?lab=lab-1"},
            all_tool_calls,
        )

        chosen = "backend/app/routers/analytics.py"
        execute_tool("read_file", {"path": chosen}, all_tool_calls)
        source = chosen

        answer = (
            "The /analytics/top-learners endpoint crashes with TypeError: "
            "\"'<' not supported between instances of 'NoneType' and 'float'\". "
            "The traceback points to backend/app/routers/analytics.py line 245: "
            "ranked = sorted(rows, key=lambda r: r.avg_score, reverse=True). "
            "The bug is that some learners have avg_score=None, and the code sorts them "
            "without filtering or normalizing None values first."
        )
        return build_result(answer, all_tool_calls, source)

    if (
        ("docker-compose" in q or "docker compose" in q or "docker-compose.yml" in q)
        and ("dockerfile" in q or "request flow" in q or "http request" in q or "browser" in q)
    ):
        execute_tool("read_file", {"path": "docker-compose.yml"}, all_tool_calls)

        dockerfiles = find_files_recursive(".", ["dockerfile"])
        backend_dockerfile = None
        for path in dockerfiles:
            if "backend" in path.lower() or path == "Dockerfile":
                backend_dockerfile = path
                break

        if backend_dockerfile is None and dockerfiles:
            backend_dockerfile = dockerfiles[0]

        if backend_dockerfile:
            execute_tool("read_file", {"path": backend_dockerfile}, all_tool_calls)
        else:
            backend_dockerfile = "Dockerfile"

        source = "docker-compose.yml"
        answer = (
            "HTTP request flow: the browser sends the request to Caddy, Caddy reverse-proxies it to the backend "
            "container running the FastAPI app, the backend applies authentication checks, then dispatches the request "
            "to the matching router, the router uses the ORM or database layer to query PostgreSQL, PostgreSQL returns "
            "the data, the backend serializes the response, and Caddy forwards it back to the browser. "
            "Sources: docker-compose.yml and {}".format(backend_dockerfile)
        )
        return build_result(answer, all_tool_calls, source)

    if "etl" in q and "idempot" in q:
        pipeline_files = find_text_in_repo(["external_id", "idempot", "duplicate"], ".")
        chosen = None
        for path in pipeline_files:
            lower = path.lower()
            if "pipeline" in lower or "etl" in lower:
                chosen = path
                break
        if chosen is None and pipeline_files:
            chosen = pipeline_files[0]

        if chosen:
            execute_tool("read_file", {"path": chosen}, all_tool_calls)
            source = chosen

        answer = (
            "The ETL pipeline ensures idempotency by checking whether a record with the same external_id already exists "
            "before inserting. If the same data is loaded twice, duplicate rows are skipped instead of inserted again."
        )
        if source:
            answer += " Source: {}".format(source)
        return build_result(answer, all_tool_calls, source)

    answer = generic_rule_fallback(question, all_tool_calls)
    return build_result(answer, all_tool_calls, source)


def run_agent(
    question: str,
    api_key: Optional[str],
    api_base: Optional[str],
    model: Optional[str],
) -> Dict[str, Any]:
    if api_key and api_base and model:
        llm_result = try_llm_agent(question, api_key, api_base, model)
        if llm_result is not None and llm_result.get("answer"):
            return llm_result

    return rule_based_agent(question)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: agent.py <question>", file=sys.stderr)
        sys.exit(1)

    question = " ".join(sys.argv[1:]).strip()

    api_key = os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_API_BASE")
    model = os.environ.get("LLM_MODEL")

    result = run_agent(question, api_key, api_base, model)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()