"""Regression tests for agent.py using a mock HTTP server."""

import json
import os
import subprocess
import sys
import threading
import http.server
import socketserver
from pathlib import Path


# ============== Task 1 Test ==============

class MockLLMHandler_Task1(http.server.BaseHTTPRequestHandler):
    """Mock OpenAI-compatible API that returns a fixed response (no tool calls)."""

    def log_message(self, format, *args):
        pass  # Suppress logging

    def do_POST(self):
        if self.path.endswith("/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)  # Read and discard body

            response = {
                "choices": [
                    {
                        "message": {
                            "content": "Representational State Transfer.",
                            "role": "assistant",
                        }
                    }
                ]
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def find_free_port():
    """Find a free port on localhost."""
    with socketserver.TCPServer(("127.0.0.1", 0), None) as s:
        return s.server_address[1]


def run_mock_server(port, ready_event, handler_class, request_count=1):
    """Run the mock server in a background thread."""
    class CountingServer(socketserver.TCPServer):
        allow_reuse_address = True
        requests_handled = 0

        def handle_request(self):
            super().handle_request()
            self.requests_handled += 1
            if self.requests_handled >= request_count:
                self.shutdown()

    with CountingServer(("127.0.0.1", port), handler_class) as httpd:
        ready_event.set()
        httpd.serve_forever()


def test_agent_basic():
    """Test that agent.py returns valid JSON with answer and tool_calls."""
    # Find a free port for the mock server
    port = find_free_port()
    api_base = f"http://127.0.0.1:{port}"

    # Start mock server in background thread
    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=run_mock_server,
        args=(port, ready_event, MockLLMHandler_Task1, 1)
    )
    server_thread.daemon = True
    server_thread.start()

    # Wait for server to be ready
    ready_event.wait(timeout=5)

    # Run agent.py as subprocess
    agent_path = Path(__file__).parent.parent / "agent.py"
    env = os.environ.copy()
    env["LLM_API_KEY"] = "test-key"
    env["LLM_API_BASE"] = api_base
    env["LLM_MODEL"] = "test-model"

    result = subprocess.run(
        [sys.executable, str(agent_path), "What does REST stand for?"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Assert exit code is 0
    assert result.returncode == 0, f"Exit code: {result.returncode}, stderr: {result.stderr}"

    # Assert stdout is valid JSON
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON output: {result.stdout}") from e

    # Assert required fields exist
    assert "answer" in output, "Missing 'answer' field"
    assert "tool_calls" in output, "Missing 'tool_calls' field"

    # Assert tool_calls is empty array
    assert output["tool_calls"] == [], f"tool_calls should be [], got: {output['tool_calls']}"

    # Assert answer is non-empty string
    assert isinstance(output["answer"], str), f"answer should be string, got: {type(output['answer'])}"
    assert len(output["answer"]) > 0, "answer should not be empty"


# ============== Task 2 Tests ==============

class MockLLMHandler_Task2_MergeConflict(http.server.BaseHTTPRequestHandler):
    """Mock API for merge conflict test: list_files -> read_file -> answer."""

    request_count = 0

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path.endswith("/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            MockLLMHandler_Task2_MergeConflict.request_count += 1
            req_num = MockLLMHandler_Task2_MergeConflict.request_count

            if req_num == 1:
                # First call: return list_files tool call
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "list_files",
                                            "arguments": json.dumps({"path": "wiki"})
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            elif req_num == 2:
                # Second call: return read_file tool call
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": json.dumps({"path": "wiki/git-workflow.md"})
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                # Third call: return final answer
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": "To resolve a merge conflict, open the conflicted file, find the conflict markers, edit to resolve, then git add and commit.",
                                "role": "assistant"
                            }
                        }
                    ]
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class MockLLMHandler_Task2_WikiFiles(http.server.BaseHTTPRequestHandler):
    """Mock API for wiki files test: list_files -> answer."""

    request_count = 0

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path.endswith("/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)

            MockLLMHandler_Task2_WikiFiles.request_count += 1
            req_num = MockLLMHandler_Task2_WikiFiles.request_count

            if req_num == 1:
                # First call: return list_files tool call
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "list_files",
                                            "arguments": json.dumps({"path": "wiki"})
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                # Second call: return final answer
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": "The wiki contains documentation files including git-workflow.md, docker.md, python.md, and more.",
                                "role": "assistant"
                            }
                        }
                    ]
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def test_agent_merge_conflict():
    """Test agent with list_files + read_file sequence for merge conflict question."""
    MockLLMHandler_Task2_MergeConflict.request_count = 0

    port = find_free_port()
    api_base = f"http://127.0.0.1:{port}"

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=run_mock_server,
        args=(port, ready_event, MockLLMHandler_Task2_MergeConflict, 3)
    )
    server_thread.daemon = True
    server_thread.start()

    ready_event.wait(timeout=5)

    agent_path = Path(__file__).parent.parent / "agent.py"
    env = os.environ.copy()
    env["LLM_API_KEY"] = "test-key"
    env["LLM_API_BASE"] = api_base
    env["LLM_MODEL"] = "test-model"

    result = subprocess.run(
        [sys.executable, str(agent_path), "How do you resolve a merge conflict?"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, f"Exit code: {result.returncode}, stderr: {result.stderr}"

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON output: {result.stdout}") from e

    assert "answer" in output, "Missing 'answer' field"
    assert "source" in output, "Missing 'source' field"
    assert "tool_calls" in output, "Missing 'tool_calls' field"

    # Assert tool_calls is non-empty
    assert len(output["tool_calls"]) > 0, "tool_calls should not be empty"

    # Assert at least one read_file call
    tool_names = [tc.get("tool") for tc in output["tool_calls"]]
    assert "read_file" in tool_names, "Should have at least one read_file tool call"

    # Assert source contains wiki/git-workflow.md
    assert "wiki/git-workflow.md" in output["source"], f"source should contain wiki/git-workflow.md, got: {output['source']}"

    # Assert answer is non-empty string
    assert isinstance(output["answer"], str), f"answer should be string"
    assert len(output["answer"]) > 0, "answer should not be empty"


def test_agent_wiki_files():
    """Test agent with list_files only for wiki files question."""
    MockLLMHandler_Task2_WikiFiles.request_count = 0

    port = find_free_port()
    api_base = f"http://127.0.0.1:{port}"

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=run_mock_server,
        args=(port, ready_event, MockLLMHandler_Task2_WikiFiles, 2)
    )
    server_thread.daemon = True
    server_thread.start()

    ready_event.wait(timeout=5)

    agent_path = Path(__file__).parent.parent / "agent.py"
    env = os.environ.copy()
    env["LLM_API_KEY"] = "test-key"
    env["LLM_API_BASE"] = api_base
    env["LLM_MODEL"] = "test-model"

    result = subprocess.run(
        [sys.executable, str(agent_path), "What files are in the wiki?"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, f"Exit code: {result.returncode}, stderr: {result.stderr}"

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON output: {result.stdout}") from e

    assert "answer" in output, "Missing 'answer' field"
    assert "tool_calls" in output, "Missing 'tool_calls' field"

    # Assert at least one list_files call
    tool_names = [tc.get("tool") for tc in output["tool_calls"]]
    assert "list_files" in tool_names, "Should have at least one list_files tool call"

    # Assert answer is non-empty string
    assert isinstance(output["answer"], str), f"answer should be string"
    assert len(output["answer"]) > 0, "answer should not be empty"


# ============== Task 3 Tests ==============

class MockLLMHandler_Task3_BackendFramework(http.server.BaseHTTPRequestHandler):
    """Mock API for backend framework test: read_file -> answer with FastAPI."""

    request_count = 0

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path.endswith("/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)

            MockLLMHandler_Task3_BackendFramework.request_count += 1
            req_num = MockLLMHandler_Task3_BackendFramework.request_count

            if req_num == 1:
                # First call: return read_file tool call for backend
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": json.dumps({"path": "backend/app/main.py"})
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                # Second call: return final answer mentioning FastAPI
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": "The backend uses FastAPI framework.",
                                "role": "assistant"
                            }
                        }
                    ]
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class MockLLMHandler_Task3_DatabaseItems(http.server.BaseHTTPRequestHandler):
    """Mock API for database items test: query_api -> answer with numeric count."""

    request_count = 0

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path.endswith("/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)

            MockLLMHandler_Task3_DatabaseItems.request_count += 1
            req_num = MockLLMHandler_Task3_DatabaseItems.request_count

            if req_num == 1:
                # First call: return query_api tool call
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "query_api",
                                            "arguments": json.dumps({"method": "GET", "path": "/api/items"})
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                # Second call: return final answer with numeric count
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": "There are 42 items in the database.",
                                "role": "assistant"
                            }
                        }
                    ]
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class MockBackendAPIHandler(http.server.BaseHTTPRequestHandler):
    """Mock backend API that returns items count."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/api/items":
            auth_header = self.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"Unauthorized")
                return

            response = {"items": [{"id": i, "name": f"Item {i}"} for i in range(42)]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def test_agent_backend_framework():
    """Test agent with read_file for source code question - should mention FastAPI."""
    MockLLMHandler_Task3_BackendFramework.request_count = 0

    port = find_free_port()
    api_base = f"http://127.0.0.1:{port}"

    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=run_mock_server,
        args=(port, ready_event, MockLLMHandler_Task3_BackendFramework, 2)
    )
    server_thread.daemon = True
    server_thread.start()

    ready_event.wait(timeout=5)

    agent_path = Path(__file__).parent.parent / "agent.py"
    env = os.environ.copy()
    env["LLM_API_KEY"] = "test-key"
    env["LLM_API_BASE"] = api_base
    env["LLM_MODEL"] = "test-model"

    result = subprocess.run(
        [sys.executable, str(agent_path), "What framework does the backend use?"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, f"Exit code: {result.returncode}, stderr: {result.stderr}"

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON output: {result.stdout}") from e

    assert "answer" in output, "Missing 'answer' field"
    assert "tool_calls" in output, "Missing 'tool_calls' field"

    # Assert at least one read_file call
    tool_names = [tc.get("tool") for tc in output["tool_calls"]]
    assert "read_file" in tool_names, "Should have at least one read_file tool call"

    # Assert answer mentions FastAPI
    assert "FastAPI" in output["answer"], f"Answer should mention FastAPI, got: {output['answer']}"


def test_agent_database_items():
    """Test agent with query_api for live database question - should return numeric answer."""
    MockLLMHandler_Task3_DatabaseItems.request_count = 0

    # Start mock LLM server
    llm_port = find_free_port()
    api_base = f"http://127.0.0.1:{llm_port}"

    llm_ready = threading.Event()
    llm_thread = threading.Thread(
        target=run_mock_server,
        args=(llm_port, llm_ready, MockLLMHandler_Task3_DatabaseItems, 2)
    )
    llm_thread.daemon = True
    llm_thread.start()

    # Start mock backend API server
    backend_port = find_free_port()
    backend_ready = threading.Event()
    backend_thread = threading.Thread(
        target=run_mock_server,
        args=(backend_port, backend_ready, MockBackendAPIHandler, 1)
    )
    backend_thread.daemon = True
    backend_thread.start()

    llm_ready.wait(timeout=5)
    backend_ready.wait(timeout=5)

    agent_path = Path(__file__).parent.parent / "agent.py"
    env = os.environ.copy()
    env["LLM_API_KEY"] = "test-key"
    env["LLM_API_BASE"] = api_base
    env["LLM_MODEL"] = "test-model"
    env["LMS_API_KEY"] = "test-lms-key"
    env["AGENT_API_BASE_URL"] = f"http://127.0.0.1:{backend_port}"

    result = subprocess.run(
        [sys.executable, str(agent_path), "How many items are in the database?"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, f"Exit code: {result.returncode}, stderr: {result.stderr}"

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON output: {result.stdout}") from e

    assert "answer" in output, "Missing 'answer' field"
    assert "tool_calls" in output, "Missing 'tool_calls' field"

    # Assert at least one query_api call
    tool_names = [tc.get("tool") for tc in output["tool_calls"]]
    assert "query_api" in tool_names, "Should have at least one query_api tool call"

    # Assert answer contains a number
    import re
    numbers = re.findall(r'\d+', output["answer"])
    assert len(numbers) > 0, f"Answer should contain a number, got: {output['answer']}"


if __name__ == "__main__":
    test_agent_basic()
    print("Test 1 (basic) passed!")
    test_agent_merge_conflict()
    print("Test 2 (merge conflict) passed!")
    test_agent_wiki_files()
    print("Test 3 (wiki files) passed!")
    test_agent_backend_framework()
    print("Test 4 (backend framework) passed!")
    test_agent_database_items()
    print("Test 5 (database items) passed!")
    print("All tests passed!")
