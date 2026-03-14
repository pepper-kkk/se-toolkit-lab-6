"""Regression test for agent.py using a mock HTTP server."""

import json
import os
import subprocess
import sys
import threading
import http.server
import socketserver
from pathlib import Path


class MockLLMHandler(http.server.BaseHTTPRequestHandler):
    """Mock OpenAI-compatible API that returns a fixed response."""

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
    with socketserver.TCPServer(("127.0.0.1", 0), MockLLMHandler) as s:
        return s.server_address[1]


def run_mock_server(port, ready_event):
    """Run the mock server in a background thread."""
    with socketserver.TCPServer(("127.0.0.1", port), MockLLMHandler) as httpd:
        ready_event.set()
        httpd.handle_request()  # Handle one request then stop


def test_agent_basic():
    """Test that agent.py returns valid JSON with answer and tool_calls."""
    # Find a free port for the mock server
    port = find_free_port()
    api_base = f"http://127.0.0.1:{port}"

    # Start mock server in background thread
    ready_event = threading.Event()
    server_thread = threading.Thread(target=run_mock_server, args=(port, ready_event))
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


if __name__ == "__main__":
    test_agent_basic()
    print("Test passed!")
