"""
Testbed Service B - API / Transform Layer

Receives requests from Service A, transforms/forwards the payload (including
any taint tokens) to Service C via HTTP POST. This simulates a typical API
gateway or middleware that doesn't sanitize user input before passing it
downstream.

The taint token (TAINT=<uuid>;) passes through B untouched - this is the
intentional vulnerability we're testing cross-service taint tracking for.

Runs on port 8080.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

SERVICE_C_URL = os.environ.get("SERVICE_C_URL", "http://service-c:9090/db/query")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8080"))


class APIHandler(BaseHTTPRequestHandler):
    """Handles incoming requests and forwards to Service C."""

    def log_message(self, format: str, *args: Any) -> None:
        """Route access logs to stdout for docker logs visibility."""
        print(f"[B] {format % args}", flush=True)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response(200, {"status": "ok", "service": "B"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/api/process":
            self._handle_process()
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_process(self) -> None:
        """
        Receive payload from A, build a 'query' containing the user_input
        (which carries the taint token), and forward to C.
        """
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            self._json_response(400, {"error": f"bad JSON: {exc}"})
            return

        user_input = payload.get("user_input", "")
        action = payload.get("action", "unknown")

        print(f"[B] Received from A: action={action} user_input={user_input!r}", flush=True)

        # Build the downstream "query" - taint propagates here because
        # user_input is embedded directly into the query string.
        query = f"SELECT * FROM records WHERE data = '{user_input}' AND action = '{action}'"
        downstream_payload = json.dumps({
            "query": query,
            "raw_input": user_input,
            "forwarded_by": "service-b",
        }).encode("utf-8")

        print(f"[B] Forwarding to C: {SERVICE_C_URL}", flush=True)

        # Forward to Service C
        try:
            req = Request(
                SERVICE_C_URL,
                data=downstream_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10.0) as resp:
                c_response = json.loads(resp.read().decode("utf-8"))
        except URLError as exc:
            print(f"[B] ERROR forwarding to C: {exc}", flush=True)
            self._json_response(502, {
                "error": f"Service C unreachable: {exc}",
                "query_sent": query,
            })
            return
        except Exception as exc:
            print(f"[B] ERROR unexpected: {exc}", flush=True)
            self._json_response(500, {"error": str(exc)})
            return

        print(f"[B] Response from C: {json.dumps(c_response)}", flush=True)

        # Return combined response to A
        self._json_response(200, {
            "status": "processed",
            "service": "B",
            "query_sent_to_c": query,
            "c_response": c_response,
            "original_input": user_input,
        })

    def _json_response(self, code: int, data: Any) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    print(f"[B] API service starting on port {LISTEN_PORT}", flush=True)
    print(f"[B] Forwarding to Service C at {SERVICE_C_URL}", flush=True)
    server = HTTPServer(("0.0.0.0", LISTEN_PORT), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[B] Shutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
