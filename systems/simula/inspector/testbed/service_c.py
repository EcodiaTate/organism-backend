"""
Testbed Service C - DB / Sink Layer

Receives "query" payloads from Service B and logs them. Simulates a database
service that executes whatever query string it receives - the taint token
(TAINT=<uuid>;) is visible here if it propagated through the chain.

This is the terminal sink: if the taint token appears in C's logs, the
cross-service taint propagation test passes.

Runs on port 9090.
"""

from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "9090"))

# Pattern to detect our taint tokens in any string
TAINT_PATTERN = re.compile(r"TAINT=[0-9a-f\-]{36};")

# In-memory log of all received queries (for /db/log endpoint)
_query_log: list[dict[str, Any]] = []


class DBHandler(BaseHTTPRequestHandler):
    """Simulates a database that logs all received queries."""

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[C] {format % args}", flush=True)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response(200, {"status": "ok", "service": "C"})
        elif self.path == "/db/log":
            self._json_response(200, {"queries": _query_log, "total": len(_query_log)})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/db/query":
            self._handle_query()
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_query(self) -> None:
        """Receive a query from Service B and 'execute' it (log + echo)."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            self._json_response(400, {"error": f"bad JSON: {exc}"})
            return

        query = payload.get("query", "")
        raw_input = payload.get("raw_input", "")
        forwarded_by = payload.get("forwarded_by", "unknown")

        # Detect taint tokens
        taint_matches = TAINT_PATTERN.findall(query)
        taint_in_raw = TAINT_PATTERN.findall(raw_input)
        all_taints = list(set(taint_matches + taint_in_raw))

        tainted = len(all_taints) > 0

        # Log the query
        entry = {
            "query": query,
            "raw_input": raw_input,
            "forwarded_by": forwarded_by,
            "tainted": tainted,
            "taint_tokens": all_taints,
        }
        _query_log.append(entry)

        print(f"[C] QUERY RECEIVED: {query!r}", flush=True)
        print(f"[C]   forwarded_by: {forwarded_by}", flush=True)
        print(f"[C]   raw_input:    {raw_input!r}", flush=True)

        if tainted:
            print(f"[C]   TAINT DETECTED: {all_taints}", flush=True)
        else:
            print("[C]   no taint detected", flush=True)

        # Simulate query execution - echo back as "results"
        self._json_response(200, {
            "status": "executed",
            "service": "C",
            "rows_affected": 0,
            "query_echo": query,
            "tainted": tainted,
            "taint_tokens_found": all_taints,
        })

    def _json_response(self, code: int, data: Any) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    print(f"[C] DB sink service starting on port {LISTEN_PORT}", flush=True)
    server = HTTPServer(("0.0.0.0", LISTEN_PORT), DBHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[C] Shutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
