"""
Testbed Service A - Client / Attacker

Sends HTTP requests containing a taint token to Service B.
The taint token format is:

    TAINT=<uuid>;

This is unique per run and trivially greppable in raw bytes, logs, or pcap.

Usage:
    python service_a.py              # one-shot: send single taint request
    python service_a.py --loop 5     # send 5 requests, 1s apart
    python service_a.py --token XYZ  # use explicit token instead of random UUID
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from urllib.error import URLError
from urllib.request import Request, urlopen


def make_taint_token(explicit: str | None = None) -> str:
    """Generate a taint token in the canonical format: TAINT=<value>;"""
    value = explicit or str(uuid.uuid4())
    return f"TAINT={value};"


def send_tainted_request(
    target_url: str,
    token: str,
    *,
    timeout_s: float = 10.0,
) -> dict:
    """
    POST a JSON payload containing the taint token to Service B.

    Returns the parsed JSON response on success, or an error dict.
    """
    payload = json.dumps({
        "user_input": token,
        "action": "lookup",
        "metadata": {"source": "service_a", "taint_marker": token},
    }).encode("utf-8")

    req = Request(
        target_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except URLError as exc:
        return {"error": str(exc), "token": token}
    except Exception as exc:
        return {"error": f"unexpected: {exc}", "token": token}


def main() -> None:
    parser = argparse.ArgumentParser(description="Testbed Service A - Taint Sender")
    parser.add_argument(
        "--target",
        default="http://service-b:8080/api/process",
        help="URL of Service B's endpoint",
    )
    parser.add_argument("--token", default=None, help="Explicit taint token value")
    parser.add_argument("--loop", type=int, default=1, help="Number of requests to send")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--wait-for-b", type=float, default=15.0, help="Seconds to wait for B to be ready")
    args = parser.parse_args()

    # Wait for Service B to be reachable
    print(f"[A] Waiting up to {args.wait_for_b}s for Service B at {args.target}...", flush=True)
    deadline = time.monotonic() + args.wait_for_b
    ready = False
    while time.monotonic() < deadline:
        try:
            req = Request(args.target.replace("/api/process", "/health"), method="GET")
            with urlopen(req, timeout=2.0):
                ready = True
                break
        except Exception:
            time.sleep(0.5)

    if not ready:
        print("[A] WARNING: Service B not reachable - sending anyway", flush=True)

    for i in range(args.loop):
        token = make_taint_token(args.token if args.loop == 1 else None)
        print(f"[A] Sending request {i + 1}/{args.loop} with token: {token}", flush=True)

        result = send_tainted_request(args.target, token)

        if "error" in result:
            print(f"[A] ERROR: {result['error']}", flush=True)
            sys.exit(1)
        else:
            print(f"[A] Response: {json.dumps(result, indent=2)}", flush=True)

            # Verify the taint token appears in the response chain
            response_str = json.dumps(result)
            taint_value = token.split("=")[1].rstrip(";")
            if taint_value in response_str:
                print("[A] PASS - taint token propagated through response", flush=True)
            else:
                print("[A] WARN - taint token not found in response", flush=True)

        if i < args.loop - 1:
            time.sleep(args.delay)

    print("[A] Done.", flush=True)


if __name__ == "__main__":
    main()
