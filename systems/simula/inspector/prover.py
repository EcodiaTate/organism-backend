"""
EcodiaOS — Inspector Vulnerability Prover (Phases 4 + 5)

Proves vulnerabilities exist by encoding security conditions as Z3 constraints,
then translates proven counterexamples into local diagnostic reproduction scripts.

The Inversion:
  Internal Simula: "Is this code correct?" → NOT(property) → UNSAT = correct
  Inspector:          "Is this code exploitable?" → NOT(security_property) → SAT = exploitable

Bounded Model Checking (BMC):
  The prover iteratively deepens the state-machine unroll depth from 1 (stateless,
  single-request) up to BMC_MAX_DEPTH (default 5).  At each depth the LLM encodes
  exactly N states and N-1 transition relations, then Z3 checks satisfiability.
  The loop terminates immediately on SAT, guaranteeing the counterexample uses
  the *shortest* exploit chain.  UNSAT at all depths proves the invariant holds
  within bounds.

When Z3 returns SAT, the counterexample is a concrete set of variable assignments
that violate the security property. Phase 5 translates these into a local-only
Security Unit Test script (targeting localhost only) for the RepairAgent to
use as a regression test during its patch-verify loop.

Uses the same Z3Bridge.check_invariant() infrastructure as Stage 2B but
inverts the interpretation: SAT means the security property can be violated.
"""

from __future__ import annotations

import ast
import json
import re
import time
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import Message
from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    InspectorConfig,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.simula.inspector.taint_types import (
        CrossServiceAttackSurface,
        TaintGraph,
    )
    from systems.simula.verification.z3_bridge import Z3Bridge

logger = structlog.get_logger().bind(system="simula.inspector.prover")


# ── Vulnerability class mapping ─────────────────────────────────────────────
# Maps attack goal keywords to the most likely vulnerability class.

_GOAL_TO_VULN_CLASS: list[tuple[re.Pattern[str], VulnerabilityClass]] = [
    (re.compile(r"unauth|authentication|login|session|token", re.I), VulnerabilityClass.BROKEN_AUTH),
    (re.compile(r"access.control|user.a.*user.b|another.user|idor|object.reference", re.I), VulnerabilityClass.BROKEN_ACCESS_CONTROL),
    (re.compile(r"sql.injection|sql.inject|sqli", re.I), VulnerabilityClass.SQL_INJECTION),
    (re.compile(r"inject(?!.*sql)", re.I), VulnerabilityClass.INJECTION),
    (re.compile(r"xss|cross.site.script", re.I), VulnerabilityClass.XSS),
    (re.compile(r"ssrf|server.side.request", re.I), VulnerabilityClass.SSRF),
    (re.compile(r"privilege.escalat|admin.function|role.bypass", re.I), VulnerabilityClass.PRIVILEGE_ESCALATION),
    (re.compile(r"reentran|recursive.call|contract.*call.*itself", re.I), VulnerabilityClass.REENTRANCY),
    (re.compile(r"race.condition|concurrent|toctou|double.spend", re.I), VulnerabilityClass.RACE_CONDITION),
    (re.compile(r"redirect|open.redirect|url.redirect", re.I), VulnerabilityClass.UNVALIDATED_REDIRECT),
    (re.compile(r"information.disclos|leak|expos|sensitive.data", re.I), VulnerabilityClass.INFORMATION_DISCLOSURE),
    (re.compile(r"deserializ|pickle|yaml.load|marshal", re.I), VulnerabilityClass.INSECURE_DESERIALIZATION),
    (re.compile(r"path.traversal|directory.traversal|\.\./", re.I), VulnerabilityClass.PATH_TRAVERSAL),
    (re.compile(r"command.inject|os.system|subprocess|shell.inject", re.I), VulnerabilityClass.COMMAND_INJECTION),
]


# ── Severity heuristics ──────────────────────────────────────────────────────
# Higher severity for more impactful vulnerability classes.

_VULN_SEVERITY_MAP: dict[VulnerabilityClass, VulnerabilitySeverity] = {
    VulnerabilityClass.SQL_INJECTION: VulnerabilitySeverity.CRITICAL,
    VulnerabilityClass.COMMAND_INJECTION: VulnerabilitySeverity.CRITICAL,
    VulnerabilityClass.REENTRANCY: VulnerabilitySeverity.CRITICAL,
    VulnerabilityClass.INSECURE_DESERIALIZATION: VulnerabilitySeverity.CRITICAL,
    VulnerabilityClass.BROKEN_AUTH: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.BROKEN_ACCESS_CONTROL: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.PRIVILEGE_ESCALATION: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.SSRF: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.PATH_TRAVERSAL: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.INJECTION: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.XSS: VulnerabilitySeverity.MEDIUM,
    VulnerabilityClass.RACE_CONDITION: VulnerabilitySeverity.MEDIUM,
    VulnerabilityClass.INFORMATION_DISCLOSURE: VulnerabilitySeverity.MEDIUM,
    VulnerabilityClass.UNVALIDATED_REDIRECT: VulnerabilitySeverity.LOW,
    VulnerabilityClass.OTHER: VulnerabilitySeverity.LOW,
}


# ── Bounded Model Checking constants ─────────────────────────────────────────
# The BMC loop starts at depth 1 (stateless, single request) and increments
# until SAT is found or MAX_DEPTH is exhausted.  This guarantees we find the
# *shortest* exploit chain — the minimum number of state transitions needed
# to violate the business invariant.

BMC_MAX_DEPTH: int = 5

# ── System prompts ───────────────────────────────────────────────────────────


def _build_attack_encoding_system_prompt(transition_depth: int) -> str:
    """
    Build a depth-specific system prompt for the Z3 constraint encoder.

    The prompt instructs the LLM to model **exactly** ``transition_depth``
    states (State_0 through State_{transition_depth - 1}).  At depth 1 this
    is a stateless single-request model; at depth 2 a two-state model, etc.

    Args:
        transition_depth: Number of application states to model (>= 1).
    """
    # Build the state chain description: "State_0 (initial) → State_1 → ..."
    state_labels = [
        f"State_{i}" + (" (initial)" if i == 0 else "")
        for i in range(transition_depth)
    ]
    state_chain = " → ".join(state_labels)

    # Build the transition count description
    transition_count = transition_depth - 1

    if transition_depth == 1:
        depth_rules = (
            "1. Model exactly 1 application state: State_0.  There are NO "
            "transitions — this is a stateless, single-request model.\n\n"
            "2. Declare Z3 variables with the `_0` suffix to denote their "
            "value in State_0:\n"
            "   - Use z3.Int for integer values (balance_0, user_id_0)\n"
            "   - Use z3.Bool for boolean flags (is_authenticated_0)\n"
            "   - Use z3.Real for numeric values (amount_0)\n\n"
            "3. Declare \"request\" variables that represent the attacker-\n"
            "   controlled input (e.g., request_amount_0, request_path_0).\n\n"
            "4. There are no transition relations at depth 1.  Encode only\n"
            "   the constraints that hold within State_0.\n\n"
            "5. Encode the BUSINESS INVARIANT that should hold in State_0\n"
            "   (e.g., \"balance >= 0\", \"is_authenticated == True\").\n\n"
            "6. Assert the NEGATION of the business invariant in State_0 so\n"
            "   that Z3 SAT proves the invariant can be violated by a single\n"
            "   request."
        )
    else:
        ", ".join(
            f"transition_{i}_{i+1}" for i in range(transition_count)
        )
        depth_rules = (
            f"1. Model exactly {transition_depth} sequential states: "
            f"{state_chain}.\n\n"
            "2. Declare SUFFIXED Z3 variables for each state.  The suffix "
            "`_N` denotes the value of that variable IN State_N:\n"
            "   - Use z3.Int for integer values "
            + "(e.g., " + ", ".join(f"balance_{i}" for i in range(min(transition_depth, 3))) + ")\n"
            "   - Use z3.Bool for boolean flags "
            + "(e.g., " + ", ".join(f"is_authenticated_{i}" for i in range(min(transition_depth, 3))) + ")\n"
            "   - Use z3.Real for numeric values "
            + "(e.g., " + ", ".join(f"amount_{i}" for i in range(min(transition_depth, 3))) + ")\n\n"
            "3. Declare \"request\" variables that represent the attacker-\n"
            "   controlled input for each transition step "
            + "(e.g., " + ", ".join(f"request_amount_{i}" for i in range(transition_count)) + ").\n\n"
            f"4. Encode exactly {transition_count} TRANSITION RELATION(S) "
            "that define how state evolves between steps.\n"
            "   Each transition must express:\n"
            "   - Pre-condition in State_N\n"
            "   - The effect of the attacker's request\n"
            "   - Post-condition producing State_{N+1}\n"
            "   Example: balance_1 == balance_0 - request_amount_0\n\n"
            f"5. Encode the BUSINESS INVARIANT that should hold in the "
            f"final state (State_{transition_depth - 1}).\n\n"
            f"6. Assert the NEGATION of the business invariant in "
            f"State_{transition_depth - 1} so that Z3 SAT proves the "
            "invariant can be violated through the multi-step exploit chain."
        )

    return (
        "You are a security constraint encoder for Z3 SMT solver, specializing in "
        "state-machine constraint solving for multi-step logic flaws.\n\n"
        "Your task: given an attack surface (code) and an attacker goal, encode the "
        "security violation as "
        + (
            "a SINGLE-STATE Z3 model."
            if transition_depth == 1
            else f"a SEQUENCE OF {transition_count} STATE TRANSITION(S) modelled in Z3."
        )
        + f"  The goal is to find a concrete exploit chain across exactly "
        f"{transition_depth} application state(s) that PROVES the vulnerability "
        "exists.\n\n"
        "## State-Machine Encoding Rules\n\n"
        f"{depth_rules}\n\n"
        "7. Your output must be a single Z3 Python expression (z3.And of all "
        + ("constraints" if transition_depth == 1 else "transition relations")
        + " + the negated invariant).  The expression must evaluate to a "
        "z3.BoolRef when executed.\n\n"
        "## Variable Declaration Format\n\n"
        "Declare ALL state-suffixed variables as a dict mapping name → Z3 type.\n"
        "Use the naming convention: `<semantic_name>_<state_index>` for state "
        "variables, and `request_<name>_<step_index>` for attacker inputs.\n\n"
        "## Output Format\n\n"
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "variable_declarations": {"var_name_0": "Int|Real|Bool", ...},\n'
        '  "z3_expression": "z3.And('
        + (
            "constraints, negated_invariant"
            if transition_depth == 1
            else f"{', '.join(f'transition_{i}_{i+1}' for i in range(transition_count))}, negated_invariant"
        )
        + ')",\n'
        '  "reasoning": "Brief explanation of the state machine and invariant violation",\n'
        f'  "state_count": {transition_depth}\n'
        "}\n\n"
        "Do NOT include any other text. Only the JSON object.\n\n"
        + _BMC_EXAMPLES_TEXT
    )


# Static examples text shared across all depths — the LLM adapts the
# example patterns to the requested depth.
_BMC_EXAMPLES_TEXT = """\
## Examples

### Example 1: Double-Spend / Negative Balance (depth=3)
Surface: POST /api/transfer, function transfer(from_id, to_id, amount)
Goal: "Drain account balance below zero via concurrent transfers"

{
  "variable_declarations": {
    "balance_0": "Int",
    "balance_1": "Int",
    "balance_2": "Int",
    "request_amount_0": "Int",
    "request_amount_1": "Int",
    "is_authenticated_0": "Bool",
    "is_authenticated_1": "Bool",
    "check_passed_0": "Bool",
    "check_passed_1": "Bool"
  },
  "z3_expression": "z3.And(balance_0 > 0, is_authenticated_0 == True, is_authenticated_1 == True, request_amount_0 > 0, request_amount_1 > 0, check_passed_0 == z3.And(request_amount_0 <= balance_0), check_passed_1 == z3.And(request_amount_1 <= balance_0), check_passed_0 == True, check_passed_1 == True, balance_1 == balance_0 - request_amount_0, balance_2 == balance_1 - request_amount_1, balance_2 < 0)",
  "reasoning": "State_0: account has positive balance. Two transfers read the SAME balance_0 for their check (TOCTOU). After both apply, balance_2 = balance_0 - amount_0 - amount_1 which can go negative. The invariant balance >= 0 is violated in State_2.",
  "state_count": 3
}

### Example 2: Privilege Escalation via Role Toggle (depth=3)
Surface: POST /api/user/role, PATCH /api/admin/settings
Goal: "Regular user escalates to admin by toggling roles across requests"

{
  "variable_declarations": {
    "role_level_0": "Int",
    "role_level_1": "Int",
    "role_level_2": "Int",
    "admin_threshold": "Int",
    "request_new_role_0": "Int",
    "request_action_1": "Bool",
    "role_change_allowed_0": "Bool",
    "admin_access_granted_1": "Bool"
  },
  "z3_expression": "z3.And(role_level_0 < admin_threshold, admin_threshold == 100, role_change_allowed_0 == True, request_new_role_0 > role_level_0, role_level_1 == request_new_role_0, request_action_1 == True, admin_access_granted_1 == (role_level_1 >= admin_threshold), role_level_2 == role_level_1, admin_access_granted_1 == True)",
  "reasoning": "State_0: user starts below admin threshold. Step 0: user changes role (unchecked upper bound). State_1: role_level >= admin_threshold. Step 1: user accesses admin endpoint — access granted. Business invariant 'non-admin cannot reach admin functions' is violated.",
  "state_count": 3
}

### Example 3: Stateless Input Validation Bypass (depth=1)
Surface: POST /api/search, function search(query)
Goal: "SQL injection via unsanitized user input"

{
  "variable_declarations": {
    "input_length_0": "Int",
    "contains_sql_metachar_0": "Bool",
    "input_sanitized_0": "Bool",
    "query_parameterized_0": "Bool",
    "sql_executed_with_input_0": "Bool"
  },
  "z3_expression": "z3.And(input_length_0 > 0, contains_sql_metachar_0 == True, input_sanitized_0 == False, query_parameterized_0 == False, sql_executed_with_input_0 == True)",
  "reasoning": "State_0: a single request with SQL metacharacters in unsanitized input is passed directly to an unparameterized query. The invariant 'user input never reaches SQL execution unsanitized' is violated in a single step.",
  "state_count": 1
}

### Example 4: Authentication Bypass via Session Fixation (depth=3)
Surface: POST /api/login, GET /api/profile
Goal: "Unauthenticated user accesses protected resource via session fixation"

{
  "variable_declarations": {
    "has_session_0": "Bool",
    "session_owner_0": "Int",
    "is_authenticated_0": "Bool",
    "has_session_1": "Bool",
    "session_owner_1": "Int",
    "is_authenticated_1": "Bool",
    "has_session_2": "Bool",
    "session_owner_2": "Int",
    "is_authenticated_2": "Bool",
    "request_fixed_session_0": "Bool",
    "request_target_user_1": "Int",
    "attacker_id": "Int",
    "victim_id": "Int",
    "access_granted_2": "Bool"
  },
  "z3_expression": "z3.And(attacker_id != victim_id, has_session_0 == False, is_authenticated_0 == False, request_fixed_session_0 == True, has_session_1 == True, session_owner_1 == attacker_id, is_authenticated_1 == False, request_target_user_1 == victim_id, session_owner_2 == victim_id, is_authenticated_2 == True, has_session_2 == True, access_granted_2 == z3.And(has_session_2 == True, is_authenticated_2 == True), access_granted_2 == True, session_owner_2 != attacker_id)",
  "reasoning": "State_0: attacker has no session. Step 0: attacker fixes a session token. State_1: session exists but unauthenticated. Step 1: victim authenticates with the fixed session. State_2: session now bound to victim, attacker reuses it. Invariant 'only the session creator has access' is violated.",
  "state_count": 3
}"""


# Legacy alias — callers that reference the constant directly get the
# depth-3 variant which matches the previous hard-coded behavior.
_ATTACK_ENCODING_SYSTEM_PROMPT = _build_attack_encoding_system_prompt(3)


_SEVERITY_CLASSIFICATION_PROMPT = """\
You are a vulnerability severity classifier.

Given a vulnerability with its Z3 counterexample, classify the severity
as one of: LOW, MEDIUM, HIGH, CRITICAL.

Severity guidelines (CVSS-aligned):
- CRITICAL (9.0-10.0): Remote code execution, authentication bypass, \
SQL injection with data exfiltration, reentrancy with fund theft
- HIGH (7.0-8.9): Privilege escalation, SSRF, broken access control, \
path traversal with sensitive file read
- MEDIUM (4.0-6.9): XSS, race conditions, information disclosure of \
non-sensitive data, CSRF
- LOW (0.1-3.9): Open redirects, minor information leaks, verbose \
error messages

Also consider the attack surface context:
- API endpoints with authentication: higher severity
- Public functions without state changes: lower severity
- Smart contract functions handling funds: higher severity

Respond with ONLY a JSON object:
{
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "reasoning": "Brief justification"
}"""


# ── Phase 5: Reproduction script generation prompt ─────────────────────────

_POC_GENERATION_SYSTEM_PROMPT = """\
You are a security verification engineer generating diagnostic reproduction \
scripts from formal Z3 counterexamples.

Your task: given a Z3 counterexample (concrete variable assignments that \
demonstrate a vulnerability condition) and the attack surface context, \
generate a Python Security Unit Test script that:
  1. Reproduces the exact input conditions Z3 identified against a LOCAL \
     development server only.
  2. Asserts the expected secure behaviour (a failing assertion documents \
     the bug for the RepairAgent to fix).
  3. Is safe, readable, and never touches a production system.

## Rules

1. The script must use the `requests` library (or `httpx` as fallback).
2. The script MUST target localhost only — `TARGET_URL` defaults to \
"http://localhost:8000". Add a comment: "# Run against local dev server only".
3. Include detailed comments explaining:
   - Which security property the Z3 counterexample violates
   - How each Z3 variable maps to a concrete request parameter
   - What the assertion checks (i.e., the expected SECURE response)
4. Structure the script as:
   - Module docstring labelled "Security Unit Test: <VulnerabilityClass>"
   - `TARGET_URL` constant at the top with the localhost comment
   - A `build_request()` function that returns (url, headers, payload) \
     WITHOUT executing the request
   - A `run_security_test()` function that calls `build_request()`, sends \
     the request, and asserts the response indicates secure behaviour
   - A `if __name__ == "__main__"` block that calls `run_security_test()`
5. Map Z3 variable assignments to concrete HTTP values:
   - Bool False → omit auth headers or use empty tokens
   - Bool True → include the header/parameter
   - Int/Real values → use as path params, query params, or body values
6. The script must be syntactically valid Python 3.10+.
7. Do NOT import anything beyond: requests, json, sys, os, urllib.parse, base64, hashlib, hmac.

## Output Format

Respond with ONLY the Python code. No markdown fences, no explanation text.
Start directly with the docstring or import statements.

## Examples

### Example 1: Broken Access Control
Z3 counterexample: is_authenticated=True, requested_user_id=999, current_user_id=1, can_access_data=True
Surface: GET /api/user/{id}

Output:
\"\"\"
Security Unit Test: Broken Access Control (IDOR)
Z3 condition: is_authenticated=True, requested_user_id=999, current_user_id=1 → can_access_data=True
Expected secure behaviour: server returns 403 Forbidden when user 1 requests user 999's data.
\"\"\"
import requests

TARGET_URL = "http://localhost:8000"  # Run against local dev server only

def build_request():
    url = f"{TARGET_URL}/api/user/999"
    headers = {"Authorization": "Bearer <valid_token_for_user_1>"}
    return url, headers, None

def run_security_test():
    url, headers, payload = build_request()
    print(f"[test] GET {url} as user 1 (should be denied)")
    response = requests.get(url, headers=headers)
    assert response.status_code == 403, (
        f"SECURITY BUG: expected 403 Forbidden, got {response.status_code}. "
        "User can read another user's data."
    )
    print("[pass] Server correctly denied cross-user data access")
    return response

if __name__ == "__main__":
    run_security_test()

### Example 2: SQL Injection
Z3 counterexample: input_length=10, contains_sql_metachar=True, input_sanitized=False, query_parameterized=False, sql_executed_with_input=True
Surface: POST /api/search, function search(query)

Output:
\"\"\"
Security Unit Test: SQL Injection
Z3 condition: contains_sql_metachar=True, input_sanitized=False, query_parameterized=False
Expected secure behaviour: server rejects or safely handles SQL metacharacters.
\"\"\"
import requests
import json

TARGET_URL = "http://localhost:8000"  # Run against local dev server only

def build_request():
    url = f"{TARGET_URL}/api/search"
    headers = {"Content-Type": "application/json"}
    payload = {"query": "' OR 1=1 --"}
    return url, headers, payload

def run_security_test():
    url, headers, payload = build_request()
    print(f"[test] POST {url} with SQL metacharacters")
    response = requests.post(url, headers=headers, json=payload)
    assert response.status_code != 500, (
        "SECURITY BUG: server threw a 500, likely due to unparameterized SQL. "
        "Input must be sanitized or queries must use parameterized statements."
    )
    print("[pass] Server handled SQL metacharacters without error")
    return response

if __name__ == "__main__":
    run_security_test()"""


# ── Cross-service constraint encoding prompt ────────────────────────────────

_CROSS_SERVICE_ENCODING_PROMPT = """\
You are a security constraint encoder for Z3 SMT solver, specializing in \
cross-service vulnerability proving.

You are given:
  1. A multi-service code context with labelled service sections
  2. A taint context (JSON) describing observed data flows between services
  3. An attacker goal that spans multiple services

Your task: encode the cross-service security violation as Z3 constraints that \
model the taint propagation chain from entry point to sink.

## Cross-Service Variables

Declare the following Z3 variable patterns:
  - tainted_at_source: Bool — True when user input enters the first service
  - cross_boundary_N: Bool — True when tainted data flows from service N \
    to service N+1 (one per hop)
  - sanitized_at_hop_N: Bool — True if the data is sanitized at hop N \
    (one per hop)
  - service_trusts_input_N: Bool — True if service N implicitly trusts \
    input from the calling service (common inter-service anti-pattern)
  - reaches_sink: Bool — True when tainted data reaches the vulnerable sink
  - sink_is_parameterized: Bool — True if the sink uses parameterized \
    queries or escaping

Plus any vulnerability-specific variables from the standard encoding rules.

## Encoding Logic

The attacker goal is satisfiable when ALL of these hold:
  1. User input is tainted at source (tainted_at_source == True)
  2. Taint propagates across every service boundary \
     (cross_boundary_N == True for each hop)
  3. No sanitization occurs at any hop \
     (sanitized_at_hop_N == False for all hops)
  4. The receiving service trusts inter-service input \
     (service_trusts_input_N == True for receiving hops)
  5. Taint reaches the sink (reaches_sink == True)
  6. The sink has no parameterization/escaping \
     (sink_is_parameterized == False)

## Taint Context Format

The taint_context JSON has:
  - sources: [{variable_name, source_service, entry_point, taint_level}]
  - sinks: [{variable_name, sink_service, sink_type, is_sanitized}]
  - flows: [{from_service, to_service, flow_type, payload_signature}]
  - involved_services: [service_names]

## Output Format

Respond with ONLY a JSON object:
{
  "variable_declarations": {"var_name": "Int|Real|Bool", ...},
  "z3_expression": "z3.And(...)",
  "reasoning": "Brief explanation of the taint chain encoding",
  "taint_chain": ["service_a -> service_b", "service_b -> service_c"]
}"""


_CROSS_SERVICE_POC_PROMPT = """\
You are a security verification engineer generating multi-step diagnostic \
reproduction scripts for cross-service vulnerabilities.

Unlike single-service PoCs, cross-service PoCs must demonstrate taint \
propagation across service boundaries. The script must:

  1. Fire a tainted payload at Service A (the entry point).
  2. Verify the taint propagates to Service B (and optionally C).
  3. Assert that the final sink in the last service is exploitable.

## Rules

1. Use the `requests` library (or `httpx` as fallback).
2. Target localhost only — each service runs on a different port \
   in the docker-compose topology.
3. Structure the script as:
   - Module docstring: "Cross-Service Security Unit Test: <VulnerabilityClass>"
   - `TARGET_URLS` dict mapping service name → localhost:port
   - `inject_tainted_payload(target_urls)` — sends the initial request
   - `verify_propagation(target_urls)` — checks intermediate services
   - `verify_sink_exploitation(target_urls)` — asserts the final exploit
   - `run_cross_service_test()` — orchestrates all three steps
   - `if __name__ == "__main__"` block

4. Include comments explaining:
   - The taint chain (Service A → B → C)
   - Which Z3 variables map to which service boundary
   - What each step verifies

5. For each service call, add timing + a short sleep (0.5s) to allow \
   inter-service propagation in docker-compose.

6. The script must be syntactically valid Python 3.10+.
7. Only import: requests, json, sys, os, urllib.parse, base64, hashlib, \
   hmac, time.

## Output Format

Respond with ONLY the Python code. No markdown fences, no explanation text.
Start directly with the docstring or import statements."""


# ── Unsafe import / network patterns for PoC safety validation ─────────────

_POC_FORBIDDEN_IMPORTS = frozenset({
    "subprocess", "os.system", "os.popen", "shutil", "ctypes",
    "socket", "multiprocessing", "threading", "signal",
    "importlib", "pickle", "shelve", "marshal",
})

_POC_ALLOWED_IMPORTS = frozenset({
    "requests", "json", "sys", "os", "urllib", "urllib.parse",
    "base64", "hashlib", "hmac", "httpx", "time",
})


# ── VulnerabilityProver ──────────────────────────────────────────────────────


class VulnerabilityProver:
    """
    Bounded Model Checker for security violations.

    Proves security violations exist by iteratively deepening a state-machine
    unroll depth (1 → BMC_MAX_DEPTH) and encoding vulnerability conditions
    as Z3 constraints at each depth.  Terminates on the first SAT, yielding
    the shortest possible exploit chain.

    The key inversion from internal Simula verification:
    - Simula checks NOT(invariant) → UNSAT means code is correct
    - Inspector checks NOT(security_property) → SAT means the security property
      can be violated (i.e., a vulnerability exists)

    When SAT, the Z3 model provides concrete variable assignments that
    demonstrate the violation. These drive a local Security Unit Test script
    for the RepairAgent to verify its patches against.
    """

    def __init__(
        self,
        z3_bridge: Z3Bridge,
        llm: LLMProvider,
        *,
        max_encoding_retries: int = 2,
        check_timeout_ms: int = 10_000,
    ) -> None:
        """
        Args:
            z3_bridge: Z3Bridge instance for constraint checking.
            llm: LLM provider for encoding attack goals as Z3 constraints.
            max_encoding_retries: Max retries if encoding fails Z3 parsing.
            check_timeout_ms: Z3 solver timeout for each constraint check.
        """
        self._z3 = z3_bridge
        self._llm = llm
        self._max_retries = max_encoding_retries
        self._check_timeout_ms = check_timeout_ms
        self._log = logger

    # ── Public API ──────────────────────────────────────────────────────────

    async def prove_vulnerability(
        self,
        surface: AttackSurface,
        attack_goal: str,
        target_url: str = "unknown",
        *,
        generate_poc: bool = False,
        config: InspectorConfig | None = None,
        max_depth: int | None = None,
    ) -> VulnerabilityReport | None:
        """
        Bounded Model Checking loop: prove a vulnerability exists by
        iteratively deepening the state-machine unroll depth.

        Starts at ``transition_depth = 1`` (a single stateless request) and
        increments until either:
          - **SAT**: the shortest exploit chain is found and returned
            immediately as a ``VulnerabilityReport``.
          - **UNSAT at max_depth**: the invariant is proven safe within
            bounds — returns ``None``.
          - **Encoding failure at every depth**: returns ``None``.

        This guarantees the counterexample uses the *minimum* number of
        state transitions needed to violate the business invariant.

        Args:
            surface: The attack surface to test.
            attack_goal: Human-readable description of the attacker's goal
                (e.g., "Unauthenticated access to protected resource").
            target_url: GitHub URL or "internal_eos" for analytics tagging.
            generate_poc: If True, generate a PoC script for proven vulns.
            config: InspectorConfig for authorized_targets validation during
                PoC generation.
            max_depth: Override BMC_MAX_DEPTH for this invocation.  Defaults
                to the module-level ``BMC_MAX_DEPTH`` constant (5).

        Returns:
            VulnerabilityReport if vulnerability proven (SAT), None if
            not exploitable (UNSAT) or encoding failed at all depths.
        """
        effective_max_depth = max_depth if max_depth is not None else BMC_MAX_DEPTH
        start = time.monotonic()
        self._log.info(
            "bmc_prove_vulnerability_start",
            entry_point=surface.entry_point,
            attack_goal=attack_goal,
            surface_type=surface.surface_type.value,
            file_path=surface.file_path,
            max_depth=effective_max_depth,
        )

        # Track encoding failures — if every depth fails encoding we log
        # differently from UNSAT-at-all-depths.
        all_encoding_failed = True

        for depth in range(1, effective_max_depth + 1):
            depth_start = time.monotonic()
            self._log.info(
                "bmc_depth_start",
                depth=depth,
                max_depth=effective_max_depth,
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
            )

            # Encode with the depth-specific prompt (with reflexion retry)
            encoding = await self._encode_attack_goal_with_retry(
                surface, attack_goal, transition_depth=depth,
            )
            if encoding is None:
                self._log.warning(
                    "bmc_encoding_failed",
                    depth=depth,
                    entry_point=surface.entry_point,
                    attack_goal=attack_goal,
                )
                # Encoding failure at this depth — try the next depth.
                # The LLM may succeed at a different unroll depth where it
                # can reason about the vulnerability more naturally.
                continue

            all_encoding_failed = False
            z3_expr_code, variable_declarations, reasoning = encoding

            # Check constraints via Z3: SAT = vulnerability proven
            check_start = time.monotonic()
            status, counterexample, structured_model = self._check_exploit_constraints(
                z3_expr_code, variable_declarations,
            )
            z3_time_ms = int((time.monotonic() - check_start) * 1000)
            depth_ms = int((time.monotonic() - depth_start) * 1000)

            if status == "sat":
                # ── SAT: shortest exploit chain found ──────────────────
                total_ms = int((time.monotonic() - start) * 1000)
                vuln_class = self._classify_vulnerability(attack_goal)
                severity = await self._classify_severity(
                    surface, attack_goal, counterexample, vuln_class,
                )

                report = VulnerabilityReport(
                    target_url=target_url,
                    vulnerability_class=vuln_class,
                    severity=severity,
                    attack_surface=surface,
                    attack_goal=attack_goal,
                    z3_counterexample=counterexample,
                    z3_constraints_code=z3_expr_code,
                )

                # Build structured evidence for SSE pipeline
                if structured_model is not None:
                    evidence = self.build_boundary_test_evidence(
                        report, variable_declarations, structured_model,
                    )
                    report.boundary_test_evidence = evidence

                # Phase 5: generate Security Unit Test if requested
                if generate_poc:
                    poc_code = await self.generate_reproduction_script(
                        report, config=config,
                    )
                    if poc_code:
                        report.proof_of_concept_code = poc_code

                self._log.info(
                    "bmc_vulnerability_proved",
                    vuln_id=report.id,
                    vulnerability_class=vuln_class.value,
                    severity=severity.value,
                    entry_point=surface.entry_point,
                    attack_goal=attack_goal,
                    bmc_depth=depth,
                    z3_time_ms=z3_time_ms,
                    total_ms=total_ms,
                    counterexample=counterexample,
                    has_poc=bool(report.proof_of_concept_code),
                )
                return report

            elif status == "unsat":
                # ── UNSAT at this depth — safe so far, try deeper ──────
                self._log.info(
                    "bmc_depth_safe",
                    depth=depth,
                    entry_point=surface.entry_point,
                    attack_goal=attack_goal,
                    z3_time_ms=z3_time_ms,
                    depth_ms=depth_ms,
                )
                # Continue to next depth — the exploit may require more steps.
                continue

            else:
                # ── Solver timeout/unknown at this depth ───────────────
                self._log.warning(
                    "bmc_depth_inconclusive",
                    depth=depth,
                    entry_point=surface.entry_point,
                    attack_goal=attack_goal,
                    status=status,
                    detail=counterexample,
                    z3_time_ms=z3_time_ms,
                )
                # Inconclusive — still try the next depth rather than
                # aborting the entire BMC run, since a different encoding
                # may produce a decisive result.
                continue

        # ── All depths exhausted ───────────────────────────────────────────
        total_ms = int((time.monotonic() - start) * 1000)

        if all_encoding_failed:
            self._log.warning(
                "bmc_all_encodings_failed",
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
                max_depth=effective_max_depth,
                total_ms=total_ms,
            )
        else:
            self._log.info(
                "bmc_invariant_holds_within_bounds",
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
                max_depth=effective_max_depth,
                total_ms=total_ms,
            )

        return None

    async def prove_vulnerability_batch(
        self,
        surface: AttackSurface,
        attack_goals: list[str],
        target_url: str = "unknown",
        *,
        generate_poc: bool = False,
        config: InspectorConfig | None = None,
    ) -> list[VulnerabilityReport]:
        """
        Test multiple attack goals against a single surface.

        Args:
            surface: The attack surface to test.
            attack_goals: List of attacker goals to test.
            target_url: Target URL for analytics.
            generate_poc: If True, generate PoC scripts for proven vulns.
            config: InspectorConfig for authorized_targets validation.

        Returns:
            List of proven vulnerabilities (may be empty).
        """
        reports: list[VulnerabilityReport] = []

        for goal in attack_goals:
            report = await self.prove_vulnerability(
                surface, goal, target_url=target_url,
                generate_poc=generate_poc,
                config=config,
            )
            if report is not None:
                reports.append(report)

        return reports

    # ── Cross-service proving ──────────────────────────────────────────────

    async def prove_cross_service_vulnerability(
        self,
        cross_surface: CrossServiceAttackSurface,
        attack_goal: str,
        target_url: str = "unknown",
        *,
        taint_graph: TaintGraph | None = None,
        generate_poc: bool = False,
        config: InspectorConfig | None = None,
    ) -> VulnerabilityReport | None:
        """
        Prove a vulnerability that spans multiple services in a docker-compose topology.

        Uses the cross-service encoding prompt which models taint propagation
        chains (tainted_at_source → cross_boundary_N → reaches_sink) instead
        of single-service security properties.

        Args:
            cross_surface: Cross-service attack surface with taint context,
                involved services, and multi-service stitched code.
            attack_goal: Human-readable cross-service attacker goal
                (e.g., "SQL injection via taint propagation from api to db-service").
            target_url: GitHub URL or "internal_eos" for analytics.
            taint_graph: Optional eBPF-observed taint graph for constraint augmentation.
            generate_poc: If True, generate a multi-step cross-service PoC.
            config: InspectorConfig for authorized_targets validation.

        Returns:
            VulnerabilityReport if the cross-service vulnerability is proven (SAT),
            None if not exploitable (UNSAT) or encoding failed.
        """
        start = time.monotonic()
        surface = cross_surface.primary_surface

        self._log.info(
            "prove_cross_service_start",
            entry_point=surface.entry_point,
            attack_goal=attack_goal,
            involved_services=cross_surface.involved_services,
            taint_sources=len(cross_surface.taint_sources),
            taint_sinks=len(cross_surface.taint_sinks),
        )

        # Build taint context JSON for the encoding prompt
        taint_context = self._build_taint_context_json(cross_surface, taint_graph)

        # Encode with cross-service prompt
        encoding = await self._encode_cross_service_goal(
            cross_surface, attack_goal, taint_context,
        )
        if encoding is None:
            self._log.warning(
                "cross_service_encoding_failed",
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
                involved_services=cross_surface.involved_services,
            )
            return None

        z3_expr_code, variable_declarations, reasoning = encoding

        # Check via Z3
        check_start = time.monotonic()
        status, counterexample, structured_model = self._check_exploit_constraints(
            z3_expr_code, variable_declarations,
        )
        z3_time_ms = int((time.monotonic() - check_start) * 1000)
        total_ms = int((time.monotonic() - start) * 1000)

        if status == "sat":
            vuln_class = self._classify_vulnerability(attack_goal)
            severity = await self._classify_severity(
                surface, attack_goal, counterexample, vuln_class,
            )

            # Cross-service vulns are at least HIGH — taint crossed a trust boundary
            if severity in (VulnerabilitySeverity.LOW, VulnerabilitySeverity.MEDIUM):
                severity = VulnerabilitySeverity.HIGH

            report = VulnerabilityReport(
                target_url=target_url,
                vulnerability_class=vuln_class,
                severity=severity,
                attack_surface=surface,
                attack_goal=attack_goal,
                z3_counterexample=counterexample,
                z3_constraints_code=z3_expr_code,
            )

            # Build structured evidence for SSE pipeline
            if structured_model is not None:
                evidence = self.build_boundary_test_evidence(
                    report, variable_declarations, structured_model,
                )
                report.boundary_test_evidence = evidence

            # Generate cross-service PoC if requested
            if generate_poc:
                poc_code = await self._generate_cross_service_poc(
                    report, cross_surface, config=config,
                )
                if poc_code:
                    report.proof_of_concept_code = poc_code

            self._log.info(
                "cross_service_vulnerability_proved",
                vuln_id=report.id,
                vulnerability_class=vuln_class.value,
                severity=severity.value,
                entry_point=surface.entry_point,
                involved_services=cross_surface.involved_services,
                z3_time_ms=z3_time_ms,
                total_ms=total_ms,
                has_poc=bool(report.proof_of_concept_code),
            )
            return report

        elif status == "unsat":
            self._log.info(
                "cross_service_vulnerability_disproved",
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
                involved_services=cross_surface.involved_services,
                z3_time_ms=z3_time_ms,
                total_ms=total_ms,
            )
            return None

        else:
            self._log.warning(
                "cross_service_check_inconclusive",
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
                status=status,
                detail=counterexample,
                z3_time_ms=z3_time_ms,
            )
            return None

    # ── Cross-service encoding helpers ─────────────────────────────────────

    async def _encode_cross_service_goal(
        self,
        cross_surface: CrossServiceAttackSurface,
        goal: str,
        taint_context: str,
    ) -> tuple[str, dict[str, str], str] | None:
        """
        Encode a cross-service attack goal using the specialized prompt
        and the reflexion retry loop.
        """
        surface = cross_surface.primary_surface

        prompt_parts = [
            "## Cross-Service Attack Surface",
            f"Entry point: {surface.entry_point}",
            f"Type: {surface.surface_type.value}",
            f"File: {surface.file_path}",
            f"Primary service: {cross_surface.service_name}",
            f"Involved services: {', '.join(cross_surface.involved_services)}",
        ]

        if surface.http_method:
            prompt_parts.append(f"HTTP method: {surface.http_method}")
        if surface.route_pattern:
            prompt_parts.append(f"Route: {surface.route_pattern}")

        # Multi-service stitched code context
        if cross_surface.cross_service_context_code:
            prompt_parts.extend([
                "",
                "## Multi-Service Source Code",
                f"```\n{cross_surface.cross_service_context_code[:8000]}\n```",
            ])

        # Taint context from eBPF observations
        prompt_parts.extend([
            "",
            "## Taint Context (eBPF-Observed Data Flows)",
            f"```json\n{taint_context}\n```",
        ])

        # Cross-service flows
        if cross_surface.cross_service_flows:
            prompt_parts.extend([
                "",
                "## Observed Cross-Service Flows",
            ])
            for flow in cross_surface.cross_service_flows:
                prompt_parts.append(
                    f"  - {flow.from_service} → {flow.to_service} "
                    f"(type: {flow.flow_type.value}, "
                    f"events: {flow.event_count})"
                )

        prompt_parts.extend([
            "",
            "## Attacker Goal",
            f"{goal}",
            "",
            "Encode the cross-service attacker goal as Z3 constraints. "
            "Model the taint propagation chain from source to sink across "
            "service boundaries. The expression should be satisfiable (SAT) "
            "when the taint reaches the sink unsanitized.",
        ])

        messages: list[Message] = [
            Message(role="user", content="\n".join(prompt_parts))
        ]

        # Reflexion retry loop (same pattern as _encode_attack_goal_with_retry)
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._llm.generate(
                    system_prompt=_CROSS_SERVICE_ENCODING_PROMPT,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2 if attempt == 1 else 0.1,
                )
            except Exception as exc:
                self._log.error(
                    "cross_service_encoding_llm_error",
                    attempt=attempt,
                    error=str(exc),
                    entry_point=surface.entry_point,
                )
                return None

            assistant_text = response.text
            messages.append(Message(role="assistant", content=assistant_text))

            parsed = self._parse_encoding_response(assistant_text)
            if parsed is None:
                error_msg = (
                    "Execution failed with the following error: "
                    "Your response could not be parsed as a JSON object. "
                    "Please correct the Z3 script and output the fixed version "
                    "as a valid JSON object with keys: variable_declarations, "
                    "z3_expression, reasoning, taint_chain."
                )
                messages.append(Message(role="user", content=error_msg))
                continue

            z3_expr, var_decls, reasoning = parsed

            execution_error = self._validate_z3_expression(z3_expr, var_decls)
            if execution_error is None:
                if attempt > 1:
                    self._log.info(
                        "cross_service_reflexion_succeeded",
                        attempt=attempt,
                        entry_point=surface.entry_point,
                    )
                return parsed

            error_msg = (
                f"Execution failed with the following error: {execution_error}. "
                "Please correct the Z3 script and output the fixed version. "
                "Respond with ONLY a corrected JSON object."
            )
            messages.append(Message(role="user", content=error_msg))

        self._log.error(
            "cross_service_reflexion_failed",
            max_retries=self._max_retries,
            entry_point=surface.entry_point,
            attack_goal=goal,
        )
        return None

    def _build_taint_context_json(
        self,
        cross_surface: CrossServiceAttackSurface,
        taint_graph: TaintGraph | None,
    ) -> str:
        """
        Build the taint context JSON string for the cross-service encoding prompt.

        Merges the CrossServiceAttackSurface metadata with the live TaintGraph
        observations (if available from eBPF).
        """
        context: dict[str, Any] = {
            "involved_services": cross_surface.involved_services,
            "sources": [
                {
                    "variable_name": s.variable_name,
                    "source_service": s.source_service,
                    "entry_point": s.entry_point,
                    "taint_level": s.taint_level.value,
                }
                for s in cross_surface.taint_sources
            ],
            "sinks": [
                {
                    "variable_name": s.variable_name,
                    "sink_service": s.sink_service,
                    "sink_type": s.sink_type.value,
                    "is_sanitized": s.is_sanitized,
                    "sanitizer_name": s.sanitizer_name,
                }
                for s in cross_surface.taint_sinks
            ],
            "flows": [
                {
                    "from_service": f.from_service,
                    "to_service": f.to_service,
                    "flow_type": f.flow_type.value,
                    "payload_signature": f.payload_signature,
                    "event_count": f.event_count,
                }
                for f in cross_surface.cross_service_flows
            ],
        }

        # Augment with live eBPF graph data if available
        if taint_graph is not None:
            context["ebpf_observed"] = True
            context["ebpf_nodes"] = [
                {"service_name": n.service_name, "service_type": n.service_type}
                for n in taint_graph.nodes
            ]
            context["ebpf_edges"] = [
                {
                    "from_service": e.from_service,
                    "to_service": e.to_service,
                    "flow_type": e.flow_type.value,
                    "event_count": e.event_count,
                }
                for e in taint_graph.edges
            ]
            context["ebpf_sources"] = [
                {
                    "variable_name": s.variable_name,
                    "source_service": s.source_service,
                    "entry_point": s.entry_point,
                }
                for s in taint_graph.sources
            ]
            context["has_unsanitized_path"] = taint_graph.has_unsanitized_path
        else:
            context["ebpf_observed"] = False

        return json.dumps(context, indent=2)

    async def _generate_cross_service_poc(
        self,
        report: VulnerabilityReport,
        cross_surface: CrossServiceAttackSurface,
        *,
        config: InspectorConfig | None = None,
    ) -> str:
        """
        Generate a multi-step cross-service PoC script.

        Uses the cross-service PoC prompt which structures the script as:
        inject_tainted_payload → verify_propagation → verify_sink_exploitation.
        """
        surface = report.attack_surface

        prompt_parts = [
            "## Cross-Service Vulnerability Details",
            f"Class: {report.vulnerability_class.value}",
            f"Severity: {report.severity.value}",
            f"Attack goal: {report.attack_goal}",
            f"Involved services: {', '.join(cross_surface.involved_services)}",
            "",
            "## Z3 Counterexample (proven exploit conditions)",
            f"{report.z3_counterexample}",
            "",
            "## Attack Surface",
            f"Entry point: {surface.entry_point}",
            f"Type: {surface.surface_type.value}",
            f"File: {surface.file_path}",
            f"Primary service: {cross_surface.service_name}",
        ]

        if surface.http_method:
            prompt_parts.append(f"HTTP method: {surface.http_method}")
        if surface.route_pattern:
            prompt_parts.append(f"Route: {surface.route_pattern}")

        # Taint chain for the PoC to reproduce
        if cross_surface.cross_service_flows:
            prompt_parts.extend(["", "## Taint Chain"])
            for flow in cross_surface.cross_service_flows:
                prompt_parts.append(
                    f"  {flow.from_service} → {flow.to_service} "
                    f"(via {flow.flow_type.value})"
                )

        if cross_surface.taint_sinks:
            prompt_parts.extend(["", "## Sink Details"])
            for sink in cross_surface.taint_sinks:
                prompt_parts.append(
                    f"  - {sink.variable_name} in {sink.sink_service} "
                    f"({sink.sink_type.value}, sanitized={sink.is_sanitized})"
                )

        if cross_surface.cross_service_context_code:
            prompt_parts.extend([
                "",
                "## Multi-Service Source Code",
                f"```\n{cross_surface.cross_service_context_code[:6000]}\n```",
            ])

        if report.z3_constraints_code:
            prompt_parts.extend([
                "",
                "## Z3 Constraints (for reference)",
                f"```python\n{report.z3_constraints_code}\n```",
            ])

        prompt_parts.extend([
            "",
            "Generate a multi-step cross-service Security Unit Test script that "
            "reproduces the taint propagation chain. Each service runs on a "
            "different localhost port in docker-compose.",
        ])

        user_prompt = "\n".join(prompt_parts)

        try:
            response = await self._llm.generate(
                system_prompt=_CROSS_SERVICE_POC_PROMPT,
                messages=[Message(role="user", content=user_prompt)],
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            self._log.error(
                "cross_service_poc_llm_error",
                vuln_id=report.id,
                error=str(exc),
            )
            return ""

        poc_code = self._parse_poc_response(response.text)
        if not poc_code:
            self._log.warning(
                "cross_service_poc_parse_failed",
                vuln_id=report.id,
            )
            return ""

        # Validate syntax
        syntax_error = self._validate_poc_syntax(poc_code)
        if syntax_error is not None:
            self._log.warning(
                "cross_service_poc_syntax_invalid",
                vuln_id=report.id,
                error=syntax_error,
            )
            return ""

        # Validate safety
        authorized_targets = config.authorized_targets if config else []
        safety_error = self._validate_poc_safety(poc_code, authorized_targets)
        if safety_error is not None:
            self._log.warning(
                "cross_service_poc_safety_violation",
                vuln_id=report.id,
                error=safety_error,
            )
            return ""

        self._log.info(
            "cross_service_poc_generated",
            vuln_id=report.id,
            script_size_bytes=len(poc_code.encode()),
            involved_services=cross_surface.involved_services,
        )

        return poc_code

    # ── Attack goal encoding ────────────────────────────────────────────────

    async def _encode_attack_goal(
        self,
        surface: AttackSurface,
        goal: str,
    ) -> tuple[str, dict[str, str], str] | None:
        """
        Use LLM to encode an attacker goal as Z3 constraints.

        Args:
            surface: The attack surface with context code.
            goal: Human-readable attacker goal.

        Returns:
            (z3_expression, variable_declarations, reasoning) or None if
            encoding fails.
        """
        # Build the user prompt with surface context
        prompt_parts = [
            "## Attack Surface",
            f"Entry point: {surface.entry_point}",
            f"Type: {surface.surface_type.value}",
            f"File: {surface.file_path}",
        ]

        if surface.http_method:
            prompt_parts.append(f"HTTP method: {surface.http_method}")
        if surface.route_pattern:
            prompt_parts.append(f"Route: {surface.route_pattern}")

        if surface.context_code:
            prompt_parts.extend([
                "",
                "## Source Code",
                f"```\n{surface.context_code[:4000]}\n```",
            ])

        prompt_parts.extend([
            "",
            "## Attacker Goal",
            f"{goal}",
            "",
            "Encode the attacker goal as Z3 constraints. The Z3 expression "
            "should be satisfiable (SAT) when the attack succeeds.",
        ])

        user_prompt = "\n".join(prompt_parts)

        try:
            response = await self._llm.generate(
                system_prompt=_ATTACK_ENCODING_SYSTEM_PROMPT,
                messages=[Message(role="user", content=user_prompt)],
                max_tokens=2048,
                temperature=0.2,
            )
        except Exception as exc:
            self._log.error(
                "encoding_llm_error",
                error=str(exc),
                entry_point=surface.entry_point,
            )
            return None

        return self._parse_encoding_response(response.text)

    async def _encode_attack_goal_with_retry(
        self,
        surface: AttackSurface,
        goal: str,
        *,
        transition_depth: int = 3,
    ) -> tuple[str, dict[str, str], str] | None:
        """
        Reflexion loop: generate Z3 encoding, attempt execution, feed any
        error back to the LLM as a follow-up user message, and retry.

        Maintains the full conversation history so the LLM has context of
        all previous failed attempts when generating a correction.

        Catches both JSON parse failures and Z3 execution errors
        (SyntaxError, NameError, Z3Exception, eval errors).

        When the surface has a non-empty taint_context, automatically
        switches to the cross-service encoding prompt for more precise
        multi-service constraint generation.

        Args:
            surface: The attack surface with context code.
            goal: Human-readable attacker goal.
            transition_depth: Number of application states to model in the
                Z3 constraint encoding (BMC unroll depth).  Defaults to 3
                for backward compatibility when called outside the BMC loop.
        """
        # Detect cross-service mode from taint context
        has_taint_context = bool(surface.taint_context)
        system_prompt = (
            _CROSS_SERVICE_ENCODING_PROMPT if has_taint_context
            else _build_attack_encoding_system_prompt(transition_depth)
        )

        # Build the initial user prompt
        prompt_parts = [
            "## Attack Surface",
            f"Entry point: {surface.entry_point}",
            f"Type: {surface.surface_type.value}",
            f"File: {surface.file_path}",
        ]
        if surface.service_name:
            prompt_parts.append(f"Service: {surface.service_name}")
        if surface.http_method:
            prompt_parts.append(f"HTTP method: {surface.http_method}")
        if surface.route_pattern:
            prompt_parts.append(f"Route: {surface.route_pattern}")
        if surface.context_code:
            prompt_parts.extend([
                "",
                "## Source Code",
                f"```\n{surface.context_code[:4000]}\n```",
            ])
        if has_taint_context:
            prompt_parts.extend([
                "",
                "## Taint Context (eBPF-Observed Data Flows)",
                f"```json\n{surface.taint_context[:4000]}\n```",
            ])
        depth_instruction = (
            f"Model exactly {transition_depth} application state(s) "
            f"(State_0"
            + (f" through State_{transition_depth - 1}" if transition_depth > 1 else "")
            + f") with {transition_depth - 1} transition(s). "
            if transition_depth > 1
            else "Model exactly 1 application state (State_0) with no transitions. "
        )
        prompt_parts.extend([
            "",
            "## Attacker Goal",
            f"{goal}",
            "",
            "## BMC Depth Constraint",
            f"{depth_instruction}"
            "Encode the attacker goal as Z3 constraints. The Z3 expression "
            "should be satisfiable (SAT) when the attack succeeds. "
            f"Set \"state_count\": {transition_depth} in your response.",
        ])

        messages: list[Message] = [
            Message(role="user", content="\n".join(prompt_parts))
        ]

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._llm.generate(
                    system_prompt=system_prompt,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2 if attempt == 0 else 0.1,
                )
            except Exception as exc:
                self._log.error(
                    "encoding_llm_error",
                    attempt=attempt,
                    error=str(exc),
                    entry_point=surface.entry_point,
                )
                return None

            assistant_text = response.text

            # Append assistant reply to history for subsequent turns
            messages.append(Message(role="assistant", content=assistant_text))

            # Try to parse the JSON response
            parsed = self._parse_encoding_response(assistant_text)
            if parsed is None:
                error_msg = (
                    "Execution failed with the following error: "
                    "Your response could not be parsed as a JSON object. "
                    "Please correct the Z3 script and output the fixed version "
                    "as a valid JSON object with keys: variable_declarations, "
                    "z3_expression, reasoning."
                )
                self._log.debug(
                    "z3_reflexion_parse_error",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    entry_point=surface.entry_point,
                )
                messages.append(Message(role="user", content=error_msg))
                continue

            z3_expr, var_decls, reasoning = parsed

            # Validate Z3 execution (catches SyntaxError, NameError, Z3Exception, etc.)
            execution_error = self._validate_z3_expression(z3_expr, var_decls)
            if execution_error is None:
                # Success — valid, executable Z3 expression
                if attempt > 0:
                    self._log.info(
                        "z3_reflexion_succeeded",
                        attempt=attempt,
                        entry_point=surface.entry_point,
                    )
                return parsed

            # Feed the execution traceback back to the LLM
            error_msg = (
                f"Execution failed with the following error: {execution_error}. "
                "Please correct the Z3 script and output the fixed version. "
                "Common issues:\n"
                "- Expression must use declared variable names exactly\n"
                "- Use z3.And, z3.Or, z3.Not, z3.Implies — not Python and/or/not\n"
                "- Comparison operators (==, !=, <, >, <=, >=) are fine on Z3 vars\n"
                "- Bool variables use == True/False, not bare references\n"
                "Respond with ONLY a corrected JSON object."
            )
            self._log.debug(
                "z3_reflexion_execution_error",
                attempt=attempt,
                max_retries=self._max_retries,
                error=execution_error,
                entry_point=surface.entry_point,
            )
            messages.append(Message(role="user", content=error_msg))

        self._log.error(
            "prover_reflexion_failed",
            max_retries=self._max_retries,
            entry_point=surface.entry_point,
            attack_goal=goal,
        )
        return None

    # ── Z3 constraint checking ──────────────────────────────────────────────

    def _check_exploit_constraints(
        self,
        z3_expr_code: str,
        variable_declarations: dict[str, str],
    ) -> tuple[str, str, dict[str, Any] | None]:
        """
        Check exploit constraints via Z3.

        Unlike Z3Bridge.check_invariant() which checks NOT(property),
        here we check the expression DIRECTLY: if SAT, the exploit
        conditions can be satisfied.

        The expression already encodes the attacker's goal (the negation
        of the security property), so we don't need to negate again.

        Args:
            z3_expr_code: Z3 Python expression encoding the attack.
            variable_declarations: Variable name → Z3 type mapping.

        Returns:
            ("sat", counterexample_str, structured_model) if exploitable,
            ("unsat", "", None) if secure,
            ("unknown", error_detail, None) if inconclusive.
        """
        try:
            import z3 as z3_lib
        except ImportError:
            return "unknown", "z3-solver not installed", None

        solver = z3_lib.Solver()
        solver.set("timeout", self._check_timeout_ms)

        # Create Z3 variables from declarations
        z3_vars: dict[str, Any] = {}
        for name, z3_type in variable_declarations.items():
            if z3_type == "Int":
                z3_vars[name] = z3_lib.Int(name)
            elif z3_type == "Real":
                z3_vars[name] = z3_lib.Real(name)
            elif z3_type == "Bool":
                z3_vars[name] = z3_lib.Bool(name)
            else:
                # Default to Real for unknown types
                z3_vars[name] = z3_lib.Real(name)

        # Evaluate the Z3 expression in a sandboxed namespace
        namespace: dict[str, Any] = {"z3": z3_lib, **z3_vars}
        try:
            expr = eval(z3_expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as exc:
            return "unknown", f"expression eval error: {exc}", None

        if not isinstance(expr, z3_lib.BoolRef):
            return "unknown", "expression did not produce a z3.BoolRef", None

        # Direct check: the expression encodes the attacker goal.
        # SAT means the attack conditions can be satisfied → vulnerability.
        solver.add(expr)
        result = solver.check()

        if result == z3_lib.sat:
            model = solver.model()
            counterexample = self._extract_z3_model(model)
            structured = self._extract_z3_model_structured(model, variable_declarations)
            return "sat", counterexample, structured
        elif result == z3_lib.unsat:
            return "unsat", "", None
        else:
            return "unknown", "solver timeout or unknown", None

    def _extract_z3_model(self, model: Any) -> str:
        """
        Convert a Z3 model to a human-readable counterexample string.

        The model contains concrete variable assignments that demonstrate
        the exploit conditions.

        Args:
            model: Z3 Model object from a SAT result.

        Returns:
            Human-readable string like "is_authenticated=False, user_id=999".
        """
        parts: list[str] = []
        try:
            for decl in model.decls():
                value = model[decl]
                # Format boolean values readably
                val_str = str(value)
                if val_str == "True":
                    val_str = "True"
                elif val_str == "False":
                    val_str = "False"
                parts.append(f"{decl.name()}={val_str}")
        except Exception as exc:
            self._log.warning("model_extraction_error", error=str(exc))
            return f"<model extraction failed: {exc}>"

        return ", ".join(sorted(parts))

    # ── State-index regex for multi-step model extraction ──────────────
    # Matches variable names ending in _N where N is a non-negative integer,
    # e.g. "balance_0", "is_authenticated_1", "request_amount_2".
    _STATE_SUFFIX_RE = re.compile(r"^(.+)_(\d+)$")

    def _extract_z3_model_structured(
        self,
        model: Any,
        variable_declarations: dict[str, str],
    ) -> dict[str, Any]:
        """
        Extract a Z3 model as a structured multi-step representation.

        Parses state-suffixed variable names (e.g. ``balance_0``,
        ``balance_1``) into an ordered list of per-step value dicts.
        Variables without a numeric suffix are placed in a top-level
        ``"globals"`` dict.

        The returned structure is::

            {
                "globals": {"admin_threshold": 100, ...},
                "steps": [
                    {"balance": 500, "is_authenticated": True, "request_amount": 200},
                    {"balance": 300, "request_amount": 350},
                    {"balance": -50},
                ],
                "state_count": 3,
                "flat": {"balance_0": 500, "balance_1": 300, ...}
            }

        ``flat`` preserves the original single-dict format for backward
        compatibility with callers that expect a plain variable→value map.

        Args:
            model: Z3 Model object from a SAT result.
            variable_declarations: The original var → Z3 type mapping,
                used to coerce values to the correct Python type.

        Returns:
            Dict with ``globals``, ``steps`` (list), ``state_count``,
            and ``flat`` keys.
        """
        flat: dict[str, Any] = {}
        # step_index → {base_name → value}
        step_buckets: dict[int, dict[str, Any]] = {}
        globals_bucket: dict[str, Any] = {}

        try:
            for decl in model.decls():
                name = decl.name()
                raw = model[decl]
                z3_type = variable_declarations.get(name, "")

                # Coerce to native Python type
                if z3_type == "Bool":
                    value: Any = str(raw) == "True"
                elif z3_type == "Int":
                    value = int(str(raw))
                elif z3_type == "Real":
                    val_str = str(raw)
                    if "/" in val_str:
                        num, den = val_str.split("/", 1)
                        value = float(int(num.strip()) / int(den.strip()))
                    else:
                        value = float(val_str)
                else:
                    value = str(raw)

                flat[name] = value

                # Bucket by state index
                m = self._STATE_SUFFIX_RE.match(name)
                if m:
                    base_name = m.group(1)
                    step_idx = int(m.group(2))
                    step_buckets.setdefault(step_idx, {})[base_name] = value
                else:
                    globals_bucket[name] = value

        except Exception as exc:
            self._log.warning("structured_model_extraction_error", error=str(exc))
            return {"globals": {}, "steps": [], "state_count": 0, "flat": flat}

        # Build the ordered steps list (fill gaps with empty dicts)
        max_idx = max(step_buckets.keys()) if step_buckets else -1
        steps: list[dict[str, Any]] = [
            step_buckets.get(i, {}) for i in range(max_idx + 1)
        ]

        return {
            "globals": globals_bucket,
            "steps": steps,
            "state_count": len(steps),
            "flat": flat,
        }

    def build_boundary_test_evidence(
        self,
        report: VulnerabilityReport,
        variable_declarations: dict[str, str],
        model_values: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a structured JSON evidence payload from a proven vulnerability.

        Maps Z3 model variables back to their corresponding HTTP request
        fields (headers, body, query parameters) using heuristics on
        variable names and the AttackSurface metadata.

        With the state-machine upgrade, ``model_values`` now contains
        ``steps`` (an ordered list of per-state dicts) and ``globals``.
        The method partitions each step's variables into HTTP field
        categories and emits ``edge_case_input`` as an **ordered list**
        of sequential payloads — one per exploit step.

        Args:
            report: The proven VulnerabilityReport.
            variable_declarations: var → Z3 type mapping from the encoding.
            model_values: Structured model from _extract_z3_model_structured,
                containing ``steps``, ``globals``, ``state_count``, and ``flat``.

        Returns:
            A JSON-serialisable dict suitable for SSE transmission::

                {
                  "analysis_result": "EVIDENCE_FOUND",
                  "details": {
                    "vuln_id": "...",
                    "vulnerability_class": "...",
                    "severity": "...",
                    "endpoint": "POST /api/test",
                    "entry_point": "app.routes.test",
                    "file_path": "src/routes.py",
                    "line_number": 42,
                    "attack_goal": "...",
                    "state_count": 3,
                    "edge_case_input": [
                      {"step": 0, "headers": {...}, "body": {...}, ...},
                      {"step": 1, "headers": {...}, "body": {...}, ...},
                      {"step": 2, "body": {...}, "flags": {...}, ...}
                    ],
                    "global_context": {"admin_threshold": 100},
                    "variable_types": {"balance_0": "Int", ...},
                    "z3_constraints": "...",
                  }
                }
        """
        surface = report.attack_surface

        # Build the endpoint string (e.g. "POST /api/users/{id}")
        method = (surface.http_method or "CALL").upper()
        route = surface.route_pattern or surface.entry_point
        endpoint = f"{method} {route}"

        _HEADER_HINTS = frozenset({
            "is_authenticated", "auth", "token", "authorization",
            "session", "cookie", "api_key", "bearer",
            "has_session",
        })
        _QUERY_HINTS = frozenset({
            "page", "limit", "offset", "sort", "filter", "search", "q",
        })
        _FLAG_HINTS = frozenset({
            "is_admin", "has_permission", "access_granted", "can_access",
            "function_requires_admin", "input_sanitized", "query_parameterized",
            "sql_executed_with_input", "contains_sql_metachar",
            "sink_is_parameterized", "reaches_sink",
            "tainted_at_source", "service_trusts_input",
            "check_passed", "role_change_allowed", "admin_access_granted",
        })

        def _partition_vars(vars_dict: dict[str, Any]) -> dict[str, Any]:
            """Partition a flat variable dict into HTTP field categories."""
            headers: dict[str, Any] = {}
            body: dict[str, Any] = {}
            query: dict[str, Any] = {}
            flags: dict[str, Any] = {}

            for var_name, value in vars_dict.items():
                name_lower = var_name.lower()
                if any(h in name_lower for h in _HEADER_HINTS):
                    headers[var_name] = value
                elif any(h in name_lower for h in _FLAG_HINTS):
                    flags[var_name] = value
                elif any(h in name_lower for h in _QUERY_HINTS):
                    query[var_name] = value
                else:
                    body[var_name] = value

            result: dict[str, Any] = {}
            if headers:
                result["headers"] = headers
            if body:
                result["body"] = body
            if query:
                result["query"] = query
            if flags:
                result["flags"] = flags
            return result

        # Extract multi-step structure
        steps: list[dict[str, Any]] = model_values.get("steps", [])
        globals_bucket: dict[str, Any] = model_values.get("globals", {})
        state_count: int = model_values.get("state_count", 0)

        # Build the ordered list of per-step payloads
        edge_case_input: list[dict[str, Any]] = []
        for step_idx, step_vars in enumerate(steps):
            if not step_vars:
                # Empty step — still include for positional correctness
                edge_case_input.append({"step": step_idx})
                continue
            partitioned = _partition_vars(step_vars)
            partitioned["step"] = step_idx
            edge_case_input.append(partitioned)

        # Fallback: if no multi-step structure was detected (legacy single-step),
        # partition the flat dict and wrap in a single-element list
        if not edge_case_input:
            flat = model_values.get("flat", model_values)
            partitioned = _partition_vars(flat)
            partitioned["step"] = 0
            edge_case_input.append(partitioned)
            state_count = 1

        return {
            "analysis_result": "EVIDENCE_FOUND",
            "details": {
                "vuln_id": report.id,
                "vulnerability_class": report.vulnerability_class.value,
                "severity": report.severity.value,
                "endpoint": endpoint,
                "entry_point": surface.entry_point,
                "file_path": surface.file_path,
                "line_number": surface.line_number,
                "attack_goal": report.attack_goal,
                "state_count": state_count,
                "edge_case_input": edge_case_input,
                "global_context": _partition_vars(globals_bucket) if globals_bucket else {},
                "variable_types": variable_declarations,
                "z3_constraints": report.z3_constraints_code,
            },
        }

    # ── Vulnerability classification ────────────────────────────────────────

    def _classify_vulnerability(self, attack_goal: str) -> VulnerabilityClass:
        """
        Classify a vulnerability based on the attack goal keywords.

        Uses regex matching against known vulnerability patterns.
        Falls back to OTHER if no pattern matches.
        """
        for pattern, vuln_class in _GOAL_TO_VULN_CLASS:
            if pattern.search(attack_goal):
                return vuln_class
        return VulnerabilityClass.OTHER

    async def _classify_severity(
        self,
        surface: AttackSurface,
        attack_goal: str,
        counterexample: str,
        vuln_class: VulnerabilityClass,
    ) -> VulnerabilitySeverity:
        """
        Classify severity using heuristic mapping first, LLM refinement
        if needed for edge cases.

        The heuristic is fast and deterministic; the LLM provides nuanced
        classification for cases where surface context matters.
        """
        # Start with heuristic severity from the vulnerability class
        base_severity = _VULN_SEVERITY_MAP.get(
            vuln_class, VulnerabilitySeverity.MEDIUM,
        )

        # Escalate based on surface type heuristics
        if surface.surface_type == AttackSurfaceType.SMART_CONTRACT_PUBLIC:
            # Smart contract vulns involving funds are always critical
            if vuln_class in (
                VulnerabilityClass.REENTRANCY,
                VulnerabilityClass.RACE_CONDITION,
            ):
                return VulnerabilitySeverity.CRITICAL

        if surface.surface_type == AttackSurfaceType.AUTH_HANDLER:
            # Auth handler vulnerabilities are at least HIGH
            if base_severity.value in ("low", "medium"):
                return VulnerabilitySeverity.HIGH

        if surface.surface_type == AttackSurfaceType.DATABASE_QUERY:
            # Database query vulns are at least HIGH (data exposure)
            if base_severity.value == "low":
                return VulnerabilitySeverity.MEDIUM

        # Use LLM for more nuanced classification of ambiguous cases
        if vuln_class == VulnerabilityClass.OTHER:
            llm_severity = await self._llm_classify_severity(
                surface, attack_goal, counterexample,
            )
            if llm_severity is not None:
                return llm_severity

        return base_severity

    async def _llm_classify_severity(
        self,
        surface: AttackSurface,
        attack_goal: str,
        counterexample: str,
    ) -> VulnerabilitySeverity | None:
        """
        Use LLM to classify severity for ambiguous vulnerability classes.

        Returns None if LLM classification fails (caller should use heuristic).
        """
        user_prompt = (
            f"Vulnerability: {attack_goal}\n"
            f"Surface: {surface.entry_point} ({surface.surface_type.value})\n"
            f"File: {surface.file_path}\n"
            f"Z3 counterexample: {counterexample}\n"
        )

        try:
            response = await self._llm.evaluate(
                prompt=(
                    f"{_SEVERITY_CLASSIFICATION_PROMPT}\n\n"
                    f"Classify this vulnerability:\n{user_prompt}"
                ),
                max_tokens=256,
                temperature=0.1,
            )
        except Exception:
            return None

        return self._parse_severity_response(response.text)

    # ── Response parsing ────────────────────────────────────────────────────

    def _parse_encoding_response(
        self,
        llm_text: str,
    ) -> tuple[str, dict[str, str], str] | None:
        """
        Parse the LLM's encoding response into Z3 expression + declarations.

        Expects JSON with keys: variable_declarations, z3_expression, reasoning.

        Returns:
            (z3_expression, variable_declarations, reasoning) or None.
        """
        text = llm_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        # Find JSON object in the response
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
            self._log.warning(
                "encoding_parse_no_json",
                text_preview=text[:200],
            )
            return None

        json_str = text[brace_start:brace_end + 1]
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self._log.warning("encoding_parse_json_error", error=str(exc))
            return None

        if not isinstance(obj, dict):
            return None

        z3_expression = obj.get("z3_expression", "")
        variable_declarations = obj.get("variable_declarations", {})
        reasoning = obj.get("reasoning", "")

        if not z3_expression or not isinstance(variable_declarations, dict):
            self._log.warning(
                "encoding_parse_missing_fields",
                has_expr=bool(z3_expression),
                has_vars=isinstance(variable_declarations, dict),
            )
            return None

        # Validate variable declarations are all valid Z3 types
        valid_types = {"Int", "Real", "Bool"}
        clean_decls: dict[str, str] = {}
        for name, z3_type in variable_declarations.items():
            if not isinstance(name, str) or not isinstance(z3_type, str):
                continue
            # Normalize type names (case-insensitive)
            normalized = z3_type.strip().capitalize()
            if normalized not in valid_types:
                normalized = "Real"  # Safe default
            clean_decls[name] = normalized

        return z3_expression, clean_decls, reasoning

    def _parse_severity_response(
        self,
        llm_text: str,
    ) -> VulnerabilitySeverity | None:
        """Parse LLM severity classification response."""
        text = llm_text.strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start == -1 or brace_end == -1:
            return None

        try:
            obj = json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            return None

        severity_str = obj.get("severity", "").strip().lower()
        try:
            return VulnerabilitySeverity(severity_str)
        except ValueError:
            return None

    # ── Phase 5: Reproduction script generation ───────────────────────────────

    async def generate_reproduction_script(
        self,
        report: VulnerabilityReport,
        *,
        config: InspectorConfig | None = None,
    ) -> str:
        """
        Generate a Security Unit Test script from a proven vulnerability.

        Takes the Z3 counterexample (concrete variable assignments that
        demonstrate a violated security property) and the attack surface
        context, then uses the LLM to produce a local-only diagnostic
        reproduction script. The script is structured as a Python test
        that asserts the expected secure response — making it directly
        consumable by the RepairAgent's patch-verify loop.

        Args:
            report: A proven VulnerabilityReport containing the surface,
                Z3 counterexample, and vulnerability goal.
            config: Optional InspectorConfig for authorized_targets validation.

        Returns:
            Python source code of the Security Unit Test script.
            Empty string if generation or safety validation fails.
        """
        start = time.monotonic()
        surface = report.attack_surface

        self._log.info(
            "reproduction_script_generation_start",
            vuln_id=report.id,
            vulnerability_class=report.vulnerability_class.value,
            severity=report.severity.value,
            entry_point=surface.entry_point,
        )

        # Build the user prompt with all context the LLM needs
        prompt_parts = [
            "## Vulnerability Details",
            f"Class: {report.vulnerability_class.value}",
            f"Severity: {report.severity.value}",
            f"Attack goal: {report.attack_goal}",
            "",
            "## Z3 Counterexample (proven exploit conditions)",
            f"{report.z3_counterexample}",
            "",
            "## Attack Surface",
            f"Entry point: {surface.entry_point}",
            f"Type: {surface.surface_type.value}",
            f"File: {surface.file_path}",
        ]

        if surface.http_method:
            prompt_parts.append(f"HTTP method: {surface.http_method}")
        if surface.route_pattern:
            prompt_parts.append(f"Route: {surface.route_pattern}")

        if surface.context_code:
            prompt_parts.extend([
                "",
                "## Source Code",
                f"```\n{surface.context_code[:4000]}\n```",
            ])

        if report.z3_constraints_code:
            prompt_parts.extend([
                "",
                "## Z3 Constraints (for reference)",
                f"```python\n{report.z3_constraints_code}\n```",
            ])

        prompt_parts.extend([
            "",
            "Generate a Python Security Unit Test script (reproduction script) that "
            "reproduces the exact violation conditions from the Z3 counterexample "
            "against a local development server and asserts the expected secure response.",
        ])

        user_prompt = "\n".join(prompt_parts)

        # Call LLM to generate the PoC
        try:
            response = await self._llm.generate(
                system_prompt=_POC_GENERATION_SYSTEM_PROMPT,
                messages=[Message(role="user", content=user_prompt)],
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            self._log.error(
                "reproduction_script_llm_error",
                vuln_id=report.id,
                error=str(exc),
            )
            return ""

        # Parse and extract the Python code from the response
        poc_code = self._parse_poc_response(response.text)
        if not poc_code:
            self._log.warning(
                "reproduction_script_parse_failed",
                vuln_id=report.id,
                response_preview=response.text[:200],
            )
            return ""

        # Validate syntax — must be parseable Python
        syntax_error = self._validate_poc_syntax(poc_code)
        if syntax_error is not None:
            self._log.warning(
                "reproduction_script_syntax_invalid",
                vuln_id=report.id,
                error=syntax_error,
            )
            # Attempt a single retry with the error fed back
            poc_code = await self._retry_reproduction_script_with_error(
                user_prompt, poc_code, syntax_error,
            )
            if not poc_code:
                return ""
            syntax_error = self._validate_poc_syntax(poc_code)
            if syntax_error is not None:
                self._log.warning(
                    "reproduction_script_syntax_retry_failed",
                    vuln_id=report.id,
                    error=syntax_error,
                )
                return ""

        # Validate safety — no forbidden imports, no unauthorized URLs
        authorized_targets = config.authorized_targets if config else []
        safety_error = self._validate_poc_safety(poc_code, authorized_targets)
        if safety_error is not None:
            self._log.warning(
                "reproduction_script_safety_violation",
                vuln_id=report.id,
                error=safety_error,
            )
            return ""

        total_ms = int((time.monotonic() - start) * 1000)
        self._log.info(
            "reproduction_script_generated",
            vuln_id=report.id,
            script_size_bytes=len(poc_code.encode()),
            total_ms=total_ms,
        )

        return poc_code

    # Keep backward-compatible alias for callers that pre-date the rename.
    generate_poc = generate_reproduction_script

    async def generate_reproduction_script_batch(
        self,
        reports: list[VulnerabilityReport],
        *,
        config: InspectorConfig | None = None,
    ) -> dict[str, str]:
        """
        Generate Security Unit Test scripts for multiple vulnerability reports.

        Args:
            reports: List of proven VulnerabilityReport objects.
            config: Optional InspectorConfig for authorized_targets validation.

        Returns:
            Dict mapping vulnerability report ID → reproduction script code.
            Only includes entries where generation succeeded.
        """
        results: dict[str, str] = {}

        for report in reports:
            script = await self.generate_reproduction_script(report, config=config)
            if script:
                results[report.id] = script

        self._log.info(
            "reproduction_script_batch_complete",
            total_reports=len(reports),
            successful_scripts=len(results),
        )

        return results

    # Backward-compatible alias.
    generate_poc_batch = generate_reproduction_script_batch

    async def _retry_reproduction_script_with_error(
        self,
        original_prompt: str,
        failed_code: str,
        error: str,
    ) -> str:
        """Retry reproduction script generation feeding back the syntax error for correction."""
        retry_prompt = (
            f"{original_prompt}\n\n"
            f"## Previous Attempt Error\n"
            f"Your previous code had a syntax error:\n"
            f"```\n{error}\n```\n\n"
            f"Previous code (first 2000 chars):\n"
            f"```python\n{failed_code[:2000]}\n```\n\n"
            f"Fix the syntax error and regenerate the Security Unit Test. "
            f"Respond with ONLY Python code."
        )

        try:
            response = await self._llm.generate(
                system_prompt=_POC_GENERATION_SYSTEM_PROMPT,
                messages=[Message(role="user", content=retry_prompt)],
                max_tokens=4096,
                temperature=0.1,
            )
        except Exception:
            return ""

        return self._parse_poc_response(response.text)

    def _parse_poc_response(self, llm_text: str) -> str:
        """
        Extract Python code from the LLM's reproduction script response.

        Handles:
        - Raw Python code (no fences)
        - Markdown ```python fences
        - Markdown ``` fences without language tag
        - Leading/trailing explanation text around code blocks

        Returns:
            The extracted Python source code, or empty string if extraction fails.
        """
        text = llm_text.strip()
        if not text:
            return ""

        # Try to extract from markdown code fences first
        # Match ```python ... ``` or ``` ... ```
        fence_pattern = re.compile(
            r"```(?:python)?\s*\n(.*?)```",
            re.DOTALL,
        )
        fenced_blocks = fence_pattern.findall(text)
        if fenced_blocks:
            # Use the longest fenced block (most likely the full script)
            code = max(fenced_blocks, key=len).strip()
            if code:
                return code  # type: ignore[no-any-return]

        # If no fences found, check if the entire response looks like Python
        # Heuristic: starts with a docstring, import, or comment
        if (
            text.startswith('"""')
            or text.startswith("'''")
            or text.startswith("import ")
            or text.startswith("from ")
            or text.startswith("#")
        ):
            return text

        # Last resort: find the first line that looks like Python and take
        # everything from there
        lines = text.split("\n")
        start_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith('"""')
                or stripped.startswith("'''")
                or stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("# ")
                or stripped.startswith("def ")
                or stripped.startswith("class ")
            ):
                start_idx = i
                break

        if start_idx >= 0:
            return "\n".join(lines[start_idx:]).strip()

        return ""

    def _validate_poc_syntax(self, poc_code: str) -> str | None:
        """
        Validate that the reproduction script is syntactically valid Python.

        Uses ast.parse() — the code is NOT executed.

        Returns:
            None if valid, or a human-readable error string if invalid.
        """
        try:
            ast.parse(poc_code, filename="<poc>", mode="exec")
        except SyntaxError as exc:
            location = f"line {exc.lineno}" if exc.lineno else "unknown location"
            return f"SyntaxError at {location}: {exc.msg}"

        return None

    def _validate_poc_safety(
        self,
        poc_code: str,
        authorized_targets: list[str],
    ) -> str | None:
        """
        Validate that the reproduction script does not contain dangerous operations.

        Checks:
        1. No forbidden imports (subprocess, socket, ctypes, etc.)
        2. No hardcoded URLs pointing to unauthorized domains
        3. No eval/exec calls (the script should be a straightforward test)

        Args:
            poc_code: The generated Python source code.
            authorized_targets: List of authorized target domains/URLs.
                If empty, URL validation is skipped (offline/localhost-only mode).

        Returns:
            None if safe, or a human-readable error string if unsafe.
        """
        # Parse the AST to inspect imports and calls
        try:
            tree = ast.parse(poc_code, filename="<poc_safety>", mode="exec")
        except SyntaxError:
            return "Code failed to parse (syntax error)"

        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name not in _POC_ALLOWED_IMPORTS:
                        if alias.name in _POC_FORBIDDEN_IMPORTS or module_name in _POC_FORBIDDEN_IMPORTS:
                            return f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    full_path = node.module
                    if full_path in _POC_FORBIDDEN_IMPORTS or root_module in _POC_FORBIDDEN_IMPORTS:
                        return f"Forbidden import: from {node.module}"

            # Check for eval/exec calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "__import__"):
                        return f"Forbidden call: {node.func.id}()"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen", "exec"):
                        return f"Forbidden call: .{node.func.attr}()"

        # Check for hardcoded URLs pointing to non-localhost, non-authorized domains
        if authorized_targets:
            url_pattern = re.compile(
                r"""(?:"|')https?://([^/"'\s:]+)""",
            )
            for match in url_pattern.finditer(poc_code):
                hostname = match.group(1).lower()
                # Allow localhost and loopback
                if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"):
                    continue
                # Allow example.com domains (placeholder)
                if hostname.endswith("example.com") or hostname.endswith("example.org"):
                    continue
                # Check against authorized targets
                is_authorized = any(
                    hostname == target.lower()
                    or hostname.endswith("." + target.lower())
                    for target in authorized_targets
                )
                if not is_authorized:
                    return (
                        f"Unauthorized target domain: {hostname} "
                        f"(authorized: {authorized_targets})"
                    )

        return None

    # ── Z3 expression validation ────────────────────────────────────────────

    def _validate_z3_expression(
        self,
        z3_expr_code: str,
        variable_declarations: dict[str, str],
    ) -> str | None:
        """
        Validate that a Z3 expression can be parsed without errors.

        Returns None if valid, or an error message string if invalid.
        """
        try:
            import z3 as z3_lib
        except ImportError:
            return None  # Can't validate without z3, assume valid

        z3_vars: dict[str, Any] = {}
        for name, z3_type in variable_declarations.items():
            if z3_type == "Int":
                z3_vars[name] = z3_lib.Int(name)
            elif z3_type == "Real":
                z3_vars[name] = z3_lib.Real(name)
            elif z3_type == "Bool":
                z3_vars[name] = z3_lib.Bool(name)
            else:
                z3_vars[name] = z3_lib.Real(name)

        namespace: dict[str, Any] = {"z3": z3_lib, **z3_vars}
        try:
            expr = eval(z3_expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as exc:
            return f"eval error: {exc}"

        if not isinstance(expr, z3_lib.BoolRef):
            return f"produced {type(expr).__name__}, expected z3.BoolRef"

        return None
