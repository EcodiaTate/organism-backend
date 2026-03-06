"""
EcodiaOS — Legal Entity Provisioning Interface

Thin client wrappers and types for legal entity formation.
Handles LLC / DAO provisioning through registered agents (Stripe Atlas,
Doola, etc.) with mandatory Human-in-the-Loop (HITL) gates for KYC
and wet-signature steps that an autonomous organism cannot perform.

Sub-modules:
    types             — Data models for entity provisioning lifecycle
    document_engine   — Constitution-to-Operating-Agreement templating
    registered_agent  — Stubbed registered-agent API client
"""
