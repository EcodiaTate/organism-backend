#!/usr/bin/env python3
"""
Test harness: trigger all 9 incident types through Thymos→Simula pipeline.
Verifies end-to-end self-healing for each error category.

Incident Classes:
  1. CRASH — Unhandled exception, system death
  2. DEGRADATION — Slow or incorrect responses
  3. CONTRACT_VIOLATION — Inter-system SLA breach
  4. LOOP_SEVERANCE — Feedback loop not transmitting
  5. DRIFT — Gradual metric deviation from baseline
  6. PREDICTION_FAILURE — Active inference errors elevated
  7. RESOURCE_EXHAUSTION — Budget exceeded
  8. COGNITIVE_STALL — Workspace cycle blocked or empty
  9. ECONOMIC_THREAT — Malicious on-chain activity detected
  10. PROTOCOL_DEGRADATION — DeFi protocol health declining
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

API_URL = "http://127.0.0.1:8000"

TEST_INCIDENTS = [
    {
        "name": "CRASH - Unhandled Exception",
        "incident_class": "crash",
        "description": "Worker process segfault in memory allocator",
        "context": {
            "http_method": "POST",
            "http_path": "/api/v1/nova/process",
            "http_status": 500,
            "request_id": "crash-001",
            "error_type": "SIGSEGV",
            "error_message": "Segmentation fault in malloc (heap corruption detected)",
            "stack_trace": ["malloc+0x1234", "nova_worker+0x5678", "main+0xabcd"],
        },
    },
    {
        "name": "DEGRADATION - Slow Response",
        "incident_class": "degradation",
        "description": "API response time exceeded SLA threshold",
        "context": {
            "http_method": "GET",
            "http_path": "/api/v1/memoria/query",
            "http_status": 200,
            "request_id": "degrad-002",
            "latency_ms": 5000.0,
            "sla_threshold_ms": 1000.0,
            "percentile": 99.0,
        },
    },
    {
        "name": "CONTRACT_VIOLATION - SLA Breach",
        "incident_class": "contract_violation",
        "description": "Synapse broadcast acknowledgement rate below contract",
        "context": {
            "source_system": "synapse",
            "metric": "broadcast_ack_rate",
            "expected_min": 0.95,
            "actual": 0.12,
            "window_cycles": 50,
            "contract_type": "inter_system_sla",
        },
    },
    {
        "name": "LOOP_SEVERANCE - Feedback Broken",
        "incident_class": "loop_severance",
        "description": "Inner-life feedback loop stopped transmitting",
        "context": {
            "loop_name": "belief_update_cycle",
            "source_system": "soma",
            "target_system": "memoria",
            "last_message_age_cycles": 200,
            "expected_interval_cycles": 10,
            "status": "no_heartbeat",
        },
    },
    {
        "name": "DRIFT - Metric Deviation",
        "incident_class": "drift",
        "description": "Nova intent generation rate drifting from baseline",
        "context": {
            "metric": "nova_intent_generation_rate",
            "baseline": 50.0,
            "current": 15.5,
            "deviation_percent": 69.0,
            "detection_window_cycles": 100,
            "drift_direction": "decreasing",
        },
    },
    {
        "name": "PREDICTION_FAILURE - Inference Error",
        "incident_class": "prediction_failure",
        "description": "Fovea predictive model showing degraded accuracy",
        "context": {
            "model": "fovea_attention_predictor",
            "baseline_accuracy": 0.92,
            "current_accuracy": 0.41,
            "sample_count": 1000,
            "confidence_interval": [0.38, 0.44],
            "alert_threshold": 0.75,
        },
    },
    {
        "name": "RESOURCE_EXHAUSTION - Budget Exceeded",
        "incident_class": "resource_exhaustion",
        "description": "Daily API budget exhausted mid-cycle",
        "context": {
            "resource_type": "api_budget",
            "daily_limit": 100.0,
            "current_spend": 105.5,
            "remaining_hours": 3.0,
            "burn_rate_usd_per_hour": 8.2,
            "days_until_reset": 0,
        },
    },
    {
        "name": "COGNITIVE_STALL - Cycle Blocked",
        "incident_class": "cognitive_stall",
        "description": "Synapse broadcast acknowledgement rate critical",
        "context": {
            "system": "synapse",
            "metric": "broadcast_ack_rate",
            "current_value": 0.0,
            "minimum_threshold": 0.3,
            "cycles_below_threshold": 50,
            "diagnosis": "Memory pressure preventing ACK transmission",
        },
    },
    {
        "name": "ECONOMIC_THREAT - Malicious Activity",
        "incident_class": "economic_threat",
        "description": "Unusual on-chain token movement pattern detected",
        "context": {
            "threat_type": "pump_and_dump",
            "token": "ECOS",
            "transaction_count_5m": 523,
            "price_volatility": 0.45,
            "volume_anomaly_zscore": 8.3,
            "confidence": 0.92,
        },
    },
    {
        "name": "PROTOCOL_DEGRADATION - DeFi Health",
        "incident_class": "protocol_degradation",
        "description": "Aave lending pool health factor declining",
        "context": {
            "protocol": "aave",
            "pool": "usdc_weth",
            "health_factor": 1.05,
            "liquidation_threshold": 1.0,
            "trend": "worsening",
            "risk_parameters": {"ltv": 0.8, "slippage": 0.15},
        },
    },
]


async def trigger_incident(incident: dict) -> dict:
    """POST an incident to Thymos error capture endpoint."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{API_URL}/api/v1/thymos/error",
                json={
                    "incident_class": incident["incident_class"],
                    "description": incident["description"],
                    "context": incident["context"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            response.raise_for_status()
            return {
                "status": "success",
                "name": incident["name"],
                "incident_class": incident["incident_class"],
                "response": response.json(),
            }
        except Exception as e:
            return {
                "status": "error",
                "name": incident["name"],
                "incident_class": incident["incident_class"],
                "error": str(e),
            }


async def main():
    print("=" * 80)
    print("THYMOS > SIMULA INTEGRATION TEST")
    print("Triggering all 9 incident classes through self-healing pipeline")
    print("=" * 80)

    results = []
    for i, incident in enumerate(TEST_INCIDENTS, 1):
        print(f"\n[{i}/{len(TEST_INCIDENTS)}] {incident['name']}")
        print(f"  Class: {incident['incident_class']}")
        print(f"  Description: {incident['description']}")

        result = await trigger_incident(incident)
        results.append(result)

        if result["status"] == "success":
            print(f"  ✓ Incident created")
            if "incident_id" in result.get("response", {}):
                print(f"    ID: {result['response']['incident_id']}")
        else:
            print(f"  ✗ Error: {result['error']}")

        await asyncio.sleep(0.5)  # Stagger to avoid thundering herd

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    success = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    print(f"✓ Incidents created: {success}/{total}")
    print("\nWatch the Simula worker terminal for proposals being generated & applied.")
    print("Expected flow per incident:")
    print("  1. incident_created → 2. triage → 3. diagnosis → 4. proposal_received")
    print("  5. simulation → 6. code_agent → 7. verify → 8. applied/rolled_back")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
