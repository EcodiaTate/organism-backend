# Repair Learning Integration: Closing the Self-Healing Loop

This document describes the complete pipeline that turns successful API repairs into learned patterns that Evo uses to prevent future errors and validate Simula proposals.

---

## Overview

**Goal**: After Thymos successfully repairs an API error, capture the fix pattern and feed it into Evo's hypothesis generation so the organism learns to avoid the error in the future.

**Flow**:
```
Thymos repairs error
  → REPAIR_COMPLETED event broadcast
    → Evo extracts pattern + registers procedural hypothesis
      → Hypothesis stored with repair metadata
        → Simula queries hypotheses during proposal validation
          → Flag proposals touching known-failure endpoints
```

---

## Architecture

### 1. Thymos: Repair Outcome → Broadcast

**File**: `systems/thymos/service.py`

**Method**: `_learn_from_success(incident, repair, diagnosis)` (line ~1805)

When a Tier 2+ repair succeeds:

1. **Persist incident** to Neo4j (for causal knowledge graph)
2. **Broadcast repair completion** via `_broadcast_repair_completed(incident, repair)`:
   - Emits `SynapseEventType.REPAIR_COMPLETED` with payload:
     - `repair_id`: incident ID
     - `incident_id`: source incident
     - `endpoint`: affected API endpoint (or system path)
     - `tier`: RepairTier name (e.g., `"KNOWN_FIX"`)
     - `incident_class`: IncidentClass value (e.g., `"contract_violation"`)
     - `fix_type`: repair action (from `RepairSpec.action`)
     - `root_cause`: diagnosed root cause
     - `antibody_id`: crystallized antibody ID (if applicable)
     - `duration_ms`: repair duration in milliseconds
     - `fix_summary`: human-readable description for Atune

3. **Feed to Evo** as an Episode for online learning

**Filter**: Only Tier 2+ (PARAMETER and above) repairs produce broadcast. NOOP repairs and transient retries do not.

**Code location**: `_broadcast_repair_completed()` at line ~2000

```python
async def _broadcast_repair_completed(self, incident: Incident, repair: RepairSpec) -> None:
    """Emit REPAIR_COMPLETED on Synapse event bus for Evo and Simula."""
    if self._synapse is None or repair.tier < RepairTier.PARAMETER:
        return

    await self._synapse._event_bus.emit(
        SynapseEvent(
            event_type=SynapseEventType.REPAIR_COMPLETED,
            source_system="thymos",
            data={
                "repair_id": incident.id,
                "incident_id": incident.id,
                "endpoint": endpoint,
                "tier": repair.tier.name,
                "incident_class": incident.incident_class.value,
                "fix_type": repair.action,
                "root_cause": incident.root_cause_hypothesis or "",
                "antibody_id": incident.antibody_id,
                "cost_usd": 0.0,
                "duration_ms": incident.resolution_time_ms or 0,
                "fix_summary": fix_summary,
            },
        )
    )
```

---

### 2. Synapse: Event Type Definition

**File**: `systems/synapse/types.py`

**Added**: `SynapseEventType.REPAIR_COMPLETED` (after `BENCHMARK_REGRESSION`)

```python
REPAIR_COMPLETED = "repair_completed"
# Payload:
#   repair_id (str), incident_id (str), endpoint (str),
#   tier (str), incident_class (str), fix_type (str),
#   root_cause (str), antibody_id (str|None),
#   cost_usd (float), duration_ms (int),
#   fix_summary (str)
```

---

### 3. Evo: Pattern Extraction → Hypothesis Registration

**File**: `systems/evo/service.py`

**New method**: `_on_repair_completed(event: Any)` (async, line ~1014)

**Subscription**: Added to `register_on_synapse()` (line ~738)

#### Flow:

1. **Subscribe to REPAIR_COMPLETED**:
   ```python
   event_bus.subscribe(SynapseEventType.REPAIR_COMPLETED, self._on_repair_completed)
   ```

2. **Extract repair pattern** from event payload:
   - Build natural-language statement: `"When {endpoint} encounters a {incident_class}, applying '{fix_type}' resolves it."`
   - Build formal test: `"Future incidents on '{endpoint}' should be resolved by '{fix_type}'; if successful in ≥3 applications, pattern holds."`

3. **Register as procedural hypothesis** via `HypothesisEngine.register_repair_hypothesis()`:
   - Category: `HypothesisCategory.PROCEDURAL`
   - Status: `HypothesisStatus.TESTING` (not PROPOSED — we have confirmation)
   - Evidence score: `1.0` (bootstrap with confirmed success)
   - Supporting episodes: `[incident_id]`
   - Complexity penalty: `0.05` (direct repair evidence is simple)

4. **Deduplication**: Skip if same endpoint+fix already active

5. **Queue pattern candidate**: Add to `_pending_candidates` as `PatternType.ACTION_SEQUENCE` for normal hypothesis generation pipeline

#### Code:
```python
async def _on_repair_completed(self, event: Any) -> None:
    """Extract repair pattern and register procedural hypothesis."""
    data = event.data or {}
    endpoint = data.get("endpoint", "")
    fix_type = data.get("fix_type", "")
    incident_class = data.get("incident_class", "")

    statement = f"When {endpoint or incident_class} encounters a {incident_class}, applying '{fix_type}' resolves it."

    h = self._hypothesis_engine.register_repair_hypothesis(
        statement=statement,
        formal_test="...",
        endpoint=endpoint,
        fix_type=fix_type,
        incident_class=incident_class,
        source_episode_id=data.get("incident_id", ""),
    )
```

---

### 4. Hypothesis Engine: Direct Registration

**File**: `systems/evo/hypothesis.py`

**New methods**:
- `register_repair_hypothesis()` — directly create hypothesis (bypasses LLM)
- `get_repair_hypotheses(endpoint)` — query hypotheses for endpoint

#### `register_repair_hypothesis()`:

- **Inputs**: statement, formal_test, endpoint, fix_type, incident_class, source_episode_id
- **Returns**: `Hypothesis | None` (None if at capacity or duplicate)
- **Dedup key**: `"{endpoint}:{fix_type}".lower()` — skip if existing hypothesis contains this pair
- **Output hypothesis**:
  - Category: `PROCEDURAL`
  - Status: `TESTING`
  - Evidence score: `1.0`
  - Supporting episodes: `[source_episode_id]`
  - Complexity penalty: `0.05`

#### `get_repair_hypotheses(endpoint)`:

- **Returns**: all active `PROCEDURAL` hypotheses mentioning the given endpoint
- **Used by**: Simula proposal validation
- **Filter**: `endpoint_lower in h.statement.lower()` and status in (TESTING, SUPPORTED)

---

### 5. Simula: Proposal Validation Against Learned Repairs

**File**: `systems/simula/service.py`

**New initialization**: `set_evo(evo)` wiring method

**New validation step**: Step 1.2 (after Telos constitutional check, before triage)

#### Flow:

1. **Wire Evo** during bootstrap (in main.py integration pass):
   ```python
   simula.set_evo(evo)
   ```

2. **During proposal processing** (after Step 1.1 Telos check):
   ```python
   await self._validate_against_learned_repairs(proposal)
   ```

3. **Validation logic** (`_validate_against_learned_repairs()`):
   - Extract endpoint targets from proposal description (regex heuristics: `/api/v*`, system names, method patterns)
   - For each endpoint, query Evo's hypothesis engine: `get_repair_hypotheses(endpoint)`
   - Check if proposal description mentions the known fixes
   - If proposal touches endpoint X but doesn't include the learned fix: **log warning**

4. **Soft warning, not blocking**:
   - Log to `proposal_missing_learned_repairs` with:
     - `proposal_id`, `endpoints`, `flagged_hypothesis_count`, `missing_count`
     - `missing_fix_summaries` (h.statement[:80] for each missing fix)
   - High-confidence mismatches (evidence_score > 2.0) logged for potential HITL escalation
   - Proposal proceeds through the normal pipeline

#### Code:
```python
async def _validate_against_learned_repairs(self, proposal: EvolutionProposal) -> None:
    """Soft validation: check proposal against Evo's learned repair patterns."""
    if self._evo is None:
        return

    endpoints = self._extract_endpoints_from_proposal(proposal)
    hypothesis_engine = getattr(self._evo, "_hypothesis_engine", None)
    if not hypothesis_engine:
        return

    for endpoint in endpoints:
        learned = hypothesis_engine.get_repair_hypotheses(endpoint)
        if learned:
            # Check if proposal includes the learned fix
            if not any(fix_type in proposal.description for fix_type in learned):
                self._logger.warning("proposal_missing_learned_repairs", ...)
```

---

## Data Types

### SynapseEvent Payload (REPAIR_COMPLETED)

```python
{
    "event_type": "repair_completed",
    "source_system": "thymos",
    "data": {
        "repair_id": str,              # Unique ID for this repair
        "incident_id": str,            # Source incident
        "endpoint": str,               # API endpoint or system path (may be empty)
        "tier": str,                   # RepairTier name
        "incident_class": str,         # IncidentClass value
        "fix_type": str,               # Repair action applied
        "root_cause": str,             # Diagnosed root cause
        "antibody_id": str | None,     # Crystallized antibody ID
        "cost_usd": float,             # LLM/compute cost
        "duration_ms": int,            # Resolution time
        "fix_summary": str,            # Human-readable summary
    }
}
```

### Hypothesis (Repair Pattern)

```python
Hypothesis(
    id="h_..." ,
    category=HypothesisCategory.PROCEDURAL,
    statement="When /api/v1/logos fails, apply 'add_missing_route' resolves it.",
    formal_test="Future /api/v1/logos incidents should be resolved by 'add_missing_route'.",
    status=HypothesisStatus.TESTING,
    evidence_score=1.0,
    supporting_episodes=["incident_id_1"],
    complexity_penalty=0.05,
    volatility_flag="normal",
)
```

---

## End-to-End Example

### Scenario: API Returns 404 on `/api/v1/logos/health`

**T=0s: Error occurs**
- Client calls `/api/v1/logos/health`
- Handler not registered → 404 returned
- Sentinel detects → creates Incident

**T=0.5s: Thymos diagnoses & repairs**
- Diagnosis: "Missing route handler"
- Repair: Apply `add_route_handler` to register the endpoint
- Verification: Handler now returns 200 ✓

**T=1.0s: Broadcast repair success**
```python
await event_bus.emit(SynapseEvent(
    event_type=SynapseEventType.REPAIR_COMPLETED,
    data={
        "repair_id": "incident_abc123",
        "endpoint": "/api/v1/logos/health",
        "tier": "KNOWN_FIX",
        "fix_type": "add_route_handler",
        "root_cause": "Missing route registration in logos module",
        "fix_summary": "[/api/v1/logos/health] KNOWN_FIX repair: add_route_handler",
    }
))
```

**T=1.1s: Evo extracts pattern**
```
_on_repair_completed() triggers:
  statement = "When /api/v1/logos/health encounters a contract_violation, applying 'add_route_handler' resolves it."

  register_repair_hypothesis():
    h.id = "h_repair_logos_404"
    h.category = PROCEDURAL
    h.status = TESTING
    h.evidence_score = 1.0
    h.supporting_episodes = ["incident_abc123"]

    → Stored in _active[h.id]
```

**T=2.0s: Future proposal touches /api/v1/logos**
```
Simula.process_proposal(
    description="Add /api/v1/logos/deprecated endpoint"
)

  _validate_against_learned_repairs():
    endpoints = ["/api/v1/logos"]
    learned = hypothesis_engine.get_repair_hypotheses("/api/v1/logos")
      → Returns [h_repair_logos_404]

    if "add_route_handler" not in proposal.description:
        log.warning("proposal_missing_learned_repairs",
            missing_fix_summaries=["When /api/v1/logos/health encounters..."]
        )
        → Soft flag for HITL review
```

---

## Metrics & Monitoring

### Key KPIs

1. **Repair → Hypothesis pipeline health**:
   - `repaired_incidents_tier_2_plus`: Count of Tier 2+ repairs
   - `repair_completed_broadcasts`: Count of events emitted
   - `repair_hypotheses_registered`: Count of new procedural hypotheses

2. **Hypothesis progression**:
   - `repair_hypotheses_testing`: Active hypotheses in TESTING state
   - `repair_hypotheses_supported`: Promoted to SUPPORTED (evidence_score > 3.0, ≥10 episodes)
   - `repair_hypotheses_integrated`: Applied to Simula mutations

3. **Prevention effectiveness**:
   - `proposal_missing_learned_repairs`: Count of proposals flagged (soft warnings)
   - `proposal_repair_pattern_match`: Count of proposals that include learned fixes (success)

4. **Example monitoring query**:
   ```
   Prevention rate = proposal_repair_pattern_match / (proposal_repair_pattern_match + proposal_missing_learned_repairs)
   Target: ≥70% of API proposals touching learned-repair endpoints include the known fix
   ```

---

## Integration Checklist

- [x] **Synapse**: Added `REPAIR_COMPLETED` event type
- [x] **Thymos**: Broadcast from `_learn_from_success()`
- [x] **Evo service**: Subscribe in `register_on_synapse()`
- [x] **Evo service**: Implement `_on_repair_completed()` handler
- [x] **Evo hypothesis engine**: Add `register_repair_hypothesis()` and `get_repair_hypotheses()`
- [x] **Simula service**: Wire `set_evo()` method
- [x] **Simula service**: Add validation step in `process_proposal()` (Step 1.2)
- [x] **Simula service**: Implement `_validate_against_learned_repairs()` and `_extract_endpoints_from_proposal()`

---

## Design Decisions

### Why Direct Hypothesis Registration (Not LLM)?

Successful repairs are **ground truth**. The organism has verified the fix works. Bypassing LLM generation:
- ✓ Eliminates hallucination risk
- ✓ Preserves repair latency (no extra LLM call)
- ✓ Starts hypothesis with bootstrapped evidence score
- ✓ Procedural category is explicit (action sequence, not world model)

### Why Soft Warning in Simula (Not Hard Block)?

Proposals may have legitimate reasons to omit known fixes:
- Different endpoint variant
- New upstream fix makes old repair obsolete
- Cross-cutting concerns (e.g., logging) apply uniformly
- Human knows better than the system

Soft warning allows:
- Simula to proceed through full pipeline
- Equor governance to make final HITL decision
- Metrics to track prediction vs. ground truth

### Why PatternCandidate Queue Too?

Queuing the repair as a pattern candidate feeds into the normal hypothesis-generation pipeline:
- Allows detectors to aggregate similar patterns
- May generate additional world-model hypotheses (e.g., "endpoints with missing handlers cluster in logs module")
- Provides cross-validation: LLM-generated hypothesis should eventually converge with repair-derived one

---

## Future Extensions

1. **Antibody → Hypothesis Linking**:
   - Include antibody effectiveness in hypothesis evidence scoring
   - If antibody effectiveness drops below threshold, deprecate hypothesis

2. **Kairos Validation**:
   - Feed repair causality to Kairos for independent causal validation
   - Boost hypothesis evidence if Kairos confirms root cause

3. **Mutation Synthesis**:
   - When repair hypothesis reaches SUPPORTED, auto-generate a Mutation proposal
   - Encode fix pattern as a reusable patch template

4. **Multi-Endpoint Correlation**:
   - Detect when same fix applies to multiple endpoints
   - Generalize hypothesis: "All endpoints missing handler: apply route registration"

---

## Files Modified

| File | Changes |
|------|---------|
| `systems/synapse/types.py` | Added `REPAIR_COMPLETED` event type |
| `systems/thymos/service.py` | Added `_broadcast_repair_completed()` method, call from `_learn_from_success()` |
| `systems/evo/service.py` | Added `_on_repair_completed()` handler, subscription in `register_on_synapse()`, import `PatternType` |
| `systems/evo/hypothesis.py` | Added `register_repair_hypothesis()` and `get_repair_hypotheses()` methods |
| `systems/simula/service.py` | Added `set_evo()`, `_validate_against_learned_repairs()`, `_extract_endpoints_from_proposal()`, and validation step in `process_proposal()` |

