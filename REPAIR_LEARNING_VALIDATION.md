# Repair Learning Integration — Validation Guide

## Testing the Repair→Hypothesis→Validation Pipeline

### Unit Tests

**Test 1: Repair Broadcast**

```python
async def test_repair_completed_broadcast():
    """Verify Thymos broadcasts REPAIR_COMPLETED on successful repair."""
    incident = Incident(
        id="test_incident_1",
        source_system="api_server",
        incident_class=IncidentClass.CONTRACT_VIOLATION,
        error_message="route not found",
        context=ApiErrorContext(endpoint="/api/v1/test", status_code=404),
    )
    repair = RepairSpec(
        tier=RepairTier.KNOWN_FIX,
        action="add_route_handler",
        target_system="api_server",
    )

    # Set up mock event bus
    thymos = ThymosService(...)
    events_captured = []
    async def capture(event):
        events_captured.append(event)
    thymos._synapse._event_bus.emit = capture

    # Run _learn_from_success
    await thymos._learn_from_success(incident, repair, diagnosis)

    # Verify event was broadcast
    assert len(events_captured) == 1
    assert events_captured[0].event_type == SynapseEventType.REPAIR_COMPLETED
    assert events_captured[0].data["endpoint"] == "/api/v1/test"
    assert events_captured[0].data["fix_type"] == "add_route_handler"
```

**Test 2: Evo Hypothesis Registration**

```python
async def test_repair_hypothesis_registration():
    """Verify Evo registers procedural hypothesis from repair event."""
    evo = EvoService(...)
    hypothesis_engine = evo._hypothesis_engine

    # Simulate REPAIR_COMPLETED event
    event = type('Event', (), {
        'data': {
            'incident_id': 'incident_123',
            'endpoint': '/api/v1/logos/health',
            'tier': 'KNOWN_FIX',
            'incident_class': 'contract_violation',
            'fix_type': 'add_route_handler',
            'root_cause': 'Missing route handler',
        }
    })()

    await evo._on_repair_completed(event)

    # Verify hypothesis was registered
    active = hypothesis_engine.get_active()
    assert len(active) > 0
    repair_h = [h for h in active if 'add_route_handler' in h.statement]
    assert len(repair_h) == 1
    assert repair_h[0].category == HypothesisCategory.PROCEDURAL
    assert repair_h[0].status == HypothesisStatus.TESTING
    assert repair_h[0].evidence_score >= 1.0
    assert 'incident_123' in repair_h[0].supporting_episodes
```

**Test 3: Hypothesis Lookup by Endpoint**

```python
def test_get_repair_hypotheses_by_endpoint():
    """Verify Evo can return hypotheses for a given endpoint."""
    hypothesis_engine = HypothesisEngine(...)

    # Manually register two repair hypotheses
    h1 = hypothesis_engine.register_repair_hypothesis(
        statement="When /api/v1/logos/health fails, add_route_handler fixes it.",
        formal_test="...",
        endpoint="/api/v1/logos/health",
        fix_type="add_route_handler",
        incident_class="contract_violation",
        source_episode_id="ep_1",
    )
    h2 = hypothesis_engine.register_repair_hypothesis(
        statement="When /api/v1/users fails, restart_service fixes it.",
        formal_test="...",
        endpoint="/api/v1/users",
        fix_type="restart_service",
        incident_class="degradation",
        source_episode_id="ep_2",
    )

    # Query for /api/v1/logos
    matches = hypothesis_engine.get_repair_hypotheses("/api/v1/logos")
    assert len(matches) == 1
    assert matches[0].id == h1.id

    # Query for /api/v1/users
    matches = hypothesis_engine.get_repair_hypotheses("/api/v1/users")
    assert len(matches) == 1
    assert matches[0].id == h2.id

    # Query for non-existent endpoint
    matches = hypothesis_engine.get_repair_hypotheses("/api/v1/nonexistent")
    assert len(matches) == 0
```

**Test 4: Simula Proposal Validation**

```python
async def test_simula_validates_against_learned_repairs():
    """Verify Simula flags proposals missing learned fixes."""
    simula = SimulaService(...)
    evo = EvoService(...)
    simula.set_evo(evo)

    # Register a repair hypothesis in Evo
    evo._hypothesis_engine.register_repair_hypothesis(
        statement="When /api/v1/logos fails, add_route_handler fixes it.",
        formal_test="...",
        endpoint="/api/v1/logos",
        fix_type="add_route_handler",
        incident_class="contract_violation",
        source_episode_id="ep_1",
    )

    # Create proposal that touches /api/v1/logos but omits the fix
    proposal = EvolutionProposal(
        description="Add /api/v1/logos/deprecated endpoint",
        rationale="Deprecate old endpoint",
        category=EvolutionProposalCategory.CODE_CHANGE,
    )

    # Simulate validation step
    logs_captured = []
    original_warning = simula._logger.warning
    simula._logger.warning = lambda *a, **kw: logs_captured.append((a, kw))

    await simula._validate_against_learned_repairs(proposal)

    # Verify warning was logged
    warning_logs = [log for log in logs_captured if 'missing_learned_repairs' in str(log)]
    assert len(warning_logs) > 0

    # Restore logger
    simula._logger.warning = original_warning
```

**Test 5: Deduplication**

```python
def test_repair_hypothesis_deduplication():
    """Verify duplicate repair hypotheses are skipped."""
    hypothesis_engine = HypothesisEngine(...)

    # Register first hypothesis
    h1 = hypothesis_engine.register_repair_hypothesis(
        statement="When /api/v1/logos fails, add_route_handler fixes it.",
        formal_test="...",
        endpoint="/api/v1/logos",
        fix_type="add_route_handler",
        incident_class="contract_violation",
        source_episode_id="ep_1",
    )
    assert h1 is not None

    # Try to register duplicate
    h2 = hypothesis_engine.register_repair_hypothesis(
        statement="When /api/v1/logos fails, add_route_handler fixes it.",
        formal_test="...",
        endpoint="/api/v1/logos",
        fix_type="add_route_handler",
        incident_class="contract_violation",
        source_episode_id="ep_2",
    )
    assert h2 is None  # Skipped due to dedup

    # Verify only one is active
    active = hypothesis_engine.get_active()
    repair_h = [h for h in active if 'add_route_handler' in h.statement]
    assert len(repair_h) == 1
```

---

### Integration Tests

**Test 6: End-to-End Repair Flow**

```python
async def test_end_to_end_repair_learning():
    """
    Full integration: error → repair → broadcast → hypothesis → validation.
    """
    # Setup
    thymos = await create_thymos_service(enable_synapse=True, enable_evo=True)
    evo = thymos._evo
    synapse = thymos._synapse
    simula = await create_simula_service()
    simula.set_evo(evo)

    # Step 1: Thymos reports and repairs an error
    incident = Incident(
        id="integration_test_1",
        source_system="api",
        incident_class=IncidentClass.CONTRACT_VIOLATION,
        error_message="POST /api/v1/logos returns 404",
        context=ApiErrorContext(endpoint="/api/v1/logos", status_code=404),
    )
    repair = RepairSpec(
        tier=RepairTier.KNOWN_FIX,
        action="add_route_handler",
        target_system="api",
        reason="Handler not registered",
    )

    # Mark incident as resolved and feed learning
    incident.repair_successful = True
    incident.resolution_time_ms = 500
    await thymos._learn_from_success(incident, repair, diagnosis=None)

    # Allow async events to process
    await asyncio.sleep(0.5)

    # Step 2: Verify hypothesis was registered in Evo
    learned_h = evo._hypothesis_engine.get_repair_hypotheses("/api/v1/logos")
    assert len(learned_h) > 0, "Hypothesis should be registered after repair broadcast"
    assert learned_h[0].category == HypothesisCategory.PROCEDURAL
    assert "add_route_handler" in learned_h[0].statement

    # Step 3: Create proposal that touches the same endpoint
    proposal = EvolutionProposal(
        description="Add rate limiting to /api/v1/logos endpoint",
        rationale="Protect from abuse",
        category=EvolutionProposalCategory.CODE_CHANGE,
    )
    proposal.id = "test_proposal_1"

    # Step 4: Validate proposal — should flag missing known fix
    logs_captured = []
    original_warning = simula._logger.warning
    def capture_log(*args, **kwargs):
        logs_captured.append((args, kwargs))
    simula._logger.warning = capture_log

    await simula._validate_against_learned_repairs(proposal)

    missing_repair_logs = [
        log for log in logs_captured
        if 'missing_learned_repairs' in str(log)
    ]
    assert len(missing_repair_logs) > 0, "Proposal should be flagged for missing repair"

    # Step 5: Create corrected proposal that includes the fix
    proposal_fixed = EvolutionProposal(
        description="Add rate limiting to /api/v1/logos endpoint. Ensure add_route_handler is invoked during init.",
        rationale="Protect from abuse",
        category=EvolutionProposalCategory.CODE_CHANGE,
    )
    proposal_fixed.id = "test_proposal_2"

    logs_captured.clear()
    await simula._validate_against_learned_repairs(proposal_fixed)

    missing_repair_logs = [
        log for log in logs_captured
        if 'missing_learned_repairs' in str(log)
    ]
    assert len(missing_repair_logs) == 0, "Corrected proposal should not be flagged"

    # Restore logger
    simula._logger.warning = original_warning

    print("✓ End-to-end repair learning test passed")
```

---

### Manual Testing

**Scenario: API Endpoint 404**

1. **Trigger error**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/logos/health
   # Returns 404 - handler not registered
   ```

2. **Monitor Thymos logs**:
   ```
   incident_created: incident_class=contract_violation, status_code=404
   repair_prescribed: action=add_route_handler, tier=KNOWN_FIX
   repair_completed_broadcast: endpoint=/api/v1/logos/health, tier=KNOWN_FIX
   ```

3. **Monitor Evo logs**:
   ```
   repair_pattern_learned: hypothesis_id=h_1234, endpoint=/api/v1/logos/health, fix_type=add_route_handler
   ```

4. **Monitor Simula logs** (when new proposal created):
   ```
   proposal_missing_learned_repairs: endpoint=/api/v1/logos, missing_count=1, missing_fix_summaries=[...]
   ```

---

### Observability

**Key Log Events**:

| Event | Logger | Level | Meaning |
|-------|--------|-------|---------|
| `repair_completed_broadcast` | thymos | INFO | Repair broadcast emitted |
| `repair_pattern_learned` | evo | INFO | Hypothesis registered |
| `proposal_missing_learned_repairs` | simula | WARNING | Proposal flagged |
| `learned_repair_validation_failed` | evo | DEBUG | Validation error (non-fatal) |

**Metrics** (if metrics collector available):

| Metric | System | Sample Values |
|--------|--------|----------------|
| `thymos.api.repair_outcome` | thymos | `{outcome: "success", tier: "KNOWN_FIX"}` |
| `repair_hypotheses_testing` | evo | Count of active repair hypotheses |
| `proposal_missing_learned_repairs_count` | simula | Count of flagged proposals |

---

## Debugging

### Hypothesis Not Registered?

Check:
1. Is `register_on_synapse()` being called during Evo init?
2. Is Synapse event bus wired to Evo before repairs happen?
3. Is repair tier ≥ PARAMETER? (NOOP repairs don't broadcast)
4. Are there existing duplicates preventing registration?

```python
# Inspect Evo state
active_hypotheses = evo._hypothesis_engine.get_all_active()
repair_hypotheses = [h for h in active_hypotheses if h.category == HypothesisCategory.PROCEDURAL]
print(f"Repair hypotheses: {len(repair_hypotheses)}")
for h in repair_hypotheses:
    print(f"  - {h.statement[:80]}")
```

### Simula Not Flagging Proposals?

Check:
1. Is `set_evo()` wired in Simula?
2. Does proposal description mention the endpoint? (heuristic matching)
3. Is the hypothesis actually in Evo's active registry?

```python
# Manually test endpoint extraction
endpoints = simula._extract_endpoints_from_proposal(proposal)
print(f"Extracted endpoints: {endpoints}")

# Manually query Evo
learned = evo._hypothesis_engine.get_repair_hypotheses(endpoints[0])
print(f"Learned hypotheses for {endpoints[0]}: {len(learned)}")
```

### High False-Positive Rate?

Adjust heuristics in `_extract_endpoints_from_proposal()`:
- Add/remove regex patterns
- Whitelist known system names
- Increase confidence threshold for method patterns

---

## Performance Considerations

- **Repair broadcast latency**: ~1ms (async, fire-and-forget)
- **Hypothesis registration**: ~50ms (dedup check + Neo4j persist)
- **Proposal validation**: ~200ms (endpoint extraction + hypothesis lookup)
- **Memory overhead**: ~1KB per repair hypothesis (small)

---

## Rollout Strategy

1. **Phase 1**: Enable in dev/staging
   - Monitor logs for errors
   - Verify broadcast/registration working

2. **Phase 2**: Enable in production (read-only mode)
   - Log warnings but don't block proposals
   - Gather metrics on false-positive rate

3. **Phase 3**: Enable soft blocking
   - Flag high-confidence mismatches
   - Route to HITL for governance review

4. **Phase 4**: Monitor KPIs
   - Track prevention rate (target: ≥70%)
   - Adjust hypothesis thresholds as needed

