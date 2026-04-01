# EcodiaOS - Evaluation Test Sets

All files are JSONL (one JSON object per line). Loaded by `TestSetManager` in
`systems/benchmarks/test_sets.py`. Used by the 5-pillar monthly evaluation
protocol (`EvaluationProtocol`) in `systems/benchmarks/evaluation_protocol.py`.

---

## Files

### `domain_tests.jsonl` - 50 items
EOS-specific reasoning tasks across 5 categories (10 each).

```json
{
  "id": "dom_001",
  "prompt": "...",
  "expected_reasoning": "...",
  "category": "economic | safety | planning | causal | social",
  "difficulty": "easy | medium | hard"
}
```

**Categories:**
- `economic` - metabolic gate decisions, yield allocation, child spawn economics
- `safety` - Thymos incident triage, immune response escalation
- `planning` - multi-step goal decomposition with constitutional constraints
- `causal` - do-calculus on EOS causal graph (Kairos-style invariants)
- `social` - federation trust, multi-agent coordination, stakeholder reasoning

**Pillar:** P1 Specialization Index (domain score). Also used as part of P2 Novelty Emergence held-out check.

---

### `general_tests.jsonl` - 50 items
General reasoning tasks (non-EOS-specific). Measures base capability retained
after specialization training (P1 general retention baseline).

```json
{
  "id": "gen_001",
  "prompt": "...",
  "expected_answer": "...",
  "category": "logic | math | language | common_sense | analogy",
  "difficulty": "easy | medium | hard"
}
```

**Categories:**
- `logic` - deductive, inductive, abductive, syllogisms
- `math` - arithmetic, algebra, combinatorics, probability
- `language` - grammar correction, synonym disambiguation, metaphor analysis
- `common_sense` - physical world reasoning, everyday cause-effect
- `analogy` - structural and relational analogies

**Pillar:** P1 Specialization Index (general retention score).

---

### `cladder_tests.jsonl` - 30 items
Causal reasoning benchmark structured on Pearl's 3-rung Ladder of Causation.
Inspired by CLadder (Jin et al., NeurIPS 2023).

```json
{
  "id": "cl_001",
  "rung": 1,
  "question": "...",
  "answer": "...",
  "variables": ["X", "Y"],
  "causal_graph": "X -> Y",
  "commonsensical": true
}
```

**Rungs:**
- Rung 1 (association, items cl_001–cl_010) - "What is?" - observational inference
- Rung 2 (intervention, items cl_011–cl_018) - "What if I do?" - do-calculus
- Rung 3 (counterfactual, items cl_019–cl_030) - "What if I had done?" - retrospective

`commonsensical: true` means real-world causal knowledge helps. `false` means
structural reasoning alone must suffice (anti-memorization probe).

**Pillar:** P3 Causal Reasoning Quality (`l1_association`, `l2_intervention`, `l3_counterfactual`).

---

### `ccr_gb_tests.jsonl` - 20 items
Fictional-world causal reasoning (Maasch et al., ICML 2025 style).
All variable names are invented (Blorp, Fnarg, Quux, Morp, etc.) to prevent
knowledge memorization. Tests whether the RE reasons from world-model
structure alone.

```json
{
  "id": "ccr_001",
  "world_model": "In world Velox: Blorp causes Fnarg. Fnarg causes Quux...",
  "scenario": "We observe Fnarg is present. Did Blorp happen?",
  "ground_truth": "...",
  "reasoning_type": "abduction | intervention | counterfactual"
}
```

**Fictional worlds:** Velox, Zanthos, Praxis, Nimbus, Quorx, Splorg, Torx, Grax, Vrex, Lumex

**Reasoning types:** abduction (8), intervention (6), counterfactual (6)

**Pillar:** P3 Causal Reasoning Quality (`ccr_validity`, `ccr_consistency`).

---

### `constitutional_scenarios.jsonl` - 30 items
Ethical catch-22 dilemmas pitting EOS constitutional drives against each other.
Used to measure Constitutional Drift (P5) - whether the RE's resolution
patterns remain stable and aligned across months.

**FROZEN - these items must NEVER appear in RE training data.**

```json
{
  "id": "eth_001",
  "scenario": "...",
  "drives_in_tension": ["growth", "care"],
  "expected_analysis": "...",
  "is_frozen": true
}
```

**Drive tensions:**
- `growth vs care` (items eth_001–eth_010)
- `honesty vs care` (items eth_011–eth_020)
- `multi-drive` (items eth_021–eth_030)

`expected_analysis` describes the *reasoning process* expected (not a single
correct answer - ethical dilemmas often have legitimate disagreement).

**Pillar:** P5 Ethical Drift Map (`coherence_wins`, `care_wins`, `growth_wins`, `honesty_wins`, `drift_magnitude`).

> **Note:** `test_sets.py:load_constitutional_scenarios()` maps `context`,
> `drives_in_conflict`, `conflict_description`, `expected_resolution_notes`.
> `evaluation_protocol.py:_eval_set()` uses a field priority chain
> (`prompt` → `question` → `context` → `scenario`) and
> (`expected_answer` → `answer` → `expected` → `expected_analysis`) to bridge
> the naming difference transparently.

---

### `held_out_episodes.jsonl` - 20 items
Frozen episodes drawn from real EOS domains. Tests whether the RE can generalize
to genuinely never-seen situations (P2 Novelty Emergence).

**FROZEN - these items must NEVER appear in RE training data. Freeze date: 2026-03-07.**

```json
{
  "id": "held_001",
  "episode_context": "...",
  "expected_action": "...",
  "expected_reasoning_quality": "...",
  "freeze_date": "2026-03-07",
  "is_frozen": true,
  "domain": "software_engineering | ethical_decision | social | economic | causal_reasoning | planning | safety"
}
```

**Domains:** software_engineering (3), ethical_decision (3), social (3),
economic (3), causal_reasoning (3), planning (3), safety (2)

**Pillar:** P2 Novelty Emergence (`success_rate`, `cosine_distance_from_training`).

---

## Frozen File Integrity

Files with `is_frozen: true` items are write-protected at the process level.
The `TestSetManager` logs a warning if it detects any `is_frozen: false` item
in `held_out_episodes.jsonl` or `constitutional_scenarios.jsonl`.

Do NOT add frozen items to training pipelines. The RE training exporter
(`core/re_training_exporter.py`) deduplicates by `episode_id` - ensure frozen
item IDs (`held_001`–`held_020`, `eth_001`–`eth_030`) are added to the
`TRAINING_EXCLUSION_LIST` when the pipeline is wired.

---

## Expanding Test Sets

The current counts (50/50/30/20/30/20) are the Round 2C seed set.
Target counts per the speciation bible: 200/200/200/100/100/100.

When expanding:
1. Maintain schema exactly (the `TestSetManager` parsers expect these fields)
2. Preserve existing item IDs - do not renumber
3. New CLadder items: balance rungs (approx 1/3 each)
4. New CCR.GB items: use only invented variable names - never real-world variables
5. New `constitutional_scenarios.jsonl` items: freeze immediately (`is_frozen: true`)
6. New `held_out_episodes.jsonl` items: freeze immediately + add IDs to exclusion list
