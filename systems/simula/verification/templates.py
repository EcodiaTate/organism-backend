"""
EcodiaOS -- Dafny Specification Templates (Stage 2A)

Pre-built Dafny spec skeletons for common EOS patterns.
The Clover loop uses these as seed templates, reducing the
number of iteration rounds needed for convergence.

Templates cover:
  - Budget adjustment (bounds checking)
  - Risk scoring (level classification)
  - Governance gating (category predicates)
  - Resource cost (non-negative estimates)
  - Constitutional alignment (bounded scores)
  - Executor contract (interface verification)
"""

from __future__ import annotations

TEMPLATES: dict[str, str] = {}


# ── Budget Adjustment Template ───────────────────────────────────────────────

TEMPLATES["budget_adjustment"] = """\
// Budget Adjustment Verification
// Ensures budget parameters remain within safe bounds after adjustment.

method VerifyBudgetAdjustment(
    oldValue: real, newValue: real, maxDeltaPercent: real
) returns (safe: bool)
    requires oldValue >= 0.0
    requires newValue >= 0.0
    requires maxDeltaPercent > 0.0
    ensures safe ==> newValue >= 0.0
    ensures safe ==> (oldValue == 0.0 || Abs(newValue - oldValue) <= maxDeltaPercent * oldValue)
{
    if oldValue == 0.0 {
        safe := newValue >= 0.0;
    } else {
        var delta := Abs(newValue - oldValue);
        safe := delta <= maxDeltaPercent * oldValue;
    }
}

function Abs(x: real): real
{
    if x >= 0.0 then x else -x
}
"""


# ── Risk Scoring Template ────────────────────────────────────────────────────

TEMPLATES["risk_scoring"] = """\
// Risk Score Computation Verification
// Ensures risk level is correctly classified from regression rate.

datatype RiskLevel = Low | Moderate | High | Unacceptable

method ClassifyRisk(
    regressionRate: real,
    thresholdUnacceptable: real,
    thresholdHigh: real
) returns (risk: RiskLevel)
    requires 0.0 <= regressionRate <= 1.0
    requires 0.0 < thresholdHigh < thresholdUnacceptable <= 1.0
    ensures regressionRate > thresholdUnacceptable ==> risk == Unacceptable
    ensures thresholdHigh < regressionRate <= thresholdUnacceptable ==> risk == High
    ensures 0.01 < regressionRate <= thresholdHigh ==> risk == Moderate
    ensures regressionRate <= 0.01 ==> risk == Low
{
    if regressionRate > thresholdUnacceptable {
        risk := Unacceptable;
    } else if regressionRate > thresholdHigh {
        risk := High;
    } else if regressionRate > 0.01 {
        risk := Moderate;
    } else {
        risk := Low;
    }
}
"""


# ── Governance Gate Template ─────────────────────────────────────────────────

TEMPLATES["governance_gate"] = """\
// Governance Gating Verification
// Ensures governed categories always require governance approval
// and forbidden categories are always rejected.

datatype ChangeCategory =
    AddExecutor | AddInputChannel | AddPatternDetector | AdjustBudget
    | ModifyContract | AddSystemCapability | ModifyCycleTiming | ChangeConsolidation
    | ModifyEquor | ModifyConstitution | ModifyInvariants | ModifySelfEvolution

predicate RequiresGovernance(cat: ChangeCategory)
{
    cat == ModifyContract
    || cat == AddSystemCapability
    || cat == ModifyCycleTiming
    || cat == ChangeConsolidation
}

predicate IsForbidden(cat: ChangeCategory)
{
    cat == ModifyEquor
    || cat == ModifyConstitution
    || cat == ModifyInvariants
    || cat == ModifySelfEvolution
}

predicate IsSelfApplicable(cat: ChangeCategory)
{
    cat == AddExecutor
    || cat == AddInputChannel
    || cat == AddPatternDetector
    || cat == AdjustBudget
}

// These predicates partition the category space.
lemma CategoryPartition(cat: ChangeCategory)
    ensures IsForbidden(cat) || RequiresGovernance(cat) || IsSelfApplicable(cat)
    ensures !(IsForbidden(cat) && RequiresGovernance(cat))
    ensures !(IsForbidden(cat) && IsSelfApplicable(cat))
    ensures !(RequiresGovernance(cat) && IsSelfApplicable(cat))
{}

method ProcessProposal(cat: ChangeCategory)
    returns (rejected: bool, needsGovernance: bool, selfApplicable: bool)
    ensures IsForbidden(cat) ==> rejected
    ensures !IsForbidden(cat) && RequiresGovernance(cat) ==> needsGovernance
    ensures !IsForbidden(cat) && IsSelfApplicable(cat) ==> selfApplicable
    ensures rejected ==> !needsGovernance && !selfApplicable
{
    rejected := IsForbidden(cat);
    needsGovernance := !rejected && RequiresGovernance(cat);
    selfApplicable := !rejected && IsSelfApplicable(cat);
}
"""


# ── Resource Cost Template ───────────────────────────────────────────────────

TEMPLATES["resource_cost"] = """\
// Resource Cost Estimation Verification
// Ensures cost estimates are non-negative and budget headroom is respected.

method EstimateResourceCost(
    llmTokensPerHour: int,
    computeMsPerCycle: int,
    memoryMb: real,
    currentBudgetUsed: real,
    totalBudget: real
) returns (withinBudget: bool, headroomPercent: real)
    requires llmTokensPerHour >= 0
    requires computeMsPerCycle >= 0
    requires memoryMb >= 0.0
    requires currentBudgetUsed >= 0.0
    requires totalBudget > 0.0
    ensures headroomPercent >= 0.0
    ensures headroomPercent <= 100.0
    ensures withinBudget <==> headroomPercent > 0.0
{
    var used := currentBudgetUsed;
    var remaining := totalBudget - used;
    if remaining <= 0.0 {
        headroomPercent := 0.0;
    } else {
        headroomPercent := (remaining / totalBudget) * 100.0;
        if headroomPercent > 100.0 {
            headroomPercent := 100.0;
        }
    }
    withinBudget := headroomPercent > 0.0;
}
"""


# ── Constitutional Alignment Template ────────────────────────────────────────

TEMPLATES["constitutional_alignment"] = """\
// Constitutional Alignment Verification
// Ensures alignment scores are bounded and composite is correct.

method ComputeConstitutionalComposite(
    coherence: real, care: real, growth: real, honesty: real
) returns (composite: real)
    requires -1.0 <= coherence <= 1.0
    requires -1.0 <= care <= 1.0
    requires -1.0 <= growth <= 1.0
    requires -1.0 <= honesty <= 1.0
    ensures -1.0 <= composite <= 1.0
    ensures composite == (coherence + care + growth + honesty) / 4.0
{
    composite := (coherence + care + growth + honesty) / 4.0;
}

method CheckAlignmentThreshold(
    composite: real, threshold: real
) returns (aligned: bool)
    requires -1.0 <= composite <= 1.0
    requires -1.0 <= threshold <= 1.0
    ensures aligned <==> composite >= threshold
{
    aligned := composite >= threshold;
}
"""


# ── Executor Contract Template ───────────────────────────────────────────────

TEMPLATES["executor_contract"] = """\
// Executor Contract Verification
// Ensures executors handle errors gracefully and return valid results.

datatype ExecutionResult = Success(output: string) | Failure(error: string)

method ExecuteAction(
    actionType: string,
    input: string,
    timeoutMs: int
) returns (result: ExecutionResult)
    requires timeoutMs > 0
    requires |actionType| > 0
    ensures result.Success? ==> |result.output| >= 0
    ensures result.Failure? ==> |result.error| > 0
{
    // Implementation mirrors Python executor logic
    if |input| == 0 {
        result := Failure("Empty input");
    } else {
        result := Success("Executed: " + actionType);
    }
}
"""


# ── Lookup ───────────────────────────────────────────────────────────────────


def get_template(proposal_category: str, spec_type: str = "") -> str | None:
    """
    Return the best-matching Dafny template for the given context.

    Args:
        proposal_category: ChangeCategory value (e.g. "modify_contract").
        spec_type: Optional specific template name.

    Returns:
        Dafny template string, or None if no match.
    """
    # Direct spec_type lookup first
    if spec_type and spec_type in TEMPLATES:
        return TEMPLATES[spec_type]

    # Category-based lookup
    category_map: dict[str, str] = {
        "adjust_budget": "budget_adjustment",
        "modify_contract": "governance_gate",
        "add_system_capability": "resource_cost",
        "modify_cycle_timing": "risk_scoring",
        "change_consolidation": "governance_gate",
        "add_executor": "executor_contract",
    }
    key = category_map.get(proposal_category)
    if key:
        return TEMPLATES.get(key)

    return None
