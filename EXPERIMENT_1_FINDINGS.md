# ARGUS — Experiment 1 Findings: Rule-Based vs Learned Governors

## Setup
- Dataset v2: 648 scenarios, 9 categories, Email + Files, EXECUTE/ASK/BLOCK.
- One category (`delete_decision`) has genuine irreducible label noise:
  gold label = f(two observable signals) with a 15% random flip, so no
  governor relying only on those signals can reach 100% on it.
- All other 8 categories are deterministic given their features.
- 3 governors compared: hand-written Rule-Based, Logistic Regression
  (scaled), XGBoost. 10 random seeds, each a fresh dataset + train + eval.

## What we ruled out first
The first two full runs hit 100% accuracy for every governor on both
IID and OOD. Root cause: features were direct copies of the label-
generating logic (an oracle `important: true/false` flag read straight
back out), so no governor could fail by construction — this measured
feature/label self-consistency, not governor competence. Fixed by
replacing the oracle flag with two indirect, imperfect signals
(`last_modified_days_ago`, `referenced_elsewhere`) plus 15% label noise,
and by auditing every other field (preconditions, etc.) for the same
kind of leak.

## Headline result (10-seed means, 95% CI)

| Governor    | IID acc          | OOD acc           | OOD false-intervention rate |
|-------------|-------------------|--------------------|------------------------------|
| rule_based  | 0.965 ± 0.015     | 0.974 ± 0.008      | 0.028 ± 0.017                |
| logistic    | 0.965 ± 0.015     | 0.949 ± 0.037      | 0.109 ± 0.099                |
| xgboost     | 0.962 ± 0.014     | 0.974 ± 0.008      | 0.028 ± 0.017                |

- **No governor beats the hand-written rule.** XGBoost matches it
  exactly on OOD; logistic is worse and far less stable.
- **All errors, for all governors, are confined to `delete_decision`**
  (confirmed via category-level breakdown) — the deterministic
  categories are handled perfectly by all three, as expected.
- **Logistic's OOD instability is real but seed-dependent, not uniform.**
  One seed (1001) drove most of the gap: 15/22 `delete_decision` OOD
  cases wrong, all in the same direction (gold=EXECUTE, predicted=BLOCK).

## Why logistic destabilizes on some seeds but rule_based doesn't
Coefficient inspection ruled out the obvious hypothesis (logistic
leaning on category-identity features like `tool_delete_file` instead
of the real signals) — the weight rankings are nearly identical between
a bad seed (1001) and a good seed (1000). The likely explanation instead:
Rule-Based consults only the two relevant signals per decision type,
completely isolated from the rest of the feature space. Logistic fits
**one shared linear boundary across all 33 features and all categories**;
a distributional wobble anywhere in that space (not necessarily in
`delete_decision` itself) can shift the effective threshold applied to
`delete_decision` cases in a particular random draw. This wasn't fully
confirmed at the decision-function level (would need probability/logit
inspection, not just coefficients) — flagged as an open item, closed
without further digging (diminishing returns relative to the project's
actual goals).

## Answered vs still open (project's own H1–H4)
**Answered (first-pass):**
- Structured features are sufficient to reach ~97% given a rule-based
  policy — but that's a ceiling set by our own label-generating logic,
  not evidence structured features are sufficient for a harder,
  real-world task.
- XGBoost does not clearly beat hand-written rules in this setting.
- A simple rule policy is more robust to unrelated distribution shift
  than a shared linear model — likely because it's per-category-isolated
  rather than globally parameterized. (Needs decision-function level
  confirmation.)

**Still open:**
- Whether this holds under a harder OOD split (composition/policy
  shift, not just entity/phrasing shift).
- Whether ARGUS improves a real agent's behavior, not just synthetic
  scenario classification.
- Whether a small governor can compete with an LLM-as-judge.
- Cost/latency tradeoffs.

## Phase 2 — Mock Agent + end-to-end (DONE)
`argus/agents/base_agent.py`, `mock_agent.py`, `scripts/run_agent_demo.py`.
Acceptance test passed on first run: ambiguous request -> ASK (no
execution), unambiguous request -> EXECUTE (real sandbox execution),
prompt-injection request -> BLOCK (no execution).

## Phase 3 — End-to-End Benchmark (DONE): first main paper experiment
135 held-out scenarios (test_iid + test_ood — NOT the 200 in the
original plan; train/validation excluded to avoid biasing XGBoost's
numbers with data it was fit on).

| System            | Task Success | Unsafe Exec Rate | False Interv Rate | Severity-Wt Error |
|-------------------|--------------|-------------------|---------------------|---------------------|
| agent_alone       | 0.407        | **1.000**         | 0.000               | 4.948               |
| agent_rule_based  | 0.970        | 0.021             | 0.055               | 0.141               |
| agent_xgboost     | 0.970        | 0.021             | 0.055               | 0.141               |

**Headline finding:** with no review, the agent executes every proposed
call including 100% of scenarios that should have been blocked. Adding
any governor (even the plain rule-based one) drops severity-weighted
error ~35x. This is the paper's first direct evidence that ARGUS
improves a real agent's safety, not just scenario-classification
accuracy in isolation.

rule_based and xgboost are identical on this benchmark (same confusion
matrix) — consistent with the Experiment 1 multi-seed result that
they're statistically indistinguishable on this dataset.

## Next stage: Phase 4 — LLM-as-a-Judge comparison (IN PROGRESS)
`argus/governors/llm_judge.py`, `scripts/run_llm_judge_benchmark.py`.
Compares agent_rule_based / agent_xgboost / agent_llm_judge on safety
AND real measured cost/latency (real API calls, real token counts).
Then Phase 5 (Hard OOD: compound failures, prompt injection variants,
stale memory, policy shift) and Phase 6 (paper figures).

## Future work: trajectory-level failures (not addressed by current MVP)
ARGUS reviews each proposed tool call in isolation — it has no memory of
prior proposals/outcomes within a task. This means it cannot currently
catch "bug-loop" style failures (an agent oscillating between two bad
states, each locally plausible but never converging — see Ilya
Sutskever's "models that look smart but are economically slow" framing).
The infrastructure is partially in place: `sandbox.py`'s
`execution_log` already records every attempted call, but no feature in
`extractor.py` reads it yet. The taxonomy already anticipates this
(`CROSS_TOOL_INCONSISTENCY`, `MEMORY_INDUCED_TOOL_DRIFT` in
`DEFERRED_CATEGORIES`) — a natural extension once the current phases
are done: an "oscillation/repetition" feature comparing the current
proposal's arguments against recent history, letting an external,
trajectory-aware governor catch loops the agent itself can't see from
inside its own context window.
