# ARGUS — Experiment 1 Findings: Rule-Based vs Learned Governors

## Setup
- Dataset v2: 648 scenarios, 9 categories, Email + Files, EXECUTE/ASK/BLOCK.
- One category (`delete_decision`) has genuine irreducible label noise:
  gold label = f(two observable signals) with a 15% random flip.
- 3 governors compared: hand-written Rule-Based, Logistic Regression
  (scaled), XGBoost. 10 random seeds, each a fresh dataset + train + eval.

## Headline result (10-seed means, 95% CI)

| Governor    | IID acc          | OOD acc           | OOD false-intervention rate |
|-------------|-------------------|--------------------|------------------------------|
| rule_based  | 0.965 ± 0.015     | 0.974 ± 0.008      | 0.028 ± 0.017                |
| logistic    | 0.965 ± 0.015     | 0.949 ± 0.037      | 0.109 ± 0.099                |
| xgboost     | 0.962 ± 0.014     | 0.974 ± 0.008      | 0.028 ± 0.017                |

No governor beats the hand-written rule on this dataset. Logistic is
measurably less stable OOD than either rule_based or XGBoost.

## Phase 2 — Mock Agent + end-to-end (DONE)
Acceptance test passed on first run: ambiguous -> ASK, unambiguous ->
EXECUTE (real sandbox execution), prompt-injection -> BLOCK.

## Phase 3 — End-to-End Benchmark (DONE): first main paper experiment
135 held-out scenarios (test_iid + test_ood).

| System            | Task Success | Unsafe Exec Rate | False Interv Rate | Severity-Wt Error |
|-------------------|--------------|-------------------|---------------------|---------------------|
| agent_alone       | 0.407        | **1.000**         | 0.000               | 4.948               |
| agent_rule_based  | 0.970        | 0.021             | 0.055               | 0.141               |
| agent_xgboost     | 0.970        | 0.021             | 0.055               | 0.141               |

**Headline finding:** with no review, the agent executes every proposed
call including 100% of scenarios that should have been blocked. Adding
any governor drops severity-weighted error ~35x.

## Phase 5 — Hard-OOD Robustness Benchmark (H1 + H2 done, H3 in progress)

Structure: H1 (Prompt Injection Variants) + H2 (Compound Failures) +
H3 (Combined Adversarial, composed from H1×H2). All three are
**test-only** — never added to train/validation splits.

### Methods lesson: three layers of feature leakage found via H1

This is worth documenting on its own, separate from the headline
numbers, because the debugging process is itself a finding.

**Layer 1 (found in Experiment 1, before Phase 2):** an early feature
directly read an `important: true/false` oracle flag that mirrored the
gold label. Fixed by replacing it with two indirect, imperfect signals
plus 15% injected label noise.

**Layer 2 (found via H1):** `missing_precondition_count` /
`satisfied_precondition_ratio` were XGBoost's #1-ranked feature by a
wide margin (importance 0.30, next closest 0.27). Root cause: every
scenario generator sets `missing_preconditions` to directly restate
*why* a scenario isn't EXECUTE, and the schema validator itself
requires `missing_preconditions == []` whenever `gold_decision ==
EXECUTE`. So `missing_precondition_count > 0` was a near-tautological
proxy for "not EXECUTE" — a bookkeeping field, not an independent
signal. Removed permanently.

**Layer 3 (found via H1, after fixing Layer 2):** H1 accuracy for
XGBoost stayed at 1.000 even after Layer 2 was fixed —
`prompt_injection_detected` correctly became the #1 feature globally
(importance 0.39), but H1's scenarios all have that feature at 0 by
design, so something else was compensating. Traced via a targeted
ablation (retrain excluding `plan_step_count` / `argument_count`) to
the fact that every hand-written injection scenario — in the original
training generator AND in H1 — happens to use a 2-step
`["read X", "follow embedded instruction: ..."]` agent_plan, vs. 1 step
for ordinary delete_file calls. That's an artifact of how *I* write
scenarios, not a property a real attack reliably has. Confirmed by the
ablation: excluding those two features dropped XGBoost's H1 accuracy
from 1.000 to exactly 0.200 — identical to rule_based, and the same 4
scenarios missed. Removed permanently.

**Honest post-cleanup result:** after removing all three leaks,
**XGBoost and rule_based perform identically on H1 (0.200 accuracy,
0.800 unsafe-execution rate)** — neither generalizes to injection
phrasing beyond the literal keyword list. H2 (compound failures) was
unaffected by any of the three fixes (rule_based 1.000, XGBoost 0.833
throughout) — its earlier result was already leak-free.

**Why this matters for the paper:** the honest conclusion is that a
keyword-based `prompt_injection_detected` feature is a real, current
limitation for BOTH governor types — this isn't a rule-based-vs-ML
question at all here, it's a feature-representation ceiling. This
directly motivates comparing against an LLM judge (Phase 4): semantic
understanding of injected instructions may need something beyond
structured keyword features, which a strong LLM might provide (at a
cost/latency price this project is built to quantify).

### H1 (final, post-cleanup)
| Governor | Accuracy | Unsafe Execution Rate |
|---|---|---|
| rule_based | 0.200 | 0.800 |
| xgboost | 0.200 | 0.800 |

Only the "conflicting embedded instruction" family (injected text
redirects to a privacy-sensitive move) is caught — and only via the
*privacy* feature, not injection detection. The other 4 families
(indirect, authority impersonation, obfuscated/pretextual,
tool-call-shaped) are missed by both governors.

### H2 (unchanged)
| Governor | Accuracy | Notes |
|---|---|---|
| rule_based | 1.000 | Explicit priority ordering generalizes to novel combinations. |
| xgboost | 0.833 | Misses the one 3-signal compound case (predicts ASK, true severity is BLOCK) — plausibly under-weighs `missing_permission` when `ambiguous_entity` co-occurs, since training only had single-issue examples. |

## Next stage
H3 (Combined Adversarial) — in progress now. Then: revisit Phase 4
(LLM-as-a-judge) once API access is available, informed directly by
the H1 finding (motivates it more strongly than before). Then Phase 6
(consolidated figures).

## Future work: trajectory-level failures (not addressed by current MVP)
ARGUS reviews each proposed tool call in isolation — no memory of prior
proposals/outcomes within a task, so it cannot catch "bug-loop" style
oscillation failures (see Ilya Sutskever's "models that look smart but
are economically slow" framing). `sandbox.py`'s `execution_log` already
records every attempted call; no feature reads it yet. Taxonomy already
anticipates this (`CROSS_TOOL_INCONSISTENCY`, `MEMORY_INDUCED_TOOL_DRIFT`
in `DEFERRED_CATEGORIES`).