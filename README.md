<div align="center">

<img width="600" height="289" alt="Basmallah-4-White-940x453" src="https://github.com/user-attachments/assets/5fdd5768-b3f0-4ffe-85f3-585f052c896a" />

</div>

---
<div align="center">

# ARGUS: A Lightweight Reliability Governor for Tool-Using AI Agents (MVP)

</div>

---
> A small decision layer that sits between an ai agent and the tools it calls,
> reviewing every proposed action before it executes.
---
## 1. the problem
AI agents that can call real tools sending emails, deleting files, moving
data cannot reliably tell the difference between an action that's safe to
run immediately, one that needs a clarifying question, and one that should be
refused outright. The common fix today is routing every action through a
large, expensive model for review. That works, but it's slow and costly to
apply consistently to every single step of every task.

--- 
## 2. the idea 
ARGUS is not a new agent. It's a governor placed between an existing agent and its tools.

Before a proposed tool call executes, ARGUS reviews it using interpretable, measurable signals (not the agent's self reported confidence and returns one of:

```
EXECUTE   — safe and unambiguous, let it run
ASK       — something is missing or ambiguous, get clarification first
VERIFY    — check further before deciding (designed for, not yet built — see §7)
BLOCK     — unsafe, unauthorized, or the result of manipulated input
```

The mvp scope deliberately restricts to `EXECUTE` / `ASK` / `BLOCK`, two domains (email, file), and a fixed set of tools (`send_email`, `create_draft`, `delete_email`, `read_file`, `move_file`, `delete_file`). `VERIFY` and broader scope were designed for from day one (see `argus/scenarios/taxonomy.py`) but intentionally deferred. 

---
## 3. architecture 

<div align="center">
<img width="1439" height="811" alt="Screenshot 2026-09-23 150210" src="https://github.com/user-attachments/assets/bd44898f-45dc-4b64-b359-3e6e0d952061" />
</div>

everything runs against a deterministic in memory sandbox (`argus/environment/snadbox.py`) no real email, filesystem, or ever touched.

---
## 4. Project Timeline

### Phase 1 — Synthetic Governor Classification
Built the scenario schema, sandbox, tool registry, feature extractor, and
three governor baselines (hand-written rules, Logistic Regression, XGBoost).
Generated 648 scenarios across 9 categories, split train/validation/
test-IID/test-OOD, with schema-level and registry-level validation on every
scenario.

**Result:** all three governors reached ~96–97% accuracy, IID and OOD alike,
with 95% confidence intervals over 10 random seeds.

### Phase 2 — Mock Agent + End-to-End Pipeline
Added an `Agent` interface, a `MockAgent`, and a demo script proving the
full chain (agent proposes → governor reviews → sandbox executes or
refuses) actually works, not just in theory.

### Phase 3 — End-to-End Benchmark (first real safety evidence)
135 held-out scenarios, agent proposals reviewed and — when approved —
actually executed against the sandbox.

| System | Task Success | Unsafe Execution Rate | Severity-Weighted Error |
|---|---|---|---|
| Agent, no review | 40.7% | **100%** | 4.95 |
| Agent + ARGUS (rule-based) | 97.0% | 2.1% | 0.14 |
| Agent + ARGUS (XGBoost) | 97.0% | 2.1% | 0.14 |

**This is the project's headline result:** with no review layer, every
scenario that should have been blocked was executed anyway. Adding ARGUS —
even the simplest rule-based version — cut that to 2.1% and reduced
weighted error ~35x.

### Phase 5 — Hard-OOD Robustness Benchmark
Because Phase 1–3 results looked suspiciously clean, we built a small
(14-scenario), hand-crafted, **test-only** stress test — never used for
training — split into three tiers:

- **H1 (Prompt Injection Variants):** 5 phrasings of "content tries to make
  the agent do something the user didn't ask for," none matching the
  keyword list the detector was built on (indirect, authority
  impersonation, obfuscated/pretextual, tool-call-shaped, conflicting
  embedded instruction).
- **H2 (Compound Failures):** 6 scenarios with 2–3 simultaneous problems
  (e.g. ambiguous recipient + missing permission + sensitive data at once),
  testing whether severity ordering holds up.
- **H3 (Combined Adversarial):** 3 scenarios composing H1 × H2 — an
  undetectable injection paired with a second, independent red flag, to
  see whether the second signal rescues the decision.

**Final, post-cleanup results** (see §5 for why "post-cleanup" matters):

| Tier | Rule-Based | XGBoost | Takeaway |
|---|---|---|---|
| H1 | 20% acc / 80% unsafe-exec | 20% acc / 80% unsafe-exec | Neither generalizes past the literal keyword list |
| H2 | 100% acc | 83% acc | Explicit rule ordering handles novel combinations better than learned weights |
| H3 | 67% acc / 33% unsafe-exec | 33% acc / 67% unsafe-exec | A second red flag sometimes rescues the decision; not always |

**Honest reading:** ARGUS dramatically improves safety over no review at
all (Phase 3), but has a real, measured, currently-unsolved weakness
against novel prompt-injection phrasing (Phase 5, H1/H3). Both are true at
once — neither cancels the other out.

---

## 5. The Three-Layer Leakage Investigation

This is arguably the most important section of this document, because it's
what makes the Phase 5 results trustworthy. Worth understanding even
briefly if you're evaluating whether to trust these numbers.

The first two full experimental runs showed **100% accuracy for every
governor, on IID and OOD alike** — which is a red flag, not a success.
Chasing that number down surfaced three separate, increasingly subtle
leaks:

**Layer 1 — the obvious oracle.** An early feature read an
`important: true/false` flag straight from the scenario's environment
state — a flag that *was* the answer. Fixed by replacing it with two
indirect, imperfect signals (file recency, cross-reference) plus 15%
injected label noise, so no governor could reach 100% by construction.

**Layer 2 — the bookkeeping leak.** After Layer 1's fix, XGBoost's #1
feature by importance (0.30, far above anything else) was
`missing_precondition_count`. Root cause: every scenario generator sets
`missing_preconditions` to directly restate *why* a scenario isn't
`EXECUTE`, and the schema itself requires that list to be empty whenever
`gold_decision == EXECUTE`. The feature was a near-tautological proxy for
"not EXECUTE," not an independent signal. Removed permanently.

**Layer 3 — the authorial-style leak.** Even after Layer 2's fix, XGBoost
still hit 100% on H1 — while `prompt_injection_detected` correctly became
the top feature overall, H1's scenarios all have it at 0 by design, so
something else was compensating. A targeted ablation (retrain excluding
`plan_step_count` / `argument_count`) proved the cause: every hand-written
injection scenario — in the original training generator *and* in the H1
test set — happens to use a 2-step
`["read X", "follow embedded instruction: ..."]` plan, versus 1 step for
ordinary file operations. That's an artifact of how the scenarios were
*written*, not a property a real attack reliably has. Confirmed
definitively: excluding those two features dropped XGBoost's H1 accuracy
from 1.000 to exactly 0.200 — identical to rule-based, missing the exact
same 4 scenarios. Removed permanently.

**Why this belongs in the README, not just a lab notebook:** anyone can
report a clean benchmark number. Reporting the number *after* actively
trying to break it, and documenting the exact mechanism of every leak
found along the way, is what makes a result worth trusting — and it's a
large part of what this project actually demonstrates methodologically.

---

## 6. Current Status & Honest Limitations

**What's validated:**
- ARGUS meaningfully improves agent safety over no review (Phase 3, real
  sandboxed execution, not just label comparison).
- The dataset and feature set have been adversarially audited for leakage
  three times over, with each fix confirmed by re-running the exact test
  that exposed it.
- A simple hand-written rule policy is at least as good as, and sometimes
  more robust than, a trained XGBoost model on this dataset — a genuinely
  useful negative result, not a failure to get ML working.

**What's NOT yet validated:**
- Robustness to prompt-injection phrasing beyond a fixed keyword list
  (Phase 5, H1) — this is the project's clearest known weakness.
- Anything beyond the synthetic sandbox: no real agent framework
  integration, no real user data, no adversarial red-teaming by anyone
  other than the project's own authors.
- Cost/latency comparison against an LLM-as-a-judge baseline (Phase 4) —
  blocked on API access, not yet run.
- Trajectory-level failures: ARGUS reviews each tool call in isolation,
  with no memory of prior proposals in the same task, so it cannot catch
  "bug-loop" style oscillation (an agent repeatedly undoing its own fixes).
  The taxonomy already anticipates this
  (`CROSS_TOOL_INCONSISTENCY`, `MEMORY_INDUCED_TOOL_DRIFT` in
  `DEFERRED_CATEGORIES`) but no feature reads it yet.

---

## 7. Is This a Sellable Product? An Honest Take

**The problem is real and commercially relevant.** AI agents executing
real actions (email, file, CRM, finance operations) without adequate
review is a live, growing risk, and "runtime governance for agent tool
calls" is a genuine, active space.

**This project, as it stands, is not a sellable product.** It's a
rigorously-tested research prototype with:
- No packaging (no installable SDK, no framework integration — LangChain,
  MCP, CrewAI, etc.)
- No demo UI
- A known, unresolved weakness in exactly the area (prompt injection) most
  security buyers would ask about first
- Zero validation outside a sandbox the project's own authors designed

**What it does have that's genuinely valuable:** a working end-to-end
pipeline, a real (not cherry-picked) safety improvement result, and —
unusually — an honestly documented account of where it breaks and why.
That combination is good foundation material for a technical pitch or an
open-source release, but turning it into something a company would deploy
in production is a substantially larger effort (integration engineering,
a real injection-defense strategy, security review, ongoing maintenance)
than what's built so far — realistically a multi-month effort, not a
follow-up session.

**Possible next steps, in increasing order of commitment (none started
yet, listed for reference):**
1. A small interactive demo (single scenario in, decision + reasoning out)
   — much smaller than a full rebuild, good for showing the idea.
2. `VERIFY` as an implemented fourth decision (already designed for in the
   taxonomy, not yet built).
3. A dedicated, non-keyword injection-detection approach — the one piece
   most worth solving before any security claim is credible.
4. Packaging as an installable library with a clean API
   (`ArgusGovernor(...).evaluate(...)`).
5. Framework integration (LangChain / MCP / CrewAI hooks).

None of this is committed — it's a menu, not a plan, to revisit once
there's time and energy for it.

---

## 8. Repository Structure

```
argus/
├── scenarios/
│   ├── taxonomy.py       # GovernorDecision, FailureCategory, RiskLevel
│   ├── schema.py         # Scenario/ToolSpec (pydantic, self-validating)
│   ├── validators.py     # cross-checks scenarios against the tool registry
│   ├── generators.py     # programmatic scenario generation (9 categories)
│   └── hard_ood.py       # H1/H2/H3 — hand-built, test-only
├── environment/
│   └── sandbox.py        # deterministic in-memory email/file environment
├── tools/
│   ├── email_tools.py
│   └── file_tools.py
├── features/
│   └── extractor.py      # Scenario -> 29 interpretable features
├── governors/
│   ├── base.py
│   ├── rule_based.py     # Baseline 1
│   └── llm_judge.py      # Baseline 4 (Phase 4, real API calls)
├── agents/
│   ├── base_agent.py
│   └── mock_agent.py
└── evaluation/
    └── metrics.py         # accuracy, macro F1, Unsafe Execution Rate,
                            # False Intervention Rate, Autonomy Coverage,
                            # Severity-Weighted Error

scripts/
├── build_dataset.py             # generate + split the dataset
├── audit_dataset.py             # leakage/consistency checks before training
├── train_baselines.py           # train logistic + XGBoost
├── evaluate_baselines.py        # IID/OOD evaluation
├── run_multi_seed.py            # 10-seed statistical comparison
├── run_agent_demo.py            # Phase 2 acceptance test
├── run_end_to_end_benchmark.py  # Phase 3 (agent alone vs. agent+ARGUS)
├── run_hard_ood_benchmark.py    # Phase 5 (H1 + H2 + H3)
├── run_llm_judge_benchmark.py   # Phase 4 (needs a real API key)
├── analyze_errors.py            # category-level error breakdown
├── inspect_xgboost_importance.py
└── ablation_no_plan_features.py

EXPERIMENT_1_FINDINGS.md   # detailed experimental log (Phases 1, 3, 5)
```

---

## 9. Reproducing the Results

```bash
pip install -r requirements.txt
python scripts/build_dataset.py
python scripts/audit_dataset.py       # should print "Dataset audit passed"
python scripts/train_baselines.py
python scripts/evaluate_baselines.py
python scripts/run_hard_ood_benchmark.py
```

Every result in this document was reproduced from a clean install at
least once before being reported.
