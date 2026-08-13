<div align="center">

<img width="600" height="289" alt="Basmallah-4-White-940x453" src="https://github.com/user-attachments/assets/5fdd5768-b3f0-4ffe-85f3-585f052c896a" />

</div>

---
<div align="center">

# ARGUS: A Lightweight Reliability Governor for Tool-Using AI Agents (MVP)

</div>

---
## What exists now

```
argus/
├── scenarios/
│   ├── taxonomy.py   # GovernorDecision, FailureCategory, RiskLevel, Reversibility
│   └── schema.py      # ToolSpec, ProposedToolCall, Scenario (pydantic, self-validating)
├── tools/
│   ├── email_tools.py # search/read/create_draft/send/delete + ToolSpecs
│   └── file_tools.py  # list/read/edit/move/delete + ToolSpecs
└── environment/
    └── sandbox.py      # deterministic in-memory Sandbox — no real network/FS/DB access

scripts/
└── sample_scenarios.py  # 10 hand-built scenarios covering EXECUTE/ASK/BLOCK
```

Run the review step:
```bash
pip install -r requirements.txt
PYTHONPATH=. python3 scripts/sample_scenarios.py
```

Every `Scenario` is validated on construction: a `gold_decision` that
contradicts its `gold_reason_codes` (e.g. EXECUTE with an unresolved
precondition, or ASK/BLOCK with no failure category) raises immediately.
This is meant to catch inconsistent gold labels before they ever reach a
training set.

## Design decisions worth flagging

- **MVP scope is deliberately narrow**: Email + Files only, and only
  `EXECUTE / ASK / BLOCK` (no VERIFY/ESCALATE yet). `MVP_DECISIONS` and
  `is_mvp_compatible()` enforce this so it's easy to tell if a generated
  scenario has drifted out of scope.
- **PROMPT_INJECTION always forces BLOCK** (`ALWAYS_BLOCK_CATEGORIES`),
  independent of any other computed risk score — this is a hard rule,
  not a learned one.
- **Reversibility metadata lives in three places on purpose**: once on
  the `ToolSpec` (general property of the tool), and denormalized onto
  each `Scenario` (property of this specific call, in case future
  scenarios want tool behavior to vary by context). Keep these in sync
  when generating scenarios programmatically.

## Next steps (not yet built)

1. `argus/scenarios/generators.py` — programmatic scenario generation
   (target: 500 scenarios for the MVP, per the failure taxonomy).
2. `argus/scenarios/validators.py` — deterministic pass/fail checkers
   that don't require an LLM judge (e.g. re-derive the gold decision
   from raw environment state + proposed call, and assert it matches).
3. `argus/agents/small_agent.py` + `base_agent.py` — the baseline
   tool-using agent that *proposes* actions (this is what the Governor
   will review — it does not exist yet).
4. `argus/governors/rule_based.py` — first governor baseline
   (`if irreversible and ambiguous: return ASK`), runnable against the
   10 sample scenarios as a smoke test before anything ML-based.
5. Feature extraction (`argus/features/`) — starts only after the
   rule-based baseline is running end-to-end.

## Constraints carried forward from the project spec

- Sandbox only. No real email, filesystem, or database is ever touched.
- Prefer interpretable, low-cost models before neural ones.
- Every scenario must have a deterministic, LLM-free way to check
  correctness.
- Don't expand past Email + Files + 3 decisions until the MVP is
  validated end to end.
