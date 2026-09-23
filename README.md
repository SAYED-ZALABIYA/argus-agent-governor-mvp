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
