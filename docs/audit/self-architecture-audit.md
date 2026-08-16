# Pi maker/checker/planner architecture audit

**Audit date:** 2026-08-13
**Scope:** local Pi fork in `~/pi`, live verification settings, runtime audit logs, tests, and the self-improvement gauntlet
**Change policy:** audit only. No protected runtime or settings file was modified.
**Overall verdict:** **The per-task maker/checker loop is real and materially useful, but the system is not yet a complete self-improvement loop. It has several high-severity fail-open, evidence-integrity, safety, and observability defects.**

## Method and confidence

This is a static architecture audit plus an observational review of local runtime records. Findings marked **Confirmed** are tied to source paths/lines, settings, or reproducible log behavior. I did not change or live-fault-inject the protected runtime; consequently, exploitability and production frequency are not quantified. Historical JSONL data spans multiple code/config eras and concurrent sessions, so event totals are reported as descriptive counts rather than success rates. The two reference articles were used as design comparators, not as mandatory specifications.

## Executive summary

The customization is substantially stronger than a one-shot “planner hands a plan to a maker” design:

- A cheap/strong model routing layer triages work into `trivial`, `standard`, and `hard` tiers.
- A planner can run before work begins and again after checker rejection.
- Independent checkpoint and final checkers receive tool receipts and repository evidence.
- Rejected work returns to a maker, can escalate to a stronger maker, and has provider fallbacks.
- Checker budgets are bounded and budget exhaustion is labeled `UNVERIFIED`.
- The implementation has extensive deterministic unit coverage.

That refutes the broad claim that there is “no loop” or “no critic.” The micro-loop is genuine.

The main architectural problem is that two different ideas are currently conflated:

1. **A runtime repair loop:** plan → make → check → retry. This exists.
2. **A harness improvement flywheel:** collect representative traces and human feedback → create durable evals → compare a candidate harness against a baseline and holdout → promote or roll back. This is only a partial gauntlet skeleton, and its primary metric contains a concrete implementation bug.

The most important flaws are:

1. **No whole-run stop condition:** ordinary runs have no maximum tool turns, elapsed time, or total cost. The checker budget bounds checker disagreements, not maker execution.
2. **Verification fails open silently:** unavailable or persistently ambiguous checkers return `undefined`; the candidate answer then exits normally without a user-visible unverified status.
3. **Raw, untrusted tool output is sent to external verifier/planner models without redaction or prompt-injection isolation.**
4. **The checker often does not see the real artifact:** evidence is truncated, excludes untracked-file contents, and verifier commands run from the session cwd rather than the relevant project root.
5. **The verdict protocol is brittle:** any occurrence of `CONFLICT` overrides `VERIFIED`, causing observed false rejections of replies beginning with `VERIFIED`.
6. **Planner output is prose, not a validated contract:** live plans exceed the injection limit and are silently cut; the CLARIFY question limit is not enforced by code.
7. **The audit trail cannot support a trustworthy improvement flywheel:** it has no run/session/cwd/config identifiers, only 15 summaries for 37 run starts in the inspected log, and logging failures are swallowed.
8. **Documented model economics do not match enforced routing:** standard/trivial maker and default triage use the session’s base model unless explicitly overridden. The live log shows expensive Anthropic triage and maker routes.
9. **Checkpoint scheduling is phase-blind and post-hoc:** read-only discovery can be rejected as “no progress,” while many tool calls—including destructive ones—can occur before a checker sees them.
10. **The macro gauntlet is not yet a valid eval loop:** its `M1` command iterates over characters rather than conflict records, and there is no baseline/holdout/promotion mechanism.
11. **Long sessions can degenerate without a stall stop:** saved sessions reach thousands of messages/tool calls, while verification summaries omit maker cost.
12. **A “run” is one prompt, not one goal:** a short follow-up such as “continue” resets hard-task state and can be routed as a new trivial task.

I would keep the overall design, but harden the contracts and evidence path before adding more model sophistication.

---

## 1. What was audited

### Runtime and prompt composition

- `packages/coding-agent/src/core/verify-routing.ts`
  - triage, planner prompt, checkpoint/final checker prompts, retries, escalation, fallbacks, audit logging
- `packages/agent/src/agent-loop.ts`
  - loop control, tool execution, checkpoint timing, final termination
- `packages/agent/src/types.ts`
  - verifier hook contracts
- `packages/coding-agent/src/core/sdk.ts`
  - settings/model resolution and hook installation
- `packages/coding-agent/src/core/settings-manager.ts`
  - verification configuration shape and merge behavior

### Configuration and behavioral evidence

- `~/.pi/agent/settings.json` (`verify` block only; read, not modified)
- `~/.pi/agent/verify-audit.jsonl`
- `~/.pi/agent/gauntlet-audit.jsonl`
- `~/pi/.pi/gauntlet/CARD.md`

### Tests

- `packages/coding-agent/test/verify-routing.test.ts`
- `packages/agent/test/agent-loop.test.ts`

### Reference designs

- OpenAI, **Build an Agent Improvement Loop with Traces, Evals, and Codex**
- Anthropic, **Loop engineering: Getting started with loops**

### Customization history

The local commit sequence shows a coherent architecture rather than an accidental prompt pile:

- `4baa6da41` — base `verifyTurn` maker/checker/failsafe hook
- `27374db7e` — verification settings and routing
- `716e3ffda` — independent evidence and thinking floors
- `ca849d00c` — plan-first and mid-build checkpoints
- `b5ce6385c` — run reset, honest exhaustion, verdict guard
- `6e17eb572` — triage tiers, clarify gate, sticky escalation, geometric checkpoints, cost logs
- `7321dfbd6` — plan-by-default clarification policy
- `95d7c4609` — provider fallbacks
- `3ad40729a` — output-token cap
- `276473693` — audits for short real tasks and no-tool answers

This progression addresses real failure modes. The findings below are the next layer, not a claim that the existing work is superficial.

---

## 2. Current architecture

```text
user prompt
    │
    ▼
triage model ───────────────► tier: trivial | standard | hard
    │
    ├─ trivial: usually skip planner/checkpoints
    │
    └─ standard/hard
          │
          ▼
      tool-less planner
          │ PLAN prose or CLARIFY prose
          ▼
      maker with full tools
          │
          ├─ tool-using turn ─► optional geometric checkpoint checker
          │                         │
          │                         ├─ VERIFIED: continue
          │                         └─ CONFLICT: corrective prompt, re-plan/escalate
          │
          └─ no-tool candidate final ─► final checker
                                            │
                                            ├─ VERIFIED: stop
                                            ├─ CONFLICT: retry/re-plan/escalate
                                            └─ ambiguous/unavailable: silently fall through
```

There is also a separate macro layer:

```text
verify-audit.jsonl + gauntlet-audit.jsonl
                 │
                 ▼
          .pi/gauntlet/CARD.md
      command metrics + model critic
                 │
                 ▼
         proposed harness amendment
       (protected changes require approval)
```

The micro-loop is implemented in code. The macro-loop currently produces proposals, but there is not yet evidence that it can reliably convert failures into durable, validated improvements.

---

## 3. Seed hypotheses: confirmed, refuted, or mitigated

| # | Seed hypothesis | Result | Evidence and assessment |
|---|---|---|---|
| 1 | No output contract for PLAN | **Partially confirmed; prose-mitigated** | The prompt asks for a “short, concrete, step-by-step plan,” investigation, assumptions, and verification (`verify-routing.ts`, planner prompt near lines 794–807). No parser expects structured plan fields, so this is not a parser bug. It is still a live control bug: plans are opaque prose, are not validated, and are silently truncated to 4,000 characters before maker injection. The current audit logged an 8,444-character plan. |
| 2 | No loop termination or re-plan protocol | **Partly refuted, partly confirmed** | Re-plan, escalation, checkpoint caps, and rejection handling exist. However, `agent-loop.ts` uses unbounded `while` loops and has no max maker turns, elapsed-time deadline, or total-cost stop. `maxRejections` only bounds one checker-conflict streak. |
| 3 | No evaluator/critic role | **Refuted for the micro-loop; confirmed for durable improvement** | Checkpoint and final checker roles are real and use fresh model calls. What is missing is OpenAI-style conversion of trace feedback into a versioned regression suite and candidate-promotion gate. The gauntlet is an early partial implementation, not yet that flywheel. |
| 4 | No safety rails for a full-tool maker | **Confirmed at the enforcement layer** | The root README explicitly says Pi has no built-in filesystem/process/network/credential permission system. The planner advertises “FULL tool access.” Project instructions discourage dangerous operations, but they are model instructions, not a sandbox or capability boundary. |
| 5 | No anti-hallucination rule for the tool-less planner | **Mitigated, not eliminated** | The planner receives a small repo snapshot and is told to put discoverable facts into investigation steps. It is not explicitly forbidden from asserting unseen paths/line numbers, and it does not receive full project instructions or file contents. |

---

## 4. Findings

Severity meanings:

- **Critical:** can cause unsafe execution, secret disclosure, or a falsely trusted result.
- **High:** regularly undermines correctness, convergence, or auditability.
- **Medium:** material quality/cost/maintenance problem with narrower impact.
- **Low:** polish or defense-in-depth.

### F-01 — No whole-run execution bound

**Severity: Critical**
**Status: Confirmed**
**Primary evidence:** `packages/agent/src/agent-loop.ts:208-212`; bounded verifier settings in `packages/coding-agent/src/core/verify-routing.ts:306-317`.

`agent-loop.ts` has an outer `while (true)` and an inner loop that continues while the maker emits tool calls or queued messages. The custom routing bounds:

- number of checkpoints,
- conflicts in one rejection streak,
- model fallback attempts.

It does **not** bound:

- total maker turns,
- total tool calls,
- elapsed time,
- total model cost,
- total mutated files or bytes.

A maker can keep issuing tools indefinitely. Geometric checkpoints eventually stop after `maxCheckpointsPerRun`; after that, a long-running maker can continue with less oversight. Provider context limits or a human abort are accidental stop mechanisms, not architecture.

This diverges from Anthropic’s goal-loop guidance: stop on **goal achieved or a maximum number of turns**.

**Concrete risk:** a stalled agent can repeatedly inspect, edit, install, call network services, or run expensive tests until manually interrupted.

**Required change:** add independent hard ceilings such as `maxMakerTurns`, `maxToolCalls`, `maxRunMs`, and optionally `maxRunCostUsd`. Exhaustion must end with a visible `STOPPED_UNVERIFIED` result, not be represented as verification.

---

### F-02 — Raw tool output crosses model/provider boundaries without redaction or injection isolation

**Severity: Critical**
**Status: Confirmed**
**Primary evidence:** raw receipt forwarding at `packages/coding-agent/src/core/verify-routing.ts:530-543`; re-planner context forwarding at `:646-660`; external model routes in the live `verify` configuration.

The final/checkpoint checker receives up to ten raw tool results. Re-planning passes the full current model context to the planner. These calls can target external OpenRouter models.

There is no redaction stage for:

- API keys or tokens printed by commands,
- credentials read from files,
- customer data,
- private source content,
- terminal output containing secrets.

There is also no robust prompt-injection boundary. Repository text and command output are inserted as ordinary user-message text. A malicious file or command output can tell the checker to ignore its rubric and reply `VERIFIED`.

**Concrete trace:** a maker that runs `env`, reads an auth file, or receives a poisoned test log can forward that content to a second provider even when the original maker route was chosen for sensitivity.

**Required change:** before every planner/checker call:

1. classify and redact secrets/PII,
2. attach provenance and sensitivity labels,
3. encode evidence as untrusted data rather than instructions,
4. enforce provider-routing policy for the evidence’s sensitivity,
5. log hashes/metadata rather than sensitive text.

Until that exists, sensitive repositories should run inside a sandbox with verifier calls disabled or routed to an approved local/ZDR model.

---

### F-03 — Verification fails open without telling the user

**Severity: High**
**Status: Confirmed**
**Primary evidence:** `packages/coding-agent/src/core/verify-routing.ts:576-585, 925-930`; swallowed hook errors at `packages/agent/src/agent-loop.ts:322-336`.

Two paths return `undefined` from verification:

- checker and fallback are unavailable,
- the checker remains ambiguous after one re-ask.

In the agent loop, `undefined` means “no verifier decision.” A no-tool candidate final then exits normally. There is no `[VERIFY] UNVERIFIED` notice for this path.

The code is honest only for **conflict budget exhaustion**. It is not honest for **checker outage, malformed output, or unresolved ambiguity**.

The inspected log contains six ambiguous audit verdicts, all followed by `skipped: ambiguous-verdict`. There were no `audit-unavailable` events in that sample, but the same silent behavior exists in code.

**Required change:** make verifier outcome explicit:

```text
VERIFIED | REJECTED | UNVERIFIED(reason) | ABORTED
```

For a final answer, `UNVERIFIED` must produce a user-visible notice and a run summary. A configurable fail-closed policy should be available for high-risk projects.

---

### F-04 — The checker’s evidence is incomplete and often rooted in the wrong directory

**Severity: High**
**Status: Confirmed by code and logs**
**Primary evidence:** `packages/coding-agent/src/core/verify-routing.ts:284-297, 480-496, 530-555`; wrong-cwd ENOENT conflicts in `verify-audit.jsonl` on 2026-08-06 and 2026-08-10.

The evidence collector:

- keeps only the last ten tool results,
- keeps only the last 1,200 characters of each,
- truncates the final diff to 6,000 characters,
- does not include untracked-file contents (`git diff HEAD` omits them),
- returns no diff outside a Git worktree,
- does not prove which command output corresponds to which claimed change.

The independent command protocol is a list of shell strings executed with `cwd` equal to the session cwd. The live global configuration uses `npm run check` and `npm test`. Several audit events show those commands running from `/Users/rahul` while the target was `/Users/rahul/quiver`, producing false ENOENT failures and repeated conflicts.

A checker can therefore:

- reject correct work because it ran the right command in the wrong directory,
- verify a large change after seeing only a prefix,
- miss the contents of a newly created file,
- miss early failures because only output tails are retained.

**Required change:** replace shell strings with typed verifier specifications:

```json
{
  "id": "project-tests",
  "argv": ["npm", "test"],
  "cwdPolicy": "nearest-package-root",
  "timeoutMs": 120000,
  "successExitCodes": [0]
}
```

Build an artifact manifest containing all changed and untracked files, hashes, bounded per-file patches, test command + resolved cwd + exit status, and explicit truncation markers. A checker must never return `VERIFIED` when required evidence was truncated or unavailable.

---

### F-05 — The verdict parser creates false conflicts and remains prompt-injectable

**Severity: High**
**Status: Confirmed by code and live audit**
**Primary evidence:** `packages/coding-agent/src/core/verify-routing.ts:126-140`; test contract at `packages/coding-agent/test/verify-routing.test.ts:67-86`; false-conflict audit events at 2026-08-07T08:30:47 and 2026-08-10T08:21:38.

`classifyCheckerVerdict()` searches the entire reply and gives any `CONFLICT` token priority over `VERIFIED`. That behavior is explicitly asserted by a unit test.

The audit log contains at least three concrete false-conflict signatures where:

- the recorded verdict is `conflict`,
- the checker text begins with `VERIFIED`,
- later prose contains the rejection token while discussing prior feedback or the rubric.

This occurred at 2026-08-07T08:30:47, 2026-08-10T08:21:38, and—self-demonstrating during this audit—2026-08-13T06:28:19. In the latest event, the final checker opened with `VERIFIED` and explicitly said the receipts substantiated the report, yet the runtime recorded a rejection because the explanation later quoted the rejection criterion.

The inverse remains possible too: an injected uppercase `VERIFIED` in evidence can be mistaken for a pass if the model echoes it without a conflict token.

**Required change:** use a structured response schema or parse an exact, isolated first-line enum. Never scan arbitrary prose for verdict substrings. Keep rationale in a separate field:

```json
{"verdict":"verified","issues":[],"evidenceGaps":[]}
```

Reject malformed output as `UNVERIFIED`, not as a silent skip.

---

### F-06 — Planner and CLARIFY outputs are not machine-enforced contracts

**Severity: High**
**Status: Confirmed**
**Primary evidence:** `packages/coding-agent/src/core/verify-routing.ts:794-842`; runaway-CLARIFY test at `packages/coding-agent/test/verify-routing.test.ts:650`; current `planned` event reports 8,444 characters.

The PLAN prompt has useful prose constraints, but runtime handling is:

1. accept arbitrary text,
2. log its full character count,
3. inject only the first 4,000 characters.

The current run logged an 8,444-character hard-tier plan and silently discarded more than half before the maker saw it. Verification steps usually occur near the end of a plan, so truncation selectively removes the most important closure criteria.

The CLARIFY branch asks for at most five questions, but the code backstop only truncates to 2,000 characters. Its test constructs 40 questions and asserts only a character cap. The maker is then instructed to filter the list, but no state transition forces it to ask at most five or stop afterward.

**Required change:** require a bounded structured planner result with explicit variants:

```json
{
  "kind": "plan",
  "assumptions": ["..."],
  "steps": [{"action":"...","evidence":"..."}],
  "successCriteria": ["..."],
  "stopConditions": ["..."]
}
```

or

```json
{"kind":"clarify","questions":[{"question":"...","why":"..."}]}
```

Validate `steps <= N`, `questions <= 5`, and serialized size before injection. If invalid, re-ask once, then surface `UNVERIFIED_PLANNER_OUTPUT` rather than slicing it.

---

### F-07 — Audit telemetry cannot support reliable longitudinal improvement

**Severity: High**
**Status: Confirmed**
**Primary evidence:** logger envelope and swallowed writes at `packages/coding-agent/src/core/verify-routing.ts:225-232`; summaries at `:387-398`; observed 37 starts vs 15 summaries.

At inspection time, the global verify log had:

- 305 events,
- 37 `run-start` events,
- 49 checker audits: 21 `verified`, 22 `conflict`, 6 `ambiguous`,
- only 15 `run-summary` events,
- 2 `budget-exhausted` events,
- 22 separate `installed` events representing multiple sessions/config eras.

These are **event counts, not quality rates**. The file lacks:

- run ID,
- session ID,
- cwd/repository ID,
- goal hash or task class beyond tier,
- configuration hash/version,
- source commit,
- parent/attempt relationship,
- end events for clarification, abort, ambiguity, provider failure, or user interruption.

Multiple Pi sessions can append to the same file, so timestamps alone cannot safely reconstruct runs. The 15/37 summary coverage shows that the denominator is incomplete. Early entries also use an older schema, making aggregate comparisons misleading.

Finally, audit-write errors are swallowed. A disk, permission, or serialization failure silently destroys the very evidence the improvement loop depends on.

**Required change:** emit a versioned trace envelope on every event:

```json
{
  "schema": 2,
  "runId": "...",
  "sessionId": "...",
  "cwdHash": "...",
  "configHash": "...",
  "commit": "...",
  "attempt": 2,
  "event": "..."
}
```

Every `run-start` must have exactly one terminal `run-summary`, including aborted, clarified, unavailable, and unverified outcomes. Logging failure should be surfaced as a health warning without crashing the task.

---

### F-08 — Enforced model routing does not match the documented cost table

**Severity: High for cost predictability; Medium for correctness**
**Status: Confirmed**
**Primary evidence:** base maker installation at `packages/coding-agent/src/core/sdk.ts:325-345`; tier fallback and default triage at `packages/coding-agent/src/core/verify-routing.ts:410-450, 738`; live Anthropic triage/demotion events.

The live settings specify only hard-tier maker overrides. For standard and trivial work, `tierView()` falls back to the session’s current base maker. The triage model also defaults to that base maker when `verify.triage.model` is unset.

Consequences:

- the “cheap DeepSeek triage” claim is not enforced,
- standard/trivial maker identity depends on whichever model the session started with,
- demotion can return to an expensive model,
- costs and family independence vary across sessions.

The live audit shows triage calls through `anthropic/claude-opus-5` and standard/trivial demotions to Anthropic, contradicting the stated inexpensive routing table.

**Required change:** explicitly configure and validate every stage for every tier, including `triage.model`, `tiers.trivial.makerModel`, and `tiers.standard` (the current type supports only trivial/hard overrides). Log the fully resolved route at `run-start`, not only at installation.

---

### F-09 — Checkpoints are phase-blind, post-hoc, and can misgrade discovery

**Severity: High**
**Status: Confirmed by the current audit run**
**Primary evidence:** tools execute before checkpoint selection at `packages/agent/src/agent-loop.ts:264-324`; count-based scheduler at `packages/coding-agent/src/core/verify-routing.ts:94-112`; this run’s first two checker conflicts.

A checkpoint is triggered by the number of **tool-using assistant turns**, not by:

- number/risk of tool calls,
- whether any mutation occurred,
- whether the agent is in discovery, implementation, or verification,
- whether meaningful evidence changed.

One assistant turn can execute several parallel tools before the count increments once. The checker runs only after those tools execute, so it cannot prevent a destructive action.

In this audit, the first hard-tier checkpoint fired after four read/search batches. There was correctly no diff yet because the task required discovery before writing. The checker rejected that as “No maker progress or diff provided.” The rejection was a false signal caused by the checkpoint design, not maker failure.

**Required change:**

- classify tool calls by risk and phase,
- require pre-action approval/sandboxing for destructive or external side effects,
- checkpoint on evidence milestones or mutations rather than raw turn count,
- teach the checkpoint contract that read-only discovery is valid when the plan is in an investigation phase,
- include a structured plan-progress cursor instead of relying on maker prose.

---

### F-10 — Rejection “budgets” do not bound total rejection churn

**Severity: High**
**Status: Confirmed**
**Primary evidence:** `packages/coding-agent/src/core/verify-routing.ts:931-973`, especially `> maxRejections` and checkpoint counter reset at `:943`.

`maxRejections: 2` accepts only when the counter becomes greater than two, so it permits three conflicts. The name and documentation say “max rejections,” making this an off-by-one semantic mismatch unless the intended concept is “max retries.”

For checkpoints, exhaustion:

1. labels the current checkpoint unverified,
2. resets `checkpointRejections` to zero,
3. demotes the maker,
4. continues the run,
5. allows later checkpoints to start another full rejection streak.

The audit log shows a `budget-exhausted` checkpoint followed by another checkpoint conflict in the same session sequence. Thus the budget is not a total budget.

**Required change:** separate and enforce:

- `maxCheckpointConflictsTotal`,
- `maxFinalRetries`,
- `maxReplans`,
- `maxEscalations`.

Once checkpoint verification is exhausted, either keep the stronger maker and disable further checks with a persistent visible unverified state, or stop the run. Do not reset and silently start a new budget.

---

### F-11 — The planner is “repo-aware” only through a lossy 2 KB snapshot

**Severity: Medium**
**Status: Confirmed**
**Primary evidence:** `packages/coding-agent/src/core/verify-routing.ts:262-280` and planner input at `:790-807`.

The planner receives:

- up to 20 `git status` lines,
- the first 200 tracked file paths,
- package name and script names,
- all truncated together to 2,000 characters.

It does not receive:

- `AGENTS.md` contents and protected-file rules,
- README architecture,
- current diff details,
- relevant source content,
- previous failed plan evidence on the initial pass.

The prompt sensibly tells it to make discovery the maker’s first step. That reduces hallucination but does not make the planner genuinely repository-grounded. It may still prescribe protected edits, nonexistent paths, wrong test commands, or an approach incompatible with project rules.

**Required change:** give the planner a bounded, harness-produced context bundle containing project instructions, repository root, changed files, task-relevant search excerpts, and available verifier commands. Explicitly forbid asserting an unseen path, symbol, line number, or command as fact.

---

### F-12 — Model independence is convention, not an invariant

**Severity: Medium**
**Status: Confirmed**
**Primary evidence:** model resolution/installation in `packages/coding-agent/src/core/sdk.ts:325-364` checks resolvability, not provider-family separation.

The current configuration uses different families for the important stages, which is good. The code does not enforce it. A user or project setting can configure maker and checker to the same model/family, or a fallback can collapse intended independence.

The SDK only checks that a final checker resolves. It does not validate:

- checker family differs from maker family,
- final checker differs from checkpoint checker,
- planner differs from maker,
- fallback preserves independence.

**Required change:** resolve provider/family metadata and emit a startup warning or disable “independent verification” claims when independence is not true. High-assurance mode should reject same-family checker routes.

---

### F-13 — Verification configuration has no runtime schema validation

**Severity: Medium**
**Status: Confirmed**
**Primary evidence:** JSON parse at `packages/coding-agent/src/core/settings-manager.ts:507-525`; direct getter at `:864-866`; configured-but-unresolvable disable path at `packages/coding-agent/src/core/sdk.ts:337-363`.

`VerifySettings` is a TypeScript interface, but settings are loaded from JSON and returned directly. Invalid values can silently alter semantics:

- negative rejection/checkpoint limits,
- invalid backoff strings,
- empty verifier commands,
- unresolved model refs,
- contradictory tier combinations.

An unresolved final checker disables the entire routing layer and produces only an audit-log event, not a prominent user warning.

**Required change:** validate merged settings at startup with a runtime schema, normalize defaults, report every rejected field, and show the fully resolved route. Treat a configured-but-unresolvable checker as a visible configuration error.

---

### F-14 — The self-improvement gauntlet’s primary metric is implemented incorrectly

**Severity: High for the macro-loop**
**Status: Confirmed**
**Primary evidence:** `~/pi/.pi/gauntlet/CARD.md:50-51`; direct reproduction found 3 conflict records but 526 string-character iterations.

`CARD.md` metric `M1` builds one string containing all conflict texts and then executes:

```js
for (const t of text) {
  const key = t.split(...)[0]
  ...
}
```

Iterating a JavaScript string yields characters, not conflict records. `M1` therefore clusters repeated characters, not recurring conflict categories. Any normal conflict prose contains characters repeated more than three times, so the metric is predisposed to fail and its output cannot identify a meaningful defect class.

The gauntlet audit contains only three rounds; two later rounds repeat failing metric `M1`. That is consistent with a broken metric, not evidence of convergence.

**Required change:** parse JSON lines into conflict records, normalize one finding/category per event, and cluster records—not characters. Add fixture tests for the metric before using it as a promotion gate.

---

### F-15 — There is no durable eval/promotion loop comparable to the OpenAI reference

**Severity: High strategically**
**Status: Confirmed**
**Primary evidence:** the current repo contains runtime/unit tests and CARD proposal metrics, but no trace-linked feedback dataset, candidate-vs-baseline gate, holdout, or promotion record; compare the fetched OpenAI workflow’s Steps 3-9.

The OpenAI reference loop preserves learning through:

1. representative traces,
2. human and model feedback,
3. generated, reviewable eval definitions,
4. a repeatable validation gate,
5. ranked harness changes,
6. implementation handoff,
7. rerunning the same evals on the candidate harness.

The current system has traces, model criticism, unit tests, and a gauntlet proposal seam. It does not yet have:

- human feedback attached to run IDs,
- automatic promotion of a real failure into a regression case,
- a versioned eval dataset,
- baseline vs candidate comparison,
- a holdout set,
- rollback criteria,
- proof that an accepted amendment improved the target metric without regressions.

Unit tests are strong but mostly mock the model outputs the test expects. They validate control flow, not end-to-end judge reliability, prompt-injection resilience, or improvement on a representative task set.

**Required change:** build the macro flywheel separately from the runtime verifier. Start reviewed, not autonomous: select failed traces, label them, create deterministic/model-graded evals, run baseline and candidate on the same set plus holdout, and require explicit approval before promotion.

---

### F-16 — Platform safety is advisory unless Pi is externally sandboxed

**Severity: Critical for untrusted repositories or autonomous use**
**Status: Confirmed and documented**
**Primary evidence:** `README.md:40-46` and `packages/coding-agent/docs/containerization.md`; full-tool declaration in the planner prompt at `packages/coding-agent/src/core/verify-routing.ts:794-796`.

Pi’s root README states that it runs with the permissions of the launching user and has no built-in permission system for filesystem, process, network, or credentials. The custom planner explicitly tells the maker it has full tool access.

The local `AGENTS.md` restrictions are valuable operational controls, but a model can violate instructions. A post-hoc checker cannot undo:

- deleted data,
- force pushes,
- secret exfiltration,
- package-install lifecycle scripts,
- external API side effects.

**Required change:** for autonomous/looped operation, run Pi in Gondolin, Docker, OpenShell, or another sandbox; use a minimal credential scope; default network egress off; and require a human/pre-action policy gate for irreversible operations. Treat the checker as quality control, not a security boundary.

---

### F-17 — Long sessions can become pathological, and the current cost log undercounts the damage

**Severity: High**
**Status: Confirmed by saved-session evidence**
**Primary evidence:** sampled files under `~/.pi/agent/sessions/--Users-rahul--`; cost accumulation at `packages/coding-agent/src/core/verify-routing.ts:380-398` tracks verifier-path calls only.

The sampled session inventory contains extreme trajectories:

- one session file is about 10 MB with roughly 2,395 assistant messages and 2,301 tool calls,
- another is about 7 MB with roughly 153 assistant messages and 327 tool calls,
- the historical transcript includes a visible degeneration where the model repeatedly says variants of “let me run the grep” without issuing a tool call,
- one maker fallback was triggered after a provider reported `Prompt tokens limit exceeded: 416480 > 215351`.

Compaction exists elsewhere in Pi, but the verification architecture has no progress/stall detector. A checker only runs at a checkpoint or candidate final, and its evidence window is too small to diagnose repeated no-op language across a long trajectory reliably.

The `run-summary` field named `verifyCalls` and its token/cost totals include triage/planner/checker calls tracked by `trackCost`, but exclude maker turns and maker fallbacks. It therefore cannot answer the operational question “what did this run cost?” or reveal that a cheap verification layer surrounded an enormous maker trajectory.

**Required change:** add deterministic stall detection (repeated near-duplicate assistant text, repeated identical tool calls/results, no artifact/evidence delta across N turns), force compaction or re-plan before context limits, and record both `verificationCost` and `totalRunCost` with maker/tool-turn counts.

---

### F-18 — Goal identity is lost across user follow-ups

**Severity: High**
**Status: Confirmed by code and audit shape**
**Primary evidence:** the reset comment and implementation at `packages/coding-agent/src/core/verify-routing.ts:327-340, 723-732`; tiering uses only the new first-user text at `:727-758`; many trivial run starts occur inside long hard-task sessions.

The architecture defines a run as **one user prompt**, not one goal episode. Every fresh prompt resets:

- the original tier,
- planning state,
- checkpoint counters,
- rejection counters,
- sticky escalation,
- fallback counters and run-cost totals.

That is reasonable for independent questions, but wrong for ordinary interactive continuations. After a hard task partially completes, a user may say “continue,” “keep going,” “proceed,” or “yes.” The router classifies only that short text plus a tiny repo snapshot. It can therefore label the continuation `trivial`, skip planning and checkpoints, and abandon the stronger maker selected for the unresolved hard goal.

The audit history has many trivial `run-start` events embedded in very long sessions, including short acknowledgement/follow-up prompts. Because there is no goal ID, the system cannot distinguish a new task from continuation of an unachieved goal.

This is the architectural difference between the current **turn-based loop** and the goal-based loop described by Anthropic.

**Required change:** introduce a persisted goal episode with `goalId`, original goal, success criteria, tier, attempt budget, and terminal state. A follow-up should inherit the active goal unless the user explicitly starts a new one or the previous goal reached a terminal outcome. Triage should see the active goal plus the new instruction.

---

## 5. Behavioral evidence and what it does—and does not—prove

### Positive evidence

The live log demonstrates that the architecture is active rather than decorative:

- planners actually run,
- checkers reject unsupported claims,
- makers retry and switch models,
- independent commands are requested,
- final answers can be rejected and later verified,
- provider fallbacks occur,
- budget exhaustion is recorded.

Examples include checker catches for:

- claimed tests that were run from the wrong directory,
- a failed edit presented as completed,
- a browser audit that crashed before verification,
- unsupported readiness claims.

These are meaningful wins.

### Negative evidence

The same log demonstrates systemic friction:

- repeated wrong-cwd verifier commands,
- six ambiguous checker outcomes that fail open,
- replies beginning `VERIFIED` classified as conflicts,
- expensive routes used for trivial/standard stages,
- checkpoint exhaustion followed by more checkpoints,
- only 15 summaries for 37 starts,
- current discovery work rejected merely because no diff existed yet,
- saved trajectories with thousands of assistant/tool events and a prompt-token-limit failure.

### Sampled behavioral episodes

I sampled 13 observable run segments from the log rather than treating aggregate event counts as quality scores. “Accepted” below means the configured checker eventually emitted `VERIFIED`; it is not an independent claim that the underlying product task truly succeeded. Routing is marked indeterminate when the audit schema did not retain enough of the original goal.

| Run start (UTC) | Route | Plan quality | Observable result | Audit tag |
|---|---|---|---|---|
| 2026-08-07 05:14 | Standard | Concrete two-step `hello.js` plan | Final checker accepted | Likely over-routed/over-planned for a tiny fixture, but converged |
| 2026-08-07 06:49 | Hard | Returned CLARIFY for a self-contained audit | Maker produced no tools; final audit skipped | **Bad plan/routing behavior**; discovery should have been the work |
| 2026-08-07 07:53 | Trivial | None | Final checker caught unsupported test claims; retry accepted after maker fallbacks | Checker added value; tier correctness indeterminate |
| 2026-08-07 08:08 | Hard | Detailed OSS-audit plan | Four checkpoint conflicts, wrong-cwd verification, budget exhaustion, no terminal summary | **Did not demonstrate convergence** |
| 2026-08-07 08:39 | Hard | Planner output was only `cap works` | No-tool final skipped | **Malformed/low-quality plan accepted as valid** |
| 2026-08-07 13:25 | Trivial | None | Checker rejected incomplete packaging evidence, then accepted retry | Useful final checker; tier correctness indeterminate |
| 2026-08-07 13:50 | Trivial | None | Checker caught a failed edit presented as completed, then accepted retry | Strong positive checker example |
| 2026-08-07 16:36 | Trivial | None | Two conflicts, provider fallback, then accepted; summary records 436,547 verification-path input tokens and $0.347586 | **Severe cost/context anomaly for a trivial route** |
| 2026-08-10 05:58 | Standard | Concrete integration plan | Three checkpoints plus final; wrong-cwd command caused rejection; eventually accepted after 13 verification calls | Converged, but with avoidable verifier churn |
| 2026-08-10 06:41 | Hard | 9,408-character plan, therefore truncated before maker injection | Five checkpoints, two ambiguous verdicts, two conflicts, checkpoint budget exhaustion, then final accepted; 17 verification calls / $0.412611 | **Weak efficiency and incomplete plan handoff** |
| 2026-08-10 07:53 | Standard | Concrete credential-integration plan | Two accepted checkpoints, then maker prompt-token-limit fallback; no terminal summary | **Context/run boundary failure; outcome unresolved** |
| 2026-08-10 08:09 | Standard | Generic continuation plan for “proceed” | Ambiguous checkpoint; first final response text began `VERIFIED` but parser recorded conflict; retry accepted | **Goal-continuity and verdict-protocol defects** |
| 2026-08-13 06:09 | Hard | 8,444-character architecture-audit plan, truncated to 4,000 for the maker | First two read-only discovery checkpoints rejected for “no progress or diff”; third accepted after this report existed | **Phase-blind checkpoint false positives** |

This sample shows both sides of the customization: it catches fabricated completion claims well, but routing, evidence binding, plan validation, context control, and stop semantics create substantial avoidable churn.

### Measurement caveat

The event file spans multiple code versions and concurrent sessions and lacks run IDs. Therefore:

- `21 verified / 22 conflict / 6 ambiguous` describes **audit events**, not task success rates;
- tier-tagged event totals are not tier run counts;
- missing summaries cannot be assigned safely to abort, failure, clarification, or concurrency;
- before/after performance claims cannot be made from this file alone.

Any future optimization loop should fix trace identity before using these numbers as a score.

---

## 6. Comparison with the reference loop patterns

| Principle | Current Pi customization | Assessment |
|---|---|---|
| Repeat work until a stop condition | Repeats on checker conflict; final verified stops | **Partial:** no whole-run turn/time/cost cap |
| Independent evaluator with fresh context | Separate checkpoint/final model calls | **Strong**, but evidence and verdict protocol are weak |
| Quantitative/deterministic verification | Checker may request allowlisted commands | **Partial:** cwd/evidence binding is unreliable |
| Trace every run | Global JSONL event log | **Partial:** no trace/run identity; incomplete terminal summaries |
| Human + model feedback | Model checker; user can inspect logs | **Weak:** feedback is not attached to durable cases |
| Convert failures into reusable evals | Manual tests and CARD metric proposals | **Weak/incomplete:** no automated reviewed regression pipeline; M1 is broken |
| Candidate vs baseline gate | Not present | **Missing** |
| Holdout to prevent overfitting | Not present | **Missing** |
| Explicit max attempts | Rejection/checkpoint limits | **Partial:** no maker-turn bound; rejection budget resets |
| Goal continuity across follow-ups | State resets on every fresh user prompt | **Missing:** short continuations can be re-triaged as trivial |
| Security boundary around tools | External sandbox recommended | **Missing in-process; must be deployed externally** |
| Cost-aware routing | Tier model routing and summaries | **Good intent, not fully enforced by config** |

---

## 7. Recommended remediation order

### Phase 0 — Do not call the verifier a security boundary

1. Use an external sandbox for autonomous runs.
2. Restrict credentials and network egress.
3. Require human approval for destructive/external side effects.

### Phase 1 — Make outcomes and evidence trustworthy

1. Replace substring verdict parsing with structured output.
2. Add explicit `UNVERIFIED` and `ABORTED` outcomes; never silently fail open.
3. Bind verifier commands to a resolved project root and typed argv.
4. Create a complete changed-artifact manifest, including untracked files and truncation flags.
5. Redact and isolate all evidence before external verifier/planner calls.

### Phase 2 — Bound the loop

1. Add maker-turn, tool-call, elapsed-time, and total-cost ceilings.
2. Persist a goal episode across follow-up prompts until success, explicit abandonment, or a hard boundary.
3. Replace streak-reset budgets with total per-goal budgets.
4. Make checkpoint cadence phase/risk aware.
5. Preserve escalation after checkpoint exhaustion instead of demoting.

### Phase 3 — Make planning a contract

1. Use a structured PLAN/CLARIFY union.
2. Enforce step/question/byte limits.
3. Include project instructions and relevant repo evidence.
4. Add explicit success and stop criteria.

### Phase 4 — Fix observability and routing

1. Add run/session/config/repo identifiers and terminal summaries.
2. Runtime-validate settings and model independence.
3. Explicitly configure every stage/tier; do not inherit an arbitrary session base model.
4. Distinguish verification overhead from total run cost.
5. Add deterministic stall and repeated-no-progress telemetry.

### Phase 5 — Build the real improvement flywheel

1. Fix and test gauntlet metrics.
2. Select and label representative failed traces.
3. Promote accepted failures into versioned regression cases.
4. Maintain regression, rolling-discovery, and holdout sets.
5. Compare baseline and candidate harnesses on identical cases.
6. Require reviewed promotion and support rollback.

---

## 8. Proposed changes (not applied)

The files involved are locally protected. These are design-level diffs for user review, not edits.

### Proposal A — Structured verdicts and explicit unverified outcome

```diff
- classifyCheckerVerdict(auditText: string): "verified" | "conflict" | "ambiguous"
+ parseCheckerResult(json: unknown):
+   | { verdict: "verified"; issues: []; evidenceGaps: [] }
+   | { verdict: "conflict"; issues: string[]; evidenceGaps: string[] }
+   | { verdict: "unverified"; reason: "malformed" | "unavailable" | "insufficient-evidence" }

- return undefined; // ambiguous/unavailable
+ return {
+   status: "unverified",
+   notice: "[VERIFY] UNVERIFIED — checker unavailable or returned no valid verdict"
+ };
```

### Proposal B — Whole-run limits

```diff
 interface VerifySettings {
+  maxMakerTurns?: number;
+  maxToolCallsPerRun?: number;
+  maxRunMs?: number;
+  maxRunCostUsd?: number;
+  unavailablePolicy?: "surface-unverified" | "fail-closed";
 }
```

The enforcement belongs in the agent loop, not solely in prompt text.

### Proposal C — Typed verifier commands

```diff
- verifierCommands?: string[];
+ verifierCommands?: Array<{
+   id: string;
+   argv: string[];
+   cwdPolicy: "session" | "git-root" | "nearest-package-root";
+   timeoutMs?: number;
+ }>;
```

### Proposal D — Structured planner output

```diff
- "Otherwise produce a short, concrete, step-by-step plan..."
+ "Return JSON matching exactly one variant:
+  PLAN { assumptions[<=5], steps[<=12], successCriteria[<=8], stopConditions[<=5] }
+  CLARIFY { questions[<=5] }.
+  Never assert an unseen path, symbol, line number, command, or test result as fact."
```

### Proposal E — Trace identity

```diff
- logVerify({ event: "run-start", chars: promptText.length });
+ logVerify({
+   schema: 2,
+   runId,
+   sessionId,
+   cwdHash,
+   configHash,
+   commit,
+   event: "run-start",
+   chars: promptText.length
+ });
```

### Proposal F — Correct the gauntlet M1 record loop

```diff
- const text = conflicts.map(l => JSON.parse(l).text || '').join('\n');
- for (const t of text) {
+ const findings = conflicts.map(l => JSON.parse(l).text || '');
+ for (const t of findings) {
    const key = normalizeConflictCategory(t);
```

A real `normalizeConflictCategory()` should be deterministic and fixture-tested; splitting the first clause is still too brittle for a promotion metric.

---

## 9. What should remain unchanged

Several choices are sound and should be preserved:

- independent checker calls do not mutate maker context directly,
- the harness—not the maker—runs verifier commands,
- project verifier commands are allowlisted,
- planner/checker do not receive tools,
- provider fallback attempts are bounded,
- hard tasks can start on the stronger maker rather than paying for a doomed cheap pass,
- sticky escalation avoids maker ping-pong,
- budget exhaustion is labeled instead of falsely called verified,
- protected local customization files require explicit human approval.

The architecture needs stronger contracts and measurement, not a wholesale rewrite.

---

## 10. Bottom line

**Is the customization flawed? Yes—but not because it lacks agents or loops.**

The per-task architecture is thoughtful and already catches real errors. Its main weaknesses are at the seams where probabilistic prose is treated as a protocol, partial evidence is treated as the artifact, and a global event log is treated as an improvement dataset.

The highest-leverage next step is not another stronger model. It is to make four things deterministic:

1. **what counts as a valid verdict,**
2. **what evidence a checker must see,**
3. **when and why a run must stop,**
4. **how a failure becomes a durable regression test.**

Once those are in place, the existing planner/maker/checker routing becomes a solid runtime repair loop, and the gauntlet can become the separate, reviewed improvement flywheel envisioned by the OpenAI and Anthropic references.
