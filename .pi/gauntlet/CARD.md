# Gauntlet card — improve Pi's own harness

This is the self-improvement card for Pi itself. It is consumed by the
gauntlet macro loop (`/gauntlet start` from the `gauntlet.ts` extension).

The objective is **not** to ship a feature in any project; it is to reduce the
recurring failure modes the verify routing and the gauntlet itself produce, by
*amending the harness from its own audit evidence* and validating each
amendment on the next round.

The two harness seams the loop exercises are:

1. **CARD self-amendment** — after each round, the critic proposes one new
   `critic:` metric derived from the round's `failing` findings. Accepted
   amendments land in this file and fire on every future round, so the eval
   gate compounds instead of staying static.
2. **Harness diff proposal** — the critic, given the last N rounds' traces and
   the current metric set, ranks one harness change (a `VerifySettings` tweak,
   a tighter `EXCELLENCE_CHARTER` clause, a prompt edit, or another metric)
   that would prevent the most failing metrics, and proposes it as a diff.
   Per `~/.pi/agent/AGENTS.md` every file listed there is protected and
   requires explicit user confirmation before any edit. The loop never edits
   them silently — it surfaces the diff and waits.

The boundary's `out_of_scope` and `escalate_human` lists exist so the loop
cannot reach into the runtime, secrets, signing, or the audit logs it reads.

```json
{
  "objective": "Reduce the recurring CONFLICT and gauntlet failure modes recorded in ~/.pi/agent/verify-audit.jsonl and ~/.pi/agent/gauntlet-audit.jsonl by amending Pi's own harness (verify settings in settings.json, the EXCELLENCE_CHARTER in verify-routing.ts, the .pi/prompts/*.md files, and this CARD) from accumulated audit evidence. Each round proposes and validates exactly one harness amendment; the bar is a measurable drop in the failing-metric count on the next round, not a claim of completion.",
  "out_of_scope": [
    "~/.pi/agent/settings.json:verify (protected — propose diffs only, never edit silently)",
    "~/pi/packages/coding-agent/src/core/verify-routing.ts (protected)",
    "~/pi/packages/coding-agent/src/core/settings-manager.ts (protected)",
    "~/pi/packages/coding-agent/src/core/sdk.ts (protected)",
    "~/pi/packages/coding-agent/src/core/agent-session.ts (protected)",
    "~/pi/packages/agent/src/agent-loop.ts (protected)",
    "~/pi/packages/agent/src/agent.ts (protected)",
    "~/pi/packages/agent/src/types.ts (protected)",
    "~/pi/packages/agent/test/agent-loop.test.ts (protected)",
    "~/pi/packages/coding-agent/test/verify-routing.test.ts (protected)",
    "~/.pi/agent/extensions/gauntlet.ts (protected)",
    "~/.pi/agent/auth.json",
    "~/.pi/agent/verify-audit.jsonl (read-only evidence source)",
    "~/.pi/agent/gauntlet-audit.jsonl (read-only evidence source)",
    "secrets, credentials, deploy keys, signing keys"
  ],
  "metrics": [
    {
      "id": "M1",
      "cmd": "node .pi/gauntlet/metrics/recurring-conflicts.mjs"
    },
    {
      "id": "M1T",
      "cmd": "node --test .pi/gauntlet/metrics/recurring-conflicts.test.mjs"
    },
    {
      "id": "M2",
      "cmd": "node /Users/rahul/pi/.pi/gauntlet/metrics/recurring-gauntlet-failures.mjs"
    },
    {
      "id": "M3",
      "critic": "Read ~/.pi/agent/verify-audit.jsonl and ~/.pi/agent/gauntlet-audit.jsonl (tail 80 lines each). Identify the single recurring failure pattern that, if a harness change prevented it, would drop the most failing metrics across both logs. Then read the protected files listed in the CARD's out_of_scope: ~/.pi/agent/settings.json (the verify key), ~/pi/packages/coding-agent/src/core/verify-routing.ts (the EXCELLENCE_CHARTER and audit prompts), and the .pi/prompts/*.md files. Propose EXACTLY ONE concrete harness amendment as a unified diff. AMBIENT-FIRST PREFERENCE (the user is non-technical and never restarts manually, so prefer changes that take effect without a rebuild): (1) best — a new critic or cmd metric for this CARD.md (takes effect next round, no restart); (2) good — a .pi/prompts/*.md edit (takes effect after one session reload); (3) acceptable — a settings.json verify-key tweak (takes effect after one session reload); (4) last resort only — a verify-routing.ts / EXCELLENCE_CHARTER source change (needs npm run build + full pi restart). Only propose category 4 when no category 1/2/3 change can prevent the failure. Output the diff in a ```diff fenced block, preceded by a one-line rationale and followed by which metric(s) it should move. Do not edit any file. Do not propose changes to out_of_scope items beyond surfacing the diff. If you find no actionable recurring pattern, reply exactly: NONE"
    }
  ],
  "boundary": {
    "max_rounds": 5,
    "max_cost_usd": 8,
    "stop_on_repeat": 2,
    "escalate_human": ["secrets", "deploy", "signing", "auth.json", "credentials"]
  },
  "criticModel": "openrouter/openai/gpt-5.6-luna"
}
```

## Operating notes

- **M1** counts recurring CONFLICT clusters in the last 50 verify audits. It
  passes when no cluster recurs ≥3 times — i.e. the verify routing is no longer
  hitting the same defect class repeatedly. It is the micro-loop health signal.
  Implemented in `.pi/gauntlet/metrics/recurring-conflicts.mjs`.
  **Fixed 2026-08-13 (audit finding F-14).** The original inline one-liner joined
  every conflict text into ONE string and then ran `for (const t of text)`.
  Iterating a string yields characters, so it clustered characters (top keys were
  `misc`, `e`, `i`, `s`, `t`) instead of conflict records — 4 records became 51
  meaningless clusters, and the metric could never identify a defect class. It now
  iterates the conflict-record array and maps each record to a stable category id.
  Exit 2 (unreadable evidence) is a failure, never a silent pass.
- **M1T** runs M1's own unit tests. A metric that gates promotion must itself be
  tested, which is precisely what F-14 proved: the broken metric shipped because
  nothing verified it. If M1T fails, treat M1's verdict as untrustworthy.
- **M2** counts recurring gauntlet failures in the last 10 rounds. It passes
  when no metric id repeats as failing ≥2 rounds — i.e. the macro loop is
  converging, not spinning. It is the macro-loop health signal.
- **M3** is the seam. The critic reads the audit logs + the protected harness
  files and proposes exactly one harness amendment as a diff. It never edits;
  the builder (a Pi session running under the verify routing) takes the diff,
  surfaces it to the user per the protected-file rule, and on approval applies
  it. The next round's M1/M2 measure whether the amendment actually moved the
  numbers.
- **Ambient-first (you never restart manually)**: M3 prefers amendments that
  take effect without a rebuild, in this order:
  1. A new metric in this CARD → takes effect next round, no reload, no restart.
  2. A `.pi/prompts/*.md` edit → takes effect after one in-session reload.
  3. A `settings.json` verify-key tweak → takes effect after one in-session reload.
  4. A `verify-routing.ts` / `EXCELLENCE_CHARTER` source change → last resort only;
     needs `npm run build` from `~/pi` + a full pi restart. The loop tells the
     builder to apply the edit, stop the round, and tell you to restart — it
     never pretends a source change is active in the current session.
- **Compounding**: when M3 proposes a *new metric* derived from a recurring
  failure, and the user accepts it, that metric lands in this CARD and fires on
  every future round. That is feedback becoming a reusable eval, Pi-native.
- **Honest stop**: `stop_on_repeat: 2` means if the same failing fingerprint
  recurs two rounds running and a strategy change has already been issued, the
  loop stops and reports the current state as NOT a pass. It never claims a
  pass it did not measure.
- **Critic**: Luna — same plug as every other checker. Pi's live lineup is
  three models (Flash, Luna, K3 spare). The gauntlet critic is a checker job,
  not a fourth company. K3 stays spare-only and is not on this loop.