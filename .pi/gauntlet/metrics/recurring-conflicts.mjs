#!/usr/bin/env node
/**
 * Gauntlet metric M1 — recurring CONFLICT clusters in recent verify audits.
 *
 * Replaces the previous inline `node -e` one-liner, which joined every conflict
 * text into ONE string and then ran `for (const t of text)`. Iterating a string
 * yields characters, so it clustered characters ("V", "E", "R", " ") instead of
 * conflict records: the metric could never identify a defect class and was
 * predisposed to fail. See docs/audit/self-architecture-audit.md finding F-14.
 *
 * This module is pure except for `main()`, so the clustering semantics are
 * unit-testable (recurring-conflicts.test.mjs).
 *
 * Exit codes (the gauntlet treats non-zero as a failing metric):
 *   0 — no conflict category recurs at or above the threshold
 *   1 — at least one recurring category (the loop still has a defect class)
 *   2 — the metric could not read its evidence (never silently "pass")
 */

import { readFileSync } from "node:fs";

export const DEFAULT_AUDIT_LOG = "/Users/rahul/.pi/agent/verify-audit.jsonl";
/** Window is measured in audit EVENTS, not raw log lines (mixed event types). */
export const DEFAULT_WINDOW = 50;
export const DEFAULT_THRESHOLD = 3;
export const DEFAULT_TOP = 5;

/**
 * Ordered category rules. First match wins, so put specific defect classes
 * before generic ones. Categories are stable ids: renaming one resets its
 * history, so treat them as an append-mostly list.
 */
const CATEGORY_RULES = [
	// Must precede every other rule: a conflict whose text OPENS with the pass
	// token is a parser artifact (finding F-05), not a real defect in the work.
	// Categorizing it separately keeps genuine defect counts honest.
	{ id: "false-conflict-verdict-parser", test: /^\s*(?:\*{0,2}|#{0,3}\s*)VERIFIED\b/ },
	{ id: "wrong-cwd-verifier-command", test: /enoent|no such file or directory|wrong directory|package\.json.*not|from `?\/users\/[^`\s]*`? instead/i },
	{ id: "claimed-verification-without-receipt", test: /claims?\b[^.]{0,80}\b(pass|passed|clean|green)|no receipt|without a receipt|not (?:backed|substantiated|supported) by/i },
	{ id: "no-progress-or-diff", test: /no (?:maker )?progress|no diff|progress section is empty|undifferentiated exploration/i },
	{ id: "failed-edit-presented-as-done", test: /edit (?:failed|did not apply)|marked done|presented as (?:complete|completed|working)/i },
	{ id: "missing-or-incomplete-tests", test: /missing tests?|no tests?\b|tests? (?:were )?not (?:run|added)|test suite (?:was )?not/i },
	{ id: "mock-placeholder-or-stub", test: /mock|placeholder|stub|fake wiring|todo|fixme/i },
	{ id: "crashed-or-errored-verification", test: /crash|exited with code|exit \d+|error during|failed with/i },
	{ id: "scope-or-goal-drift", test: /drift|instead of actually|roleplay|not the requested|does not satisfy the goal/i },
];

/** Parse a JSONL audit log into records, skipping unparseable lines. */
export function parseAuditLines(raw) {
	const records = [];
	for (const line of String(raw).split("\n")) {
		const trimmed = line.trim();
		if (!trimmed) continue;
		try {
			records.push(JSON.parse(trimmed));
		} catch {
			// A corrupt line must not abort the metric; it is simply not evidence.
		}
	}
	return records;
}

/**
 * The most recent `window` audit events, oldest-first.
 * P2: only count events from the current schema version (schema 2) so old
 * config-era conflicts don't keep the metric failing after a real fix.
 * Events without a schema field are pre-schema-2 and are excluded.
 */
export function selectRecentAuditEvents(records, window = DEFAULT_WINDOW) {
	const audits = records.filter((r) => r && r.event === "audit" && r.schema === 2);
	return window > 0 ? audits.slice(-window) : audits;
}

/** Conflict findings inside a window of audit events: one string per record. */
export function selectConflictFindings(auditEvents) {
	return auditEvents
		.filter((e) => e.verdict === "conflict")
		.map((e) => (typeof e.text === "string" ? e.text : ""))
		.filter((text) => text.trim().length > 0);
}

/**
 * Map one conflict finding to a stable category id.
 *
 * Deterministic and total: unmatched findings fall back to a normalized first
 * clause so a new defect class still clusters with itself instead of being
 * dropped. Never returns an empty string.
 */
export function normalizeConflictCategory(finding) {
	const text = String(finding ?? "");
	for (const rule of CATEGORY_RULES) {
		if (rule.test.test(text)) return rule.id;
	}
	const firstClause = text
		.replace(/[*_`#>\\]/g, " ")
		.split(/[:;.,\n]/)[0]
		.toLowerCase()
		.replace(/\s+/g, " ")
		.trim()
		.slice(0, 40)
		.trim();
	return firstClause.length > 0 ? `other:${firstClause}` : "other:unclassified";
}

/**
 * Cluster conflict FINDINGS (one entry per audit record) by category.
 * Returns descending counts, ties broken by category id for determinism.
 */
export function clusterConflicts(findings) {
	const counts = new Map();
	for (const finding of findings) {
		if (typeof finding !== "string") continue;
		const category = normalizeConflictCategory(finding);
		counts.set(category, (counts.get(category) ?? 0) + 1);
	}
	return [...counts.entries()]
		.map(([category, count]) => ({ category, count }))
		.sort((a, b) => b.count - a.count || a.category.localeCompare(b.category));
}

/** Recurring = at or above `threshold` hits, capped to the top `top` clusters. */
export function recurringClusters(clusters, threshold = DEFAULT_THRESHOLD, top = DEFAULT_TOP) {
	return clusters.slice(0, top).filter((c) => c.count >= threshold);
}

/** Whole-metric evaluation over a raw log body. Pure; used by tests and main. */
export function evaluate(raw, options = {}) {
	const window = options.window ?? DEFAULT_WINDOW;
	const threshold = options.threshold ?? DEFAULT_THRESHOLD;
	const top = options.top ?? DEFAULT_TOP;
	const records = parseAuditLines(raw);
	const auditEvents = selectRecentAuditEvents(records, window);
	const findings = selectConflictFindings(auditEvents);
	const clusters = clusterConflicts(findings);
	const recurring = recurringClusters(clusters, threshold, top);
	return { auditEvents: auditEvents.length, findings, clusters, recurring, window, threshold };
}

export function formatReport(result) {
	const scope = `${result.findings.length} conflict finding(s) in last ${result.auditEvents} audit event(s)`;
	if (result.recurring.length === 0) {
		return `no recurring CONFLICT category (>=${result.threshold} hits); ${scope}`;
	}
	const lines = result.recurring.map((c) => `- ${c.category} (${c.count} findings)`);
	return `recurring CONFLICT categories (>=${result.threshold} hits); ${scope}:\n${lines.join("\n")}`;
}

function parseArgs(argv) {
	const options = { logPath: DEFAULT_AUDIT_LOG };
	for (const arg of argv) {
		const [key, value] = arg.split("=");
		if (key === "--log" && value) options.logPath = value;
		else if (key === "--window" && value) options.window = Number(value);
		else if (key === "--threshold" && value) options.threshold = Number(value);
		else if (key === "--top" && value) options.top = Number(value);
	}
	return options;
}

export function main(argv = process.argv.slice(2)) {
	const options = parseArgs(argv);
	let raw;
	try {
		raw = readFileSync(options.logPath, "utf8");
	} catch (error) {
		// Missing/unreadable evidence is a metric failure, not a pass.
		console.error(`M1 could not read ${options.logPath}: ${error.message}`);
		return 2;
	}
	const result = evaluate(raw, options);
	console.log(formatReport(result));
	return result.recurring.length === 0 ? 0 : 1;
}

if (import.meta.url === `file://${process.argv[1]}`) {
	process.exit(main());
}
