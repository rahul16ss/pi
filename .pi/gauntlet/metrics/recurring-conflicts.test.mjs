#!/usr/bin/env node
/**
 * Regression tests for gauntlet metric M1 (finding F-14).
 *
 * The defect: the previous implementation joined all conflict texts into one
 * string and iterated it, clustering CHARACTERS instead of conflict records.
 * `clusters records, never characters` is the direct regression guard.
 *
 * Run: node --test .pi/gauntlet/metrics/recurring-conflicts.test.mjs
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import {
	clusterConflicts,
	evaluate,
	formatReport,
	main,
	normalizeConflictCategory,
	parseAuditLines,
	recurringClusters,
	selectConflictFindings,
	selectRecentAuditEvents,
} from "./recurring-conflicts.mjs";

const auditLine = (fields) => JSON.stringify({ ts: "2026-08-13T00:00:00.000Z", event: "audit", schema: 2, ...fields });

test("clusters records, never characters (F-14 regression)", () => {
	const findings = [
		"The harness ran `npm test` from `/Users/rahul` instead of the repo root, so ENOENT.",
		"No maker progress or diff provided.",
	];
	const clusters = clusterConflicts(findings);
	const total = clusters.reduce((sum, c) => sum + c.count, 0);

	// One cluster entry per distinct category, and counts sum to record count.
	assert.equal(total, findings.length);
	assert.ok(clusters.length <= findings.length);
	// The old bug produced single-character categories such as "V" or "t".
	for (const cluster of clusters) {
		assert.ok(cluster.category.length > 1, `single-character category leaked: ${cluster.category}`);
	}
	// Joined-string iteration would have produced ~115 clusters, not <= 2.
	assert.ok(clusters.length <= 2);
});

test("distinct defect classes form distinct clusters", () => {
	const clusters = clusterConflicts([
		"ENOENT: no such file or directory, open '/Users/rahul/package.json'",
		"The maker claims npm test passed but there is no receipt proving it.",
		"No maker progress or diff provided.",
	]);
	assert.equal(clusters.length, 3);
	assert.deepEqual(
		clusters.map((c) => c.category).sort(),
		["claimed-verification-without-receipt", "no-progress-or-diff", "wrong-cwd-verifier-command"],
	);
	for (const cluster of clusters) assert.equal(cluster.count, 1);
});

test("same defect class collapses into one cluster with the right count", () => {
	const clusters = clusterConflicts([
		"The harness ran `npm test` from the wrong directory (ENOENT).",
		"npm test failed with ENOENT because it ran from /Users/rahul instead of the project.",
		"Verification ran in the wrong directory again; no such file or directory.",
	]);
	assert.equal(clusters.length, 1);
	assert.equal(clusters[0].category, "wrong-cwd-verifier-command");
	assert.equal(clusters[0].count, 3);
});

test("empty findings produce no clusters and no recurrence", () => {
	assert.deepEqual(clusterConflicts([]), []);
	assert.deepEqual(recurringClusters([], 3), []);
	const result = evaluate("");
	assert.equal(result.findings.length, 0);
	assert.deepEqual(result.recurring, []);
	assert.match(formatReport(result), /^no recurring CONFLICT category/);
});

test("conflicts whose text opens with VERIFIED are categorized as parser artifacts (F-05)", () => {
	// Real shape observed in verify-audit.jsonl on 2026-08-13: the checker opened
	// with VERIFIED, then quoted the rejection criterion, and the old prose scan
	// recorded a conflict. That is a parser artifact, not a defect in the work.
	const text = "VERIFIED\n\nThe receipts substantiate the claims, so the CONFLICT criterion was not met.";
	assert.equal(normalizeConflictCategory(text), "false-conflict-verdict-parser");
	assert.equal(normalizeConflictCategory("**VERIFIED**\n\nNo issues; nothing meets CONFLICT."), "false-conflict-verdict-parser");
	// A genuine rejection that merely mentions verification stays a real defect class.
	assert.equal(
		normalizeConflictCategory("The maker claims npm test passed but no receipt proves it. CONFLICT"),
		"claimed-verification-without-receipt",
	);
});

test("unmatched findings still cluster with themselves instead of vanishing", () => {
	const category = normalizeConflictCategory("Quantum flux capacitor misaligned; retry later");
	assert.ok(category.startsWith("other:"));
	assert.equal(normalizeConflictCategory("Quantum flux capacitor misaligned; different tail"), category);
	assert.equal(normalizeConflictCategory(""), "other:unclassified");
	assert.equal(normalizeConflictCategory(undefined), "other:unclassified");
});

test("only conflict audit events inside the window count as findings", () => {
	const raw = [
		auditLine({ verdict: "conflict", text: "old ENOENT wrong directory" }),
		auditLine({ verdict: "verified", text: "VERIFIED" }),
		JSON.stringify({ event: "run-start", chars: 10 }),
		"{ not json",
		auditLine({ verdict: "ambiguous", text: "RUN: npm test" }),
		auditLine({ verdict: "conflict", text: "No maker progress or diff provided." }),
	].join("\n");

	const records = parseAuditLines(raw);
	assert.equal(records.length, 5, "corrupt line is skipped, valid ones survive");

	const windowed = selectRecentAuditEvents(records, 2);
	assert.equal(windowed.length, 2);
	assert.deepEqual(selectConflictFindings(windowed), ["No maker progress or diff provided."]);

	const wide = selectConflictFindings(selectRecentAuditEvents(records, 50));
	assert.equal(wide.length, 2);
});

test("threshold decides pass/fail and exit code semantics", () => {
	const conflict = (text) => auditLine({ verdict: "conflict", text });
	const raw = [conflict("ENOENT wrong directory"), conflict("ENOENT wrong directory again"), conflict("ENOENT third time")].join("\n");

	const failing = evaluate(raw, { threshold: 3 });
	assert.equal(failing.recurring.length, 1);
	assert.equal(failing.recurring[0].count, 3);
	assert.match(formatReport(failing), /recurring CONFLICT categories/);

	const passing = evaluate(raw, { threshold: 4 });
	assert.deepEqual(passing.recurring, []);
});

test("unreadable evidence exits 2 rather than passing silently", () => {
	const code = main(["--log=/nonexistent/verify-audit.jsonl"]);
	assert.equal(code, 2);
});
