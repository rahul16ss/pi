#!/usr/bin/env node
/**
 * Tests for gauntlet metric M2 (recurring gauntlet failures).
 * Run: node --test .pi/gauntlet/metrics/recurring-gauntlet-failures.test.mjs
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import {
	clusterFailingMetrics,
	collectFailingMetrics,
	evaluate,
	formatReport,
	main,
	parseGauntletLines,
	recurringFailures,
	selectRecentRounds,
} from "./recurring-gauntlet-failures.mjs";

const roundLine = (fields) => JSON.stringify({ ts: "2026-08-13T00:00:00.000Z", event: "round", ...fields });

test("parses valid round lines and skips corrupt ones", () => {
	const raw = [roundLine({ failing: ["M1"] }), "{ not json", roundLine({ failing: [] })].join("\n");
	const records = parseGauntletLines(raw);
	assert.equal(records.length, 2);
});

test("collects failing metrics from rounds", () => {
	const rounds = [
		{ event: "round", failing: ["M1", "M2"] },
		{ event: "round", failing: ["M1"] },
		{ event: "round", failing: [] },
	];
	assert.deepEqual(collectFailingMetrics(rounds), ["M1", "M2", "M1"]);
});

test("clusters by metric id", () => {
	assert.deepEqual(clusterFailingMetrics(["M1", "M2", "M1"]), [
		{ metric: "M1", count: 2 },
		{ metric: "M2", count: 1 },
	]);
});

test("recurring = threshold or more", () => {
	assert.equal(recurringFailures([{ metric: "M1", count: 2 }], 2).length, 1);
	assert.equal(recurringFailures([{ metric: "M1", count: 1 }], 2).length, 0);
});

test("evaluate returns no recurring when no metric repeats", () => {
	const raw = [roundLine({ failing: ["M1"] }), roundLine({ failing: ["M2"] })].join("\n");
	const result = evaluate(raw);
	assert.equal(result.recurring.length, 0);
	assert.match(formatReport(result), /no recurring gauntlet failure/);
});

test("evaluate returns recurring when same metric fails twice", () => {
	const raw = [roundLine({ failing: ["M1"] }), roundLine({ failing: ["M1"] })].join("\n");
	const result = evaluate(raw, { threshold: 2 });
	assert.equal(result.recurring.length, 1);
	assert.equal(result.recurring[0].metric, "M1");
	assert.match(formatReport(result), /recurring gauntlet failures/);
});

test("unreadable evidence exits 2 rather than passing silently", () => {
	const code = main(["--log=/nonexistent/gauntlet-audit.jsonl"]);
	assert.equal(code, 2);
});