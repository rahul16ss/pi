#!/usr/bin/env node
/**
 * Gauntlet metric M2 — recurring gauntlet metric failures in recent rounds.
 *
 * Passes when no metric id fails in >=2 of the last 10 gauntlet rounds.
 * Extracted from the inline CARD.md one-liner so it can be unit-tested.
 *
 * Exit codes:
 *   0 — no metric id recurs as failing across rounds
 *   1 — at least one metric id fails in >=2 recent rounds
 *   2 — the metric could not read its evidence (never silently pass)
 */

import { readFileSync } from "node:fs";

export const DEFAULT_GAUNTLET_LOG = "/Users/rahul/.pi/agent/gauntlet-audit.jsonl";
export const DEFAULT_WINDOW = 10;
export const DEFAULT_THRESHOLD = 2;

export function parseGauntletLines(raw) {
	const records = [];
	for (const line of String(raw).split("\n")) {
		const trimmed = line.trim();
		if (!trimmed) continue;
		try {
			records.push(JSON.parse(trimmed));
		} catch {
			// skip corrupt lines
		}
	}
	return records;
}

export function selectRecentRounds(records, window = DEFAULT_WINDOW) {
	const rounds = records.filter((r) => r && r.event === "round");
	return window > 0 ? rounds.slice(-window) : rounds;
}

export function collectFailingMetrics(rounds) {
	return rounds.flatMap((r) => {
		const failing = Array.isArray(r.failing) ? r.failing : [];
		return failing.map((f) => String(f).trim().slice(0, 40));
	});
}

export function clusterFailingMetrics(metrics) {
	const counts = new Map();
	for (const m of metrics) {
		if (!m) continue;
		counts.set(m, (counts.get(m) ?? 0) + 1);
	}
	return [...counts.entries()]
		.map(([metric, count]) => ({ metric, count }))
		.sort((a, b) => b.count - a.count || a.metric.localeCompare(b.metric));
}

export function recurringFailures(clusters, threshold = DEFAULT_THRESHOLD) {
	return clusters.filter((c) => c.count >= threshold);
}

export function evaluate(raw, options = {}) {
	const window = options.window ?? DEFAULT_WINDOW;
	const threshold = options.threshold ?? DEFAULT_THRESHOLD;
	const records = parseGauntletLines(raw);
	const rounds = selectRecentRounds(records, window);
	const failing = collectFailingMetrics(rounds);
	const clusters = clusterFailingMetrics(failing);
	const recurring = recurringFailures(clusters, threshold);
	return { rounds: rounds.length, failing, clusters, recurring, window, threshold };
}

export function formatReport(result) {
	const scope = `${result.failing.length} failing metric(s) in last ${result.rounds} round(s)`;
	if (result.recurring.length === 0) {
		return `no recurring gauntlet failure (>=${result.threshold} rounds); ${scope}`;
	}
	const lines = result.recurring.map((c) => `- ${c.metric} (${c.count} rounds)`);
	return `recurring gauntlet failures (>=${result.threshold} rounds); ${scope}:\n${lines.join("\n")}`;
}

export function main(argv = process.argv.slice(2)) {
	let logPath = DEFAULT_GAUNTLET_LOG;
	for (const arg of argv) {
		const [key, value] = arg.split("=");
		if (key === "--log" && value) logPath = value;
	}
	let raw;
	try {
		raw = readFileSync(logPath, "utf8");
	} catch (error) {
		console.error(`M2 could not read ${logPath}: ${error.message}`);
		return 2;
	}
	const result = evaluate(raw);
	console.log(formatReport(result));
	return result.recurring.length === 0 ? 0 : 1;
}

if (import.meta.url === `file://${process.argv[1]}`) {
	process.exit(main());
}