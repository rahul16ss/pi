/**
 * Unit tests for the gauntlet extension: pure scoring plus the fake-runtime
 * macro loop (start / metrics / critic skip / persist / settle / stop).
 *
 * The macro-loop verdict parser (parseCard, decideNext, extractAmendment,
 * buildRoundPrompt, …) shipped without tests; F-14 proved untested metrics
 * ship broken. These tests lock the contracts the audit called out.
 */
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	amendmentApplyInstruction,
	buildRoundPrompt,
	CRITIC_DISABLE_VERIFY_JSON,
	CRITIC_GIT_SNAPSHOT_CMD,
	classifyAmendment,
	createGauntletSession,
	DEFAULT_BOUNDARY,
	decideNext,
	evaluateRound,
	extractAmendment,
	fingerprintRound,
	type GauntletCard,
	type GauntletHost,
	type GauntletState,
	handleGauntletCommand,
	handleGauntletSettled,
	handleGauntletTransportFailure,
	humanEscalationHit,
	isTransportFailure,
	loadGauntletState,
	type MetricResult,
	parseCard,
	parseFindings,
	type RoundRecord,
	saveGauntletState,
	sumVerifyCostSince,
} from "../../../../.pi/agent/extensions/gauntlet.ts";
import { InMemorySettingsStorage, SettingsManager } from "../src/core/settings-manager.ts";

const dirs: string[] = [];
afterEach(() => {
	for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function card(over: Partial<GauntletCard> = {}): GauntletCard {
	return {
		objective: "Hold the bar.",
		metrics: [{ id: "M1", cmd: "true" }],
		boundary: { max_rounds: 5, stop_on_repeat: 2, escalate_human: ["secrets"] },
		...over,
	};
}

function state(over: Partial<GauntletState> = {}): GauntletState {
	return {
		active: true,
		startedAt: "2026-08-15T00:00:00.000Z",
		round: 1,
		status: "running",
		repeatCount: 0,
		lastFingerprint: "",
		strategyDirectiveIssued: false,
		rounds: [],
		...over,
	};
}

function record(over: Partial<RoundRecord> = {}): RoundRecord {
	return {
		round: 1,
		at: "2026-08-15T00:00:01.000Z",
		results: [{ id: "M1", pass: false, evidence: "fail", findings: ["P0: broken"] }],
		fingerprint: "m1|p0: broken",
		...over,
	};
}

describe("parseCard", () => {
	it("reads a valid json block", () => {
		const md = `# Card\n\n\`\`\`json\n${JSON.stringify({
			objective: "Ship it.",
			metrics: [{ id: "M1", cmd: "npm test" }],
		})}\n\`\`\`\n`;
		const { card: parsed, error } = parseCard(md);
		expect(error).toBeUndefined();
		expect(parsed?.objective).toBe("Ship it.");
		expect(parsed?.metrics[0]?.id).toBe("M1");
	});

	it("rejects a missing json block", () => {
		expect(parseCard("# no json").error).toMatch(/no ```json/i);
	});

	it("rejects a metric with both cmd and critic", () => {
		const md = `\`\`\`json\n${JSON.stringify({
			objective: "x",
			metrics: [{ id: "M1", cmd: "true", critic: "review" }],
		})}\n\`\`\`\n`;
		expect(parseCard(md).error).toMatch(/exactly one/i);
	});

	it("rejects an empty metrics array", () => {
		const md = `\`\`\`json\n${JSON.stringify({ objective: "x", metrics: [] })}\n\`\`\`\n`;
		expect(parseCard(md).error).toMatch(/non-empty array/i);
	});

	it("rejects an empty objective", () => {
		const md = `\`\`\`json\n${JSON.stringify({ objective: "  ", metrics: [{ id: "M1", cmd: "true" }] })}\n\`\`\`\n`;
		expect(parseCard(md).error).toMatch(/objective/i);
	});

	it("rejects a metric with no id", () => {
		const md = `\`\`\`json\n${JSON.stringify({ objective: "x", metrics: [{ cmd: "true" }] })}\n\`\`\`\n`;
		expect(parseCard(md).error).toMatch(/needs an id/i);
	});

	it("rejects a metric with neither cmd nor critic", () => {
		const md = `\`\`\`json\n${JSON.stringify({ objective: "x", metrics: [{ id: "M1" }] })}\n\`\`\`\n`;
		expect(parseCard(md).error).toMatch(/exactly one/i);
	});

	it("rejects unparseable json", () => {
		const md = "```json\n{not valid json}\n```";
		expect(parseCard(md).error).toMatch(/does not parse/i);
	});
});

describe("fingerprintRound / parseFindings", () => {
	it("fingerprints by failing ids and finding headlines, not noisy evidence", () => {
		const a: MetricResult[] = [{ id: "M1", pass: false, evidence: "noise A", findings: ["P0: missing tests"] }];
		const b: MetricResult[] = [
			{ id: "M1", pass: false, evidence: "noise B totally different", findings: ["P0: missing tests"] },
		];
		expect(fingerprintRound(a)).toBe(fingerprintRound(b));
	});

	it("parses P0/P1 lines and treats NONE as empty", () => {
		expect(parseFindings("P0: boom\nP1: debt\n")).toEqual(["P0: boom", "P1: debt"]);
		expect(parseFindings("NONE")).toEqual([]);
	});
});

describe("extractAmendment", () => {
	it("pulls the diff, rationale, and moves: ids", () => {
		const reply = [
			"Tighten the independence gate.",
			"```diff",
			"--- a/.pi/gauntlet/CARD.md",
			"+++ b/.pi/gauntlet/CARD.md",
			"+new metric",
			"```",
			"moves: M1, M2",
		].join("\n");
		const am = extractAmendment(reply, "M3");
		expect(am?.rationale).toBe("Tighten the independence gate.");
		expect(am?.diff).toContain("CARD.md");
		expect(am?.targetMetrics).toEqual(["M1", "M2"]);
		expect(am?.sourceMetric).toBe("M3");
	});

	it("returns undefined when there is no fenced diff", () => {
		expect(extractAmendment("P0: still broken", "M3")).toBeUndefined();
	});

	it("returns empty targetMetrics when moves: line is absent", () => {
		const reply = ["Rationale line.", "```diff", "--- a/x.ts", "+++ b/x.ts", "+code", "```"].join("\n");
		const am = extractAmendment(reply, "M3");
		expect(am).toBeTruthy();
		expect(am?.targetMetrics).toEqual([]);
		expect(am?.rationale).toBe("Rationale line.");
	});
});

describe("decideNext", () => {
	it("passes only a clean round with no amendment", () => {
		const next = decideNext({
			state: state(),
			card: card(),
			record: record({
				results: [{ id: "M1", pass: true, evidence: "ok", findings: [] }],
				fingerprint: "",
			}),
		});
		expect(next).toEqual({ action: "pass" });
	});

	it("does not pass when a harness amendment is pending", () => {
		const next = decideNext({
			state: state(),
			card: card(),
			record: record({
				results: [{ id: "M1", pass: true, evidence: "ok", findings: [] }],
				fingerprint: "",
				amendment: {
					diff: "--- a/x\n+++ b/x",
					rationale: "tighten",
					targetMetrics: ["M1"],
					sourceMetric: "M3",
				},
			}),
		});
		expect(next.action).not.toBe("pass");
	});

	it("stops on human escalation before max-rounds", () => {
		const next = decideNext({
			state: state({ round: 1 }),
			card: card(),
			record: record({
				results: [{ id: "M1", pass: false, evidence: "touched secrets", findings: [] }],
			}),
		});
		expect(next).toEqual({ action: "stop", reason: "human-escalation (secrets)" });
	});

	it("stops on max-rounds", () => {
		const next = decideNext({
			state: state({ round: 5 }),
			card: card(),
			record: record(),
		});
		expect(next).toEqual({ action: "stop", reason: "max-rounds (5)" });
	});

	it("stops on cost budget", () => {
		const next = decideNext({
			state: state(),
			card: card({ boundary: { max_rounds: 5, max_cost_usd: 1, stop_on_repeat: 2 } }),
			record: record(),
			costUsd: 1.5,
		});
		expect(next.action).toBe("stop");
		if (next.action === "stop") expect(next.reason).toMatch(/cost-budget/);
	});

	it("issues strategy-change once on repeat, then stops", () => {
		const first = decideNext({
			state: state({ repeatCount: 2, strategyDirectiveIssued: false }),
			card: card(),
			record: record(),
		});
		expect(first).toEqual({ action: "strategy-change" });
		const second = decideNext({
			state: state({ repeatCount: 2, strategyDirectiveIssued: true }),
			card: card(),
			record: record(),
		});
		expect(second.action).toBe("stop");
	});

	it("continues when there is still budget and no repeat", () => {
		const next = decideNext({
			state: state({ round: 1, repeatCount: 0 }),
			card: card(),
			record: record(),
		});
		expect(next).toEqual({ action: "continue" });
	});
});

describe("classifyAmendment / amendmentApplyInstruction / buildRoundPrompt", () => {
	it("classifies card, prompt, settings, and source diffs", () => {
		expect(classifyAmendment("--- a/.pi/gauntlet/CARD.md\n+++ b/.pi/gauntlet/CARD.md")).toBe("card");
		expect(classifyAmendment("--- a/.pi/prompts/verify.md\n+++ b/.pi/prompts/verify.md")).toBe("prompt");
		expect(classifyAmendment("--- a/settings.json\n+++ b/settings.json")).toBe("settings");
		expect(classifyAmendment("--- a/src/core/verify-routing.ts\n+++ b/src/core/verify-routing.ts")).toBe("source");
	});

	it("classifies bare card.md without the .pi/gauntlet path prefix", () => {
		expect(classifyAmendment("--- a/CARD.md\n+++ b/CARD.md")).toBe("card");
	});

	it("does not classify a source file that merely mentions card.md in a comment", () => {
		const diff = ["--- a/src/foo.ts", "+++ b/src/foo.ts", "+// see CARD.md for details"].join("\n");
		expect(classifyAmendment(diff)).toBe("source");
	});

	it("source instruction tells the builder to stop and not claim the change is live", () => {
		const text = amendmentApplyInstruction("source");
		expect(text).toMatch(/npm run build/i);
		expect(text).toMatch(/NOT ambient/);
	});

	it("card instruction says it takes effect next round without a restart", () => {
		const text = amendmentApplyInstruction("card");
		expect(text).toMatch(/next round/i);
		expect(text).not.toMatch(/npm run build/i);
	});

	it("prepends a pending amendment and the largest gap", () => {
		const prompt = buildRoundPrompt({
			card: card(),
			record: record({
				amendment: {
					diff: "--- a/.pi/gauntlet/CARD.md\n+++ b/.pi/gauntlet/CARD.md\n+metric",
					rationale: "Add a metric.",
					targetMetrics: ["M1"],
					sourceMetric: "M3",
				},
			}),
			state: state({ round: 1 }),
			strategyChange: false,
		});
		expect(prompt).toMatch(/PROPOSED HARNESS AMENDMENT/);
		expect(prompt).toMatch(/Add a metric/);
		expect(prompt).toMatch(/LARGEST GAP/);
		expect(prompt).toMatch(/explicit approval/);
	});

	it("includes a STRATEGY CHANGE directive when strategyChange is true", () => {
		const prompt = buildRoundPrompt({
			card: card(),
			record: record(),
			state: state({ round: 2 }),
			strategyChange: true,
		});
		expect(prompt).toMatch(/STRATEGY CHANGE REQUIRED/);
		expect(prompt).toMatch(/different approach/i);
	});

	it("lists prior failed approaches when rounds have fingerprints", () => {
		const prompt = buildRoundPrompt({
			card: card(),
			record: record({ round: 3 }),
			state: state({
				round: 3,
				rounds: [
					{ round: 1, at: "", results: [], fingerprint: "m1|p0: old failure" },
					{ round: 2, at: "", results: [], fingerprint: "m1|p0: second failure" },
					{ round: 3, at: "", results: [], fingerprint: "m1|p0: current" },
				],
			}),
			strategyChange: false,
		});
		expect(prompt).toMatch(/already failed/i);
		expect(prompt).toContain("m1|p0: old failure");
		expect(prompt).toContain("m1|p0: second failure");
		expect(prompt).not.toContain("m1|p0: current");
	});

	it("includes out_of_scope items when the card has them", () => {
		const prompt = buildRoundPrompt({
			card: card({ out_of_scope: ["secrets", "deploy keys"] }),
			record: record(),
			state: state({ round: 1 }),
			strategyChange: false,
		});
		expect(prompt).toMatch(/Out of scope/i);
		expect(prompt).toContain("secrets");
		expect(prompt).toContain("deploy keys");
	});
});

describe("sumVerifyCostSince / isTransportFailure / humanEscalationHit", () => {
	it("sums run-summary costs on or after the start timestamp", () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-"));
		dirs.push(dir);
		const log = join(dir, "audit.jsonl");
		writeFileSync(
			log,
			[
				JSON.stringify({ event: "run-summary", ts: "2026-08-14T00:00:00.000Z", costUsd: 9 }),
				JSON.stringify({ event: "run-summary", ts: "2026-08-15T01:00:00.000Z", costUsd: 1.25 }),
				JSON.stringify({ event: "triage", ts: "2026-08-15T02:00:00.000Z", costUsd: 0.5 }),
				"{not json}\n",
			].join("\n"),
		);
		expect(sumVerifyCostSince(log, "2026-08-15T00:00:00.000Z")).toBe(1.25);
		expect(sumVerifyCostSince(join(dir, "missing.jsonl"), "2026-08-15T00:00:00.000Z")).toBe(0);
	});

	it("critic project settings disable verify routing via empty checkerModel", () => {
		const storage = new InMemorySettingsStorage();
		storage.withLock("global", () =>
			JSON.stringify({
				verify: { checkerModel: "openrouter/openai/gpt-5.6-luna" },
			}),
		);
		storage.withLock("project", () => '{"verify":{"checkerModel":""}}');
		const mgr = SettingsManager.fromStorage(storage);
		expect(mgr.getVerifySettings()?.checkerModel).toBe("");
	});

	it("treats only hard stop/error as transport failure", () => {
		expect(isTransportFailure({ role: "assistant", stopReason: "error" }).failed).toBe(true);
		expect(isTransportFailure({ role: "assistant", stopReason: "aborted" }).failed).toBe(true);
		expect(isTransportFailure({ role: "assistant", errorMessage: "boom" }).failed).toBe(true);
		expect(isTransportFailure({ role: "assistant", content: "the previous approach failed" }).failed).toBe(false);
		expect(isTransportFailure({ role: "user", content: "error happened" }).failed).toBe(false);
		expect(isTransportFailure({ role: "toolResult", stopReason: "error" }).failed).toBe(false);
	});

	it("hits escalate_human keywords case-insensitively", () => {
		expect(
			humanEscalationHit(card(), record({ results: [{ id: "M1", pass: false, evidence: "SECRETS leaked" }] })),
		).toBe("secrets");
		expect(DEFAULT_BOUNDARY.max_rounds).toBe(5);
	});

	it("escalates when a passing metric's evidence still mentions a keyword", () => {
		expect(
			humanEscalationHit(
				card({ boundary: { max_rounds: 5, stop_on_repeat: 2, escalate_human: ["credentials"] } }),
				record({ results: [{ id: "M1", pass: true, evidence: "No hardcoded credentials found" }] }),
			),
		).toBe("credentials");
	});

	it("returns undefined when no escalation keywords match", () => {
		expect(
			humanEscalationHit(card(), record({ results: [{ id: "M1", pass: false, evidence: "broken tests" }] })),
		).toBeUndefined();
	});

	it("returns undefined when the card has no escalate_human keywords", () => {
		const noEsc = card({ boundary: { max_rounds: 5, stop_on_repeat: 2 } });
		expect(
			humanEscalationHit(noEsc, record({ results: [{ id: "M1", pass: false, evidence: "touched secrets" }] })),
		).toBeUndefined();
	});
});

function writeCard(dir: string, over: Partial<GauntletCard> = {}): void {
	mkdirSync(join(dir, ".pi", "gauntlet"), { recursive: true });
	const body = card({
		metrics: [
			{ id: "M1", cmd: "npm test" },
			{ id: "M2", critic: "review the diff" },
		],
		boundary: { max_rounds: 5, stop_on_repeat: 2 },
		...over,
	});
	writeFileSync(
		join(dir, ".pi", "gauntlet", "CARD.md"),
		`# Card\n\n\`\`\`json\n${JSON.stringify(body, null, 2)}\n\`\`\`\n`,
	);
}

function fakeHost(dir: string): {
	host: GauntletHost;
	notices: string[];
	messages: string[];
	calls: Array<{ command: string; args: string[]; cwd: string }>;
	criticSettings: string[];
	setCmdCode: (code: number) => void;
	setCritic: (res: { code: number; stdout: string; stderr?: string }) => void;
} {
	const notices: string[] = [];
	const messages: string[] = [];
	const calls: Array<{ command: string; args: string[]; cwd: string }> = [];
	const criticSettings: string[] = [];
	let cmdCode = 0;
	let critic = { code: 0, stdout: "NONE", stderr: "" };
	const host: GauntletHost = {
		exec: async (command, args, opts) => {
			calls.push({ command, args, cwd: opts.cwd });
			if (command === "pi") {
				criticSettings.push(readFileSync(join(opts.cwd, ".pi", "settings.json"), "utf8"));
				return { code: critic.code, stdout: critic.stdout, stderr: critic.stderr, killed: false };
			}
			if (command === "bash" && args[1] === CRITIC_GIT_SNAPSHOT_CMD) {
				return { code: 0, stdout: " M tracked.ts\n", stderr: "", killed: false };
			}
			return { code: cmdCode, stdout: cmdCode === 0 ? "ok" : "fail", stderr: "", killed: false };
		},
		notify: (message) => notices.push(message),
		sendUserMessage: (content) => messages.push(content),
		waitForIdle: async () => {},
		auditLogPath: join(dir, "audit.jsonl"),
	};
	return {
		host,
		notices,
		messages,
		calls,
		criticSettings,
		setCmdCode: (code) => {
			cmdCode = code;
		},
		setCritic: (res) => {
			critic = { stderr: "", ...res };
		},
	};
}

describe("gauntlet runtime loop", () => {
	it("start without a card notifies an error and does not exec", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		const { host, notices, calls } = fakeHost(dir);
		await handleGauntletCommand("start", dir, createGauntletSession(), host);
		expect(notices.some((n) => /no gauntlet card/i.test(n))).toBe(true);
		expect(calls).toHaveLength(0);
		expect(loadGauntletState(dir)).toBeUndefined();
	});

	it("failing command metrics skip the critic and persist a continue round", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		writeCard(dir);
		const session = createGauntletSession();
		const { host, messages, calls, criticSettings, setCmdCode } = fakeHost(dir);
		setCmdCode(1);
		await handleGauntletCommand("start", dir, session, host);
		expect(calls.some((c) => c.command === "pi")).toBe(false);
		expect(calls.some((c) => c.args[1] === CRITIC_GIT_SNAPSHOT_CMD)).toBe(false);
		expect(criticSettings).toHaveLength(0);
		const saved = loadGauntletState(dir);
		expect(saved?.active).toBe(true);
		expect(saved?.status).toBe("running");
		expect(saved?.round).toBe(1);
		expect(saved?.rounds[0]?.results.find((r) => r.id === "M2")?.evidence).toMatch(
			/skipped: command metrics failing/i,
		);
		expect(session.awaitingRound).toBe(true);
		expect(messages[0]).toMatch(/LARGEST GAP/i);
		expect(messages[0]).not.toMatch(/GAUNTLET PASSED/);
	});

	it("passing commands run the critic with verify disabled in a temp cwd", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		writeCard(dir);
		const { host, notices, messages, calls, criticSettings } = fakeHost(dir);
		await handleGauntletCommand("start", dir, createGauntletSession(), host);
		const piCall = calls.find((c) => c.command === "pi");
		expect(piCall?.args).toEqual(
			expect.arrayContaining(["-p", "--no-session", "--model", "openrouter/openai/gpt-5.6-luna"]),
		);
		expect(piCall?.cwd).not.toBe(dir);
		expect(piCall?.cwd).toMatch(/pi-gauntlet-critic-/);
		expect(criticSettings).toEqual([CRITIC_DISABLE_VERIFY_JSON]);
		expect(calls.some((c) => c.command === "bash" && c.args[1] === CRITIC_GIT_SNAPSHOT_CMD && c.cwd === dir)).toBe(
			true,
		);
		const saved = loadGauntletState(dir);
		expect(saved?.status).toBe("passed");
		expect(saved?.active).toBe(false);
		expect(notices.some((n) => /GAUNTLET PASSED/i.test(n))).toBe(true);
		expect(messages[0]).toMatch(/GAUNTLET PASSED/);
	});

	it("a critic that cannot run is fail-closed, not a pass", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		writeCard(dir);
		const { host, messages, setCritic } = fakeHost(dir);
		setCritic({ code: 1, stdout: "", stderr: "model down" });
		await handleGauntletCommand("start", dir, createGauntletSession(), host);
		const saved = loadGauntletState(dir);
		expect(saved?.status).toBe("running");
		expect(saved?.rounds[0]?.results.find((r) => r.id === "M2")?.pass).toBe(false);
		expect(saved?.rounds[0]?.results.find((r) => r.id === "M2")?.findings?.[0]).toMatch(/could not run/i);
		expect(messages[0]).toMatch(/NOT A PASS|LARGEST GAP/i);
		expect(messages.join("\n")).not.toMatch(/GAUNTLET PASSED/);
	});

	it("stop persists stopped-user and clears the awaiting flag", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		writeCard(dir);
		const session = createGauntletSession();
		const { host, setCmdCode, notices } = fakeHost(dir);
		setCmdCode(1);
		await handleGauntletCommand("start", dir, session, host);
		expect(session.awaitingRound).toBe(true);
		await handleGauntletCommand("stop", dir, session, host);
		expect(session.awaitingRound).toBe(false);
		expect(loadGauntletState(dir)?.status).toBe("stopped-user");
		expect(notices.some((n) => /stopped by user/i.test(n))).toBe(true);
	});

	it("status with no state tells the user how to arm", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		const { host, notices } = fakeHost(dir);
		await handleGauntletCommand("status", dir, createGauntletSession(), host);
		expect(notices[0]).toMatch(/no state/i);
	});

	it("agent_settled re-enters the loop only while a round is awaiting", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		writeCard(dir);
		const session = createGauntletSession();
		const { host, calls, setCmdCode } = fakeHost(dir);
		setCmdCode(1);
		await handleGauntletCommand("start", dir, session, host);
		const afterStart = calls.length;
		await handleGauntletSettled(dir, session, host);
		expect(loadGauntletState(dir)?.round).toBe(2);
		expect(calls.length).toBeGreaterThan(afterStart);
		expect(session.awaitingRound).toBe(true);

		const idle = createGauntletSession();
		const beforeIdle = calls.length;
		await handleGauntletSettled(dir, idle, host);
		expect(calls.length).toBe(beforeIdle);
		expect(loadGauntletState(dir)?.round).toBe(2);
	});

	it("a provider failure while awaiting a round stops honestly", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		writeCard(dir);
		const session = createGauntletSession();
		const { host, notices, setCmdCode } = fakeHost(dir);
		setCmdCode(1);
		await handleGauntletCommand("start", dir, session, host);
		handleGauntletTransportFailure({ role: "assistant", stopReason: "error" }, dir, session, host);
		expect(session.awaitingRound).toBe(false);
		expect(loadGauntletState(dir)?.status).toBe("stopped-provider-error");
		expect(loadGauntletState(dir)?.active).toBe(false);
		expect(notices.some((n) => /provider failure/i.test(n))).toBe(true);
	});

	it("evaluateRound is a no-op while already evaluating or shutting down", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		writeCard(dir);
		saveGauntletState(dir, {
			active: true,
			startedAt: "2026-08-16T00:00:00.000Z",
			round: 0,
			status: "running",
			repeatCount: 0,
			lastFingerprint: "",
			strategyDirectiveIssued: false,
			rounds: [],
		});
		const { host, calls } = fakeHost(dir);
		const evaluating = createGauntletSession();
		evaluating.evaluating = true;
		await evaluateRound(dir, evaluating, host);
		expect(calls).toHaveLength(0);
		expect(loadGauntletState(dir)?.round).toBe(0);

		const shutting = createGauntletSession();
		shutting.shuttingDown = true;
		await evaluateRound(dir, shutting, host);
		expect(calls).toHaveLength(0);
		expect(loadGauntletState(dir)?.round).toBe(0);
	});

	it("start resumes a running state instead of resetting the round counter", async () => {
		const dir = mkdtempSync(join(tmpdir(), "gauntlet-loop-"));
		dirs.push(dir);
		writeCard(dir);
		saveGauntletState(dir, {
			active: true,
			startedAt: "2026-08-16T00:00:00.000Z",
			round: 2,
			status: "running",
			repeatCount: 0,
			lastFingerprint: "",
			strategyDirectiveIssued: false,
			rounds: [],
		});
		const { host, setCmdCode } = fakeHost(dir);
		setCmdCode(1);
		await handleGauntletCommand("start", dir, createGauntletSession(), host);
		expect(loadGauntletState(dir)?.round).toBe(3);
	});
});
