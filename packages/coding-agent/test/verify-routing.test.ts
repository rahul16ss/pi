import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage, BeforeFirstTurnContext, VerifyTurnContext } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it } from "vitest";
import type { VerifySettings } from "../src/core/settings-manager.ts";
import {
	boundContextForModel,
	buildTriagePrompt,
	CHECKER_VERDICT_LINE_INSTRUCTION,
	checkerRefForKind,
	classifyCheckerVerdict,
	classifyTriageTier,
	createVerifyRouting,
	EXCELLENCE_CHARTER,
	extractModelFamily,
	messageText,
	parseCheckerVerdict,
	redactEvidence,
	selectAuditChecker,
	shouldEscalateMaker,
	shouldPlanFirst,
	shouldRunCheckpoint,
	tallyMakerTrace,
} from "../src/core/verify-routing.ts";

function modelWithId(id: string): Model<any> {
	return {
		id,
		name: id,
		api: "openai-completions",
		provider: "openrouter",
		baseUrl: "https://openrouter.ai/api/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 8192,
	};
}

function assistant(text: string, modelId = "mock", extra: Partial<AssistantMessage> = {}): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "openai-completions",
		provider: "openrouter",
		model: modelId,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
		...extra,
	};
}

function toolResult(toolName: string, text: string): AgentMessage {
	return {
		role: "toolResult",
		toolCallId: `call_${toolName}`,
		toolName,
		content: [{ type: "text", text }],
		isError: false,
		timestamp: Date.now(),
	};
}

describe("extractModelFamily / selectAuditChecker", () => {
	it("treats openrouter-prefixed ids as the inner company family", () => {
		expect(extractModelFamily("openrouter/openai/gpt-5.6-luna")).toBe("openai");
		expect(extractModelFamily("openai/gpt-5.6-luna")).toBe("openai");
		expect(extractModelFamily("openrouter/deepseek/deepseek-v4-pro-0813")).toBe("deepseek");
	});

	it("keeps an independent spare as availability fallback", () => {
		const pick = selectAuditChecker({
			liveMakerRef: "openrouter/deepseek/deepseek-v4-pro-0813",
			checkerRef: "openrouter/openai/gpt-5.6-luna",
			checkerFallbackRef: "openrouter/meta/muse-spark-1.2",
		});
		expect(pick.primaryRef).toContain("gpt-5.6-luna");
		expect(pick.fallbackRef).toContain("muse-spark");
		expect(pick.swapped).toBe(false);
	});

	it("promotes the spare when the live maker collides with the checker family", () => {
		const pick = selectAuditChecker({
			liveMakerRef: "openai/gpt-5.6-luna",
			checkerRef: "openrouter/openai/gpt-5.6-luna",
			checkerFallbackRef: "openrouter/meta/muse-spark-1.2",
		});
		expect(pick.primaryRef).toContain("muse-spark");
		expect(pick.fallbackRef).toBeUndefined();
		expect(pick.swapped).toBe(true);
		expect(pick.reason).toBe("maker-family-collides-with-checker");
	});

	it("drops a spare that is the same family as the live maker or the checker", () => {
		expect(
			selectAuditChecker({
				liveMakerRef: "openrouter/meta/muse-spark-1.2",
				checkerRef: "openrouter/openai/gpt-5.6-luna",
				checkerFallbackRef: "meta/muse-spark-1.2",
			}).fallbackRef,
		).toBeUndefined();
		expect(
			selectAuditChecker({
				liveMakerRef: "openrouter/deepseek/deepseek-v4-pro-0813",
				checkerRef: "openrouter/openai/gpt-5.6-luna",
				checkerFallbackRef: "openai/gpt-5.6-luna",
			}).fallbackRef,
		).toBeUndefined();
	});
});

describe("classifyCheckerVerdict", () => {
	it("prefers CONFLICT and does not treat unverified as VERIFIED", () => {
		expect(classifyCheckerVerdict("VERIFIED")).toBe("verified");
		expect(classifyCheckerVerdict("Looks good.\nVERIFIED")).toBe("verified");
		expect(classifyCheckerVerdict("Missing tests. CONFLICT")).toBe("conflict");
		expect(classifyCheckerVerdict("claims are unverified")).toBe("ambiguous");
		expect(classifyCheckerVerdict("claims are unverified. CONFLICT")).toBe("conflict");
		expect(classifyCheckerVerdict("")).toBe("ambiguous");
		expect(classifyCheckerVerdict("maybe ok")).toBe("ambiguous");
	});

	it("an isolated first-line verdict is authoritative and its rationale is never scanned (F-05)", () => {
		// Observed live on 2026-08-13: the final checker opened with VERIFIED and
		// then quoted the rejection criterion, and the prose scan recorded a
		// rejection — wasting a full maker+checker cycle.
		expect(
			classifyCheckerVerdict(
				"VERIFIED\n\nThe receipts substantiate the claims:\n- report exists\n\nNo grounds for CONFLICT were found.",
			),
		).toBe("verified");
		expect(classifyCheckerVerdict("VERIFIED\nThe CONFLICT criterion was not met.")).toBe("verified");
		expect(classifyCheckerVerdict("VERDICT: VERIFIED\nNothing here meets CONFLICT.")).toBe("verified");
		expect(classifyCheckerVerdict("**VERIFIED**\n\nTests ran; no CONFLICT.")).toBe("verified");

		// A leading rejection verdict still rejects, whatever the rationale says.
		expect(classifyCheckerVerdict("VERDICT: CONFLICT\nThe tests were never VERIFIED.")).toBe("conflict");
		expect(classifyCheckerVerdict("CONFLICT\n\nThe maker never ran the suite.")).toBe("conflict");

		// Provenance is reported so adoption of the structured format is measurable.
		expect(parseCheckerVerdict("VERDICT: VERIFIED\nwhy").via).toBe("verdict-line");
		expect(parseCheckerVerdict("VERIFIED\nwhy").via).toBe("bare-token-line");
		expect(parseCheckerVerdict("Missing tests. CONFLICT").via).toBe("legacy-scan");
		expect(parseCheckerVerdict("").via).toBe("empty");
	});

	it("a leading verdict token with same-line rationale is authoritative (F-05 same-line case)", () => {
		// The exact live 2026-08-13 false-conflict event: the checker opened with
		// "VERIFIED." and then explained on the SAME line, mentioning CONFLICT in
		// the explanation. The isolated-line rules miss this because the rationale
		// is on the same line as the token, not a later line. The leading-token
		// stage treats the first word as the verdict and ignores the rest.
		expect(
			classifyCheckerVerdict(
				"VERIFIED. The receipts substantiate the report. The rejection criterion CONFLICT was not triggered because the evidence is complete.",
			),
		).toBe("verified");
		expect(parseCheckerVerdict("VERIFIED. The receipts substantiate the report. CONFLICT mentioned.").via).toBe(
			"leading-token-line",
		);
		expect(classifyCheckerVerdict("CONFLICT. The maker never ran the suite; VERIFIED is not earned.")).toBe(
			"conflict",
		);
		expect(parseCheckerVerdict("CONFLICT. The maker never ran the suite; VERIFIED is not earned.").via).toBe(
			"leading-token-line",
		);
		// Markdown decoration around the token still counts.
		expect(classifyCheckerVerdict("**VERIFIED.** Tests passed; no CONFLICT criterion met.")).toBe("verified");
		// A leading verdict with an em-dash rationale separator.
		expect(classifyCheckerVerdict("VERIFIED — the evidence is complete; CONFLICT does not apply.")).toBe("verified");

		// Discussion of the token (no sentence terminator after it) is NOT a verdict.
		// "VERIFIED is required" and "VERIFIED claims" are prose about the token,
		// not a verdict — they must fall through to legacy-scan so CONFLICT still wins.
		expect(parseCheckerVerdict("VERIFIED is required by the rubric. CONFLICT found.").via).toBe("legacy-scan");
		expect(classifyCheckerVerdict("VERIFIED is required by the rubric. CONFLICT found.")).toBe("conflict");
		expect(parseCheckerVerdict("VERIFIED claims in the log are false. CONFLICT").via).toBe("legacy-scan");
		expect(classifyCheckerVerdict("VERIFIED claims in the log are false. CONFLICT")).toBe("conflict");
	});

	it("keeps the legacy trailing-verdict protocol working so real rejections never fail open", () => {
		// The pre-existing prompt asked for "the specific problem, then the word
		// CONFLICT", so trailing verdicts must keep classifying or genuine
		// rejections would degrade to ambiguous (which fails open).
		expect(classifyCheckerVerdict("Missing tests. CONFLICT")).toBe("conflict");
		expect(classifyCheckerVerdict("Looks good.\nVERIFIED")).toBe("verified");
		// An injected pass token inside quoted evidence must not manufacture a pass.
		expect(classifyCheckerVerdict("The log claims 'VERIFIED' but the suite failed. CONFLICT")).toBe("conflict");
		// A verdict-shaped line that is not isolated is not a verdict line.
		expect(parseCheckerVerdict("The rubric says VERDICT: VERIFIED is required. CONFLICT").via).toBe("legacy-scan");
	});

	it("never reads a negated or hedged VERIFIED as a pass", () => {
		expect(classifyCheckerVerdict("This cannot be VERIFIED")).toBe("ambiguous");
		expect(classifyCheckerVerdict("The claims are not VERIFIED yet")).toBe("ambiguous");
		expect(classifyCheckerVerdict("I can't say VERIFIED without receipts")).toBe("ambiguous");
		expect(classifyCheckerVerdict("un-verified claims here")).toBe("ambiguous");
		// Lowercase prose is not the exact verdict token.
		expect(classifyCheckerVerdict("I verified the tests myself")).toBe("ambiguous");
		// A pass after an unrelated negation in an earlier sentence still counts.
		expect(classifyCheckerVerdict("No issues found.\nVERIFIED")).toBe("verified");
	});
});

describe("classifyTriageTier", () => {
	it("parses the first clear tier token, case-insensitively", () => {
		expect(classifyTriageTier("TRIVIAL")).toBe("trivial");
		expect(classifyTriageTier("This is a STANDARD task.")).toBe("standard");
		expect(classifyTriageTier("hard")).toBe("hard");
		expect(classifyTriageTier("Verdict: HARD — concurrency involved")).toBe("hard");
		expect(classifyTriageTier("no idea")).toBeUndefined();
		expect(classifyTriageTier("")).toBeUndefined();
	});
});

describe("buildTriagePrompt", () => {
	it("asks the model to treat greetings as TRIVIAL, including when a name is used", () => {
		const prompt = buildTriagePrompt({ promptText: "Hi Pi" });
		expect(prompt).toContain("Hi Pi");
		expect(prompt).toContain("TRIVIAL");
		expect(prompt).toContain("greeting");
		expect(prompt).toContain("assistant's name");
		expect(prompt).not.toContain("A coding goal is already in progress");
	});

	it("tells the model that an active goal's follow-up is not small talk", () => {
		const prompt = buildTriagePrompt({ promptText: "yes", goalActive: true });
		expect(prompt).toContain("already in progress");
		expect(prompt).toContain("yes");
	});
});

describe("tallyMakerTrace", () => {
	it("counts unique tool results and assistant turns from the full trace", () => {
		const a1 = assistant("one");
		const a2 = assistant("two");
		const t1 = toolResult("bash", "ok");
		const t2 = toolResult("read", "ok");
		const msgs: AgentMessage[] = [{ role: "user", content: "do work", timestamp: Date.now() }, a1, t1, a2, t2];
		const first = tallyMakerTrace(msgs.slice(0, 3), a1);
		const second = tallyMakerTrace(msgs, a2);
		expect(first.toolCalls).toBe(1);
		expect(first.makerTurns).toBe(1);
		expect(second.toolCalls).toBe(2);
		expect(second.makerTurns).toBe(2);
	});
});

describe("shouldEscalateMaker", () => {
	it("escalates on/after the configured rejection count", () => {
		expect(shouldEscalateMaker({ rejections: 1, escalateAfterRejections: 1 })).toBe(true);
		expect(shouldEscalateMaker({ rejections: 1, escalateAfterRejections: 2 })).toBe(false);
		expect(shouldEscalateMaker({ rejections: 2, escalateAfterRejections: 2 })).toBe(true);
		expect(shouldEscalateMaker({ rejections: 1, escalateAfterRejections: 0 })).toBe(false);
	});
});

describe("checkerRefForKind", () => {
	it("uses checkpointCheckerModel for mid-build and checkerModel for final", () => {
		const verify = {
			checkerModel: "openrouter/moonshotai/kimi-k3",
			checkpointCheckerModel: "openrouter/z-ai/glm-5.2",
		};
		expect(checkerRefForKind(verify, "checkpoint")).toBe("openrouter/z-ai/glm-5.2");
		expect(checkerRefForKind(verify, "final")).toBe("openrouter/moonshotai/kimi-k3");
		expect(checkerRefForKind({ checkerModel: "openrouter/moonshotai/kimi-k3" }, "checkpoint")).toBe(
			"openrouter/moonshotai/kimi-k3",
		);
	});
});

describe("shouldRunCheckpoint", () => {
	it("geometric backoff audits at N, 2N, 4N and never leaves a long tail unaudited", () => {
		const geo = (toolTurnCount: number, checkpointsSoFar: number) =>
			shouldRunCheckpoint({ toolTurnCount, everyN: 6, checkpointsSoFar, maxCheckpoints: 8, backoff: "geometric" });
		expect(geo(5, 0)).toBe(false);
		expect(geo(6, 0)).toBe(true);
		expect(geo(11, 1)).toBe(false);
		expect(geo(12, 1)).toBe(true);
		expect(geo(24, 2)).toBe(true);
		expect(geo(48, 3)).toBe(true); // fixed cadence with a cap of 4 would have gone dark here
	});

	it("fires every N tool turns within the budget", () => {
		expect(shouldRunCheckpoint({ toolTurnCount: 2, everyN: 2, checkpointsSoFar: 0, maxCheckpoints: 3 })).toBe(true);
		expect(shouldRunCheckpoint({ toolTurnCount: 4, everyN: 2, checkpointsSoFar: 1, maxCheckpoints: 3 })).toBe(true);
		expect(shouldRunCheckpoint({ toolTurnCount: 3, everyN: 2, checkpointsSoFar: 0, maxCheckpoints: 3 })).toBe(false);
	});

	it("disables when everyN or max is 0", () => {
		expect(shouldRunCheckpoint({ toolTurnCount: 8, everyN: 0, checkpointsSoFar: 0, maxCheckpoints: 3 })).toBe(false);
		expect(shouldRunCheckpoint({ toolTurnCount: 8, everyN: 8, checkpointsSoFar: 0, maxCheckpoints: 0 })).toBe(false);
	});

	it("stops after max checkpoints", () => {
		expect(shouldRunCheckpoint({ toolTurnCount: 8, everyN: 2, checkpointsSoFar: 3, maxCheckpoints: 3 })).toBe(false);
	});
});

describe("shouldPlanFirst", () => {
	it("requires planFirst, long enough prompt, and not already planned", () => {
		expect(
			shouldPlanFirst({
				planFirst: true,
				promptText: "x".repeat(80),
				minPromptChars: 80,
				alreadyPlanned: false,
			}),
		).toBe(true);
		expect(
			shouldPlanFirst({
				planFirst: true,
				promptText: "hi",
				minPromptChars: 80,
				alreadyPlanned: false,
			}),
		).toBe(false);
		expect(
			shouldPlanFirst({
				planFirst: false,
				promptText: "x".repeat(80),
				minPromptChars: 80,
				alreadyPlanned: false,
			}),
		).toBe(false);
		expect(
			shouldPlanFirst({
				planFirst: true,
				promptText: "x".repeat(80),
				minPromptChars: 80,
				alreadyPlanned: true,
			}),
		).toBe(false);
	});
});

describe("createVerifyRouting", () => {
	const dirs: string[] = [];
	afterEach(() => {
		for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
	});

	function setup(settings: Partial<VerifySettings> = {}) {
		const dir = mkdtempSync(join(tmpdir(), "pi-verify-"));
		dirs.push(dir);
		const auditLogPath = join(dir, "verify-audit.jsonl");
		const models = new Map<string, Model<any>>([
			["openrouter/deepseek/deepseek-v4-flash-0731", modelWithId("deepseek/deepseek-v4-flash-0731")],
			["openrouter/openai/gpt-5.6-luna", modelWithId("openai/gpt-5.6-luna")],
			["openrouter/anthropic/claude-opus-5", modelWithId("anthropic/claude-opus-5")],
			["openrouter/z-ai/glm-5.2", modelWithId("z-ai/glm-5.2")],
			["openrouter/moonshotai/kimi-k3", modelWithId("moonshotai/kimi-k3")],
			["openrouter/meta/muse-spark-1.2", modelWithId("meta/muse-spark-1.2")],
		]);
		const resolveModel = (ref: string | undefined) => {
			if (!ref) return undefined;
			return models.get(ref) ?? [...models.values()].find((m) => m.id === ref || ref.endsWith(`/${m.id}`));
		};
		const calls: string[] = [];
		const replies: Record<string, string[]> = {
			"openai/gpt-5.6-luna": [],
			"anthropic/claude-opus-5": ["1. inspect\n2. implement\n3. test"],
			"z-ai/glm-5.2": [],
			"moonshotai/kimi-k3": [],
			"meta/muse-spark-1.2": [],
		};
		const routing = createVerifyRouting({
			verify: {
				checkerModel: "openrouter/moonshotai/kimi-k3",
				checkpointCheckerModel: "openrouter/z-ai/glm-5.2",
				plannerModel: "openrouter/anthropic/claude-opus-5",
				escalationMakerModel: "openrouter/openai/gpt-5.6-luna",
				checkerThinkingLevel: "max",
				checkpointCheckerThinkingLevel: "max",
				plannerThinkingLevel: "max",
				escalationMakerThinkingLevel: "max",
				planFirst: true,
				planMinPromptChars: 20,
				checkpointEveryToolTurns: 2,
				maxCheckpointsPerRun: 3,
				maxRejections: 2,
				planAfterRejections: 2,
				escalateAfterRejections: 1,
				auditOnlyAfterTools: true,
				triage: { enabled: false },
				...settings,
			},
			makerModel: modelWithId("deepseek/deepseek-v4-flash-0731"),
			makerThinkingLevel: "max",
			cwd: dir,
			auditLogPath,
			resolveModel,
		});
		expect(routing).toBeTruthy();

		const runModel = async (model: Model<any>, _msgs: AgentMessage[]) => {
			calls.push(model.id);
			const queue = replies[model.id] ?? [];
			const text = queue.length > 0 ? (queue.shift() as string) : `answer from ${model.id}`;
			return assistant(text, model.id);
		};

		return { routing: routing!, calls, replies, auditLogPath, runModel, dir };
	}

	function readAudit(path: string) {
		return readFileSync(path, "utf8")
			.trim()
			.split("\n")
			.filter(Boolean)
			.map((line) => JSON.parse(line));
	}

	it("plans first with the planner model and returns maker for execution", async () => {
		const { routing, calls, runModel, auditLogPath } = setup();
		const ctx = {
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{
					role: "user",
					content: "Implement a robust add() helper with tests and lint.",
					timestamp: Date.now(),
				},
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext;

		const prep = await routing.beforeFirstTurn(ctx);
		expect(calls).toEqual(["anthropic/claude-opus-5"]);
		expect(prep?.messages?.[0] && messageText(prep.messages[0] as any)).toContain("[PLANNER]");
		expect(prep?.model?.id).toBe("deepseek/deepseek-v4-flash-0731");

		// A second user prompt is a new run: state resets and the planner runs again.
		const second = await routing.beforeFirstTurn(ctx);
		expect(second?.messages?.length).toBeGreaterThan(0);
		expect(calls).toEqual(["anthropic/claude-opus-5", "anthropic/claude-opus-5"]);
		const audit = readAudit(auditLogPath);
		expect(audit.filter((e) => e.event === "planned").length).toBe(2);
		expect(audit.filter((e) => e.event === "run-start").length).toBe(2);
	});

	it("does not reset or plan on continues that carry no new user message", async () => {
		const { routing, calls, runModel } = setup();
		const prep = await routing.beforeFirstTurn({
			context: {
				systemPrompt: "",
				messages: [{ role: "user", content: "Implement a robust add() helper with tests.", timestamp: Date.now() }],
				tools: [],
			},
			newMessages: [],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		expect(prep).toBeUndefined();
		expect(calls).toEqual([]);
	});

	it("lets triage classify a greeting so the planner never runs", async () => {
		const { routing, calls, replies, runModel, auditLogPath } = setup({
			triage: { enabled: true },
			planFirst: true,
		});
		replies["deepseek/deepseek-v4-flash-0731"] = ["TRIVIAL"];
		const prep = await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "Hi Pi", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		expect(prep?.messages).toBeUndefined();
		expect(calls).toEqual(["deepseek/deepseek-v4-flash-0731"]);
		const audit = readAudit(auditLogPath);
		expect(audit.some((e) => e.event === "triage" && e.tier === "trivial" && e.via === "heuristic")).toBe(false);
		expect(audit.some((e) => e.event === "triage" && e.tier === "trivial")).toBe(true);
		expect(audit.some((e) => e.event === "plan-skipped" && e.reason === "trivial-tier")).toBe(true);
	});

	it("schedules checkpoints every N tool turns up to max", () => {
		const { routing } = setup({ checkpointEveryToolTurns: 2, maxCheckpointsPerRun: 2 });
		expect(routing.shouldCheckpoint({ toolTurnCount: 1 })).toBe(false);
		expect(routing.shouldCheckpoint({ toolTurnCount: 2 })).toBe(true);
		// simulate two checkpoints consumed via verifyTurn kind=checkpoint below;
		// shouldCheckpoint itself only looks at checkpointsSoFar which increments in verifyTurn.
		// Before any checkpoint verify, turn 4 is still eligible:
		expect(routing.shouldCheckpoint({ toolTurnCount: 4 })).toBe(true);
	});

	it("runs checkpoint on glm and escalates maker to luna on first reject", async () => {
		const { routing, calls, replies, runModel } = setup({ planAfterRejections: 99 });
		replies["z-ai/glm-5.2"] = ["Progress looks fine. VERIFIED"];
		const verified = await routing.verifyTurn({
			kind: "checkpoint",
			toolTurnCount: 2,
			message: assistant("edited files"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "edited"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(verified?.status).toBe("verified");
		if (verified?.status === "verified") {
			expect(verified.model?.id).toBe("deepseek/deepseek-v4-flash-0731");
		}
		expect(calls).toEqual(["z-ai/glm-5.2"]);

		replies["z-ai/glm-5.2"] = ["Missing tests. CONFLICT"];
		const rejected = await routing.verifyTurn({
			kind: "checkpoint",
			toolTurnCount: 4,
			message: assistant("still hacking"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "no tests"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(rejected?.status).toBe("rejected");
		if (rejected?.status === "rejected") {
			expect(rejected.model?.id).toBe("openai/gpt-5.6-luna");
			expect(rejected.thinkingLevel).toBe("max");
			expect(rejected.correctivePrompt).toContain("MAKER/CHECKER/PLANNER");
			expect(rejected.correctivePrompt).not.toContain("Feedback: .");
		}
	});

	it("does not reroute a VERIFIED-first final answer whose rationale mentions the rejection token (F-05)", async () => {
		const { routing, calls, replies, runModel, auditLogPath } = setup({ planFirst: false });
		// Verbatim shape of the false rejection recorded on 2026-08-13.
		replies["moonshotai/kimi-k3"] = [
			"VERIFIED\n\nThe receipts substantiate the final answer's checkable claims:\n" +
				"- the report exists and tests ran\n\nNo grounds for CONFLICT were found.",
		];
		const verified = await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "npm test ok"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);

		// Accepted, and the run ends on the base maker: no retry, no escalation.
		expect(verified?.status).toBe("verified");
		if (verified?.status === "verified") {
			expect(verified.model?.id).toBe("deepseek/deepseek-v4-flash-0731");
		}
		// Exactly one checker call: no re-ask, no corrective maker cycle.
		expect(calls).toEqual(["moonshotai/kimi-k3"]);

		const audit = readAudit(auditLogPath);
		const auditEvent = audit.find((e) => e.event === "audit");
		expect(auditEvent?.verdict).toBe("verified");
		expect(auditEvent?.verdictVia).toBe("bare-token-line");
		expect(audit.some((e) => e.event === "rejected")).toBe(false);
		expect(audit.find((e) => e.event === "run-summary")?.outcome).toBe("verified");
	});

	it("still rejects and escalates on a genuine leading CONFLICT verdict", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({ planFirst: false, planAfterRejections: 99 });
		replies["moonshotai/kimi-k3"] = ["VERDICT: CONFLICT\n\nNo tests ran; the claim is unsupported."];
		const rejected = await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "ls only"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);

		expect(rejected?.status).toBe("rejected");
		if (rejected?.status === "rejected") {
			expect(rejected.model?.id).toBe("openai/gpt-5.6-luna");
			expect(rejected.correctivePrompt).toContain("No tests ran");
		}
		const audit = readAudit(auditLogPath);
		expect(audit.find((e) => e.event === "audit")?.verdictVia).toBe("verdict-line");
		expect(audit.some((e) => e.event === "rejected")).toBe(true);
	});

	it("instructs checkers to emit an isolated verdict line", async () => {
		const { routing, replies, runModel } = setup({ planFirst: false });
		const seenPrompts: string[] = [];
		const recordingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
			seenPrompts.push(msgs.map((m) => messageText(m as any)).join("\n"));
			return runModel(model, msgs);
		};
		replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
		await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "npm test ok"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel: recordingRunModel,
		} as unknown as VerifyTurnContext);
		expect(seenPrompts.join("\n")).toContain(CHECKER_VERDICT_LINE_INSTRUCTION);
	});

	it("uses kimi for final audits", async () => {
		const { routing, calls, replies, runModel } = setup({ planFirst: false });
		replies["moonshotai/kimi-k3"] = ["All good. VERIFIED"];
		const verified = await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "npm test ok"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(verified?.status).toBe("verified");
		expect(calls).toEqual(["moonshotai/kimi-k3"]);
	});

	it("skips final audit for trivial tier when auditOnlyAfterTools and no tool results", async () => {
		const { routing, calls, replies, runModel, auditLogPath } = setup({ triage: { enabled: true } });
		replies["deepseek/deepseek-v4-flash-0731"] = ["TRIVIAL"];
		await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "Hi Pi", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		calls.length = 0;
		const result = await routing.verifyTurn({
			kind: "final",
			message: assistant("hello"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "Hi Pi", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(result).toBeUndefined();
		expect(calls).toEqual([]);
		expect(readAudit(auditLogPath).some((e) => e.reason === "no-tools")).toBe(true);
	});

	it("standard tier audits a no-tool final answer even with auditOnlyAfterTools", async () => {
		const { routing, calls, replies, runModel } = setup();
		replies["moonshotai/kimi-k3"] = ["All good. VERIFIED"];
		const verified = await routing.verifyTurn({
			kind: "final",
			message: assistant("here is the explanation you asked for"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "explain why this design is safe", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(verified?.status).toBe("verified");
		expect(calls).toEqual(["moonshotai/kimi-k3"]);
	});

	it("on final reject: escalates maker to luna, and on 2nd reject also plans with opus", async () => {
		const { routing, calls, replies, runModel } = setup({
			planAfterRejections: 2,
			maxRejections: 2,
			planFirst: false,
			escalateAfterRejections: 1,
		});
		replies["moonshotai/kimi-k3"] = ["No tests. CONFLICT", "Still no tests. CONFLICT"];
		replies["anthropic/claude-opus-5"] = ["1) write tests\n2) run npm test"];

		const base = {
			kind: "final" as const,
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "ls only"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		};

		const first = await routing.verifyTurn(base as unknown as VerifyTurnContext);
		expect(first?.status).toBe("rejected");
		if (first?.status === "rejected") {
			expect(first.model?.id).toBe("openai/gpt-5.6-luna");
		}
		expect(calls).toEqual(["moonshotai/kimi-k3"]);

		const second = await routing.verifyTurn(base as unknown as VerifyTurnContext);
		expect(second?.status).toBe("rejected");
		if (second?.status === "rejected") {
			expect(second.correctivePrompt).toContain("The planner produced this plan");
			expect(second.model?.id).toBe("openai/gpt-5.6-luna");
		}
		expect(calls).toEqual(["moonshotai/kimi-k3", "moonshotai/kimi-k3", "anthropic/claude-opus-5"]);
	});

	it("never rejects on empty/ambiguous checker output (no Feedback: . storm)", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({ planFirst: false });
		replies["moonshotai/kimi-k3"] = [""]; // empty
		const result = await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "ok"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		// F-03: ambiguous verdict now surfaces an explicit UNVERIFIED notice
		// instead of returning undefined (silent fail-open). The user must see
		// that the checker never signed off. P1-3: distinct unverified status.
		expect(result?.status).toBe("unverified");
		if (result?.status === "unverified") {
			expect(result.notice).toMatch(/UNVERIFIED.*ambiguous or unavailable/);
		}
		expect(readAudit(auditLogPath).some((e) => e.reason === "ambiguous-verdict")).toBe(true);
		expect(readAudit(auditLogPath).some((e) => e.event === "run-summary" && e.outcome === "unverified")).toBe(true);
	});

	it("logs budget-exhausted and accepts final answer after max conflicts", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({
			planFirst: false,
			planAfterRejections: 99,
			maxRejections: 1,
		});
		replies["moonshotai/kimi-k3"] = ["bad CONFLICT", "still bad CONFLICT"];
		const base = {
			kind: "final" as const,
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "ok"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		};
		// With maxRejections: 1, the first conflict is a rejection (1 > 1 = false),
		// the second conflict exhausts (2 > 1 = true).
		// P1-3: budget exhaustion returns unverified status, not verified.
		expect((await routing.verifyTurn(base as unknown as VerifyTurnContext))?.status).toBe("rejected");
		const accepted = await routing.verifyTurn(base as unknown as VerifyTurnContext);
		expect(accepted?.status).toBe("unverified");
		if (accepted?.status === "unverified") {
			expect(accepted.model?.id).toBe("deepseek/deepseek-v4-flash-0731");
			expect(accepted.notice).toContain("[VERIFY] UNVERIFIED");
			expect(accepted.notice).toContain("still bad");
		}
		const exhausted = readAudit(auditLogPath).find((e) => e.event === "budget-exhausted");
		expect(exhausted?.accepted).toBe("unverified");
	});

	it("triage HARD routes the run to the strong planner and the stronger maker", async () => {
		const { routing, calls, replies, runModel, auditLogPath } = setup({ triage: { enabled: true } });
		replies["deepseek/deepseek-v4-flash-0731"] = ["HARD"];
		const prep = await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{
					role: "user",
					content: "Rework session resume to survive provider failover races.",
					timestamp: Date.now(),
				},
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		// Triage on the cheap maker, plan on the (strong) planner, make on the escalation-class maker.
		expect(calls).toEqual(["deepseek/deepseek-v4-flash-0731", "anthropic/claude-opus-5"]);
		expect(prep?.model?.id).toBe("openai/gpt-5.6-luna");
		expect(prep?.thinkingLevel).toBe("high");
		expect(readAudit(auditLogPath).some((e) => e.event === "triage" && e.tier === "hard")).toBe(true);
	});

	it("triage TRIVIAL skips the planner and checkpoints, and finals on the cheap checker", async () => {
		const { routing, calls, replies, runModel } = setup({ triage: { enabled: true } });
		replies["deepseek/deepseek-v4-flash-0731"] = ["TRIVIAL"];
		const prep = await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{
					role: "user",
					content: "Bump the copyright year in the site footer to 2026 please.",
					timestamp: Date.now(),
				},
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		expect(prep?.messages).toBeUndefined();
		expect(prep?.model?.id).toBe("deepseek/deepseek-v4-flash-0731");
		expect(prep?.thinkingLevel).toBe("medium");
		expect(calls).toEqual(["deepseek/deepseek-v4-flash-0731"]); // triage only, no planner
		expect(routing.shouldCheckpoint({ toolTurnCount: 6 })).toBe(false);

		replies["z-ai/glm-5.2"] = ["VERIFIED"];
		const verified = await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{
					role: "user",
					content: "Bump the copyright year in the site footer to 2026 please.",
					timestamp: Date.now(),
				},
				toolResult("bash", "edited footer"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(verified?.status).toBe("verified");
		expect(calls).toEqual(["deepseek/deepseek-v4-flash-0731", "z-ai/glm-5.2"]); // glm, not kimi
	});

	it("unparseable or failing triage falls back to STANDARD", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({ triage: { enabled: true } });
		replies["deepseek/deepseek-v4-flash-0731"] = ["dunno, somewhere in the middle"];
		await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Add input validation to the settings loader with tests.", timestamp: Date.now() },
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		expect(readAudit(auditLogPath).some((e) => e.event === "triage" && e.tier === "standard")).toBe(true);
	});

	it("planner CLARIFY makes the maker relay questions and stop instead of guessing", async () => {
		const { routing, replies, runModel, auditLogPath } = setup();
		replies["anthropic/claude-opus-5"] = [
			"CLARIFY:\n1. Which auth method should sessions use?\n2. Is downtime acceptable?",
		];
		const prep = await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{
					role: "user",
					content: "Migrate authentication to the new provider sometime soon.",
					timestamp: Date.now(),
				},
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		const injected = prep?.messages?.[0] && messageText(prep.messages[0] as any);
		expect(injected).toContain("FIRST investigate");
		expect(injected).toContain("never ask the user anything you can discover");
		expect(injected).toContain("at most 5 questions");
		expect(injected).toContain("Which auth method");
		expect(injected).not.toContain("[PLANNER] Execute this plan");
		expect(readAudit(auditLogPath).some((e) => e.event === "clarify")).toBe(true);
	});

	it("caps a runaway CLARIFY question dump and biases the planner prompt toward planning", async () => {
		const { routing, replies, runModel } = setup();
		const wall = `CLARIFY:\n${Array.from({ length: 40 }, (_, i) => `${i + 1}. Question ${i + 1}?`).join("\n")}${"x".repeat(5_000)}`;
		replies["anthropic/claude-opus-5"] = [wall];
		const seenPrompts: string[] = [];
		const recordingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
			seenPrompts.push(msgs.map((m) => messageText(m as any)).join("\n"));
			return runModel(model, msgs);
		};
		const prep = await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{
					role: "user",
					content: "Audit this project end to end and improve whatever needs improving.",
					timestamp: Date.now(),
				},
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel: recordingRunModel,
		} as unknown as BeforeFirstTurnContext);
		const injected = (prep?.messages?.[0] && messageText(prep.messages[0] as any)) ?? "";
		// Relayed questions are hard-capped even when the planner misbehaves.
		// A character budget is not the contract — the maker must see at most 5
		// numbered questions, not a 2000-character slice of a 40-question dump.
		const numbered = (injected.match(/^\s*\d+[.)]\s+/gm) ?? []).length;
		expect(numbered).toBeLessThanOrEqual(5);
		expect(injected.length).toBeLessThan(2_600);
		// The planner instruction itself demands plan-by-default and forbids discoverable questions.
		const plannerPrompt = seenPrompts.join("\n");
		expect(plannerPrompt).toContain("PLAN BY DEFAULT");
		expect(plannerPrompt).toContain("Never ask the user anything discoverable");
		expect(plannerPrompt).toContain("never CLARIFY for them");
		expect(plannerPrompt).toContain("AT MOST 5 numbered questions");
	});

	it("keeps the escalated maker after a verified checkpoint (sticky), demotes on final", async () => {
		const { routing, replies, runModel } = setup({ planAfterRejections: 99 });
		const base = (msgs: AgentMessage[]) =>
			({
				kind: "checkpoint",
				toolTurnCount: 2,
				message: assistant("progress"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: msgs,
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			}) as unknown as VerifyTurnContext;
		const msgs: AgentMessage[] = [
			{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
			toolResult("bash", "wip"),
		];

		replies["z-ai/glm-5.2"] = ["No tests yet, drifting. CONFLICT"];
		const rejected = await routing.verifyTurn(base(msgs));
		expect(rejected?.status).toBe("rejected");
		if (rejected?.status === "rejected") expect(rejected.model?.id).toBe("openai/gpt-5.6-luna");

		replies["z-ai/glm-5.2"] = ["Back on track. VERIFIED"];
		const verifiedCheckpoint = await routing.verifyTurn(base(msgs));
		expect(verifiedCheckpoint?.status).toBe("verified");
		if (verifiedCheckpoint?.status === "verified") {
			expect(verifiedCheckpoint.model?.id).toBe("openai/gpt-5.6-luna"); // sticky, no ping-pong
		}

		replies["moonshotai/kimi-k3"] = ["All good. VERIFIED"];
		const verifiedFinal = await routing.verifyTurn({
			...(base(msgs) as object),
			kind: "final",
		} as unknown as VerifyTurnContext);
		expect(verifiedFinal?.status).toBe("verified");
		if (verifiedFinal?.status === "verified") {
			expect(verifiedFinal.model?.id).toBe("deepseek/deepseek-v4-flash-0731"); // run over, demote
		}
	});

	it("checker audits carry the excellence charter and log cost fields; final emits run-summary", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({ planFirst: false });
		const seenPrompts: string[] = [];
		const recordingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
			seenPrompts.push(msgs.map((m) => messageText(m as any)).join("\n"));
			return runModel(model, msgs);
		};
		replies["moonshotai/kimi-k3"] = ["All good. VERIFIED"];
		const verified = await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "npm test ok"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel: recordingRunModel,
		} as unknown as VerifyTurnContext);
		expect(verified?.status).toBe("verified");
		expect(seenPrompts.join("\n")).toContain(EXCELLENCE_CHARTER.slice(0, 60));

		const audit = readAudit(auditLogPath);
		const auditEvent = audit.find((e) => e.event === "audit");
		expect(auditEvent?.tokensIn).toBeDefined();
		expect(auditEvent?.tokensOut).toBeDefined();
		expect(auditEvent?.costUsd).toBeDefined();
		const summary = audit.find((e) => e.event === "run-summary");
		expect(summary?.outcome).toBe("verified");
		expect(summary?.verifyCalls).toBeGreaterThan(0);
	});

	it("re-asks once on an ambiguous verdict before failing open", async () => {
		const { routing, calls, replies, runModel } = setup({ planFirst: false });
		replies["moonshotai/kimi-k3"] = ["hmm, mixed feelings", "Fine on second look. VERIFIED"];
		const verified = await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "npm test ok"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(verified?.status).toBe("verified");
		expect(calls).toEqual(["moonshotai/kimi-k3", "moonshotai/kimi-k3"]);
	});

	it("falls back to plannerFallbackModel when the planner is rate-limited upstream", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({
			plannerModel: "openrouter/openai/gpt-5.6-luna",
			plannerFallbackModel: "openrouter/z-ai/glm-5.2",
		});
		replies["z-ai/glm-5.2"] = ["1. inspect\n2. implement\n3. verify"];
		const errored = assistant("", "openai/gpt-5.6-luna");
		(errored as { stopReason: string }).stopReason = "error";
		(errored as { errorMessage?: string }).errorMessage = "temporarily rate-limited upstream";
		const failingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
			if (model.id === "openai/gpt-5.6-luna") return errored;
			return runModel(model, msgs);
		};
		const prep = await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{
					role: "user",
					content: "Implement a robust add() helper with tests and lint checks.",
					timestamp: Date.now(),
				},
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel: failingRunModel,
		} as unknown as BeforeFirstTurnContext);
		expect(prep?.messages?.[0] && messageText(prep.messages[0] as any)).toContain("1. inspect");
		const fallback = readAudit(auditLogPath).find((e) => e.event === "fallback");
		expect(fallback?.stage).toBe("planner");
		expect(fallback?.to).toBe("z-ai/glm-5.2");
	});

	it("falls back to checkerFallbackModel when the checker errors, and never re-asks an errored checker", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({
			planFirst: false,
			checkerFallbackModel: "openrouter/moonshotai/kimi-k3",
			checkerModel: "openrouter/z-ai/glm-5.2",
		});
		replies["moonshotai/kimi-k3"] = ["All good. VERIFIED"];
		const failingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
			if (model.id === "z-ai/glm-5.2") {
				const errored = assistant("", "z-ai/glm-5.2");
				(errored as { stopReason: string }).stopReason = "error";
				(errored as { errorMessage?: string }).errorMessage = "rate-limited";
				return errored;
			}
			return runModel(model, msgs);
		};
		const verified = await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "npm test ok"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel: failingRunModel,
		} as unknown as VerifyTurnContext);
		expect(verified?.status).toBe("verified");
		const audit = readAudit(auditLogPath);
		expect(audit.find((e) => e.event === "fallback")?.to).toBe("moonshotai/kimi-k3");
		expect(audit.find((e) => e.event === "audit")?.checker).toBe("moonshotai/kimi-k3");
	});

	it("skips the audit gracefully when checker and fallback are both unavailable", async () => {
		const { routing, auditLogPath } = setup({ planFirst: false, checkerFallbackModel: undefined });
		const alwaysErr = async (model: Model<any>) => {
			const errored = assistant("", model.id);
			(errored as { stopReason: string }).stopReason = "error";
			(errored as { errorMessage?: string }).errorMessage = "rate-limited";
			return errored;
		};
		const result = await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "ok"),
			],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel: alwaysErr,
		} as unknown as VerifyTurnContext);
		// F-03: checker + fallback both unavailable now surfaces an explicit
		// UNVERIFIED notice instead of returning undefined (silent fail-open).
		// P1-3: distinct unverified status, not verified-with-notice.
		expect(result?.status).toBe("unverified");
		if (result?.status === "unverified") {
			expect(result.notice).toMatch(/UNVERIFIED.*ambiguous or unavailable/);
		}
		expect(readAudit(auditLogPath).some((e) => e.reason === "ambiguous-verdict")).toBe(true);
		expect(readAudit(auditLogPath).some((e) => e.event === "run-summary" && e.outcome === "unverified")).toBe(true);
	});

	it("onTurnError swaps a rate-limited maker to the fallback, capped per run", async () => {
		const { routing, auditLogPath } = setup({ makerFallbackModel: "openrouter/moonshotai/kimi-k3" });
		const errored = assistant("", "openai/gpt-5.6-luna");
		(errored as { stopReason: string }).stopReason = "error";
		(errored as { errorMessage?: string }).errorMessage = "temporarily rate-limited upstream";
		const ctx = {
			message: errored,
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [],
		} as unknown as Parameters<typeof routing.onTurnError>[0];

		// Same-model guard: if the fallback itself is what failed, give up.
		const failedFallback = assistant("", "moonshotai/kimi-k3");
		(failedFallback as { stopReason: string }).stopReason = "error";
		expect(
			await routing.onTurnError({ ...(ctx as object), message: failedFallback } as Parameters<
				typeof routing.onTurnError
			>[0]),
		).toBeUndefined();

		const first = await routing.onTurnError(ctx);
		expect(first?.model?.id).toBe("moonshotai/kimi-k3");
		const second = await routing.onTurnError(ctx);
		expect(second?.model?.id).toBe("moonshotai/kimi-k3");
		// Capped: a third failure in the same run ends it instead of looping.
		expect(await routing.onTurnError(ctx)).toBeUndefined();
		expect(readAudit(auditLogPath).filter((e) => e.event === "maker-fallback").length).toBe(2);
	});

	it("uses the spare checker when a maker-fallback collides with the checker family", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({
			planFirst: false,
			checkerModel: "openrouter/openai/gpt-5.6-luna",
			checkerFallbackModel: "openrouter/meta/muse-spark-1.2",
			checkerFallbackThinkingLevel: "xhigh",
			makerFallbackModel: "openrouter/openai/gpt-5.6-luna",
			checkpointCheckerModel: "openrouter/openai/gpt-5.6-luna",
		});
		replies["meta/muse-spark-1.2"] = ["All good. VERIFIED"];
		const errored = assistant("", "deepseek/deepseek-v4-flash-0731");
		(errored as { stopReason: string }).stopReason = "error";
		(errored as { errorMessage?: string }).errorMessage = "temporarily rate-limited upstream";
		await routing.onTurnError({
			message: errored,
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [],
		} as unknown as Parameters<typeof routing.onTurnError>[0]);

		const thinkingByModel: Record<string, string | undefined> = {};
		const recordingRunModel = async (model: Model<any>, msgs: AgentMessage[], opts?: { thinkingLevel?: string }) => {
			thinkingByModel[model.id] = opts?.thinkingLevel;
			return runModel(model, msgs);
		};
		const verified = await routing.verifyTurn({
			kind: "final",
			message: assistant("done", "openai/gpt-5.6-luna"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [
				{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
				toolResult("bash", "ok"),
			],
			config: { model: modelWithId("openai/gpt-5.6-luna"), convertToLlm: () => [] },
			runModel: recordingRunModel,
		} as unknown as VerifyTurnContext);
		expect(verified?.status).toBe("verified");
		const audit = readAudit(auditLogPath);
		expect(audit.some((e) => e.event === "checker-swap" && String(e.to).includes("muse-spark"))).toBe(true);
		expect(thinkingByModel["meta/muse-spark-1.2"]).toBe("xhigh");
		expect(thinkingByModel["openai/gpt-5.6-luna"]).toBeUndefined();
	});

	it("installs when the session maker matches the checker if an independent spare exists", () => {
		const dir = mkdtempSync(join(tmpdir(), "pi-verify-"));
		dirs.push(dir);
		const routing = createVerifyRouting({
			verify: {
				checkerModel: "openrouter/openai/gpt-5.6-luna",
				checkerFallbackModel: "openrouter/meta/muse-spark-1.2",
			},
			makerModel: modelWithId("openai/gpt-5.6-luna"),
			makerThinkingLevel: "max",
			cwd: dir,
			auditLogPath: join(dir, "a.jsonl"),
			resolveModel: (ref) => (ref ? modelWithId(ref.replace(/^openrouter\//, "")) : undefined),
		});
		expect(routing).toBeTruthy();
	});

	it("returns undefined when checker model cannot be resolved", () => {
		const dir = mkdtempSync(join(tmpdir(), "pi-verify-"));
		dirs.push(dir);
		const routing = createVerifyRouting({
			verify: { checkerModel: "openrouter/missing/model" },
			makerModel: modelWithId("deepseek/deepseek-v4-flash-0731"),
			makerThinkingLevel: "max",
			cwd: dir,
			auditLogPath: join(dir, "a.jsonl"),
			resolveModel: () => undefined,
		});
		expect(routing).toBeUndefined();
	});

	it("F-01: stops with STOPPED_UNVERIFIED when maxMakerTurns is reached", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({
			planFirst: false,
			maxMakerTurns: 2,
			maxRejections: 99, // don't let rejection budget fire first
		});
		replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
		replies["z-ai/glm-5.2"] = ["VERDICT: VERIFIED"];
		const user = { role: "user" as const, content: "do work", timestamp: Date.now() };
		const a1 = assistant("turn 1");
		const a2 = assistant("turn 2");
		const t1 = toolResult("bash", "ok");
		const first = await routing.verifyTurn({
			kind: "checkpoint",
			toolTurnCount: 1,
			message: a1,
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [user, a1, t1],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(first?.status).toBe("verified");
		if (first?.status === "verified") {
			expect(first.notice).toBeUndefined();
		}
		const second = await routing.verifyTurn({
			kind: "final",
			message: a2,
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [user, a1, t1, a2],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(second?.status).toBe("unverified");
		if (second?.status === "unverified") {
			expect(second.notice).toMatch(/STOPPED_UNVERIFIED.*max-maker-turns/);
		}
		expect(
			readAudit(auditLogPath).some((e) => e.event === "run-stopped" && e.reason?.includes("max-maker-turns")),
		).toBe(true);
		expect(readAudit(auditLogPath).some((e) => e.event === "run-summary" && e.outcome === "stopped")).toBe(true);
	});

	it("does not double-count cumulative tool results across checkpoints", async () => {
		const { routing, replies, runModel } = setup({
			planFirst: false,
			maxToolCallsPerRun: 10,
			maxRejections: 99,
		});
		replies["z-ai/glm-5.2"] = ["VERDICT: VERIFIED", "VERDICT: VERIFIED", "VERDICT: VERIFIED", "VERDICT: VERIFIED"];
		const user = { role: "user" as const, content: "do work", timestamp: Date.now() };
		const tools = [toolResult("a", "1"), toolResult("b", "2"), toolResult("c", "3"), toolResult("d", "4")];
		const msgs: AgentMessage[] = [user];
		for (let i = 0; i < 4; i++) {
			const turn = assistant(`t${i}`);
			msgs.push(turn, tools[i]);
			const result = await routing.verifyTurn({
				kind: "checkpoint",
				toolTurnCount: i + 1,
				message: turn,
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [...msgs],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			expect(result?.status).toBe("verified");
		}
	});

	it("sums maker cost from the whole trace, not only the last checkpoint message", async () => {
		const { routing, replies, runModel } = setup({
			planFirst: false,
			maxRunCostUsd: 2.5,
			maxRejections: 99,
		});
		replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
		const paid = (text: string, cost: number) =>
			assistant(text, "mock", {
				usage: {
					input: 10,
					output: 10,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 20,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: cost },
				},
			});
		const user = { role: "user" as const, content: "do work", timestamp: Date.now() };
		const a1 = paid("one", 1);
		const a2 = paid("two", 1);
		const a3 = paid("three", 1);
		const result = await routing.verifyTurn({
			kind: "final",
			message: a3,
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [user, a1, a2, a3],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(result?.status).toBe("unverified");
		if (result?.status === "unverified") {
			expect(result.notice).toMatch(/STOPPED_UNVERIFIED.*max-run-cost/);
		}
	});

	it("shouldStopAfterTurn stops between checkpoints when the tool budget is spent", async () => {
		const { routing } = setup({ planFirst: false, maxToolCallsPerRun: 2 });
		const user = { role: "user" as const, content: "do work", timestamp: Date.now() };
		const a1 = assistant("working");
		const newMessages: AgentMessage[] = [user, a1, toolResult("a", "1"), toolResult("b", "2")];
		const context = { systemPrompt: "", messages: [...newMessages], tools: [] };
		const stop = routing.shouldStopAfterTurn({
			message: a1,
			toolResults: [],
			context,
			newMessages,
		});
		expect(stop).toBe(true);
		expect(newMessages.some((m) => m.role === "user" && messageText(m).includes("STOPPED_UNVERIFIED"))).toBe(true);
	});

	it("F-07: every audit event carries a runId and schema=2 envelope", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({ planFirst: false });
		replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
		await routing.verifyTurn({
			kind: "final",
			message: assistant("done"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "do work", timestamp: Date.now() }, toolResult("bash", "ok")],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		const events = readAudit(auditLogPath);
		// run-start, audit, demoted, run-summary — all should carry runId + schema.
		for (const e of events) {
			expect(typeof e.runId).toBe("string");
			expect(e.runId.length).toBeGreaterThan(0);
			expect(e.schema).toBe(2);
		}
		// All events in one run share the same runId.
		const ids = new Set(events.map((e) => e.runId));
		expect(ids.size).toBe(1);
	});

	it("P0-2: yes after a finished standard run stays non-trivial", async () => {
		const { routing, replies, runModel, auditLogPath } = setup({
			planFirst: false,
			triage: { enabled: true },
		});
		// First prompt: a real task that gets triaged as standard.
		replies["deepseek/deepseek-v4-flash-0731"] = ["STANDARD"];
		const realCtx = {
			context: {
				systemPrompt: "",
				messages: [],
				tools: [],
			},
			newMessages: [{ role: "user", content: "Build add() with tests.", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext;
		await routing.beforeFirstTurn(realCtx);
		const realAudit = readAudit(auditLogPath);
		expect(realAudit.some((e) => e.event === "triage" && e.tier === "standard")).toBe(true);

		// The real CLARIFY path finishes a no-tool standard turn and emits a
		// run-summary. A test that never calls verifyTurn cannot catch the
		// goalActive wipe inside emitRunSummary.
		replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
		await routing.verifyTurn({
			kind: "final",
			message: assistant("I asked the user two questions and stopped."),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "Build add() with tests.", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as VerifyTurnContext);
		expect(readAudit(auditLogPath).some((e) => e.event === "run-summary")).toBe(true);

		replies["deepseek/deepseek-v4-flash-0731"] = ["STANDARD"];
		const yesCtx = {
			context: {
				systemPrompt: "",
				messages: [],
				tools: [],
			},
			newMessages: [{ role: "user", content: "yes", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext;
		await routing.beforeFirstTurn(yesCtx);
		const yesTriage = readAudit(auditLogPath).filter((e) => e.event === "triage");
		const last = yesTriage[yesTriage.length - 1];
		expect(last?.via).not.toBe("heuristic");
		expect(last?.tier).not.toBe("trivial");
	});
});

describe("boundContextForModel", () => {
	const smallModel = { ...modelWithId("test/small"), contextWindow: 10_000 };
	const largeModel = { ...modelWithId("test/large"), contextWindow: 1_000_000 };

	it("returns messages unchanged when they fit the model context window", () => {
		const msgs: AgentMessage[] = [
			{ role: "user", content: "Build add() with tests.", timestamp: Date.now() },
			{ role: "assistant", content: [{ type: "text", text: "done" }], timestamp: Date.now() } as AgentMessage,
		];
		expect(boundContextForModel(msgs, largeModel)).toBe(msgs);
	});

	it("truncates from the front when over budget, keeping the goal + most recent", () => {
		const goal: AgentMessage = { role: "user", content: "Build the app.", timestamp: Date.now() };
		const msgs: AgentMessage[] = [goal];
		for (let i = 0; i < 200; i++) {
			msgs.push({
				role: i % 2 === 0 ? "assistant" : "user",
				content: [{ type: "text", text: "x".repeat(500) }],
				timestamp: Date.now(),
			} as AgentMessage);
		}
		const bounded = boundContextForModel(msgs, smallModel);
		expect(bounded[0]).toBe(goal);
		expect(bounded.length).toBeLessThan(20);
		expect(bounded.length).toBeGreaterThan(1);
		expect(bounded[bounded.length - 1]).toBe(msgs[msgs.length - 1]);
	});

	it("returns just the goal when even that barely fits", () => {
		const tinyModel = { ...modelWithId("test/tiny"), contextWindow: 8200 };
		const goal: AgentMessage = { role: "user", content: "Build the app.", timestamp: Date.now() };
		const msgs: AgentMessage[] = [
			goal,
			{
				role: "assistant",
				content: [{ type: "text", text: "x".repeat(100_000) }],
				timestamp: Date.now(),
			} as AgentMessage,
		];
		const bounded = boundContextForModel(msgs, tinyModel);
		expect(bounded.length).toBe(1);
		expect(bounded[0]).toBe(goal);
	});

	it("returns a truncated goal when the window is smaller than the response reserve", () => {
		const tinyModel = { ...modelWithId("test/tiny"), contextWindow: 1_000 };
		const goal: AgentMessage = {
			role: "user",
			content: "Build the app with a full test suite.",
			timestamp: Date.now(),
		};
		const msgs: AgentMessage[] = [
			goal,
			{
				role: "assistant",
				content: [{ type: "text", text: "x".repeat(50_000) }],
				timestamp: Date.now(),
			} as AgentMessage,
		];
		const bounded = boundContextForModel(msgs, tinyModel);
		expect(bounded.length).toBe(1);
		expect(messageText(bounded[0] as AgentMessage)).toMatch(/TRUNCATED/i);
		expect(messageText(bounded[0] as AgentMessage)).not.toContain("x".repeat(100));
	});

	it("handles messages with no user goal (keeps only recent)", () => {
		const msgs: AgentMessage[] = [];
		for (let i = 0; i < 100; i++) {
			msgs.push({
				role: "assistant",
				content: [{ type: "text", text: "x".repeat(500) }],
				timestamp: Date.now(),
			} as AgentMessage);
		}
		const bounded = boundContextForModel(msgs, smallModel);
		expect(bounded.length).toBeLessThan(20);
		expect(bounded.length).toBeGreaterThan(0);
	});

	it("returns an empty array for an empty input", () => {
		expect(boundContextForModel([], smallModel)).toEqual([]);
	});
});

describe("redactEvidence", () => {
	it("redacts OpenRouter, OpenAI, GitHub, and xAI API keys", () => {
		const keys = [
			`sk-or-v1-${"a".repeat(64)}`,
			`sk-${"b".repeat(48)}`,
			`ghp_${"A".repeat(36)}`,
			`xai-${"c".repeat(48)}`,
		];
		for (const key of keys) {
			const redacted = redactEvidence(key);
			expect(redacted).not.toContain(key);
			expect(redacted).toMatch(/REDACTED/i);
		}
	});

	it("preserves git SHA-1 (40 hex) and SHA-256 (64 hex) commit hashes", () => {
		const sha1 = "a1b2c3d4e5f6789012345678901234567890abcd";
		const sha256 = "a".repeat(64);
		expect(sha1.length).toBe(40);
		expect(sha256.length).toBe(64);
		expect(redactEvidence(`commit ${sha1} updated`)).toContain(sha1);
		expect(redactEvidence(`commit ${sha256} updated`)).toContain(sha256);
	});

	it("redacts non-hex 40+ char tokens", () => {
		const token = "X".repeat(40);
		expect(redactEvidence(`value: ${token}`)).not.toContain(token);
	});

	it("captures the key name for key=value format (no literal $1)", () => {
		const redacted = redactEvidence(`api_key=${"A".repeat(40)}`);
		expect(redacted).not.toContain("$1");
		expect(redacted).not.toContain("A".repeat(32));
		expect(redacted).toMatch(/api_key=\[REDACTED\]/i);
	});

	it("redacts key:value format with a colon separator", () => {
		const redacted = redactEvidence(`api_key: ${"A".repeat(40)}`);
		expect(redacted).not.toContain("A".repeat(32));
		expect(redacted).toMatch(/api_key=\[REDACTED\]/i);
	});

	it("does not redact short values in key=value format", () => {
		expect(redactEvidence("api_key: enable")).toContain("enable");
		expect(redactEvidence("auth: bearer")).toContain("bearer");
	});

	it("redacts .env-style lines with sensitive key names", () => {
		const redacted = redactEvidence(`MY_API_KEY=sk_test_${"x".repeat(40)}`);
		expect(redacted).not.toContain("sk_test_");
		expect(redacted).toMatch(/MY_API_KEY=\[REDACTED\]/i);
	});

	it("redacts AWS access keys", () => {
		const aws = "AKIA" + "ABCDEFGHJKLMNPQRS";
		expect(redactEvidence(aws)).not.toContain(aws);
		expect(redactEvidence(aws)).toMatch(/REDACTED/i);
	});
});
