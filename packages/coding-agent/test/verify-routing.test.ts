import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage, BeforeFirstTurnContext, VerifyTurnContext } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it } from "vitest";
import type { VerifySettings } from "../src/core/settings-manager.ts";
import {
	checkerRefForKind,
	classifyCheckerVerdict,
	classifyTriageTier,
	createVerifyRouting,
	EXCELLENCE_CHARTER,
	isConversationalPrompt,
	messageText,
	shouldEscalateMaker,
	shouldPlanFirst,
	shouldRunCheckpoint,
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

function assistant(text: string, modelId = "mock"): AssistantMessage {
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

describe("isConversationalPrompt", () => {
	it("catches greetings and acknowledgments, not real tasks", () => {
		expect(isConversationalPrompt("hi")).toBe(true);
		expect(isConversationalPrompt("thanks, looks great!")).toBe(true);
		expect(isConversationalPrompt("ok")).toBe(true);
		expect(isConversationalPrompt("got it")).toBe(true);
		expect(isConversationalPrompt("fix the race condition in session resume")).toBe(false);
		expect(isConversationalPrompt("refactor the settings loader to be async")).toBe(false);
	});

	it("does not treat a short real task as conversational", () => {
		expect(isConversationalPrompt("fix the typo")).toBe(false);
		expect(isConversationalPrompt("bump the version")).toBe(false);
		expect(isConversationalPrompt("explain why X")).toBe(false);
		expect(isConversationalPrompt("is this safe?")).toBe(false);
		expect(isConversationalPrompt("review this")).toBe(false);
	});

	it("still treats short acknowledgments as conversational", () => {
		expect(isConversationalPrompt("ok")).toBe(true);
		expect(isConversationalPrompt("yes")).toBe(true);
		expect(isConversationalPrompt("done")).toBe(true);
		expect(isConversationalPrompt("sounds good")).toBe(true);
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

	it("skips planFirst for short conversational prompts (heuristic trivial tier)", async () => {
		const { routing, calls, runModel, auditLogPath } = setup();
		const prep = await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "hi", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		// No planner or triage call runs, but the trivial tier still routes the maker.
		expect(prep?.messages).toBeUndefined();
		expect(prep?.model?.id).toBe("deepseek/deepseek-v4-flash-0731");
		expect(prep?.thinkingLevel).toBe("medium");
		expect(calls).toEqual([]);
		const audit = readAudit(auditLogPath);
		expect(audit.some((e) => e.event === "triage" && e.tier === "trivial" && e.via === "heuristic")).toBe(true);
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
		const { routing, calls, runModel, auditLogPath } = setup();
		// Route the run into the trivial tier via the conversational heuristic.
		await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "ok", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		const result = await routing.verifyTurn({
			kind: "final",
			message: assistant("hello"),
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "hi", timestamp: Date.now() }],
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
		expect(result).toBeUndefined();
		expect(readAudit(auditLogPath).some((e) => e.reason === "ambiguous-verdict")).toBe(true);
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
		expect((await routing.verifyTurn(base as unknown as VerifyTurnContext))?.status).toBe("rejected");
		const accepted = await routing.verifyTurn(base as unknown as VerifyTurnContext);
		expect(accepted?.status).toBe("verified");
		if (accepted?.status === "verified") {
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
		expect(result).toBeUndefined();
		expect(readAudit(auditLogPath).some((e) => e.reason === "ambiguous-verdict")).toBe(true);
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
});
