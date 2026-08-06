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
	createVerifyRouting,
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

		const second = await routing.beforeFirstTurn(ctx);
		expect(second).toBeUndefined();
		expect(readAudit(auditLogPath).some((e) => e.event === "planned")).toBe(true);
	});

	it("skips planFirst for short conversational prompts", async () => {
		const { routing, calls, runModel, auditLogPath } = setup();
		const prep = await routing.beforeFirstTurn({
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content: "hi", timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext);
		expect(prep).toBeUndefined();
		expect(calls).toEqual([]);
		expect(readAudit(auditLogPath).some((e) => e.event === "plan-skipped")).toBe(true);
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

	it("skips final audit when auditOnlyAfterTools and no tool results", async () => {
		const { routing, calls, runModel, auditLogPath } = setup();
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
		}
		expect(readAudit(auditLogPath).some((e) => e.event === "budget-exhausted")).toBe(true);
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
