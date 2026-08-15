/**
 * Harness promotion eval gate.
 *
 * Inspired by OpenAI's agent-improvement loop: traces and review findings
 * become durable evals, and those evals are the validation gate before a
 * harness change can be treated as done.
 * https://developers.openai.com/cookbook/examples/agents_sdk/agent_improvement_loop
 *
 * Rules:
 *   - A case encodes a user-visible contract, not a comment in source.
 *   - Tests that skip the real path (for example, never calling verifyTurn)
 *     do not count as coverage.
 *   - `already holding` must stay green. `promotion blockers` stay red until
 *     the remaining defects are actually fixed. Do not mark blockers with
 *     it.fails / it.skip — that would greenwash the suite.
 */
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage, BeforeFirstTurnContext, VerifyTurnContext } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it } from "vitest";
import { InMemorySettingsStorage, SettingsManager, type VerifySettings } from "../src/core/settings-manager.ts";
import {
	boundContextForModel,
	collectTouchedPaths,
	createVerifyRouting,
	DIFF_EVIDENCE_HARD_CAP_CHARS,
	evidenceTruncationBlocksVerified,
	extractModelFamily,
	GOAL_DIFF_TRUNCATION_MARKER,
	gatherDiffEvidence,
	injectPlan,
	messageText,
	OTHER_DIFF_TRUNCATION_MARKER,
	redactEvidence,
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

function toolCall(name: string, args: Record<string, unknown>): AgentMessage {
	return {
		...assistant("", "mock"),
		content: [{ type: "toolCall", id: `call_${name}`, name, arguments: args }],
		stopReason: "toolUse",
	} as AgentMessage;
}

function modelFamily(ref: string): string {
	return extractModelFamily(ref);
}

function numberedQuestions(text: string): number {
	return (text.match(/^\s*\d+[.)]\s+/gm) ?? []).length;
}

function gitInit(dir: string): void {
	execSync("git init", { cwd: dir, stdio: "ignore" });
	execSync('git config user.email "eval@test"', { cwd: dir, stdio: "ignore" });
	execSync('git config user.name "eval"', { cwd: dir, stdio: "ignore" });
}

describe("harness eval gate", () => {
	const dirs: string[] = [];
	afterEach(() => {
		for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
	});

	function setup(
		settings: Partial<VerifySettings> = {},
		cwd?: string,
		sessionMakerId = "deepseek/deepseek-v4-flash-0731",
	) {
		const dir = cwd ?? mkdtempSync(join(tmpdir(), "pi-eval-"));
		if (!cwd) dirs.push(dir);
		const auditLogPath = join(dir, "verify-audit.jsonl");
		const models = new Map<string, Model<any>>([
			["openrouter/deepseek/deepseek-v4-flash-0731", modelWithId("deepseek/deepseek-v4-flash-0731")],
			["openrouter/openai/gpt-5.6-luna", modelWithId("openai/gpt-5.6-luna")],
			["openrouter/meta/muse-spark-1.2", modelWithId("meta/muse-spark-1.2")],
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
			"anthropic/claude-opus-5": ["1. inspect\n2. implement\n3. Verify by running npm test"],
			"z-ai/glm-5.2": [],
			"moonshotai/kimi-k3": [],
			"deepseek/deepseek-v4-flash-0731": [],
			"meta/muse-spark-1.2": [],
		};
		const routing = createVerifyRouting({
			verify: {
				checkerModel: "openrouter/moonshotai/kimi-k3",
				checkpointCheckerModel: "openrouter/z-ai/glm-5.2",
				plannerModel: "openrouter/anthropic/claude-opus-5",
				escalationMakerModel: "openrouter/openai/gpt-5.6-luna",
				checkerThinkingLevel: "max",
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
			makerModel: modelWithId(sessionMakerId),
			makerThinkingLevel: "high",
			cwd: dir,
			auditLogPath,
			resolveModel,
		});

		const runModel = async (model: Model<any>, _msgs: AgentMessage[]) => {
			calls.push(model.id);
			const queue = replies[model.id] ?? [];
			const text = queue.length > 0 ? (queue.shift() as string) : `answer from ${model.id}`;
			return assistant(text, model.id);
		};

		const readAudit = () =>
			existsSync(auditLogPath)
				? readFileSync(auditLogPath, "utf8")
						.trim()
						.split("\n")
						.filter(Boolean)
						.map((line) => JSON.parse(line) as Record<string, unknown>)
				: [];

		return { routing, calls, replies, auditLogPath, runModel, dir, readAudit, resolveModel };
	}

	function userTurn(content: string, runModel: BeforeFirstTurnContext["runModel"]): BeforeFirstTurnContext {
		return {
			context: { systemPrompt: "", messages: [], tools: [] },
			newMessages: [{ role: "user", content, timestamp: Date.now() }],
			config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
			runModel,
		} as unknown as BeforeFirstTurnContext;
	}

	describe("already holding — must not regress", () => {
		it("eval: empty project checkerModel disables routing after merge", () => {
			const storage = new InMemorySettingsStorage();
			storage.withLock("global", () =>
				JSON.stringify({
					verify: {
						checkerModel: "openrouter/openai/gpt-5.6-luna",
						plannerModel: "openrouter/openai/gpt-5.6-luna",
					},
				}),
			);
			storage.withLock("project", () => JSON.stringify({ verify: { checkerModel: "" } }));
			const mgr = SettingsManager.fromStorage(storage);
			expect(mgr.getVerifySettings()?.checkerModel).toBe("");
			expect(mgr.getVerifySettings()?.plannerModel).toBe("openrouter/openai/gpt-5.6-luna");
		});

		it("eval: createVerifyRouting refuses to install when checkerModel is blank", () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-"));
			dirs.push(dir);
			const routing = createVerifyRouting({
				verify: { checkerModel: "" },
				makerModel: modelWithId("deepseek/deepseek-v4-flash-0731"),
				makerThinkingLevel: "high",
				cwd: dir,
				auditLogPath: join(dir, "a.jsonl"),
				resolveModel: () => undefined,
			});
			expect(routing).toBeUndefined();
		});

		it("eval: injectPlan keeps a trailing verify step when the plan is too long", () => {
			const body = `${"do a thing\n".repeat(400)}7. Verify by running npm test and npm run check\n`;
			const injected = injectPlan(body);
			expect(injected).toContain("Verify by running npm test");
			expect(injected).toMatch(/truncated/i);
		});

		it("eval: injectPlan keeps verification when the verification section itself exceeds the budget", () => {
			const verification = `7. Verify by running ${"npm test && ".repeat(400)}npm run check\n`;
			const body = `1. Investigate the bug\n2. Patch the file\n${verification}`;
			const injected = injectPlan(body);
			expect(injected).toContain("Verify by running");
			expect(injected).toMatch(/truncated/i);
			expect(injected).not.toContain("1. Investigate the bug");
		});

		it("eval: gatherDiffEvidence includes untracked new files on a final audit", () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-git-"));
			dirs.push(dir);
			gitInit(dir);
			writeFileSync(join(dir, "tracked.txt"), "old\n");
			execSync("git add tracked.txt && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			writeFileSync(join(dir, "brand-new.ts"), "export const n = 1;\n");
			const evidence = gatherDiffEvidence(dir, "final");
			expect(evidence).toMatch(/brand-new\.ts/);
			expect(evidence).toMatch(/untracked/i);
			expect(evidence).toContain("export const n = 1");
		});

		it("eval: untracked evidence must not run the filename as a shell command", () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-"));
			dirs.push(dir);
			gitInit(dir);
			writeFileSync(join(dir, "tracked.txt"), "old\n");
			execSync("git add tracked.txt && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			const evil = 'x"; touch PWNED; echo "safe';
			writeFileSync(join(dir, evil), "secret-body\n");
			const evidence = gatherDiffEvidence(dir, "final");
			expect(existsSync(join(dir, "PWNED"))).toBe(false);
			expect(evidence).toContain("secret-body");
		});

		it("eval: live standard maker and live checker are different families", () => {
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			if (!existsSync(livePath)) throw new Error("live settings.json is the contract under test");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as {
				verify?: { checkerModel?: string; tiers?: { standard?: { makerModel?: string } } };
			};
			const maker = live.verify?.tiers?.standard?.makerModel;
			const checker = live.verify?.checkerModel;
			expect(maker).toBeTruthy();
			expect(checker).toBeTruthy();
			expect(modelFamily(maker!)).not.toBe(modelFamily(checker!));
		});

		it("eval: live enabled models are Flash and Luna; K3 stays off the picker", () => {
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			if (!existsSync(livePath)) throw new Error("live settings.json is the contract under test");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as { enabledModels?: string[] };
			const enabled = live.enabledModels ?? [];
			const normalized = enabled.map((id) => id.replace(/^openrouter\//, "")).sort();
			expect(normalized).toEqual(["deepseek/deepseek-v4-flash-0731", "openai/gpt-5.6-luna"].sort());
			expect(enabled.join(" ")).not.toMatch(
				/muse-spark|grok-4\.6|kimi-k3|claude-fable|gpt-5\.6-sol|deepseek-v4-pro|claude-opus/i,
			);
		});

		it("eval: live hard audits stay independent — same-family Luna/Luna promotes the K3 spare", () => {
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			if (!existsSync(livePath)) throw new Error("live settings.json is the contract under test");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as {
				verify?: {
					checkerModel?: string;
					checkerFallbackModel?: string;
					tiers?: {
						hard?: { makerModel?: string; checkerModel?: string; checkerFallbackModel?: string };
					};
				};
			};
			const hardMaker = live.verify?.tiers?.hard?.makerModel;
			const hardChecker = live.verify?.tiers?.hard?.checkerModel ?? live.verify?.checkerModel;
			const spare = live.verify?.tiers?.hard?.checkerFallbackModel ?? live.verify?.checkerFallbackModel;
			expect(hardMaker).toBeTruthy();
			expect(hardChecker).toBeTruthy();
			expect(spare).toBeTruthy();
			if (modelFamily(hardMaker!) === modelFamily(hardChecker!)) {
				expect(modelFamily(spare!)).not.toBe(modelFamily(hardMaker!));
			} else {
				expect(modelFamily(hardMaker!)).not.toBe(modelFamily(hardChecker!));
			}
		});

		it("eval: live hard planner must be a different family from the hard maker", () => {
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			if (!existsSync(livePath)) throw new Error("live settings.json is the contract under test");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as {
				verify?: {
					strongPlannerModel?: string;
					plannerModel?: string;
					tiers?: { hard?: { makerModel?: string; plannerModel?: string } };
				};
			};
			const hardMaker = live.verify?.tiers?.hard?.makerModel;
			const hardPlanner =
				live.verify?.tiers?.hard?.plannerModel ?? live.verify?.strongPlannerModel ?? live.verify?.plannerModel;
			expect(hardMaker).toBeTruthy();
			expect(hardPlanner).toBeTruthy();
			expect(modelFamily(hardMaker!)).not.toBe(modelFamily(hardPlanner!));
		});

		it("eval: live checker fallback is a required third family, not an optional same-family retry", () => {
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			if (!existsSync(livePath)) throw new Error("live settings.json is the contract under test");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as {
				verify?: {
					checkerModel?: string;
					checkerFallbackModel?: string;
					tiers?: {
						standard?: { makerModel?: string; checkerFallbackModel?: string };
						hard?: { makerModel?: string; checkerFallbackModel?: string };
					};
				};
			};
			const checker = live.verify?.checkerModel;
			const fallback = live.verify?.checkerFallbackModel;
			const standardMaker = live.verify?.tiers?.standard?.makerModel;
			const hardMaker = live.verify?.tiers?.hard?.makerModel;
			expect(checker).toBeTruthy();
			expect(fallback).toBeTruthy();
			expect(modelFamily(fallback!)).not.toBe(modelFamily(checker!));
			expect(modelFamily(fallback!)).not.toBe(modelFamily(standardMaker ?? ""));
			expect(modelFamily(fallback!)).not.toBe(modelFamily(hardMaker ?? ""));
		});

		it("eval: live bash sandbox is installed and denies auth.json", () => {
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			if (!existsSync(livePath)) throw new Error("live settings.json is the contract under test");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as { packages?: string[] };
			expect(live.packages?.some((p) => p.includes("sandbox"))).toBe(true);
			const sandboxDir = join(homedir(), ".pi", "agent", "extensions", "sandbox");
			expect(existsSync(join(sandboxDir, "index.ts"))).toBe(true);
			const cfgPath = join(homedir(), ".pi", "agent", "extensions", "sandbox.json");
			expect(existsSync(cfgPath)).toBe(true);
			const cfg = JSON.parse(readFileSync(cfgPath, "utf8")) as { filesystem?: { denyRead?: string[] } };
			expect(cfg.filesystem?.denyRead?.some((p) => p.includes("auth.json") || p.includes(".ssh"))).toBe(true);
		});

		it("eval: live gauntlet critic is Luna, the live checker — not a fourth model", () => {
			const src = readFileSync(join(homedir(), ".pi", "agent", "extensions", "gauntlet.ts"), "utf8");
			expect(src).not.toMatch(/muse-spark-1\.2:xhigh/);
			expect(src).toMatch(/criticModel \?\? "openrouter\/openai\/gpt-5\.6-luna"/);
			expect(src).not.toMatch(/criticModel \?\? "openrouter\/google\//);
			expect(src).not.toMatch(/criticModel \?\? "openrouter\/moonshotai\//);
			const cardPath = join(homedir(), "pi", ".pi", "gauntlet", "CARD.md");
			if (!existsSync(cardPath)) throw new Error("live gauntlet CARD.md is the contract under test");
			const cardMd = readFileSync(cardPath, "utf8");
			const json = /```json\s*\n([\s\S]*?)\n\s*```/.exec(cardMd)?.[1];
			if (!json) throw new Error("CARD.md has no json block");
			const card = JSON.parse(json) as { criticModel?: string };
			expect(card.criticModel).toBe("openrouter/openai/gpt-5.6-luna");
			expect(card.criticModel).not.toMatch(/kimi-k3|gemini|spark|grok|claude|sol/i);
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as {
				verify?: { checkerModel?: string };
			};
			expect(card.criticModel).toBe(live.verify?.checkerModel);
		});

		it("eval: live spare checker is Kimi K3, a third-family plug, not a daily enabledModel", () => {
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			if (!existsSync(livePath)) throw new Error("live settings.json is the contract under test");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as {
				enabledModels?: string[];
				verify?: {
					checkerFallbackModel?: string;
					escalationMakerModel?: string;
				};
			};
			const fallback = live.verify?.checkerFallbackModel;
			expect(fallback).toMatch(/moonshotai\/kimi-k3$/);
			expect(fallback).not.toMatch(/muse-spark|grok-4\.6/);
			expect(live.enabledModels?.some((id) => modelFamily(id) === modelFamily(fallback!))).toBe(false);
			expect(JSON.stringify(live.verify ?? {})).not.toMatch(/"makerModel": "openrouter\/moonshotai\/kimi-k3"/);
			expect(live.verify?.escalationMakerModel).not.toMatch(/kimi-k3/);
		});

		it("eval: live models.json pins rated sampling under the provider, not a discarded top-level key", () => {
			const modelsPath = join(homedir(), ".pi", "agent", "models.json");
			expect(existsSync(modelsPath)).toBe(true);
			const models = JSON.parse(readFileSync(modelsPath, "utf8")) as {
				modelOverrides?: unknown;
				providers?: {
					openrouter?: {
						modelOverrides?: Record<
							string,
							{
								maxTokens?: number;
								samplingParams?: Record<string, unknown>;
								thinkingLevelMap?: Record<string, string | null>;
							}
						>;
					};
				};
			};
			expect(models.modelOverrides).toBeUndefined();
			const overrides = models.providers?.openrouter?.modelOverrides;
			expect(overrides).toBeTruthy();

			const settingsPath = join(homedir(), ".pi", "agent", "settings.json");
			const live = JSON.parse(readFileSync(settingsPath, "utf8")) as {
				maxOutputTokens?: number;
				verify?: {
					checkerModel?: string;
					checkerFallbackModel?: string;
					tiers?: { standard?: { makerModel?: string } };
				};
			};
			const plugIds = [
				live.verify?.tiers?.standard?.makerModel,
				live.verify?.checkerModel,
				live.verify?.checkerFallbackModel,
				"openrouter/deepseek/deepseek-v4-flash-0731",
			]
				.filter((id): id is string => Boolean(id))
				.map((id) => id.replace(/^openrouter\//, ""));

			for (const id of plugIds) {
				const override = overrides?.[id];
				expect(override, `missing rated override for ${id}`).toBeTruthy();
				expect(override?.maxTokens ?? 0).toBeGreaterThanOrEqual(128000);
			}

			const flash = overrides?.["deepseek/deepseek-v4-flash-0731"];
			expect(flash?.samplingParams?.temperature).toBe(1);
			expect(flash?.samplingParams?.top_p).toBe(0.95);
			expect(flash?.thinkingLevelMap?.max).toBe("max");
			expect(flash?.thinkingLevelMap?.xhigh).toBe("high");

			const luna = overrides?.["openai/gpt-5.6-luna"];
			expect(luna?.samplingParams?.temperature).toBeUndefined();
			expect(luna?.maxTokens).toBe(128000);

			const kimi = overrides?.["moonshotai/kimi-k3"];
			expect(kimi?.samplingParams?.temperature).toBeUndefined();
			expect(kimi?.maxTokens).toBe(262144);
			expect(kimi?.thinkingLevelMap?.off).toBe("low");
			expect(kimi?.thinkingLevelMap?.max).toBe("max");

			const grok = overrides?.["x-ai/grok-4.6"];
			expect(grok?.samplingParams?.temperature).toBeUndefined();
			expect(grok?.maxTokens).toBe(128000);
			expect(grok?.thinkingLevelMap?.off).toBe("low");
			expect(grok?.thinkingLevelMap?.max).toBe("xhigh");
			expect(grok?.thinkingLevelMap?.xhigh).toBe("xhigh");

			expect(live.maxOutputTokens).toBe(128000);
		});

		it("eval: same-family maker and checker still install when an independent spare exists", () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-"));
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
	});

	describe("already holding — last-pass contracts must not regress", () => {
		it("eval: after a finished standard run, answering yes must not skip the loop", async () => {
			const { routing, replies, runModel, readAudit } = setup({
				planFirst: false,
				triage: { enabled: true },
			});
			if (!routing) throw new Error("routing required");
			replies["deepseek/deepseek-v4-flash-0731"] = ["STANDARD"];
			replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];

			await routing.beforeFirstTurn(userTurn("Build add() with tests.", runModel));
			await routing.verifyTurn({
				kind: "final",
				message: assistant("I asked the user two questions and stopped."),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [{ role: "user", content: "Build add() with tests.", timestamp: Date.now() }],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);

			expect(readAudit().some((e) => e.event === "run-summary")).toBe(true);

			replies["deepseek/deepseek-v4-flash-0731"] = ["STANDARD"];
			await routing.beforeFirstTurn(userTurn("yes", runModel));
			const yesTriage = readAudit().filter((e) => e.event === "triage");
			const last = yesTriage[yesTriage.length - 1];
			expect(last?.via).not.toBe("heuristic");
			expect(last?.tier).not.toBe("trivial");
		});

		it("eval: after a finished standard run, thanks is triaged as TRIVIAL", async () => {
			const { routing, replies, runModel, readAudit } = setup({
				planFirst: false,
				triage: { enabled: true },
			});
			if (!routing) throw new Error("routing required");
			replies["deepseek/deepseek-v4-flash-0731"] = ["STANDARD", "TRIVIAL"];
			replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
			await routing.beforeFirstTurn(userTurn("Build add() with tests.", runModel));
			await routing.verifyTurn({
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
			await routing.beforeFirstTurn(userTurn("thanks", runModel));
			const last = readAudit()
				.filter((e) => e.event === "triage")
				.at(-1);
			expect(last?.via).not.toBe("heuristic");
			expect(last?.tier).toBe("trivial");
		});

		it("eval: same-family maker and checker must not install as independent verification", () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-"));
			dirs.push(dir);
			const routing = createVerifyRouting({
				verify: { checkerModel: "openrouter/openai/gpt-5.6-luna" },
				makerModel: modelWithId("openai/gpt-5.6-luna"),
				makerThinkingLevel: "max",
				cwd: dir,
				auditLogPath: join(dir, "a.jsonl"),
				resolveModel: (ref) => (ref ? modelWithId(ref.replace(/^openrouter\//, "")) : undefined),
			});
			expect(routing).toBeUndefined();
		});

		it("eval: checker fallback must not be the hard maker's family", async () => {
			const { routing, replies, runModel, readAudit } = setup({
				planFirst: false,
				checkerModel: "openrouter/openai/gpt-5.6-luna",
				checkerFallbackModel: "openrouter/meta/muse-spark-1.2",
				tiers: {
					hard: {
						makerModel: "openrouter/meta/muse-spark-1.2",
						makerThinkingLevel: "xhigh",
					},
				},
				triage: { enabled: true },
			});
			if (!routing) throw new Error("routing required");
			replies["deepseek/deepseek-v4-flash-0731"] = ["HARD"];
			await routing.beforeFirstTurn(userTurn("Redesign auth across the whole repo.", runModel));

			const erroringRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
				if (model.id === "openai/gpt-5.6-luna") {
					return assistant("unavailable", model.id, {
						stopReason: "error",
						errorMessage: "429 rate limited",
					});
				}
				return runModel(model, msgs);
			};
			replies["meta/muse-spark-1.2"] = ["VERDICT: VERIFIED"];
			await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Redesign auth across the whole repo.", timestamp: Date.now() },
					toolResult("bash", "edited"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel: erroringRunModel,
			} as unknown as VerifyTurnContext);

			const fallbacks = readAudit().filter((e) => e.event === "fallback");
			for (const event of fallbacks) {
				expect(modelFamily(String(event.to))).not.toBe("meta");
			}
		});

		it("eval: live hard-tier checker fallback must exist and be a third family", () => {
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			if (!existsSync(livePath)) throw new Error("live settings.json is the contract under test");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as {
				verify: {
					checkerModel?: string;
					checkerFallbackModel?: string;
					tiers?: { hard?: { makerModel?: string; checkerFallbackModel?: string } };
				};
			};
			const hardMaker = live.verify.tiers?.hard?.makerModel;
			const checker = live.verify.checkerModel;
			const fallback = live.verify.tiers?.hard?.checkerFallbackModel ?? live.verify.checkerFallbackModel;
			expect(hardMaker).toBeTruthy();
			expect(fallback).toBeTruthy();
			expect(modelFamily(hardMaker!)).not.toBe(modelFamily(fallback!));
			expect(modelFamily(fallback!)).not.toBe(modelFamily(checker ?? ""));
		});

		it("eval: a 40-question CLARIFY injects at most 5 questions to the maker", async () => {
			const { routing, replies, runModel } = setup();
			if (!routing) throw new Error("routing required");
			const wall = `CLARIFY:\n${Array.from({ length: 40 }, (_, i) => `${i + 1}. Question ${i + 1}?`).join("\n")}`;
			replies["anthropic/claude-opus-5"] = [wall];
			const prep = await routing.beforeFirstTurn(
				userTurn("Migrate authentication to the new provider sometime soon.", runModel),
			);
			const injected = prep?.messages?.[0] ? messageText(prep.messages[0] as AgentMessage) : "";
			expect(numberedQuestions(injected)).toBeLessThanOrEqual(5);
		});

		it("eval: OpenRouter-style keys in tool output must not reach the checker", () => {
			const leaked = `sk-or-v1-${"a".repeat(64)}`;
			const redacted = redactEvidence(leaked);
			expect(redacted).not.toContain("sk-or-v1-");
			expect(redacted).not.toContain("a".repeat(32));
		});

		it("eval: key=value redaction must not emit a literal $1", () => {
			const leaked = `api_key=${"A".repeat(40)}`;
			const redacted = redactEvidence(leaked);
			expect(redacted).not.toContain("A".repeat(32));
			expect(redacted).not.toContain("$1");
			expect(redacted).toMatch(/api_key=\[REDACTED\]/i);
		});

		it("eval: git commit SHAs in evidence must survive redaction", () => {
			const sha = "a1b2c3d4e5f6789012345678901234567890abcd";
			expect(sha.length).toBe(40);
			const evidence = `commit ${sha} updated src/foo.ts`;
			expect(redactEvidence(evidence)).toContain(sha);
		});

		it("eval: VERIFIED is illegal when the harness truncated the diff", async () => {
			const { routing, replies, runModel, dir } = setup({ planFirst: false, triage: { enabled: false } });
			if (!routing) throw new Error("routing required");
			gitInit(dir);
			writeFileSync(join(dir, "tracked.ts"), "old\n");
			execSync("git add tracked.ts && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			writeFileSync(join(dir, "tracked.ts"), `${"x".repeat(DIFF_EVIDENCE_HARD_CAP_CHARS + 10_000)}\n`);
			replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Ship the refactor.", timestamp: Date.now() },
					toolCall("edit", { path: "tracked.ts" }),
					toolResult("edit", "patched tracked.ts"),
					toolResult("bash", "npm test ok"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			const evidence = gatherDiffEvidence(dir, "final", { focusPaths: ["tracked.ts"] });
			expect(evidence).toContain(GOAL_DIFF_TRUNCATION_MARKER);
			expect(result?.status).not.toBe("verified");
		});

		it("eval: npm test must run from the nearest package.json, not the git root", async () => {
			const repo = mkdtempSync(join(tmpdir(), "pi-eval-mono-"));
			dirs.push(repo);
			gitInit(repo);
			writeFileSync(
				join(repo, "package.json"),
				JSON.stringify({
					name: "root",
					scripts: { test: "node -e \"process.stdout.write('RAN_FROM_GIT_ROOT')\"" },
				}),
			);
			const pkg = join(repo, "pkg");
			mkdirSync(pkg);
			writeFileSync(
				join(pkg, "package.json"),
				JSON.stringify({
					name: "pkg",
					scripts: { test: "node -e \"process.stdout.write('RAN_FROM_NEAREST_PACKAGE')\"" },
				}),
			);
			// Commit so the script source is not sitting in untracked evidence.
			execSync("git add -A && git commit --no-gpg-sign -m init", { cwd: repo, stdio: "ignore" });
			const { routing, replies, runModel } = setup(
				{ planFirst: false, verifierCommands: ["npm test"], triage: { enabled: false } },
				pkg,
			);
			if (!routing) throw new Error("routing required");
			const captured: string[] = [];
			const capturingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
				captured.push(msgs.map((m) => messageText(m)).join("\n"));
				return runModel(model, msgs);
			};
			replies["moonshotai/kimi-k3"] = ["RUN: npm test", "VERDICT: VERIFIED"];
			await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Add a helper.", timestamp: Date.now() },
					toolResult("bash", "edited"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel: capturingRunModel,
			} as unknown as VerifyTurnContext);
			const runOutput = captured.join("\n").split("Output of `npm test`").slice(1).join("\n");
			expect(runOutput).toContain("RAN_FROM_NEAREST_PACKAGE");
			expect(runOutput).not.toContain("RAN_FROM_GIT_ROOT");
		});
	});

	describe("already holding — fourth-pass contracts must not regress", () => {
		it("eval: standard.makerModel must be the actual standard builder, not a dead settings key", async () => {
			const { routing, runModel } = setup(
				{
					planFirst: false,
					triage: { enabled: false },
					tiers: {
						standard: {
							makerModel: "openrouter/deepseek/deepseek-v4-flash-0731",
							makerThinkingLevel: "high",
						},
					},
				},
				undefined,
				"openai/gpt-5.6-luna",
			);
			if (!routing) throw new Error("routing must install: session Luna vs checker Kimi are different families");
			const prep = await routing.beforeFirstTurn(userTurn("Build add() with tests.", runModel));
			expect(prep?.model?.id).toBe("deepseek/deepseek-v4-flash-0731");
		});

		it("eval: hard-tier checker must not be the hard maker's family even when the session maker differs", async () => {
			const { routing, replies, runModel } = setup({
				planFirst: false,
				checkerModel: "openrouter/openai/gpt-5.6-luna",
				triage: { enabled: true },
				tiers: {
					hard: {
						makerModel: "openrouter/meta/muse-spark-1.2",
						checkerModel: "openrouter/meta/muse-spark-1.2",
					},
				},
			});
			if (!routing) throw new Error("routing required");
			replies["deepseek/deepseek-v4-flash-0731"] = ["HARD"];
			replies["meta/muse-spark-1.2"] = ["VERDICT: VERIFIED"];
			await routing.beforeFirstTurn(userTurn("Redesign auth across the whole repo.", runModel));
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Redesign auth across the whole repo.", timestamp: Date.now() },
					toolResult("bash", "edited"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			expect(result?.status).not.toBe("verified");
		});

		it("eval: openrouter-prefixed ids of the same family must not install as independent verification", () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-"));
			dirs.push(dir);
			const routing = createVerifyRouting({
				verify: { checkerModel: "openrouter/openai/gpt-5.6-luna" },
				makerModel: modelWithId("openrouter/openai/gpt-5.6-luna"),
				makerThinkingLevel: "max",
				cwd: dir,
				auditLogPath: join(dir, "a.jsonl"),
				resolveModel: (ref) => (ref ? modelWithId(ref) : undefined),
			});
			expect(routing).toBeUndefined();
		});

		it("eval: an unnumbered 40-question CLARIFY still injects at most 5 questions", async () => {
			const { routing, replies, runModel } = setup();
			if (!routing) throw new Error("routing required");
			const wall = `CLARIFY:\n${Array.from({ length: 40 }, (_, i) => `Please decide item ${i + 1}?`).join("\n")}`;
			replies["anthropic/claude-opus-5"] = [wall];
			const prep = await routing.beforeFirstTurn(
				userTurn("Migrate authentication to the new provider sometime soon.", runModel),
			);
			const injected = prep?.messages?.[0] ? messageText(prep.messages[0] as AgentMessage) : "";
			const questionBlock = injected.split("Planner's open questions:")[1] ?? injected;
			const questionLines = questionBlock
				.split("\n")
				.map((l) => l.trim())
				.filter((l) => l.length > 0 && /\?/.test(l));
			expect(questionLines.length).toBeLessThanOrEqual(5);
		});

		it("eval: a truncated untracked file must not be accepted as verified", async () => {
			const { routing, replies, runModel, dir } = setup({ planFirst: false, triage: { enabled: false } });
			if (!routing) throw new Error("routing required");
			gitInit(dir);
			writeFileSync(join(dir, "tracked.ts"), "old\n");
			execSync("git add tracked.ts && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			writeFileSync(join(dir, "brand-new.ts"), `${"x".repeat(DIFF_EVIDENCE_HARD_CAP_CHARS + 10_000)}\n`);
			replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Add the new module.", timestamp: Date.now() },
					toolCall("write", { path: "brand-new.ts" }),
					toolResult("write", "wrote brand-new.ts"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			const evidence = gatherDiffEvidence(dir, "final", { focusPaths: ["brand-new.ts"] });
			expect(evidence).toContain(GOAL_DIFF_TRUNCATION_MARKER);
			expect(result?.status).not.toBe("verified");
		});

		it("eval: live standard checker fallback must exist and be a third family", () => {
			const livePath = join(homedir(), ".pi", "agent", "settings.json");
			if (!existsSync(livePath)) throw new Error("live settings.json is the contract under test");
			const live = JSON.parse(readFileSync(livePath, "utf8")) as {
				verify: {
					checkerModel?: string;
					checkerFallbackModel?: string;
					tiers?: { standard?: { makerModel?: string; checkerFallbackModel?: string } };
				};
			};
			const standardMaker = live.verify.tiers?.standard?.makerModel;
			const checker = live.verify.checkerModel;
			const fallback = live.verify.tiers?.standard?.checkerFallbackModel ?? live.verify.checkerFallbackModel;
			expect(standardMaker).toBeTruthy();
			expect(fallback).toBeTruthy();
			expect(modelFamily(standardMaker!)).not.toBe(modelFamily(fallback!));
			expect(modelFamily(fallback!)).not.toBe(modelFamily(checker ?? ""));
		});

		it("eval: checker audit messages must fit the checker context window", async () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-"));
			dirs.push(dir);
			gitInit(dir);
			writeFileSync(join(dir, "tracked.ts"), "old\n");
			execSync("git add tracked.ts && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			writeFileSync(join(dir, "tracked.ts"), `${"x".repeat(12_000)}\n`);
			const tinyChecker = { ...modelWithId("moonshotai/kimi-k3"), contextWindow: 9_000 };
			const routing = createVerifyRouting({
				verify: {
					checkerModel: "openrouter/moonshotai/kimi-k3",
					planFirst: false,
					triage: { enabled: false },
					auditOnlyAfterTools: true,
				},
				makerModel: modelWithId("deepseek/deepseek-v4-flash-0731"),
				makerThinkingLevel: "high",
				cwd: dir,
				auditLogPath: join(dir, "a.jsonl"),
				resolveModel: (ref) => {
					if (!ref) return undefined;
					if (ref.includes("kimi-k3")) return tinyChecker;
					return modelWithId(ref.replace(/^openrouter\//, ""));
				},
			});
			if (!routing) throw new Error("routing required");
			let checkerChars = 0;
			const capturingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
				if (model.id.includes("kimi")) {
					checkerChars = msgs.reduce((sum, m) => sum + messageText(m).length, 0);
				}
				return assistant("VERDICT: CONFLICT", model.id);
			};
			await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Ship the refactor.", timestamp: Date.now() },
					toolResult("bash", "ok"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel: capturingRunModel,
			} as unknown as VerifyTurnContext);
			const maxChars = (tinyChecker.contextWindow - 8192) * 4;
			expect(checkerChars).toBeGreaterThan(0);
			expect(checkerChars).toBeLessThanOrEqual(maxChars);
		});
	});

	describe("already holding — fifth-pass contracts must not regress", () => {
		it("eval: session Luna plus checker Luna must still install when standard.makerModel is DeepSeek", async () => {
			const { routing, runModel } = setup(
				{
					planFirst: false,
					triage: { enabled: false },
					checkerModel: "openrouter/openai/gpt-5.6-luna",
					tiers: {
						standard: {
							makerModel: "openrouter/deepseek/deepseek-v4-flash-0731",
							makerThinkingLevel: "high",
						},
					},
				},
				undefined,
				"openai/gpt-5.6-luna",
			);
			expect(routing).toBeTruthy();
			const prep = await routing!.beforeFirstTurn(userTurn("Build add() with tests.", runModel));
			expect(prep?.model?.id).toBe("deepseek/deepseek-v4-flash-0731");
		});

		it("eval: hard Spark must not grade hard Spark when the maker comes from escalationMakerModel", async () => {
			const { routing, replies, runModel } = setup({
				planFirst: false,
				checkerModel: "openrouter/openai/gpt-5.6-luna",
				escalationMakerModel: "openrouter/meta/muse-spark-1.2",
				triage: { enabled: true },
				tiers: {
					hard: {
						checkerModel: "openrouter/meta/muse-spark-1.2",
					},
				},
			});
			if (!routing) throw new Error("routing required");
			replies["deepseek/deepseek-v4-flash-0731"] = ["HARD"];
			replies["meta/muse-spark-1.2"] = ["VERDICT: VERIFIED"];
			await routing.beforeFirstTurn(userTurn("Redesign auth across the whole repo.", runModel));
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Redesign auth across the whole repo.", timestamp: Date.now() },
					toolResult("bash", "edited"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			expect(result?.status).not.toBe("verified");
		});

		it("eval: a 40-line CLARIFY without question marks still injects at most 5 lines", async () => {
			const { routing, replies, runModel } = setup();
			if (!routing) throw new Error("routing required");
			const wall = `CLARIFY:\n${Array.from({ length: 40 }, (_, i) => `Decide the scope for item ${i + 1}.`).join("\n")}`;
			replies["anthropic/claude-opus-5"] = [wall];
			const prep = await routing.beforeFirstTurn(
				userTurn("Migrate authentication to the new provider sometime soon.", runModel),
			);
			const injected = prep?.messages?.[0] ? messageText(prep.messages[0] as AgentMessage) : "";
			const questionBlock = injected.split("Planner's open questions:")[1] ?? injected;
			const lines = questionBlock
				.split("\n")
				.map((l) => l.trim())
				.filter((l) => l.length > 0);
			expect(lines.length).toBeLessThanOrEqual(5);
		});

		it("eval: openrouter-prefixed hard maker must still fall back to DeepSeek, not skip it as family openrouter", async () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-"));
			dirs.push(dir);
			const routing = createVerifyRouting({
				verify: {
					checkerModel: "openrouter/openai/gpt-5.6-luna",
					checkerFallbackModel: "openrouter/deepseek/deepseek-v4-flash-0731",
					planFirst: false,
					triage: { enabled: true },
					auditOnlyAfterTools: true,
					tiers: {
						hard: {
							makerModel: "openrouter/meta/muse-spark-1.2",
							makerThinkingLevel: "xhigh",
						},
					},
				},
				makerModel: modelWithId("openrouter/deepseek/deepseek-v4-flash-0731"),
				makerThinkingLevel: "high",
				cwd: dir,
				auditLogPath: join(dir, "a.jsonl"),
				resolveModel: (ref) => (ref ? modelWithId(ref) : undefined),
			});
			if (!routing) throw new Error("routing required");
			const auditPath = join(dir, "a.jsonl");
			let usedTriage = false;
			const runModel = async (model: Model<any>, _msgs: AgentMessage[]) => {
				if (model.id.includes("luna")) {
					return assistant("unavailable", model.id, { stopReason: "error", errorMessage: "429" });
				}
				if (model.id.includes("deepseek") && !usedTriage) {
					usedTriage = true;
					return assistant("HARD", model.id);
				}
				return assistant("VERDICT: VERIFIED", model.id);
			};
			await routing.beforeFirstTurn(userTurn("Redesign auth across the whole repo.", runModel));
			await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Redesign auth across the whole repo.", timestamp: Date.now() },
					toolResult("bash", "edited"),
				],
				config: { model: modelWithId("openrouter/deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			const events = existsSync(auditPath)
				? readFileSync(auditPath, "utf8")
						.trim()
						.split("\n")
						.filter(Boolean)
						.map((line) => JSON.parse(line) as Record<string, unknown>)
				: [];
			expect(events.some((e) => e.event === "fallback" && String(e.to).includes("deepseek"))).toBe(true);
		});

		it("eval: checker RUN follow-up messages must also fit the checker context window", async () => {
			const repo = mkdtempSync(join(tmpdir(), "pi-eval-"));
			dirs.push(repo);
			gitInit(repo);
			writeFileSync(join(repo, "tracked.ts"), "old\n");
			writeFileSync(
				join(repo, "package.json"),
				JSON.stringify({ name: "pkg", scripts: { test: "node -e \"process.stdout.write('ok')\"" } }),
			);
			execSync("git add -A && git commit --no-gpg-sign -m init", { cwd: repo, stdio: "ignore" });
			writeFileSync(join(repo, "tracked.ts"), `${"x".repeat(12_000)}\n`);
			const tinyChecker = { ...modelWithId("moonshotai/kimi-k3"), contextWindow: 9_000 };
			const routing = createVerifyRouting({
				verify: {
					checkerModel: "openrouter/moonshotai/kimi-k3",
					planFirst: false,
					triage: { enabled: false },
					auditOnlyAfterTools: true,
					verifierCommands: ["npm test"],
				},
				makerModel: modelWithId("deepseek/deepseek-v4-flash-0731"),
				makerThinkingLevel: "high",
				cwd: repo,
				auditLogPath: join(repo, "a.jsonl"),
				resolveModel: (ref) => {
					if (!ref) return undefined;
					if (ref.includes("kimi-k3")) return tinyChecker;
					return modelWithId(ref.replace(/^openrouter\//, ""));
				},
			});
			if (!routing) throw new Error("routing required");
			const checkerPayloads: number[] = [];
			let kimiCalls = 0;
			const capturingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
				if (model.id.includes("kimi")) {
					checkerPayloads.push(msgs.reduce((sum, m) => sum + messageText(m).length, 0));
					kimiCalls += 1;
					if (kimiCalls === 1) return assistant("RUN: npm test", model.id);
					return assistant("VERDICT: VERIFIED", model.id);
				}
				return assistant("ok", model.id);
			};
			await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Ship the refactor.", timestamp: Date.now() },
					toolResult("bash", "ok"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel: capturingRunModel,
			} as unknown as VerifyTurnContext);
			const maxChars = (tinyChecker.contextWindow - 8192) * 4;
			expect(checkerPayloads.length).toBeGreaterThanOrEqual(2);
			for (const n of checkerPayloads) {
				expect(n).toBeLessThanOrEqual(maxChars);
			}
		});

		it("eval: RUN command output must be redacted before it reaches the checker", async () => {
			const repo = mkdtempSync(join(tmpdir(), "pi-eval-redact-run-"));
			dirs.push(repo);
			gitInit(repo);
			const leaked = `sk-or-v1-${"a".repeat(64)}`;
			writeFileSync(
				join(repo, "package.json"),
				JSON.stringify({
					name: "pkg",
					scripts: { test: `node -e "process.stdout.write(${JSON.stringify(`OPENROUTER_API_KEY=${leaked}`)})"` },
				}),
			);
			execSync("git add -A && git commit --no-gpg-sign -m init", { cwd: repo, stdio: "ignore" });
			writeFileSync(join(repo, "tracked.ts"), "x\n");
			const { routing, replies, runModel } = setup(
				{ planFirst: false, verifierCommands: ["npm test"], triage: { enabled: false } },
				repo,
			);
			if (!routing) throw new Error("routing required");
			const captured: string[] = [];
			const capturingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
				captured.push(msgs.map((m) => messageText(m)).join("\n"));
				return runModel(model, msgs);
			};
			replies["moonshotai/kimi-k3"] = ["RUN: npm test", "VERDICT: VERIFIED"];
			await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Add a helper.", timestamp: Date.now() },
					toolResult("bash", "edited"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel: capturingRunModel,
			} as unknown as VerifyTurnContext);
			const followUp = captured.slice(1).join("\n");
			expect(followUp).toMatch(/Output of `npm test`/);
			expect(followUp).not.toContain(leaked);
			expect(followUp).not.toContain("sk-or-v1-");
		});

		it("eval: truncation-override must inspect the last bounded checker payload after RUN", async () => {
			const repo = mkdtempSync(join(tmpdir(), "pi-eval-run-bound-"));
			dirs.push(repo);
			gitInit(repo);
			writeFileSync(join(repo, "tracked.ts"), "old\n");
			writeFileSync(
				join(repo, "package.json"),
				JSON.stringify({
					name: "pkg",
					scripts: {
						test: "node -e \"process.stdout.write(('verifier output line\\n').repeat(220))\"",
					},
				}),
			);
			execSync("git add -A && git commit --no-gpg-sign -m init", { cwd: repo, stdio: "ignore" });
			writeFileSync(join(repo, "tracked.ts"), "new\n");
			const tinyChecker = { ...modelWithId("moonshotai/kimi-k3"), contextWindow: 9_000 };
			const routing = createVerifyRouting({
				verify: {
					checkerModel: "openrouter/moonshotai/kimi-k3",
					planFirst: false,
					triage: { enabled: false },
					auditOnlyAfterTools: true,
					verifierCommands: ["npm test"],
				},
				makerModel: modelWithId("deepseek/deepseek-v4-flash-0731"),
				makerThinkingLevel: "high",
				cwd: repo,
				auditLogPath: join(repo, "a.jsonl"),
				resolveModel: (ref) => {
					if (!ref) return undefined;
					if (ref.includes("kimi-k3")) return tinyChecker;
					return modelWithId(ref.replace(/^openrouter\//, ""));
				},
			});
			if (!routing) throw new Error("routing required");
			const payloads: string[] = [];
			let kimiCalls = 0;
			const capturingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
				if (model.id.includes("kimi")) {
					payloads.push(msgs.map((m) => messageText(m)).join("\n"));
					kimiCalls += 1;
					if (kimiCalls === 1) return assistant("RUN: npm test", model.id);
					return assistant("VERDICT: VERIFIED", model.id);
				}
				return assistant("ok", model.id);
			};
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Ship the helper.", timestamp: Date.now() },
					toolResult("bash", "ok"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel: capturingRunModel,
			} as unknown as VerifyTurnContext);
			expect(payloads.length).toBeGreaterThanOrEqual(2);
			expect(payloads[0]).toMatch(/git diff HEAD/i);
			expect(payloads[payloads.length - 1]).not.toMatch(/git diff HEAD/i);
			expect(result?.status).not.toBe("verified");
			expect((result as any).notice).toMatch(/UNVERIFIED/i);
		});
	});

	describe("promotion blockers — stay red until the contract holds", () => {
		it("eval: a mixed CLARIFY of 3 numbered questions plus 37 extra lines still injects at most 5 lines", async () => {
			const { routing, replies, runModel } = setup();
			if (!routing) throw new Error("routing required");
			const wall = `CLARIFY:\n1. Auth method?\n2. Scope?\n3. Deadline?\n${Array.from({ length: 37 }, (_, i) => `Also consider item ${i + 1}.`).join("\n")}`;
			replies["anthropic/claude-opus-5"] = [wall];
			const prep = await routing.beforeFirstTurn(
				userTurn("Migrate authentication to the new provider sometime soon.", runModel),
			);
			const injected = prep?.messages?.[0] ? messageText(prep.messages[0] as AgentMessage) : "";
			const questionBlock = injected.split("Planner's open questions:")[1] ?? injected;
			const lines = questionBlock
				.split("\n")
				.map((l) => l.trim())
				.filter((l) => l.length > 0);
			expect(lines.length).toBeLessThanOrEqual(5);
		});

		it("eval: boundContextForModel must shorten an oversized single goal message", () => {
			const tiny = { ...modelWithId("test/tiny"), contextWindow: 9_000 };
			const huge: AgentMessage = {
				role: "user",
				content: "x".repeat(20_000),
				timestamp: Date.now(),
			};
			const bounded = boundContextForModel([huge], tiny);
			const maxChars = (tiny.contextWindow - 8192) * 4;
			expect(messageText(bounded[0] as AgentMessage).length).toBeLessThanOrEqual(maxChars);
		});

		it("eval: boundContextForModel with contextWindow 1000 (smaller than reserve) does not throw and does not forward the tail", () => {
			const tiny = { ...modelWithId("test/tiny"), contextWindow: 1_000 };
			const goal: AgentMessage = { role: "user", content: "Build add() with tests.", timestamp: Date.now() };
			const tail: AgentMessage = {
				role: "assistant",
				content: [{ type: "text", text: "y".repeat(20_000) }],
				timestamp: Date.now(),
			} as AgentMessage;
			const bounded = boundContextForModel([goal, tail], tiny);
			expect(bounded.length).toBe(1);
			expect(messageText(bounded[0] as AgentMessage)).toMatch(/TRUNCATED/i);
			expect(messageText(bounded[0] as AgentMessage)).not.toContain("y".repeat(50));
		});

		it("eval: trivial work must not be Luna grading Luna after a DeepSeek standard pin lets routing install", async () => {
			const { routing, replies, runModel } = setup(
				{
					planFirst: false,
					triage: { enabled: true },
					checkerModel: "openrouter/openai/gpt-5.6-luna",
					checkpointCheckerModel: "openrouter/openai/gpt-5.6-luna",
					tiers: {
						standard: {
							makerModel: "openrouter/deepseek/deepseek-v4-flash-0731",
							makerThinkingLevel: "high",
						},
					},
				},
				undefined,
				"openai/gpt-5.6-luna",
			);
			if (!routing) throw new Error("routing must install because standard maker is DeepSeek");
			replies["openai/gpt-5.6-luna"] = ["TRIVIAL", "VERDICT: VERIFIED"];
			await routing.beforeFirstTurn(userTurn("Rename the helper.", runModel));
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("renamed"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Rename the helper.", timestamp: Date.now() },
					toolResult("bash", "mv a.ts b.ts"),
				],
				config: { model: modelWithId("openai/gpt-5.6-luna"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			expect(result?.status).not.toBe("verified");
		});

		it("eval: hard Spark must not grade hard Spark when the checker is only the global checkerModel", async () => {
			const { routing, replies, runModel } = setup({
				planFirst: false,
				checkerModel: "openrouter/meta/muse-spark-1.2",
				triage: { enabled: true },
				tiers: {
					hard: {
						makerModel: "openrouter/meta/muse-spark-1.2",
						makerThinkingLevel: "xhigh",
					},
				},
			});
			if (!routing) throw new Error("routing required");
			replies["deepseek/deepseek-v4-flash-0731"] = ["HARD"];
			replies["meta/muse-spark-1.2"] = ["VERDICT: VERIFIED"];
			await routing.beforeFirstTurn(userTurn("Redesign auth across the whole repo.", runModel));
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Redesign auth across the whole repo.", timestamp: Date.now() },
					toolResult("bash", "edited"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			expect(result?.status).not.toBe("verified");
		});

		it("eval: checker prompt must be bounded when the context window is smaller than the reserve", async () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-"));
			dirs.push(dir);
			gitInit(dir);
			writeFileSync(join(dir, "tracked.ts"), "old\n");
			execSync("git add tracked.ts && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			writeFileSync(join(dir, "tracked.ts"), `${"x".repeat(12_000)}\n`);
			const tinyChecker = { ...modelWithId("moonshotai/kimi-k3"), contextWindow: 1_000 };
			const routing = createVerifyRouting({
				verify: {
					checkerModel: "openrouter/moonshotai/kimi-k3",
					planFirst: false,
					triage: { enabled: false },
					auditOnlyAfterTools: true,
				},
				makerModel: modelWithId("deepseek/deepseek-v4-flash-0731"),
				makerThinkingLevel: "high",
				cwd: dir,
				auditLogPath: join(dir, "a.jsonl"),
				resolveModel: (ref) => {
					if (!ref) return undefined;
					if (ref.includes("kimi-k3")) return tinyChecker;
					return modelWithId(ref.replace(/^openrouter\//, ""));
				},
			});
			if (!routing) throw new Error("routing required");
			let checkerChars = 0;
			const capturingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
				if (model.id.includes("kimi")) {
					checkerChars = msgs.reduce((sum, m) => sum + messageText(m).length, 0);
				}
				return assistant("VERDICT: CONFLICT", model.id);
			};
			await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Ship the refactor.", timestamp: Date.now() },
					toolResult("bash", "ok"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel: capturingRunModel,
			} as unknown as VerifyTurnContext);
			expect(checkerChars).toBeGreaterThan(0);
			expect(checkerChars).toBeLessThanOrEqual(tinyChecker.contextWindow * 4);
		});

		it("eval: a complete mid-size diff that bounding dropped must not be accepted as verified", async () => {
			const dir = mkdtempSync(join(tmpdir(), "pi-eval-"));
			dirs.push(dir);
			gitInit(dir);
			writeFileSync(join(dir, "tracked.ts"), "old\n");
			execSync("git add tracked.ts && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			writeFileSync(join(dir, "tracked.ts"), `${"y".repeat(4_000)}\n`);
			const tinyChecker = { ...modelWithId("moonshotai/kimi-k3"), contextWindow: 9_000 };
			const routing = createVerifyRouting({
				verify: {
					checkerModel: "openrouter/moonshotai/kimi-k3",
					planFirst: false,
					triage: { enabled: false },
					auditOnlyAfterTools: true,
				},
				makerModel: modelWithId("deepseek/deepseek-v4-flash-0731"),
				makerThinkingLevel: "high",
				cwd: dir,
				auditLogPath: join(dir, "a.jsonl"),
				resolveModel: (ref) => {
					if (!ref) return undefined;
					if (ref.includes("kimi-k3")) return tinyChecker;
					return modelWithId(ref.replace(/^openrouter\//, ""));
				},
			});
			if (!routing) throw new Error("routing required");
			let checkerSawDiff = false;
			const capturingRunModel = async (model: Model<any>, msgs: AgentMessage[]) => {
				if (model.id.includes("kimi") && msgs.some((m) => messageText(m).includes("y".repeat(20)))) {
					checkerSawDiff = true;
				}
				return assistant("VERDICT: VERIFIED", model.id);
			};
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Ship the refactor.", timestamp: Date.now() },
					toolResult("bash", "ok"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel: capturingRunModel,
			} as unknown as VerifyTurnContext);
			const evidence = gatherDiffEvidence(dir, "final", { skipUntracked: ["a.jsonl"] });
			expect(evidence).not.toContain(GOAL_DIFF_TRUNCATION_MARKER);
			if (!checkerSawDiff) {
				expect(result?.status).not.toBe("verified");
			}
		});

		it("eval: GitHub and xAI-style keys in tool output must not reach the checker", () => {
			const github = `ghp_${"A".repeat(36)}`;
			const xai = `xai-${"b".repeat(48)}`;
			expect(redactEvidence(github)).not.toContain("ghp_");
			expect(redactEvidence(xai)).not.toContain("xai-");
		});

		it("eval: a small complete diff that mentions more chars may still be verified", async () => {
			const { routing, replies, runModel, dir } = setup({ planFirst: false, triage: { enabled: false } });
			if (!routing) throw new Error("routing required");
			gitInit(dir);
			writeFileSync(join(dir, "tracked.ts"), "old\n");
			execSync("git add tracked.ts && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			writeFileSync(join(dir, "tracked.ts"), "export const note = 'more chars';\n");
			replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Add a note constant.", timestamp: Date.now() },
					toolResult("bash", "ok"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			expect(result?.status).toBe("verified");
		});

		it("eval: a large dirty tree unrelated to this run must not void a verified goal", async () => {
			const { routing, replies, runModel, dir, readAudit } = setup({ planFirst: false, triage: { enabled: false } });
			if (!routing) throw new Error("routing required");
			gitInit(dir);
			writeFileSync(join(dir, "unrelated.ts"), "old-unrelated\n");
			writeFileSync(join(dir, "helper.ts"), "export const before = 1;\n");
			execSync("git add unrelated.ts helper.ts && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			writeFileSync(join(dir, "unrelated.ts"), `${"x".repeat(DIFF_EVIDENCE_HARD_CAP_CHARS + 10_000)}\n`);
			writeFileSync(
				join(dir, "helper.ts"),
				"export function hasLeadingTilde(path: string): boolean { return path.startsWith('~'); }\n",
			);
			const focused = gatherDiffEvidence(dir, "final", { focusPaths: ["helper.ts"] });
			expect(focused).toContain("hasLeadingTilde");
			expect(focused).toContain(OTHER_DIFF_TRUNCATION_MARKER);
			expect(focused).not.toContain(GOAL_DIFF_TRUNCATION_MARKER);
			expect(evidenceTruncationBlocksVerified(focused)).toBe(false);
			expect(collectTouchedPaths([toolCall("edit", { path: "helper.ts" })], dir)).toEqual(["helper.ts"]);
			replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("done"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Add hasLeadingTilde to helper.ts and test it.", timestamp: Date.now() },
					toolCall("edit", { path: "helper.ts" }),
					toolResult("edit", "updated helper.ts"),
					toolResult("bash", "npx vitest run paths.test.ts ok"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			expect(result?.status).toBe("verified");
			expect(readAudit().some((e) => e.event === "truncation-override")).toBe(false);
		});

		it("eval: a read-only task in a dirty tree may still be verified", async () => {
			const { routing, replies, runModel, dir } = setup({ planFirst: false, triage: { enabled: false } });
			if (!routing) throw new Error("routing required");
			gitInit(dir);
			writeFileSync(join(dir, "types.ts"), "export type Flag = boolean;\n");
			writeFileSync(join(dir, "unrelated.ts"), "old\n");
			execSync("git add types.ts unrelated.ts && git commit --no-gpg-sign -m init", { cwd: dir, stdio: "ignore" });
			writeFileSync(join(dir, "unrelated.ts"), `${"y".repeat(DIFF_EVIDENCE_HARD_CAP_CHARS + 10_000)}\n`);
			replies["moonshotai/kimi-k3"] = ["VERDICT: VERIFIED"];
			const result = await routing.verifyTurn({
				kind: "final",
				message: assistant("No lint issues."),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Read types.ts and check lint.", timestamp: Date.now() },
					toolCall("read", { path: "types.ts" }),
					toolResult("read", "export type Flag = boolean;"),
					toolResult("bash", "biome check packages/agent/src/types.ts ok"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			expect(collectTouchedPaths([toolCall("read", { path: "types.ts" })], dir)).toEqual(["types.ts"]);
			expect(result?.status).toBe("verified");
		});

		it("eval: other-tree truncation must not block VERIFIED", () => {
			const text = `git diff HEAD (files this run edited):\n+ok\n${OTHER_DIFF_TRUNCATION_MARKER}`;
			expect(evidenceTruncationBlocksVerified(text)).toBe(false);
			expect(evidenceTruncationBlocksVerified(`${text}\n${GOAL_DIFF_TRUNCATION_MARKER}`)).toBe(true);
		});

		it("eval: hard Spark must not grade hard Spark at a checkpoint either", async () => {
			const { routing, replies, runModel } = setup({
				planFirst: false,
				checkerModel: "openrouter/openai/gpt-5.6-luna",
				checkpointCheckerModel: "openrouter/meta/muse-spark-1.2",
				triage: { enabled: true },
				tiers: {
					hard: {
						makerModel: "openrouter/meta/muse-spark-1.2",
						makerThinkingLevel: "xhigh",
					},
				},
			});
			if (!routing) throw new Error("routing required");
			replies["deepseek/deepseek-v4-flash-0731"] = ["HARD"];
			replies["meta/muse-spark-1.2"] = ["VERDICT: VERIFIED"];
			await routing.beforeFirstTurn(userTurn("Redesign auth across the whole repo.", runModel));
			const result = await routing.verifyTurn({
				kind: "checkpoint",
				toolTurnCount: 4,
				message: assistant("progress"),
				context: { systemPrompt: "", messages: [], tools: [] },
				newMessages: [
					{ role: "user", content: "Redesign auth across the whole repo.", timestamp: Date.now() },
					toolResult("bash", "wip"),
				],
				config: { model: modelWithId("deepseek/deepseek-v4-flash-0731"), convertToLlm: () => [] },
				runModel,
			} as unknown as VerifyTurnContext);
			expect(result?.status).not.toBe("verified");
		});
	});
});
