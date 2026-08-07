/**
 * Maker / checker / planner routing for mid-build and final verification.
 *
 * Phases:
 *   0. triage        — one cheap classification call routes the run to a tier
 *                      (trivial | standard | hard); each tier picks the
 *                      cheapest models that hold quality
 *   1. planFirst     — planner model decomposes the goal before the maker
 *                      starts (skipped for trivial); an ambiguous goal makes
 *                      the planner reply CLARIFY and the maker relays the
 *                      questions to the user instead of guessing
 *   2. make          — tier maker model executes with tools
 *   3. checkpoint    — mid-build checker audits progress on a geometric
 *                      schedule (N, 2N, 4N … tool turns)
 *   4. escalate      — after repeated rejects, retries use the escalation
 *                      maker (sticky for the rest of the run)
 *   5. final         — final checker audits the no-tool-call answer against
 *                      the real artifact (git diff), may order verifier
 *                      commands run by the harness; planner may re-plan
 *
 * Planner and checkers never execute tools. Only makers do. Budget-exhausted
 * acceptance is honest: the user sees "[VERIFY] UNVERIFIED", never a fake pass.
 * Every model call logs token/cost usage to the audit log; a run-summary event
 * totals the verification overhead per run.
 */

import { execSync } from "node:child_process";
import { appendFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type {
	AgentMessage,
	BeforeFirstTurnContext,
	BeforeFirstTurnResult,
	ThinkingLevel,
	TurnVerifyResult,
	VerifyTurnContext,
} from "@earendil-works/pi-agent-core";
import type {
	AssistantMessage,
	Model,
	TextContent,
	ToolResultMessage,
	UserMessage,
} from "@earendil-works/pi-ai/compat";
import type { VerifySettings, VerifyTierOverrides } from "./settings-manager.ts";

export type VerifyKind = "final" | "checkpoint";
export type VerifyTier = "trivial" | "standard" | "hard";

/**
 * Quality bar shared by every checker audit. Tech debt is a rejection, not a
 * note: the point of the checker is that this garbage never survives a run.
 */
export const EXCELLENCE_CHARTER =
	"Hold this work to the standard of the world's best engineers. Treat each of the following as grounds for " +
	"CONFLICT, not a side note: mock or placeholder data, stubbed or fake wiring presented as working, TODO/FIXME " +
	"markers left in the change, dead code, swallowed or silently ignored errors, hardcoded secrets, behavior " +
	"claimed without a receipt proving it, and obvious unhandled edge cases.";

export interface VerifyRoutingHooks {
	beforeFirstTurn: (ctx: BeforeFirstTurnContext) => Promise<BeforeFirstTurnResult | undefined>;
	shouldCheckpoint: (ctx: { toolTurnCount: number }) => boolean;
	verifyTurn: (ctx: VerifyTurnContext) => Promise<TurnVerifyResult | undefined>;
}

export interface CreateVerifyRoutingOptions {
	verify: VerifySettings;
	makerModel: Model<any>;
	makerThinkingLevel: ThinkingLevel;
	cwd: string;
	auditLogPath: string;
	resolveModel: (ref: string | undefined) => Model<any> | undefined;
}

export function messageText(m: AssistantMessage | ToolResultMessage | UserMessage | AgentMessage): string {
	const content = (m as { content?: unknown }).content;
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";
	return content
		.filter((c): c is TextContent => !!c && typeof c === "object" && (c as TextContent).type === "text")
		.map((c) => c.text)
		.join(" ")
		.trim();
}

/** Pure: whether a mid-build checkpoint should run for this tool-turn count. */
export function shouldRunCheckpoint(opts: {
	toolTurnCount: number;
	everyN: number;
	checkpointsSoFar: number;
	maxCheckpoints: number;
	/** "geometric" audits at N, 2N, 4N … so long runs never go unaudited. Default: "fixed" (every N). */
	backoff?: "fixed" | "geometric";
}): boolean {
	const { toolTurnCount, everyN, checkpointsSoFar, maxCheckpoints } = opts;
	if (everyN <= 0) return false;
	if (maxCheckpoints <= 0) return false;
	if (checkpointsSoFar >= maxCheckpoints) return false;
	if (toolTurnCount <= 0) return false;
	if (opts.backoff === "geometric") {
		return toolTurnCount >= everyN * 2 ** checkpointsSoFar;
	}
	return toolTurnCount % everyN === 0;
}

/** Pure: whether plan-first should run for this prompt. */
export function shouldPlanFirst(opts: {
	planFirst: boolean;
	promptText: string;
	minPromptChars: number;
	alreadyPlanned: boolean;
}): boolean {
	if (!opts.planFirst || opts.alreadyPlanned) return false;
	return opts.promptText.trim().length >= opts.minPromptChars;
}

/**
 * Classify a checker reply.
 * Prefer CONFLICT when present. Never treat the substring inside "unverified"
 * as VERIFIED, and never treat a negated VERIFIED ("cannot be VERIFIED") as a
 * pass. The positive match is case-sensitive: the audit prompt demands the
 * exact uppercase token, so prose like "I verified the tests" stays ambiguous.
 */
export function classifyCheckerVerdict(auditText: string): "verified" | "conflict" | "ambiguous" {
	const text = auditText.trim();
	if (!text) return "ambiguous";
	if (/\bCONFLICT\b/i.test(text)) return "conflict";
	if (/\bun-?verified\b/i.test(text)) return "ambiguous";
	if (/\b(?:not|cannot|can't|isn't|is not|never|won't|no)\b[^.\n]{0,60}?\bVERIFIED\b/i.test(text)) return "ambiguous";
	if (/\bVERIFIED\b/.test(text)) return "verified";
	return "ambiguous";
}

/** Pure: whether retries should use the escalated maker. */
export function shouldEscalateMaker(opts: { rejections: number; escalateAfterRejections: number }): boolean {
	const after = opts.escalateAfterRejections;
	if (after <= 0) return false;
	return opts.rejections >= after;
}

/** Pure: parse a triage reply into a tier. First clear token wins. */
export function classifyTriageTier(reply: string): VerifyTier | undefined {
	const match = /\b(TRIVIAL|STANDARD|HARD)\b/i.exec(reply);
	if (!match) return undefined;
	return match[1].toLowerCase() as VerifyTier;
}

/** Pure: conversational prompts that never need triage, planning, or audits. */
export function isConversationalPrompt(promptText: string): boolean {
	const text = promptText.trim();
	if (text.length < 20) return true;
	return /^(hi|hey|hello|thanks|thank you|ok|okay|yes|no|nice|cool|great|good)\b[\s\S]{0,40}$/i.test(text);
}

function tierOverrides(verify: VerifySettings, tier: VerifyTier): VerifyTierOverrides | undefined {
	if (tier === "trivial") return verify.tiers?.trivial;
	if (tier === "hard") return verify.tiers?.hard;
	return undefined;
}

export function checkerRefForKind(
	verify: VerifySettings,
	kind: VerifyKind,
	tier: VerifyTier = "standard",
): string | undefined {
	const overrides = tierOverrides(verify, tier);
	if (kind === "checkpoint") {
		return overrides?.checkpointCheckerModel ?? verify.checkpointCheckerModel ?? verify.checkerModel;
	}
	if (tier === "trivial") {
		// Trivial finals get the cheap checker: the artifact is small and the
		// checkpoint checker is the cost-effective tier for it.
		return overrides?.checkerModel ?? verify.checkpointCheckerModel ?? verify.checkerModel;
	}
	return overrides?.checkerModel ?? verify.checkerModel;
}

export function checkerThinkingForKind(
	verify: VerifySettings,
	kind: VerifyKind,
	tier: VerifyTier = "standard",
): ThinkingLevel | undefined {
	const overrides = tierOverrides(verify, tier);
	if (kind === "checkpoint") {
		return (
			overrides?.checkpointCheckerThinkingLevel ??
			verify.checkpointCheckerThinkingLevel ??
			verify.checkerThinkingLevel
		);
	}
	if (tier === "trivial") {
		return overrides?.checkerThinkingLevel ?? "medium";
	}
	return overrides?.checkerThinkingLevel ?? verify.checkerThinkingLevel;
}

export function createVerifyAuditLogger(auditLogPath: string): (entry: Record<string, unknown>) => void {
	return (entry: Record<string, unknown>): void => {
		try {
			appendFileSync(auditLogPath, `${JSON.stringify({ ts: new Date().toISOString(), ...entry })}\n`);
		} catch {
			// audit logging must never break the loop
		}
	};
}

function usageFields(m: AssistantMessage): { tokensIn: number; tokensOut: number; costUsd: number } {
	const usage = (m as { usage?: { input?: number; output?: number; cost?: { total?: number } } }).usage;
	return {
		tokensIn: usage?.input ?? 0,
		tokensOut: usage?.output ?? 0,
		costUsd: usage?.cost?.total ?? 0,
	};
}

function guardedExec(cwd: string, command: string, timeout = 2_000): string {
	try {
		return execSync(command, {
			cwd,
			timeout,
			encoding: "utf8",
			maxBuffer: 1024 * 1024,
			stdio: ["ignore", "pipe", "ignore"],
		}).trim();
	} catch {
		return "";
	}
}

function headLines(text: string, n: number): string {
	return text.split("\n").slice(0, n).join("\n");
}

/**
 * Cheap synchronous repo snapshot shared by triage and the planner so neither
 * plans in a vacuum. Never throws; empty string outside a repo.
 */
export function gatherRepoSignals(cwd: string): string {
	const parts: string[] = [];
	const status = guardedExec(cwd, "git status --short");
	if (status) parts.push(`git status --short:\n${headLines(status, 20)}`);
	const files = guardedExec(cwd, "git ls-files");
	if (files) parts.push(`files (first 200):\n${headLines(files, 200)}`);
	try {
		const pkg = JSON.parse(readFileSync(join(cwd, "package.json"), "utf8")) as {
			name?: string;
			scripts?: Record<string, string>;
		};
		parts.push(`package.json: name=${pkg.name ?? "?"} scripts=${Object.keys(pkg.scripts ?? {}).join(", ")}`);
	} catch {
		// no package.json — fine
	}
	return parts.join("\n\n").slice(0, 2_000);
}

/**
 * The real artifact for checker audits: working-tree diff against HEAD.
 * Final audits get the truncated diff; checkpoints get the cheap stat view.
 */
export function gatherDiffEvidence(cwd: string, kind: VerifyKind): string {
	const status = guardedExec(cwd, "git status --short");
	const stat = guardedExec(cwd, "git diff --stat HEAD", 5_000);
	const parts: string[] = [];
	if (status) parts.push(`git status --short:\n${headLines(status, 30)}`);
	if (stat) parts.push(`git diff --stat HEAD:\n${headLines(stat, 40)}`);
	if (kind === "final") {
		const diff = guardedExec(cwd, "git diff HEAD", 5_000);
		if (diff) parts.push(`git diff HEAD (truncated):\n${diff.slice(0, 6_000)}`);
	}
	return parts.join("\n\n");
}

export function createVerifyRouting(options: CreateVerifyRoutingOptions): VerifyRoutingHooks | undefined {
	const { verify, makerModel, makerThinkingLevel, cwd, resolveModel } = options;
	const finalChecker = resolveModel(verify.checkerModel);
	if (!verify.checkerModel || !finalChecker) return undefined;

	const logVerify = createVerifyAuditLogger(options.auditLogPath);
	const everyN = verify.checkpointEveryToolTurns ?? 0;
	const maxCheckpoints = verify.maxCheckpointsPerRun ?? 8;
	const backoff = verify.checkpointBackoff ?? "geometric";
	const planFirst = verify.planFirst ?? Boolean(verify.plannerModel);
	const minPromptChars = verify.planMinPromptChars ?? 80;
	const maxRejections = verify.maxRejections ?? 2;
	const planAfterRejections = verify.planAfterRejections ?? 1;
	const escalateAfterRejections = verify.escalateAfterRejections ?? 1;
	const verifierCommands = verify.verifierCommands ?? [];
	const maxCheckerRuns = verify.maxCheckerRuns ?? 2;
	const stickyEscalation = verify.stickyEscalation ?? true;
	const triageEnabled = verify.triage?.enabled ?? true;

	let finalRejections = 0;
	let checkpointRejections = 0;
	let checkpointsSoFar = 0;
	let alreadyPlanned = false;
	let makerEscalated = false;
	let tier: VerifyTier = "standard";
	const runCost = { calls: 0, tokensIn: 0, tokensOut: 0, costUsd: 0 };

	// A run is one user prompt. Without this reset, `alreadyPlanned` and the
	// checkpoint budget leak across tasks and every task after the first runs
	// with no plan and no mid-build audits.
	const resetRunState = (): void => {
		finalRejections = 0;
		checkpointRejections = 0;
		checkpointsSoFar = 0;
		alreadyPlanned = false;
		makerEscalated = false;
		tier = "standard";
		runCost.calls = 0;
		runCost.tokensIn = 0;
		runCost.tokensOut = 0;
		runCost.costUsd = 0;
	};

	const trackCost = (m: AssistantMessage): { tokensIn: number; tokensOut: number; costUsd: number } => {
		const u = usageFields(m);
		runCost.calls++;
		runCost.tokensIn += u.tokensIn;
		runCost.tokensOut += u.tokensOut;
		runCost.costUsd += u.costUsd;
		return u;
	};

	const emitRunSummary = (outcome: "verified" | "unverified"): void => {
		logVerify({
			event: "run-summary",
			tier,
			outcome,
			verifyCalls: runCost.calls,
			tokensIn: runCost.tokensIn,
			tokensOut: runCost.tokensOut,
			costUsd: Number(runCost.costUsd.toFixed(6)),
		});
	};

	interface TierView {
		skipPlanner: boolean;
		plannerRef?: string;
		plannerThinking: ThinkingLevel;
		maker: Model<any>;
		makerThinking: ThinkingLevel;
		everyN: number;
		escalationRef?: string;
		escalationThinking: ThinkingLevel;
	}

	const tierView = (): TierView => {
		const t = tierOverrides(verify, tier);
		if (tier === "trivial") {
			return {
				skipPlanner: t?.skipPlanner ?? true,
				plannerRef: t?.plannerModel ?? verify.plannerModel,
				plannerThinking: t?.plannerThinkingLevel ?? verify.plannerThinkingLevel ?? "max",
				maker: resolveModel(t?.makerModel) ?? makerModel,
				makerThinking: t?.makerThinkingLevel ?? "medium",
				everyN: t?.checkpointEveryToolTurns ?? 0,
				escalationRef: t?.escalationMakerModel ?? verify.escalationMakerModel,
				escalationThinking: t?.escalationMakerThinkingLevel ?? verify.escalationMakerThinkingLevel ?? "max",
			};
		}
		if (tier === "hard") {
			return {
				skipPlanner: t?.skipPlanner ?? false,
				plannerRef: t?.plannerModel ?? verify.strongPlannerModel ?? verify.plannerModel,
				plannerThinking:
					t?.plannerThinkingLevel ?? verify.strongPlannerThinkingLevel ?? verify.plannerThinkingLevel ?? "max",
				// Hard tasks start on the stronger maker: a doomed cheap-maker first
				// attempt costs a full attempt + audit + rejection cycle for nothing.
				maker: resolveModel(t?.makerModel ?? verify.escalationMakerModel) ?? makerModel,
				makerThinking: t?.makerThinkingLevel ?? "high",
				everyN: t?.checkpointEveryToolTurns ?? everyN,
				escalationRef: t?.escalationMakerModel ?? verify.escalationMakerModel,
				escalationThinking: t?.escalationMakerThinkingLevel ?? verify.escalationMakerThinkingLevel ?? "max",
			};
		}
		return {
			skipPlanner: false,
			plannerRef: verify.plannerModel,
			plannerThinking: verify.plannerThinkingLevel ?? "max",
			maker: makerModel,
			makerThinking: makerThinkingLevel,
			everyN,
			escalationRef: verify.escalationMakerModel,
			escalationThinking: verify.escalationMakerThinkingLevel ?? "max",
		};
	};

	const demoteMaker = (): { model: Model<any>; thinkingLevel: ThinkingLevel } => {
		makerEscalated = false;
		const view = tierView();
		return { model: view.maker, thinkingLevel: view.makerThinking };
	};

	const currentEscalation = (): { model: Model<any>; thinkingLevel: ThinkingLevel } => {
		const view = tierView();
		const escalation = resolveModel(view.escalationRef);
		if (escalation) return { model: escalation, thinkingLevel: view.escalationThinking };
		return { model: view.maker, thinkingLevel: view.makerThinking };
	};

	const chooseRetryMaker = (
		rejections: number,
	): { model: Model<any>; thinkingLevel: ThinkingLevel; escalated: boolean } => {
		const view = tierView();
		const escalation = resolveModel(view.escalationRef);
		// Escalating to the model already making is a no-op, not an escalation.
		const escalationDistinct = escalation && escalation.id !== view.maker.id;
		if (escalationDistinct && (makerEscalated || shouldEscalateMaker({ rejections, escalateAfterRejections }))) {
			makerEscalated = true;
			return { model: escalation, thinkingLevel: view.escalationThinking, escalated: true };
		}
		return { model: view.maker, thinkingLevel: view.makerThinking, escalated: false };
	};

	const execVerifierCommand = (command: string): string => {
		try {
			return execSync(command, {
				cwd,
				timeout: 120_000,
				encoding: "utf8",
				maxBuffer: 4 * 1024 * 1024,
				stdio: ["ignore", "pipe", "pipe"],
			}).slice(-4_000);
		} catch (error) {
			const err = error as { status?: number; stdout?: string | Buffer; stderr?: string | Buffer };
			return (
				`exit ${err.status ?? "?"}\n` +
				`${String(err.stdout ?? "").slice(-2_000)}\n${String(err.stderr ?? "").slice(-2_000)}`.trim()
			);
		}
	};

	const runCheckerAudit = async (
		vctx: VerifyTurnContext,
		kind: VerifyKind,
	): Promise<{ verdict: "verified" | "conflict" | "ambiguous"; auditText: string; ran: string | null }> => {
		const checkerRef = checkerRefForKind(verify, kind, tier);
		const activeChecker = resolveModel(checkerRef);
		if (!activeChecker) {
			return { verdict: "ambiguous", auditText: "", ran: null };
		}
		const thinkingLevel = checkerThinkingForKind(verify, kind, tier);

		const runInstruction = verifierCommands.length
			? ` If you need fresh evidence before judging, reply EXACTLY "RUN: <command>" with one of: ${verifierCommands.join(
					", ",
				)}. The harness — not the maker — runs it and you receive the raw output. You may request up to ${maxCheckerRuns} commands, one per reply.`
			: "";
		const auditPrompt =
			(kind === "checkpoint"
				? "You are an independent mid-build checker. You see the goal, recent tool receipts, the working-tree " +
					"diff summary, and the maker's latest progress — not the maker's private reasoning. Reply EXACTLY: " +
					"VERIFIED if progress is on track toward the goal (partial progress is OK). If the maker is stuck, " +
					"wrong, skipping required verification, or drifting, reply with the specific problem, then the word " +
					"CONFLICT."
				: "You are an independent checker. You are given the goal, the real tool receipts, the actual " +
					"working-tree diff, and the maker's final answer — not the maker's reasoning. Judge the artifact, " +
					"not the summary. Reply EXACTLY: VERIFIED only if the answer fully and correctly satisfies the goal " +
					"AND the receipts show real verification (tests/build/lint actually ran and passed). If the answer " +
					"is wrong, incomplete, or its claims are unverified, reply with the specific problem, then the word " +
					"CONFLICT.") +
			` ${EXCELLENCE_CHARTER}` +
			runInstruction;

		const firstUser = vctx.newMessages.find((m) => m.role === "user");
		const goalText = firstUser ? messageText(firstUser) : "";
		const receipts = vctx.newMessages
			.filter((m): m is ToolResultMessage => m.role === "toolResult")
			.slice(-10)
			.map((m) => `--- ${m.toolName} ---\n${messageText(m).slice(-1200)}`);

		const auditMessages: AgentMessage[] = [{ role: "user", content: `Goal:\n${goalText}`, timestamp: Date.now() }];
		if (receipts.length > 0) {
			auditMessages.push({
				role: "user",
				content: `Tool receipts (real outputs):\n\n${receipts.join("\n\n")}`,
				timestamp: Date.now(),
			});
		}
		const diffEvidence = gatherDiffEvidence(cwd, kind);
		if (diffEvidence) {
			auditMessages.push({
				role: "user",
				content: `Working-tree evidence (gathered by the harness, not the maker):\n\n${diffEvidence}`,
				timestamp: Date.now(),
			});
		}
		auditMessages.push({
			role: "user",
			content:
				kind === "checkpoint"
					? `Latest maker progress under review:\n${messageText(vctx.message)}`
					: `Candidate final answer under review:\n${messageText(vctx.message)}`,
			timestamp: Date.now(),
		});
		auditMessages.push({ role: "user", content: auditPrompt, timestamp: Date.now() });

		let audit = await vctx.runModel(activeChecker, auditMessages, { thinkingLevel });
		trackCost(audit);
		let auditText = messageText(audit);

		const ran: string[] = [];
		while (ran.length < maxCheckerRuns) {
			const requested = /^RUN:\s*(\S.*)$/m.exec(auditText)?.[1]?.trim();
			if (!requested || !verifierCommands.includes(requested) || ran.includes(requested)) break;
			ran.push(requested);
			const output = execVerifierCommand(requested);
			auditMessages.push({
				role: "user",
				content: `Output of \`${requested}\` (run by the harness, tail):\n${output}`,
				timestamp: Date.now(),
			});
			audit = await vctx.runModel(activeChecker, auditMessages, { thinkingLevel });
			trackCost(audit);
			auditText = messageText(audit);
		}

		let verdict = classifyCheckerVerdict(auditText);
		if (verdict === "ambiguous") {
			// One strict re-ask: an ambiguous verdict otherwise drops the audit
			// silently, wasting the checker call.
			auditMessages.push({
				role: "user",
				content:
					'Your reply did not contain a clear verdict. Reply EXACTLY "VERIFIED", or state the specific problem followed by the word "CONFLICT".',
				timestamp: Date.now(),
			});
			audit = await vctx.runModel(activeChecker, auditMessages, { thinkingLevel });
			trackCost(audit);
			auditText = messageText(audit);
			verdict = classifyCheckerVerdict(auditText);
		}

		logVerify({
			event: "audit",
			kind,
			tier,
			checker: activeChecker.id,
			verdict,
			ran: ran.length > 0 ? ran : null,
			stop: audit.stopReason,
			err: audit.errorMessage?.slice(0, 200) ?? null,
			text: auditText.slice(0, 240),
			...usageFields(audit),
		});

		return { verdict, auditText, ran: ran[0] ?? null };
	};

	const escalatePlanner = async (vctx: VerifyTurnContext, feedback: string): Promise<string> => {
		// Re-plans after rejection use the strong planner: recovery after the
		// cheap path failed is where the expensive model earns its price.
		const plannerRef = verify.strongPlannerModel ?? verify.plannerModel;
		const planner = resolveModel(plannerRef);
		if (!planner) return "";
		const plan = await vctx.runModel(
			planner,
			[
				...vctx.context.messages,
				{
					role: "user",
					content:
						`A checker rejected the previous work. Feedback: ${feedback.slice(0, 800)}\n\n` +
						"You are a senior planner. Do NOT answer the task yourself. Break the problem down into " +
						"a short, concrete, step-by-step plan that a maker model can execute to produce " +
						"a correct result. Address the checker's feedback and require a DIFFERENT strategy from " +
						"the rejected approach.",
					timestamp: Date.now(),
				},
			],
			{ thinkingLevel: verify.strongPlannerThinkingLevel ?? verify.plannerThinkingLevel ?? "max" },
		);
		trackCost(plan);
		return messageText(plan);
	};

	const rejectWithOptionalPlan = async (
		vctx: VerifyTurnContext,
		kind: VerifyKind,
		auditText: string,
		rejections: number,
	): Promise<TurnVerifyResult> => {
		let planText = "";
		if (rejections >= planAfterRejections) {
			planText = await escalatePlanner(vctx, auditText);
		}
		const retry = chooseRetryMaker(rejections);
		logVerify({
			event: "rejected",
			kind,
			tier,
			rejections,
			planned: planText.length > 0,
			planner: planText ? (verify.strongPlannerModel ?? verify.plannerModel) : null,
			makerEscalated: retry.escalated,
			retryMaker: retry.model.id,
		});
		return {
			status: "rejected",
			correctivePrompt: planText
				? `[MAKER/CHECKER/PLANNER] Your previous ${kind === "checkpoint" ? "progress" : "answer"} was rejected by the checker. Feedback: ${auditText.slice(0, 800)}. ` +
					`The planner produced this plan:\n${planText.slice(0, 2000)}\n` +
					"Execute the plan step by step; do not repeat the rejected approach."
				: `[MAKER/CHECKER/PLANNER] Your previous ${kind === "checkpoint" ? "progress" : "answer"} was rejected by the checker. Feedback: ${auditText.slice(0, 800)}. ` +
					`Rework ${kind === "checkpoint" ? "the next steps" : "the answer"} correctly; do not repeat the rejected approach.`,
			model: retry.model,
			thinkingLevel: retry.thinkingLevel,
		};
	};

	const triagePrompt = (promptText: string, signals: string): string =>
		"Classify this coding task by complexity. Reply with EXACTLY one word: TRIVIAL, STANDARD, or HARD.\n" +
		"TRIVIAL: a small, fully specified, low-risk change (typo, rename, config tweak, one-line fix) or a question needing no code change.\n" +
		"STANDARD: a normal feature, fix, or refactor across a few files with a clear approach.\n" +
		"HARD: multi-file or architectural work, tricky debugging, concurrency, data migrations, security-sensitive code, or anything where the approach itself is uncertain.\n\n" +
		`Task:\n${promptText.slice(0, 2_000)}` +
		(signals ? `\n\nRepo signals:\n${signals}` : "");

	const clarifyDirective = (questions: string): string =>
		"[PLANNER] Open questions block this goal. FIRST investigate the repository and environment yourself with " +
		"read-only tools and answer every question you can from evidence — never ask the user anything you can " +
		"discover. THEN ask the user only what remains and is genuinely theirs to answer (intent, scope trade-offs, " +
		"budgets, external access): at most 5 questions, one sentence each. Then STOP and wait for their answers — " +
		"do not write code and do not guess on the questions you relay.\n\nPlanner's open questions:\n\n" +
		questions;

	return {
		beforeFirstTurn: async (ctx) => {
			// Only a fresh user prompt starts a new run. Continues/retries carry no
			// new user message and must not reset state or re-plan mid-task.
			const firstUser = ctx.newMessages.find((m) => m.role === "user");
			if (!firstUser) return undefined;
			const promptText = messageText(firstUser as UserMessage);
			resetRunState();
			logVerify({ event: "run-start", chars: promptText.length });

			const signals = gatherRepoSignals(cwd);

			if (isConversationalPrompt(promptText)) {
				tier = "trivial";
				logVerify({ event: "triage", tier, via: "heuristic" });
			} else if (triageEnabled) {
				const triageModel = resolveModel(verify.triage?.model) ?? makerModel;
				try {
					const reply = await ctx.runModel(
						triageModel,
						[{ role: "user", content: triagePrompt(promptText, signals), timestamp: Date.now() }],
						{ thinkingLevel: "off" },
					);
					const cost = trackCost(reply);
					tier = classifyTriageTier(messageText(reply)) ?? "standard";
					logVerify({ event: "triage", tier, via: triageModel.id, ...cost });
				} catch (error) {
					tier = "standard";
					logVerify({
						event: "triage",
						tier,
						via: "triage-error",
						err: error instanceof Error ? error.message.slice(0, 200) : String(error).slice(0, 200),
					});
				}
			} else {
				tier = "standard";
			}

			const view = tierView();
			// Always route the maker/thinking for the tier, plan or no plan.
			const baseResult: BeforeFirstTurnResult = { model: view.maker, thinkingLevel: view.makerThinking };

			const planner = resolveModel(view.plannerRef);
			// With triage active the tier decides whether to plan; without it the
			// legacy prompt-length gate applies.
			const planGate =
				triageEnabled || tier === "trivial"
					? !view.skipPlanner && planFirst
					: shouldPlanFirst({ planFirst, promptText, minPromptChars, alreadyPlanned });
			if (!planner || !planGate) {
				logVerify({
					event: "plan-skipped",
					reason: !planner ? "no-planner" : tier === "trivial" ? "trivial-tier" : "plan-first-disabled",
					tier,
					chars: promptText.length,
				});
				return baseResult;
			}

			alreadyPlanned = true;
			try {
				const plan = await ctx.runModel(
					planner,
					[
						{
							role: "user",
							content:
								`Goal:\n${promptText}\n\n` +
								(signals ? `Repo signals:\n${signals}\n\n` : "") +
								"You are a senior planner with no tool access, but the maker executing your plan has FULL " +
								"tool access: it can read any file, search the repo, and run commands. PLAN BY DEFAULT.\n" +
								"Never ask the user anything discoverable from the repository or by running commands — make " +
								"investigating those facts the first steps of the plan instead. Audits, reviews, and " +
								"'figure out what to improve' goals are self-contained: discovery IS the work, so never " +
								"CLARIFY for them. Where the goal leaves room, state explicit assumptions in the plan and " +
								"proceed rather than asking.\n" +
								"Reply CLARIFY only when a decision is genuinely the user's to make (product intent, scope " +
								"trade-offs, spend approvals, external accounts/credentials) AND guessing wrong would waste " +
								"most of the work: first line CLARIFY, then AT MOST 5 numbered questions, one sentence each " +
								"with a short why-it-matters, no preamble.\n" +
								"Otherwise produce a short, concrete, step-by-step plan the maker can execute — " +
								"investigation steps first, assumptions stated explicitly, and how to verify " +
								"(tests/build/lint) at the end. Do NOT implement or claim the work is done.",
							timestamp: Date.now(),
						},
					],
					{ thinkingLevel: view.plannerThinking },
				);
				const cost = trackCost(plan);
				const planText = messageText(plan);
				if (!planText) {
					logVerify({ event: "plan-skipped", reason: "empty-plan", planner: view.plannerRef, tier });
					return baseResult;
				}
				if (/^\s*CLARIFY\b/i.test(planText)) {
					// Backstop: even a misbehaving planner cannot relay a wall of text.
					const questions = (planText.replace(/^\s*CLARIFY\W*/i, "").trim() || planText).slice(0, 2_000);
					logVerify({ event: "clarify", planner: planner.id, tier, text: questions.slice(0, 240), ...cost });
					return {
						...baseResult,
						messages: [{ role: "user", content: clarifyDirective(questions), timestamp: Date.now() }],
					};
				}
				logVerify({
					event: "planned",
					planner: planner.id,
					tier,
					chars: planText.length,
					text: planText.slice(0, 240),
					...cost,
				});
				return {
					...baseResult,
					messages: [
						{
							role: "user",
							content: `[PLANNER] Execute this plan step by step with tools. Do not skip verification steps.\n\n${planText.slice(0, 4000)}`,
							timestamp: Date.now(),
						},
					],
				};
			} catch (error) {
				logVerify({
					event: "plan-skipped",
					reason: "planner-error",
					tier,
					err: error instanceof Error ? error.message.slice(0, 200) : String(error).slice(0, 200),
				});
				return baseResult;
			}
		},

		shouldCheckpoint: ({ toolTurnCount }) =>
			shouldRunCheckpoint({
				toolTurnCount,
				everyN: tierView().everyN,
				checkpointsSoFar,
				maxCheckpoints,
				backoff,
			}),

		verifyTurn: async (vctx) => {
			const kind: VerifyKind = vctx.kind ?? "final";

			if (kind === "final" && verify.auditOnlyAfterTools && !vctx.newMessages.some((m) => m.role === "toolResult")) {
				logVerify({ event: "skipped", reason: "no-tools", kind, tier });
				return undefined;
			}

			if (kind === "checkpoint") {
				checkpointsSoFar++;
				logVerify({ event: "checkpoint", toolTurnCount: vctx.toolTurnCount ?? null, n: checkpointsSoFar, tier });
			}

			const { verdict, auditText } = await runCheckerAudit(vctx, kind);

			if (verdict === "verified") {
				if (kind === "final") finalRejections = 0;
				else checkpointRejections = 0;
				// Sticky escalation: once a run needed the stronger maker, demoting on
				// the next verified checkpoint invites fail→escalate→pass→demote→fail
				// ping-pong, which costs more than the stronger maker's tokens.
				const keepEscalated = stickyEscalation && makerEscalated && kind === "checkpoint";
				const next = keepEscalated ? currentEscalation() : demoteMaker();
				logVerify({ event: keepEscalated ? "kept-escalated" : "demoted", kind, tier, maker: next.model.id });
				if (kind === "final") emitRunSummary("verified");
				return {
					status: "verified",
					model: next.model,
					thinkingLevel: next.thinkingLevel,
				};
			}

			if (verdict !== "conflict" || auditText.length === 0) {
				logVerify({ event: "skipped", reason: "ambiguous-verdict", kind, tier });
				return undefined;
			}

			if (kind === "checkpoint") {
				checkpointRejections++;
				if (checkpointRejections > maxRejections) {
					logVerify({
						event: "budget-exhausted",
						kind,
						tier,
						rejections: checkpointRejections,
						accepted: "unverified",
					});
					checkpointRejections = 0;
					const base = demoteMaker();
					// Accept progress and keep building, but say so honestly.
					return {
						status: "verified",
						model: base.model,
						thinkingLevel: base.thinkingLevel,
						notice:
							"[VERIFY] UNVERIFIED — mid-build progress accepted on checker budget " +
							`(${maxRejections} rejections). Last checker feedback: ${auditText.slice(0, 400)}`,
					};
				}
				return rejectWithOptionalPlan(vctx, kind, auditText, checkpointRejections);
			}

			finalRejections++;
			if (finalRejections > maxRejections) {
				logVerify({ event: "budget-exhausted", kind, tier, rejections: finalRejections, accepted: "unverified" });
				finalRejections = 0;
				const base = demoteMaker();
				emitRunSummary("unverified");
				// Accept the final answer and stop (verified ends the loop), but the
				// user must see that the checker never signed off.
				return {
					status: "verified",
					model: base.model,
					thinkingLevel: base.thinkingLevel,
					notice:
						"[VERIFY] UNVERIFIED — final answer accepted on checker budget " +
						`(${maxRejections} rejections). Treat with suspicion. Last checker feedback: ${auditText.slice(0, 400)}`,
				};
			}
			return rejectWithOptionalPlan(vctx, kind, auditText, finalRejections);
		},
	};
}
