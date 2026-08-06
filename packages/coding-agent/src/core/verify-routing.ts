/**
 * Maker / checker / planner routing for mid-build and final verification.
 *
 * Phases:
 *   1. planFirst     — planner model decomposes the goal before the maker starts
 *   2. make          — base maker model executes with tools
 *   3. checkpoint    — every N tool turns, mid-build checker audits progress
 *   4. escalate      — after repeated rejects, retries use escalationMakerModel
 *   5. final         — final checker audits the no-tool-call answer; planner may re-plan
 *
 * Planner and checkers never execute tools. Only base/escalation makers do.
 */

import { execSync } from "node:child_process";
import { appendFileSync } from "node:fs";
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
import type { VerifySettings } from "./settings-manager.ts";

export type VerifyKind = "final" | "checkpoint";

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
}): boolean {
	const { toolTurnCount, everyN, checkpointsSoFar, maxCheckpoints } = opts;
	if (everyN <= 0) return false;
	if (maxCheckpoints <= 0) return false;
	if (checkpointsSoFar >= maxCheckpoints) return false;
	if (toolTurnCount <= 0) return false;
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
 * Prefer CONFLICT when present. Never treat the substring inside "unverified" as VERIFIED.
 */
export function classifyCheckerVerdict(auditText: string): "verified" | "conflict" | "ambiguous" {
	const text = auditText.trim();
	if (!text) return "ambiguous";
	if (/\bCONFLICT\b/i.test(text)) return "conflict";
	if (/\bunverified\b/i.test(text)) return "ambiguous";
	if (/\bVERIFIED\b/i.test(text)) return "verified";
	return "ambiguous";
}

/** Pure: whether retries should use the escalated maker. */
export function shouldEscalateMaker(opts: { rejections: number; escalateAfterRejections: number }): boolean {
	const after = opts.escalateAfterRejections;
	if (after <= 0) return false;
	return opts.rejections >= after;
}

export function checkerRefForKind(verify: VerifySettings, kind: VerifyKind): string | undefined {
	if (kind === "checkpoint") {
		return verify.checkpointCheckerModel ?? verify.checkerModel;
	}
	return verify.checkerModel;
}

export function checkerThinkingForKind(verify: VerifySettings, kind: VerifyKind): ThinkingLevel | undefined {
	if (kind === "checkpoint") {
		return verify.checkpointCheckerThinkingLevel ?? verify.checkerThinkingLevel;
	}
	return verify.checkerThinkingLevel;
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

export function createVerifyRouting(options: CreateVerifyRoutingOptions): VerifyRoutingHooks | undefined {
	const { verify, makerModel, makerThinkingLevel, cwd, resolveModel } = options;
	const finalChecker = resolveModel(verify.checkerModel);
	if (!verify.checkerModel || !finalChecker) return undefined;

	const logVerify = createVerifyAuditLogger(options.auditLogPath);
	const everyN = verify.checkpointEveryToolTurns ?? 0;
	const maxCheckpoints = verify.maxCheckpointsPerRun ?? 3;
	const planFirst = verify.planFirst ?? Boolean(verify.plannerModel);
	const minPromptChars = verify.planMinPromptChars ?? 80;
	const maxRejections = verify.maxRejections ?? 2;
	const planAfterRejections = verify.planAfterRejections ?? 1;
	const escalateAfterRejections = verify.escalateAfterRejections ?? 1;
	const verifierCommands = verify.verifierCommands ?? [];

	let finalRejections = 0;
	let checkpointRejections = 0;
	let checkpointsSoFar = 0;
	let alreadyPlanned = false;
	let makerEscalated = false;

	const demoteMaker = (): { model: Model<any>; thinkingLevel: ThinkingLevel } => {
		makerEscalated = false;
		return { model: makerModel, thinkingLevel: makerThinkingLevel };
	};

	const chooseRetryMaker = (
		rejections: number,
	): { model: Model<any>; thinkingLevel: ThinkingLevel; escalated: boolean } => {
		const escalation = resolveModel(verify.escalationMakerModel);
		if (escalation && shouldEscalateMaker({ rejections, escalateAfterRejections })) {
			makerEscalated = true;
			return {
				model: escalation,
				thinkingLevel: verify.escalationMakerThinkingLevel ?? "max",
				escalated: true,
			};
		}
		if (makerEscalated && escalation) {
			return {
				model: escalation,
				thinkingLevel: verify.escalationMakerThinkingLevel ?? "max",
				escalated: true,
			};
		}
		return { model: makerModel, thinkingLevel: makerThinkingLevel, escalated: false };
	};

	const runCheckerAudit = async (
		vctx: VerifyTurnContext,
		kind: VerifyKind,
	): Promise<{ verdict: "verified" | "conflict" | "ambiguous"; auditText: string; ran: string | null }> => {
		const checkerRef = checkerRefForKind(verify, kind);
		const activeChecker = resolveModel(checkerRef);
		if (!activeChecker) {
			return { verdict: "ambiguous", auditText: "", ran: null };
		}
		const thinkingLevel = checkerThinkingForKind(verify, kind);

		const auditPrompt =
			kind === "checkpoint"
				? "You are an independent mid-build checker. You see the goal, recent tool receipts, and the maker's latest " +
					"progress — not the maker's private reasoning. Reply EXACTLY: VERIFIED if progress is on track toward the " +
					"goal (partial progress is OK). If the maker is stuck, wrong, skipping required verification, or drifting, " +
					"reply with the specific problem, then the word CONFLICT." +
					(verifierCommands.length
						? ` If you need fresh evidence, reply EXACTLY "RUN: <command>" with one of: ${verifierCommands.join(", ")}.`
						: "")
				: "You are an independent checker. You are given the goal, the real tool receipts, and the maker's " +
					"final answer — not the maker's reasoning. Reply EXACTLY: VERIFIED only if the answer fully and " +
					"correctly satisfies the goal AND the receipts show real verification (tests/build/lint actually " +
					"ran and passed). If the answer is wrong, incomplete, or its claims are unverified, reply with " +
					"the specific problem, then the word CONFLICT." +
					(verifierCommands.length
						? ` If you need fresh evidence before judging, reply EXACTLY "RUN: <command>" with one of: ${verifierCommands.join(", ")}. The harness — not the maker — runs it and you receive the raw output.`
						: "");

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
		let auditText = messageText(audit);

		const requested = /^RUN:\s*(\S.*)$/m.exec(auditText)?.[1]?.trim();
		if (requested && verifierCommands.includes(requested)) {
			let output: string;
			try {
				output = execSync(requested, {
					cwd,
					timeout: 120_000,
					encoding: "utf8",
					maxBuffer: 4 * 1024 * 1024,
					stdio: ["ignore", "pipe", "pipe"],
				}).slice(-4000);
			} catch (error) {
				const err = error as { status?: number; stdout?: string | Buffer; stderr?: string | Buffer };
				output =
					`exit ${err.status ?? "?"}\n` +
					`${String(err.stdout ?? "").slice(-2000)}\n${String(err.stderr ?? "").slice(-2000)}`.trim();
			}
			audit = await vctx.runModel(
				activeChecker,
				[
					...auditMessages,
					{
						role: "user",
						content: `Output of \`${requested}\` (run by the harness, tail):\n${output}`,
						timestamp: Date.now(),
					},
				],
				{ thinkingLevel },
			);
			auditText = messageText(audit);
		}

		const verdict = classifyCheckerVerdict(auditText);

		logVerify({
			event: "audit",
			kind,
			checker: activeChecker.id,
			verdict,
			ran: requested ?? null,
			stop: audit.stopReason,
			err: audit.errorMessage?.slice(0, 200) ?? null,
			text: auditText.slice(0, 240),
		});

		return { verdict, auditText, ran: requested ?? null };
	};

	const escalatePlanner = async (vctx: VerifyTurnContext, feedback: string): Promise<string> => {
		const planner = resolveModel(verify.plannerModel);
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
						"a correct result. Address the checker's feedback.",
					timestamp: Date.now(),
				},
			],
			{ thinkingLevel: verify.plannerThinkingLevel ?? "max" },
		);
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
			rejections,
			planned: planText.length > 0,
			planner: planText ? verify.plannerModel : null,
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

	return {
		beforeFirstTurn: async (ctx) => {
			const planner = resolveModel(verify.plannerModel);
			if (!planner) return undefined;

			const firstUser =
				ctx.newMessages.find((m) => m.role === "user") ?? ctx.context.messages.find((m) => m.role === "user");
			const promptText = firstUser ? messageText(firstUser as UserMessage) : "";
			if (!shouldPlanFirst({ planFirst, promptText, minPromptChars, alreadyPlanned })) {
				logVerify({
					event: "plan-skipped",
					reason: alreadyPlanned ? "already-planned" : "prompt-too-short",
					chars: promptText.length,
				});
				return undefined;
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
								"You are a senior planner. Do NOT implement or claim the work is done. Produce a short, concrete, " +
								"step-by-step plan a maker model can execute. Include how to verify (tests/build/lint) at the end.",
							timestamp: Date.now(),
						},
					],
					{ thinkingLevel: verify.plannerThinkingLevel ?? "max" },
				);
				const planText = messageText(plan);
				if (!planText) {
					logVerify({ event: "plan-skipped", reason: "empty-plan", planner: verify.plannerModel });
					return undefined;
				}
				logVerify({
					event: "planned",
					planner: planner.id,
					chars: planText.length,
					text: planText.slice(0, 240),
				});
				const base = demoteMaker();
				return {
					messages: [
						{
							role: "user",
							content: `[PLANNER] Execute this plan step by step with tools. Do not skip verification steps.\n\n${planText.slice(0, 4000)}`,
							timestamp: Date.now(),
						},
					],
					model: base.model,
					thinkingLevel: base.thinkingLevel,
				};
			} catch (error) {
				logVerify({
					event: "plan-skipped",
					reason: "planner-error",
					err: error instanceof Error ? error.message.slice(0, 200) : String(error).slice(0, 200),
				});
				return undefined;
			}
		},

		shouldCheckpoint: ({ toolTurnCount }) =>
			shouldRunCheckpoint({
				toolTurnCount,
				everyN,
				checkpointsSoFar,
				maxCheckpoints,
			}),

		verifyTurn: async (vctx) => {
			const kind: VerifyKind = vctx.kind ?? "final";

			if (kind === "final" && verify.auditOnlyAfterTools && !vctx.newMessages.some((m) => m.role === "toolResult")) {
				logVerify({ event: "skipped", reason: "no-tools", kind });
				return undefined;
			}

			if (kind === "checkpoint") {
				checkpointsSoFar++;
				logVerify({ event: "checkpoint", toolTurnCount: vctx.toolTurnCount ?? null, n: checkpointsSoFar });
			}

			const { verdict, auditText } = await runCheckerAudit(vctx, kind);

			if (verdict === "verified") {
				if (kind === "final") finalRejections = 0;
				else checkpointRejections = 0;
				const base = demoteMaker();
				logVerify({ event: "demoted", kind, maker: base.model.id });
				return {
					status: "verified",
					model: base.model,
					thinkingLevel: base.thinkingLevel,
				};
			}

			if (verdict !== "conflict" || auditText.length === 0) {
				logVerify({ event: "skipped", reason: "ambiguous-verdict", kind });
				return undefined;
			}

			if (kind === "checkpoint") {
				checkpointRejections++;
				if (checkpointRejections > maxRejections) {
					logVerify({ event: "budget-exhausted", kind, rejections: checkpointRejections });
					checkpointRejections = 0;
					const base = demoteMaker();
					// Accept progress and keep building on the base maker.
					return { status: "verified", model: base.model, thinkingLevel: base.thinkingLevel };
				}
				return rejectWithOptionalPlan(vctx, kind, auditText, checkpointRejections);
			}

			finalRejections++;
			if (finalRejections > maxRejections) {
				logVerify({ event: "budget-exhausted", kind, rejections: finalRejections });
				finalRejections = 0;
				const base = demoteMaker();
				// Accept the final answer as-is and stop (verified ends the loop).
				return { status: "verified", model: base.model, thinkingLevel: base.thinkingLevel };
			}
			return rejectWithOptionalPlan(vctx, kind, auditText, finalRejections);
		},
	};
}
