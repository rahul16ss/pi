/**
 * Maker / checker / planner routing for mid-build and final verification.
 *
 * Architecture: roles are stable, models are plugs. Independence is a
 * family rule with a required third-family spare checker. When the live
 * maker collides with the primary checker family (session cycle, maker
 * fallback, sticky escalation), the spare is promoted so the maker never
 * grades its own work.
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

import { execFileSync, execSync } from "node:child_process";
import { appendFileSync, lstatSync, readFileSync, realpathSync, type Stats } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";
import type {
	AgentLoopTurnUpdate,
	AgentMessage,
	BeforeFirstTurnContext,
	BeforeFirstTurnResult,
	ShouldStopAfterTurnContext,
	ThinkingLevel,
	TurnErrorContext,
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
import { isSafeVerifierCommand, sanitizeVerifierCommands } from "./verifier-commands.ts";

/**
 * Rough token estimate: ~4 characters per token. Deliberately conservative
 * (overestimates slightly) so we truncate before the provider rejects us.
 */
const CHARS_PER_TOKEN = 4;
/** Tokens reserved for the model's response + system prompt overhead. */
const RESPONSE_RESERVE_TOKENS = 8192;

/**
 * Bound a message list to fit the target model's context window so cross-model
 * routing never hits a "prompt tokens limit exceeded" rejection (audit F-17).
 * Keeps the most recent messages and, if the original list started with a user
 * goal, preserves that goal at the front so the model never loses the task.
 * Pure; exported for tests.
 */
export function boundContextForModel(messages: AgentMessage[], model: Model<any>): AgentMessage[] {
	const maxInputTokens = (model.contextWindow ?? 1_000_000) - RESPONSE_RESERVE_TOKENS;
	const maxChars = Math.max(0, maxInputTokens) * CHARS_PER_TOKEN;
	let totalChars = 0;
	for (const m of messages) {
		totalChars += messageText(m).length + 16; // +16 for role/overhead
	}
	if (totalChars <= maxChars) return messages; // fits, no truncation needed

	// Over budget: keep the first user message (the goal) + the most recent
	// messages that fit. Drop from the front (oldest) after the goal.
	const firstUser = messages.find((m) => m.role === "user");
	const goal = firstUser ? [firstUser] : [];
	const goalChars = goal.reduce((sum, m) => sum + messageText(m).length + 16, 0);
	const remainingChars = maxChars - goalChars;
	if (remainingChars <= 0) {
		// Even the goal doesn't fit — return just the goal, truncated to the
		// window with a visible marker. A whole message that exceeds the
		// window must not be forwarded verbatim (the "single 20k message"
		// defect): the helper's own comment promised truncation.
		if (goal.length === 0) return [];
		const goalTruncationMarker = "\n[... TRUNCATED: goal exceeded the checker context window and was shortened ...]";
		const budget = Math.max(0, maxChars - goalTruncationMarker.length - 1);
		const truncatedGoalText = messageText(goal[0]).slice(0, budget);
		return [
			{
				...(goal[0] as UserMessage),
				content: truncatedGoalText + goalTruncationMarker,
			} as UserMessage,
		];
	}
	const recent: AgentMessage[] = [];
	let recentChars = 0;
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i] === firstUser) continue; // already in goal
		const msgChars = messageText(messages[i]).length + 16;
		if (recentChars + msgChars > remainingChars) break;
		recent.unshift(messages[i]);
		recentChars += msgChars;
	}
	return [...goal, ...recent];
}

export type VerifyKind = "final" | "checkpoint";
export type VerifyTier = "trivial" | "standard" | "hard";

/**
 * F-02/P1: Redact patterns that look like secrets/credentials before evidence
 * is forwarded to external checker/planner models. This is a best-effort
 * regex pass, not a security boundary — it catches the common shapes (API
 * keys, tokens, passwords in env output, .env file contents) so accidental
 * exposure doesn't cross provider boundaries. Pure; exported for tests.
 */
export function redactEvidence(text: string): string {
	let redacted = text;
	// OpenRouter API keys: sk-or-v1-<64 chars>
	redacted = redacted.replace(/sk-or-v[0-9]+-[A-Za-z0-9_-]{20,}/g, "[REDACTED_OPENROUTER_KEY]");
	// OpenAI-style API keys: sk-<48+ chars>
	redacted = redacted.replace(/sk-[A-Za-z0-9]{40,}/g, "[REDACTED_OPENAI_KEY]");
	// GitHub personal access / fine-grained tokens: ghp_ / gho_ / ghu_ / ghs_ / github_pat_...
	redacted = redacted.replace(/gh[pousr]_[A-Za-z0-9]{20,}/g, "[REDACTED_GITHUB_KEY]");
	redacted = redacted.replace(/github_pat_[A-Za-z0-9_]{20,}/g, "[REDACTED_GITHUB_KEY]");
	// xAI API keys: xai-<...>
	redacted = redacted.replace(/xai-[A-Za-z0-9_-]{20,}/g, "[REDACTED_XAI_KEY]");
	// API keys in key=value or key: value format. Capture the key name so the
	// replacement is `api_key=[REDACTED]`, not a literal `$1 [REDACTED]`.
	redacted = redacted.replace(
		/(api[_-]?key|token|secret|password|auth)\s*[:=]\s*['"]?[A-Za-z0-9_-]{32,}['"]?/gi,
		"$1=[REDACTED]",
	);
	// Generic long token patterns (40+ alphanumeric chars on their own).
	// Git SHA-1 (40 hex) and SHA-256 (64 hex) are ubiquitous in diff evidence
	// and must survive — they are not secrets.
	redacted = redacted.replace(/(?<=[:=\s])['"]?([A-Za-z0-9_-]{40,})['"]?(?=\s|$)/g, (match, token: string) =>
		/^[0-9a-f]{40}$/i.test(token) || /^[0-9a-f]{64}$/i.test(token) ? match : "[REDACTED]",
	);
	// AWS-style keys (AKIA...)
	redacted = redacted.replace(/AKIA[0-9A-Z]{16}/g, "[REDACTED_AWS_KEY]");
	// .env-style lines with KEY=value where value looks sensitive
	redacted = redacted.replace(/^(\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)\w*)\s*=\s*\S+$/gim, "$1=[REDACTED]");
	return redacted;
}

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
	onTurnError: (ctx: TurnErrorContext) => Promise<AgentLoopTurnUpdate | undefined>;
	/** Every turn, including tool turns that skip verifyTurn. */
	shouldStopAfterTurn: (ctx: ShouldStopAfterTurnContext) => boolean;
}

/** A verifier reply that is a provider failure, not a verdict. */
export function isErrorReply(m: AssistantMessage): boolean {
	return m.stopReason === "error" || Boolean(m.errorMessage && m.errorMessage.trim().length > 0);
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

export type CheckerVerdict = "verified" | "conflict" | "ambiguous";

/**
 * The machine-readable verdict contract every checker prompt now demands.
 * A dedicated first line makes the verdict a parsed field instead of something
 * inferred from prose, which is what caused false rejections (see below).
 */
export const CHECKER_VERDICT_LINE_INSTRUCTION =
	'Your FIRST line must be exactly "VERDICT: VERIFIED" or "VERDICT: CONFLICT" and nothing else. ' +
	"Put all reasoning, quoted rules, and evidence AFTER that line. Never write the words VERIFIED or " +
	"CONFLICT on the first line for any other purpose.";

/** Exact-match enum on an isolated verdict line. No prose scanning. */
const VERDICT_LINE = /^\s*(?:[*_#>\s-]*)VERDICT\s*[:=-]\s*\**\s*(VERIFIED|CONFLICT)\b\s*\**\s*\.?\s*$/;
/** A first line that is *only* the bare pass token (a very common checker style). */
const BARE_VERIFIED_LINE = /^\s*(?:[*_#>\s-]*)VERIFIED\b[\s!.*_]*$/;
/** A first line that is *only* the bare rejection token. */
const BARE_CONFLICT_LINE = /^\s*(?:[*_#>\s-]*)CONFLICT\b[\s!.*_]*$/;
/**
 * A first line that LEADS with the pass token and then continues as rationale
 * on the same line (e.g. "VERIFIED. The receipts substantiate the report.").
 * This is still the checker's verdict — the first word of its first line — not
 * evidence. Matches the live 2026-08-13 false-conflict event that the isolated-
 * line rules missed because the prose was on the same line as VERIFIED, not on
 * a later line. The sentence-terminator anchor prevents matching "VERIFIED is
 * required" or "VERIFIED claims" (no terminator after the token) — those are
 * discussion, not a verdict.
 */
const LEADING_VERIFIED_LINE = /^\s*(?:[*_#>\s-]*)\**\s*VERIFIED\b\s*\**\s*[.!—\-–]+\**\s*(?:\s|$)/;
const LEADING_CONFLICT_LINE = /^\s*(?:[*_#>\s-]*)\**\s*CONFLICT\b\s*\**\s*[.!—\-–]+\**\s*(?:\s|$)/;

/** First line with visible content, ignoring blank/decoration-only lines. */
function firstMeaningfulLine(text: string): string {
	for (const line of text.split("\n")) {
		const trimmed = line.trim();
		if (trimmed.length === 0) continue;
		if (/^[-=*_#\s]+$/.test(trimmed)) continue; // markdown rule / decoration
		return trimmed;
	}
	return "";
}

/**
 * Classify a checker reply.
 *
 * Three-stage, deliberately ordered:
 *
 *  1. STRUCTURED (authoritative): an isolated first-line verdict — either
 *     "VERDICT: <enum>" or a line containing only the bare token — is taken as
 *     the verdict verbatim. The rest of the reply is rationale and is NEVER
 *     scanned. This is what fixes the false-conflict defect: a checker that
 *     opens with VERIFIED and later quotes the rejection criterion ("the
 *     CONFLICT criterion was not met") used to be recorded as a rejection,
 *     wasting a full maker+checker cycle. Observed live on 2026-08-13; see
 *     docs/audit/self-architecture-audit.md finding F-05.
 *
 *  2. LEADING-TOKEN (authoritative for same-line rationale): a first line that
 *     leads with VERIFIED/CONFLICT followed by a sentence terminator and more
 *     prose on the same line (e.g. "VERIFIED. The receipts substantiate...").
 *     The token opening the checker's own first line is the verdict; the rest
 *     is explanation. This closes the gap the isolated-line rules miss when
 *     the rationale is on the same line as the token, not on a later line. The
 *     sentence-terminator anchor prevents matching "VERIFIED is required" or
 *     "VERIFIED claims" (discussion, not a verdict).
 *
 *  3. LEGACY SCAN (compatibility only): replies with no isolated or leading
 *     verdict token fall back to the historical prose scan, because the
 *     pre-existing protocol asked for "the specific problem, then the word
 *     CONFLICT" — i.e. the verdict trails the rationale. Dropping that would
 *     turn real rejections into ambiguous verdicts, which fail open. CONFLICT
 *     still wins here, so an injected VERIFIED inside evidence cannot
 *     manufacture a pass.
 *
 * Never treats the substring inside "unverified" as VERIFIED, and never treats a
 * negated VERIFIED ("cannot be VERIFIED") as a pass. Positive matching is
 * case-sensitive: prose like "I verified the tests" stays ambiguous.
 */
export function classifyCheckerVerdict(auditText: string): CheckerVerdict {
	return parseCheckerVerdict(auditText).verdict;
}

/**
 * Same classification as `classifyCheckerVerdict`, plus which stage decided it.
 * `via` is logged so the improvement loop can measure structured-format
 * adoption instead of guessing whether checkers follow the contract.
 */
export function parseCheckerVerdict(auditText: string): {
	verdict: CheckerVerdict;
	via: "verdict-line" | "bare-token-line" | "leading-token-line" | "legacy-scan" | "empty";
} {
	const text = auditText.trim();
	if (!text) return { verdict: "ambiguous", via: "empty" };

	const first = firstMeaningfulLine(text);

	const tagged = VERDICT_LINE.exec(first);
	if (tagged) {
		return { verdict: tagged[1] === "VERIFIED" ? "verified" : "conflict", via: "verdict-line" };
	}
	if (BARE_VERIFIED_LINE.test(first)) return { verdict: "verified", via: "bare-token-line" };
	if (BARE_CONFLICT_LINE.test(first)) return { verdict: "conflict", via: "bare-token-line" };
	// A first line that leads with the token then continues as rationale on the
	// same line. This closes the F-05 gap for replies like
	// "VERIFIED. <rationale mentioning CONFLICT>" that the isolated-line rules
	// miss because the prose isn't on a later line. The token opening the
	// checker's own first line is the verdict; the rest is explanation.
	if (LEADING_VERIFIED_LINE.test(first)) return { verdict: "verified", via: "leading-token-line" };
	if (LEADING_CONFLICT_LINE.test(first)) return { verdict: "conflict", via: "leading-token-line" };

	if (/\bCONFLICT\b/i.test(text)) return { verdict: "conflict", via: "legacy-scan" };
	if (/\bun-?verified\b/i.test(text)) return { verdict: "ambiguous", via: "legacy-scan" };
	if (/\b(?:not|cannot|can't|isn't|is not|never|won't|no)\b[^.\n]{0,60}?\bVERIFIED\b/i.test(text)) {
		return { verdict: "ambiguous", via: "legacy-scan" };
	}
	if (/\bVERIFIED\b/.test(text)) return { verdict: "verified", via: "legacy-scan" };
	return { verdict: "ambiguous", via: "legacy-scan" };
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

/**
 * Ask the triage model whether this message is small talk or real work.
 * Greetings, thanks, and follow-ups are the model's job — not a word list.
 */
export function buildTriagePrompt(opts: { promptText: string; signals?: string; goalActive?: boolean }): string {
	const goalNote = opts.goalActive
		? "A coding goal is already in progress. If this message continues that work or answers a question about it, pick STANDARD or HARD for that work. If this message is only closing the conversation, pick TRIVIAL.\n\n"
		: "";
	const signals = opts.signals?.trim() ? `\n\nRepo signals:\n${opts.signals}` : "";
	return (
		"Classify this user message. Reply with EXACTLY one word: TRIVIAL, STANDARD, or HARD.\n" +
		"TRIVIAL: greetings and small talk with no coding request; or a tiny fully specified change (typo, rename, one-line fix).\n" +
		"STANDARD: a normal feature, fix, refactor, or a real question about the code or design.\n" +
		"HARD: multi-file or architectural work, tricky debugging, concurrency, migrations, security-sensitive code, or an uncertain approach.\n" +
		"A greeting is TRIVIAL even when it uses the assistant's name. Do not treat small talk as a planning task.\n" +
		goalNote +
		`Message:\n${opts.promptText.slice(0, 2_000)}` +
		signals
	);
}

function tierOverrides(verify: VerifySettings, tier: VerifyTier): VerifyTierOverrides | undefined {
	if (tier === "trivial") return verify.tiers?.trivial;
	if (tier === "standard") return verify.tiers?.standard;
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

/**
 * Company/family of a model ref. Strips the `openrouter/` provider prefix so
 * `openrouter/openai/gpt-5.6-luna` and `openai/gpt-5.6-luna` are the same
 * family. Roles in this harness are families, not model ids — plugs can be
 * swapped whenever a better model ships.
 */
export function extractModelFamily(ref: string): string {
	return (
		ref
			.replace(/^openrouter\//, "")
			.split("/")[0]
			?.toLowerCase() ?? ""
	);
}

export function familiesDiffer(a: string, b: string): boolean {
	const fa = extractModelFamily(a);
	const fb = extractModelFamily(b);
	return Boolean(fa && fb && fa !== fb);
}

export type AuditCheckerPick = {
	primaryRef: string;
	fallbackRef?: string;
	swapped: boolean;
	reason?: string;
};

/**
 * Pick the checker (and optional availability spare) for one audit.
 *
 * Architecture, not lineup: the spare checker is a required third family.
 * If the live maker collides with the configured checker family (session
 * cycle, maker-fallback, sticky escalation), the spare becomes the primary
 * checker so the maker never grades its own work. The availability fallback
 * is dropped when it is the same family as the live maker or the (possibly
 * swapped) primary checker.
 */
export function selectAuditChecker(opts: {
	liveMakerRef: string;
	checkerRef: string;
	checkerFallbackRef?: string;
}): AuditCheckerPick {
	const makerFam = extractModelFamily(opts.liveMakerRef);
	const checkerFam = extractModelFamily(opts.checkerRef);
	const fallbackFam = opts.checkerFallbackRef ? extractModelFamily(opts.checkerFallbackRef) : "";

	let primaryRef = opts.checkerRef;
	let fallbackRef = opts.checkerFallbackRef;
	let swapped = false;
	let reason: string | undefined;

	if (makerFam && checkerFam && makerFam === checkerFam) {
		if (fallbackFam && fallbackFam !== makerFam) {
			primaryRef = opts.checkerFallbackRef as string;
			fallbackRef = undefined;
			swapped = true;
			reason = "maker-family-collides-with-checker";
		}
	}

	const primaryFam = extractModelFamily(primaryRef);
	if (fallbackRef) {
		const fbFam = extractModelFamily(fallbackRef);
		if (!fbFam || fbFam === makerFam || fbFam === primaryFam) {
			fallbackRef = undefined;
		}
	}

	return { primaryRef, fallbackRef, swapped, reason };
}

export function checkerFallbackThinkingForKind(
	verify: VerifySettings,
	kind: VerifyKind,
	tier: VerifyTier = "standard",
): ThinkingLevel | undefined {
	const overrides = tierOverrides(verify, tier);
	return (
		overrides?.checkerFallbackThinkingLevel ??
		verify.checkerFallbackThinkingLevel ??
		checkerThinkingForKind(verify, kind, tier)
	);
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

/** Count maker turns, tool results, and maker cost from the live trace — not by incrementing. */
export function tallyMakerTrace(
	messages: AgentMessage[],
	current?: AgentMessage,
): { makerTurns: number; toolCalls: number; tokensIn: number; tokensOut: number; costUsd: number } {
	const seen = new Set<AgentMessage>();
	const list: AgentMessage[] = [];
	for (const m of messages) {
		if (!seen.has(m)) {
			seen.add(m);
			list.push(m);
		}
	}
	if (current && !seen.has(current)) list.push(current);
	let makerTurns = 0;
	let toolCalls = 0;
	let tokensIn = 0;
	let tokensOut = 0;
	let costUsd = 0;
	for (const m of list) {
		if (m.role === "toolResult") toolCalls++;
		if (m.role === "assistant") {
			makerTurns++;
			const u = usageFields(m as AssistantMessage);
			tokensIn += u.tokensIn;
			tokensOut += u.tokensOut;
			costUsd += u.costUsd;
		}
	}
	return { makerTurns, toolCalls, tokensIn, tokensOut, costUsd };
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

/** Cap untracked reads so a FIFO, device, or huge file cannot hang or OOM the checker. */
export const UNTRACKED_FILE_MAX_BYTES = 1_048_576;

function isUnsafeUntrackedTarget(stat: Stats): boolean {
	return stat.isFIFO() || stat.isSocket() || stat.isCharacterDevice() || stat.isBlockDevice() || stat.isDirectory();
}

/** Read an untracked file without interpolating its name into a shell command.
 * Resolves symlinks so an in-repo link targeting a file outside the repo
 * cannot leak external content to the checker. Skips FIFOs, devices, sockets,
 * and directories. Oversized regular files return a truncation marker instead
 * of being read in full. */
export function readUntrackedFile(cwd: string, file: string): string {
	if (!file || file.includes("\0") || isAbsolute(file)) return "";
	const resolved = resolve(cwd, file);
	const rel = relative(cwd, resolved);
	if (!rel || rel.startsWith("..") || isAbsolute(rel)) return "";
	try {
		const linkStat = lstatSync(resolved);
		if (isUnsafeUntrackedTarget(linkStat)) return "";
		const real = realpathSync(resolved);
		const realRel = relative(realpathSync(cwd), real);
		if (!realRel || realRel.startsWith("..") || isAbsolute(realRel)) return "";
		const targetStat = lstatSync(real);
		if (!targetStat.isFile() || isUnsafeUntrackedTarget(targetStat)) return "";
		if (targetStat.size > UNTRACKED_FILE_MAX_BYTES) {
			return `[... ${Number(targetStat.size)} more chars ...]\n`;
		}
		return readFileSync(real, { encoding: "utf8" });
	} catch {
		return "";
	}
}

function headLines(text: string, n: number): string {
	return text.split("\n").slice(0, n).join("\n");
}

/**
 * F-06: Inject a plan into the maker prompt without silently truncating
 * verification steps. If the plan exceeds the budget, the last section
 * (which typically contains verification steps) is moved to the front so
 * it's never cut. A visible truncation marker is added when the plan is
 * shortened. Pure; exported for tests.
 */
const PLAN_INJECT_BUDGET = 4_000;

export function injectPlan(planText: string, header?: string): string {
	const prefix = header ?? "[PLANNER] Execute this plan step by step with tools. Do not skip verification steps.\n\n";
	if (planText.length <= PLAN_INJECT_BUDGET) {
		return prefix + planText;
	}
	// Plan is too long. Try to preserve verification steps by splitting on
	// common section markers and keeping the last section (verification) + the
	// first section (investigation) at the expense of middle detail.
	const lines = planText.split("\n");
	// Find a verification section (usually near the end, marked by "verify",
	// "test", "check", "lint", "build" in a heading-like line)
	let verifyStart = -1;
	for (let i = lines.length - 1; i >= 0; i--) {
		if (/^(?:\d+[.)]\s*)?(?:verify|test|check|lint|build|run)\b/i.test(lines[i].trim())) {
			verifyStart = i;
			break;
		}
	}
	if (verifyStart > 0) {
		const verification = lines.slice(verifyStart).join("\n");
		const remaining = PLAN_INJECT_BUDGET - verification.length - 200; // 200 for header + marker
		if (remaining > 200) {
			const beginning = lines.slice(0, Math.max(1, Math.floor(remaining / 80))).join("\n");
			return (
				prefix +
				beginning +
				"\n[... plan detail truncated to fit context budget; verification steps preserved ...]\n" +
				verification
			);
		}
		// Verification section itself exceeds the budget. Keep as much of it as
		// fits — never drop it in favor of the plan head, which is the contract
		// this function advertises ("verification steps preserved").
		const oversizedMarker = "\n[... TRUNCATED: verification section exceeded context budget and was shortened ...]";
		const budget = Math.max(0, PLAN_INJECT_BUDGET - oversizedMarker.length);
		return prefix + verification.slice(0, budget) + oversizedMarker;
	}
	// No verification section found — truncate with marker
	return (
		prefix +
		planText.slice(0, PLAN_INJECT_BUDGET) +
		"\n[... TRUNCATED: plan exceeded context budget; verification steps may be missing — ask the planner to shorten if needed ...]"
	);
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

/** Hard cap so a 1M-token checker never dumps a multi-megabyte dirty tree. */
export const DIFF_EVIDENCE_HARD_CAP_CHARS = 80_000;
export const GOAL_DIFF_TRUNCATION_MARKER = "[... TRUNCATED: goal-relevant diff not fully shown ...]";
export const OTHER_DIFF_TRUNCATION_MARKER = "[... TRUNCATED: other working-tree files not touched this run ...]";

const MUTATING_TOOL_NAMES = new Set(["edit", "write"]);
const READ_TOOL_NAMES = new Set(["read"]);

export interface DiffEvidenceOptions {
	maxChars?: number;
	focusPaths?: string[];
	checker?: { contextWindow?: number };
	/** Untracked files to omit (harness audit logs, not maker work). */
	skipUntracked?: string[];
}

/**
 * Size the working-tree evidence to the checker window, then cap it.
 * ~40% of the remaining input budget, never above DIFF_EVIDENCE_HARD_CAP_CHARS.
 */
export function diffEvidenceBudgetChars(model?: { contextWindow?: number }): number {
	const windowTokens = model?.contextWindow ?? 128_000;
	const inputTokens = Math.max(0, windowTokens - RESPONSE_RESERVE_TOKENS);
	const inputChars = inputTokens * CHARS_PER_TOKEN;
	const share = Math.floor(inputChars * 0.4);
	return Math.max(512, Math.min(share, DIFF_EVIDENCE_HARD_CAP_CHARS));
}

/**
 * VERIFIED is illegal only when the *goal-relevant* change was truncated.
 * Summarizing unrelated dirty files must not void a pass.
 */
export function evidenceTruncationBlocksVerified(text: string): boolean {
	if (!text) return false;
	const stripped = text.split(OTHER_DIFF_TRUNCATION_MARKER).join("");
	if (stripped.includes(GOAL_DIFF_TRUNCATION_MARKER)) return true;
	if (/\[\.\.\. TRUNCATED: goal exceeded/i.test(stripped)) return true;
	if (/\[\.\.\. TRUNCATED:[^\]]*\]/i.test(stripped)) return true;
	if (/\[\.\.\. \d+ more chars not shown \.\.\.\]/i.test(stripped)) return true;
	if (/\[\.\.\. \d+ more chars \.\.\.\]/i.test(stripped)) return true;
	return false;
}

function toRepoRelativePath(cwd: string, filePath: string | undefined): string | null {
	if (!filePath || filePath.includes("\0")) return null;
	const resolved = isAbsolute(filePath) ? filePath : resolve(cwd, filePath);
	const rel = relative(cwd, resolved);
	if (!rel || rel.startsWith("..") || isAbsolute(rel)) return null;
	return rel.split("\\").join("/");
}

/**
 * Paths this run actually touched, from structured tool-call arguments
 * (not from scanning prose). Mutating edit/write paths win; if the maker
 * never wrote a file, read paths are the focus so a dirty tree does not
 * swallow a Q&A or lint task.
 */
export function collectTouchedPaths(messages: AgentMessage[], cwd: string): string[] {
	const mutated = new Set<string>();
	const read = new Set<string>();
	for (const m of messages) {
		if ((m as { role?: string }).role !== "assistant") continue;
		const content = (m as { content?: unknown }).content;
		if (!Array.isArray(content)) continue;
		for (const block of content) {
			if (!block || typeof block !== "object") continue;
			const b = block as { type?: string; name?: string; arguments?: unknown };
			if (b.type !== "toolCall" || !b.name) continue;
			const args = b.arguments;
			const path =
				args && typeof args === "object" && typeof (args as { path?: unknown }).path === "string"
					? (args as { path: string }).path
					: undefined;
			const rel = toRepoRelativePath(cwd, path);
			if (!rel) continue;
			if (MUTATING_TOOL_NAMES.has(b.name)) mutated.add(rel);
			else if (READ_TOOL_NAMES.has(b.name)) read.add(rel);
		}
	}
	return mutated.size > 0 ? [...mutated] : [...read];
}

function guardedGit(cwd: string, args: string[], timeout = 5_000): string {
	try {
		return execFileSync("git", args, {
			cwd,
			timeout,
			encoding: "utf8",
			maxBuffer: 4 * 1024 * 1024,
			stdio: ["ignore", "pipe", "ignore"],
		}).trim();
	} catch {
		return "";
	}
}

function takeWithMarker(text: string, budget: number, marker: string): { text: string; truncated: boolean } {
	if (budget <= 0) return { text: marker, truncated: true };
	if (text.length <= budget) return { text, truncated: false };
	const keep = Math.max(0, budget - marker.length - 1);
	return { text: `${text.slice(0, keep)}\n${marker}`, truncated: true };
}

function listUntracked(cwd: string): string[] {
	const raw = guardedGit(cwd, ["ls-files", "-z", "--others", "--exclude-standard"], 3_000);
	if (!raw) return [];
	return raw.split("\0").filter(Boolean).slice(0, 40);
}

function formatUntracked(files: string[], cwd: string): string {
	const parts: string[] = [];
	for (const file of files) {
		const content = readUntrackedFile(cwd, file);
		if (content) parts.push(`--- ${file} (untracked) ---\n${content}`);
	}
	return parts.join("\n\n");
}

/**
 * The real artifact for checker audits: working-tree diff against HEAD,
 * including untracked new files (which `git diff HEAD` omits). Final audits
 * get the truncated diff; checkpoints get the cheap stat view.
 *
 * Files this run edited (or, if none, files it read) are packed first. Other
 * dirty files are summarized and may be truncated without voiding VERIFIED.
 */
export function gatherDiffEvidence(cwd: string, kind: VerifyKind, opts: DiffEvidenceOptions = {}): string {
	const maxChars = opts.maxChars ?? diffEvidenceBudgetChars(opts.checker);
	const focusPaths = [
		...new Set((opts.focusPaths ?? []).map((p) => toRepoRelativePath(cwd, p)).filter((p): p is string => Boolean(p))),
	];
	const focusSet = new Set(focusPaths);

	const status = guardedExec(cwd, "git status --short");
	const stat = guardedGit(cwd, ["diff", "--stat", "HEAD"], 5_000);
	const overviewParts: string[] = [];
	if (status) overviewParts.push(`git status --short:\n${headLines(status, 30)}`);
	if (stat) overviewParts.push(`git diff --stat HEAD:\n${headLines(stat, 40)}`);
	if (kind !== "final") {
		const overview = overviewParts.join("\n\n");
		return takeWithMarker(overview, maxChars, GOAL_DIFF_TRUNCATION_MARKER).text;
	}

	const skipUntracked = new Set(
		(opts.skipUntracked ?? []).map((p) => toRepoRelativePath(cwd, p)).filter((p): p is string => Boolean(p)),
	);
	const untracked = listUntracked(cwd).filter((f) => !skipUntracked.has(f));
	const focusUntracked = focusSet.size === 0 ? untracked : untracked.filter((f) => focusSet.has(f));
	const otherUntracked = focusSet.size === 0 ? [] : untracked.filter((f) => !focusSet.has(f));

	const goalDiff =
		focusPaths.length > 0
			? guardedGit(cwd, ["diff", "HEAD", "--", ...focusPaths], 5_000)
			: guardedGit(cwd, ["diff", "HEAD"], 5_000);
	const otherDiff =
		focusPaths.length > 0
			? guardedGit(cwd, ["diff", "HEAD", "--", ".", ...focusPaths.map((p) => `:(exclude)${p}`)], 5_000)
			: "";

	const goalChunks: string[] = [];
	if (goalDiff) {
		goalChunks.push(
			focusPaths.length > 0
				? `git diff HEAD (files this run edited):\n${goalDiff}`
				: `git diff HEAD (working tree, tracked changes):\n${goalDiff}`,
		);
	}
	const goalUntracked = formatUntracked(focusUntracked, cwd);
	if (goalUntracked) {
		goalChunks.push(`Untracked new files (not in git diff HEAD, shown separately):\n${goalUntracked}`);
	}

	const otherChunks: string[] = [];
	if (otherDiff) otherChunks.push(`git diff HEAD (other working-tree files):\n${otherDiff}`);
	const otherUntrackedBody = formatUntracked(otherUntracked.slice(0, 20), cwd);
	if (otherUntrackedBody) otherChunks.push(`Other untracked files:\n${otherUntrackedBody}`);

	const overview = overviewParts.join("\n\n");
	const packed: string[] = overview ? [overview] : [];
	let remaining = maxChars - overview.length - (overview ? 2 : 0);

	const goalBody = goalChunks.join("\n\n");
	if (goalBody) {
		const taken = takeWithMarker(goalBody, remaining, GOAL_DIFF_TRUNCATION_MARKER);
		packed.push(taken.text);
		remaining -= taken.text.length + 2;
		if (taken.truncated) return packed.join("\n\n");
	}

	const otherBody = otherChunks.join("\n\n");
	if (otherBody) {
		if (remaining < 120) packed.push(OTHER_DIFF_TRUNCATION_MARKER);
		else packed.push(takeWithMarker(otherBody, remaining, OTHER_DIFF_TRUNCATION_MARKER).text);
	}

	return packed.join("\n\n");
}

export function createVerifyRouting(options: CreateVerifyRoutingOptions): VerifyRoutingHooks | undefined {
	const { verify, makerModel, makerThinkingLevel, cwd, resolveModel } = options;
	const finalChecker = resolveModel(verify.checkerModel);
	if (!verify.checkerModel || !finalChecker) return undefined;

	const auditLog = createVerifyAuditLogger(options.auditLogPath);
	/** F-07: every event carries the run identity envelope automatically. */
	const logVerify = (entry: Record<string, unknown>): void => {
		auditLog({ ...traceEnvelope(), ...entry });
	};

	// Independence is a family rule, not a lineup rule. The session maker and
	// the primary checker should differ. If they do not, an independent spare
	// checker (or a standard-tier maker override) can still make verification
	// possible — models will change; the spare slot must not.
	const makerFamily = extractModelFamily(makerModel.id);
	const checkerFamily = extractModelFamily(verify.checkerModel);
	const standardMakerFamily = verify.tiers?.standard?.makerModel
		? extractModelFamily(verify.tiers.standard.makerModel)
		: undefined;
	const spareFamily = verify.checkerFallbackModel ? extractModelFamily(verify.checkerFallbackModel) : "";
	const sessionBreaksIndependence = makerFamily === checkerFamily;
	const standardFixesIndependence = standardMakerFamily !== undefined && standardMakerFamily !== checkerFamily;
	const spareCoversSession = Boolean(spareFamily && spareFamily !== makerFamily);
	if (sessionBreaksIndependence && !standardFixesIndependence && !spareCoversSession) {
		auditLog({
			schema: 2,
			event: "disabled",
			reason: `same-family (${makerFamily}): maker ${makerModel.id} and checker ${verify.checkerModel} cannot independently verify`,
		});
		return undefined;
	}
	if (sessionBreaksIndependence && (standardFixesIndependence || spareCoversSession)) {
		auditLog({
			schema: 2,
			event: "independence-warning",
			reason: standardFixesIndependence
				? `session maker ${makerModel.id} and checker ${verify.checkerModel} are the same family, but standard.makerModel (${verify.tiers?.standard?.makerModel}) is independent`
				: `session maker ${makerModel.id} and checker ${verify.checkerModel} are the same family; audits will promote checkerFallbackModel ${verify.checkerFallbackModel}`,
		});
	}
	// Check tier-level independence for hard and standard tiers. Don't
	// refuse to install — instead, flag the tier so verifyTurn can refuse
	// to verify at runtime when the active tier has a same-family checker.
	//
	// Kind-aware: a tier can be independent for final audits but not for
	// checkpoints (or vice versa) because the two kinds pick the checker
	// from different fields. `tierIndependenceBroken[tier][kind]` records
	// the result for each (tier, kind) pair so verifyTurn can refuse only
	// the kind that is actually self-review.
	//
	// Effective maker for a tier: explicit makerModel, or (for hard) the
	// escalationMakerModel, or (for standard/trivial) the session default.
	// Effective checker for a kind: the kind-specific tier override, then
	// the kind-specific global field, then the global checker. When the
	// tier sets no checker at all, the global checker family is the
	// fallback — if that is the same family as the maker, self-review
	// happens via the global checker, not just the tier override.
	const sessionMakerFamily = extractModelFamily(makerModel.id);
	const globalCheckerFamily = extractModelFamily(verify.checkerModel);
	const tierIndependenceBroken: Record<string, Record<VerifyKind, boolean>> = {};
	for (const tierName of ["hard", "standard", "trivial"] as const) {
		const t = verify.tiers?.[tierName];
		const effectiveMakerRef =
			t?.makerModel ??
			(tierName === "hard" ? (verify.tiers?.hard?.escalationMakerModel ?? verify.escalationMakerModel) : undefined);
		const effectiveMakerFamily = effectiveMakerRef
			? extractModelFamily(effectiveMakerRef)
			: tierName === "hard"
				? undefined
				: sessionMakerFamily;
		if (effectiveMakerFamily === undefined) continue;
		for (const kindName of ["final", "checkpoint"] as const) {
			const checkerRef = checkerRefForKind(verify, kindName, tierName);
			const kindCheckerFamily = checkerRef ? extractModelFamily(checkerRef) : globalCheckerFamily;
			if (effectiveMakerFamily === kindCheckerFamily) {
				const tierSpare = verify.tiers?.[tierName]?.checkerFallbackModel ?? verify.checkerFallbackModel;
				const spareCovers = Boolean(tierSpare) && extractModelFamily(tierSpare as string) !== effectiveMakerFamily;
				if (spareCovers) {
					auditLog({
						schema: 2,
						event: "independence-warning",
						tier: tierName,
						kind: kindName,
						reason: `${tierName} tier effective maker ${effectiveMakerRef ?? makerModel.id} and ${kindName} checker ${checkerRef ?? verify.checkerModel} are the same family (${effectiveMakerFamily}) — audits will promote the spare checker`,
					});
					continue;
				}
				tierIndependenceBroken[tierName] ??= { final: false, checkpoint: false };
				tierIndependenceBroken[tierName][kindName] = true;
				auditLog({
					schema: 2,
					event: "independence-warning",
					tier: tierName,
					kind: kindName,
					reason: `${tierName} tier effective maker ${effectiveMakerRef ?? makerModel.id} and ${kindName} checker ${checkerRef ?? verify.checkerModel} are the same family (${effectiveMakerFamily}) — ${kindName} verification on this tier will be refused`,
				});
			}
		}
	}
	const everyN = verify.checkpointEveryToolTurns ?? 0;
	const maxCheckpoints = verify.maxCheckpointsPerRun ?? 8;
	const backoff = verify.checkpointBackoff ?? "geometric";
	const planFirst = verify.planFirst ?? Boolean(verify.plannerModel);
	const minPromptChars = verify.planMinPromptChars ?? 80;
	const maxRejections = verify.maxRejections ?? 2;
	const planAfterRejections = verify.planAfterRejections ?? 1;
	const escalateAfterRejections = verify.escalateAfterRejections ?? 1;
	const verifierCommands = sanitizeVerifierCommands(verify.verifierCommands);
	const maxCheckerRuns = verify.maxCheckerRuns ?? 2;
	const stickyEscalation = verify.stickyEscalation ?? true;
	const triageEnabled = verify.triage?.enabled ?? true;
	// F-01 whole-run bounds. 0 means unlimited (legacy behavior).
	const maxMakerTurns = verify.maxMakerTurns ?? 0;
	const maxToolCallsPerRun = verify.maxToolCallsPerRun ?? 0;
	const maxRunMs = verify.maxRunMs ?? 0;
	const maxRunCostUsd = verify.maxRunCostUsd ?? 0;
	// F-03 fail-open policy for unavailable/ambiguous checkers.
	const unavailablePolicy = verify.unavailablePolicy ?? "surface-unverified";

	let finalRejections = 0;
	let checkpointRejections = 0;
	let totalCheckpointRejections = 0; // F-10: never resets per run
	let totalFinalRejections = 0; // F-10: never resets per run
	let checkpointsSoFar = 0;
	let alreadyPlanned = false;
	let makerEscalated = false;
	let makerFallbacksUsed = 0;
	let liveMaker: Model<any> = makerModel;
	let tier: VerifyTier = "standard";
	const runCost = { calls: 0, tokensIn: 0, tokensOut: 0, costUsd: 0 };
	// F-01 whole-run counters + start time, reset per run.
	let makerTurns = 0;
	let toolCallsTotal = 0;
	let makerCostUsd = 0;
	let runStartedAt = 0;
	// F-07 run identity. Generated once per run; carried on every audit event.
	let runId = "";
	let runSummarized = false; // F-07: prevents double-summaries
	// P0-2: track whether a goal is currently active (a run that got past
	// triage and started real work). Short replies like "yes", "ok", "done"
	// after an active goal are follow-ups, not new conversational prompts.
	let goalActive = false;
	let missingSpareWarned = false;

	const warnMissingSpareOnce = (): void => {
		if (missingSpareWarned) return;
		missingSpareWarned = true;
		if (!verify.checkerFallbackModel) {
			logVerify({
				event: "independence-warning",
				reason: "checkerFallbackModel is unset — checker availability has no independent spare family",
			});
		} else if (spareFamily === checkerFamily) {
			logVerify({
				event: "independence-warning",
				reason: `checkerFallbackModel ${verify.checkerFallbackModel} is the same family as checkerModel (${checkerFamily}) — the spare is not independent`,
			});
		}
	};

	// A run is one user prompt. Without this reset, `alreadyPlanned` and the
	// checkpoint budget leak across tasks and every task after the first runs
	// with no plan and no mid-build audits.
	const resetRunState = (): void => {
		finalRejections = 0;
		checkpointRejections = 0;
		totalCheckpointRejections = 0;
		totalFinalRejections = 0;
		checkpointsSoFar = 0;
		alreadyPlanned = false;
		makerEscalated = false;
		makerFallbacksUsed = 0;
		liveMaker = makerModel;
		tier = "standard";
		runCost.calls = 0;
		runCost.tokensIn = 0;
		runCost.tokensOut = 0;
		runCost.costUsd = 0;
		makerTurns = 0;
		toolCallsTotal = 0;
		makerCostUsd = 0;
		runStartedAt = Date.now();
		// F-07: stable per-run id so traces can be reconstructed across sessions.
		runId = `r_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
		runSummarized = false;
		// P0-2: goalActive is NOT reset here. It persists across runs so that
		// a follow-up answering a CLARIFY question inherits the active goal
		// and goes through the full verify stack. Triage decides whether the
		// new message is small talk (TRIVIAL, which clears the goal) or more
		// work. It is set to true when triage assigns a non-trivial tier.
	};

	/** F-07 common envelope fields attached to every audit event for this run. */
	const traceEnvelope = (): Record<string, unknown> => {
		// Lazily assign a runId if verifyTurn is called without beforeFirstTurn
		// (e.g. in unit tests, or if triage is disabled). Ensures every event
		// carries a non-empty identity.
		if (!runId) {
			runId = `r_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
			runStartedAt = Date.now();
		}
		return { runId, schema: 2 };
	};

	/**
	 * F-01 + F-03: check the whole-run bounds. Returns a stop reason when a
	 * hard ceiling is breached, else undefined. Caller emits a visible
	 * STOPPED_UNVERIFIED notice and ends the run — never a silent pass.
	 */
	const checkRunBounds = (_kind: "checkpoint" | "final"): string | undefined => {
		const totalCost = makerCostUsd + runCost.costUsd;
		if (maxMakerTurns > 0 && makerTurns >= maxMakerTurns) {
			return `max-maker-turns (${maxMakerTurns})`;
		}
		if (maxToolCallsPerRun > 0 && toolCallsTotal >= maxToolCallsPerRun) {
			return `max-tool-calls (${maxToolCallsPerRun})`;
		}
		if (maxRunMs > 0 && Date.now() - runStartedAt >= maxRunMs) {
			return `max-run-ms (${maxRunMs})`;
		}
		if (maxRunCostUsd > 0 && totalCost >= maxRunCostUsd) {
			return `max-run-cost ($${totalCost.toFixed(4)} >= $${maxRunCostUsd})`;
		}
		return undefined;
	};

	const applyRunTally = (messages: AgentMessage[], current?: AgentMessage): void => {
		const t = tallyMakerTrace(messages, current);
		makerTurns = t.makerTurns;
		toolCallsTotal = t.toolCalls;
		makerCostUsd = t.costUsd;
	};

	const stoppedNotice = (reason: string): string =>
		`[VERIFY] STOPPED_UNVERIFIED — run ended on ${reason}. ` +
		`The answer below was NOT audited by the checker. Treat with suspicion.`;

	/**
	 * Run a verifier call with an availability fallback: on a provider error
	 * (upstream rate limit etc.), retry once on the configured fallback model.
	 * Returns the reply plus the model that actually produced it so follow-up
	 * calls in the same audit stay on the working model.
	 */
	const callWithFallback = async (
		runModel: VerifyTurnContext["runModel"],
		primary: Model<any>,
		fallbackRef: string | undefined,
		messages: AgentMessage[],
		thinkingLevel: ThinkingLevel | undefined,
		stage: string,
		fallbackThinkingLevel?: ThinkingLevel,
	): Promise<{ reply: AssistantMessage; model: Model<any> }> => {
		let reply: AssistantMessage | undefined;
		let failure = "";
		try {
			reply = await runModel(primary, messages, { thinkingLevel });
			trackCost(reply);
			if (!isErrorReply(reply)) return { reply, model: primary };
			failure = reply.errorMessage ?? reply.stopReason;
		} catch (error) {
			failure = error instanceof Error ? error.message : String(error);
		}
		const fallback = resolveModel(fallbackRef);
		if (!fallback || fallback.id === primary.id) {
			if (reply) return { reply, model: primary };
			throw new Error(failure);
		}
		logVerify({ event: "fallback", stage, from: primary.id, to: fallback.id, err: failure.slice(0, 160) });
		const second = await runModel(fallback, messages, {
			thinkingLevel: fallbackThinkingLevel ?? thinkingLevel,
		});
		trackCost(second);
		return { reply: second, model: fallback };
	};

	const trackCost = (m: AssistantMessage): { tokensIn: number; tokensOut: number; costUsd: number } => {
		const u = usageFields(m);
		runCost.calls++;
		runCost.tokensIn += u.tokensIn;
		runCost.tokensOut += u.tokensOut;
		runCost.costUsd += u.costUsd;
		return u;
	};

	const emitRunSummary = (outcome: "verified" | "unverified" | "stopped"): void => {
		if (runSummarized) return;
		runSummarized = true;
		// goalActive persists across run boundaries so a short follow-up
		// still goes through the verify stack when triage says it continues
		// the work. It is cleared when triage returns TRIVIAL.
		logVerify({
			...traceEnvelope(),
			event: "run-summary",
			tier,
			outcome,
			verifyCalls: runCost.calls,
			tokensIn: runCost.tokensIn,
			tokensOut: runCost.tokensOut,
			costUsd: Number((makerCostUsd + runCost.costUsd).toFixed(6)),
			makerCostUsd: Number(makerCostUsd.toFixed(6)),
			verifyCostUsd: Number(runCost.costUsd.toFixed(6)),
			makerTurns,
			toolCallsTotal,
			elapsedMs: runStartedAt ? Date.now() - runStartedAt : 0,
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
			skipPlanner: t?.skipPlanner ?? false,
			plannerRef: t?.plannerModel ?? verify.plannerModel,
			plannerThinking: t?.plannerThinkingLevel ?? verify.plannerThinkingLevel ?? "max",
			// P1/Gate 1: read the standard-tier maker override, don't fall
			// through to the session default. This is the field that pins
			// independence when the session model is cycled to Luna.
			maker: resolveModel(t?.makerModel) ?? makerModel,
			makerThinking: t?.makerThinkingLevel ?? makerThinkingLevel,
			everyN,
			escalationRef: t?.escalationMakerModel ?? verify.escalationMakerModel,
			escalationThinking: t?.escalationMakerThinkingLevel ?? verify.escalationMakerThinkingLevel ?? "max",
		};
	};

	const demoteMaker = (): { model: Model<any>; thinkingLevel: ThinkingLevel } => {
		makerEscalated = false;
		const view = tierView();
		liveMaker = view.maker;
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
			liveMaker = escalation;
			return { model: escalation, thinkingLevel: view.escalationThinking, escalated: true };
		}
		liveMaker = view.maker;
		return { model: view.maker, thinkingLevel: view.makerThinking, escalated: false };
	};

	// F-04: resolve the working directory for verifier commands (npm test,
	// npm run check) to the nearest package.json from the session cwd, not
	// the git root. In a monorepo, running from the git root would run the
	// wrong tests or fail for the wrong reason. Falls back to cwd if no
	// package.json is found.
	const projectRoot = (() => {
		// Walk up from cwd to find the nearest package.json
		let dir = cwd;
		for (let i = 0; i < 10; i++) {
			try {
				readFileSync(join(dir, "package.json"), "utf8");
				return dir;
			} catch {
				const parent = join(dir, "..");
				if (parent === dir) break;
				dir = parent;
			}
		}
		return cwd;
	})();

	const execVerifierCommand = (command: string): string => {
		if (!isSafeVerifierCommand(command) || !verifierCommands.includes(command)) {
			return "refused: verifier command is not a single allowlisted command";
		}
		try {
			return execSync(command, {
				cwd: projectRoot,
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
	): Promise<{
		verdict: "verified" | "conflict" | "ambiguous";
		auditText: string;
		ran: string | null;
		/** The messages the checker actually received (already bounded). */
		boundedMessages: AgentMessage[];
		/** The full, unbounded evidence the harness gathered for this audit. */
		fullEvidence: string;
	}> => {
		const checkerRef = checkerRefForKind(verify, kind, tier) ?? verify.checkerModel;
		if (!checkerRef) {
			return { verdict: "ambiguous", auditText: "", ran: null, boundedMessages: [], fullEvidence: "" };
		}
		const currentMaker = liveMaker ?? tierView().maker;
		const pick = selectAuditChecker({
			liveMakerRef: currentMaker.id,
			checkerRef,
			checkerFallbackRef: tierOverrides(verify, tier)?.checkerFallbackModel ?? verify.checkerFallbackModel,
		});
		const activeChecker = resolveModel(pick.primaryRef);
		if (!activeChecker) {
			return { verdict: "ambiguous", auditText: "", ran: null, boundedMessages: [], fullEvidence: "" };
		}
		if (!familiesDiffer(currentMaker.id, activeChecker.id)) {
			logVerify({
				event: "independence-refused",
				kind,
				tier,
				reason: "live maker and resolved checker are the same family",
				maker: currentMaker.id,
				checker: activeChecker.id,
			});
			return { verdict: "ambiguous", auditText: "", ran: null, boundedMessages: [], fullEvidence: "" };
		}
		if (pick.swapped) {
			logVerify({
				event: "checker-swap",
				kind,
				tier,
				reason: pick.reason,
				from: checkerRef,
				to: activeChecker.id,
				maker: currentMaker.id,
			});
		}
		const thinkingLevel = pick.swapped
			? checkerFallbackThinkingForKind(verify, kind, tier)
			: checkerThinkingForKind(verify, kind, tier);
		const fallbackThinkingLevel = checkerFallbackThinkingForKind(verify, kind, tier);

		const runInstruction = verifierCommands.length
			? ` If you need fresh evidence before judging, reply EXACTLY "RUN: <command>" with one of: ${verifierCommands.join(
					", ",
				)}. The harness — not the maker — runs it and you receive the raw output. You may request up to ${maxCheckerRuns} commands, one per reply.`
			: "";
		// The verdict-line instruction goes LAST so it is the most recent constraint
		// before the checker answers. It matters that EXCELLENCE_CHARTER itself
		// contains the uppercase rejection token: prose-scanning a reply that quotes
		// the rubric is exactly how false rejections happened (finding F-05).
		const auditPrompt =
			(kind === "checkpoint"
				? "You are an independent mid-build checker. You see the goal, recent tool receipts, the working-tree " +
					"diff summary, and the maker's latest progress — not the maker's private reasoning. Return the " +
					"VERIFIED verdict if progress is on track toward the goal (partial progress is OK). " +
					"IMPORTANT: read-only discovery (inspecting files, searching, reading code) IS valid progress when " +
					"the task requires investigation before implementation — do not reject it just because no diff exists " +
					"yet. Only return the CONFLICT verdict if the maker is genuinely stuck, wrong, skipping required " +
					"verification, repeating failed approaches, or drifting from the goal. State the specific problem " +
					"after the verdict line. If the evidence notes other working-tree files not touched this run, " +
					"those are pre-existing workspace dirt — do not CONFLICT solely because of them."
				: "You are an independent checker. You are given the goal, the real tool receipts, the actual " +
					"working-tree diff, and the maker's final answer — not the maker's reasoning. Judge the artifact, " +
					"not the summary. Return the VERIFIED verdict only if the answer fully and correctly satisfies the " +
					"goal AND the receipts show real verification (tests/build/lint actually ran and passed). Return the " +
					"CONFLICT verdict if the answer is wrong, incomplete, or its claims are unverified, and state the " +
					"specific problem after the verdict line. If the evidence notes other working-tree files not " +
					"touched this run, those are pre-existing workspace dirt — do not CONFLICT solely because of them.") +
			` ${EXCELLENCE_CHARTER}` +
			runInstruction +
			` ${CHECKER_VERDICT_LINE_INSTRUCTION}`;

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
				content: `Tool receipts (real outputs, redacted for cross-model safety):\n\n${receipts.map((r) => redactEvidence(r)).join("\n\n")}`,
				timestamp: Date.now(),
			});
		}
		const diffEvidence = gatherDiffEvidence(cwd, kind, {
			checker: activeChecker,
			focusPaths: collectTouchedPaths(vctx.newMessages, cwd),
			skipUntracked: [options.auditLogPath],
		});
		if (diffEvidence) {
			auditMessages.push({
				role: "user",
				content: `Working-tree evidence (gathered by the harness, not the maker; redacted):\n\n${redactEvidence(diffEvidence)}`,
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

		let audit: AssistantMessage;
		let workingChecker = activeChecker;
		let workingThinking = thinkingLevel;
		let boundedMessages: AgentMessage[] = [];
		const fullEvidence = diffEvidence;
		try {
			const bounded = boundContextForModel(auditMessages, activeChecker);
			const first = await callWithFallback(
				vctx.runModel,
				activeChecker,
				pick.fallbackRef,
				bounded,
				thinkingLevel,
				kind === "checkpoint" ? "checkpoint-checker" : "final-checker",
				fallbackThinkingLevel,
			);
			audit = first.reply;
			workingChecker = first.model;
			if (first.model.id !== activeChecker.id) workingThinking = fallbackThinkingLevel;
			boundedMessages = bounded;
		} catch (error) {
			// Both checker and fallback unavailable: return ambiguous so the
			// verifyTurn F-03 branch surfaces an explicit UNVERIFIED notice to
			// the user. No longer a silent fail-open.
			logVerify({
				event: "audit-unavailable",
				kind,
				tier,
				checker: activeChecker.id,
				err: error instanceof Error ? error.message.slice(0, 200) : String(error).slice(0, 200),
			});
			return { verdict: "ambiguous", auditText: "", ran: null, boundedMessages: [], fullEvidence: "" };
		}
		let auditText = messageText(audit);

		const ran: string[] = [];
		while (ran.length < maxCheckerRuns) {
			const requested = /^RUN:\s*(\S.*)$/m.exec(auditText)?.[1]?.trim();
			if (!requested || !verifierCommands.includes(requested) || ran.includes(requested)) break;
			ran.push(requested);
			const output = redactEvidence(execVerifierCommand(requested));
			auditMessages.push({
				role: "user",
				content: `Output of \`${requested}\` (run by the harness, tail):\n${output}`,
				timestamp: Date.now(),
			});
			// Truncation-override inspects boundedMessages. Re-bind on every
			// follow-up so a later call that drops the diff cannot VERIFIED
			// against the first payload that still had it.
			const runBounded = boundContextForModel(auditMessages, workingChecker);
			boundedMessages = runBounded;
			audit = await vctx.runModel(workingChecker, runBounded, {
				thinkingLevel: workingThinking,
			});
			trackCost(audit);
			auditText = messageText(audit);
		}

		let parsed = parseCheckerVerdict(auditText);
		let verdict = parsed.verdict;
		if (verdict === "ambiguous" && !isErrorReply(audit)) {
			// One strict re-ask: an ambiguous verdict otherwise drops the audit
			// silently, wasting the checker call.
			auditMessages.push({
				role: "user",
				content: `Your reply did not contain a clear verdict. ${CHECKER_VERDICT_LINE_INSTRUCTION}`,
				timestamp: Date.now(),
			});
			const reaskBounded = boundContextForModel(auditMessages, workingChecker);
			boundedMessages = reaskBounded;
			audit = await vctx.runModel(workingChecker, reaskBounded, {
				thinkingLevel: workingThinking,
			});
			trackCost(audit);
			auditText = messageText(audit);
			parsed = parseCheckerVerdict(auditText);
			verdict = parsed.verdict;
		}

		logVerify({
			event: "audit",
			kind,
			tier,
			checker: workingChecker.id,
			verdict,
			// How the verdict was decided. Lets the improvement loop measure whether
			// checkers actually follow the structured contract instead of guessing.
			verdictVia: parsed.via,
			ran: ran.length > 0 ? ran : null,
			stop: audit.stopReason,
			err: audit.errorMessage?.slice(0, 200) ?? null,
			text: auditText.slice(0, 240),
			...usageFields(audit),
		});

		return { verdict, auditText, ran: ran[0] ?? null, boundedMessages, fullEvidence };
	};

	const escalatePlanner = async (vctx: VerifyTurnContext, feedback: string): Promise<string> => {
		// Re-plans after rejection use the strong planner: recovery after the
		// cheap path failed is where the expensive model earns its price.
		const plannerRef = verify.strongPlannerModel ?? verify.plannerModel;
		const planner = resolveModel(plannerRef);
		if (!planner) return "";
		try {
			const { reply: plan } = await callWithFallback(
				vctx.runModel,
				planner,
				verify.plannerFallbackModel,
				[
					...boundContextForModel(vctx.context.messages, planner),
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
				verify.strongPlannerThinkingLevel ?? verify.plannerThinkingLevel ?? "max",
				"re-planner",
			);
			return messageText(plan);
		} catch {
			// Planner and fallback both unavailable: retry without a plan.
			return "";
		}
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
				? `${injectPlan(
						planText,
						`[MAKER/CHECKER/PLANNER] Your previous ${kind === "checkpoint" ? "progress" : "answer"} was rejected by the checker. Feedback: ${auditText.slice(0, 800)}.\n\nThe planner produced this plan:\n`,
					)}\nExecute the plan step by step; do not repeat the rejected approach.`
				: `[MAKER/CHECKER/PLANNER] Your previous ${kind === "checkpoint" ? "progress" : "answer"} was rejected by the checker. Feedback: ${auditText.slice(0, 800)}. ` +
					`Rework ${kind === "checkpoint" ? "the next steps" : "the answer"} correctly; do not repeat the rejected approach.`,
			model: retry.model,
			thinkingLevel: retry.thinkingLevel,
		};
	};

	const clarifyDirective = (questions: string): string =>
		"[PLANNER] Open questions block this goal. FIRST investigate the repository and environment yourself with " +
		"read-only tools and answer every question you can from evidence — never ask the user anything you can " +
		"discover. THEN ask the user only what remains and is genuinely theirs to answer (intent, scope trade-offs, " +
		"budgets, external access): at most 5 questions, one sentence each. Then STOP and wait for their answers — " +
		"do not write code and do not guess on the questions you relay.\n\nPlanner's open questions:\n\n" +
		questions;

	return {
		beforeFirstTurn: async (ctx) => {
			warnMissingSpareOnce();
			// Only a fresh user prompt starts a new run. Continues/retries carry no
			// new user message and must not reset state or re-plan mid-task.
			const firstUser = ctx.newMessages.find((m) => m.role === "user");
			if (!firstUser) return undefined;
			const promptText = messageText(firstUser as UserMessage);
			const hadGoal = goalActive;
			resetRunState();
			logVerify({ event: "run-start", chars: promptText.length });

			const signals = gatherRepoSignals(cwd);

			if (triageEnabled) {
				const triageModel = resolveModel(verify.triage?.model) ?? makerModel;
				try {
					const reply = await ctx.runModel(
						triageModel,
						[
							{
								role: "user",
								content: buildTriagePrompt({ promptText, signals, goalActive: hadGoal }),
								timestamp: Date.now(),
							},
						],
						{ thinkingLevel: verify.triage?.thinkingLevel ?? "off" },
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

			goalActive = tier !== "trivial";

			const view = tierView();
			liveMaker = view.maker;
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
				const { reply: plan } = await callWithFallback(
					ctx.runModel,
					planner,
					verify.plannerFallbackModel,
					[
						{
							role: "user",
							content:
								`Goal:\n${promptText}\n\n` +
								(signals ? `Repo signals:\n${signals}\n\n` : "") +
								"You are a senior planner with no tool access, but the maker executing your plan has FULL " +
								"tool access: it can read any file, search the repo, and run commands. PLAN BY DEFAULT.\n" +
								"If the user is only greeting you or making small talk with no coding request, do not CLARIFY " +
								"and do not invent a goal — there isn't one. Reply with a one-line plan: greet back and wait.\n" +
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
					view.plannerThinking,
					"planner",
				);
				const cost = usageFields(plan);
				const planText = messageText(plan);
				if (!planText) {
					logVerify({ event: "plan-skipped", reason: "empty-plan", planner: view.plannerRef, tier });
					return baseResult;
				}
				if (/^\s*CLARIFY\b/i.test(planText)) {
					// Backstop: even a misbehaving planner cannot relay a wall of text.
					// F-06: enforce a hard cap of 5 questions. Count numbered lines,
					// lines ending in ?, and any non-empty line as a fallback — a
					// planner that avoids both numbering and ? must still be capped.
					const rawQuestions = planText.replace(/^\s*CLARIFY\W*/i, "").trim() || planText;
					const allLines = rawQuestions.split("\n").filter((l) => l.trim().length > 0);
					const questionLines = allLines.filter((l) => /^\s*\d+[.)]\s+/m.test(l) || /\?\s*$/.test(l.trim()));
					// If question-like lines were found, cap those. Otherwise cap
					// all non-empty lines (catches "Decide the scope for item N.")
					const linesToCap = questionLines.length > 0 ? questionLines : allLines;
					// Always cap the relayed block to 5 lines. A mixed CLARIFY
					// (a few numbered questions plus many unnumbered "also consider"
					// lines) used to inject all 40 lines because questionLines were
					// found and capped, but the unnumbered extras were still
					// included via the rawQuestions.slice(0, 2_000) fallback below.
					const cappedQuestions =
						linesToCap.length > 5
							? linesToCap.slice(0, 5).join("\n")
							: linesToCap.length > 0
								? linesToCap.join("\n")
								: rawQuestions.slice(0, 2_000);
					const questions = cappedQuestions;
					logVerify({
						event: "clarify",
						planner: planner.id,
						tier,
						text: questions.slice(0, 240),
						...cost,
						questionsCapped: linesToCap.length > 5,
					});
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
							content: injectPlan(planText),
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

		onTurnError: async ({ message }) => {
			// A maker turn died on a provider error after retries. Swap to the
			// availability fallback and retry instead of killing the run.
			if (makerFallbacksUsed >= 2) return undefined;
			const failedId = (message as { model?: string }).model ?? "";
			const fallback = resolveModel(verify.makerFallbackModel);
			if (!fallback || fallback.id === failedId) return undefined;
			makerFallbacksUsed++;
			liveMaker = fallback;
			logVerify({
				event: "maker-fallback",
				from: failedId || null,
				to: fallback.id,
				n: makerFallbacksUsed,
				err: message.errorMessage?.slice(0, 160) ?? null,
			});
			return { model: fallback };
		},

		verifyTurn: async (vctx) => {
			warnMissingSpareOnce();
			const kind: VerifyKind = vctx.kind ?? "final";

			applyRunTally(vctx.newMessages, vctx.message);

			// F-01: enforce whole-run hard ceilings before doing any more work.
			const stopReason = checkRunBounds(kind);
			if (stopReason) {
				logVerify({ event: "run-stopped", kind, tier, reason: stopReason, makerTurns, toolCallsTotal });
				emitRunSummary("stopped");
				const base = demoteMaker();
				return {
					status: "unverified",
					model: base.model,
					thinkingLevel: base.thinkingLevel,
					notice: stoppedNotice(stopReason),
				};
			}

			// Gate 2: if this tier's independence is broken (maker and checker
			// are the same family) for THIS kind of audit, refuse to verify —
			// self-review is not a second opinion. Surface an honest UNVERIFIED
			// notice. Independence is kind-aware: a tier may be independent for
			// final audits but not for checkpoints (or vice versa) because the
			// two kinds pick the checker from different fields.
			if (tierIndependenceBroken[tier]?.[kind]) {
				logVerify({
					event: "independence-refused",
					kind,
					tier,
					reason: "tier maker and checker are the same family",
				});
				if (kind === "final") emitRunSummary("unverified");
				const base = demoteMaker();
				return {
					status: "unverified",
					model: base.model,
					thinkingLevel: base.thinkingLevel,
					notice: `[VERIFY] UNVERIFIED — the ${tier} tier's ${kind} checker is the same family as its maker. Independent verification is not possible. Treat with suspicion.`,
				};
			}

			// `auditOnlyAfterTools` skips the final audit when the maker used no
			// tools. That is correct for the trivial tier (no artifact to audit),
			// but for standard/hard a no-tool final answer is exactly where a
			// second opinion matters most — explanations, design reviews, and
			// reasoning have no diff to check, so the checker is the only back
			// pressure. Gate the skip on the trivial tier only.
			if (
				kind === "final" &&
				verify.auditOnlyAfterTools &&
				tier === "trivial" &&
				!vctx.newMessages.some((m) => m.role === "toolResult")
			) {
				logVerify({ event: "skipped", reason: "no-tools", kind, tier });
				// F-03: a trivial no-tool skip is a deliberate pass, not a silent
				// fail-open. Emit a summary so the trace has a terminal event.
				emitRunSummary("verified");
				return undefined;
			}

			if (kind === "checkpoint") {
				checkpointsSoFar++;
				logVerify({ event: "checkpoint", toolTurnCount: vctx.toolTurnCount ?? null, n: checkpointsSoFar, tier });
			}

			const { verdict, auditText, boundedMessages, fullEvidence } = await runCheckerAudit(vctx, kind);

			// F-04/Gate 7: if the evidence was truncated, VERIFIED is illegal.
			// The checker cannot pass work it has not fully seen. Override the
			// verdict to unverified with a notice explaining the truncation.
			//
			// Check what the checker ACTUALLY received (boundedMessages), not a
			// second copy of the diff. The old code re-gathered the diff here,
			// which is a different snapshot from what bounding forwarded — a
			// mid-size diff that bounding silently dropped never tripped this
			// gate because the re-gathered copy still contained the diff text.
			//
			// Match the harness's own truncation markers precisely, not loose
			// substrings. The old `/more chars/i` regex matched any evidence
			// containing those two words, so a one-line constant named
			// `moreChars` (or a comment saying "more chars") was treated as a
			// truncation marker and falsely rejected. The real markers are
			// structured phrases emitted by gatherDiffEvidence / boundContextForModel.
			if (verdict === "verified") {
				const checkerSaw = boundedMessages.map((m) => messageText(m)).join("\n");
				const truncatedGoal =
					evidenceTruncationBlocksVerified(checkerSaw) || evidenceTruncationBlocksVerified(fullEvidence);
				const fullHadDiff = /git diff HEAD|untracked/i.test(fullEvidence);
				const checkerSawDiff = /git diff HEAD|untracked/i.test(checkerSaw);
				const diffDroppedByBounding = fullHadDiff && !checkerSawDiff;
				if (truncatedGoal || diffDroppedByBounding) {
					logVerify({
						event: "truncation-override",
						kind,
						tier,
						reason: diffDroppedByBounding
							? "checker verified but bounding dropped the diff evidence"
							: "checker verified but evidence was truncated",
					});
					const base = demoteMaker();
					if (kind === "final") emitRunSummary("unverified");
					return {
						status: "unverified",
						model: base.model,
						thinkingLevel: base.thinkingLevel,
						notice:
							kind === "final"
								? "[VERIFY] UNVERIFIED — the goal-relevant diff was truncated (or dropped by bounding) and the checker could not see the full change. VERIFIED is illegal when evidence is incomplete."
								: "[VERIFY] UNVERIFIED — checkpoint evidence was truncated or dropped. Keep building; this checkpoint is not a pass.",
					};
				}
			}

			if (verdict === "verified") {
				if (kind === "final") finalRejections = 0;
				else checkpointRejections = 0;
				// Sticky escalation: once a run needed the stronger maker, demoting on
				// the next verified checkpoint invites fail→escalate→pass→demote→fail
				// ping-pong, which costs more than the stronger maker's tokens.
				const keepEscalated = stickyEscalation && makerEscalated && kind === "checkpoint";
				const next = keepEscalated ? currentEscalation() : demoteMaker();
				liveMaker = next.model;
				logVerify({ event: keepEscalated ? "kept-escalated" : "demoted", kind, tier, maker: next.model.id });
				if (kind === "final") emitRunSummary("verified");
				return {
					status: "verified",
					model: next.model,
					thinkingLevel: next.thinkingLevel,
				};
			}

			// F-03: ambiguous/unavailable checker. The old code returned
			// `undefined` here, which the agent loop treats as "no verifier
			// decision" and the final answer exits normally with no user-visible
			// unverified status. Now: surface an explicit UNVERIFIED notice so
			// the user always sees that the checker never signed off, and emit
			// a terminal run-summary so the trace is complete.
			if (verdict !== "conflict" || auditText.length === 0) {
				logVerify({ event: "skipped", reason: "ambiguous-verdict", kind, tier });
				if (kind === "final") {
					emitRunSummary("unverified");
					const base = demoteMaker();
					const notice =
						"[VERIFY] UNVERIFIED — the checker returned no clear verdict (ambiguous or unavailable). " +
						"The answer was NOT audited. Treat with suspicion.";
					if (unavailablePolicy === "fail-closed" && finalRejections < maxRejections) {
						// Fail-closed: force one rejection cycle instead of accepting.
						finalRejections++;
						return rejectWithOptionalPlan(vctx, kind, notice, finalRejections);
					}
					return {
						status: "unverified",
						model: base.model,
						thinkingLevel: base.thinkingLevel,
						notice,
					};
				}
				// Checkpoint ambiguous: accept progress but label it honestly.
				const ambiguousBase = demoteMaker();
				return {
					status: "unverified",
					model: ambiguousBase.model,
					thinkingLevel: ambiguousBase.thinkingLevel,
					notice: `[VERIFY] UNVERIFIED — mid-build checkpoint ${checkpointsSoFar} returned no clear verdict. Progress accepted; treat with suspicion.`,
				};
			}

			if (kind === "checkpoint") {
				checkpointRejections++;
				totalCheckpointRejections++;
				// F-10: use total budget, not per-streak. The per-streak counter
				// can reset and allow endless churn. The total never resets
				// during a run, so the loop converges or stops honestly.
				if (totalCheckpointRejections > maxRejections) {
					logVerify({
						event: "budget-exhausted",
						kind,
						tier,
						rejections: totalCheckpointRejections,
						accepted: "unverified",
					});
					// F-10: do NOT reset checkpointRejections to 0. The total
					// budget is exhausted; further checkpoints must see this.
					checkpointRejections = maxRejections; // prevent re-entry
					const base = demoteMaker();
					// Accept progress and keep building, but say so honestly.
					return {
						status: "unverified",
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
			totalFinalRejections++;
			if (totalFinalRejections > maxRejections) {
				logVerify({
					event: "budget-exhausted",
					kind,
					tier,
					rejections: totalFinalRejections,
					accepted: "unverified",
				});
				finalRejections = 0;
				const base = demoteMaker();
				emitRunSummary("unverified");
				// Accept the final answer and stop, but the user must see that
				// the checker never signed off (P1-3: distinct unverified status).
				return {
					status: "unverified",
					model: base.model,
					thinkingLevel: base.thinkingLevel,
					notice:
						"[VERIFY] UNVERIFIED — final answer accepted on checker budget " +
						`(${maxRejections} rejections). Treat with suspicion. Last checker feedback: ${auditText.slice(0, 400)}`,
				};
			}
			return rejectWithOptionalPlan(vctx, kind, auditText, finalRejections);
		},

		shouldStopAfterTurn: (ctx) => {
			applyRunTally(ctx.newMessages, ctx.message);
			const reason = checkRunBounds("checkpoint");
			if (!reason) return false;
			if (!runSummarized) {
				logVerify({ event: "run-stopped", kind: "turn", tier, reason, makerTurns, toolCallsTotal });
				emitRunSummary("stopped");
			}
			const alreadyNoted = ctx.newMessages.some(
				(m) => m.role === "user" && messageText(m).includes("STOPPED_UNVERIFIED"),
			);
			if (!alreadyNoted) {
				const notice = { role: "user" as const, content: stoppedNotice(reason), timestamp: Date.now() };
				ctx.newMessages.push(notice);
				ctx.context.messages.push(notice);
			}
			return true;
		},
	};
}
