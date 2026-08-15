/**
 * Read piped stdin without hanging forever when a parent process leaves
 * the pipe open (common for script spawners and `pi -p` from an IDE).
 */

import type { Readable } from "node:stream";

export type PipedStdin = Readable & { isTTY?: boolean };

export interface ReadPipedStdinOptions {
	/** When false, stop waiting if no data arrives within idleTimeoutMs. Default true. */
	waitForEnd?: boolean;
	idleTimeoutMs?: number;
}

/**
 * Returns undefined when stdin is a TTY, empty, or (in optional mode) stays
 * open with no data. Trims the captured text.
 */
export async function readPipedStdin(
	stdin: PipedStdin = process.stdin,
	opts: ReadPipedStdinOptions = {},
): Promise<string | undefined> {
	if (stdin.isTTY) return undefined;
	if (stdin.readableEnded || stdin.destroyed) return undefined;

	const waitForEnd = opts.waitForEnd ?? true;
	const idleTimeoutMs = opts.idleTimeoutMs ?? 50;

	return new Promise((resolve) => {
		let data = "";
		let settled = false;
		let idleTimer: ReturnType<typeof setTimeout> | undefined;

		const finish = (): void => {
			if (settled) return;
			settled = true;
			clearTimeout(idleTimer);
			stdin.off("data", onData);
			stdin.off("end", onEnd);
			stdin.off("error", onEnd);
			resolve(data.trim() || undefined);
		};

		const onData = (chunk: string | Buffer): void => {
			data += String(chunk);
			if (!waitForEnd) {
				clearTimeout(idleTimer);
				idleTimer = setTimeout(finish, idleTimeoutMs);
			}
		};

		const onEnd = (): void => finish();

		if (!waitForEnd) {
			idleTimer = setTimeout(finish, idleTimeoutMs);
		}
		if (typeof stdin.setEncoding === "function") stdin.setEncoding("utf8");
		stdin.on("data", onData);
		stdin.on("end", onEnd);
		stdin.on("error", onEnd);
		if (typeof stdin.resume === "function") stdin.resume();
	});
}
