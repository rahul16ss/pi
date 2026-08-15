import { PassThrough } from "node:stream";
import { describe, expect, it } from "vitest";
import { readPipedStdin } from "../src/cli/piped-stdin.ts";

describe("readPipedStdin", () => {
	it("returns undefined for a TTY", async () => {
		const stdin = Object.assign(new PassThrough(), { isTTY: true });
		await expect(readPipedStdin(stdin)).resolves.toBeUndefined();
	});

	it("waits for end when stdin is the prompt", async () => {
		const stdin = new PassThrough();
		const pending = readPipedStdin(stdin, { waitForEnd: true });
		stdin.write("hello from pipe\n");
		stdin.end();
		await expect(pending).resolves.toBe("hello from pipe");
	});

	it("does not hang when a CLI prompt is already present and stdin stays open", async () => {
		const stdin = new PassThrough();
		const pending = readPipedStdin(stdin, { waitForEnd: false, idleTimeoutMs: 30 });
		await expect(pending).resolves.toBeUndefined();
	});

	it("still merges data already in the pipe when a CLI prompt is present", async () => {
		const stdin = new PassThrough();
		const pending = readPipedStdin(stdin, { waitForEnd: false, idleTimeoutMs: 30 });
		stdin.write("from pipe\n");
		stdin.end();
		await expect(pending).resolves.toBe("from pipe");
	});
});
