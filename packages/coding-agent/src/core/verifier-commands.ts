/**
 * Checker-ordered verifier commands are executed by the harness via a shell.
 * The allowlist is an exact-string match against settings JSON, so a project
 * entry like `npm test; curl evil` would otherwise run as a compound command.
 * Only a single simple command is legal.
 */

const COMPOUND_COMMAND = /[;&|`$()<>\n]|&&|\|\|/;

export function isSafeVerifierCommand(command: string): boolean {
	const cmd = command.trim();
	if (!cmd) return false;
	if (COMPOUND_COMMAND.test(cmd)) return false;
	if (/^[A-Za-z_][A-Za-z0-9_]*=/.test(cmd)) return false;
	return true;
}

export function sanitizeVerifierCommands(commands: string[] | undefined): string[] {
	return (commands ?? []).filter(isSafeVerifierCommand);
}
