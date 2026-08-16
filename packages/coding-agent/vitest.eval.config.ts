import { fileURLToPath } from "node:url";
import { defineConfig, mergeConfig } from "vitest/config";
import baseConfig, { workspaceSourcePaths } from "../../vitest.base.ts";

/** Machine-local eval-gate / gauntlet tests excluded from default CI vitest. */
export default mergeConfig(
	baseConfig,
	defineConfig({
		test: {
			globals: true,
			environment: "node",
			testTimeout: 30000,
			env: { PI_OFFLINE: "1" },
			unstubEnvs: true,
			exclude: ["**/node_modules/**", "**/dist/**"],
			server: {
				deps: {
					external: [/@silvia-odwyer\/photon-node/],
				},
				fs: {
					allow: [fileURLToPath(new URL("../../../.pi/agent/extensions", import.meta.url))],
				},
			},
		},
		resolve: {
			alias: [
				{
					find: /^@earendil-works\/pi-client$/,
					replacement: fileURLToPath(new URL("../client/src/index.ts", import.meta.url)),
				},
				{
					find: /^@earendil-works\/pi-protocol$/,
					replacement: fileURLToPath(new URL("../protocol/src/index.ts", import.meta.url)),
				},
				{ find: /^@mariozechner\/pi-ai$/, replacement: workspaceSourcePaths.aiIndex },
				{ find: /^@mariozechner\/pi-ai\/oauth$/, replacement: workspaceSourcePaths.aiOAuth },
				{ find: /^@mariozechner\/pi-agent-core$/, replacement: workspaceSourcePaths.agentIndex },
				{ find: /^@mariozechner\/pi-tui$/, replacement: workspaceSourcePaths.tuiIndex },
			],
		},
	}),
);
