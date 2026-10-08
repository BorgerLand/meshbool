import type { Plugin } from "@borger/plugin_sdk";

//per-client history of freebuild actions. only the owner
//needs it, and it's never presented
export const meshbool = {
	name: "meshbooldonotuse",
	tracked: false,
	rsSimulationFQN: "::meshbool::donotuse",
	nodePackageName: "meshbool",
} satisfies Plugin<"meshbooldonotuse">;
