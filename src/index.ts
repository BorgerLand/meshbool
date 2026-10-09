import type { Plugin } from "@borger/plugin_sdk";

export const meshbool = {
	name: "meshbooldonotuse",
	tracked: false,
	rsSimulationFQN: "::meshbool::donotuse",
	nodePackageName: "meshbool",
} satisfies Plugin<"meshbooldonotuse">;
