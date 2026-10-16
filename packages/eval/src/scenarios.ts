import { readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";
import { Scenario } from "./types.js";

export const DEFAULT_SCENARIOS_DIR = resolve(fileURLToPath(new URL(".", import.meta.url)), "../scenarios");

export function loadScenarios(dir = DEFAULT_SCENARIOS_DIR): Scenario[] {
  return readdirSync(dir)
    .filter((f) => f.endsWith(".yaml") || f.endsWith(".yml"))
    .map((f) => {
      const parsed = Scenario.safeParse(YAML.parse(readFileSync(join(dir, f), "utf8")));
      if (!parsed.success) {
        throw new Error(`Invalid scenario ${f}:\n${parsed.error.issues.map((i) => `  - ${i.path.join(".")}: ${i.message}`).join("\n")}`);
      }
      return parsed.data;
    });
}

export function findScenario(id: string, dir = DEFAULT_SCENARIOS_DIR): Scenario {
  const s = loadScenarios(dir).find((x) => x.id === id);
  if (!s) throw new Error(`Scenario "${id}" not found in ${dir}`);
  return s;
}
