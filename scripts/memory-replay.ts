/**
 * Offline replay of memory consolidation over a recorded conversation. Re-runs the summariser
 * (default haiku) over the recorded turns with the same window/cadence the engine uses, prints the
 * fact count after each consolidation, and reports whether given strings survive in memory. Lets a
 * consolidation-prompt change be tested in minutes instead of an hour-long 200-turn run, and gives
 * memory snapshots for recordings made before `memorySnapshots` existed.
 *
 *   npm run memory-replay -- eval-results/long-run-v9/conversations/haru-ko-long__v9-structured-sonnet__r1.json \
 *     --expect "성수동,달빛 고양이,보리" [--rules v1|v2|v3] [--model claude-cli:haiku] [--window 20] [--every 10] [--max-facts 25] [--out snapshots.json]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { DEFAULT_MEMORY_CONFIG, MemoryManager, emptyMemory, findCharacter, type MemoryState, type Turn } from "@persona/engine";
import { resolveProvider } from "../packages/eval/src/providers.js";
import { findScenario } from "../packages/eval/src/scenarios.js";

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith("--"));
if (!file) {
  console.error("usage: memory-replay <conversation.json> [--expect a,b] [--rules v1|v2|v3] [--model spec] [--window n] [--every n] [--max-facts n] [--out file]");
  process.exit(1);
}
const flag = (name: string) => {
  const i = args.indexOf(`--${name}`);
  return i === -1 ? undefined : args[i + 1];
};
const expect = (flag("expect") ?? "").split(",").map((s) => s.trim()).filter(Boolean);
const rules = (flag("rules") ?? "v1") as "v1" | "v2" | "v3" | "v4";
const spec = flag("model") ?? "claude-cli:haiku";
const windowTurns = Number(flag("window") ?? DEFAULT_MEMORY_CONFIG.windowTurns);
const consolidateEvery = Number(flag("every") ?? DEFAULT_MEMORY_CONFIG.consolidateEvery);
const maxFacts = Number(flag("max-facts") ?? DEFAULT_MEMORY_CONFIG.maxFacts);

const rec = JSON.parse(readFileSync(file, "utf8")) as { scenarioId: string; turns: Turn[] };
const scenario = findScenario(rec.scenarioId);
const card = findCharacter(scenario.characterId);
const { provider, model } = resolveProvider(spec);
const manager = new MemoryManager(provider, { windowTurns, consolidateEvery, model, rules, maxFacts });

const has = (m: MemoryState, s: string) => m.summary.includes(s) || m.facts.some((f) => f.text.includes(s));
const firstSaid = (s: string) => rec.turns.find((t) => t.role === "user" && t.content.includes(s))?.index;

(async () => {
  let memory = emptyMemory();
  const turns: Turn[] = [];
  const snapshots: { afterTurnIndex: number; memory: MemoryState }[] = [];
  let calls = 0;
  let promptTokens = 0;
  console.log(`# memory-replay · ${file} · rules ${rules} · ${spec} · window ${windowTurns} · every ${consolidateEvery} · max facts ${maxFacts}`);
  console.log(`\n| after turn | facts | ${expect.join(" | ")} |`);
  console.log(`|---:|---:|${expect.map(() => "---").join("|")}|`);
  for (const t of rec.turns) {
    turns.push(t);
    if (!manager.needsConsolidation(turns, memory)) continue;
    memory = await manager.consolidate(card, scenario.lang, turns, memory);
    calls++;
    promptTokens += manager.lastUsage?.promptTokens ?? 0;
    snapshots.push({ afterTurnIndex: t.index, memory: structuredClone(memory) });
    const cells = expect.map((s) => {
      const said = firstSaid(s);
      if (said === undefined || said > memory.coveredUpTo) return "·";
      return has(memory, s) ? "O" : "X";
    });
    console.log(`| ${t.index} | ${memory.facts.length} | ${cells.join(" | ")} |`);
  }
  console.log(`\n${calls} consolidations · ${promptTokens.toLocaleString()} prompt tokens · ${memory.facts.length} facts at the end${memory.archive ? ` · ${memory.archive.length} in the archive` : ""}`);
  for (const s of expect) {
    const said = firstSaid(s);
    const seen = snapshots.filter((x) => said !== undefined && x.memory.coveredUpTo >= said);
    const lost = seen.filter((x) => !has(x.memory, s)).map((x) => x.afterTurnIndex);
    console.log(`- "${s}": first said at turn ${said ?? "never"}; in memory at ${seen.length - lost.length}/${seen.length} checkpoints${lost.length ? `; missing after turns ${lost.join(", ")}` : ""}`);
  }
  console.log("\nFinal facts:");
  for (const f of memory.facts) console.log(`- ${f.text}${f.source === "unknown" ? "" : `  [${f.source}${f.by ? `: ${f.by}` : ""}]`}`);
  if (rules === "v3") {
    const bySource: Record<string, number> = {};
    for (const f of memory.facts) bySource[f.source] = (bySource[f.source] ?? 0) + 1;
    console.log(`\nSources at the end: ${Object.entries(bySource).map(([k, v]) => `${k} ${v}`).join(" · ")}`);
  }
  const out = flag("out");
  if (out) {
    writeFileSync(out, JSON.stringify(snapshots, null, 2));
    console.log(`\nsnapshots → ${out}`);
  }
})();
