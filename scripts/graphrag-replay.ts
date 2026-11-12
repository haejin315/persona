/**
 * Replays Graph RAG retrieval over a finished recording against the stores that run left behind
 * (docs/14 round 8): the same user messages, the same carried context, other retrieval options.
 * For every probe of the scenario it reports whether the expected answer is in what was retrieved,
 * how many seeds the message woke and how long the stores took - so a retrieval change can be
 * measured in minutes on a 5,000-turn graph instead of in another 30-hour run.
 *
 *   npx tsx scripts/graphrag-replay.ts eval-results/long-run-5000/conversations/<record>.json \
 *     [--variants '[{}, {"maxSeeds": 16}]'] [--base '{"chunkK":3,...}'] [--sample 50]
 *
 * The stores hold the whole run, so vectors are filtered to the turns the summariser had covered
 * at the probe (as in the run); graph edges carry no turn filter in the store, so edges from after
 * the probe are dropped afterwards and "(no longer)" reflects the end of the run. Every variant
 * sees the same stores, so the comparison between variants is fair; absolute numbers are not the
 * run's.
 */
import { readFileSync } from "node:fs";
import { Neo4jGraph } from "../packages/engine/src/memory/graphdb.js";
import { GraphRag, type GraphRagOptions } from "../packages/engine/src/memory/graphrag.js";
import { localEmbedder } from "../packages/engine/src/memory/recall.js";
import { QdrantIndex } from "../packages/engine/src/memory/vectordb.js";
import type { ConversationRecord } from "../packages/eval/src/types.js";
import { findScenario } from "../packages/eval/src/scenarios.js";

const args = process.argv.slice(2);
const flag = (name: string, dflt?: string) => {
  const i = args.indexOf(`--${name}`);
  return i === -1 ? dflt : args[i + 1];
};
const file = args.find((a) => !a.startsWith("--") && a.endsWith(".json"));
if (!file) throw new Error("record file required");
const rec = JSON.parse(readFileSync(file, "utf8")) as ConversationRecord;
const scenario = findScenario(rec.scenarioId);
const ns = flag("ns", rec.id.toLowerCase().replace(/[^a-z0-9]+/g, "-"))!;
const armRag = (rec.arm as { graphRag?: Record<string, unknown> }).graphRag ?? {};
const { at: _at, render: _render, carry, ...armOpts } = armRag as { at?: string; render?: string; carry?: number } & Record<string, unknown>;
const base = { ...armOpts, ...JSON.parse(flag("base", "{}")!) } as Partial<GraphRagOptions>;
const variants = JSON.parse(flag("variants", '[{}, {"maxSeeds": 16}, {"maxTokenNames": 40}, {"maxSeeds": 16, "maxTokenNames": 40}, {"tokenLinking": false}]')!) as Partial<GraphRagOptions>[];
const sample = Number(flag("sample", "0"));
const norm = (s: string) => s.replace(/\s+/g, "");
const bands = [
  [30, 300],
  [300, 1500],
  [1500, 99999],
] as const;

const users = rec.turns.filter((t) => t.role === "user");
const byIndex = new Map(rec.turns.map((t) => [t.index, t]));
const charName = rec.turns.find((t) => t.role === "character") ? "하루" : "하루";
const ignore = [charName, rec.persona.name];
const sideFile = `packages/eval/scenarios/${rec.scenarioId}.facts.json`;
let side: { facts: { expected: string; atUserTurn: number; replaces?: string }[] } | undefined;
try {
  side = JSON.parse(readFileSync(sideFile, "utf8"));
} catch {
  side = undefined;
}
interface Case {
  userTurn: number;
  message: string;
  context: string[];
  turnToMax: number;
  expected?: string;
  distance?: number;
  changed?: boolean;
}
const cases: Case[] = [];
for (const p of scenario.probes) {
  const u = users[p.atUserTurn - 1];
  if (!u) continue;
  const prev = rec.turns.filter((t) => t.role === "character" && t.index < u.index).slice(-(carry ?? 0));
  const snap = rec.memorySnapshots.filter((s) => s.afterTurnIndex <= u.index).slice(-1)[0];
  const revealed = side?.facts.filter((f) => f.expected === p.expected && f.atUserTurn < p.atUserTurn).sort((a, b) => b.atUserTurn - a.atUserTurn)[0];
  const fromBeats = scenario.beats.filter((b) => b.fromUserTurn < p.atUserTurn && b.reveal.some((f) => f.includes(p.expected))).map((b) => b.fromUserTurn).slice(-1)[0];
  const revealedAt = revealed?.atUserTurn ?? fromBeats;
  cases.push({ userTurn: p.atUserTurn, message: u.content, context: prev.map((t) => t.content), turnToMax: snap?.memory.coveredUpTo ?? 0, expected: p.expected, distance: revealedAt !== undefined ? p.atUserTurn - revealedAt : undefined, changed: !!revealed?.replaces });
}
if (sample > 0) {
  const step = Math.max(1, Math.floor(users.length / sample));
  for (let i = step; i < users.length; i += step) {
    const u = users[i]!;
    if (cases.some((c) => c.userTurn === i + 1)) continue;
    const prev = rec.turns.filter((t) => t.role === "character" && t.index < u.index).slice(-(carry ?? 0));
    const snap = rec.memorySnapshots.filter((s) => s.afterTurnIndex <= u.index).slice(-1)[0];
    cases.push({ userTurn: i + 1, message: u.content, context: prev.map((t) => t.content), turnToMax: snap?.memory.coveredUpTo ?? 0 });
  }
}
cases.sort((a, b) => a.userTurn - b.userTurn);

const embedder = localEmbedder(base.embedModel as string | undefined);
const vectors = new QdrantIndex();
const graph = new Neo4jGraph(ns);
await embedder.warm?.();
console.log(`${rec.id}: ns ${ns}, ${cases.length} cases (${cases.filter((c) => c.expected).length} probes), base ${JSON.stringify(base)}`);
const recorded = cases.map((c) => {
  const u = users[c.userTurn - 1]!;
  const reply = byIndex.get(u.index + 1);
  const m = reply?.meta as { retrieval?: { seeds: string[]; ms: number }; recalled?: { text: string }[] } | undefined;
  return { seeds: m?.retrieval?.seeds.length, ms: m?.retrieval?.ms, hit: c.expected && m?.recalled ? m.recalled.some((x) => norm(x.text).includes(norm(c.expected!))) : undefined };
});
const probeCases = cases.map((c, i) => ({ c, i })).filter(({ c }) => c.expected);
const bandStats = (hits: (boolean | undefined)[]) =>
  bands.map(([lo, hi]) => {
    const ps = probeCases.filter(({ c }) => c.distance !== undefined && c.distance >= lo && c.distance <= hi);
    return `${lo}-${hi}: ${ps.filter(({ i }) => hits[i]).length}/${ps.length}`;
  }).join("  ") + `  changed: ${probeCases.filter(({ c, i }) => c.changed && hits[i]).length}/${probeCases.filter(({ c }) => c.changed).length}`;
const mean = (a: number[]) => (a.length ? Math.round(a.reduce((s, x) => s + x, 0) / a.length) : NaN);
const med = (a: number[]) => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1]! : NaN; };
console.log(`recorded run:   ${bandStats(recorded.map((r) => r.hit))}  seeds med ${med(recorded.map((r) => r.seeds ?? 0))}  ms med ${med(recorded.map((r) => r.ms ?? 0))}`);

for (const v of variants) {
  const rag = new GraphRag({ ...(base as GraphRagOptions), ...v, ns, embedder, vectors, graph });
  const hits: (boolean | undefined)[] = [];
  const seedsN: number[] = [];
  const msN: number[] = [];
  const flips: string[] = [];
  for (const [i, c] of cases.entries()) {
    const t0 = performance.now();
    const r = await rag.retrieve(c.message, { turnToMax: c.turnToMax, ignore, context: c.context });
    msN.push(Math.round(performance.now() - t0));
    seedsN.push(r.seeds.length);
    if (!c.expected) {
      hits.push(undefined);
      continue;
    }
    const limit = c.turnToMax + 20;
    const texts = [...r.edges.filter((e) => e.turnTo <= limit).map((e) => `${e.subject} ${e.relation} ${e.object}: ${e.text}`), ...r.facts.map((f) => f.text), ...r.chunks.map((ch) => ch.text)];
    const hit = texts.some((t) => norm(t).includes(norm(c.expected!)));
    hits.push(hit);
    if (recorded[i]!.hit !== undefined && hit !== recorded[i]!.hit) flips.push(`t${c.userTurn}[${c.expected}] ${recorded[i]!.hit ? "lost" : "gained"} (seeds ${recorded[i]!.seeds}→${r.seeds.length})`);
  }
  console.log(`${JSON.stringify(v).padEnd(44)} ${bandStats(hits)}  seeds med ${med(seedsN)} max ${Math.max(...seedsN)}  ms med ${med(msN)} mean ${mean(msN)} max ${Math.max(...msN)}`);
  if (flips.length) console.log(`   vs recorded: ${flips.join("; ")}`);
}
await graph.close?.();
await vectors.close?.();
