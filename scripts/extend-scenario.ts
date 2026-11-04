/**
 * Extends a scenario to N user turns (docs/13, plan C). A model writes the new beats a thousand
 * turns at a time, seeing every fact revealed so far and the last beats for voice; the probes are
 * then placed mechanically (packages/eval/src/extend.ts) over the whole fact list, so the longest
 * recall distance is a design parameter, not luck.
 *
 *   npx tsx scripts/extend-scenario.ts --from haru-ko-1000 --to 5000 --id haru-ko-5000 \
 *     --out packages/eval/scenarios/haru-ko-5000.yaml [--model claude-cli:opus] [--every 15] [--probes 60] [--seed 5000]
 *
 * Writes the YAML plus a sidecar <out>.facts.json (every fact with expected/ask/replaces) so the
 * probe layout can be re-placed without calling the model again (--replace-only).
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import YAML from "yaml";
import { findCharacter } from "@persona/engine";
import { chunkPrompt, parseBeatsJson, placeProbes, type FactEntry, type GeneratedBeat } from "../packages/eval/src/extend.js";
import { resolveProvider } from "../packages/eval/src/providers.js";
import { findScenario } from "../packages/eval/src/scenarios.js";
import { Scenario } from "../packages/eval/src/types.js";

const args = process.argv.slice(2);
const flag = (name: string, dflt?: string) => {
  const i = args.indexOf(`--${name}`);
  return i === -1 ? dflt : args[i + 1];
};
const fromId = flag("from", "haru-ko-1000")!;
const toTurns = Number(flag("to", "5000"));
const newId = flag("id", `${fromId.replace(/-\d+$/, "")}-${toTurns}`)!;
const out = flag("out", `packages/eval/scenarios/${newId}.yaml`)!;
const modelSpec = flag("model", "claude-cli:opus")!;
const every = Number(flag("every", "15"));
const probeCount = Number(flag("probes", "60"));
const seed = Number(flag("seed", String(toTurns)));
const replaceOnly = args.includes("--replace-only");
const factsFile = out.replace(/\.ya?ml$/, "") + ".facts.json";

const src = findScenario(fromId);
const card = findCharacter(src.characterId) as unknown as { description?: Record<string, string>; summary?: Record<string, string> };
const blurb = card.description?.[src.lang] ?? card.summary?.[src.lang] ?? "";
/** Last user turn the source scenario was written for (its beats end before it). */
const srcEnd = Number(flag("src-turns", String(Math.ceil(Math.max(...src.beats.map((b) => b.fromUserTurn)) / 100) * 100)));
const srcTurns = srcEnd;
const { provider, model } = resolveProvider(modelSpec);

async function askJson(system: string, user: string): Promise<string> {
  const res = await provider.generate([{ role: "system", content: system }, { role: "user", content: user }], { model, temperature: 0.7 });
  return res.text;
}

/** The source scenario's reveals are plain sentences; ask the model for expected/ask (and replaces) for each. */
async function annotateSource(): Promise<FactEntry[]> {
  const reveals = src.beats.flatMap((b) => b.reveal.map((fact) => ({ fact, atUserTurn: b.fromUserTurn })));
  const examples = src.probes.map((p) => `- expected "${p.expected}": ${p.ask}`).join("\n");
  const system = `You annotate facts from a roleplay evaluation scenario so that a simulated user can later ask the character about them. Respond with JSON only: {"facts": [{"fact": "<copied exactly>", "expected": "<shortest answer: one proper noun, number, date or title as the fact writes it>", "ask": "<Korean instruction in the style of the examples>", "replaces": "<expected of an earlier fact this one changes, or omit>", "probeable": true|false}]}. probeable is false for facts with no crisp answer (a feeling, "the deadline went fine"). Keep the order and the count; one entry per input fact.`;
  const user = `Example asks already in the scenario:\n${examples}\n\nFACTS (in story order, with the user turn where each is revealed):\n${reveals.map((r, i) => `${i + 1}. [turn ${r.atUserTurn}] ${r.fact}`).join("\n")}\n\nJSON only.`;
  const text = await askJson(system, user);
  const s = text.trim().replace(/^```(?:json)?\s*|\s*```$/g, "");
  const parsed = JSON.parse(s.slice(s.indexOf("{"), s.lastIndexOf("}") + 1)) as { facts: any[] };
  if (!Array.isArray(parsed.facts) || parsed.facts.length !== reveals.length) throw new Error(`annotate: got ${parsed.facts?.length} entries for ${reveals.length} facts`);
  const outFacts: FactEntry[] = [];
  parsed.facts.forEach((f, i) => {
    if (f.probeable === false) return;
    if (typeof f.expected !== "string" || typeof f.ask !== "string") return;
    outFacts.push({ fact: reveals[i]!.fact, expected: f.expected.trim(), ask: f.ask.trim(), atUserTurn: reveals[i]!.atUserTurn, ...(typeof f.replaces === "string" && f.replaces.trim() ? { replaces: f.replaces.trim() } : {}) });
  });
  return outFacts;
}

async function main() {
  let facts: FactEntry[];
  let beats: GeneratedBeat[];
  if (replaceOnly && existsSync(factsFile)) {
    const saved = JSON.parse(readFileSync(factsFile, "utf8")) as { facts: FactEntry[]; beats: GeneratedBeat[] };
    facts = saved.facts;
    beats = saved.beats;
    console.log(`loaded ${facts.length} facts and ${beats.length} generated beats from ${factsFile}`);
  } else {
    console.log(`annotating ${src.beats.flatMap((b) => b.reveal).length} source facts with ${modelSpec}…`);
    facts = await annotateSource();
    console.log(`  ${facts.length} probeable, ${facts.filter((f) => f.replaces).length} marked as changes`);
    beats = [];
    let lastBeats: GeneratedBeat[] = src.beats.slice(-8).map((b) => ({ fromUserTurn: b.fromUserTurn, goal: b.goal, reveal: b.reveal.map((fact) => ({ fact, expected: "", ask: "" })) }));
    const chunk = 1000;
    for (let from = srcEnd + 1; from <= toTurns; from += chunk) {
      const to = Math.min(from + chunk - 1, toTurns);
      const spec = { from, to, every, facts: 70, changes: 4, yearsHint: "about two" };
      const { system, user } = chunkPrompt(src, blurb, facts, lastBeats, spec);
      console.log(`beats ${from}-${to}: asking ${modelSpec} (${facts.length} known facts)…`);
      const t0 = Date.now();
      let got: GeneratedBeat[] | undefined;
      for (let attempt = 0; attempt < 3 && !got; attempt++) {
        try {
          got = parseBeatsJson(await askJson(system, user));
        } catch (e) {
          console.log(`  attempt ${attempt + 1} failed: ${(e as Error).message.slice(0, 200)}`);
        }
      }
      if (!got) throw new Error(`no beats for ${from}-${to}`);
      got.sort((a, b) => a.fromUserTurn - b.fromUserTurn);
      // Keep the turn grid the prompt asked for even if the model drifted.
      got.forEach((b, i) => (b.fromUserTurn = from + i * every));
      got = got.filter((b) => b.fromUserTurn <= to);
      console.log(`  ${got.length} beats, ${got.flatMap((b) => b.reveal).length} facts (${got.flatMap((b) => b.reveal).filter((r) => r.replaces).length} changes) in ${Math.round((Date.now() - t0) / 1000)} s`);
      for (const b of got) for (const r of b.reveal) facts.push({ fact: r.fact, expected: r.expected, ask: r.ask, atUserTurn: b.fromUserTurn, ...(r.replaces ? { replaces: r.replaces } : {}) });
      beats.push(...got);
      lastBeats = got.slice(-8);
    }
    writeFileSync(factsFile, JSON.stringify({ facts, beats }, null, 2));
  }

  // Probes: the source scenario's own stay; new ones go over [srcTurns+1, toTurns].
  const span = toTurns - srcTurns;
  const bands = [
    { min: 30, max: 300, count: Math.round(probeCount / 3) },
    { min: 300, max: 1500, count: Math.round(probeCount / 3) },
    { min: 1500, max: Math.max(1600, span - 200), count: probeCount - 2 * Math.round(probeCount / 3) },
  ];
  // Never ask about things that move with time (an age, "this year's best moment", "what I read these days").
  const excludeRe = new RegExp(flag("exclude", "살$|올해|요즘|이번 주|지난 주|이번 달|다음 달")!);
  const placed = placeProbes(facts, { from: beats[0]!.fromUserTurn, to: toTurns, every, offset: Math.floor(every / 2), bands, margin: 20, preferChanged: Math.round(probeCount / 6), seed, exclude: (f) => excludeRe.test(f.expected) || excludeRe.test(f.fact) });
  const probes = [...src.probes, ...placed.map((p) => ({ atUserTurn: p.atUserTurn, ask: p.ask, expected: p.expected }))].sort((a, b) => a.atUserTurn - b.atUserTurn);

  const scenario = {
    ...src,
    id: newId,
    beats: [...src.beats, ...beats.map((b) => ({ fromUserTurn: b.fromUserTurn, goal: b.goal, reveal: b.reveal.map((r) => r.fact) }))],
    probes,
  };
  const parsed = Scenario.safeParse(scenario);
  if (!parsed.success) throw new Error(`generated scenario invalid:\n${parsed.error.issues.map((i) => `  - ${i.path.join(".")}: ${i.message}`).join("\n")}`);
  for (let i = 1; i < scenario.beats.length; i++) if (scenario.beats[i]!.fromUserTurn <= scenario.beats[i - 1]!.fromUserTurn) throw new Error(`beats not increasing at ${i}`);

  const header = `# ${newId}: ${fromId} carried on to ${toTurns} user turns (docs/13, plan C). Beats after turn ${srcTurns} were written by\n# ${modelSpec} from the facts revealed so far (scripts/extend-scenario.ts); probes after turn ${srcTurns} were placed by\n# packages/eval/src/extend.ts over every fact with distance bands 30-300 / 300-1500 / 1500+ user turns, changed facts\n# asked only for their latest value. Facts and the generated beats are in ${factsFile.split("/").pop()}.\n`;
  writeFileSync(out, header + YAML.stringify(scenario, { lineWidth: 0 }));
  const dist = placed.map((p) => p.distance);
  console.log(`wrote ${out}: ${scenario.beats.length} beats, ${scenario.probes.length} probes (${placed.length} new: ${placed.filter((p) => p.changed).length} on changed facts; distance min ${Math.min(...dist)} median ${dist.slice().sort((a, b) => a - b)[dist.length >> 1]} max ${Math.max(...dist)})`);
  for (const b of bands) console.log(`  band ${b.min}-${b.max}: ${placed.filter((p) => p.distance >= b.min && p.distance <= b.max).length}/${b.count}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
