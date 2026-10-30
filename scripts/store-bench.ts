/**
 * Store-only scale benchmark (docs/13, measurement A). No LLM. Seeds from a real 1000-turn archive
 * and grows it to the entry count a conversation of N user turns would produce, then times the
 * queries a reply would make: the trigger check, one entity page, a narrowed vector search, and
 * (for comparison only) a full scan.
 *
 *   npx tsx scripts/store-bench.ts --turns 10000 [--embedder e5|bigram] [--keep] [--out path]
 */
import { existsSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { bigramEmbedder, localEmbedder } from "../packages/engine/src/memory/recall.js";
import { MemoryStore } from "../packages/engine/src/memory/store.js";
import type { ArchivedFact } from "../packages/engine/src/types.js";

const arg = (name: string, dflt: string) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1]! : dflt;
};
const turns = Number(arg("turns", "10000"));
const embedderName = arg("embedder", "e5");
const keep = process.argv.includes("--keep");
const out = arg("out", join(tmpdir(), `persona-store-bench-${turns}.sqlite`));
const SEED = "eval-results/long-run-1000-recall2/conversations/haru-ko-1000__v13-ledger25-recall-e5v2-sonnet__r0.json";
const SPEAKERS = new Set(["하루", "민수"]);

const ms = (t0: number) => Math.round((performance.now() - t0) * 10) / 10;
const median = (xs: number[]) => xs.slice().sort((a, b) => a - b)[xs.length >> 1]!;
const time = (fn: () => unknown, n = 20) => {
  const xs: number[] = [];
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    fn();
    xs.push(performance.now() - t0);
  }
  return Math.round(median(xs) * 100) / 100;
};
const mb = (b: number) => `${(b / 1024 / 1024).toFixed(0)} MB`;

// ---- seed -------------------------------------------------------------------------------------
const seed = (JSON.parse(readFileSync(SEED, "utf8")) as { memory: { archive: ArchivedFact[] } }).memory.archive;
const seedUserTurns = 1000;
const perTurn = seed.length / seedUserTurns;
const copies = Math.max(1, Math.round(turns / seedUserTurns));
const target = seed.length * copies;
console.log(`seed ${seed.length} entries from ${seedUserTurns} user turns (${perTurn.toFixed(2)}/turn) → ${copies} copies = ${target.toLocaleString()} entries for ${turns.toLocaleString()} user turns`);

const embedder = embedderName === "bigram" ? bigramEmbedder(384) : localEmbedder();
let t0 = performance.now();
await embedder.warm?.();
console.log(`embedder ${embedder.id} warm ${ms(t0)} ms`);
t0 = performance.now();
const seedVecs: number[][] = [];
for (let i = 0; i < seed.length; i += 64) seedVecs.push(...(await embedder.embed(seed.slice(i, i + 64).map((f) => f.text), "passage")));
console.log(`seed embeddings ${ms(t0)} ms (${seedVecs[0]!.length} dims)`);

// Gaussian noise on a seed vector, renormalised: a different fact in the same neighbourhood.
let rnd = 12345;
const rand = () => ((rnd = (rnd * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const gauss = () => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
const perturb = (v: number[], sigma: number) => {
  const w = v.map((x) => x + sigma * gauss());
  const n = Math.sqrt(w.reduce((a, x) => a + x * x, 0)) || 1;
  return w.map((x) => x / n);
};

// ---- load ---------------------------------------------------------------------------------------
if (existsSync(out)) rmSync(out);
const store = new MemoryStore(out);
const rss0 = process.memoryUsage().rss;
t0 = performance.now();
const BATCH = 5000;
let batch: ArchivedFact[] = [];
let vecs: number[][] = [];
const flush = () => {
  if (batch.length) store.appendFacts(batch, vecs);
  batch = [];
  vecs = [];
};
for (let c = 0; c < copies; c++) {
  const shift = c * seedUserTurns * 2;
  for (let i = 0; i < seed.length; i++) {
    const f = seed[i]!;
    const ents = (f.entities ?? []).map((e) => (SPEAKERS.has(e) || c === 0 ? e : `${e}${c}`));
    let text = f.text;
    if (c > 0) for (const e of f.entities ?? []) if (!SPEAKERS.has(e)) text = text.split(e).join(`${e}${c}`);
    batch.push({ ...f, text, turnFrom: f.turnFrom + shift, turnTo: f.turnTo + shift, ...(ents.length ? { entities: ents } : {}) });
    vecs.push(c === 0 ? seedVecs[i]! : perturb(seedVecs[i]!, 0.05));
    if (batch.length >= BATCH) flush();
  }
  if (c % 50 === 0 && c > 0) process.stdout.write(`  copy ${c}/${copies} ${ms(t0)} ms\n`);
}
flush();
const loadMs = ms(t0);
const n = store.factCount();
const nEnt = store.entities().length;
const size = statSync(out).size;
console.log(`loaded ${n.toLocaleString()} facts, ${nEnt.toLocaleString()} entities in ${(loadMs / 1000).toFixed(1)} s · file ${mb(size)} · rss ${mb(process.memoryUsage().rss)} (was ${mb(rss0)})`);

// ---- queries ------------------------------------------------------------------------------------
const lastCopy = copies - 1;
const suffix = lastCopy === 0 ? "" : String(lastCopy);
const lastTurn = copies * seedUserTurns * 2;
const message = `보리${suffix}는 잘 지내요? 하늘 동물병원${suffix}은 또 갔어요?`;
const ledger = "- 민수는 아침마다 아메리카노를 마신다\n- 민수는 성수동에서 망원동으로 이사했다";
const [qv] = await embedder.embed([message], "query");

const tTrigger = time(() => store.entitiesIn(message, { ledger, ignore: ["하루", "민수"], maxShare: 0.05, minFacts: 2 }));
const fired = store.entitiesIn(message, { ledger, ignore: ["하루", "민수"], maxShare: 0.05, minFacts: 2 }).map((e) => `${e.name}(${e.nFacts})`);
const tPage = time(() => store.entityPage(`보리${suffix}`));
const pageLen = store.entityPage(`보리${suffix}`).length;
const tPageOld = time(() => store.entityPage("보리"));
const tVecEntity = time(() => store.vectorSearch(qv!, { k: 5, entity: `보리${suffix}` }));
const tVecRange = time(() => store.vectorSearch(qv!, { k: 5, turnFrom: lastTurn - 2000, turnTo: lastTurn }));
const rangeN = store.factsInRange(lastTurn - 2000, lastTurn).length;
const tVecRange10 = time(() => store.vectorSearch(qv!, { k: 5, turnFrom: lastTurn - 20000, turnTo: lastTurn }), 5);
const range10N = store.factsInRange(Math.max(0, lastTurn - 20000), lastTurn).length;
const tFull = time(() => store.vectorSearch(qv!, { k: 5 }), n > 500000 ? 1 : 3);
const top = store.vectorSearch(qv!, { k: 1 })[0];
const rssEnd = process.memoryUsage().rss;

console.log(`
| user turns | entries | entities | load | file | rss after queries | trigger check | entity page (${pageLen} rows) | page, copy 0 | vector · entity | vector · last 2,000 turns (${rangeN}) | vector · last 20,000 turns (${range10N}) | vector · full scan |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ${turns.toLocaleString()} | ${n.toLocaleString()} | ${nEnt.toLocaleString()} | ${(loadMs / 1000).toFixed(1)} s | ${mb(size)} | ${mb(rssEnd)} | ${tTrigger} ms | ${tPage} ms | ${tPageOld} ms | ${tVecEntity} ms | ${tVecRange} ms | ${tVecRange10} ms | ${tFull} ms |
`);
console.log(`trigger fired on: ${fired.join(", ") || "(nothing)"} · full-scan top-1: "${top?.fact.text.slice(0, 60)}" cos ${top?.cos.toFixed(3)}`);
store.close();
if (!keep) rmSync(out, { force: true });
else console.log(`kept ${out}`);
