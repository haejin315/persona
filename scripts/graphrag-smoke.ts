/**
 * Graph RAG smoke against the live stores (docs/14): Qdrant + Neo4j from docker-compose.yml,
 * multilingual-e5-small locally. Ingests a real 1000-turn recording (its 2,000 transcript turns
 * as chunks, its archive facts as vectors; the archive predates triples, so the graph gets the
 * triples the summariser would have produced for a few of them) and times retrieval for the
 * scenario's probe questions.
 *
 *   npx tsx scripts/graphrag-smoke.ts [eval-results/long-run-1000/conversations/<record>.json]
 */
import { readFileSync } from "node:fs";
import { Neo4jGraph } from "../packages/engine/src/memory/graphdb.js";
import { GraphRag, renderGraphRag } from "../packages/engine/src/memory/graphrag.js";
import { localEmbedder } from "../packages/engine/src/memory/recall.js";
import { QdrantIndex } from "../packages/engine/src/memory/vectordb.js";
import type { ArchivedFact, Turn } from "../packages/engine/src/types.js";

const file = process.argv[2] ?? "eval-results/long-run-1000-recall2/conversations/haru-ko-1000__v13-ledger25-recall-e5v2-sonnet__r0.json";
const rec = JSON.parse(readFileSync(file, "utf8")) as { turns: Turn[]; memory: { archive: ArchivedFact[] } };
const ms = (t0: number) => `${Math.round(performance.now() - t0)} ms`;

const ns = "smoke";
const embedder = localEmbedder();
const vectors = new QdrantIndex();
const graph = new Neo4jGraph(ns);
const rag = new GraphRag({ ns, embedder, vectors, graph, chunkK: 3, factK: 5, hops: 2, edgeLimit: 12, chunking: { user: "민수", character: "하루" } });

let t0 = performance.now();
await embedder.warm?.();
console.log(`e5 warm ${ms(t0)}`);
t0 = performance.now();
await rag.clear();
console.log(`cleared ${ns} ${ms(t0)}`);

// Triples for a handful of archive facts (what rules v5 asks the summariser for).
const triples: Record<string, { subject: string; relation: string; object: string }> = {
  "하늘 동물병원": { subject: "보리", relation: "다닌다", object: "하늘 동물병원" },
  "성수동": { subject: "민수", relation: "산다", object: "성수동" },
  "망원동": { subject: "민수", relation: "이사한다", object: "망원동" },
  "브롬톤": { subject: "민수", relation: "탄다", object: "브롬톤" },
  "라떼": { subject: "민수", relation: "마신다", object: "라떼" },
  "지연": { subject: "지연 선배", relation: "제안했다", object: "스튜디오 합류" },
};
const archive = rec.memory.archive.map((f) => {
  const key = Object.keys(triples).find((k) => f.text.includes(k));
  return key ? { ...f, triple: triples[key] } : f;
});

t0 = performance.now();
for (let i = 0; i < archive.length; i += 128) await rag.ingestFacts(archive.slice(i, i + 128));
console.log(`facts ${archive.length} ingested ${ms(t0)} (triples on ${archive.filter((f) => f.triple).length})`);
t0 = performance.now();
const last = rec.turns.at(-1)!.index;
let n = 0;
for (let a = 0; a <= last; a += 200) n += await rag.ingestTurns(rec.turns, a, Math.min(last, a + 199));
console.log(`chunks ${n} from ${rec.turns.length} turns ingested ${ms(t0)}`);
console.log("counts", await rag.counts());

const questions = [
  "보리가 다니는 동물병원 이름 기억하세요?",
  "제가 어디로 이사한다고 했죠?",
  "자전거 뭐 샀다고 했는지 기억나세요?",
  "요즘 제가 뭐 마시는지 아세요?",
  "제 사수 선배 얘기 했었잖아요, 그분 이름 기억나세요?",
  "오늘 날씨가 참 좋네요.",
];
for (const q of questions) {
  t0 = performance.now();
  const r = await rag.retrieve(q, { turnToMax: last - 20, ignore: ["하루", "민수"] });
  const dt = ms(t0);
  console.log(`\nQ: ${q}  (${dt}; seeds ${JSON.stringify(r.seeds)}, edges ${r.edges.length}, facts ${r.facts.length}, chunks ${r.chunks.length})`);
  for (const e of r.edges.slice(0, 4)) console.log(`   G d${e.depth} t${e.turnTo} ${e.subject} —${e.relation}→ ${e.object}${e.current ? "" : " (no longer)"}`);
  for (const f of r.facts.slice(0, 3)) console.log(`   F ${f.score.toFixed(3)} t${f.turnTo} ${f.text.slice(0, 70)}`);
  for (const c of r.chunks.slice(0, 2)) console.log(`   C ${c.score.toFixed(3)} t${c.turnFrom}-${c.turnTo} ${c.text.replace(/\n/g, " | ").slice(0, 110)}`);
}
const block = renderGraphRag(await rag.retrieve(questions[0]!, { turnToMax: last - 20, ignore: ["하루", "민수"] }), rec.turns.length, "민수");
console.log(`\nblock chars ${block.length}, first 600:\n${block.slice(0, 600)}`);
await rag.close();
