import { describe, expect, it } from "vitest";
import { loadAllCharacters } from "../characters.js";
import type { LLMProvider } from "../providers/types.js";
import { ChatSession } from "../session.js";
import { emptyMemory, type ArchivedFact, type Turn } from "../types.js";
import { chunkAll, chunkRange } from "./chunker.js";
import { MemoryGraph } from "./graphdb.js";
import { GraphRag, renderGraphRag } from "./graphrag.js";
import { bigramEmbedder } from "./recall.js";
import { MemoryVectorIndex, uuidFor } from "./vectordb.js";

const cards = loadAllCharacters();
const turn = (index: number, role: Turn["role"], content: string): Turn => ({ index, role, content, ts: index });
const transcript: Turn[] = [];
for (let i = 0; i < 12; i++) transcript.push(turn(i, i % 2 === 0 ? "user" : "character", i === 4 ? "보리가 하늘 동물병원에 다녀왔어요. 네 살이에요." : `turn ${i}`));

const facts: ArchivedFact[] = [
  { text: "민수는 고양이 보리를 키운다", source: "user", turnFrom: 0, turnTo: 3, entities: ["민수", "보리"], triple: { subject: "민수", relation: "키운다", object: "보리" } },
  { text: "보리는 하늘 동물병원에 다닌다", source: "user", turnFrom: 4, turnTo: 7, entities: ["보리", "하늘 동물병원"], triple: { subject: "보리", relation: "다닌다", object: "하늘 동물병원" } },
  { text: "민수는 아침마다 아메리카노를 마신다", source: "user", turnFrom: 0, turnTo: 3, triple: { subject: "민수", relation: "마신다", object: "아메리카노" } },
  { text: "민수는 위가 아파서 라떼로 바꿨다", source: "user", turnFrom: 8, turnTo: 11, triple: { subject: "민수", relation: "마신다", object: "라떼" } },
  { text: "하늘 동물병원은 성수동에 있다", source: "user", turnFrom: 8, turnTo: 11, triple: { subject: "하늘 동물병원", relation: "있다", object: "성수동" } },
];

describe("Graph RAG memory (docs/14)", () => {
  it("chunks the transcript on a fixed grid of overlapping exchanges, once per window", () => {
    const all = chunkAll(transcript, { exchanges: 2, overlap: 1 });
    expect(all.map((c) => c.id)).toEqual(["c0-3", "c2-5", "c4-7", "c6-9", "c8-11"]);
    expect(all[1]!.text.split("\n")).toHaveLength(4);
    // The windows ending in a range are produced exactly once across consecutive ranges.
    const a = chunkRange(transcript, 0, 5).map((c) => c.id);
    const b = chunkRange(transcript, 6, 11).map((c) => c.id);
    expect([...a, ...b]).toEqual(all.map((c) => c.id));
    expect(uuidFor("c0-3")).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(uuidFor("c0-3")).toBe(uuidFor("c0-3"));
  });

  it("the graph keeps one edge per triple, marks a replaced object as no longer current, and walks hops", async () => {
    const g = new MemoryGraph();
    await g.addTriples(facts.map((f) => ({ ...f.triple!, text: f.text, turnFrom: f.turnFrom, turnTo: f.turnTo })));
    expect(await g.count()).toEqual({ nodes: 6, edges: 5 });
    const one = await g.neighborhood(["보리"], { hops: 1, limit: 10 });
    expect(one.map((e) => e.object).sort()).toEqual(["보리", "하늘 동물병원"]);
    const two = await g.neighborhood(["보리"], { hops: 2, limit: 10 });
    expect(two.some((e) => e.object === "성수동" && e.depth === 2)).toBe(true);
    expect(two.some((e) => e.object === "라떼" && e.depth === 2)).toBe(true);
    const drinks = two.filter((e) => e.relation === "마신다");
    expect(drinks.find((e) => e.object === "아메리카노")!.current).toBe(false);
    expect(drinks.find((e) => e.object === "라떼")!.current).toBe(true);
  });

  it("retrieves graph edges, nearest facts and verbatim chunks for a message, and renders one block", async () => {
    const rag = new GraphRag({ ns: "t", embedder: bigramEmbedder(256), vectors: new MemoryVectorIndex(), graph: new MemoryGraph(), chunkK: 2, factK: 3, hops: 2 });
    await rag.ingestFacts(facts);
    expect(await rag.ingestTurns(transcript, 0, 11)).toBe(5);
    expect(await rag.counts()).toEqual({ chunks: 5, facts: 5, nodes: 6, edges: 5 });
    const r = await rag.retrieve("보리 병원은 어디였죠?", { turnToMax: 7, ignore: ["하루", "민수"] });
    expect(r.seeds).toEqual(["보리"]);
    expect(r.edges.some((e) => e.object === "하늘 동물병원")).toBe(true);
    expect(r.chunks.every((c) => c.turnTo <= 7)).toBe(true);
    expect(r.chunks[0]!.text).toContain("하늘 동물병원");
    expect(r.facts).toHaveLength(3);
    const block = renderGraphRag(r, 40, "민수");
    expect(block).toContain("KNOWLEDGE GRAPH");
    expect(block).toContain("보리 —다닌다→ 하늘 동물병원");
    expect(block).toContain("RELATED PAST EXCHANGES");
    expect(block).toMatch(/\[about \d+ exchanges ago\]/);
    // A hub (a node on more than a tenth of all edges, once there are more than 10) seeds nothing.
    const hubby = new MemoryGraph();
    const many = Array.from({ length: 30 }, (_, i) => ({ subject: "오늘", relation: "날씨", object: `날씨${i}`, text: `오늘 날씨 ${i}`, turnFrom: i, turnTo: i }));
    await hubby.addTriples([...many, { subject: "보리", relation: "다닌다", object: "하늘 동물병원", text: "x", turnFrom: 0, turnTo: 1 }]);
    const h = new GraphRag({ ns: "h", embedder: bigramEmbedder(256), vectors: new MemoryVectorIndex(), graph: hubby });
    expect((await h.retrieve("오늘 보리 어때요", {})).seeds).toEqual(["보리"]);
    // Speaker names seed nothing; with no seed and no stores touched, the block is empty.
    const empty = new GraphRag({ ns: "e", embedder: bigramEmbedder(256), vectors: new MemoryVectorIndex(), graph: new MemoryGraph() });
    expect(renderGraphRag(await empty.retrieve("고마워요, 하루 씨", { ignore: ["하루"] }), 10, "민수")).toBe("");
  });

  it("a session with graphRag shows the retrieved block as a context message and records the retrieval on the reply", async () => {
    let prompt: string[] = [];
    const provider: LLMProvider = {
      id: "fake",
      defaultModel: "fake-1",
      async generate(messages) {
        prompt = messages.map((m) => `${m.role}/${m.kind ?? "-"}: ${m.content}`);
        return { text: '"하늘 동물병원이요."', model: "fake-1", latencyMs: 1 };
      },
      async *stream() {
        yield '"하늘 동물병원이요."';
      },
    };
    const vectors = new MemoryVectorIndex();
    const graph = new MemoryGraph();
    const embedder = bigramEmbedder(256);
    const pre = new GraphRag({ ns: "s", embedder, vectors, graph });
    await pre.ingestFacts(facts);
    await pre.ingestTurns(transcript, 0, 11);
    const haru = cards.find((c) => c.id === "haru")!;
    const s = new ChatSession({ card: haru, lang: "ko", provider, promptVersion: "v13-structured", memory: { enabled: false }, graphRag: { ns: "s", embedder, vectors, graph } });
    s.memory = { ...emptyMemory(), coveredUpTo: 11 };
    const reply = await s.send('"보리 병원 이름 기억하세요?"');
    const ctx = prompt.find((p) => p.startsWith("system/context"));
    expect(ctx).toContain("KNOWLEDGE GRAPH");
    expect(ctx).toContain("하늘 동물병원");
    expect(prompt[0]).not.toContain("KNOWLEDGE GRAPH");
    expect(reply.meta?.retrieval).toMatchObject({ seeds: ["보리"] });
    expect(reply.meta?.retrieval!.edges).toBeGreaterThan(0);
    expect(reply.meta?.recalled?.some((x) => x.why === "graph")).toBe(true);
    expect(reply.meta?.recalled?.some((x) => x.why === "chunk")).toBe(true);
    s.close();
  });
});
