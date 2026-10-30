import { describe, expect, it } from "vitest";
import { loadAllCharacters } from "../characters.js";
import type { LLMProvider } from "../providers/types.js";
import { ChatSession } from "../session.js";
import { emptyMemory, type ArchivedFact, type Turn } from "../types.js";
import { chunkAll, chunkRange } from "./chunker.js";
import { MemoryGraph } from "./graphdb.js";
import { GraphRag, factId, renderGraphRag } from "./graphrag.js";
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
  { text: "민수는 위가 아파서 라떼로 바꿨다", source: "user", turnFrom: 8, turnTo: 11, triple: { subject: "민수", relation: "마신다", object: "라떼", replaces: true } },
  { text: "하늘 동물병원은 성수동에 있다", source: "user", turnFrom: 8, turnTo: 11, triple: { subject: "하늘 동물병원", relation: "있다", object: "성수동" } },
  { text: "민수는 어릴 때 만화를 그렸다", source: "user", turnFrom: 4, turnTo: 7, triple: { subject: "민수", relation: "그렸다", object: "만화" } },
  { text: "민수가 그린 만화 제목은 달빛 고양이다", source: "user", turnFrom: 8, turnTo: 11, triple: { subject: "민수", relation: "그렸다", object: "달빛 고양이" } },
  { text: "하루는 보리 사진을 꼭 보여 달라고 했다", source: "self", turnFrom: 4, turnTo: 7, triple: { subject: "하루", relation: "부탁했다", object: "보리 사진" } },
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

  it("the graph keeps one edge per triple, supersedes only when the summariser said so, and walks hops", async () => {
    const g = new MemoryGraph();
    await g.addTriples(facts.map((f) => ({ ...f.triple!, text: f.text, turnFrom: f.turnFrom, turnTo: f.turnTo })));
    expect(await g.count()).toEqual({ nodes: 10, edges: 8 });
    const one = await g.neighborhood(["보리"], { hops: 1, limit: 10 });
    expect(one.map((e) => e.object).sort()).toEqual(["보리", "하늘 동물병원"]);
    const two = await g.neighborhood(["보리"], { hops: 2, limit: 20 });
    expect(two.some((e) => e.object === "성수동" && e.depth === 2)).toBe(true);
    expect(two.some((e) => e.object === "라떼" && e.depth === 2)).toBe(true);
    // 마신다 라떼 carried replaces: true, so 아메리카노 is no longer current.
    const drinks = two.filter((e) => e.relation === "마신다");
    expect(drinks.find((e) => e.object === "아메리카노")!.current).toBe(false);
    expect(drinks.find((e) => e.object === "라떼")!.current).toBe(true);
    // 그렸다 달빛 고양이 did not: both drawings stay true (round 1 marked 897 such lines "no longer").
    const drew = two.filter((e) => e.relation === "그렸다");
    expect(drew).toHaveLength(2);
    expect(drew.every((e) => e.current)).toBe(true);
    // With 민수 declared a hub, a path crosses it only along one relation: from 만화 the walk reaches
    // 달빛 고양이 (그렸다 both ways) but not 보리 or 라떼 (키운다, 마신다).
    const viaHub = await g.neighborhood(["만화"], { hops: 2, limit: 20, hubs: ["민수"] });
    expect(viaHub.map((e) => e.object).sort()).toEqual(["달빛 고양이", "만화"]);
    const free = await g.neighborhood(["만화"], { hops: 2, limit: 20 });
    expect(free.some((e) => e.object === "라떼")).toBe(true);
  });

  it("retrieves graph edges, nearest facts and verbatim chunks for a message, and renders one block", async () => {
    const rag = new GraphRag({ ns: "t", embedder: bigramEmbedder(256), vectors: new MemoryVectorIndex(), graph: new MemoryGraph(), chunkK: 2, factK: 3, hops: 2 });
    await rag.ingestFacts(facts);
    expect(await rag.ingestTurns(transcript, 0, 11)).toBe(5);
    expect(await rag.counts()).toEqual({ chunks: 5, facts: 8, nodes: 10, edges: 8 });
    const r = await rag.retrieve("보리 병원은 어디였죠?", { turnToMax: 7, ignore: ["하루", "민수"] });
    // Whole name and token: "보리" also wakes the node "보리 사진".
    expect(r.seeds).toEqual(["보리", "보리 사진"]);
    expect(r.edges.some((e) => e.object === "하늘 동물병원")).toBe(true);
    expect(r.chunks.every((c) => c.turnTo <= 7)).toBe(true);
    expect(r.chunks[0]!.text).toContain("하늘 동물병원");
    // Facts come in two pools: about the user (factK), then the character's own lines (selfK).
    expect(r.facts.filter((f) => !f.self)).toHaveLength(3);
    expect(r.facts.filter((f) => f.self).map((f) => f.text)).toEqual(["하루는 보리 사진을 꼭 보여 달라고 했다"]);
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

  it("links a node by a token of its name and ranks the neighbourhood by similarity to the message, not recency", async () => {
    // 민수 is a hub with many recent edges; the one the message is about is the oldest.
    const g = new MemoryGraph();
    const seats: ArchivedFact[] = Array.from({ length: 12 }, (_, i) => ({
      text: `민수는 ${i}번째 창가 자리에 앉았다`,
      source: "user",
      turnFrom: 100 + i,
      turnTo: 100 + i,
      triple: { subject: "민수", relation: "앉았다", object: `창가 자리 ${i}` },
    }));
    const visits: ArchivedFact[] = Array.from({ length: 8 }, (_, i) => ({
      text: `민수는 ${i}번째 서점에 들른다`,
      source: "user",
      turnFrom: 200 + i,
      turnTo: 200 + i,
      triple: { subject: "민수", relation: "들른다", object: `서점 ${i}` },
    }));
    const home: ArchivedFact = { text: "민수는 이사 전 대전 본가에 들른다", source: "user", turnFrom: 10, turnTo: 10, triple: { subject: "민수", relation: "들른다", object: "대전 본가" } };
    // The same triple restated later: one line in the result, not two.
    const again: ArchivedFact = { text: "민수는 대전 본가에 들른다고 다시 말했다", source: "user", turnFrom: 300, turnTo: 300, triple: { subject: "민수", relation: "들른다", object: "대전 본가" } };
    const all = [home, again, ...seats, ...visits];
    const rag = new GraphRag({ ns: "r", embedder: bigramEmbedder(256), vectors: new MemoryVectorIndex(), graph: g, edgeLimit: 3, edgeCandidates: 60, factK: 2, selfK: 0 });
    await rag.ingestFacts(all);
    // "본가" is a token of the node "대전 본가": the node seeds; 민수 (hub, on all 21 edges) does not.
    const r = await rag.retrieve("제 본가가 어디라고 했는지 기억하세요?", { ignore: ["하루", "민수"] });
    expect(r.seeds).toEqual(["대전 본가"]);
    // Depth 1 is the 본가 edge; depth 2 crosses the hub 민수 only along 들른다, so the 8 bookshop
    // visits are candidates and the 12 seats are not. Ranking by similarity to the message puts the
    // edge the message is about first, although it is the oldest, and records its score.
    expect(r.edges).toHaveLength(3);
    expect(r.edges[0]!.object).toBe("대전 본가");
    expect(r.edges[0]!.score).toBeGreaterThan(r.edges[1]!.score ?? 0);
    expect(r.edges.every((e) => e.relation === "들른다")).toBe(true);
    expect(r.edges.filter((e) => e.object === "대전 본가")).toHaveLength(1);
    expect(factId(home)).toBe("f10-10:민수는 이사 전 대전 본가에 들른다");
    // Without ranking (edgeCandidates 0) the graph's own order wins: depth 1 first, then newest.
    const plain = new GraphRag({ ns: "p", embedder: bigramEmbedder(256), vectors: new MemoryVectorIndex(), graph: g, edgeLimit: 3, edgeCandidates: 0, tokenLinking: false });
    const p = await plain.retrieve("대전 본가 기억하세요?", { ignore: ["하루", "민수"] });
    expect(p.edges[0]!.object).toBe("대전 본가");
    expect(p.edges[1]!.object).toBe("서점 7");
    expect(p.edges[0]!.score).toBeUndefined();
    // Whole-name linking only: "본가" alone wakes nothing.
    expect((await plain.retrieve("제 본가가 어디였죠", { ignore: ["하루", "민수"] })).seeds).toEqual([]);
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
    expect(reply.meta?.retrieval?.seeds).toContain("보리");
    expect(reply.meta?.retrieval!.edges).toBeGreaterThan(0);
    const graphLines = reply.meta?.recalled?.filter((x) => x.why === "graph") ?? [];
    expect(graphLines.length).toBeGreaterThan(0);
    // Ranked: the recorded score is a cosine, not a hop depth.
    expect(graphLines.every((x) => x.score <= 1)).toBe(true);
    expect(reply.meta?.recalled?.some((x) => x.why === "chunk")).toBe(true);
    s.close();
  });
});
