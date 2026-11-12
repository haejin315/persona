import { describe, expect, it } from "vitest";
import { loadAllCharacters } from "../characters.js";
import type { LLMProvider } from "../providers/types.js";
import { ChatSession } from "../session.js";
import { emptyMemory, type ArchivedFact, type Turn } from "../types.js";
import { chunkAll, chunkRange } from "./chunker.js";
import { MemoryGraph } from "./graphdb.js";
import { GraphRag, factId, renderGraphRag, speechOf, speechSentences } from "./graphrag.js";
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

  it("keeps the top edge's siblings next to it, and the v2 block separates the user's facts from the character's own words", async () => {
    // The message is about drawing comics; the title is a sibling edge (same subject and relation)
    // that cosine alone ranks below unrelated-but-wordier neighbours.
    const g = new MemoryGraph();
    const comics: ArchivedFact = { text: "민수는 어릴 때 만화를 그렸다", source: "user", turnFrom: 10, turnTo: 10, triple: { subject: "민수", relation: "그렸다", object: "만화" } };
    const title: ArchivedFact = { text: "제목은 달빛 고양이였다", source: "user", turnFrom: 12, turnTo: 12, triple: { subject: "민수", relation: "그렸다", object: "달빛 고양이" } };
    const noise: ArchivedFact[] = Array.from({ length: 6 }, (_, i) => ({
      text: `민수는 어릴 때 만화 ${i}권을 헌책방에서 읽었다`,
      source: "user",
      turnFrom: 20 + i,
      turnTo: 20 + i,
      triple: { subject: "민수", relation: "읽었다", object: `만화 ${i}권` },
    }));
    const promise: ArchivedFact = { text: "하루는 토요일에 만화 얘기를 듣기로 했다", source: "self", turnFrom: 14, turnTo: 14, triple: { subject: "하루", relation: "듣기로 했다", object: "만화 얘기" } };
    const vectors = new MemoryVectorIndex();
    const embedder = bigramEmbedder(256);
    const on = new GraphRag({ ns: "x", embedder, vectors, graph: g, edgeLimit: 3, expandTop: 2, factK: 2, selfK: 1 });
    await on.ingestFacts([comics, title, ...noise, promise]);
    const q = "제가 어릴 때 만화를 그린 적이 있어요";
    const r = await on.retrieve(q, { ignore: ["하루", "민수"] });
    expect(r.edges[0]!.object).toBe("만화");
    expect(r.edges.slice(0, 3).some((e) => e.object === "달빛 고양이")).toBe(true);
    const off = new GraphRag({ ns: "x", embedder, vectors, graph: g, edgeLimit: 3, expandTop: 0, factK: 2, selfK: 1 });
    const r0 = await off.retrieve(q, { ignore: ["하루", "민수"] });
    expect(r0.edges[0]!.object).toBe("만화");
    // v2 block: the character's own edge and self fact sit under their own heading with the rule on past plans.
    const hand = {
      seeds: ["만화"],
      edges: [
        { subject: "민수", relation: "그렸다", object: "만화", text: comics.text, turnFrom: 10, turnTo: 10, current: true, depth: 1, score: 0.9 },
        { subject: "하루", relation: "듣기로 했다", object: "만화 얘기", text: promise.text, turnFrom: 14, turnTo: 14, current: true, depth: 2, score: 0.8 },
      ],
      chunks: [],
      facts: [
        { text: title.text, turnFrom: 12, turnTo: 12, score: 0.8, self: false },
        { text: promise.text, turnFrom: 14, turnTo: 14, score: 0.7, self: true },
      ],
    };
    const block = renderGraphRag(hand, 100, "민수", { character: "하루", style: "v2" });
    expect(block).toContain("# WHAT YOU REMEMBER ABOUT THIS");
    expect(block).toContain("## ABOUT 민수");
    expect(block).toContain("## WHAT YOU SAID OR PLANNED THEN");
    expect(block.indexOf("하루 —듣기로 했다→ 만화 얘기")).toBeGreaterThan(block.indexOf("## WHAT YOU SAID OR PLANNED THEN"));
    expect(block.indexOf("민수 —그렸다→ 만화")).toBeLessThan(block.indexOf("## WHAT YOU SAID OR PLANNED THEN"));
    expect(renderGraphRag(hand, 100, "민수", { character: "하루", style: "v1" })).toContain("KNOWLEDGE GRAPH");
  });

  it("a tail-placed block is folded after the transcript, a head-placed one before it", async () => {
    const { foldMessages } = await import("../providers/claudeCli.js");
    const msgs = (kind: "context" | "context-tail") => [
      { role: "system" as const, content: "SYS" },
      { role: "system" as const, kind, content: "BLOCK" },
      { role: "user" as const, content: "u1" },
      { role: "assistant" as const, content: "a1" },
      { role: "user" as const, content: "u2" },
    ];
    const head = foldMessages(msgs("context"));
    expect(head.system).toBe("SYS");
    expect(head.prompt.indexOf("BLOCK")).toBeLessThan(head.prompt.indexOf("[USER]\nu1"));
    const tail = foldMessages(msgs("context-tail"));
    expect(tail.system).toBe("SYS");
    expect(tail.prompt.indexOf("BLOCK")).toBeGreaterThan(tail.prompt.indexOf("[USER]\nu2"));
    expect(tail.prompt).toContain("BEARS ON THE LAST MESSAGE");
  });

  it("round 8: a named replacement retires only the edge it names; true retires every edge under the same subject and relation", async () => {
    const g = new MemoryGraph();
    const base = { text: "", turnFrom: 1, turnTo: 1 };
    await g.addTriples([
      { ...base, subject: "민수", relation: "산다", object: "성수동" },
      { ...base, subject: "민수", relation: "산다", object: "펜탁스 ME" },
      { ...base, subject: "민수", relation: "간다", object: "태호 결혼식" },
      { ...base, subject: "민수", relation: "약속했다", object: "태호 결혼식" },
    ]);
    await g.addTriples([{ ...base, turnFrom: 5, turnTo: 5, subject: "민수", relation: "산다", object: "망원동", replaces: "성수동" }]);
    await g.addTriples([{ ...base, turnFrom: 6, turnTo: 6, subject: "민수", relation: "다녀왔다", object: "태호 결혼식", replaces: "간다" }]);
    const current = Object.fromEntries((await g.neighborhood(["민수"], { hops: 1, limit: 20 })).map((e) => [`${e.relation} ${e.object}`, e.current]));
    expect(current).toEqual({ "산다 성수동": false, "산다 펜탁스 ME": true, "산다 망원동": true, "간다 태호 결혼식": false, "약속했다 태호 결혼식": true, "다녀왔다 태호 결혼식": true });
    // A name that matches nothing retires nothing (the old fact may never have become a triple).
    await g.addTriples([{ ...base, turnFrom: 7, turnTo: 7, subject: "민수", relation: "산다", object: "상암동", replaces: "연희동" }]);
    expect((await g.neighborhood(["민수"], { hops: 1, limit: 20 })).find((e) => e.object === "망원동")!.current).toBe(true);
    // true is the old rule: everything under 산다 goes.
    await g.addTriples([{ ...base, turnFrom: 8, turnTo: 8, subject: "민수", relation: "산다", object: "연남동", replaces: true }]);
    const after = (await g.neighborhood(["민수"], { hops: 1, limit: 20 })).filter((e) => e.relation === "산다");
    expect(after.filter((e) => e.current).map((e) => e.object)).toEqual(["연남동"]);
  });

  it("round 8: seeds come in tiers (whole name in the message first, a token in the carried context last) and maxSeeds / maxTokenNames cap them", async () => {
    const g = new MemoryGraph();
    // Six "… 소리" nodes among 400+ names: with the share rule alone (2% of the names, floor 3) the
    // token 소리 links them all once the graph is large; the absolute cap stops that.
    const sounds: ArchivedFact[] = Array.from({ length: 6 }, (_, i) => ({ text: `하루는 ${i}번 소리를 들었다`, source: "self", turnFrom: 10 + i, turnTo: 10 + i, triple: { subject: "하루", relation: "들었다", object: `${["그라인더", "문", "바람", "그릇", "의자", "알람"][i]} 소리` } }));
    const others: ArchivedFact[] = Array.from({ length: 400 }, (_, i) => ({ text: `민수는 물건 ${i}을 샀다`, source: "user", turnFrom: 100 + i, turnTo: 100 + i, triple: { subject: "민수", relation: "샀다", object: `물건 ${i}` } }));
    const job: ArchivedFact = { text: "민수는 루멘 디자인팀 팀장이 됐다", source: "user", turnFrom: 400, turnTo: 400, triple: { subject: "민수", relation: "맡았다", object: "루멘 디자인팀 팀장" } };
    const cat: ArchivedFact = { text: "민수는 보리를 키운다", source: "user", turnFrom: 1, turnTo: 1, triple: { subject: "민수", relation: "키운다", object: "보리" } };
    const all = [cat, job, ...sounds, ...others];
    const message = "루멘에서 제 직책이 뭐였는지 기억하세요?";
    const context = ["*그라인더 소리가 난다.* \"보리는 잘 있어요?\""];
    const open = new GraphRag({ ns: "s", embedder: bigramEmbedder(256), vectors: new MemoryVectorIndex(), graph: g, edgeLimit: 3, factK: 0, selfK: 0 });
    await open.ingestFacts(all);
    const r = await open.retrieve(message, { ignore: ["하루", "민수"], context });
    // Tier order: 루멘 디자인팀 팀장 (token 루멘 in the message), 보리 (whole name in the context), then the 소리 nodes (token in the context).
    expect(r.seeds[0]).toBe("루멘 디자인팀 팀장");
    expect(r.seeds[1]).toBe("보리");
    expect(r.seeds.filter((x) => x.endsWith("소리"))).toHaveLength(6);
    const capped = new GraphRag({ ns: "s", embedder: bigramEmbedder(256), vectors: new MemoryVectorIndex(), graph: g, edgeLimit: 3, factK: 0, selfK: 0, maxSeeds: 2 });
    expect((await capped.retrieve(message, { ignore: ["하루", "민수"], context })).seeds).toEqual(["루멘 디자인팀 팀장", "보리"]);
    const strict = new GraphRag({ ns: "s", embedder: bigramEmbedder(256), vectors: new MemoryVectorIndex(), graph: g, edgeLimit: 3, factK: 0, selfK: 0, maxTokenNames: 3 });
    // 소리 is in 6 names > 3: no token link, only the one sound named whole in the context; 루멘 is in one name: still links.
    expect((await strict.retrieve(message, { ignore: ["하루", "민수"], context })).seeds).toEqual(["루멘 디자인팀 팀장", "보리", "그라인더 소리"]);
  });

  it("round 5: spoken sentences are the query, the character's last reply seeds too, and a plan becoming the event supersedes the plan", async () => {
    // Stage directions are not spoken; quotes are stripped; sentences split.
    expect(speechSentences('*문손잡이를 잡은 채 잠깐 멈춘다.*  "아, 하나만요. 제가 어릴 때 공책에 만화를 그린 적이 있어요."  "내일 또 올게요."')).toEqual([
      "아, 하나만요.",
      "제가 어릴 때 공책에 만화를 그린 적이 있어요.",
      "내일 또 올게요.",
    ]);
    expect(speechOf("*컵을 내려놓는다.* \"네.\"")).toBe('"네."');
    const vectors = new MemoryVectorIndex();
    const graph = new MemoryGraph();
    const embedder = bigramEmbedder(256);
    const message = '*문손잡이를 잡은 채 잠깐 멈춘다.*  "아, 하나만요. 제가 어릴 때 공책에 만화를 그린 적이 있어요."  "갑자기 생각나서요. 내일 또 올게요."';
    const bySentence = new GraphRag({ ns: "q", embedder, vectors, graph, query: "sentences", factK: 1, selfK: 0, edgeLimit: 0 });
    await bySentence.ingestFacts(facts);
    const r = await bySentence.retrieve(message, { ignore: ["하루", "민수"] });
    // The 만화 fact is nearest to its own sentence; its score is that sentence's, not the whole message's.
    expect(r.facts[0]!.text).toBe("민수는 어릴 때 만화를 그렸다");
    const [sv] = await embedder.embed(["제가 어릴 때 공책에 만화를 그린 적이 있어요."], "query");
    const direct = await vectors.search("q-facts", sv!, { limit: 1, filter: { self: false } });
    expect(r.facts[0]!.score).toBeCloseTo(direct[0]!.score, 6);
    const whole = new GraphRag({ ns: "q", embedder, vectors, graph, query: "message", factK: 1, selfK: 0, edgeLimit: 0 });
    const [mv] = await embedder.embed([message], "query");
    const wholeDirect = await vectors.search("q-facts", mv!, { limit: 1, filter: { self: false } });
    expect((await whole.retrieve(message, { ignore: ["하루", "민수"] })).facts[0]!.score).toBeCloseTo(wholeDirect[0]!.score, 6);
    // Context seeds the graph (the character's own last reply named the hospital), the message alone does not.
    const seeded = new GraphRag({ ns: "q", embedder, vectors, graph, selfK: 0 });
    expect((await seeded.retrieve('"네, 갈게요."', { ignore: ["하루", "민수"] })).seeds).toEqual([]);
    const withContext = await seeded.retrieve('"네, 갈게요."', { ignore: ["하루", "민수"], context: ['"하늘 동물병원 이름은 제가 들은 기억이 없어요."'] });
    expect(withContext.seeds).toContain("하늘 동물병원");
    expect(withContext.edges.some((e) => e.object === "하늘 동물병원")).toBe(true);
    // (subject, object) supersede: the report that the wedding was attended replaces the plan to go.
    const g2 = new MemoryGraph();
    await g2.addTriples([
      { subject: "민수", relation: "간다", object: "태호 결혼식", text: "민수는 태호 결혼식에 간다", turnFrom: 1, turnTo: 1 },
      { subject: "민수", relation: "좋아한다", object: "태호", text: "민수는 태호를 좋아한다", turnFrom: 1, turnTo: 1 },
      { subject: "민수", relation: "다녀왔다", object: "태호 결혼식", text: "민수는 태호 결혼식에 다녀왔다", turnFrom: 9, turnTo: 9, replaces: true },
    ]);
    const edges = await g2.neighborhood(["태호 결혼식", "태호"], { hops: 1, limit: 10 });
    expect(edges.find((e) => e.relation === "간다")!.current).toBe(false);
    expect(edges.find((e) => e.relation === "다녀왔다")!.current).toBe(true);
    expect(edges.find((e) => e.relation === "좋아한다")!.current).toBe(true);
  });

  it("round 6: interleave fusion gives every spoken sentence its first edge, each with its siblings, before any sentence's second", async () => {
    const vectors = new MemoryVectorIndex();
    const graph = new MemoryGraph();
    const embedder = bigramEmbedder(256);
    const rows: ArchivedFact[] = [
      { text: "민수는 어릴 때 만화를 그렸다", source: "user", turnFrom: 1, turnTo: 1, triple: { subject: "민수", relation: "그렸다", object: "만화" } },
      { text: "민수는 어릴 때 '달빛 고양이'라는 만화를 그렸다", source: "user", turnFrom: 2, turnTo: 2, triple: { subject: "민수", relation: "그렸다", object: "달빛 고양이" } },
      { text: "민수는 어릴 때 만화풍 그림도 그렸다", source: "user", turnFrom: 3, turnTo: 3, triple: { subject: "민수", relation: "그렸다", object: "만화풍 그림" } },
      { text: "보리는 하늘 동물병원에 다닌다", source: "user", turnFrom: 4, turnTo: 4, triple: { subject: "보리", relation: "다닌다", object: "하늘 동물병원" } },
      { text: "보리는 하늘 동물병원 처방 사료를 먹는다", source: "user", turnFrom: 5, turnTo: 5, triple: { subject: "보리", relation: "먹는다", object: "처방 사료" } },
      { text: "하늘 동물병원은 큰길 쪽에 있다", source: "user", turnFrom: 6, turnTo: 6, triple: { subject: "하늘 동물병원", relation: "있다", object: "큰길" } },
    ];
    const message = '"보리가 다니는 하늘 동물병원이요." "제가 어릴 때 만화를 그린 적이 있어요."';
    const mk = (fusion: "max" | "interleave") => new GraphRag({ ns: "f6", embedder, vectors, graph, query: "sentences", fusion, factK: 2, selfK: 0, edgeLimit: 12, edgeCandidates: 300, expandTop: 2 });
    await mk("max").ingestFacts(rows);
    const sentences = speechSentences(message);
    expect(sentences).toHaveLength(2);
    // Each sentence's nearest fact, measured directly.
    const heads = await Promise.all(sentences.map(async (s) => { const [qv] = await embedder.embed([s], "query"); const [h] = await vectors.search("f6-facts", qv!, { limit: 1 }); return String(h!.payload.text); }));
    expect(heads[0]).not.toBe(heads[1]);
    const r = await mk("interleave").retrieve(message, { ignore: ["하루", "민수"] });
    const texts = r.edges.map((e) => e.text);
    // The first edge is one sentence's head; its siblings (same subject and relation, at most 2) follow; then the other head.
    expect(heads).toContain(texts[0]);
    const first = r.edges[0]!;
    let k = 1;
    while (k < r.edges.length && r.edges[k]!.subject === first.subject && r.edges[k]!.relation === first.relation) k++;
    expect(k - 1).toBeLessThanOrEqual(2);
    expect(texts[k]).toBe(heads.find((h) => h !== texts[0]));
    // Facts too: with factK 2, one from each sentence rather than two from the nearer one.
    expect(r.facts.map((f) => f.text).sort()).toEqual([...heads].sort());
    // Max fusion keeps its single lead: the first edge is the best-scoring fact over both sentences.
    const m = await mk("max").retrieve(message, { ignore: ["하루", "민수"] });
    expect(m.edges.every((e, i) => i === 0 || (e.score ?? 0) <= (m.edges[0]!.score ?? 0))).toBe(true);
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
