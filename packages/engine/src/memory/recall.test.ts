import { describe, expect, it } from "vitest";
import { loadAllCharacters } from "../characters.js";
import type { LLMProvider } from "../providers/types.js";
import { ChatSession } from "../session.js";
import { emptyMemory, type ArchivedFact } from "../types.js";
import { MemoryManager } from "./manager.js";
import { Recall, bigramEmbedder, nearDuplicate } from "./recall.js";

const cards = loadAllCharacters();

const archive: ArchivedFact[] = [
  { text: "민수는 고양이 한 마리를 키우고 이름은 보리다", source: "user", turnFrom: 10, turnTo: 20, entities: ["민수", "보리"] },
  { text: "민수는 다음 달 성수동으로 이사한다", source: "user", turnFrom: 20, turnTo: 30, entities: ["민수", "성수동"] },
  { text: "민수의 생일은 11월 3일이다", source: "user", turnFrom: 100, turnTo: 110, entities: ["민수"] },
  { text: "하루는 아침 6시 반에 셔터를 올린다", source: "self", turnFrom: 2, turnTo: 10, entities: ["하루"] },
];

describe("recall (docs/12)", () => {
  it("bigram embedder: unit vectors, and the paraphrase scores above the stranger", async () => {
    const e = bigramEmbedder();
    const [a, b, c] = await e.embed(["성수동으로 이사 간다", "성수동 이사", "셔터를 올린다"], "passage");
    const norm = Math.sqrt(a!.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeCloseTo(1, 5);
    const dot = (x: number[], y: number[]) => x.reduce((s, v, i) => s + v * y[i]!, 0);
    expect(dot(a!, b!)).toBeGreaterThan(dot(a!, c!));
  });

  it("an entity in the message pulls its facts first; what the ledger already holds is left out; k caps", async () => {
    const r = new Recall(bigramEmbedder(), { k: 2 });
    const hits = await r.query("얘 이름 기억하세요? 보리요.", archive, [{ text: "민수의 생일은 11월 3일이다", source: "user" }]);
    expect(hits.length).toBe(2);
    expect(hits[0]!.why).toBe("entity");
    expect(hits[0]!.fact.text).toContain("보리");
    expect(hits.some((h) => h.fact.text.includes("11월 3일"))).toBe(false);
  });

  it("the speakers' names are not entities, near-duplicates collapse, and minZ hides what merely resembles the message (docs/12 result 2)", async () => {
    const dup: ArchivedFact[] = [
      ...archive,
      { text: "민수는 고양이 한 마리를 키우고 그 이름은 보리다", source: "user", turnFrom: 30, turnTo: 40, entities: ["민수", "보리"] },
    ];
    // "하루 씨" names the character: without the exclusion every 하루-tagged entry is an entity hit.
    const plain = new Recall(bigramEmbedder(), { k: 5 });
    expect((await plain.query("고마워요, 하루 씨.", dup)).filter((h) => h.why === "entity").length).toBe(1);
    const r = new Recall(bigramEmbedder(), { k: 5, ignoreEntities: ["하루", "민수"] });
    expect((await r.query("고마워요, 하루 씨.", dup)).filter((h) => h.why === "entity").length).toBe(0);
    // 보리 is still an entity, and the two restatements of the cat fact count once.
    expect(nearDuplicate(dup[0]!.text, dup[4]!.text)).toBe(true);
    expect(nearDuplicate(dup[0]!.text, dup[1]!.text)).toBe(false);
    const hits = await r.query("보리는 잘 지내요?", dup);
    expect(hits.filter((h) => h.fact.text.includes("고양이")).length).toBe(1);
    expect(hits[0]!.why).toBe("entity");
    // With minZ a message that resembles nothing shows fewer than the k that fit without it.
    const gated = new Recall(bigramEmbedder(), { k: 5, ignoreEntities: ["하루", "민수"], minZ: 1.5 });
    expect((await r.query("오늘 날씨가 참 좋네요.", dup)).length).toBe(4); // five entries, the two cat restatements count once
    expect((await gated.query("오늘 날씨가 참 좋네요.", dup)).length).toBeLessThanOrEqual(2);
    // An entity hit is shown whatever the floor.
    expect((await gated.query("성수동 이사는 잘 되셨어요?", dup)).some((h) => h.fact.text.includes("성수동") && h.why === "entity")).toBe(true);
  });

  it("memory rules v4 keep every new fact in the archive with its turn range and entities, and hide the archive from the summariser", async () => {
    const seen: string[] = [];
    const reply = JSON.stringify({
      summary: "둘은 이름을 나눴다.",
      facts: [{ text: "민수의 이름은 민수다", source: "user" }],
      relationship: { stage: "acquaintance", notes: [] },
      new_facts: [
        { text: "민수의 이름은 민수다", source: "user", entities: ["민수"] },
        { text: "민수는 고양이 보리를 키운다", source: "user", entities: ["민수", "보리"] },
        { text: "민수는 고양이 보리를 키운다", source: "user" }, // duplicate: kept once
      ],
    });
    const provider: LLMProvider = {
      id: "fake",
      defaultModel: "fake-1",
      async generate(messages) {
        seen.push(messages.map((m) => m.content).join("\n"));
        return { text: reply, model: "fake-1", latencyMs: 1 };
      },
      async *stream() {
        yield reply;
      },
    };
    const m = new MemoryManager(provider, { windowTurns: 2, consolidateEvery: 1, rules: "v4" });
    const turns = ["안녕하세요", "저는 민수예요", "고양이 보리를 키워요", "네"].map((content, index) => ({
      role: (index % 2 ? "user" : "character") as "user" | "character",
      content,
      index,
      ts: 0,
    }));
    const existing = { ...emptyMemory(), archive: [{ text: "옛 사실", source: "user" as const, turnFrom: 0, turnTo: 0 }] };
    const out = await m.consolidate(cards[0]!, "ko", turns, existing);
    expect(out.facts.length).toBe(1);
    expect(out.archive?.map((f) => f.text)).toEqual(["옛 사실", "민수의 이름은 민수다", "민수는 고양이 보리를 키운다"]);
    expect(out.archive?.[2]).toMatchObject({ turnFrom: 0, turnTo: 1, entities: ["민수", "보리"] });
    expect(seen[0]).toContain('"new_facts"');
    expect(seen[0]).not.toContain("옛 사실"); // the archive is uncapped and never fed back
  });

  it("a session with recall shows the fitting archive entries in a per-turn context message and records them on the reply", async () => {
    let prompt: string[] = [];
    const provider: LLMProvider = {
      id: "fake",
      defaultModel: "fake-1",
      async generate(messages) {
        prompt = messages.map((m) => `${m.role}/${m.kind ?? "-"}: ${m.content}`);
        return { text: '"보리요."', model: "fake-1", latencyMs: 1 };
      },
      async *stream() {
        yield '"보리요."';
      },
    };
    const haru = cards.find((c) => c.id === "haru")!;
    const s = new ChatSession({ card: haru, lang: "ko", provider, promptVersion: "v13-structured", memory: { enabled: false }, recall: { embedder: bigramEmbedder(), k: 2 } });
    s.memory = { ...emptyMemory(), archive };
    const turn = await s.send("*사진을 내민다* \"보리 기억하세요?\"");
    const ctx = prompt.find((p) => p.startsWith("system/context"));
    expect(ctx).toContain("older entries");
    expect(ctx).toContain("보리");
    expect(ctx).toMatch(/\[about \d+ exchanges ago\] /); // each entry says how old it is
    expect(prompt[0]).not.toContain("older entries"); // the static system prompt is untouched
    expect(turn.meta?.recalled?.[0]).toMatchObject({ why: "entity", turnTo: 20 });
    expect(turn.meta?.recalled?.[0]?.text).toContain("보리");
  });
});
