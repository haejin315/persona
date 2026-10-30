import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { ArchivedFact, Turn } from "../types.js";
import { bigramEmbedder } from "./recall.js";
import { MemoryStore, bareName } from "./store.js";

const facts: ArchivedFact[] = [
  { text: "민수는 고양이 한 마리를 키우고 이름은 보리다", source: "user", turnFrom: 10, turnTo: 20, entities: ["민수", "보리"] },
  { text: "민수는 다음 달 성수동으로 이사한다", source: "user", turnFrom: 20, turnTo: 30, entities: ["민수", "성수동"] },
  { text: "민수는 아침마다 아메리카노를 마신다", source: "user", turnFrom: 30, turnTo: 40, entities: ["민수", "아메리카노"] },
  { text: "하루는 아침 6시 반에 셔터를 올린다", source: "self", turnFrom: 2, turnTo: 10, entities: ["하루 씨"] },
  { text: "보리는 하늘 동물병원에 다닌다", source: "user", turnFrom: 400, turnTo: 410, entities: ["보리", "하늘 동물병원"] },
  { text: "민수는 위가 아파서 라떼로 바꿨다", source: "user", turnFrom: 1700, turnTo: 1710, entities: ["민수", "라떼"] },
];

describe("MemoryStore (docs/13)", () => {
  it("keeps facts with their entity edges, serves one proper noun as a time-ordered page, and marks what changed", async () => {
    const s = new MemoryStore();
    const emb = bigramEmbedder(256);
    const ids = s.appendFacts(facts, await emb.embed(facts.map((f) => f.text), "passage"));
    expect(ids).toHaveLength(6);
    expect(s.factCount()).toBe(6);
    // Entities are indexed bare: "하루 씨" becomes "하루".
    expect(bareName("하루 씨")).toBe("하루");
    expect(s.entities().map((e) => e.name)).toContain("하루");
    expect(s.entities().find((e) => e.name === "민수")).toMatchObject({ nFacts: 4, firstTurn: 10, lastTurn: 1710 });
    // The page for 보리: both facts, oldest first.
    const page = s.entityPage("보리");
    expect(page.map((f) => f.turnTo)).toEqual([20, 410]);
    expect(page[0]!.status).toBe("current");
    // 아메리카노 → 라떼: the old fact is superseded, still on the page unless excluded.
    s.supersede(ids[2]!, ids[5]!);
    expect(s.fact(ids[2]!)).toMatchObject({ status: "superseded", supersededBy: ids[5] });
    expect(s.entityPage("민수").filter((f) => f.status === "superseded")).toHaveLength(1);
    expect(s.entityPage("민수", { includeSuperseded: false })).toHaveLength(3);
    // The archive comes back in the session's shape.
    const back = s.loadArchive();
    expect(back).toHaveLength(6);
    expect(back[0]).toMatchObject({ text: facts[0]!.text, turnFrom: 10, turnTo: 20, entities: ["민수", "보리"] });
    expect((back[0] as unknown as Record<string, unknown>).status).toBeUndefined();
    s.close();
  });

  it("triggers on discriminative entities the message names and the ledger lacks", () => {
    const s = new MemoryStore();
    s.appendFacts(facts);
    // 민수 is tagged on 4 of 6 facts: not discriminative at a 50% share; 보리 (2 of 6) is.
    const names = (xs: { name: string }[]) => xs.map((x) => x.name);
    expect(names(s.discriminativeEntities({ maxShare: 0.5, minCap: 0 }))).not.toContain("민수");
    expect(names(s.discriminativeEntities({ maxShare: 0.5, minCap: 0 }))).toContain("보리");
    // A young store excludes nothing by share: the floor is 10 facts.
    expect(names(s.discriminativeEntities({ maxShare: 0.5 }))).toContain("민수");
    // The message names 보리 and 하루 씨; 하루 is excluded as a speaker, 보리 fires only when the ledger lacks it.
    expect(names(s.entitiesIn("보리는 잘 지내요, 하루 씨?", { maxShare: 0.5, minCap: 0, ignore: ["하루"] }))).toEqual(["보리"]);
    expect(s.entitiesIn("보리는 잘 지내요, 하루 씨?", { maxShare: 0.5, minCap: 0, ignore: ["하루"], ledger: "- 민수는 고양이 보리를 키운다" })).toEqual([]);
    expect(names(s.entitiesIn("오늘 날씨가 좋네요", { maxShare: 0.5, minCap: 0 }))).toEqual([]);
    s.close();
  });

  it("searches vectors over a narrowed candidate set, and over the whole store only when asked", async () => {
    const s = new MemoryStore();
    const emb = bigramEmbedder(256);
    s.appendFacts(facts, await emb.embed(facts.map((f) => f.text), "passage"));
    const [q] = await emb.embed(["보리 동물병원은 어디예요?"], "query");
    const all = s.vectorSearch(q!, { k: 2 });
    expect(all[0]!.fact.text).toContain("하늘 동물병원");
    const narrowed = s.vectorSearch(q!, { k: 5, entity: "민수" });
    expect(narrowed.every((h) => h.fact.entities?.includes("민수"))).toBe(true);
    expect(narrowed).toHaveLength(4);
    const ranged = s.vectorSearch(q!, { k: 5, turnFrom: 0, turnTo: 40 });
    expect(ranged.map((h) => h.fact.turnTo).every((t) => t <= 40)).toBe(true);
    // Mixed dimensions are refused.
    expect(() => s.appendFacts([facts[0]!], [[1, 0, 0]])).toThrow(/dims/);
    s.close();
  });

  it("persists the transcript, ledger snapshots and summaries in one file and reopens it", () => {
    const dir = mkdtempSync(join(tmpdir(), "persona-store-"));
    const path = join(dir, "haru-minsu.sqlite");
    const turns: Turn[] = [
      { index: 0, role: "user", content: "안녕하세요", ts: 1 },
      { index: 1, role: "character", content: "어서 오세요", ts: 2, meta: { latencyMs: 1200 } } as Turn,
    ];
    try {
      const s = new MemoryStore(path);
      s.appendTurns(turns);
      s.appendTurns(turns); // idempotent on index
      s.appendFacts(facts.slice(0, 2));
      s.saveLedger(30, { summary: "둘은 이름을 나눴다.", facts: [facts[0]!], relationship: { stage: "acquaintance", notes: [] }, coveredUpTo: 30, archive: facts });
      s.putSummary({ level: 1, turnFrom: 0, turnTo: 100, text: "첫 장" });
      s.putSummary({ level: 1, turnFrom: 100, turnTo: 200, text: "둘째 장" });
      expect(s.sizeBytes()).toBeGreaterThan(0);
      s.close();

      const r = new MemoryStore(path);
      expect(r.turnCount()).toBe(2);
      expect(r.turns(0, 1)[1]).toMatchObject({ index: 1, role: "character", meta: { latencyMs: 1200 } });
      expect(r.factCount()).toBe(2);
      const ledger = r.loadLedger();
      expect(ledger?.turnTo).toBe(30);
      expect(ledger?.memory.summary).toBe("둘은 이름을 나눴다.");
      expect((ledger?.memory as Record<string, unknown>).archive).toBeUndefined();
      expect(r.summaries(1).map((x) => x.text)).toEqual(["둘째 장", "첫 장"]);
      r.close();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
