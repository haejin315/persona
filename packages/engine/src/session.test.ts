import { describe, expect, it } from "vitest";
import { loadAllCharacters } from "./characters.js";
import { DEFAULT_MEMORY_CONFIG, MemoryManager } from "./memory/manager.js";
import { DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS } from "./prompt/builder.js";
import type { LLMProvider } from "./providers/types.js";
import { ChatSession } from "./session.js";
import { emptyMemory, type Turn } from "./types.js";

/** Provider that echoes a canned reply; keeps tests offline. */
const fake = (reply = "ok"): LLMProvider => ({
  id: "fake",
  defaultModel: "fake-1",
  async generate() {
    return { text: reply, model: "fake-1", latencyMs: 1, usage: { promptTokens: 10, completionTokens: 2 } };
  },
  async *stream(_m, opts) {
    yield reply;
    opts?.onDone?.({ model: "fake-1", latencyMs: 1, usage: { promptTokens: 10, completionTokens: 2 } });
  },
});

const cards = loadAllCharacters();
const persona = { name: "민수", description: "28살 회사원" };

describe("character cards", () => {
  it("load and have ko + ja style blocks", () => {
    expect(cards.map((c) => c.id).sort()).toEqual(["cho", "haru", "imo", "jung", "kael", "yoon"]);
    for (const c of cards) {
      expect(c.style.ko).toBeDefined();
      expect(c.style.ja).toBeDefined();
      expect(c.scenario.firstMessage.ko).toBeTruthy();
      expect(c.scenario.firstMessage.ja).toBeTruthy();
    }
  });

  it("first messages and examples use the *action* \"speech\" convention", () => {
    for (const c of cards) {
      for (const lang of ["ko", "ja", "en"] as const) {
        const fm = c.scenario.firstMessage[lang]!;
        expect(fm).toMatch(/\*[^*]+\*/);
        expect(fm).toMatch(/["「][^"」]+["」]/);
        for (const ex of c.style[lang]!.examples) expect(ex.character).toMatch(/\*[^*]+\*/);
      }
    }
  });
});

describe("prompt versions", () => {
  it.each(Object.keys(PROMPT_VERSIONS))("%s builds a system prompt for every card/lang", (pv) => {
    for (const card of cards) {
      for (const lang of ["ko", "ja"] as const) {
        const s = new ChatSession({ card, lang, provider: fake(), promptVersion: pv, persona });
        s.open();
        const msgs = s.buildMessages();
        expect(msgs[0]!.role).toBe("system");
        expect(msgs[0]!.content).toContain(card.name[lang]!);
        expect(msgs[0]!.content).toContain("민수");
        expect(msgs[0]!.content).not.toContain("{{user}}");
        expect(msgs.filter((m) => m.kind !== "context")).toHaveLength(2); // system + first message (+ optional context)
        for (const m of msgs.filter((m) => m.kind === "context")) expect(m.content).not.toContain("{{user}}");
      }
    }
  });

  it("v16 speaks of a folded [days a-b] paragraph only when there is one, and is v13 otherwise", () => {
    const world = { day: 31, weekday: "Wednesday", slot: "morning", place: "카페 '모퉁이'", daysSinceUser: 30 } as const;
    const build = (pv: string, summary: string) => {
      const s = new ChatSession({ card: cards[0]!, lang: "ko", provider: fake(), promptVersion: pv, persona });
      s.memory = { ...emptyMemory(), summary };
      s.world = world;
      return s.buildMessages()[0]!.content;
    };
    const folded = "[days 2-23] day 2 (Tuesday) 오후 책다락에서 민수의 책 두 권을 받아 왔다.";
    expect(build("v16-structured", folded)).toContain("have faded");
    expect(build("v16-structured", "평범한 한 달이었다.")).not.toContain("have faded");
    expect(build("v16-structured", "평범한 한 달이었다.")).toBe(build("v13-structured", "평범한 한 달이었다."));
    expect(build("v13-structured", folded)).not.toContain("have faded");
  });

  it("v2 carries memory facts into the prompt", () => {
    const s = new ChatSession({ card: cards[0]!, lang: "ko", provider: fake(), promptVersion: "v2-structured" });
    s.memory = { ...emptyMemory(), facts: [{ text: "유저의 이름은 민수다", source: "user" }] };
    expect(s.buildMessages()[0]!.content).toContain("유저의 이름은 민수다");
  });

  it("v3-cached keeps the system prompt identical while memory changes (context message carries it)", () => {
    const s = new ChatSession({ card: cards[0]!, lang: "ko", provider: fake(), promptVersion: "v3-cached" });
    const before = s.buildMessages();
    s.memory = { ...emptyMemory(), facts: [{ text: "유저의 이름은 민수다", source: "user" }], relationship: { stage: "friendly", notes: [] } };
    const after = s.buildMessages();
    expect(after[0]!.content).toBe(before[0]!.content); // static system prompt unchanged
    expect(after[1]!.kind).toBe("context");
    expect(after[1]!.content).toContain("유저의 이름은 민수다");
    expect(after[1]!.content).toContain("friendly");
    expect(before[0]!.content).not.toContain("Facts to remember");
  });

  it("claude-cli folding keeps context messages out of the system prompt", async () => {
    const { foldMessages } = await import("./providers/claudeCli.js");
    const s = new ChatSession({ card: cards[0]!, lang: "ko", provider: fake(), promptVersion: "v3-cached", persona });
    s.open();
    s.memory = { ...emptyMemory(), facts: [{ text: "유저는 고양이 보리를 키운다", source: "user" }] };
    const { system, prompt } = foldMessages(s.buildMessages());
    expect(system).not.toContain("보리");
    expect(prompt).toContain("보리");
    expect(prompt.indexOf("[CURRENT STATE")).toBe(0);
  });

  it("fills {{user}} in the first message", () => {
    const s = new ChatSession({ card: cards.find((c) => c.id === "kael")!, lang: "ko", provider: fake(), persona });
    expect(s.open().content).toContain("민수의 짐");
  });
});

describe("session", () => {
  it("records turns with indices, meta and usage", async () => {
    const s = new ChatSession({ card: cards[0]!, lang: "ko", provider: fake("안녕") });
    s.open();
    const t = await s.send("hi");
    expect(t.index).toBe(2);
    expect(t.role).toBe("character");
    expect(t.content).toBe("안녕");
    expect(t.meta?.promptVersion).toBe(DEFAULT_PROMPT_VERSION);
    expect(t.meta?.usage?.promptTokens).toBe(10);
  });

  it("streams and records usage from onDone", async () => {
    const s = new ChatSession({ card: cards[0]!, lang: "ko", provider: fake("스트림") });
    s.open();
    const gen = s.sendStream("hi");
    let out = "";
    let r = await gen.next();
    while (!r.done) {
      out += r.value;
      r = await gen.next();
    }
    expect(out).toBe("스트림");
    expect(r.value.meta?.usage?.completionTokens).toBe(2);
  });

  it("continue adds a character turn without a user turn", async () => {
    const s = new ChatSession({ card: cards[0]!, lang: "ko", provider: fake() });
    s.open();
    await s.continue();
    expect(s.turns.map((t) => t.role)).toEqual(["character", "character"]);
  });

  it("regenerate replaces the last character turn", async () => {
    const s = new ChatSession({ card: cards[0]!, lang: "ko", provider: fake("둘") });
    s.open();
    await s.send("hi");
    await s.regenerate();
    expect(s.turns).toHaveLength(3);
    expect(s.turns.at(-1)!.content).toBe("둘");
  });

  it("undoLastReply removes only a trailing character turn", async () => {
    const s = new ChatSession({ card: cards[0]!, lang: "ko", provider: fake() });
    s.open();
    await s.send("hi");
    expect(s.undoLastReply()?.role).toBe("character");
    expect(s.undoLastReply()).toBeUndefined();
  });
});

describe("memory manager", () => {
  const turns = (n: number): Turn[] =>
    Array.from({ length: n }, (_, i) => ({
      role: i % 2 ? "character" : "user",
      content: `t${i}`,
      index: i,
      ts: 0,
    }));

  it("keeps only the window in the prompt", () => {
    const mm = new MemoryManager(fake(), { ...DEFAULT_MEMORY_CONFIG, windowTurns: 4 });
    expect(mm.recentWindow(turns(10)).map((t) => t.index)).toEqual([6, 7, 8, 9]);
  });

  it("asks for consolidation once enough turns fall outside the window", () => {
    const mm = new MemoryManager(fake(), { windowTurns: 4, consolidateEvery: 3 });
    expect(mm.needsConsolidation(turns(6), emptyMemory())).toBe(false); // 2 outside
    expect(mm.needsConsolidation(turns(7), emptyMemory())).toBe(true); // 3 outside
    expect(mm.needsConsolidation(turns(8), { ...emptyMemory(), coveredUpTo: 2 })).toBe(false);
  });

  it("merges the JSON the model returns", async () => {
    const reply = JSON.stringify({
      summary: "둘은 비 오는 날 처음 이름을 나눴다.",
      facts: ["유저 이름: 민수"],
      relationship: { stage: "acquaintance", notes: ["이름을 알게 됨"] },
    });
    const mm = new MemoryManager(fake(reply), { windowTurns: 2, consolidateEvery: 1 });
    const m = await mm.consolidate(cards[0]!, "ko", turns(6), emptyMemory());
    expect(m.facts).toEqual([{ text: "유저 이름: 민수", source: "unknown" }]);
    expect(m.relationship.stage).toBe("acquaintance");
    expect(m.coveredUpTo).toBe(3); // turns 0..3 folded, 4..5 remain in window
  });

  it("rules v3 keeps the source the model gives each fact and drops malformed ones", async () => {
    const reply = JSON.stringify({
      summary: "s",
      facts: [
        { text: "민수는 성수동으로 이사했다", source: "user" },
        { text: "하루는 여섯 시 반에 문을 연다", source: "self" },
        { text: "민수의 회사가 판교로 옮긴다", source: "told", by: "이모" },
        { text: "출처가 이상한 사실", source: "divine" },
        "문자열 사실",
        { source: "user" },
      ],
      relationship: { stage: "friendly", notes: [] },
    });
    const mm = new MemoryManager(fake(reply), { windowTurns: 2, consolidateEvery: 1, rules: "v3" });
    const m = await mm.consolidate(cards[0]!, "ko", turns(6), emptyMemory());
    expect(m.facts).toEqual([
      { text: "민수는 성수동으로 이사했다", source: "user" },
      { text: "하루는 여섯 시 반에 문을 연다", source: "self" },
      { text: "민수의 회사가 판교로 옮긴다", source: "told", by: "이모" },
      { text: "출처가 이상한 사실", source: "unknown" },
      { text: "문자열 사실", source: "unknown" },
    ]);
  });
});
