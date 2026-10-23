import { describe, expect, it } from "vitest";
import { findCharacter } from "../characters.js";
import type { LLMProvider } from "../providers/types.js";
import { ChatSession } from "../session.js";
import { advanceDay, extractTasks } from "./llm.js";
import { syncCharacter } from "./sync.js";
import { World, type WorldFile } from "./world.js";

const def: WorldFile = {
  id: "test-town",
  places: [
    { id: "cafe", name: "카페 '모퉁이'" },
    { id: "bookshop", name: "헌책방 '책다락'", hours: "11시~19시" },
    { id: "home", name: "집" },
  ],
  characters: [
    { id: "haru", name: "하루", tier: "main", routine: { morning: "cafe", afternoon: "cafe", evening: "home", night: "home" } },
    { id: "yoon", name: "윤", tier: "main", routine: { morning: "bookshop", afternoon: "bookshop", evening: "bookshop", night: "home" } },
  ],
};

/** Provider that answers every call with the JSON the world layer asks for. */
const fakeWorldLLM = (): LLMProvider & { calls: string[] } => {
  const p = {
    id: "fake",
    defaultModel: "fake-1",
    calls: [] as string[],
    async generate(messages: { role: string; content: string }[]) {
      const sys = messages[0]!.content;
      p.calls.push(sys.slice(0, 40));
      let text: string;
      if (sys.includes("AGREED to do later")) {
        text = JSON.stringify({ tasks: [{ what: "책다락에 가서 민수가 맡겨 둔 책이 들어왔는지 확인하고 받아 온다", where: "bookshop", dueDay: 1, dueSlot: "afternoon" }] });
      } else if (sys.includes("two characters met")) {
        text = JSON.stringify({
          scene: '*문이 열린다.* "안녕하세요."',
          result: "하루가 윤에게서 민수의 책을 받아 왔다.",
          details: ["윤이 책을 신문지에 싸 주었다", "둘째 권 표지가 조금 바랬다"],
          told: { haru: ["윤은 다음 주에 가게를 사흘 쉰다고 했다"], yoon: ["하루는 민수가 어릴 때 만화를 그렸다고 전했다"] },
        });
      } else if (sys.includes("did an errand")) {
        text = JSON.stringify({ outcome: "done", result: "하루가 책다락에서 민수의 책을 받아 왔다.", details: ["사장님이 책을 신문지에 싸 주었다", "둘째 권 표지가 바랬다"] });
      } else {
        text = '*컵을 내려놓는다.* "네."';
      }
      return { text, model: "fake-1", latencyMs: 1, usage: { promptTokens: 10, completionTokens: 5 } };
    },
    async *stream() {
      yield "";
    },
  };
  return p;
};

describe("World (code only)", () => {
  it("validates routines against the place list", () => {
    expect(() => World.create({ ...def, characters: [{ id: "x", name: "x", tier: "main", routine: { morning: "nowhere" } }] })).toThrow(/unknown place/);
  });

  it("a due task moves the character; others at the same place are found", () => {
    const w = World.create(def);
    expect(w.whereIs("haru", 2, "afternoon")).toBe("cafe");
    w.addTask({ assignee: "haru", what: "책 받아 오기", where: "bookshop", dueDay: 2, dueSlot: "afternoon", from: "민수" });
    expect(w.whereIs("haru", 2, "afternoon")).toBe("bookshop");
    expect(w.whoElseAt("bookshop", 2, "afternoon", "haru")).toEqual(["yoon"]);
    expect(w.whoElseAt("bookshop", 2, "morning", "haru")).toEqual(["yoon"]);
    expect(w.whoElseAt("cafe", 2, "afternoon", "haru")).toEqual([]);
  });

  it("recording an event closes its task and facts carry the ledger's source", () => {
    const w = World.create(def);
    const t = w.addTask({ assignee: "haru", what: "책", where: "bookshop", dueDay: 2, from: "민수" });
    w.setClock(3);
    const ev = w.record({ day: 2, slot: "afternoon", where: "bookshop", who: ["haru", "yoon"], scope: "personal", result: "책을 받아 왔다", details: ["신문지에 싸 주었다"], knownBy: { haru: { source: "witnessed" }, yoon: { source: "witnessed" } }, origin: "scene", taskId: t.id });
    w.record({ day: 2, slot: "afternoon", where: "bookshop", who: ["haru", "yoon"], scope: "personal", result: "윤은 다음 주에 쉰다고 했다", details: [], knownBy: { haru: { source: "told", by: "윤" } }, origin: "scene" });
    expect(t.status).toBe("done");
    expect(t.eventId).toBe(ev.id);
    const haru = w.factsFor("haru", w.unsynced("haru"));
    expect(haru.map((f) => f.source)).toEqual(["witnessed", "witnessed", "told"]);
    expect(haru[0]!.text).toContain("yesterday");
    expect(haru[2]).toMatchObject({ by: "윤" });
    // yoon knows the meeting but was not told the second line
    expect(w.factsFor("yoon", w.unsynced("yoon"))).toHaveLength(2);
    w.markSynced("haru");
    expect(w.unsynced("haru")).toHaveLength(0);
  });
});

describe("World + model + session", () => {
  const card = findCharacter("haru");
  const yoonCard = { ...findCharacter("haru"), id: "yoon", name: { ko: "윤", ja: "ユン", en: "Yoon" } };

  it("extracts a task, resolves it alone when nobody else is there, and the character's prompt gains NOW + witnessed facts", async () => {
    const llm = fakeWorldLLM();
    const w = World.create({ ...def, characters: [def.characters[0]!] });
    const session = new ChatSession({ card, lang: "ko", provider: llm, persona: { name: "민수" }, memory: { enabled: false } });
    await session.send('"내일 책다락 가시면 제 책 좀 받아다 주세요."');
    const tasks = await extractTasks({ provider: llm }, { world: w, card, lang: "ko", persona: { name: "민수" }, turns: session.turns, characterId: "haru" });
    expect(tasks).toHaveLength(1);
    w.addTask(tasks[0]!);
    w.sawUser("haru");

    const before = session.buildMessages()[0]!.content;
    expect(before).not.toContain("# NOW");

    const events = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards: { haru: card } });
    expect(w.day).toBe(2);
    expect(events).toHaveLength(1);
    expect(events[0]!.origin).toBe("errand");
    expect(w.state.tasks[0]!.status).toBe("done");

    const added = syncCharacter(session, w, "haru");
    expect(added).toHaveLength(3); // result + 2 details
    const facts = session.memory.facts;
    expect(facts.every((f) => f.source === "witnessed")).toBe(true);
    expect(facts[0]!.text).toMatch(/^yesterday, afternoon, 헌책방/);
    const after = session.buildMessages()[0]!.content;
    expect(after).toContain("# NOW\nDay 2, morning. You are at 카페");
    expect(after).toContain("A day has passed since you last spoke with 민수");
    expect(after).toContain("책을 받아 왔다");
  });

  it("resolves the task as a scene when another main character is at the place, and `told` lines reach only the one who heard them", async () => {
    const llm = fakeWorldLLM();
    const w = World.create(def);
    w.addTask({ assignee: "haru", what: "책 받아 오기", where: "bookshop", dueDay: 1, dueSlot: "afternoon", from: "민수" });
    const events = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards: { haru: card, yoon: yoonCard }, persona: { name: "민수" } });
    expect(events).toHaveLength(1);
    expect(events[0]!.origin).toBe("scene");
    expect(events[0]!.scene).toContain("안녕하세요");
    // 1 scene event + 1 told line for each side
    expect(w.state.ledger).toHaveLength(3);
    const haru = w.factsFor("haru", w.unsynced("haru"));
    const yoon = w.factsFor("yoon", w.unsynced("yoon"));
    expect(haru.find((f) => f.source === "told")).toMatchObject({ by: "윤", text: expect.stringContaining("사흘 쉰다") });
    expect(yoon.find((f) => f.source === "told")).toMatchObject({ by: "하루", text: expect.stringContaining("만화") });
    expect(haru.some((f) => f.text.includes("만화"))).toBe(false);
    expect(llm.calls).toHaveLength(1); // one call for the whole scene
  });

  it("a routine day costs no calls, and a task due later today stays pending", async () => {
    const llm = fakeWorldLLM();
    const w = World.create(def);
    w.addTask({ assignee: "haru", what: "내일 일", where: "bookshop", dueDay: 2, dueSlot: "afternoon", from: "민수" });
    const events = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards: { haru: card } });
    expect(events).toEqual([]);
    expect(llm.calls).toHaveLength(0);
    expect(w.day).toBe(2);
    expect(w.state.tasks[0]!.status).toBe("pending");
  });
});
