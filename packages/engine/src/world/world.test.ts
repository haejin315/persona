import { describe, expect, it } from "vitest";
import { findCharacter } from "../characters.js";
import type { LLMProvider } from "../providers/types.js";
import { ChatSession } from "../session.js";
import { advanceDay, applyAttention, catchUp, decideAttention, extractTasks, newsFor } from "./llm.js";
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
const fakeWorldLLM = (opts: { errand?: "done" | "failed" } = {}): LLMProvider & { calls: string[]; systems: string[] } => {
  const p = {
    id: "fake",
    defaultModel: "fake-1",
    calls: [] as string[],
    systems: [] as string[],
    async generate(messages: { role: string; content: string }[]) {
      const sys = messages[0]!.content;
      p.calls.push(sys.slice(0, 40));
      p.systems.push(sys);
      let text: string;
      if (sys.includes("AGREED to do later")) {
        text = JSON.stringify({ tasks: [{ what: "책다락에 가서 민수가 맡겨 둔 책이 들어왔는지 확인하고 받아 온다", where: "bookshop", dueDay: 1, dueSlot: "afternoon" }] });
      } else if (sys.includes("two characters met")) {
        text = JSON.stringify({
          scene: '*문이 열린다.* "안녕하세요."',
          result: "하루가 윤에게서 민수의 책을 받아 왔다.",
          details: ["윤이 책을 신문지에 싸 주었다", "둘째 권 표지가 조금 바랬다"],
          told: { haru: ["윤은 다음 주에 가게를 사흘 쉰다고 했다"], yoon: ["[about 민수] 하루는 민수가 어릴 때 만화를 그렸다고 전했다"] },
        });
      } else if (sys.includes("Decide what, if anything, happened on day")) {
        const day = Number(sys.match(/happened on day (\d+)/)?.[1] ?? 0);
        text = JSON.stringify({ events: day % 2 === 0 ? [{ slot: "morning", where: "cafe", who: ["haru"], scope: "personal", result: `day ${day}: 원두가 바뀌었다`, details: ["새 원두는 신맛이 덜하다"] }] : [] });
      } else if (sys.includes("pass with nobody watching")) {
        const m = sys.match(/Days (\d+) to (\d+)/)!;
        text = JSON.stringify({ events: [{ day: Number(m[1]) + 1, slot: "afternoon", where: "bookshop", who: ["yoon"], scope: "regional", result: `days ${m[1]}-${m[2]}: 책다락이 일주일 쉬었다 열었다`, details: [], effects: { bookshop: null } }] });
      } else if (sys.includes("Choose who lives in full")) {
        text = JSON.stringify({ main: ["haru"], why: { haru: "holds the errand" } });
      } else if (sys.includes("check a story event")) {
        text = JSON.stringify({ ok: true, problems: [] });
      } else if (sys.includes("did an errand")) {
        text = JSON.stringify({ outcome: opts.errand ?? "done", result: "하루가 책다락에서 민수의 책을 받아 왔다.", details: ["사장님이 책을 신문지에 싸 주었다", "둘째 권 표지가 바랬다"] });
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

  it("the same outing asked for twice is one task; a task at the routine place is dropped", async () => {
    const w = World.create(def);
    const a = w.addTask({ assignee: "haru", what: "민수 책 받아 오기", where: "bookshop", dueDay: 1, dueSlot: "afternoon", from: "민수" });
    const b = w.addTask({ assignee: "haru", what: "이모 책 받아 오기", where: "bookshop", dueDay: 1, dueSlot: "afternoon", from: "이모" });
    expect(b).toBe(a);
    expect(w.state.tasks).toHaveLength(1);
    expect(a.what).toBe("민수 책 받아 오기 / 이모 책 받아 오기");
    const llm = {
      id: "fake",
      defaultModel: "fake-1",
      async generate() {
        return { text: JSON.stringify({ tasks: [{ what: "7시에 카페 문 열기", where: "cafe", dueDay: 2, dueSlot: "morning" }, { what: "책", where: "bookshop", dueDay: 2, dueSlot: "afternoon" }] }), model: "fake-1", latencyMs: 1 };
      },
      async *stream() {
        yield "";
      },
    };
    const card = findCharacter("haru");
    const found = await extractTasks({ provider: llm }, { world: w, card, lang: "ko", persona: { name: "민수" }, turns: [], characterId: "haru" });
    expect(found.map((t) => t.what)).toEqual(["책"]);
  });

  it("weekdays follow the clock; an event's effects change a place's condition and reach the NOW block", () => {
    const w = World.create({ ...def, startWeekday: "Friday" });
    expect(w.weekday).toBe("Friday");
    expect(w.weekdayOf(4)).toBe("Monday");
    w.record({ day: 1, slot: "evening", where: "bookshop", who: ["yoon"], scope: "regional", result: "수도관이 터졌다", details: [], knownBy: { yoon: { source: "witnessed" } }, origin: "catchup", effects: { bookshop: "임시 휴업 (수도관 파열), day 4까지" } });
    expect(w.condition("bookshop")).toContain("임시 휴업");
    expect(w.condition("cafe")).toBeUndefined();
    const block = w.blockFor("yoon");
    expect(block.weekday).toBe("Friday");
    expect(block.placeCondition).toContain("임시 휴업");
    expect(w.blockFor("haru").placeCondition).toBeUndefined();
    w.record({ day: 4, slot: "morning", where: "bookshop", who: ["yoon"], scope: "regional", result: "다시 열었다", details: [], knownBy: {}, origin: "catchup", effects: { bookshop: null } });
    expect(w.condition("bookshop")).toBeUndefined();
    expect(() => w.record({ day: 1, slot: "morning", where: "cafe", who: [], scope: "world", result: "x", details: [], knownBy: {}, origin: "catchup", effects: { nowhere: "y" } })).toThrow(/unknown place/);
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
    expect(haru[0]!.text).toMatch(/^afternoon, 헌책방/);
    expect(haru[0]!.day).toBe(2);
    expect(haru[2]).toMatchObject({ by: "윤", day: 2 });
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
    expect(facts[0]!.text).toMatch(/^afternoon, 헌책방/);
    const after = session.buildMessages()[0]!.content;
    expect(after).toContain("# NOW\nDay 2 (Tuesday), morning. You are at 카페");
    expect(after).toContain("A day has passed since you last spoke with 민수");
    expect(after).toContain("- [day 1, yesterday] afternoon, 헌책방 '책다락': 하루가 책다락에서 민수의 책을 받아 왔다.");
    // the label follows the clock: two days later the same fact reads "2 days ago"
    w.setClock(3);
    session.world = w.blockFor("haru");
    expect(session.buildMessages()[0]!.content).toContain("[day 1, 2 days ago] afternoon");
    w.setClock(2);
    session.world = w.blockFor("haru");
    // the next user message carries the day boundary into the transcript
    await session.send('"어제 다녀오셨어요?"');
    const msgs = session.buildMessages();
    const lastUser = [...msgs].reverse().find((m) => m.role === "user")!;
    expect(lastUser.content).toMatch(/^\[The next day: day 2, morning, at 카페 '모퉁이'\]\n"어제/);
    expect(session.turns.at(-2)!.meta?.boundary).toBeDefined();
    expect(events[0]!.checked).toEqual({ ok: true, problems: [], retried: false });
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
    expect(llm.calls).toHaveLength(2); // the scene + its check
  });

  it("runs a routine scene only when co-located characters have news, at most scenesPerDay", async () => {
    const llm = fakeWorldLLM();
    const w = World.create({
      ...def,
      places: [...def.places, { id: "market", name: "시장" }],
      characters: [
        ...def.characters,
        { id: "imo", name: "이모", tier: "main", routine: { morning: "market", afternoon: "cafe", evening: "bookshop", night: "home" } },
      ],
    });
    const imoCard = { ...card, id: "imo", name: { ko: "이모", ja: "おば", en: "Aunt" } };
    const cards = { haru: card, yoon: yoonCard, imo: imoCard };
    // Day 1: nobody knows anything yet -> no routine scene even though imo and yoon share the evening.
    expect(await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards })).toEqual([]);
    expect(llm.calls).toHaveLength(0);
    // Day 2: a task scene haru x yoon in the afternoon gives yoon news; imo meets yoon in the evening.
    w.addTask({ assignee: "haru", what: "책", where: "bookshop", dueDay: 2, dueSlot: "afternoon", from: "민수" });
    const events = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards });
    expect(events.map((e) => [...e.who].sort())).toEqual([
      ["haru", "yoon"],
      ["imo", "yoon"],
    ]);
    expect(events[1]!.slot).toBe("evening");
    // the fake scene reports told lines for haru/yoon only; the second scene's teller for yoon is imo
    expect(w.factsFor("yoon", w.unsynced("yoon")).some((f) => f.source === "told" && f.by === "이모")).toBe(true);
    // a told line marked [about 민수] becomes a fact about the user's things, without the mark in its text
    const marked = w.factsFor("yoon", w.unsynced("yoon")).find((f) => f.source === "told" && f.about === "user");
    expect(marked?.text).toMatch(/하루는 민수가 어릴 때 만화를 그렸다고 전했다$/);
    expect(marked?.text).not.toContain("[about");
    expect(w.factsFor("imo", w.unsynced("imo")).some((f) => f.source === "witnessed")).toBe(true);
    // haru and imo share the cafe in the afternoon but the cap (1) was spent
    expect(llm.calls.filter((c) => c.startsWith("You write what happened"))).toHaveLength(2);
  });

  it("two errands at the same place and slot are one outing", async () => {
    const llm = fakeWorldLLM();
    const w = World.create({ ...def, characters: [def.characters[0]!] });
    w.addTask({ assignee: "haru", what: "이모 책", where: "bookshop", dueDay: 1, dueSlot: "afternoon", from: "이모" });
    w.addTask({ assignee: "haru", what: "민수 책", where: "bookshop", dueDay: 1, dueSlot: "afternoon", from: "민수" });
    const events = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards: { haru: card } });
    expect(events).toHaveLength(1);
    // addTask already merged them; the grouping in advanceDay covers tasks that differ in slot text
    expect(w.state.tasks.map((t) => t.status)).toEqual(["done"]);
    expect(w.state.tasks[0]!.what).toBe("이모 책 / 민수 책");
    expect(w.state.tasks.map((t) => t.eventId)).toEqual([events[0]!.id]);
    expect(llm.calls.filter((c) => c.startsWith("You decide what happened"))).toHaveLength(1);
  });

  it("a task in a slot the character spent with the user is deferred, not done", async () => {
    const llm = fakeWorldLLM();
    const w = World.create({ ...def, characters: [def.characters[0]!] });
    w.addTask({ assignee: "haru", what: "책 받아 오기", where: "bookshop", dueDay: 1, dueSlot: "afternoon", from: "민수" });
    w.setClock(1, "afternoon");
    w.recordChat("haru", "민수", "cafe");
    expect(w.whereIs("haru", 1, "afternoon")).toBe("bookshop"); // the plan
    const events = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards: { haru: card }, persona: { name: "민수" } });
    expect(events).toHaveLength(1);
    expect(events[0]!.origin).toBe("catchup");
    expect(events[0]!.result).toContain("did not go to 헌책방");
    expect(w.state.tasks[0]).toMatchObject({ status: "pending", dueDay: 2 });
    expect(llm.calls).toHaveLength(0);
    // next day the errand happens
    const later = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards: { haru: card }, persona: { name: "민수" } });
    expect(later[0]!.origin).toBe("errand");
    expect(w.state.tasks[0]!.status).toBe("done");
    // a slot spent with the user has no routine scene either
    const w2 = World.create({ ...def, places: [...def.places, { id: "market", name: "시장" }], characters: [def.characters[0]!, { id: "imo", name: "이모", tier: "main", routine: { morning: "market", afternoon: "cafe", evening: "bookshop", night: "market" } }] });
    w2.record({ day: 1, slot: "morning", where: "cafe", who: ["haru"], scope: "personal", result: "x", details: ["y"], knownBy: { haru: { source: "witnessed" } }, origin: "errand" });
    w2.setClock(1, "afternoon");
    w2.recordChat("haru", "민수", "cafe");
    const imoCard = { ...card, id: "imo", name: { ko: "이모", ja: "おば", en: "Aunt" } };
    const none = await advanceDay(w2, { llm: { provider: llm }, lang: "ko", cards: { haru: card, imo: imoCard } });
    expect(none).toEqual([]);
    // news from the evening is not news in that day's afternoon (no time travel)
    const w3 = World.create({ ...def, places: [...def.places, { id: "market", name: "시장" }], characters: [def.characters[0]!, { id: "imo", name: "이모", tier: "main", routine: { morning: "market", afternoon: "cafe", evening: "bookshop", night: "market" } }] });
    w3.record({ day: 1, slot: "evening", where: "bookshop", who: ["imo"], scope: "regional", result: "수도관이 터졌다", details: [], knownBy: { imo: { source: "witnessed" } }, origin: "catchup" });
    expect(await advanceDay(w3, { llm: { provider: llm }, lang: "ko", cards: { haru: card, imo: imoCard } })).toEqual([]);
    const next = await advanceDay(w3, { llm: { provider: llm }, lang: "ko", cards: { haru: card, imo: imoCard } });
    expect(next.map((e) => [e.day, e.slot])).toEqual([[2, "afternoon"]]);
    const facts = w.factsFor("haru", w.unsynced("haru"));
    expect(facts.map((f) => f.day)).toEqual([1, 2, 2, 2]); // deferral, errand + 2 details (the chat itself is not a fact)
  });

  it("the world's own events appear only when eventsPerDay > 0, and a long gap is spans plus recent days", async () => {
    const llm = fakeWorldLLM();
    const w = World.create({ ...def, characters: [def.characters[0]!, { ...def.characters[1]!, routine: { ...def.characters[1]!.routine, night: "bookshop" } }] });
    const cards = { haru: card, yoon: yoonCard };
    expect(await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards })).toEqual([]); // default: off
    const one = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards, eventsPerDay: 1 }); // day 2 ends
    expect(one.map((e) => e.origin)).toEqual(["world"]);
    expect(one[0]!.knownBy).toEqual({ haru: { source: "witnessed" } });
    llm.calls.length = 0;
    // 40 days away from day 3: days 3..35 in two spans (30 + 3), days 36..42 day by day
    const r = await catchUp(w, { llm: { provider: llm }, lang: "ko", cards, eventsPerDay: 1, recentDays: 7, spanDays: 30 }, 40);
    expect(w.day).toBe(43);
    expect(r.inSpans).toBe(33);
    expect(r.dayByDay).toBe(7);
    const spanCalls = llm.calls.filter((c) => c.startsWith("You are the world of a quiet") ).length;
    expect(spanCalls).toBe(2 + 7); // 2 spans + 7 days
    expect(r.events.filter((e) => e.origin === "span")).toHaveLength(2);
    expect(r.events.filter((e) => e.origin === "span").every((e) => e.day >= 3 && e.day <= 35)).toBe(true);
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

  it("a failed errand (the shop was shut) is remembered, retried the next day, then given up", async () => {
    const llm = fakeWorldLLM({ errand: "failed" });
    const w = World.create(def);
    const t = w.addTask({ assignee: "haru", what: "책", where: "bookshop", dueDay: 1, dueSlot: "afternoon", from: "민수" });
    const first = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards: { haru: card }, verifier: null });
    expect(first).toHaveLength(1);
    expect(first[0]!.origin).toBe("errand");
    expect(first[0]!.taskId).toBeUndefined();
    expect(t).toMatchObject({ status: "pending", dueDay: 2, attempts: 1 });
    const second = await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards: { haru: card }, verifier: null });
    expect(second).toHaveLength(1);
    expect(t.status).toBe("done");
    expect(t.eventId).toBe(second[0]!.id);
    expect(w.factsFor("haru", w.unsynced("haru")).map((f) => f.day)).toEqual([1, 1, 1, 2, 2, 2]);
  });

  it("an errand due inside a span is a scene when the other main is there, so both remember it", async () => {
    const llm = fakeWorldLLM();
    const w = World.create(def);
    w.addTask({ assignee: "haru", what: "책", where: "bookshop", dueDay: 5, dueSlot: "afternoon", from: "민수" });
    const r = await catchUp(w, { llm: { provider: llm }, lang: "ko", cards: { haru: card, yoon: yoonCard }, eventsPerDay: 0, recentDays: 7, spanDays: 30 }, 40);
    const scene = r.events.find((e) => e.origin === "scene");
    expect(scene).toBeDefined();
    expect(scene!.day).toBe(5);
    expect(Object.keys(scene!.knownBy).sort()).toEqual(["haru", "yoon"]);
    expect(w.state.tasks[0]!.status).toBe("done");
    expect(w.factsFor("yoon", w.unsynced("yoon")).some((f) => f.day === 5 && f.source === "witnessed")).toBe(true);
  });

  it("a regional event is rumour around its region by the end of the day, and scenes can carry a what-to-tell list", async () => {
    const llm = fakeWorldLLM();
    const places = [
      { id: "cafe", name: "카페", region: "동네" },
      { id: "bookshop", name: "책방", region: "동네", hours: "11시~19시" },
      { id: "home", name: "집" },
      { id: "market", name: "시장", region: "동네" },
      { id: "far", name: "성수", region: "성수" },
    ];
    const w = World.create({
      ...def,
      places,
      characters: [
        ...def.characters,
        { id: "imo", name: "이모", tier: "main", routine: { morning: "market", afternoon: "cafe", evening: "bookshop", night: "home" } },
        { id: "kael", name: "카엘", tier: "main", routine: { morning: "far", afternoon: "far", evening: "far", night: "far" } },
      ],
    });
    const imoCard = { ...card, id: "imo", name: { ko: "이모", ja: "おば", en: "Aunt" } };
    const kaelCard = { ...card, id: "kael", name: { ko: "카엘", ja: "カエル", en: "Kael" } };
    const cards = { haru: card, yoon: yoonCard, imo: imoCard, kael: kaelCard };
    // the notice is seen in the morning, so in the afternoon the aunt is back at the cafe with Haru
    w.record({ day: 1, slot: "morning", where: "market", who: ["imo"], scope: "regional", result: "단수 안내문이 붙었다", details: ["화요일부터"], knownBy: { imo: { source: "witnessed" } }, origin: "catchup" });
    w.record({ day: 1, slot: "morning", where: "cafe", who: ["haru"], scope: "personal", result: "우유가 늦었다", details: [], knownBy: { haru: { source: "witnessed" } }, origin: "world" });
    await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards, sceneNews: true, persona: { name: "민수" }, userBackDay: 8 });
    const notice = w.state.ledger[0]!;
    expect(notice.knownBy.haru).toEqual({ source: "rumor" });
    expect(notice.knownBy.yoon).toEqual({ source: "rumor" });
    expect(notice.knownBy.imo).toEqual({ source: "witnessed" });
    expect(notice.knownBy.kael).toBeUndefined(); // another region
    expect(w.state.ledger[1]!.knownBy.yoon).toBeUndefined(); // personal events do not spread
    expect(w.factsFor("yoon", w.unsynced("yoon")).map((f) => f.source)).toContain("rumor");
    // a rumour is not knowing: the witness still has this as news for yoon, and yoon has nothing for the witness
    expect(newsFor(w, "imo", "yoon", 3, "morning").map((e) => e.result)).toContain("단수 안내문이 붙었다");
    expect(newsFor(w, "yoon", "imo", 3, "morning")).toEqual([]);
    // the afternoon scene haru x imo (both have news) carried the what-to-tell lists
    const scene = llm.systems.find((x) => x.startsWith("You write what happened"));
    expect(scene).toBeDefined();
    expect(scene).toContain("has to tell");
    expect(scene).toContain("단수 안내문이 붙었다");
    expect(scene).toContain("우유가 늦었다");
    expect(scene).toContain("will not be back before day 8");
    // off by default: no list
    const llm2 = fakeWorldLLM();
    const w2 = World.create({ ...def, places, characters: [...def.characters, { id: "imo", name: "이모", tier: "main", routine: { morning: "market", afternoon: "cafe", evening: "bookshop", night: "home" } }] });
    w2.record({ day: 1, slot: "morning", where: "cafe", who: ["haru"], scope: "personal", result: "우유가 늦었다", details: [], knownBy: { haru: { source: "witnessed" } }, origin: "world" });
    await advanceDay(w2, { llm: { provider: llm2 }, lang: "ko", cards: { haru: card, imo: imoCard } });
    expect(llm2.systems.find((x) => x.startsWith("You write what happened"))).not.toContain("has to tell");
  });

  it("what happened to the user's things is a marked fact, and the character is told to raise it unasked", () => {
    const w = World.create(def);
    w.record({ day: 1, slot: "evening", where: "bookshop", who: ["haru"], scope: "personal", result: "수도관이 터졌다", details: [], knownBy: { haru: { source: "witnessed" }, yoon: { source: "rumor" } }, origin: "catchup", userEffects: { "민수의 책": "1권 표지가 젖었다" } });
    expect(w.state.user).toEqual({ "민수의 책": "1권 표지가 젖었다" });
    // someone who only heard around gets the event, not what it did to the user's things
    expect(w.factsFor("yoon", w.unsynced("yoon")).some((f) => f.about === "user")).toBe(false);
    const facts = w.factsFor("haru", w.unsynced("haru"));
    expect(facts.find((f) => f.about === "user")).toMatchObject({ text: "민수의 책: 1권 표지가 젖었다", source: "witnessed", day: 1 });
    const session = new ChatSession({ card, lang: "ko", provider: fakeWorldLLM(), persona: { name: "민수" }, memory: { enabled: false } });
    w.setClock(2);
    syncCharacter(session, w, "haru");
    const sys = session.buildMessages()[0]!.content;
    expect(sys).toContain("[about 민수's things] 민수의 책: 1권 표지가 젖었다");
    expect(sys).toContain("bring it up early and unasked");
    // back to usual clears the state; a session without such a fact gets no such sentence
    w.record({ day: 2, slot: "morning", where: "bookshop", who: ["haru"], scope: "personal", result: "책을 말렸다", details: [], knownBy: { haru: { source: "witnessed" } }, origin: "catchup", userEffects: { "민수의 책": null } });
    expect(w.state.user).toEqual({});
    const plain = new ChatSession({ card, lang: "ko", provider: fakeWorldLLM(), persona: { name: "민수" }, memory: { enabled: false } });
    const w2 = World.create(def);
    w2.setClock(2);
    syncCharacter(plain, w2, "haru");
    expect(plain.buildMessages()[0]!.content).not.toContain("bring it up early and unasked");
  });

  it("a witness keeps the details of errands and of the last days; an older ordinary day keeps its result only", () => {
    const w = World.create(def);
    const t = w.addTask({ assignee: "haru", what: "책", where: "bookshop", dueDay: 1, from: "민수" });
    w.record({ day: 1, slot: "morning", where: "cafe", who: ["haru"], scope: "personal", result: "우유가 늦었다", details: ["기사가 길이 막혔다고 했다"], knownBy: { haru: { source: "witnessed" } }, origin: "world" });
    w.record({ day: 1, slot: "afternoon", where: "bookshop", who: ["haru"], scope: "personal", result: "책을 받아 왔다", details: ["표지가 바랬다"], knownBy: { haru: { source: "witnessed" } }, origin: "errand", taskId: t.id });
    w.record({ day: 7, slot: "morning", where: "cafe", who: ["haru"], scope: "personal", result: "원두가 바뀌었다", details: ["신맛이 덜하다"], knownBy: { haru: { source: "witnessed" } }, origin: "world" });
    w.setClock(8);
    const texts = w.factsFor("haru", w.unsynced("haru")).map((f) => f.text);
    expect(texts).toContain("표지가 바랬다");
    expect(texts).toContain("신맛이 덜하다");
    expect(texts).not.toContain("기사가 길이 막혔다고 했다");
    expect(texts.some((x) => x.includes("우유가 늦었다"))).toBe(true);
  });
});

describe("attention (docs/10 T10)", () => {
  const cards = { haru: findCharacter("haru"), yoon: findCharacter("yoon") };

  it("one call names who lives in full; the others become background extras with a sketch, and a rumour still reaches them", async () => {
    const w = World.create(def);
    w.state.lastSeen.haru = 1;
    w.addTask({ assignee: "haru", what: "책다락에 가서 민수의 책을 받아 온다", where: "bookshop", dueDay: 2, from: "민수" });
    const llm = fakeWorldLLM();
    const usage = { promptTokens: 0, completionTokens: 0, cachedPromptTokens: 0, calls: 0 };
    const d = await decideAttention(w, { llm: { provider: llm }, lang: "ko", cards, persona: { name: "민수" }, recentChats: [{ characterId: "haru", day: 1, transcript: "민수: 책 좀 받아다 주세요\n하루: 네." }] }, { daysAway: 7, cap: 2 }, usage);
    expect(d.main).toEqual(["haru"]);
    expect(d.why.haru).toBe("holds the errand");
    expect(usage.calls).toBe(1);
    const sys = llm.systems.find((s) => s.includes("Choose who lives in full"))!;
    expect(sys).toContain("책다락에 가서 민수의 책을 받아 온다");
    expect(sys).toContain("Last spoke with 민수: day 1 (0 days ago)");
    expect(sys).toContain("민수: 책 좀 받아다 주세요");
    expect(w.character("haru").tier).toBe("main");
    expect(w.character("yoon").tier).toBe("named");
    expect(w.character("yoon").sketch).toBe(cards.yoon.identity.role);
    // background characters with a card still hear what goes around town (zero calls)
    w.record({ day: 1, slot: "afternoon", where: "bookshop", who: ["haru"], scope: "world", result: "책다락 앞 골목이 파였다", details: [], knownBy: { haru: { source: "witnessed" } }, origin: "catchup" });
    await advanceDay(w, { llm: { provider: llm }, lang: "ko", cards, persona: { name: "민수" }, verifier: null });
    expect(w.state.ledger.find((e) => e.result === "책다락 앞 골목이 파였다")!.knownBy.yoon).toEqual({ source: "rumor" });
    // back to living in full
    applyAttention(w, cards, ["haru", "yoon"]);
    expect(w.character("yoon").tier).toBe("main");
  });
});
