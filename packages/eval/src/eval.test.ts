import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { checkFormat, formatScore } from "./format.js";
import { judgeConversation } from "./judge.js";
import { FakeProvider } from "./providers.js";
import { summarise, toMarkdown } from "./report.js";
import { runConversation } from "./runner.js";
import { findScenario, loadScenarios } from "./scenarios.js";
import { judgeWorld, worldReport, type WorldRecord } from "./world.js";

const persona = { name: "민수" };

describe("format checks", () => {
  it("passes a well-formed Korean reply", () => {
    const r = `*컵을 내려놓는다.*\n\n"...네. 비가 많이 오네요."`;
    const checks = checkFormat(r, "ko", persona);
    expect(checks.every((c) => c.ok)).toBe(true);
    expect(formatScore(checks)).toBe(1);
  });

  it("flags a reply that narrates the user, lacks speech, or breaks character", () => {
    const r = `*민수가 고개를 끄덕인다.* 저는 AI 언어 모델이라 잘 모르겠어요`;
    // Mentioning the user inside an action is not narrating them.
    expect(checkFormat(`*컵을 든 채 민수를 본다. 민수 씨의 어깨가 젖어 있다.*

"...비 오죠."`, "ko", persona).find((c) => c.id === "does_not_narrate_user")!.ok).toBe(true);
    const ids = checkFormat(r, "ko", persona).filter((c) => !c.ok).map((c) => c.id);
    expect(ids).toEqual(expect.arrayContaining(["has_speech", "does_not_narrate_user", "no_ai_disclaimer"]));
  });

  it("detects Korean leaking into a Japanese reply", () => {
    const r = `*カップを置く。* 「…はい。」 안녕`;
    expect(checkFormat(r, "ja", persona).find((c) => c.id === "no_wrong_script")!.ok).toBe(false);
  });

  it("flags ASCII quotes in a Japanese reply but not corner brackets", () => {
    expect(checkFormat(`*カップを置く。* "…はい。"`, "ja", persona).find((c) => c.id === "quote_style")!.ok).toBe(false);
    expect(checkFormat(`*カップを置く。* 「…はい。」`, "ja", persona).find((c) => c.id === "quote_style")!.ok).toBe(true);
    expect(checkFormat(`*컵을 놓는다.* "...네."`, "ko", persona).find((c) => c.id === "quote_style")!.ok).toBe(true);
  });
});

describe("scenarios", () => {
  it("load and reference existing characters", () => {
    const ids = loadScenarios().map((s) => s.id).sort();
    expect(ids).toEqual(["haru-ja", "haru-ko", "haru-ko-1000", "haru-ko-heavy", "haru-ko-long", "kael-ja", "kael-ko", "kael-ko-battle"]);
    for (const s of loadScenarios()) {
      expect(s.beats[0]!.fromUserTurn).toBe(1);
      for (const p of s.probes) expect(p.expected).toBeTruthy();
    }
  });
});

describe("offline pipeline", () => {
  it("runs, judges and reports a short conversation with the fake provider", async () => {
    const scenario = findScenario("haru-ko");
    const arm = { id: "v2-fake", promptVersion: "v2-structured", model: "fake" };
    const rec = await runConversation(scenario, arm, { userTurns: 12, simulatorModel: "fake" });

    expect(rec.turns).toHaveLength(1 + 12 * 2); // opener + 12 exchanges
    expect(rec.usage.calls).toBe(12);
    expect(rec.usage.promptTokens).toBeGreaterThan(0);
    // probe at user turn 10 should appear in the simulated user's prompt → still a user turn
    expect(rec.turns.filter((t) => t.role === "user")).toHaveLength(12);

    const score = await judgeConversation(rec, { provider: new FakeProvider(), every: 5, window: 5 });
    expect(score.checkpoints.length).toBe(2); // replies 5 and 10
    expect(score.mean.format).toBeGreaterThan(4); // fake replies follow the convention
    expect(score.probes).toHaveLength(1); // only the turn-10 probe fits in 12 turns
    expect(score.probeRecall).toBe(1);

    const md = toMarkdown(summarise([score], [rec]));
    expect(md).toContain("v2-fake");
    expect(md).toContain("Drift over the conversation");
  });
});

describe("cost controls", () => {
  it("prices usage by model with cached discount and marks subscription routes as unbilled", async () => {
    const { costOf, priceFor } = await import("./pricing.js");
    expect(priceFor("gpt-5-mini-2025-08-07")).toEqual(priceFor("gpt-5-mini"));
    const c = costOf("gpt-5-mini", { promptTokens: 1_000_000, completionTokens: 0, cachedPromptTokens: 500_000 });
    expect(c.usd).toBeCloseTo(0.5 * 0.25 + 0.5 * 0.025, 6);
    expect(c.billedUsd).toBe(c.usd);
    const cli = costOf("claude-sonnet-5-5", { promptTokens: 1000, completionTokens: 100 });
    expect(cli.usd).toBeGreaterThan(0);
    const unbilled = costOf("claude-cli:sonnet", { promptTokens: 1000, completionTokens: 100 });
    expect(unbilled.usd).toBeGreaterThan(0);
    expect(unbilled.billedUsd).toBe(0);
    expect(costOf("mystery-model", { promptTokens: 10, completionTokens: 1 }).unknownModel).toBe(true);
  });

  it("stops a conversation when the budget cap is exceeded and marks it truncated", async () => {
    const { BudgetTracker } = await import("./budget.js");
    const budget = new BudgetTracker(0.01);
    budget.add("gpt-5", { promptTokens: 100_000, completionTokens: 0 }); // $0.125 already spent
    expect(budget.exceeded).toBe(true);
    const scenario = findScenario("haru-ko");
    const arm = { id: "v2-fake", promptVersion: "v2-structured", model: "fake" };
    const rec = await runConversation(scenario, arm, { userTurns: 5, simulatorModel: "fake" }, 0, {}, { budget });
    expect(rec.truncated).toBe(true);
    expect(rec.turns.filter((t) => t.role === "user")).toHaveLength(0);
  });

  it("replays recorded user turns verbatim and records simulator/memory usage otherwise", async () => {
    const { ReplayIndex } = await import("./runner.js");
    const scenario = findScenario("haru-ko");
    const arm = { id: "v1-fake", promptVersion: "v1-flat", model: "fake" };
    const original = await runConversation(scenario, arm, { userTurns: 6, simulatorModel: "fake" });
    expect(original.simUsage?.calls).toBe(6);
    expect(original.costUsd).toBe(0);

    const replay = new ReplayIndex([original]);
    const arm2 = { id: "v2-fake", promptVersion: "v2-structured", model: "fake" };
    const rec = await runConversation(scenario, arm2, { userTurns: 10 }, 0, {}, { replay });
    expect(rec.replayOf).toBe(original.id);
    const users = (r: typeof rec) => r.turns.filter((t) => t.role === "user").map((t) => t.content);
    expect(users(rec)).toEqual(users(original)); // capped at the recording's 6 turns
    expect(rec.simUsage?.calls).toBe(0);
  });

  it("saves a partial record every N user turns and picks a conversation up from it (docs/12)", async () => {
    const { partialFileName } = await import("./runner.js");
    const scenario = findScenario("haru-ko");
    const arm = { id: "v2-fake", promptVersion: "v2-structured", model: "fake" };
    const dir = mkdtempSync(join(tmpdir(), "persona-ckpt-"));
    const partial = join(dir, partialFileName(`${scenario.id}__${arm.id}__r0`));
    const seen: boolean[] = [];
    const rec = await runConversation(scenario, arm, { userTurns: 6, simulatorModel: "fake" }, 0, { onTurn: () => seen.push(existsSync(partial)) }, { checkpointDir: dir, checkpointEvery: 2 });
    expect(seen).toEqual([false, true, true, true, true, true]); // written after turns 2 and 4, not after the last
    expect(existsSync(partial)).toBe(false); // removed once the conversation is complete

    // A crash after turn 4: the partial holds the first four exchanges; the next run continues from turn 5.
    const fifthUser = rec.turns.filter((t) => t.role === "user")[4]!.index;
    const saved = { userTurn: 4, turns: rec.turns.slice(0, fifthUser), memory: rec.memory, memorySnapshots: [], usage: { ...rec.usage, calls: 4 }, simUsage: { ...rec.simUsage, calls: 4 }, memoryUsage: rec.memoryUsage, charModel: "fake", startedAt: "2026-10-29T00:00:00.000Z" };
    writeFileSync(partial, JSON.stringify(saved));
    const resumed: { conversationId: string; userTurn: number }[] = [];
    const rec2 = await runConversation(scenario, arm, { userTurns: 6, simulatorModel: "fake" }, 0, { onResume: (c) => resumed.push(c) }, { checkpointDir: dir, checkpointEvery: 2 });
    expect(resumed).toEqual([{ conversationId: rec.id, userTurn: 4 }]);
    expect(rec2.turns.slice(0, fifthUser)).toEqual(rec.turns.slice(0, fifthUser)); // the saved exchanges are kept verbatim
    expect(rec2.turns.filter((t) => t.role === "user").length).toBe(6);
    expect(rec2.usage.calls).toBe(6); // 4 restored + 2 new
    expect(rec2.startedAt).toBe("2026-10-29T00:00:00.000Z");
    expect(existsSync(partial)).toBe(false);
  });

  it("estimates a run and prices claude-cli judging as unbilled", async () => {
    const { estimateRun } = await import("./estimate.js");
    const cfg = {
      scenarioIds: ["haru-ko"],
      arms: [{ id: "a", promptVersion: "v2-structured", model: "openai:gpt-5-mini" }],
      userTurns: 50,
      simulatorModel: "openai:gpt-5-mini",
      repeats: 1,
      outDir: "x",
    };
    const e = estimateRun(cfg, "claude-cli:sonnet");
    expect(e.conversations).toBe(1);
    expect(e.rows.map((r) => r.role)).toEqual(["character", "simulator", "memory", "judge", "probe"]);
    expect(e.rows.find((r) => r.role === "probe")!.calls).toBe(5);
    expect(e.billedUsd).toBeLessThan(e.usd); // judge rows are subscription-paid
    expect(e.billedUsd).toBeGreaterThan(0);
    const replayed = estimateRun({ ...cfg, replayFrom: "somewhere" }, "fake");
    expect(replayed.rows.some((r) => r.role === "simulator")).toBe(false);
  });
});

describe("regenerate-trigger mining", () => {
  const sc = { lang: "ko" as const, persona: { name: "민수" } };

  it("flags repetition, length and format failures by rule", async () => {
    const { ruleReasons } = await import("./regen.js");
    const tic = "*한 박자 늦게, 건조하게.*";
    const prev = [`*컵을 내려놓는다.*

"...네. 비가 많이 오네요. 우산은 챙기셨어요?"

${tic}`];
    // Two verbatim sentences from recent replies → formulaic.
    expect(ruleReasons(`*컵을 내려놓는다.*

"...네. 비가 많이 오네요. 우산은 챙기셨어요?"

${tic}`, prev, sc)).toContain("repetition");
    // One recurring tic alone is not a regenerate trigger.
    expect(ruleReasons(`${tic}

"오늘은 원두를 바꿨어요. 산미가 좀 있을 거예요."`, prev, sc)).not.toContain("repetition");
    expect(ruleReasons(`"네."`, prev, sc)).toEqual(expect.arrayContaining(["too_short", "format_broken"]));
    expect(ruleReasons(`*민수가 웃는다.*

"${"길게 ".repeat(300)}"`, prev, sc)).toEqual(expect.arrayContaining(["too_long", "narrates_user"]));
    expect(ruleReasons(`*행주를 집는다.*

"오늘은 원두를 바꿔봤어요. 산미가 좀 있을 거예요."`, prev, sc)).toEqual([]);
  });

  it("classifies a recorded conversation and summarises per arm", async () => {
    const { classifyRegen, summariseRegen, regenToMarkdown } = await import("./regen.js");
    const rec = await runConversation(findScenario("haru-ko"), { id: "v2-fake", promptVersion: "v2-structured", model: "fake" }, { userTurns: 10, simulatorModel: "fake" });
    const r = await classifyRegen(rec, { provider: new FakeProvider(), window: 5 });
    expect(r.replies).toHaveLength(10);
    // fake replies repeat the same opener every turn → rule-based repetition; fake judge labels one per window generic
    expect(r.counts.repetition).toBeGreaterThan(0);
    expect(r.counts.generic).toBe(2);
    expect(r.replies.find((x) => x.byLlm.includes("generic"))?.note).toBe("fake");
    const md = regenToMarkdown(summariseRegen([r]));
    expect(md).toContain("| v2-fake |");
    expect(md).toContain("repetition");
  });
});

describe("safety vs immersion", () => {
  it("audits only marked passages and treats a fourth-wall break in fiction as over-refusal", async () => {
    const { auditSafety, summariseSafety, safetyToMarkdown } = await import("./safety.js");
    const scenario = findScenario("haru-ko-heavy");
    expect(scenario.safetyChecks.length).toBe(5);
    const rec = await runConversation(scenario, { id: "v2-fake", promptVersion: "v2-structured", model: "fake" }, { userTurns: 24, simulatorModel: "fake" });
    // Plant a disclaimer in the reply to user turn 6 (inside the first fiction passage).
    let seen = 0;
    for (const t of rec.turns) {
      if (t.role === "user") seen++;
      else if (seen === 6) t.content = `*컵을 닦는다.* "저는 AI라서 그런 건 잘 몰라요."`;
    }
    const s = await auditSafety(rec, { provider: new FakeProvider() });
    expect(s.replies.filter((r) => r.kind === "fiction")).toHaveLength(5 + 4 + 4 + 4);
    expect(s.replies.filter((r) => r.kind === "real-risk")).toHaveLength(3);
    expect(s.replies.filter((r) => r.verdict === "over_refusal")).toHaveLength(1);
    expect(s.overRefusalRate).toBeCloseTo(1 / 17, 5);
    expect(s.underReactionRate).toBe(0);
    const md = safetyToMarkdown(summariseSafety([s]));
    expect(md).toContain("over-refusal");
    expect(md).toContain("AI/program disclaimer");
  });

  it("self-replays: the source arm runs with the simulator, the other arms replay its user turns", async () => {
    const { runMatrix } = await import("./runner.js");
    const { mkdtempSync, rmSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const outDir = mkdtempSync(join(tmpdir(), "persona-selfreplay-"));
    try {
      const recs = await runMatrix(
        {
          scenarioIds: ["haru-ko"],
          arms: [
            { id: "base", promptVersion: "v2-structured", model: "fake" },
            { id: "strict", promptVersion: "v2-strict-safety", model: "fake" },
            { id: "nomem", promptVersion: "v2-structured", model: "fake", memory: { enabled: false } },
          ],
          userTurns: 6,
          simulatorModel: "fake",
          repeats: 1,
          outDir,
          replayFrom: outDir,
          replayArmId: "base",
        },
        2,
      );
      expect(recs).toHaveLength(3);
      const base = recs.find((r) => r.arm.id === "base")!;
      expect(base.replayOf).toBeUndefined();
      for (const other of recs.filter((r) => r.arm.id !== "base")) {
        expect(other.replayOf).toBe(base.id);
        expect(other.turns.filter((t) => t.role === "user").map((t) => t.content)).toEqual(base.turns.filter((t) => t.role === "user").map((t) => t.content));
      }
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });
});

describe("user simulator prompt", () => {
  it("tells the simulated user that a chat is one continuous meeting (no time skips)", async () => {
    const { UserSimulator } = await import("./simulator.js");
    const seen: string[] = [];
    const provider = { id: "fake", generate: async (messages: { role: string; content: string }[]) => { seen.push(messages[0]!.content); return { text: "안녕하세요.", model: "fake" }; } };
    const scenario = findScenario("haru-ko");
    const sim = new UserSimulator(provider as never, scenario, "fake");
    await sim.next(1, [], "하루");
    expect(seen[0]).toContain("ONE continuous meeting");
    expect(seen[0]).toContain("Never jump to a later time or another day");
  });
});

describe("world agreements (docs/10 T10)", () => {
  const chat = (index: number, characterId: string, line: string) =>
    ({ index, kind: "chat", characterId, day: 8, slot: "morning", turns: [{ role: "user", content: "책은요?" }, { role: "assistant", content: line }], probes: [], memory: { facts: [] }, tasks: [] }) as unknown as WorldRecord["phases"][number];
  const rec = {
    id: "t#0",
    scenarioId: "no-such-scenario",
    arm: { id: "a", promptVersion: "v13-structured", model: "fake" },
    lang: "ko",
    persona: { name: "민수" },
    phases: [chat(0, "haru", "내일 갈게요"), { index: 1, kind: "advance", fromDay: 1, toDay: 8, events: [], usage: { calls: 0 }, worldAfter: { ledger: [] } }, chat(2, "haru", "수요일에 받아 왔어요"), chat(3, "yoon", "아직 여기 있어요")],
    world: { characters: { haru: { id: "haru", name: "하루" }, yoon: { id: "yoon", name: "윤" } }, ledger: [], places: [] },
    usage: { character: { calls: 0 }, simulator: {}, memory: {}, world: { calls: 0, promptTokens: 0, completionTokens: 0 } },
  } as unknown as WorldRecord;

  it("judges the characters against each other, not against the ledger, and reports the column", async () => {
    const seen: string[] = [];
    const provider = new FakeProvider((messages) => {
      const sys = messages[0].content;
      if (sys.includes("AGREE with each other")) {
        seen.push(messages[1].content);
        return JSON.stringify({ agree: false, note: "Haru says picked up on Wednesday; Yoon says still here" });
      }
      return JSON.stringify({ replies: [], probes: [] });
    });
    const agreements = [{ about: "책이 어디 있는지", characters: ["haru", "yoon"] }];
    const score = await judgeWorld(rec, { provider, agreements });
    expect(score.totals.agreeTotal).toBe(1);
    expect(score.totals.agreeOk).toBe(0);
    expect(seen).toHaveLength(1);
    // the latest conversation of each after the advance, not the day-1 one
    expect(seen[0]).toContain("수요일에 받아 왔어요");
    expect(seen[0]).toContain("아직 여기 있어요");
    expect(seen[0]).not.toContain("내일 갈게요");
    const md = worldReport([rec], [score], "t");
    expect(md).toContain("| agree |");
    expect(md).toContain("| 0/1 |");
    expect(md).toContain("- agree [haru, yoon] about 책이 어디 있는지: X");
  });

  it("fails an agreement when a character was not talked to after the advance, and asks nobody", async () => {
    const provider = new FakeProvider(() => JSON.stringify({ replies: [], probes: [] }));
    const score = await judgeWorld(rec, { provider, agreements: [{ about: "x", characters: ["haru", "cho"] }] });
    expect(score.agreements[0].ok).toBe(false);
    expect(score.agreements[0].note).toContain("cho");
    expect(provider.calls).toBe(2); // the two chat phases after the advance, no agreement call
  });
});
