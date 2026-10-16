import { describe, expect, it } from "vitest";
import { checkFormat, formatScore } from "./format.js";
import { judgeConversation } from "./judge.js";
import { FakeProvider } from "./providers.js";
import { summarise, toMarkdown } from "./report.js";
import { runConversation } from "./runner.js";
import { findScenario, loadScenarios } from "./scenarios.js";

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
    const ids = checkFormat(r, "ko", persona).filter((c) => !c.ok).map((c) => c.id);
    expect(ids).toEqual(expect.arrayContaining(["has_speech", "does_not_narrate_user", "no_ai_disclaimer"]));
  });

  it("detects Korean leaking into a Japanese reply", () => {
    const r = `*カップを置く。* 「…はい。」 안녕`;
    expect(checkFormat(r, "ja", persona).find((c) => c.id === "no_wrong_script")!.ok).toBe(false);
  });
});

describe("scenarios", () => {
  it("load and reference existing characters", () => {
    const ids = loadScenarios().map((s) => s.id).sort();
    expect(ids).toEqual(["haru-ja", "haru-ko", "kael-ko"]);
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
