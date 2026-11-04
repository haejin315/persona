import { describe, expect, it } from "vitest";
import { mulberry32, parseBeatsJson, placeProbes, type FactEntry } from "./extend.js";

const fact = (atUserTurn: number, expected: string, replaces?: string): FactEntry => ({ fact: `fact ${expected}`, expected, ask: `ask ${expected}`, atUserTurn, ...(replaces ? { replaces } : {}) });

describe("extend: probe placement (docs/13 plan C)", () => {
  const facts: FactEntry[] = [
    fact(30, "성수동"),
    fact(140, "달빛 고양이"),
    fact(340, "라떼"),
    fact(810, "망원동", "성수동"),
    fact(1105, "온도"),
    fact(1300, "아메리카노", "라떼"),
    ...Array.from({ length: 60 }, (_, i) => fact(1015 + i * 60, `fact-${i}`)),
  ];
  const opts = { from: 1001, to: 5000, every: 15, offset: 7, bands: [{ min: 30, max: 300, count: 6 }, { min: 300, max: 1500, count: 6 }, { min: 1500, max: 4500, count: 6 }], margin: 20, preferChanged: 2, seed: 7 };

  it("is deterministic for a seed, asks each fact once, one probe per slot, and only after the reveal margin", () => {
    const a = placeProbes(facts, opts);
    const b = placeProbes(facts, opts);
    expect(a).toEqual(b);
    expect(a.length).toBe(18);
    expect(new Set(a.map((p) => p.expected)).size).toBe(a.length);
    expect(new Set(a.map((p) => p.atUserTurn)).size).toBe(a.length);
    for (const p of a) {
      expect((p.atUserTurn - 1001 - 7) % 15).toBe(0);
      expect(p.atUserTurn - p.revealedAt).toBeGreaterThanOrEqual(20);
    }
    for (const band of opts.bands) expect(a.filter((p) => p.distance >= band.min && p.distance <= band.max).length).toBe(band.count);
  });

  it("never asks a replaced fact after its replacement, and asks the replacing fact for its new value", () => {
    const a = placeProbes(facts, opts);
    expect(a.some((p) => p.expected === "성수동")).toBe(false); // replaced at 810, before the probe range
    expect(a.some((p) => p.expected === "라떼" && p.atUserTurn >= 1300)).toBe(false);
    expect(a.filter((p) => p.changed).length).toBeGreaterThanOrEqual(2);
    expect(a.some((p) => p.expected === "망원동" || p.expected === "아메리카노")).toBe(true);
  });

  it("drops excluded facts and ambiguous answers, but allows an answer that returns after its earlier fact was replaced", () => {
    const dup: FactEntry[] = [
      fact(1, "아메리카노"),
      fact(340, "라떼", "아메리카노"),
      fact(1100, "10월"),
      fact(1200, "10월"), // same answer, earlier one still live → ambiguous, dropped
      fact(1500, "아메리카노", "라떼"), // the answer returns after 라떼 replaced the first → allowed
      fact(1600, "네 살"),
      ...Array.from({ length: 20 }, (_, i) => fact(1015 + i * 100, `f-${i}`)),
    ];
    const o = { ...opts, bands: [{ min: 30, max: 4000, count: 30 }], preferChanged: 0, exclude: (f: FactEntry) => /살$/.test(f.expected) };
    const a = placeProbes(dup, o);
    expect(a.filter((p) => p.expected === "10월").length).toBe(1);
    expect(a.find((p) => p.expected === "10월")!.revealedAt).toBe(1100);
    expect(a.some((p) => p.expected === "아메리카노" && p.revealedAt === 1500)).toBe(true);
    expect(a.some((p) => p.expected === "네 살")).toBe(false);
  });

  it("mulberry32 repeats for a seed and differs across seeds", () => {
    const r1 = mulberry32(1), r2 = mulberry32(1), r3 = mulberry32(2);
    const s1 = [r1(), r1(), r1()], s2 = [r2(), r2(), r2()], s3 = [r3(), r3(), r3()];
    expect(s1).toEqual(s2);
    expect(s1).not.toEqual(s3);
    for (const x of s1) expect(x >= 0 && x < 1).toBe(true);
  });
});

describe("extend: beats JSON from the model", () => {
  it("accepts a fenced answer, keeps well-formed reveals and drops malformed ones", () => {
    const text = '```json\n{"beats":[{"fromUserTurn":1001,"goal":" 봄. ","reveal":[{"fact":"민수는 X를 샀다","expected":"X","ask":"묻는다","replaces":" "},{"fact":"bad"}]},{"fromUserTurn":1016,"goal":"g"}]}\n```';
    const beats = parseBeatsJson(text);
    expect(beats).toEqual([
      { fromUserTurn: 1001, goal: "봄.", reveal: [{ fact: "민수는 X를 샀다", expected: "X", ask: "묻는다" }] },
      { fromUserTurn: 1016, goal: "g", reveal: [] },
    ]);
  });

  it("rejects output without a beats array", () => {
    expect(() => parseBeatsJson("{\"x\":1}")).toThrow(/no beats/);
    expect(() => parseBeatsJson("nothing here")).toThrow(/no JSON/);
  });
});
