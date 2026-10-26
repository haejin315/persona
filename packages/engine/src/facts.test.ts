import { describe, expect, it } from "vitest";
import { emptyMemory, normalizeMemory, renderFact, toFact, type MemoryState } from "./types.js";

describe("facts with provenance", () => {
  it("accepts both stored shapes", () => {
    expect(toFact("민수는 고양이 보리를 키운다")).toEqual({ text: "민수는 고양이 보리를 키운다", source: "unknown" });
    expect(toFact({ text: " a ", source: "told", by: " 이모 " })).toEqual({ text: "a", source: "told", by: "이모" });
    expect(toFact({ text: "a", source: "nonsense" })).toEqual({ text: "a", source: "unknown" });
    expect(toFact("")).toBeUndefined();
    expect(toFact({ source: "user" })).toBeUndefined();
    expect(toFact(42)).toBeUndefined();
  });

  it("normalises an old recording's string facts", () => {
    const old = { ...emptyMemory(), facts: ["x", "", "y"] as unknown as MemoryState["facts"] };
    expect(normalizeMemory(old).facts).toEqual([
      { text: "x", source: "unknown" },
      { text: "y", source: "unknown" },
    ]);
  });

  it("renders the source only when asked and only when known", () => {
    const f = { text: "민수는 성수동으로 이사했다", source: "user" as const };
    expect(renderFact(f, "민수", false)).toBe("민수는 성수동으로 이사했다");
    expect(renderFact(f, "민수", true)).toBe("민수는 성수동으로 이사했다 (민수 told you)");
    expect(renderFact({ text: "a", source: "self" }, "민수", true)).toBe("a (you said this yourself)");
    expect(renderFact({ text: "a", source: "told", by: "이모" }, "민수", true)).toBe("a (이모 told you)");
    expect(renderFact({ text: "a", source: "told" }, "민수", true)).toBe("a (someone told you)");
    expect(renderFact({ text: "a", source: "rumor" }, "민수", true)).toBe("a (heard around, not seen by you)");
    expect(renderFact({ text: "a", source: "unknown" }, "민수", true)).toBe("a");
  });
});
