import { describe, expect, it } from "vitest";
import { firstSpeechComplete, firstSpeechLine, openerOf, stripRepeatedOpener } from "./filters.js";

const OPENERS = ["...아,", "…아,", "음,"];
const prevWithOpener = '*컵을 내려놓는다.*\n\n"...아, 그건 몰랐어요."\n\n*행주를 접는다.*';
const prevWithout = '*컵을 내려놓는다.*\n\n"그건 몰랐어요."';

describe("firstSpeechLine / openerOf", () => {
  it("finds the first quoted line after actions", () => {
    expect(firstSpeechLine(prevWithOpener)?.line).toBe('"...아, 그건 몰랐어요."');
    expect(openerOf(firstSpeechLine(prevWithOpener)?.line, OPENERS)).toBe("...아,");
    expect(openerOf(firstSpeechLine(prevWithout)?.line, OPENERS)).toBeUndefined();
  });
  it("handles Japanese quotes", () => {
    expect(openerOf("「…あ、そうですか。」", ["…あ、", "んー、"])).toBe("…あ、");
  });
});

describe("stripRepeatedOpener", () => {
  it("drops the opener when the previous reply also opened with one", () => {
    const cur = '*냅킨을 꺼낸다.*\n\n"음, 들켰네요. 다음엔 안 옮길게요."\n\n"토요일에 말할게요."';
    const r = stripRepeatedOpener(cur, prevWithOpener, OPENERS);
    expect(r.stripped).toBe(true);
    expect(r.text).toBe('*냅킨을 꺼낸다.*\n\n"들켰네요. 다음엔 안 옮길게요."\n\n"토요일에 말할게요."');
  });
  it("leaves the reply alone when the previous one had no opener", () => {
    const cur = '"음, 들켰네요."';
    expect(stripRepeatedOpener(cur, prevWithout, OPENERS)).toEqual({ text: cur, stripped: false });
  });
  it("leaves the reply alone when there is no previous reply or no opener now", () => {
    expect(stripRepeatedOpener('"음, 네."', undefined, OPENERS).stripped).toBe(false);
    expect(stripRepeatedOpener('"네, 그럴게요."', prevWithOpener, OPENERS).stripped).toBe(false);
  });
  it("keeps a line that would become empty", () => {
    expect(stripRepeatedOpener('"음,"', prevWithOpener, OPENERS).stripped).toBe(false);
  });
  it("only touches the first spoken line", () => {
    const cur = '"음, 네."\n\n"음, 그리고요."';
    expect(stripRepeatedOpener(cur, prevWithOpener, OPENERS).text).toBe('"네."\n\n"음, 그리고요."');
  });
});

describe("firstSpeechComplete", () => {
  it("is true once a newline follows the first spoken line", () => {
    expect(firstSpeechComplete('*컵을 든다.*\n\n"...아, 그')).toBe(false);
    expect(firstSpeechComplete('*컵을 든다.*\n\n"...아, 그래요."\n')).toBe(true);
    expect(firstSpeechComplete("*컵을 든다.*\n\n")).toBe(false);
  });
});
