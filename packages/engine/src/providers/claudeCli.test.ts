import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { newStreamState, parseStreamLine, systemPromptArgs } from "./claudeCli.js";

describe("claude-cli system prompt argument (docs/12)", () => {
  it("passes a short prompt on the command line and a long one through a temp file that cleanup removes", () => {
    expect(systemPromptArgs("")).toMatchObject({ args: [] });
    expect(systemPromptArgs("짧다", 10).args).toEqual(["--system-prompt", "짧다"]);
    const long = "민수는 보리를 키운다. ".repeat(10);
    const { args, cleanup } = systemPromptArgs(long, 50);
    expect(args[0]).toBe("--system-prompt-file");
    expect(readFileSync(args[1]!, "utf8")).toBe(long);
    cleanup();
    expect(existsSync(args[1]!)).toBe(false);
    cleanup(); // idempotent
  });
});

describe("claude-cli stream-json parsing (docs/11)", () => {
  it("yields text deltas only, skips thinking, and keeps the result event", () => {
    const st = newStreamState();
    const lines = [
      '{"type":"system","subtype":"init"}',
      '{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}}',
      '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me think"}}}',
      '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"NOT TEXT BLOCK"}}}',
      '{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"text"}}}',
      '{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"*컵을 내려놓는다.* "}}}',
      "not json at all",
      '{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"\\"네.\\""}}}',
      '{"type":"result","subtype":"success","is_error":false,"result":"*컵을 내려놓는다.* \\"네.\\"","usage":{"input_tokens":2,"cache_read_input_tokens":8000,"output_tokens":150},"modelUsage":{"claude-sonnet-4-6":{}}}',
    ];
    const out = lines.map((l) => parseStreamLine(l, st)).filter(Boolean);
    expect(out).toEqual(["*컵을 내려놓는다.* ", '"네."']);
    expect(st.text).toBe('*컵을 내려놓는다.* "네."');
    expect(st.result?.usage?.output_tokens).toBe(150);
    expect(Object.keys(st.result?.modelUsage)[0]).toBe("claude-sonnet-4-6");
  });
});
