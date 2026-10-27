import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import type { ChatMessage } from "../types.js";
import type { GenerateOptions, GenerateResult, LLMProvider } from "./types.js";

/**
 * Runs prompts through the Claude Code CLI (`claude -p`) so a Claude subscription can power the
 * engine instead of prepaid API credit. Only the CLI itself is used — no tokens are extracted —
 * which is the sanctioned way to use a subscription programmatically.
 *
 * Flags strip Claude Code's own scaffolding down to ~800 fixed tokens (measured 2026-10-16:
 * 22K → 830 with --strict-mcp-config --setting-sources "" --exclude-dynamic-system-prompt-sections)
 * so usage numbers reflect our prompt, and the fixed part is prompt-cached across calls.
 *
 * Limits: one process per call (~3 s of start-up before the API call, measured 2026-10-27), no
 * temperature/max-token control, and the subscription's rolling usage window applies instead of
 * a per-token bill. `stream()` is real streaming (stream-json with partial messages): text deltas
 * are yielded as they arrive, thinking deltas are not, and the first text delta's time is reported.
 * `thinking: "off"` is honoured only by models that are not adaptive (Sonnet 5.5 and later always
 * think); a pinned claude-sonnet-4-6 answers without a thinking block (docs/11).
 */
export class ClaudeCliProvider implements LLMProvider {
  readonly id = "claude-cli";
  readonly defaultModel: string;

  constructor(opts: { defaultModel?: string; bin?: string; thinking?: "off" | "adaptive" } = {}) {
    this.defaultModel = opts.defaultModel ?? "sonnet";
    this.bin = opts.bin ?? process.env.CLAUDE_CLI_BIN ?? "claude";
    this.thinking = opts.thinking;
  }
  private bin: string;
  private thinking?: "off" | "adaptive";

  private args(model: string, system: string, format: "json" | "stream-json"): string[] {
    const args = ["-p", "--output-format", format, "--model", model, "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--exclude-dynamic-system-prompt-sections", "--no-session-persistence", "--max-turns", "1"];
    if (format === "stream-json") args.push("--verbose", "--include-partial-messages");
    if (system) args.push("--system-prompt", system);
    return args;
  }

  private env(thinking?: "off" | "adaptive"): NodeJS.ProcessEnv {
    // Running inside a Claude Code session sets these; the nested CLI refuses to start with them.
    const env = { ...process.env };
    delete env.CLAUDECODE;
    delete env.CLAUDE_CODE_ENTRYPOINT;
    if ((thinking ?? this.thinking) === "off") {
      // Documented to switch thinking off on non-adaptive models only (Sonnet 4.6, Haiku 4.5).
      env.CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING = "1";
      env.MAX_THINKING_TOKENS = "0";
    }
    return env;
  }

  async generate(messages: ChatMessage[], opts?: GenerateOptions): Promise<GenerateResult> {
    const model = opts?.model ?? this.defaultModel;
    const { system, prompt } = foldMessages(messages);
    const args = this.args(model, system, "json");
    const env = this.env(opts?.thinking);

    const t0 = Date.now();
    const out = await new Promise<string>((resolve, reject) => {
      const child = spawn(this.bin, args, { env, stdio: ["pipe", "pipe", "pipe"], signal: opts?.signal });
      let stdout = "";
      let stderr = "";
      child.stdout.setEncoding("utf8").on("data", (d) => (stdout += d));
      child.stderr.setEncoding("utf8").on("data", (d) => (stderr += d));
      child.on("error", reject);
      child.on("close", (code) => {
        if (code === 0) resolve(stdout);
        else reject(new Error(`claude exited ${code}: ${stderr || stdout}`.slice(0, 500)));
      });
      child.stdin.end(prompt, "utf8");
    });

    const json = parseResultJson(out);
    if (!json) throw new Error(`claude-cli: could not parse output: ${out.slice(0, 200)}`);
    if (json.is_error) throw cliError(json);
    return { ...resultOf(json, model), latencyMs: Date.now() - t0 };
  }

  /**
   * Real streaming: `--output-format stream-json --include-partial-messages`. Text deltas are
   * yielded as they arrive; thinking deltas are skipped; the result event carries usage. On a CLI
   * that sends no partial messages the whole text is yielded once at the end.
   */
  async *stream(messages: ChatMessage[], opts?: GenerateOptions): AsyncIterable<string> {
    const model = opts?.model ?? this.defaultModel;
    const { system, prompt } = foldMessages(messages);
    const t0 = Date.now();
    const child = spawn(this.bin, this.args(model, system, "stream-json"), { env: this.env(opts?.thinking), stdio: ["pipe", "pipe", "pipe"], signal: opts?.signal });
    let stderr = "";
    child.stderr.setEncoding("utf8").on("data", (d) => (stderr += d));
    const exit = new Promise<number | null>((resolve, reject) => {
      child.on("error", reject);
      child.on("close", resolve);
    });
    child.stdin.end(prompt, "utf8");

    const state = newStreamState();
    let firstTokenMs: number | undefined;
    for await (const line of createInterface({ input: child.stdout })) {
      const delta = parseStreamLine(line, state);
      if (!delta) continue;
      if (firstTokenMs === undefined) firstTokenMs = Date.now() - t0;
      yield delta;
    }
    const code = await exit;
    if (!state.result) throw new Error(`claude exited ${code}: ${(stderr || "no result event").slice(0, 500)}`);
    if (state.result.is_error) throw cliError(state.result);
    if (!state.text && typeof state.result.result === "string" && state.result.result) {
      firstTokenMs = Date.now() - t0;
      yield state.result.result;
    }
    const r = resultOf(state.result, model);
    opts?.onDone?.({ model: r.model, usage: r.usage, latencyMs: Date.now() - t0, firstTokenMs });
  }
}

export interface StreamState {
  /** content block index → type ("thinking" | "text" | ...). */
  blocks: Map<number, string>;
  /** Text yielded so far. */
  text: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  result?: any;
}
export const newStreamState = (): StreamState => ({ blocks: new Map(), text: "" });

/** One stream-json line → the text delta it carries; thinking deltas and every other event give "". */
export function parseStreamLine(line: string, state: StreamState): string {
  const s = line.trim();
  if (!s.startsWith("{")) return "";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let ev: any;
  try {
    ev = JSON.parse(s);
  } catch {
    return "";
  }
  if (ev.type === "result") {
    state.result = ev;
    return "";
  }
  if (ev.type !== "stream_event") return "";
  const e = ev.event ?? {};
  if (e.type === "content_block_start") {
    state.blocks.set(Number(e.index), String(e.content_block?.type ?? ""));
    return "";
  }
  if (e.type === "content_block_delta" && e.delta?.type === "text_delta" && state.blocks.get(Number(e.index)) === "text") {
    const t = String(e.delta.text ?? "");
    state.text += t;
    return t;
  }
  return "";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function resultOf(json: any, model: string): Omit<GenerateResult, "latencyMs"> {
  const u = json.usage ?? {};
  const cached = Number(u.cache_read_input_tokens ?? 0);
  const fresh = Number(u.input_tokens ?? 0) + Number(u.cache_creation_input_tokens ?? 0);
  // Keep the "claude-cli:" prefix on the reported model so pricing knows the subscription paid.
  const canonical = Object.keys(json.modelUsage ?? {})[0] ?? model;
  return {
    text: String(json.result ?? ""),
    model: `claude-cli:${canonical}`,
    usage: { promptTokens: fresh + cached, completionTokens: Number(u.output_tokens ?? 0), cachedPromptTokens: cached },
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function cliError(json: any): Error {
  const msg = String(json.result ?? "unknown error");
  const err = new Error(`claude-cli: ${msg.slice(0, 300)}`) as Error & { rateLimited?: boolean };
  err.rateLimited = /limit|rate|overloaded|429/i.test(msg);
  return err;
}

/** The CLI takes one system prompt and one user prompt; fold multi-turn history into a transcript. */
export function foldMessages(messages: ChatMessage[]): { system: string; prompt: string } {
  // Static instructions → --system-prompt (cached by the CLI when byte-identical across calls).
  // Per-turn context → head of the stdin prompt, so the system prompt never changes mid-conversation.
  const system = messages.filter((m) => m.role === "system" && m.kind !== "context").map((m) => m.content).join("\n\n");
  const context = messages.filter((m) => m.kind === "context").map((m) => m.content).join("\n\n");
  const rest = messages.filter((m) => m.role !== "system");
  const head = context ? `[CURRENT STATE - treat as true, do not quote]\n${context}\n\n` : "";
  if (rest.length === 1) return { system, prompt: head + rest[0]!.content };
  const prompt = [
    head + "The conversation so far is below. Write ONLY the next ASSISTANT message, with no label.",
    "",
    ...rest.map((m) => `[${m.role === "assistant" ? "ASSISTANT" : "USER"}]\n${m.content}`),
  ].join("\n\n");
  return { system, prompt };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function parseResultJson(out: string): any | null {
  const start = out.indexOf("{");
  if (start === -1) return null;
  try {
    return JSON.parse(out.slice(start));
  } catch {
    const m = out.match(/\{"type":"result"[\s\S]*\}/);
    if (!m) return null;
    try {
      return JSON.parse(m[0]);
    } catch {
      return null;
    }
  }
}
