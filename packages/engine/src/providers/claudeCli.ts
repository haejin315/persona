import { spawn } from "node:child_process";
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
 * Limits: one process per call (~1s start-up), no temperature/max-token control, and the
 * subscription's rolling usage window applies instead of a per-token bill.
 */
export class ClaudeCliProvider implements LLMProvider {
  readonly id = "claude-cli";
  readonly defaultModel: string;

  constructor(opts: { defaultModel?: string; bin?: string } = {}) {
    this.defaultModel = opts.defaultModel ?? "sonnet";
    this.bin = opts.bin ?? process.env.CLAUDE_CLI_BIN ?? "claude";
  }
  private bin: string;

  async generate(messages: ChatMessage[], opts?: GenerateOptions): Promise<GenerateResult> {
    const model = opts?.model ?? this.defaultModel;
    const { system, prompt } = foldMessages(messages);

    const args = [
      "-p",
      "--output-format",
      "json",
      "--model",
      model,
      "--tools",
      "",
      "--strict-mcp-config",
      "--setting-sources",
      "",
      "--exclude-dynamic-system-prompt-sections",
      "--no-session-persistence",
      "--max-turns",
      "1",
    ];
    if (system) args.push("--system-prompt", system);

    // Running inside a Claude Code session sets these; the nested CLI refuses to start with them.
    const env = { ...process.env };
    delete env.CLAUDECODE;
    delete env.CLAUDE_CODE_ENTRYPOINT;

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
    if (json.is_error) {
      const msg = String(json.result ?? "unknown error");
      const err = new Error(`claude-cli: ${msg.slice(0, 300)}`) as Error & { rateLimited?: boolean };
      err.rateLimited = /limit|rate|overloaded|429/i.test(msg);
      throw err;
    }

    const u = json.usage ?? {};
    const cached = Number(u.cache_read_input_tokens ?? 0);
    const fresh = Number(u.input_tokens ?? 0) + Number(u.cache_creation_input_tokens ?? 0);
    // Keep the "claude-cli:" prefix on the reported model so pricing knows the subscription paid.
    const canonical = Object.keys(json.modelUsage ?? {})[0] ?? model;
    return {
      text: String(json.result ?? ""),
      model: `claude-cli:${canonical}`,
      latencyMs: Date.now() - t0,
      usage: { promptTokens: fresh + cached, completionTokens: Number(u.output_tokens ?? 0), cachedPromptTokens: cached },
    };
  }

  /** The CLI returns whole messages; emulate streaming by yielding the finished text. */
  async *stream(messages: ChatMessage[], opts?: GenerateOptions): AsyncIterable<string> {
    const r = await this.generate(messages, opts);
    yield r.text;
    opts?.onDone?.({ model: r.model, usage: r.usage, latencyMs: r.latencyMs });
  }
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
