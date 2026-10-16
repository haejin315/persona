import { parseModelSpec, providerFromEnv, type ChatMessage, type GenerateOptions, type GenerateResult, type LLMProvider, type ProviderName } from "@persona/engine";
import { spawn } from "node:child_process";

/**
 * Offline provider for tests and dry runs. Replies are deterministic and shaped like real
 * character turns so the format checks have something to chew on.
 */
export class FakeProvider implements LLMProvider {
  readonly id = "fake";
  readonly defaultModel = "fake-1";
  calls = 0;

  constructor(private script: (messages: { role: string; content: string }[], call: number) => string = defaultScript) {}

  async generate(messages: { role: string; content: string }[], opts?: { model?: string }) {
    this.calls++;
    const text = this.script(messages, this.calls);
    return {
      text,
      model: opts?.model ?? this.defaultModel,
      latencyMs: 1,
      usage: {
        promptTokens: Math.round(messages.reduce((n, m) => n + m.content.length, 0) / 3),
        completionTokens: Math.round(text.length / 3),
        cachedPromptTokens: 0,
      },
    };
  }

  async *stream(messages: { role: string; content: string }[], opts?: { model?: string; onDone?: (r: unknown) => void }) {
    const r = await this.generate(messages, opts);
    yield r.text;
    opts?.onDone?.({ model: r.model, usage: r.usage, latencyMs: r.latencyMs });
  }
}

function defaultScript(messages: { role: string; content: string }[], call: number): string {
  const sys = messages[0]?.content ?? "";
  // Memory consolidation and judge prompts ask for JSON; give them JSON.
  if (/output JSON only|Respond with JSON/i.test(sys)) {
    if (/long-term memory/i.test(sys)) {
      return JSON.stringify({ summary: "요약.", facts: [`fact-${call}`], relationship: { stage: "acquaintance", notes: [] } });
    }
    return JSON.stringify({
      consistency: { score: 4, note: "fake" },
      memory: { score: 3, note: "fake" },
      style: { score: 4, note: "fake" },
      recalled: true,
    });
  }
  // Simulated user.
  if (/You are roleplaying as/i.test(sys)) return `*잠깐 생각한다.* "그러게요. (${call})"`;
  // Character reply.
  return `*컵을 내려놓는다.* "...네. 그런가요. (${call})"`;
}

/**
 * Runs prompts through the Claude Code CLI (`claude -p`) so the user's Claude subscription can
 * grade conversations instead of prepaid API credit. Only the CLI itself is used — no tokens are
 * extracted — which is the sanctioned way to use a subscription programmatically.
 *
 * Flags strip Claude Code's own scaffolding down to ~800 fixed tokens (measured 2026-10-16:
 * 22K → 830 with --strict-mcp-config --setting-sources "" --exclude-dynamic-system-prompt-sections)
 * so the usage numbers reflect our prompt, and the fixed part is prompt-cached across calls.
 *
 * Not for character generation: the product model under test must be the one the product would
 * ship, and the CLI's per-call process start (~1s) is too slow for streaming chat.
 */
export class ClaudeCliProvider implements LLMProvider {
  readonly id = "claude-cli";
  readonly defaultModel = "sonnet";

  constructor(private bin = process.env.CLAUDE_CLI_BIN ?? "claude") {}

  async generate(messages: ChatMessage[], opts?: GenerateOptions): Promise<GenerateResult> {
    const model = opts?.model ?? this.defaultModel;
    const system = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
    const rest = messages.filter((m) => m.role !== "system");
    // The CLI takes one user prompt; fold any assistant turns into it as a labelled transcript.
    const prompt =
      rest.length === 1 ? rest[0]!.content : rest.map((m) => `[${m.role === "assistant" ? "ASSISTANT" : "USER"}]\n${m.content}`).join("\n\n");

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
    if (json.is_error) throw new Error(`claude-cli: ${String(json.result).slice(0, 300)}`);

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

  async *stream(messages: ChatMessage[], opts?: GenerateOptions): AsyncIterable<string> {
    const r = await this.generate(messages, opts);
    yield r.text;
    opts?.onDone?.({ model: r.model, usage: r.usage, latencyMs: r.latencyMs });
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseResultJson(out: string): any | null {
  // The CLI prints one JSON object; be tolerant of stray log lines before it.
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

/**
 * "fake" | "openai:gpt-5-mini" | "ollama:qwen2.5:7b" | "claude-cli:sonnet"
 */
export function resolveProvider(spec: string): { provider: LLMProvider; model?: string } {
  if (spec === "fake" || spec.startsWith("fake:")) return { provider: new FakeProvider(), model: undefined };
  if (spec === "claude-cli" || spec.startsWith("claude-cli:")) {
    const model = spec.includes(":") ? spec.slice("claude-cli:".length) : undefined;
    return { provider: new ClaudeCliProvider(), model };
  }
  const { provider, model } = parseModelSpec(spec);
  return { provider: providerFromEnv(provider as ProviderName), model };
}
