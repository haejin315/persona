import { providerFromSpec, type LLMProvider } from "@persona/engine";

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
 * "fake" | "openai:gpt-5-mini" | "ollama:qwen2.5:7b" | "claude-cli:sonnet"
 */
export function resolveProvider(spec: string): { provider: LLMProvider; model?: string } {
  if (spec === "fake" || spec.startsWith("fake:")) return { provider: new FakeProvider(), model: undefined };
  return providerFromSpec(spec);
}
