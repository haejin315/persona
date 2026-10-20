import { ClaudeCliProvider } from "./claudeCli.js";
import { OpenAICompatibleProvider } from "./openaiCompatible.js";
import type { LLMProvider } from "./types.js";

export * from "./types.js";
export { OpenAICompatibleProvider } from "./openaiCompatible.js";
export { ClaudeCliProvider, foldMessages, parseResultJson } from "./claudeCli.js";

export type ProviderName = "claude-cli" | "openai" | "ollama";

/** Default model spec for the character: CHAT_MODEL env, else the Claude subscription via the CLI. */
export const DEFAULT_MODEL_SPEC = "claude-cli:sonnet";

/** Build a provider from environment variables. Throws a readable error when a key is missing. */
export function providerFromEnv(name: ProviderName = "claude-cli"): LLMProvider {
  switch (name) {
    case "claude-cli":
      return new ClaudeCliProvider();
    case "openai": {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) throw new Error("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.");
      return new OpenAICompatibleProvider({ id: "openai", apiKey, defaultModel: "gpt-5-mini" });
    }
    case "ollama":
      return new OpenAICompatibleProvider({
        id: "ollama",
        apiKey: "ollama", // Ollama ignores the key but the SDK requires one
        baseURL: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1",
        defaultModel: process.env.OLLAMA_MODEL ?? "qwen2.5:7b",
      });
    default:
      throw new Error(`unknown provider "${name}" (expected claude-cli | openai | ollama)`);
  }
}

/**
 * Parse "provider" or "provider:model" (e.g. "claude-cli:haiku", "openai:gpt-5-mini",
 * "ollama:qwen2.5:7b"). A bare model name with no known provider prefix is treated as OpenAI,
 * which keeps old CHAT_MODEL=gpt-5-mini style values working.
 */
export function parseModelSpec(spec: string): { provider: ProviderName; model?: string } {
  const known: ProviderName[] = ["claude-cli", "openai", "ollama"];
  for (const p of known) {
    if (spec === p) return { provider: p };
    if (spec.startsWith(`${p}:`)) return { provider: p, model: spec.slice(p.length + 1) };
  }
  return { provider: "openai", model: spec };
}

/** Provider + model for a spec string. */
export function providerFromSpec(spec: string = process.env.CHAT_MODEL ?? DEFAULT_MODEL_SPEC): { provider: LLMProvider; model?: string } {
  const { provider, model } = parseModelSpec(spec);
  return { provider: providerFromEnv(provider), model };
}
