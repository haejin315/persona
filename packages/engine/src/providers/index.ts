import { OpenAICompatibleProvider } from "./openaiCompatible.js";
import type { LLMProvider } from "./types.js";

export * from "./types.js";
export { OpenAICompatibleProvider } from "./openaiCompatible.js";

export type ProviderName = "openai" | "ollama";

/** Build a provider from environment variables. Throws a readable error when a key is missing. */
export function providerFromEnv(name: ProviderName = "openai"): LLMProvider {
  switch (name) {
    case "openai": {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) throw new Error("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.");
      return new OpenAICompatibleProvider({
        id: "openai",
        apiKey,
        defaultModel: process.env.CHAT_MODEL ?? "gpt-5-mini",
      });
    }
    case "ollama":
      return new OpenAICompatibleProvider({
        id: "ollama",
        apiKey: "ollama", // Ollama ignores the key but the SDK requires one
        baseURL: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1",
        defaultModel: process.env.OLLAMA_MODEL ?? "qwen2.5:7b",
      });
  }
}

/** Parse "provider" or "provider:model" (e.g. "openai:gpt-4o", "ollama:qwen2.5:7b"). */
export function parseModelSpec(spec: string): { provider: ProviderName; model?: string } {
  const i = spec.indexOf(":");
  if (i === -1) return { provider: spec as ProviderName };
  return { provider: spec.slice(0, i) as ProviderName, model: spec.slice(i + 1) };
}
