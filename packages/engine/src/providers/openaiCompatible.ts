import OpenAI from "openai";
import type { ChatMessage, TokenUsage } from "../types.js";
import type { GenerateOptions, GenerateResult, LLMProvider } from "./types.js";

export interface OpenAICompatibleConfig {
  id: string;
  apiKey: string;
  baseURL?: string;
  defaultModel: string;
}

type OpenAIUsage = OpenAI.CompletionUsage | null | undefined;

function toUsage(u: OpenAIUsage): TokenUsage | undefined {
  if (!u) return undefined;
  return {
    promptTokens: u.prompt_tokens,
    completionTokens: u.completion_tokens,
    cachedPromptTokens: u.prompt_tokens_details?.cached_tokens,
  };
}

/**
 * Works for OpenAI itself and for anything speaking the OpenAI chat-completions dialect:
 * Ollama (`http://localhost:11434/v1`), vLLM, LM Studio, OpenRouter, etc.
 */
export class OpenAICompatibleProvider implements LLMProvider {
  readonly id: string;
  readonly defaultModel: string;
  private client: OpenAI;

  constructor(cfg: OpenAICompatibleConfig) {
    this.id = cfg.id;
    this.defaultModel = cfg.defaultModel;
    this.client = new OpenAI({ apiKey: cfg.apiKey, baseURL: cfg.baseURL });
  }

  async generate(messagesIn: ChatMessage[], opts: GenerateOptions = {}): Promise<GenerateResult> {
    const messages = messagesIn.map(({ role, content }) => ({ role, content }));
    const model = opts.model ?? this.defaultModel;
    const t0 = performance.now();
    const res = await this.client.chat.completions.create(
      {
        model,
        messages,
        temperature: opts.temperature ?? 0.9,
        max_completion_tokens: opts.maxTokens ?? 600,
      },
      { signal: opts.signal },
    );
    return {
      text: res.choices[0]?.message?.content ?? "",
      model: res.model ?? model,
      usage: toUsage(res.usage),
      latencyMs: Math.round(performance.now() - t0),
    };
  }

  async *stream(messagesIn: ChatMessage[], opts: GenerateOptions = {}): AsyncIterable<string> {
    const messages = messagesIn.map(({ role, content }) => ({ role, content }));
    const model = opts.model ?? this.defaultModel;
    const t0 = performance.now();
    let usage: TokenUsage | undefined;
    let seenModel = model;
    const stream = await this.client.chat.completions.create(
      {
        model,
        messages,
        temperature: opts.temperature ?? 0.9,
        max_completion_tokens: opts.maxTokens ?? 600,
        stream: true,
        stream_options: { include_usage: true },
      },
      { signal: opts.signal },
    );
    for await (const chunk of stream) {
      if (chunk.model) seenModel = chunk.model;
      if (chunk.usage) usage = toUsage(chunk.usage);
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) yield delta;
    }
    opts.onDone?.({ model: seenModel, usage, latencyMs: Math.round(performance.now() - t0) });
  }
}
