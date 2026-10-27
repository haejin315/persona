import type { ChatMessage, TokenUsage } from "../types.js";

export interface GenerateResult {
  text: string;
  model: string;
  usage?: TokenUsage;
  latencyMs: number;
  /** Streaming only: time to the first text delta (what the user waits before anything appears). */
  firstTokenMs?: number;
}

export interface GenerateOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  /** Abort signal for cancellation (e.g. user navigated away). */
  signal?: AbortSignal;
  /**
   * "off" asks the provider to answer without extended thinking (docs/11: a reply spends ~5 s
   * thinking before its first visible character). Providers that cannot honour it ignore it.
   */
  thinking?: "off" | "adaptive";
  /** Streaming only: called once after the last delta with model/usage/latency. */
  onDone?: (result: Omit<GenerateResult, "text">) => void;
}

/**
 * A provider turns messages into text. Every LLM — OpenAI, Ollama, Anthropic, a self-hosted
 * model — is wrapped behind this so the engine, eval harness and server never care which.
 */
export interface LLMProvider {
  readonly id: string;
  readonly defaultModel: string;
  generate(messages: ChatMessage[], opts?: GenerateOptions): Promise<GenerateResult>;
  stream(messages: ChatMessage[], opts?: GenerateOptions): AsyncIterable<string>;
}
