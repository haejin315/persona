import type { TokenUsage } from "@persona/engine";

/**
 * List prices in USD per 1M tokens (input / cached input / output). Used for budget caps and the
 * `estimate` command; the unit-economics script has the same table with sources.
 * `billed: false` marks routes that do not draw on prepaid credit (e.g. the Claude Code
 * subscription CLI) — their cost is still *computed* at list price so the report can show what the
 * same calls would have cost on the API.
 */
export interface Price {
  in: number;
  cachedIn: number;
  out: number;
  billed?: boolean;
}

export const PRICES: Record<string, Price> = {
  "gpt-5-nano": { in: 0.05, cachedIn: 0.005, out: 0.4 },
  "gpt-5-mini": { in: 0.25, cachedIn: 0.025, out: 2.0 },
  "gpt-4.1-mini": { in: 0.4, cachedIn: 0.1, out: 1.6 },
  "gpt-5": { in: 1.25, cachedIn: 0.125, out: 10 },
  "gpt-5.4": { in: 2.5, cachedIn: 0.25, out: 15 },
  "gpt-5.5": { in: 5, cachedIn: 0.5, out: 30 },
  "claude-haiku-5-5": { in: 0.1, cachedIn: 0.01, out: 0.5 },
  "claude-sonnet-5-5": { in: 2.0, cachedIn: 0.2, out: 10 },
  "claude-opus-5-5": { in: 4.0, cachedIn: 0.2, out: 20 },
  // Claude Code CLI aliases (the subscription pays — priceFor marks any "claude-cli:" model unbilled).
  haiku: { in: 0.1, cachedIn: 0.01, out: 0.5 },
  sonnet: { in: 2.0, cachedIn: 0.2, out: 10 },
  opus: { in: 4.0, cachedIn: 0.2, out: 20 },
  // Local / offline — free.
  fake: { in: 0, cachedIn: 0, out: 0, billed: false },
  ollama: { in: 0, cachedIn: 0, out: 0, billed: false },
};

/** Find a price by model name, tolerating provider prefixes and dated snapshots. */
export function priceFor(model: string): Price | undefined {
  // Anything that went through the Claude Code CLI is paid by the subscription, not credit.
  if (model.startsWith("claude-cli")) {
    const inner = model.includes(":") ? model.slice(model.indexOf(":") + 1) : "sonnet";
    const p = priceFor(inner) ?? PRICES.sonnet!;
    return { ...p, billed: false };
  }
  const name = model.replace(/^openai:/, "");
  if (PRICES[name]) return PRICES[name];
  if (name.startsWith("fake")) return PRICES.fake;
  if (name.startsWith("ollama")) return PRICES.ollama;
  // Longest matching prefix: "gpt-5-mini-2025-08-07" → "gpt-5-mini", not "gpt-5".
  const keys = Object.keys(PRICES)
    .filter((k) => name.startsWith(k))
    .sort((a, b) => b.length - a.length);
  return keys[0] ? PRICES[keys[0]] : undefined;
}

export interface Cost {
  /** List-price cost of the calls. */
  usd: number;
  /** Portion that actually draws on prepaid credit. */
  billedUsd: number;
  /** True when the model is unknown and cost was assumed 0. */
  unknownModel: boolean;
}

export function costOf(model: string, usage: TokenUsage): Cost {
  const p = priceFor(model);
  if (!p) return { usd: 0, billedUsd: 0, unknownModel: true };
  const cached = usage.cachedPromptTokens ?? 0;
  const fresh = Math.max(0, usage.promptTokens - cached);
  const usd = (fresh * p.in + cached * p.cachedIn + usage.completionTokens * p.out) / 1e6;
  return { usd, billedUsd: p.billed === false ? 0 : usd, unknownModel: false };
}

export const fmtUsd = (x: number) => `$${x.toFixed(x < 0.1 ? 4 : 2)}`;
