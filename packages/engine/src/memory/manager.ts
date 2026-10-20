import type { LLMProvider } from "../providers/types.js";
import type { CharacterCard, Lang, MemoryState, TokenUsage, Turn } from "../types.js";

export interface MemoryConfig {
  /** Number of most recent turns kept verbatim in the prompt. */
  windowTurns: number;
  /** Consolidate memory once this many turns have fallen outside the window un-summarised. */
  consolidateEvery: number;
  /** Model used for summarisation / fact extraction (use a cheap one). */
  model?: string;
  /**
   * false = never consolidate: the character only sees the verbatim window. Used as the
   * "no long-term memory" arm in the memory experiment (set windowTurns high to get the
   * "whole transcript in context" arm instead).
   */
  enabled?: boolean;
}

export const DEFAULT_MEMORY_CONFIG: MemoryConfig = {
  windowTurns: 20,
  consolidateEvery: 10,
};

const LANG_NAME: Record<Lang, string> = { ko: "Korean", ja: "Japanese", en: "English" };

/**
 * Keeps the long-term memory current. Turns older than the window get folded into a rolling
 * summary + fact ledger, so the character can "remember" turn 3 at turn 3000 without the whole
 * transcript in context.
 */
export class MemoryManager {
  /** Usage of the most recent consolidation call, for cost accounting. */
  lastUsage?: TokenUsage & { model: string };

  constructor(
    private provider: LLMProvider,
    private cfg: MemoryConfig = DEFAULT_MEMORY_CONFIG,
  ) {}

  /** Turns that should be in the prompt verbatim. */
  recentWindow(turns: Turn[]): Turn[] {
    return turns.slice(-this.cfg.windowTurns);
  }

  /** True when enough un-summarised turns have accumulated outside the window. */
  needsConsolidation(turns: Turn[], memory: MemoryState): boolean {
    if (this.cfg.enabled === false) return false;
    const outsideWindow = turns.length - this.cfg.windowTurns;
    if (outsideWindow <= 0) return false;
    const pending = outsideWindow - (memory.coveredUpTo + 1);
    return pending >= this.cfg.consolidateEvery;
  }

  async consolidate(
    card: CharacterCard,
    lang: Lang,
    turns: Turn[],
    memory: MemoryState,
  ): Promise<MemoryState> {
    const outsideWindow = turns.length - this.cfg.windowTurns;
    const from = memory.coveredUpTo + 1;
    const toExclusive = Math.max(from, outsideWindow);
    const slice = turns.slice(from, toExclusive);
    if (slice.length === 0) return memory;

    const name = card.name[lang] ?? card.id;
    const transcript = slice
      .map((t) => `${t.role === "user" ? "User" : name}: ${t.content}`)
      .join("\n");

    const system = [
      `You maintain the long-term memory of a roleplay character named ${name}.`,
      "Given the EXISTING memory and NEW transcript lines, output JSON only:",
      "{",
      `  "summary": "updated running summary, 3-8 sentences, past tense, written in ${LANG_NAME[lang]}",`,
      '  "facts": ["short standalone facts worth remembering verbatim: names, promises, preferences, events, nicknames"],',
      '  "relationship": { "stage": "one of: initial | acquaintance | friendly | close | intimate | strained", "notes": ["how the relationship changed and why"] }',
      "}",
      "Keep facts from existing memory unless contradicted. Max 25 facts; merge duplicates.",
      `Attribute every fact to the right person and start it with their name (the user or ${name}). What the user tells ${name} about themselves is a fact about the USER, even when ${name} reacts to it.`,
    ].join("\n");

    const res = await this.provider.generate(
      [
        { role: "system", content: system },
        {
          role: "user",
          content: `EXISTING MEMORY:\n${JSON.stringify(memory, null, 2)}\n\nNEW TRANSCRIPT (turns ${from}-${toExclusive - 1}):\n${transcript}`,
        },
      ],
      { model: this.cfg.model, temperature: 0.2, maxTokens: 900 },
    );

    if (res.usage) this.lastUsage = { ...res.usage, model: res.model };
    const parsed = safeJson(res.text);
    if (!parsed) return { ...memory, coveredUpTo: toExclusive - 1 };

    const facts = Array.isArray(parsed.facts)
      ? parsed.facts.filter((f: unknown): f is string => typeof f === "string").slice(0, 25)
      : memory.facts;
    const notes = Array.isArray(parsed.relationship?.notes)
      ? parsed.relationship.notes.filter((n: unknown): n is string => typeof n === "string").slice(0, 8)
      : memory.relationship.notes;

    return {
      summary: typeof parsed.summary === "string" ? parsed.summary : memory.summary,
      facts,
      relationship: {
        stage: typeof parsed.relationship?.stage === "string" ? parsed.relationship.stage : memory.relationship.stage,
        notes,
      },
      coveredUpTo: toExclusive - 1,
    };
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function safeJson(text: string): any | null {
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) return null;
  try {
    return JSON.parse(m[0]);
  } catch {
    return null;
  }
}
