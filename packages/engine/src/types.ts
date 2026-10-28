import { z } from "zod";

/** Languages the engine can speak. Each character card carries a style block per language. */
export const Lang = z.enum(["ko", "ja", "en"]);
export type Lang = z.infer<typeof Lang>;

/**
 * Per-language speech style. This is the part that breaks first when a model is swapped
 * or a language is added, so it is isolated from the identity block on purpose.
 */
export const SpeechStyle = z.object({
  /** One-paragraph description of how the character talks in this language. */
  summary: z.string(),
  /** Politeness level: Korean 존댓말/반말, Japanese 敬語 level. Free text, read by the LLM. */
  politeness: z.string(),
  /** How the character addresses the user (호칭 / 呼び方). May contain {{user}}. */
  addressUser: z.string(),
  /** First-person pronoun / self-reference (나/저, 俺/僕/私 …). */
  selfReference: z.string(),
  /** Signature phrases, verbal tics, punctuation habits. */
  quirks: z.array(z.string()).default([]),
  /**
   * Literal opener beats this character uses ("...아,", "음,"), so the engine can tell when two
   * consecutive replies open the same way and drop the second one (filters.ts). Optional.
   */
  openers: z.array(z.string()).default([]),
  /** Things this character would never say in this language (tone guardrails). */
  avoid: z.array(z.string()).default([]),
  /** Few-shot example exchanges, written in the *action* "speech" format. */
  examples: z
    .array(z.object({ user: z.string(), character: z.string() }))
    .default([]),
});
export type SpeechStyle = z.infer<typeof SpeechStyle>;

export const CharacterCard = z.object({
  id: z.string(),
  name: z.record(Lang, z.string()),
  genre: z.enum(["romance_daily", "fantasy_adventure"]),
  tagline: z.record(Lang, z.string()).optional(),

  /** Who the character is — identity, backstory, personality. Language-neutral facts. */
  identity: z.object({
    age: z.string().optional(),
    role: z.string(),
    appearance: z.string().optional(),
    personality: z.array(z.string()),
    backstory: z.string(),
    values: z.array(z.string()).default([]),
    secrets: z.array(z.string()).default([]),
  }),

  /** Setting the story lives in. */
  world: z.object({
    summary: z.string(),
    rules: z.array(z.string()).default([]),
    locations: z.array(z.string()).default([]),
  }),

  /** Opening scene and the character's relationship to the user at turn 0. Text may use {{user}}. */
  scenario: z.object({
    relationshipToUser: z.string(),
    openingSituation: z.string(),
    firstMessage: z.record(Lang, z.string()),
    /** Hooks the character can pull on when the conversation stalls. */
    plotHooks: z.array(z.string()).default([]),
  }),

  /** Facts that must never drift, no matter how long the conversation runs. */
  hardConstraints: z.array(z.string()).default([]),

  /** Immersion-preserving safety boundaries, written in-character where possible. */
  boundaries: z.array(z.string()).default([]),

  style: z.record(Lang, SpeechStyle),
});
export type CharacterCard = z.infer<typeof CharacterCard>;

/**
 * The user's own character in the story. Mirrors what WHIF/Crack/Zeta call a "페르소나":
 * the model plays the character, the user plays this persona, and both narrate.
 */
export const UserPersona = z.object({
  name: z.string(),
  /** One or two sentences: who they are in this story. Optional. */
  description: z.string().optional(),
});
export type UserPersona = z.infer<typeof UserPersona>;

export const DEFAULT_PERSONA: UserPersona = { name: "유저" };

/** Token usage reported by a provider for one generation. */
export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  /** Prompt tokens served from the provider's prompt cache, when reported. */
  cachedPromptTokens?: number;
}

/** A single chat turn as stored in a session. */
export interface Turn {
  role: "user" | "character";
  content: string;
  /** Turn index, 0-based, counting both sides. */
  index: number;
  ts: number;
  /** Model and prompt version that produced a character turn (for eval attribution). */
  meta?: {
    model?: string;
    promptVersion?: string;
    latencyMs?: number;
    /** Streamed replies: time to the first visible character (docs/11). */
    firstTokenMs?: number;
    /** Streamed replies: the provider's start-up overlapped with the user's typing (docs/11). */
    warm?: boolean;
    usage?: TokenUsage;
    filters?: string[];
    /**
     * On a user turn: time passed before it ("The next day: day 2, morning, at 카페"). Rendered as a
     * bracketed line ahead of the message in the prompt and in the memory transcript, so the
     * conversation window does not read as one continuous sitting (docs/10 T2).
     */
    boundary?: string;
    /** Archive entries shown next to MEMORY for this reply, with their match score (docs/12). */
    recalled?: { text: string; score: number; why: "entity" | "vector" }[];
  };
}

/**
 * How the character knows a fact (docs/10, step 1). "user": the user said it to the character.
 * "self": the character said, promised or decided it (opening hours, promises) - the character must
 * stay consistent with these. "witnessed": it happened in the scene. "told": a third person told the
 * character (`by`). "rumor": heard without a source. "unknown": stored before sources existed.
 */
export type FactSource = "user" | "self" | "witnessed" | "told" | "rumor" | "unknown";
export const FACT_SOURCES: readonly FactSource[] = ["user", "self", "witnessed", "told", "rumor", "unknown"];

export interface Fact {
  text: string;
  source: FactSource;
  /** Who told the character, when `source` is "told". */
  by?: string;
  /**
   * World day the fact belongs to (facts that came from the ledger). The prompt renders it relative
   * to today ("yesterday", "3 days ago") at build time; a label baked into the text would go stale.
   */
  day?: number;
  /** The fact is about the user's things or a promise to the user (from an event's userEffects): the character brings it up unasked. */
  about?: "user";
}

/**
 * A fact as the summariser first stated it, kept for good in `MemoryState.archive` (docs/12). The
 * prompt's fact list is capped and merges entries; the archive is append-only and uncapped, so a
 * detail the list had to drop can still be found and shown when it fits the moment.
 */
export interface ArchivedFact extends Fact {
  /** Transcript turns the fact was extracted from. */
  turnFrom: number;
  turnTo: number;
  /** Proper nouns in the fact (people, pets, places, shops, titles): the exact index for recall. */
  entities?: string[];
}

/** Accept a stored fact in either shape (old recordings hold plain strings). */
export function toFact(f: unknown): Fact | undefined {
  if (typeof f === "string") return f.trim() ? { text: f.trim(), source: "unknown" } : undefined;
  if (f && typeof f === "object") {
    const o = f as Record<string, unknown>;
    if (typeof o.text !== "string" || !o.text.trim()) return undefined;
    const source = FACT_SOURCES.includes(o.source as FactSource) ? (o.source as FactSource) : "unknown";
    const by = typeof o.by === "string" && o.by.trim() ? o.by.trim() : undefined;
    const day = typeof o.day === "number" && Number.isFinite(o.day) ? Math.floor(o.day) : undefined;
    return { text: o.text.trim(), source, ...(by ? { by } : {}), ...(day !== undefined ? { day } : {}) };
  }
  return undefined;
}

export const factText = (f: Fact | string): string => (typeof f === "string" ? f : f.text);

/** Bring a stored memory (possibly from an old recording with string facts) to the current shape. */
export function normalizeMemory(m: MemoryState): MemoryState {
  return { ...m, facts: (m.facts as unknown[]).map(toFact).filter((f): f is Fact => !!f) };
}

/**
 * One prompt line per fact. With `provenance` the way the character knows it follows the fact, so the
 * prompt can ask the character to speak only from what reached them and to say how it did.
 */
export function renderFact(f: Fact, user: string, provenance: boolean, today?: number): string {
  const when = `${f.day === undefined ? "" : `[day ${f.day}${today === undefined ? "" : `, ${relativeDay(f.day, today)}`}] `}${f.about === "user" ? `[about ${user}'s things] ` : ""}`;
  if (!provenance || f.source === "unknown") return when + f.text;
  const how: Record<Exclude<FactSource, "unknown">, string> = {
    user: `${user} told you`,
    self: "you said this yourself",
    witnessed: "you saw this happen",
    told: f.by ? `${f.by} told you` : "someone told you",
    rumor: "heard around, not seen by you",
  };
  return `${when}${f.text} (${how[f.source]})`;
}

/** "today" / "yesterday" / "3 days ago" / "tomorrow" / "in 2 days". */
export function relativeDay(day: number, today: number): string {
  const diff = today - day;
  if (diff === 0) return "today";
  if (diff === 1) return "yesterday";
  if (diff > 1) return `${diff} days ago`;
  return diff === -1 ? "tomorrow" : `in ${-diff} days`;
}

/** Long-term memory distilled from the conversation. */
export interface MemoryState {
  /** Rolling summary of everything older than the short-term window. */
  summary: string;
  /** Discrete facts worth keeping verbatim, each with how the character knows it. */
  facts: Fact[];
  /** Relationship stage as the engine perceives it; drives tone progression. */
  relationship: { stage: string; notes: string[] };
  /** Turn index up to which the summary/facts are current. */
  coveredUpTo: number;
  /** Every fact ever extracted, uncapped (memory rules v4); absent on older recordings. */
  archive?: ArchivedFact[];
}

export const emptyMemory = (): MemoryState => ({
  summary: "",
  facts: [],
  relationship: { stage: "initial", notes: [] },
  coveredUpTo: -1,
});

/** Minimal provider-neutral message format. */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
  /**
   * "context" marks instructions that change every turn (memory, relationship stage, what was
   * just said). Providers keep them OUT of the cached static system prompt: the Claude CLI route
   * caches the system prompt only when it is byte-identical, so one changing line there costs
   * the whole cache (measured: v3-recent cached 0 tokens vs 3,449 for v3).
   */
  kind?: "context";
}

/** A turn as it appears in a transcript or prompt: the day boundary, if any, ahead of the text. */
export const turnText = (t: Turn): string => (t.meta?.boundary ? `[${t.meta.boundary}]\n${t.content}` : t.content);

/** Replace {{user}} placeholders with the persona's name. */
export function fillUser(text: string, persona: UserPersona): string {
  return text.replaceAll("{{user}}", persona.name);
}
