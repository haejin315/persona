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
  meta?: { model?: string; promptVersion?: string; latencyMs?: number; usage?: TokenUsage };
}

/** Long-term memory distilled from the conversation. */
export interface MemoryState {
  /** Rolling summary of everything older than the short-term window. */
  summary: string;
  /** Discrete facts worth keeping verbatim: user's name, promises, shared events. */
  facts: string[];
  /** Relationship stage as the engine perceives it; drives tone progression. */
  relationship: { stage: string; notes: string[] };
  /** Turn index up to which the summary/facts are current. */
  coveredUpTo: number;
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
}

/** Replace {{user}} placeholders with the persona's name. */
export function fillUser(text: string, persona: UserPersona): string {
  return text.replaceAll("{{user}}", persona.name);
}
