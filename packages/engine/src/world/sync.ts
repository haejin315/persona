import type { LLMProvider } from "../providers/types.js";
import type { ChatSession } from "../session.js";
import type { Fact, TokenUsage } from "../types.js";
import type { World } from "./world.js";

/**
 * Fold what the character came to know off-screen into their memory, as facts carrying the
 * ledger's source, and set the `# NOW` block. Call it before the user speaks to the character on
 * a new day. Returns the facts that were added.
 */
export function syncCharacter(session: ChatSession, world: World, characterId: string): Fact[] {
  const events = world.unsynced(characterId);
  const have = new Set(session.memory.facts.map((f) => f.text));
  const fresh = world.factsFor(characterId, events).filter((f) => !have.has(f.text));
  if (fresh.length) session.memory = { ...session.memory, facts: [...session.memory.facts, ...fresh] };
  world.markSynced(characterId);
  const block = world.blockFor(characterId);
  session.world = block;
  if (block.daysSinceUser > 0 && session.turns.length) {
    session.markBoundary(`${block.daysSinceUser === 1 ? "The next day" : `${block.daysSinceUser} days later`}: day ${block.day}, ${block.slot}, at ${block.place}`);
  }
  return fresh;
}

export interface CompactResult {
  folded: number;
  usage?: TokenUsage & { model: string };
}

/**
 * A long gap leaves more dated facts than the prompt should carry (docs/10 T6). Facts older than
 * `keepDays` are folded into the running summary in one call when the ledger facts exceed `maxFacts`;
 * the newest stay verbatim. What is folded is still true, just no longer itemised.
 */
export async function compactWorldFacts(
  session: ChatSession,
  world: World,
  llm: { provider: LLMProvider; model?: string },
  opts: { maxFacts?: number; keepDays?: number } = {},
): Promise<CompactResult> {
  const maxFacts = opts.maxFacts ?? 25;
  const keepDays = opts.keepDays ?? 7;
  const dated = session.memory.facts.filter((f) => f.day !== undefined);
  if (dated.length <= maxFacts) return { folded: 0 };
  const old = dated.filter((f) => world.day - (f.day ?? world.day) > keepDays);
  if (!old.length) return { folded: 0 };
  const lines = old.map((f) => `- day ${f.day} (${world.weekdayOf(f.day!)}): ${f.text} [${f.source}${f.by ? `:${f.by}` : ""}]`).join("\n");
  const res = await llm.provider.generate(
    [
      {
        role: "system",
        content: `You compress a character's memory. Rewrite the dated facts below as 2-5 sentences of past-tense summary in the same language as the facts, keeping every proper noun, number, weekday and who-told-whom, dropping nothing that would matter a month later. Date each thing exactly as "day N (Weekday)" with the Latin word day and the day number (the character's clock counts days the same way: today is day ${world.day}); never a bare weekday and never a calendar-style "N일", which read as a recent date months later. Output the summary text only.`,
      },
      { role: "user", content: `EXISTING SUMMARY:\n${session.memory.summary || "(none)"}\n\nFACTS TO FOLD IN:\n${lines}` },
    ],
    { model: llm.model, temperature: 0.2, maxTokens: 500 },
  );
  let text = res.text.trim();
  if (!text) return { folded: 0 };
  // The fold is labelled with its day range, so the paragraph reads as "long ago" next to NOW's "Day N".
  const days = old.map((f) => f.day!);
  text = `[days ${Math.min(...days)}-${Math.max(...days)}] ${text}`;
  const oldSet = new Set(old);
  session.memory = {
    ...session.memory,
    summary: session.memory.summary ? `${session.memory.summary}\n${text}` : text,
    facts: session.memory.facts.filter((f) => !oldSet.has(f)),
  };
  return { folded: old.length, usage: res.usage ? { ...res.usage, model: res.model } : undefined };
}
