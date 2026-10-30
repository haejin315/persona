import type { LLMProvider } from "../providers/types.js";
import { toFact, turnText, type ArchivedFact, type CharacterCard, type Fact, type Lang, type MemoryState, type TokenUsage, type Turn } from "../types.js";

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
  /**
   * Consolidation prompt rules. "v1" = attribution rule only. "v2" adds the proper-noun rule: in the
   * 200-turn v9 run the user said 성수동 three times and the fact list at turn 180 held "4층, 엘리베이터
   * 없음, 빵집 냄새" but not the name (docs/02); the summariser kept the details and dropped the noun.
   */
  rules?: "v1" | "v2" | "v3" | "v4" | "v5" | "v6";
  /**
   * Cap on the fact ledger (default 25). The 200-turn replays sit at the cap from turn ~50 on, and
   * under that pressure the summariser merges facts and loses names (docs/02 "엉뚱한 칸").
   * "v4" (docs/12) keeps the v3 list and adds the archive: every fact in the new transcript lines,
   * stated once with its proper nouns tagged, appended to `memory.archive` uncapped.
   */
  maxFacts?: number;
  /**
   * Character budget for the whole fact list (docs/13). The count cap alone did not bound the
   * prompt: at 1000 turns the 25 slots held their facts by growing into paragraphs (374 → 9,041
   * chars, +75% prompt tokens). With a budget the summariser orders by importance and the list is
   * cut where the budget ends; what is cut stays in the archive for recall. Absent = count cap only.
   */
  maxFactChars?: number;
}

export const DEFAULT_MEMORY_CONFIG: MemoryConfig = {
  windowTurns: 20,
  consolidateEvery: 10,
  maxFacts: 25,
  // v3 since 2026-10-22: proper nouns kept (v2) plus a source per fact (docs/10 step 1).
  // The 50-turn A/B kept recall, fact count and cost equal to v2.
  rules: "v3",
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
    const maxFacts = this.cfg.maxFacts ?? 25;
    const v6 = this.cfg.rules === "v6";
    const v5 = this.cfg.rules === "v5" || v6;
    const v3 = this.cfg.rules === "v3" || this.cfg.rules === "v4" || v5;
    const v4 = this.cfg.rules === "v4" || v5;
    const transcript = slice
      .map((t) => `${t.role === "user" ? "User" : name}: ${turnText(t)}`)
      .join("\n");

    const system = [
      `You maintain the long-term memory of a roleplay character named ${name}.`,
      "Given the EXISTING memory and NEW transcript lines, output JSON only:",
      "{",
      `  "summary": "updated running summary, 3-8 sentences, past tense, written in ${LANG_NAME[lang]}",`,
      v3
        ? `  "facts": [{"text": "short standalone fact worth remembering verbatim: names, promises, preferences, events, nicknames", "source": "user | self | witnessed | told | rumor", "by": "who told ${name} - only when source is told"}],`
        : '  "facts": ["short standalone facts worth remembering verbatim: names, promises, preferences, events, nicknames"],',
      '  "relationship": { "stage": "one of: initial | acquaintance | friendly | close | intimate | strained", "notes": ["how the relationship changed and why"] }',
      ...(v4
        ? [
            `  "new_facts": [{"text": "one concrete fact stated in the NEW TRANSCRIPT lines, in ${LANG_NAME[lang]}", "source": "user | self | witnessed | told | rumor", "by": "only when source is told", "entities": ["proper nouns in this fact: people, pets, places, shops, titles - verbatim"]${v5 ? `, "triple": {"subject": "one entity or concept", "relation": "short verb phrase", "object": "one entity or concept"${v6 ? ', "replaces": "true only when this changes what held before about the same subject and relation"' : ""}}` : ""}}]`,
          ]
        : []),
      "}",
      `Keep facts from existing memory unless contradicted. Max ${maxFacts} facts; merge duplicates.`,
      ...(this.cfg.maxFactChars
        ? [
            `The fact list is also capped at ${this.cfg.maxFactChars} characters in total and is cut there: one short line per fact, no paragraphs, most important first (names, promises, what is current now). What falls off the end is not lost - it stays in the archive.`,
          ]
        : []),
      `Attribute every fact to the right person and start it with their name (the user or ${name}). What the user tells ${name} about themselves is a fact about the USER, even when ${name} reacts to it.`,
      ...(this.cfg.rules === "v2" || v3
        ? [
            `Proper nouns are the memory. A place, person, pet, shop or title the user names is kept verbatim inside a fact, and merging facts never drops a name an earlier fact had. A fact about where someone lives or goes names the place; "moved to a 4th-floor flat" without the neighbourhood is a lost fact.`,
          ]
        : []),
      ...(v3
        ? [
            `"source" is how ${name} knows the fact. "user": the user said it to ${name}. "self": ${name} said, promised or decided it (opening hours, habits, promises) - keep these, ${name} must stay consistent with them. "witnessed": it happened in the scene in front of ${name}. "told": a third person told ${name} - put their name in "by". "rumor": heard with no source. A fact keeps the source it already has; repeating a fact does not make it better known.`,
          ]
        : []),
      ...(v4
        ? [
            `"new_facts" is the archive, separate from "facts": every concrete thing the NEW TRANSCRIPT lines state - a name, a number, a date, a title, a place, a preference, a promise, a change of plan - one fact per detail, attributed the same way, names verbatim. Do not merge it with existing memory and do not cap it; it keeps what the ${maxFacts}-fact list has to drop. Tag each with the proper nouns it contains.`,
          ]
        : []),
      ...(v5
        ? [
            `"triple" is the same fact as a knowledge-graph edge: "subject" and "object" are each ONE entity or concept - a person, pet, place, shop, drink, object, habit, date, job - as a single word or short noun phrase in ${LANG_NAME[lang]} (the user by name, ${name} by name; never a sentence), "relation" is a short verb phrase in ${LANG_NAME[lang]} (키운다, 다닌다, 마신다, 산다, 일한다, 좋아한다, 약속했다, 바꿨다). A node is something one could ask about later - a named person, pet, place, shop, brand, title, job, a specific date like 11월 3일 - never a time word (오늘, 내일, 어제, 이번 주, weekdays), weather, a quantity or a passing gesture; a fact about those has no triple. When something changes, the new fact's triple has the same subject and relation with the new object (민수 마신다 라떼 replacing 민수 마신다 아메리카노). Omit "triple" when the fact has no clear subject and object.`,
          ]
        : []),
      ...(v6
        ? [
            `"replaces": true on a triple means the old fact with the same subject and relation no longer holds (a new drink, a new home, a changed plan). Two facts that both stay true (민수 그렸다 만화 and 민수 그렸다 달빛 고양이; two different promises) are NOT a replacement - leave "replaces" out. This is a judgement about the story, not about the words. A node is also never a phrase built on a time word (토요일 컵, 금요일 마감, 내일 얘기 are not nodes - the node there is 컵, 마감, or none): what one could ask about later is the thing, not the day.`,
          ]
        : []),
    ].join("\n");
    // The archive is never shown to the summariser: it is uncapped, and the list it maintains is "facts".
    const { archive: _archive, ...ledger } = memory;

    const res = await this.provider.generate(
      [
        { role: "system", content: system },
        {
          role: "user",
          // v1/v2 keep the string shape they were measured with; v3 shows the stored sources.
          content: `EXISTING MEMORY:\n${JSON.stringify(v3 ? ledger : { ...ledger, facts: ledger.facts.map((f) => f.text) }, null, 2)}\n\nNEW TRANSCRIPT (turns ${from}-${toExclusive - 1}):\n${transcript}`,
        },
      ],
      { model: this.cfg.model, temperature: 0.2, maxTokens: v4 ? 1800 : 900 },
    );

    if (res.usage) this.lastUsage = { ...res.usage, model: res.model };
    const parsed = safeJson(res.text);
    if (!parsed) return { ...memory, coveredUpTo: toExclusive - 1 };

    let facts: Fact[] = Array.isArray(parsed.facts)
      ? (parsed.facts as unknown[]).map(toFact).filter((f): f is Fact => !!f).slice(0, maxFacts)
      : memory.facts;
    if (this.cfg.maxFactChars) facts = cutToBudget(facts, this.cfg.maxFactChars);
    const known = new Set((memory.archive ?? []).map((f) => f.text));
    const fresh: ArchivedFact[] = v4 && Array.isArray(parsed.new_facts)
      ? (parsed.new_facts as unknown[]).flatMap((raw) => {
          const f = toFact(raw);
          if (!f || known.has(f.text)) return [];
          known.add(f.text);
          const ents = raw && typeof raw === "object" && Array.isArray((raw as { entities?: unknown }).entities)
            ? ((raw as { entities: unknown[] }).entities.filter((e): e is string => typeof e === "string" && e.trim().length >= 2).map((e) => e.trim()).slice(0, 8))
            : [];
          const t = raw && typeof raw === "object" ? (raw as { triple?: unknown }).triple : undefined;
          const triple =
            t && typeof t === "object" && ["subject", "relation", "object"].every((k) => typeof (t as Record<string, unknown>)[k] === "string" && ((t as Record<string, string>)[k] ?? "").trim())
              ? {
                  subject: (t as Record<string, string>).subject!.trim(),
                  relation: (t as Record<string, string>).relation!.trim(),
                  object: (t as Record<string, string>).object!.trim(),
                  ...((t as Record<string, unknown>).replaces === true ? { replaces: true } : {}),
                }
              : undefined;
          return [{ ...f, turnFrom: from, turnTo: toExclusive - 1, ...(ents.length ? { entities: ents } : {}), ...(triple ? { triple } : {}) }];
        })
      : [];
    const archive = v4 || memory.archive ? [...(memory.archive ?? []), ...fresh] : undefined;
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
      ...(archive ? { archive } : {}),
    };
  }
}

/** Keep facts in the summariser's order while their total length fits the budget (docs/13). */
export function cutToBudget(facts: Fact[], budget: number): Fact[] {
  const out: Fact[] = [];
  let used = 0;
  for (const f of facts) {
    if (used + f.text.length > budget) break;
    out.push(f);
    used += f.text.length;
  }
  return out;
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
