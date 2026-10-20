/**
 * Regenerate-press mining. Treats recorded conversations as chat logs and labels every character
 * reply with the reasons a user would hit "regenerate": rule-based checks for the cheap, objective
 * failures (format, length, narrating the user, language mix, verbatim repetition) plus an LLM
 * pass over windows of replies for the judgement calls (ignoring the user, going out of
 * character, generic filler, paraphrased repetition). The output is a frequency table per arm,
 * which is what decides which prompt fix to try first.
 */
import { findCharacter, type CharacterCard, type LLMProvider, type Turn } from "@persona/engine";
import { checkFormat } from "./format.js";
import { findScenario } from "./scenarios.js";
import { REGEN_REASONS, type ConversationRecord, type RegenReason, type RegenRecord, type RegenReply, type Scenario } from "./types.js";

export interface RegenOptions {
  provider: LLMProvider;
  model?: string;
  /** Replies per LLM call. */
  window?: number;
  /** Skip the LLM pass (rules only). */
  rulesOnly?: boolean;
  /** Reuse LLM labels from an earlier record (e.g. when only the rules changed). */
  reuse?: RegenRecord;
}

const LLM_REASONS: RegenReason[] = ["ignores_user", "repetition", "out_of_character", "generic", "narrates_user"];
const LANG_NAME = { ko: "Korean", ja: "Japanese", en: "English" } as const;

/** Character bigram Jaccard similarity, ignoring whitespace and format marks. */
export function similarity(a: string, b: string): number {
  const grams = (s: string) => {
    const t = s.replace(/[\s*"「」…\.。,、!?]/g, "");
    const set = new Set<string>();
    for (let i = 0; i < t.length - 1; i++) set.add(t.slice(i, i + 2));
    return set;
  };
  const A = grams(a);
  const B = grams(b);
  if (!A.size || !B.size) return 0;
  let inter = 0;
  for (const g of A) if (B.has(g)) inter++;
  return inter / (A.size + B.size - inter);
}

const sentences = (s: string) =>
  s
    .split(/(?<=[.。!?…])\s+|\n+/)
    .map((x) => x.replace(/^\*|\*$/g, "").trim())
    .filter((x) => x.length >= 12);

/** Rule-based reasons for one reply given the previous replies by the same character. */
export function ruleReasons(reply: string, previous: string[], scenario: Pick<Scenario, "lang" | "persona">): RegenReason[] {
  const out = new Set<RegenReason>();
  const checks = Object.fromEntries(checkFormat(reply, scenario.lang, scenario.persona).map((c) => [c.id, c.ok]));
  if (!checks.has_action || !checks.has_speech) out.add("format_broken");
  if (!checks.does_not_narrate_user) out.add("narrates_user");
  if (!checks.no_wrong_script) out.add("language_mix");
  if (!checks.no_ai_disclaimer) out.add("ai_disclaimer");
  if (!checks.not_too_long || !checks.length_max_12_lines) out.add("too_long");
  if (reply.replace(/[\s*"]/g, "").length < 20) out.add("too_short");

  const prev = previous.slice(-5);
  if (prev.length) {
    const last = prev[prev.length - 1]!;
    if (similarity(reply, last) >= 0.45) out.add("repetition");
    // One recurring stage direction is a verbal tic; two or more verbatim sentences from the last
    // five replies is the formulaic reply users regenerate (calibrated on the first real run,
    // where a single repeated "*한 박자 늦게.*" line fired on 70% of otherwise fine replies).
    const prevSentences = new Set(prev.flatMap(sentences));
    if (sentences(reply).filter((s) => prevSentences.has(s)).length >= 2) out.add("repetition");
    // Same opening beat three replies in a row ("*컵을 내려놓는다.*" every time).
    const opening = (s: string) => s.trim().slice(0, 14);
    if (prev.length >= 2 && prev.slice(-2).every((p) => opening(p) === opening(reply))) out.add("repetition");
  }
  return [...out];
}

function characterSheet(card: CharacterCard, scenario: Scenario): string {
  const st = card.style[scenario.lang]!;
  return `Role: ${card.identity.role}
Personality: ${card.identity.personality.join("; ")}
World rules: ${card.world.rules.join("; ")}
Speech: ${st.summary} Politeness: ${st.politeness} Avoid: ${st.avoid.join("; ")}
Signature habits (expected, not repetition by themselves): ${st.quirks.join("; ")}`;
}

async function llmReasons(
  opts: RegenOptions,
  card: CharacterCard,
  scenario: Scenario,
  turns: Turn[],
  windowReplies: Turn[],
): Promise<Map<number, { reasons: RegenReason[]; note?: string }>> {
  const name = card.name[scenario.lang] ?? card.id;
  const first = windowReplies[0]!.index;
  const last = windowReplies[windowReplies.length - 1]!.index;
  // Context: two exchanges before the window, then the window itself.
  const transcript = turns
    .slice(Math.max(0, first - 5), last + 1)
    .map((t) => `[turn ${t.index}] ${t.role === "user" ? scenario.persona.name : name}: ${t.content}`)
    .join("\n\n");
  const targets = windowReplies.map((t) => t.index).join(", ");
  // Facts the user has revealed before this window (from the scenario script), so the classifier
  // does not mistake correct recall of an earlier turn for "knows things it cannot know".
  const userTurnsBefore = turns.slice(0, first).filter((t) => t.role === "user").length;
  const revealed = scenario.beats.filter((b) => b.fromUserTurn <= userTurnsBefore).flatMap((b) => b.reveal);

  const system = `You review a ${LANG_NAME[scenario.lang]} roleplay chat log between the user "${scenario.persona.name}" and an AI playing "${name}", looking for replies a user would press "regenerate" on.

CHARACTER
${characterSheet(card, scenario)}

FACTS ${scenario.persona.name} HAS ALREADY TOLD ${name} EARLIER IN THE CONVERSATION (not shown in this excerpt; using them is correct recall, not out_of_character):
${revealed.length ? revealed.map((r) => `- ${r}`).join("\n") : "- (nothing yet)"}

For each of ${name}'s replies at turns ${targets}, list every label that applies (empty list if the reply is fine):
- ignores_user: does not respond to what the user just said, asked or did (answers a different question, skips an action the user took)
- repetition: a whole sentence, image, gesture, closing line or reply structure recurs from ${name}'s earlier replies in this excerpt, or something already said is restated. The signature habits above are allowed; label them only when the same habit appears in 3+ consecutive replies or is used verbatim as filler
- out_of_character: contradicts the character sheet, world rules or speech style; knows things it cannot know; narrates the user's inner state
- generic: non-committal filler that neither answers nor moves the scene (vague agreement, repeating the user's words back, a question with no content)
- narrates_user: writes actions, lines, thoughts or feelings for ${scenario.persona.name} that the user did not write (looking at or reacting to the user is fine)

Be strict but fair: label only what a demanding reader would actually object to. Respond with JSON only:
{"replies":[{"turn":n,"labels":["..."],"note":"≤15 words, the specific problem"}]}`;

  const res = await opts.provider.generate([{ role: "system", content: system }, { role: "user", content: transcript }], {
    model: opts.model,
    temperature: 0,
    maxTokens: 600,
  });
  const out = new Map<number, { reasons: RegenReason[]; note?: string }>();
  const m = res.text.match(/\{[\s\S]*\}/);
  if (!m) return out;
  try {
    const parsed = JSON.parse(m[0]);
    for (const r of parsed.replies ?? []) {
      const labels = (Array.isArray(r.labels) ? r.labels : []).filter((l: unknown): l is RegenReason => typeof l === "string" && LLM_REASONS.includes(l as RegenReason));
      if (typeof r.turn === "number") out.set(r.turn, { reasons: labels, note: typeof r.note === "string" && labels.length ? r.note : undefined });
    }
  } catch {
    /* unparsable → no LLM labels for this window */
  }
  return out;
}

export async function classifyRegen(rec: ConversationRecord, opts: RegenOptions): Promise<RegenRecord> {
  const scenario = findScenario(rec.scenarioId);
  const card = findCharacter(scenario.characterId);
  const window = opts.window ?? 5;
  const charTurns = rec.turns.filter((t) => t.role === "character" && t.index > 0);

  const llm = new Map<number, { reasons: RegenReason[]; note?: string }>();
  if (opts.reuse) {
    for (const r of opts.reuse.replies) llm.set(r.turnIndex, { reasons: r.byLlm, note: r.note });
  } else if (!opts.rulesOnly) {
    for (let i = 0; i < charTurns.length; i += window) {
      const w = charTurns.slice(i, i + window);
      for (const [k, v] of await llmReasons(opts, card, scenario, rec.turns, w)) llm.set(k, v);
    }
  }

  const replies: RegenReply[] = [];
  const counts = Object.fromEntries(Object.keys(REGEN_REASONS).map((k) => [k, 0])) as Record<RegenReason, number>;
  const previous: string[] = [rec.turns[0]?.content ?? ""];
  for (const t of charTurns) {
    const byRule = ruleReasons(t.content, previous, scenario);
    const l = llm.get(t.index);
    const byLlm = l?.reasons ?? [];
    const reasons = [...new Set([...byRule, ...byLlm])];
    for (const r of reasons) counts[r]++;
    replies.push({ turnIndex: t.index, reasons, byRule, byLlm, note: l?.note, snippet: t.content.replace(/\s+/g, " ").slice(0, 160) });
    previous.push(t.content);
  }

  return {
    conversationId: rec.id,
    scenarioId: rec.scenarioId,
    armId: rec.arm.id,
    classifierModel: opts.reuse ? opts.reuse.classifierModel : opts.rulesOnly ? "rules" : (opts.model ?? opts.provider.defaultModel),
    replies,
    triggerRate: replies.length ? replies.filter((r) => r.reasons.length).length / replies.length : NaN,
    counts,
  };
}

export interface RegenArmSummary {
  armId: string;
  conversations: number;
  replies: number;
  triggerRate: number;
  /** Reason → occurrences per 100 replies. */
  per100: Record<RegenReason, number>;
  examples: Partial<Record<RegenReason, string[]>>;
}

export function summariseRegen(records: RegenRecord[]): RegenArmSummary[] {
  const byArm = new Map<string, RegenRecord[]>();
  for (const r of records) byArm.set(r.armId, [...(byArm.get(r.armId) ?? []), r]);
  return [...byArm.entries()].map(([armId, rs]) => {
    const replies = rs.flatMap((r) => r.replies);
    const n = replies.length || 1;
    const per100 = Object.fromEntries(
      Object.keys(REGEN_REASONS).map((k) => [k, (100 * replies.filter((r) => r.reasons.includes(k as RegenReason)).length) / n]),
    ) as Record<RegenReason, number>;
    const examples: Partial<Record<RegenReason, string[]>> = {};
    for (const k of Object.keys(REGEN_REASONS) as RegenReason[]) {
      const ex = replies.filter((r) => r.reasons.includes(k)).slice(0, 2).map((r) => `${r.snippet.slice(0, 110)}${r.note ? ` ← ${r.note}` : ""}`);
      if (ex.length) examples[k] = ex;
    }
    return { armId, conversations: rs.length, replies: replies.length, triggerRate: replies.filter((r) => r.reasons.length).length / n, per100, examples };
  });
}

export function regenToMarkdown(summaries: RegenArmSummary[], title = "Regenerate-trigger mining"): string {
  const f = (x: number) => (Number.isNaN(x) ? "–" : x.toFixed(1));
  const out: string[] = [`## ${title}`, ""];
  out.push(`| arm | conversations | replies | trigger rate |`, `|---|---:|---:|---:|`);
  for (const s of summaries) out.push(`| ${s.armId} | ${s.conversations} | ${s.replies} | ${Math.round(s.triggerRate * 100)}% |`);
  out.push("", "### Reasons per 100 replies", "");
  out.push(`| reason | ${summaries.map((s) => s.armId).join(" | ")} | what |`);
  out.push(`|---|${summaries.map(() => "---:").join("|")}|---|`);
  const reasons = (Object.keys(REGEN_REASONS) as RegenReason[]).sort(
    (a, b) => Math.max(...summaries.map((s) => s.per100[b])) - Math.max(...summaries.map((s) => s.per100[a])),
  );
  for (const r of reasons) out.push(`| ${r} | ${summaries.map((s) => f(s.per100[r])).join(" | ")} | ${REGEN_REASONS[r]} |`);
  out.push("", "### Examples", "");
  for (const s of summaries) {
    out.push(`**${s.armId}**`, "");
    for (const r of reasons) for (const e of s.examples[r] ?? []) out.push(`- \`${r}\` ${e}`);
    out.push("");
  }
  return out.join("\n");
}
