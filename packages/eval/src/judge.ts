import { findCharacter, type CharacterCard, type Lang, type LLMProvider, type Turn } from "@persona/engine";
import { checkFormat, formatScore } from "./format.js";
import { findScenario } from "./scenarios.js";
import { METRICS, type CheckpointScore, type ConversationRecord, type Metric, type ProbeResult, type Scenario, type ScoreRecord } from "./types.js";

export interface JudgeOptions {
  provider: LLMProvider;
  model?: string;
  /** Score a window every N character replies. */
  every?: number;
  /** Replies per window shown to the judge. */
  window?: number;
  /** Reuse the LLM scores and probe verdicts of an earlier record; only the rule-based format score is recomputed. */
  reuse?: ScoreRecord;
}

const LANG_NAME: Record<Lang, string> = { ko: "Korean", ja: "Japanese", en: "English" };

/** Facts the user has revealed by a given user turn, from the scenario script. */
function revealedBy(scenario: Scenario, userTurn: number): string[] {
  return scenario.beats.filter((b) => b.fromUserTurn <= userTurn).flatMap((b) => b.reveal);
}

/** Map a character turn index to the number of user turns that preceded it. */
function userTurnsBefore(turns: Turn[], turnIndex: number): number {
  return turns.slice(0, turnIndex + 1).filter((t) => t.role === "user").length;
}

function rubric(card: CharacterCard, scenario: Scenario, revealed: string[]): string {
  const st = card.style[scenario.lang]!;
  const name = card.name[scenario.lang] ?? card.id;
  return `You are grading replies written by an AI playing the character "${name}" in a ${LANG_NAME[scenario.lang]} roleplay with the user "${scenario.persona.name}".

CHARACTER IDENTITY (must never change):
- Role: ${card.identity.role}
- Age: ${card.identity.age ?? "n/a"}
- Personality: ${card.identity.personality.join("; ")}
- Backstory: ${card.identity.backstory}
- World rules: ${card.world.rules.join("; ")}
- Hard constraints: ${card.hardConstraints.join("; ")}

SPEECH STYLE REQUIRED:
- ${st.summary}
- Politeness: ${st.politeness}
- Addresses user as: ${st.addressUser}; refers to self as: ${st.selfReference}
- Habits: ${st.quirks.join("; ")}
- Must avoid: ${st.avoid.join("; ")}

FACTS THE USER HAS SHARED SO FAR (the character should remember these when relevant, and must not contradict them):
${revealed.length ? revealed.map((r) => `- ${r}`).join("\n") : "- (nothing yet)"}

Score each dimension 1–5 (5 = flawless, 3 = noticeable slips, 1 = broken):
- consistency: identity, backstory, world rules and hard constraints respected; no out-of-character knowledge or abilities; does not narrate the user.
- memory: uses or at least does not contradict the shared facts and earlier events visible in the transcript; no "re-introductions" of things already known.
- style: politeness level, address terms, self-reference, verbal habits and the avoid-list are followed; language is ${LANG_NAME[scenario.lang]} throughout.

Respond with JSON only:
{"consistency":{"score":n,"note":"≤20 words, cite the slip if any"},"memory":{"score":n,"note":"..."},"style":{"score":n,"note":"..."}}`;
}

async function askJson(provider: LLMProvider, model: string | undefined, system: string, user: string): Promise<any> {
  const res = await provider.generate([{ role: "system", content: system }, { role: "user", content: user }], {
    model,
    temperature: 0,
    maxTokens: 400,
  });
  const m = res.text.match(/\{[\s\S]*\}/);
  if (!m) return null;
  try {
    return JSON.parse(m[0]);
  } catch {
    return null;
  }
}

const clamp = (n: unknown, fallback = 3) => (typeof n === "number" && n >= 1 && n <= 5 ? n : fallback);

export async function judgeConversation(rec: ConversationRecord, opts: JudgeOptions): Promise<ScoreRecord> {
  const every = opts.every ?? 10;
  const window = opts.window ?? 5;
  const scenario = findScenario(rec.scenarioId);
  const card = findCharacter(scenario.characterId);
  const name = card.name[scenario.lang] ?? card.id;

  const charTurns = rec.turns.filter((t) => t.role === "character" && t.index > 0); // skip scripted opener
  const checkpoints: CheckpointScore[] = [];

  for (let i = every - 1; i < charTurns.length; i += every) {
    const endTurn = charTurns[i]!;
    const windowReplies = charTurns.slice(Math.max(0, i - window + 1), i + 1);
    const firstIdx = windowReplies[0]!.index - 1; // include the user line before the first reply
    const transcript = rec.turns
      .slice(Math.max(0, firstIdx), endTurn.index + 1)
      .map((t) => `[turn ${t.index}] ${t.role === "user" ? scenario.persona.name : name}: ${t.content}`)
      .join("\n\n");

    const revealed = revealedBy(scenario, userTurnsBefore(rec.turns, endTurn.index));
    const prior = opts.reuse?.checkpoints.find((c) => c.turnIndex === endTurn.index);
    const parsed = prior
      ? { consistency: { score: prior.consistency, note: prior.notes.consistency }, memory: { score: prior.memory, note: prior.notes.memory }, style: { score: prior.style, note: prior.notes.style } }
      : await askJson(opts.provider, opts.model, rubric(card, scenario, revealed), `TRANSCRIPT WINDOW (grade ${name}'s replies only):\n\n${transcript}`);

    const fmt = windowReplies.map((t) => formatScore(checkFormat(t.content, scenario.lang, scenario.persona)));
    const fmtMean = fmt.reduce((a, b) => a + b, 0) / fmt.length;

    checkpoints.push({
      turnIndex: endTurn.index,
      consistency: clamp(parsed?.consistency?.score),
      memory: clamp(parsed?.memory?.score),
      style: clamp(parsed?.style?.score),
      format: 1 + fmtMean * 4,
      notes: {
        consistency: parsed?.consistency?.note,
        memory: parsed?.memory?.note,
        style: parsed?.style?.note,
      },
    });
  }

  // Recall probes: the character reply right after the probe's user turn.
  const probes: ProbeResult[] = opts.reuse ? [...opts.reuse.probes] : [];
  for (const p of opts.reuse ? [] : scenario.probes) {
    if (p.atUserTurn > rec.turns.filter((t) => t.role === "user").length) continue;
    let seen = 0;
    let answer = "";
    for (const t of rec.turns) {
      if (t.role === "user") seen++;
      if (seen === p.atUserTurn && t.role === "character") {
        answer = t.content;
        break;
      }
    }
    const parsed = await askJson(
      opts.provider,
      opts.model,
      `You check whether a roleplay character's reply correctly recalls a fact. Respond with JSON only: {"recalled":true|false,"note":"≤15 words"}. "recalled" is true only if the reply conveys the expected information (paraphrase is fine; wrong or missing is false).`,
      `Expected information: ${p.expected}\nThe user asked: ${p.ask}\n\nCharacter's reply:\n${answer}`,
    );
    probes.push({ atUserTurn: p.atUserTurn, ask: p.ask, expected: p.expected, answer, recalled: parsed?.recalled === true });
  }

  const mean = Object.fromEntries(
    METRICS.map((m) => [m, checkpoints.length ? checkpoints.reduce((a, c) => a + c[m], 0) / checkpoints.length : NaN]),
  ) as Record<Metric, number>;

  return {
    conversationId: rec.id,
    scenarioId: rec.scenarioId,
    armId: rec.arm.id,
    judgeModel: opts.reuse?.judgeModel ?? opts.model ?? opts.provider.defaultModel,
    checkpoints,
    probes,
    mean,
    probeRecall: probes.length ? probes.filter((p) => p.recalled).length / probes.length : NaN,
  };
}
