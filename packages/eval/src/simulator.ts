import type { ChatMessage, LLMProvider, TokenUsage, Turn } from "@persona/engine";
import type { ConversationRecord, Scenario } from "./types.js";

const LANG_NAME = { ko: "Korean", ja: "Japanese", en: "English" } as const;

export interface UserMessage {
  text: string;
  model?: string;
  usage?: TokenUsage;
}

/** Anything that can produce the user's next message: an LLM simulator or a recorded script. */
export interface UserSource {
  next(userTurn: number, recent: Turn[], characterName: string): Promise<UserMessage>;
  /** Number of turns available, when finite (replay). */
  readonly length?: number;
}

/**
 * Plays the user. Given the scenario's current beat and the recent transcript, writes the next
 * user message in the *action* "speech" convention. On probe turns it asks the probe question
 * (worked into the scene, not as a quiz).
 */
export class UserSimulator implements UserSource {
  constructor(
    private provider: LLMProvider,
    private scenario: Scenario,
    private model?: string,
  ) {}

  currentBeat(userTurn: number) {
    const beats = this.scenario.beats;
    let active = beats[0]!;
    for (const b of beats) if (b.fromUserTurn <= userTurn) active = b;
    return active;
  }

  probeAt(userTurn: number) {
    return this.scenario.probes.find((p) => p.atUserTurn === userTurn);
  }

  async next(userTurn: number, recent: Turn[], characterName: string): Promise<UserMessage> {
    const beat = this.currentBeat(userTurn);
    const probe = this.probeAt(userTurn);
    const s = this.scenario;

    const transcript = recent
      .slice(-8)
      .map((t) => `${t.role === "user" ? s.persona.name : characterName}: ${t.content}`)
      .join("\n\n");

    const system = `You are roleplaying as ${s.persona.name}${s.persona.description ? `, ${s.persona.description}` : ""}, a human user chatting with the character ${characterName} in an interactive story.
Write ONLY ${s.persona.name}'s next message, in ${LANG_NAME[s.lang]}.

Writing style: ${s.userStyle}
Format: actions and narration between asterisks *...*, spoken words in double quotes "...". 1–3 short paragraphs. Never write ${characterName}'s actions or lines.

Current goal: ${beat.goal}
${beat.reveal.length ? `Work these facts about yourself into the conversation naturally if you haven't already (do not list them):\n${beat.reveal.map((r) => `- ${r}`).join("\n")}` : ""}
${probe ? `\nIMPORTANT for this message: ${probe.ask} Ask it in character, casually, without hinting at the answer. The question must be specific enough that the only correct answer is "${probe.expected}" (never say it yourself). You must have told ${characterName} this fact earlier; if you have not, state it plainly in this message instead of asking.` : ""}
Do not break character. Do not summarise. Do not add meta commentary.`;

    const messages: ChatMessage[] = [
      { role: "system", content: system },
      { role: "user", content: `Recent conversation:\n\n${transcript}\n\nWrite ${s.persona.name}'s next message.` },
    ];
    const res = await this.provider.generate(messages, { model: this.model, temperature: 0.9, maxTokens: 300 });
    return { text: res.text.trim(), model: res.model, usage: res.usage };
  }
}

/**
 * Replays the user side of a recorded conversation verbatim. Costs nothing and gives every arm
 * the same input, so differences between arms come from the character side only. The trade-off:
 * the user no longer reacts to what this character actually said.
 */
export class ScriptedUser implements UserSource {
  private script: string[];

  constructor(source: ConversationRecord | string[], public readonly sourceId?: string) {
    this.script = Array.isArray(source) ? source : source.turns.filter((t) => t.role === "user").map((t) => t.content);
    if (!Array.isArray(source)) this.sourceId = source.id;
  }

  get length() {
    return this.script.length;
  }

  async next(userTurn: number): Promise<UserMessage> {
    const text = this.script[userTurn - 1];
    if (text === undefined) throw new Error(`replay script has only ${this.script.length} user turns; asked for turn ${userTurn}`);
    return { text };
  }
}
