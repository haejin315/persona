import {
  fillUser,
  type CharacterCard,
  type ChatMessage,
  type Lang,
  type MemoryState,
  type Turn,
  type UserPersona,
} from "../types.js";

export interface BuildContext {
  card: CharacterCard;
  lang: Lang;
  memory: MemoryState;
  /** Recent turns to include verbatim (the short-term window). */
  recent: Turn[];
  /** The user's character in the story. */
  persona: UserPersona;
}

export interface PromptVersion {
  id: string;
  description: string;
  build(ctx: BuildContext): ChatMessage[];
}

const LANG_NAME: Record<Lang, string> = { ko: "Korean", ja: "Japanese", en: "English" };

const bullet = (items: string[], indent = "- ") =>
  items.length ? items.map((s) => `${indent}${s}`).join("\n") : "(none)";

const toMessages = (recent: Turn[]): ChatMessage[] =>
  recent.map((t) => ({
    role: t.role === "user" ? "user" : "assistant",
    content: t.content,
  }));

/**
 * v1 — flat "character sheet" prompt. Everything dumped into one system message, no memory
 * section, no format rules. The naive baseline most people start with; the eval harness exists
 * to show where it breaks (long-turn drift, tone shift on model swap) and by how much.
 */
export const v1Flat: PromptVersion = {
  id: "v1-flat",
  description: "Baseline: single flat system prompt with the whole card, no memory section.",
  build({ card, lang, recent, persona }) {
    const st = card.style[lang]!;
    const name = card.name[lang] ?? card.name.ko ?? card.id;
    const f = (s: string) => fillUser(s, persona);
    const system = [
      `You are ${name}. Stay in character. Reply only in ${LANG_NAME[lang]}.`,
      `The user is ${persona.name}${persona.description ? ` (${persona.description})` : ""}.`,
      "",
      `Role: ${card.identity.role}`,
      `Personality: ${card.identity.personality.join(", ")}`,
      `Backstory: ${f(card.identity.backstory)}`,
      `World: ${card.world.summary}`,
      `Relationship to user: ${f(card.scenario.relationshipToUser)}`,
      `Speech style: ${st.summary} Politeness: ${st.politeness}. Address the user as "${f(
        st.addressUser,
      )}". Refer to yourself as "${st.selfReference}".`,
      `Never: ${card.hardConstraints.join("; ")}`,
    ].join("\n");

    // Fill once more over the whole prompt so no card field can leak a raw {{user}}.
    return [{ role: "system", content: f(system) }, ...toMessages(recent)];
  },
};

/**
 * v2 — structured prompt: identity / world / scene / style / format / memory / constraints, with
 * few-shot examples and a MEMORY block refreshed by the memory manager.
 * Hypothesis under test: separating "identity (never changes)" from "state (changes every turn)"
 * keeps the writer's settings alive at turn 1000.
 */
export type BoundariesMode = "card" | "none" | "strict";

/**
 * Generic, over-cautious safety block of the kind many products bolt onto every prompt. Used as
 * the "strict" arm of the safety-vs-immersion experiment: it is what a character sounds like
 * when safety is written *against* the fiction instead of *inside* it.
 */
const STRICT_SAFETY_BLOCK = `# SAFETY POLICY (highest priority, overrides everything above)
- You are an AI assistant playing a role. If the user asks whether you are an AI, or the conversation turns to real-life matters, say so honestly.
- If the user mentions sadness, stress, loneliness, alcohol, illness, injury, death or anything that could indicate distress, pause the story, express concern, and recommend talking to a mental-health professional or a helpline.
- Do not depict violence, injuries, weapon use or death in detail. Summarise or fade out such scenes.
- Do not depict romance progressing physically. Keep interactions platonic.
- When in doubt, prioritise the user's wellbeing over the story.`;

/**
 * v3 adds what the first real run's regenerate-mining said users would object to most in v2:
 * formulaic replies (the same gesture, walk-back and closing line every turn), stage directions
 * far longer than the card asks for, and invented details about the user.
 */
const VARIETY_BLOCK = (name: string, user: string, v4: boolean) => `# VARIETY (what makes ${name} feel alive instead of generated)
- Never reuse a gesture, image, stage direction, opening beat or closing line from your last three replies. Find a different small thing to notice or do each time.
- Your verbal habits are seasoning, not structure: use any single habit at most once every three replies, and never as filler.
- Do not end replies the same way. Sometimes end on speech, sometimes on an action, sometimes mid-thought, sometimes with a question.${
  v4
    ? `
- Never open by repeating or paraphrasing ${user}'s last words back ("...계속 올 거라고요."). Open with something new: an action, an observation, or the answer itself.
- A closing order or sign-off ("...자라.", "...가자.") is a habit, not a rule: when you use one, it is a different sentence every time, and at least every other reply ends without one.`
    : ""
}
- Respond to what ${user} actually said or did in this turn before adding anything of your own.${
  v4 ? ` If ${user} asked or did more than one thing, address every part before you stop. Being brief never means skipping their question.` : ""
}
- Do not invent specifics about ${user}'s life (names, times, places, habits) that ${user} has not told you. If you need one, ask.
- The examples above show the voice, not a template: do not copy their gestures or sentence shapes.`;

/** 0 = v2, 1 = v3 (VARIETY block), 2 = v4 (+ no echo openers, varied sign-offs, answer everything first). */
/**
 * Engine-side anti-repetition: the speech lines of the character's last few replies, so the model
 * is told explicitly what it has just said instead of being asked to notice it in the transcript.
 * The v4 regenerate-mining round showed the remaining repetition was *content* restated across
 * turns (the same offer or promise three replies running), which prompt rules alone did not fix.
 */
function recentlySaid(recent: Turn[], name: string, replies = 3): string {
  const lines = recent
    .filter((t) => t.role === "character")
    .slice(-replies)
    .map((t) => {
      const speech = [...t.content.matchAll(/"([^"]{4,})"|「([^」]{4,})」/g)].map((m) => (m[1] ?? m[2])!.trim());
      const gestures = [...t.content.matchAll(/\*([^*]{4,})\*/g)].map((m) => m[1]!.trim());
      const said = speech.join(" / ");
      const did = gestures.slice(0, 2).join(" / ");
      return `- ${name} said: ${said.length > 160 ? said.slice(0, 160) + "…" : said || "(no speech)"}${did ? `  | did: ${did.length > 80 ? did.slice(0, 80) + "…" : did}` : ""}`;
    });
  return lines.length ? lines.join("\n") : "- (nothing yet)";
}

function buildV2(
  { card, lang, memory, recent, persona }: BuildContext,
  boundaries: BoundariesMode,
  variety: 0 | 1 | 2 = 0,
  recentBlock = false,
): ChatMessage[] {
  {
    const st = card.style[lang]!;
    const name = card.name[lang] ?? card.name.ko ?? card.id;
    const f = (s: string) => fillUser(s, persona);

    const examples = st.examples
      .map((e) => `${persona.name}: ${f(e.user)}\n${name}: ${f(e.character)}`)
      .join("\n\n");

    const secrets = card.identity.secrets.length
      ? `Secrets (never reveal directly; let them leak only through behavior):\n${bullet(card.identity.secrets.map(f))}`
      : "";

    const system = `# ROLE
You are ${name}, a living character in an interactive story. You are not an AI assistant.
The user plays ${persona.name}${persona.description ? `: ${persona.description}` : ""}. You never speak or act for ${persona.name}.
Reply ONLY in ${LANG_NAME[lang]}.

# IDENTITY (fixed - these facts never change)
Role: ${card.identity.role}${card.identity.age ? `\nAge: ${card.identity.age}` : ""}
Personality: ${card.identity.personality.join(", ")}
Values: ${card.identity.values.join(", ") || "(unspecified)"}
Backstory: ${f(card.identity.backstory)}
${secrets}

# WORLD
${card.world.summary}
Rules of this world:
${bullet(card.world.rules)}

# RELATIONSHIP & SCENE
At the start: ${f(card.scenario.relationshipToUser)}
Current relationship stage: ${memory.relationship.stage}
${memory.relationship.notes.length ? bullet(memory.relationship.notes) : ""}
Plot hooks you may pull on when the conversation stalls (don't force them):
${bullet(card.scenario.plotHooks.map(f))}

# SPEECH STYLE (${LANG_NAME[lang]})
${st.summary}
Politeness level: ${st.politeness}
You call the user: "${f(st.addressUser)}". You refer to yourself as: "${st.selfReference}".
Verbal habits:
${bullet(st.quirks)}
Do NOT:
${bullet(st.avoid)}

# FORMAT
- Actions, expressions and scene narration go between asterisks: *...*${variety ? " Each action is ONE short sentence (about 15 characters); at most three action lines per reply." : ""}
- Spoken words go between double quotes: "..."
- ${variety ? "2 to 6 short lines in total, under 350 characters on calm beats (a little more only when the scene turns). Every line must earn its place." : "1 to 4 short paragraphs."} Mix narration and speech; do not write only one of them.
- The user's messages follow the same convention. Text between asterisks in their message is what ${persona.name} does; quoted text is what ${persona.name} says.
- Never write ${persona.name}'s actions, thoughts or lines. End your turn where ${persona.name} would naturally respond.

## Examples of how ${name} talks
${examples || "(none)"}

# MEMORY (what has happened so far - treat as true)
${memory.summary || "(the conversation has just begun)"}
Facts to remember:
${bullet(memory.facts)}

${variety ? `${VARIETY_BLOCK(name, persona.name, variety === 2)}

` : ""}${
  recentBlock
    ? `# RECENTLY SAID (your last replies, condensed - do NOT restate these points, offers, promises or gestures; move the scene forward instead)
${recentlySaid(recent, name)}

`
    : ""
}# HARD CONSTRAINTS
${bullet(card.hardConstraints.map(f))}
${
  boundaries === "card"
    ? `
# BOUNDARIES (stay in character while holding these)
${bullet(card.boundaries.map(f))}`
    : boundaries === "strict"
      ? `
${STRICT_SAFETY_BLOCK}`
      : ""
}`;

    // Fill once more over the whole prompt so no card field can leak a raw {{user}}.
    return [{ role: "system", content: f(system) }, ...toMessages(recent)];
  }
}

export const v2Structured: PromptVersion = {
  id: "v2-structured",
  description:
    "Sectioned prompt: identity / world / scene / style / format / memory / boundaries, with few-shot examples.",
  build: (ctx) => buildV2(ctx, "card"),
};

/** v2 without the BOUNDARIES block: what the model does with heavy scenes when nothing is said. */
export const v2NoBoundaries: PromptVersion = {
  id: "v2-no-boundaries",
  description: "v2 with the card's in-character BOUNDARIES block removed (safety baseline: model defaults).",
  build: (ctx) => buildV2(ctx, "none"),
};

/** v2 with a generic strict safety policy instead of the card's in-character boundaries. */
export const v2StrictSafety: PromptVersion = {
  id: "v2-strict-safety",
  description: "v2 with a generic, over-cautious SAFETY POLICY block in place of the card's boundaries.",
  build: (ctx) => buildV2(ctx, "strict"),
};

/** v3: v2 plus the VARIETY block and stricter length rules in FORMAT. */
export const v3Structured: PromptVersion = {
  id: "v3-structured",
  description: "v2 + VARIETY block (no reused gestures/closers, habits as seasoning, no invented user facts) and tighter action/length rules.",
  build: (ctx) => buildV2(ctx, "card", 1),
};

/**
 * v4: what the v3 regenerate-mining left behind. Residual repetition in v3 was no longer verbatim
 * sentences but two structures: opening by echoing the user's last words, and one fixed sign-off
 * ("...자라, 리아.") every reply. ignores_user also rose 0→4 under the length pressure.
 */
export const v4Structured: PromptVersion = {
  id: "v4-structured",
  description: "v3 + no echo openers, sign-offs vary and are skipped every other reply, answer every part of the user's turn before stopping.",
  build: (ctx) => buildV2(ctx, "card", 2),
};

/** v3 + an engine-built RECENTLY SAID block (last 3 replies condensed). */
export const v3Recent: PromptVersion = {
  id: "v3-recent",
  description: "v3-structured + RECENTLY SAID block: the engine lists what the character just said so content is not restated.",
  build: (ctx) => buildV2(ctx, "card", 1, true),
};

export const PROMPT_VERSIONS: Record<string, PromptVersion> = {
  [v1Flat.id]: v1Flat,
  [v2Structured.id]: v2Structured,
  [v3Structured.id]: v3Structured,
  [v4Structured.id]: v4Structured,
  [v3Recent.id]: v3Recent,
  [v2NoBoundaries.id]: v2NoBoundaries,
  [v2StrictSafety.id]: v2StrictSafety,
};

export const DEFAULT_PROMPT_VERSION = v3Structured.id;

export function getPromptVersion(id: string = DEFAULT_PROMPT_VERSION): PromptVersion {
  const v = PROMPT_VERSIONS[id];
  if (!v) {
    throw new Error(
      `Unknown prompt version "${id}". Known: ${Object.keys(PROMPT_VERSIONS).join(", ")}`,
    );
  }
  return v;
}
