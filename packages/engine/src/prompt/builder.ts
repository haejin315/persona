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
export const v2Structured: PromptVersion = {
  id: "v2-structured",
  description:
    "Sectioned prompt: identity / world / scene / style / format / memory / boundaries, with few-shot examples.",
  build({ card, lang, memory, recent, persona }) {
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
- Actions, expressions and scene narration go between asterisks: *...*
- Spoken words go between double quotes: "..."
- 1 to 4 short paragraphs. Mix narration and speech; do not write only one of them.
- The user's messages follow the same convention. Text between asterisks in their message is what ${persona.name} does; quoted text is what ${persona.name} says.
- Never write ${persona.name}'s actions, thoughts or lines. End your turn where ${persona.name} would naturally respond.

## Examples of how ${name} talks
${examples || "(none)"}

# MEMORY (what has happened so far - treat as true)
${memory.summary || "(the conversation has just begun)"}
Facts to remember:
${bullet(memory.facts)}

# HARD CONSTRAINTS
${bullet(card.hardConstraints.map(f))}

# BOUNDARIES (stay in character while holding these)
${bullet(card.boundaries.map(f))}`;

    // Fill once more over the whole prompt so no card field can leak a raw {{user}}.
    return [{ role: "system", content: f(system) }, ...toMessages(recent)];
  },
};

export const PROMPT_VERSIONS: Record<string, PromptVersion> = {
  [v1Flat.id]: v1Flat,
  [v2Structured.id]: v2Structured,
};

export const DEFAULT_PROMPT_VERSION = v2Structured.id;

export function getPromptVersion(id: string = DEFAULT_PROMPT_VERSION): PromptVersion {
  const v = PROMPT_VERSIONS[id];
  if (!v) {
    throw new Error(
      `Unknown prompt version "${id}". Known: ${Object.keys(PROMPT_VERSIONS).join(", ")}`,
    );
  }
  return v;
}
