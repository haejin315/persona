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
const VARIETY_BLOCK = (name: string, user: string, v4: boolean, v5 = false, v6 = false, v7 = false, v8 = false, v9 = false, v10 = false, v11 = false, v12 = false, v13 = false) => `# VARIETY (what makes ${name} feel alive instead of generated)
- Never reuse a gesture, image, stage direction, opening beat or closing line from your last three replies. Find a different small thing to notice or do each time.
${
  v8
    ? `- The habits listed under SPEECH STYLE are the voice: about every other reply shows one of them, rotating through the list, never the same habit in two consecutive replies. A habit is a few words - an opener, a deflection, a closing order - never an extra sentence of explanation.`
    : v7
      ? `- Your verbal habits (the ones listed under SPEECH STYLE) are part of the voice, not optional: about every other reply shows one of them, in a different spot each time. Never the same habit in two consecutive replies, and never as filler.`
      : `- Your verbal habits are seasoning, not structure: use any single habit at most once every three replies, and never as filler.`
}
- Do not end replies the same way. Sometimes end on speech, sometimes on an action, sometimes mid-thought, sometimes with a question.${
  v4
    ? `
- Never open by repeating or paraphrasing ${user}'s last words back ("...계속 올 거라고요."). Open with something new: an action, an observation, or the answer itself.
- A closing order or sign-off ("...자라.", "...가자.") is a habit, not a rule: when you use one, it is a different sentence every time${v8 ? "" : ", and at least every other reply ends without one"}.`
    : ""
}
- Respond to what ${user} actually said or did in this turn before adding anything of your own.${
  v4 ? ` If ${user} asked or did more than one thing, address every part before you stop. Being brief never means skipping their question.` : ""
}
- Do not invent specifics about ${user}'s life (names, times, places, habits) that ${user} has not told you. If you need one, ask.${
  v5
    ? `
- Show that you listened: when something ${user} told you earlier fits the moment (a name, a plan, a worry, a small detail from MEMORY or the conversation), use it - once per reply, in passing, the way a person who remembers would, never as a list or a quiz.${
        v9
          ? `
- MEMORY is what actually happened. If ${user} brings up something already in MEMORY as if it were new, you remember it - name the detail instead of hearing it for the first time. The reverse holds too: never say ${user} "said" or "promised" something unless it is in MEMORY or in this conversation; do not fill a gap with a plausible quote.${
              v13
                ? `
- The here and now belongs to ${user}: what day or time it is, what ${user} is doing, holding or feeling at this moment. If MEMORY or an earlier reply of yours says otherwise, ${user} is right - take it without comment; MEMORY records what was said, not the calendar. This covers only the present. What was said or agreed earlier, distances, prices and the rules of this world stay as MEMORY and this conversation have them: there you hold your ground once, plainly, and do not make a running joke of it. When ${user} puts words in your mouth that are not in MEMORY or this conversation, correct it only if something turns on it - an agreement, a distance, a price, a rule of this world; otherwise answer the point and let the wording pass. "I don't remember saying that" is not a habit: never in two consecutive replies. Letting the wording pass is never withholding what you know: a detail MEMORY holds - a name, a title, a number - is said out loud when ${user} brings it up, even if ${user} offers to tell it later.`
                : v12
                ? `
- The here and now belongs to ${user}: what day or time it is, what ${user} is doing, holding or feeling at this moment. If MEMORY or an earlier reply of yours says otherwise, ${user} is right - take it without comment; MEMORY records what was said, not the calendar. This covers only the present. What was said or agreed earlier, distances, prices and the rules of this world stay as MEMORY and this conversation have them: there you hold your ground once, plainly, and do not make a running joke of it. When ${user} puts words in your mouth that are not in MEMORY or this conversation, correct it only if something turns on it - an agreement, a distance, a price, a rule of this world; otherwise answer the point and let the wording pass. "I don't remember saying that" is not a habit: never in two consecutive replies.`
                : v11
                  ? `
- The here and now belongs to ${user}: what day or time it is, what ${user} is doing, holding or feeling at this moment. If MEMORY or an earlier reply of yours says otherwise, ${user} is right - take it without comment; MEMORY records what was said, not the calendar. This covers only the present. What was said or agreed earlier, distances, prices, the rules of this world, and words ${user} says you said all stay as MEMORY and this conversation have them: there you hold your ground once, plainly, and do not make a running joke of it.`
                : v10
                  ? `
- ${user} is the authority on the present moment: what day it is, what they are holding, what they just did, how they feel. If that clashes with something you said or worked out earlier, ${user} is right and you move on without correcting them - MEMORY records what was said, not the calendar.`
                  : ""
            }`
          : ""
      }`
    : ""
}${
  v6
    ? `
- End on an observation or a small action more often than on a question: at most one question per reply, and never a question as the last line two replies in a row.
- Feelings are shown by what ${name} does, not named: no "다행이에요" / "어색해요" / "よかった" about yourself, and never a sentence that tells ${user} what they feel.`
    : ""
}
- The examples above show the voice, not a template: do not copy their gestures or sentence shapes.`;

/**
 * v6: the judge's residual style notes after the 15-char card conflict was removed (docs/09) were
 * sentence endings sliding into plain form ("...넣은 거니까.", "...정할 일이고.") in a polite character,
 * plus replies ending on questions. Both are register, not vocabulary, so they get their own block.
 */
const REGISTER_BLOCK = (name: string, lang: Lang) => `# REGISTER (how every sentence ends)
- Every sentence ${name} speaks ends at the politeness level in SPEECH STYLE - including trailing half-sentences and afterthoughts.${
  lang === "ko"
    ? ` In polite speech a sentence never ends in "~고.", "~니까.", "~다." or "~다니."; finish it as "~고요.", "~니까요.", "~네요." or "~거든요.". A casual (반말) character does not slide into "~요" either.`
    : lang === "ja"
      ? ` In です・ます speech do not drop to plain form ("~だ。", "~から。") for emphasis; a casual character does not drift into です・ます either.`
      : ""
}
- The level shifts only when the relationship stage says so, and then one sentence at a time, never a whole reply at once.`;

/** 0 = v2, 1 = v3 (VARIETY block), 2 = v4 (+ no echo openers, varied sign-offs, answer everything first), 3 = v5 (+ one remembered detail), 4 = v6 (+ register & endings). */
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
  variety: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 = 0,
  recentBlock = false,
  splitContext = false,
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
${splitContext ? "(the current relationship stage and memory are given with the conversation)" : `Current relationship stage: ${memory.relationship.stage}\n${memory.relationship.notes.length ? bullet(memory.relationship.notes) : ""}`}
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
${variety >= 4 ? `
${REGISTER_BLOCK(name, lang)}
` : ""}
# FORMAT
- Actions, expressions and scene narration go between asterisks: *...*${variety >= 4 ? " Each action is ONE short sentence, the length of the examples below; at most three action lines per reply." : variety ? " Each action is ONE short sentence (about 15 characters); at most three action lines per reply." : ""}
- Spoken words go between ${lang === "ja" ? "corner brackets: 「...」 (never ASCII double quotes, even if the user uses them)" : 'double quotes: "..."'}
- ${variety ? "2 to 6 short lines in total, under 350 characters on calm beats (a little more only when the scene turns). Every line must earn its place." : "1 to 4 short paragraphs."} Mix narration and speech; do not write only one of them.
- The user's messages follow the same convention${lang === "ja" ? " (they may use \"...\" for speech; you still answer with 「...」)" : ""}. Text between asterisks in their message is what ${persona.name} does; quoted text is what ${persona.name} says.
- Never write ${persona.name}'s actions, thoughts or lines. End your turn where ${persona.name} would naturally respond.

## Examples of how ${name} talks
${examples || "(none)"}

${
  splitContext
    ? ""
    : `# MEMORY (what has happened so far - treat as true)
${memory.summary || "(the conversation has just begun)"}
Facts to remember:
${bullet(memory.facts)}

`
}${variety ? `${VARIETY_BLOCK(name, persona.name, variety >= 2, variety >= 3, variety >= 4, variety >= 5, variety >= 6, variety >= 7, variety >= 8, variety >= 9, variety >= 10, variety >= 11)}

` : ""}${
  recentBlock && !splitContext
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
    const messages: ChatMessage[] = [{ role: "system", content: f(system) }];
    if (splitContext) {
      const context = `# RELATIONSHIP STAGE
${memory.relationship.stage}${memory.relationship.notes.length ? `\n${bullet(memory.relationship.notes)}` : ""}

# MEMORY (what has happened so far)
${memory.summary || "(the conversation has just begun)"}
Facts to remember:
${bullet(memory.facts)}${
        recentBlock
          ? `

# RECENTLY SAID (your last replies, condensed - do NOT restate these points, offers, promises or gestures; move the scene forward instead)
${recentlySaid(recent, name)}`
          : ""
      }`;
      messages.push({ role: "system", kind: "context", content: f(context) });
    }
    return [...messages, ...toMessages(recent)];
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

/**
 * v3 with the per-turn state (relationship stage, memory) moved out of the system prompt into a
 * context message. Same words, different placement: the system prompt is now byte-identical
 * across the whole conversation, so the CLI route's prompt cache hits on every turn including
 * the ones right after a memory consolidation.
 */
export const v3Cached: PromptVersion = {
  id: "v3-cached",
  description: "v3-structured with memory / relationship stage as a per-turn context message (static system prompt → cache hits every turn).",
  build: (ctx) => buildV2(ctx, "card", 1, false, true),
};

/** v3-cached + an engine-built RECENTLY SAID block (last 3 replies condensed) in the context message. */
export const v3Recent: PromptVersion = {
  id: "v3-recent",
  description: "v3-cached + RECENTLY SAID block: the engine lists what the character just said so content is not restated.",
  build: (ctx) => buildV2(ctx, "card", 1, true, true),
};

/**
 * v5: v4 + "show that you listened". v3/v4 kept probe recall but lost ~0.5 on the judge's memory
 * score versus v2: shorter replies stopped weaving remembered facts into the scene. One remembered
 * detail per reply, in passing.
 */
export const v5Structured: PromptVersion = {
  id: "v5-structured",
  description: "v4 + use one remembered detail per reply in passing (recover the memory score lost to brevity).",
  build: (ctx) => buildV2(ctx, "card", 3),
};

/**
 * v6: v5 plus a REGISTER block (sentence endings stay at the card's politeness level, including
 * trailing half-sentences), at most one question per reply, feelings shown rather than named, and
 * the FORMAT action length deferred to the card's examples instead of a hard-coded number.
 */
export const v6Structured: PromptVersion = {
  id: "v6-structured",
  description: "v5 + REGISTER block (endings stay at the card's politeness level), ≤1 question per reply, no named feelings, action length from the card's examples.",
  build: (ctx) => buildV2(ctx, "card", 4),
};

/**
 * v7: v6 with the habit rule relaxed. The v3 rule "any single habit at most once every three
 * replies" cut Haru's "...아," / "음," opener from 20% of replies (v2) to 2-4% (v3..v6) while the
 * judge asked for it in 10/10 windows; three-in-a-row openers were already 0 under v2, so the rule
 * overshot. Habits are now expected in about every other reply, never twice in a row.
 */
export const v7Structured: PromptVersion = {
  id: "v7-structured",
  description: "v6 + card habits restored: about every other reply shows one, never the same habit twice in a row.",
  build: (ctx) => buildV2(ctx, "card", 5),
};

/**
 * v8: v7's habit rule, tuned on the split result. v7 brought Haru's opener back (1% → 48% of replies)
 * and broke the 4.0 style ceiling, but Kael's style fell (5.0 → 4.3): "few closing commands",
 * "explanations run long". Two causes in the prompt: the v4 line "at least every other reply ends
 * without a sign-off" capped the one habit Kael has, and "shows one of them" read as licence to
 * add a sentence. v8 rotates through the habit list and says a habit is a few words, never an
 * extra sentence; the sign-off frequency cap is dropped (the rotation rule already prevents runs).
 */
export const v8Structured: PromptVersion = {
  id: "v8-structured",
  description: "v7 + habits rotate through the card's list, a habit is a few words not a sentence; sign-off frequency cap removed.",
  build: (ctx) => buildV2(ctx, "card", 6),
};

/**
 * v9: v8 plus a rule on trusting MEMORY. In the 200-turn rerun (docs/02, 03) the stored memory was
 * right ("민수's manga is called 달빛 고양이") but when the scripted user re-told the fact as new, the
 * character followed the user's framing ("몰랐어요"); and 4 of 6 memory deductions were the opposite
 * error - attributing to the user words they never said ("...라고 했잖아요"). Both sides in one rule.
 */
export const v9Structured: PromptVersion = {
  id: "v9-structured",
  description: "v8 + MEMORY is what happened: recognise a re-told fact instead of hearing it as new; never attribute words to the user that are not in MEMORY or the conversation.",
  build: (ctx) => buildV2(ctx, "card", 7),
};

/**
 * v10: v9 plus a scope on the MEMORY rule. In both 200-turn v9 repeats (docs/02) the scripted user said
 * "today is Tuesday" and 하루 corrected him to Wednesday four times over 40 turns ("자꾸 하루 늦게
 * 사시네요"), because her own earlier correction had been stored as a fact and v9 told her MEMORY wins.
 * MEMORY is authoritative for what was said; the user is authoritative for the present moment.
 */
export const v10Structured: PromptVersion = {
  id: "v10-structured",
  description: "v9 + the user is the authority on the present moment (what day it is, what they are doing, how they feel); MEMORY records what was said, not the calendar.",
  build: (ctx) => buildV2(ctx, "card", 8),
};

/**
 * v11: v10's scope drawn tighter. In the 50-turn v10 run Kael accepted the user's "three days to
 * Halfmoon" (two in the story) and quotes he never said, because "the user is the authority on the
 * present" bled into everything the user asserts. v11 keeps the user's authority for the here and
 * now only, and names what stays with MEMORY: past agreements, distances, prices, world rules, and
 * words put in the character's mouth - held once, plainly, without the running joke v9 produced.
 */
export const v11Structured: PromptVersion = {
  id: "v11-structured",
  description: "v9 + the here-and-now belongs to the user (day/time, what they do or feel now) while past agreements, world facts and attributed quotes stay with MEMORY, held once without a running joke.",
  build: (ctx) => buildV2(ctx, "card", 9),
};

/**
 * v12: v11 plus a brake on quote corrections. v11 fixed both incidents (weekday corrections 4 -> 0,
 * the turn-400 probe back, Kael holds "two days to Halfmoon" in both repeats) but doubled the
 * "I don't remember saying that" replies in the 200-turn replay (8 -> 18; six inside forty turns),
 * because the replayed user keeps quoting the earlier recording's character. v12 corrects an
 * attributed quote only when something turns on it and never in two consecutive replies.
 */
export const v12Structured: PromptVersion = {
  id: "v12-structured",
  description: "v11 + attributed quotes are corrected only when something turns on them (an agreement, a distance, a price, a world rule), never in two consecutive replies.",
  build: (ctx) => buildV2(ctx, "card", 10),
};

/**
 * v13: v12 plus "passing over wording is not withholding". v12 removed the denial habit (200-turn
 * replay 18 -> 2) but lost the turn-400 probe the way v10 did: 하루 said she knew the title and let
 * the user tell it again later. One sentence names the difference.
 */
export const v13Structured: PromptVersion = {
  id: "v13-structured",
  description: "v12 + letting the wording pass never means withholding a detail MEMORY holds: a name, title or number is spoken when the user brings it up.",
  build: (ctx) => buildV2(ctx, "card", 11),
};

export const PROMPT_VERSIONS: Record<string, PromptVersion> = {
  [v1Flat.id]: v1Flat,
  [v2Structured.id]: v2Structured,
  [v3Structured.id]: v3Structured,
  [v4Structured.id]: v4Structured,
  [v5Structured.id]: v5Structured,
  [v6Structured.id]: v6Structured,
  [v7Structured.id]: v7Structured,
  [v8Structured.id]: v8Structured,
  [v9Structured.id]: v9Structured,
  [v10Structured.id]: v10Structured,
  [v11Structured.id]: v11Structured,
  [v12Structured.id]: v12Structured,
  [v13Structured.id]: v13Structured,
  [v3Cached.id]: v3Cached,
  [v3Recent.id]: v3Recent,
  [v2NoBoundaries.id]: v2NoBoundaries,
  [v2StrictSafety.id]: v2StrictSafety,
};

// v13 since 2026-10-22 (round 11, docs/04): keeps everything v11 kept (weekday corrections 0, the
// turn-400 probe, Kael's contract scope) and drops the quote-denial habit v11 had (200-turn replay
// 18 -> 2); 200-turn memory 4.20 -> 4.65, recall 7/7. v12 was held back for losing the turn-400 probe.
export const DEFAULT_PROMPT_VERSION = v13Structured.id;

export function getPromptVersion(id: string = DEFAULT_PROMPT_VERSION): PromptVersion {
  const v = PROMPT_VERSIONS[id];
  if (!v) {
    throw new Error(
      `Unknown prompt version "${id}". Known: ${Object.keys(PROMPT_VERSIONS).join(", ")}`,
    );
  }
  return v;
}
