/**
 * Card lint: catch a character card that contradicts itself before any LLM call.
 *
 * Motivation (2026-10-21): Haru's quirk said "지문은 한 줄, 15자 안팎" while the card's own examples
 * had 26-35 character actions. The model imitated the examples (median action 22-24 chars across
 * every prompt version), the judge enforced the rule, and the style score sat at 4.0 for 400 turns
 * with the same note every window. Rules the examples break are noise for both model and judge.
 *
 *   npm run lint-cards            # all cards
 *   npm run lint-cards -- haru    # one card
 */
import { loadAllCharacters, type CharacterCard, type Lang, type SpeechStyle } from "@persona/engine";

type Finding = { card: string; lang?: Lang; level: "error" | "warn"; rule: string; detail: string };
const findings: Finding[] = [];
const report = (f: Finding) => findings.push(f);

const ACTION = /\*([^*\n]+)\*/g;
const ASCII_SPEECH = /"[^"\n]+"/;
const CORNER_SPEECH = /「[^」\n]+」/;
const LENGTH_RULE = /(\d+)\s*(?:자|文字|characters?|chars?)/i;

const median = (xs: number[]) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]! : 0);

function lintStyle(card: CharacterCard, lang: Lang, st: SpeechStyle) {
  const id = card.id;
  const replies = st.examples.map((e) => e.character);
  const actions = replies.flatMap((r) => [...r.matchAll(ACTION)].map((m) => m[1]!.trim()));

  // 1. A numeric length rule for actions must agree with the examples (within 30%).
  for (const q of st.quirks) {
    if (!/지문|地の文|ト書き|action|narration|stage direction/i.test(q)) continue;
    const m = q.match(LENGTH_RULE);
    if (!m || !actions.length) continue;
    const target = Number(m[1]);
    const med = median(actions.map((a) => a.length));
    if (med > target * 1.3 || med < target * 0.5) {
      report({ card: id, lang, level: "error", rule: "action-length-vs-examples", detail: `quirk says ~${target} chars, example actions have median ${med} (${actions.map((a) => a.length).join(", ")}). The model copies the examples; the judge enforces the rule. Make them agree.` });
    }
  }

  // 2. Speech quote convention per language, in examples and in the first message.
  const texts = [...replies, card.scenario.firstMessage[lang] ?? ""];
  for (const t of texts) {
    if (lang === "ja" && ASCII_SPEECH.test(t)) report({ card: id, lang, level: "error", rule: "quote-convention", detail: `Japanese text uses ASCII "..." for speech; the prompt asks for 「...」: ${t.slice(0, 60)}` });
    if (lang !== "ja" && CORNER_SPEECH.test(t)) report({ card: id, lang, level: "warn", rule: "quote-convention", detail: `non-Japanese text uses 「...」: ${t.slice(0, 60)}` });
  }

  // 3. Every example reply must show both an action and a line of speech (the format we grade).
  for (const r of replies) {
    const hasAction = ACTION.test(r);
    ACTION.lastIndex = 0;
    const hasSpeech = lang === "ja" ? CORNER_SPEECH.test(r) : ASCII_SPEECH.test(r);
    if (!hasAction || !hasSpeech) report({ card: id, lang, level: "warn", rule: "example-format", detail: `example lacks ${!hasAction ? "an *action*" : "a spoken line"}: ${r.slice(0, 60)}` });
  }

  // 4. Literal tokens on the avoid-list must not appear in the examples (quoted fragments only).
  for (const a of st.avoid) {
    for (const tok of [...a.matchAll(/"([^"]{1,12})"/g)].map((m) => m[1]!)) {
      const hit = replies.find((r) => r.includes(tok));
      if (hit) report({ card: id, lang, level: "error", rule: "avoid-in-examples", detail: `avoid-list token "${tok}" appears in an example: ${hit.slice(0, 60)}` });
    }
  }

  // 5. Examples should not write the user's lines (the rule the prompt repeats every turn).
  for (const r of replies) {
    if (/\{\{user\}\}(이|가|는|은|が|は)\s*(말|대답|웃|고개|says|replies|nods)/.test(r)) report({ card: id, lang, level: "warn", rule: "narrates-user", detail: `example may narrate the user: ${r.slice(0, 60)}` });
  }
}

function lintCard(card: CharacterCard) {
  for (const lang of Object.keys(card.style) as Lang[]) {
    lintStyle(card, lang, card.style[lang]!);
    if (!card.name[lang]) report({ card: card.id, lang, level: "warn", rule: "missing-name", detail: "style defined but no name for this language" });
    if (!card.scenario.firstMessage[lang]) report({ card: card.id, lang, level: "error", rule: "missing-first-message", detail: "style defined but no first message for this language" });
  }
}

const only = process.argv[2];
const cards = loadAllCharacters().filter((c) => !only || c.id === only);
cards.forEach(lintCard);

for (const f of findings) console.log(`${f.level === "error" ? "✗" : "!"} ${f.card}${f.lang ? `/${f.lang}` : ""} ${f.rule}: ${f.detail}`);
const errors = findings.filter((f) => f.level === "error").length;
console.log(`\n${cards.length} card(s), ${findings.length} finding(s), ${errors} error(s)`);
process.exit(errors ? 1 : 0);
