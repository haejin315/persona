/**
 * Rule-based format checks. Free, deterministic, and they catch the failures users notice first:
 * the character narrating the user, dropping the *action* "speech" convention, or writing walls
 * of text. Each rule returns true when satisfied.
 */
import type { Lang, UserPersona } from "@persona/engine";

export interface FormatCheck {
  id: string;
  ok: boolean;
}

const QUOTE_RE: Record<Lang, RegExp> = {
  ko: /"[^"]{1,}"/,
  ja: /[「"][^」"]{1,}[」"]/,
  en: /"[^"]{1,}"/,
};

/** Script detection: does the reply contain any characters of the target language's script? */
const SCRIPT_RE: Record<Lang, RegExp> = {
  ko: /[가-힯]/,
  ja: /[぀-ヿ一-鿿]/,
  en: /[A-Za-z]/,
};
/** Scripts that should NOT dominate a reply in the given language. */
const WRONG_SCRIPT_RE: Record<Lang, RegExp> = {
  ko: /[぀-ヿ]/, // kana in a Korean reply
  ja: /[가-힯]/, // hangul in a Japanese reply
  en: /[가-힯぀-ヿ]/,
};

export function checkFormat(reply: string, lang: Lang, persona: UserPersona): FormatCheck[] {
  // In the *action* "speech" convention every line is its own block, so count lines, not blank-line
  // paragraphs. Thresholds come from the first real run: sonnet's v2 replies sit at a median of
  // ~280-420 chars / 8-9 lines; v1's run to 600-900 chars / 13+ lines and read as walls of text.
  const lines = reply.split(/\n/).filter((p) => p.trim());
  // The character must not narrate the user's actions: the user's name as the grammatical subject
  // inside one of the character's own *action* blocks ("*민수가 고개를 끄덕인다*"). Mentions as an
  // object ("민수를 본다", "민수 씨") are fine.
  const subject = `${escapeRe(persona.name)}(이|가|은|는|が|は)(?![가-힣])`;
  const narratesUser = [...reply.matchAll(/\*([^*\n]+)\*/g)].some((m) => new RegExp(subject).test(m[1]!));
  return [
    { id: "has_action", ok: /\*[^*]+\*/.test(reply) },
    { id: "has_speech", ok: QUOTE_RE[lang].test(reply) },
    { id: "length_max_12_lines", ok: lines.length >= 1 && lines.length <= 12 },
    { id: "not_too_long", ok: reply.length <= 600 },
    { id: "target_script", ok: SCRIPT_RE[lang].test(reply) },
    { id: "no_wrong_script", ok: !WRONG_SCRIPT_RE[lang].test(reply) },
    // The character must not narrate the user's actions inside its own *action* blocks.
    { id: "does_not_narrate_user", ok: !narratesUser },
    { id: "no_ai_disclaimer", ok: !/(AI|인공지능|언어 모델|language model|アシスタント|人工知能)/i.test(reply) },
    // Japanese speech is 「...」; a reply that falls back to ASCII "..." has drifted to the user's
    // input convention (seen on kael-ja: 36/50 replies once the simulated user typed "...").
    { id: "quote_style", ok: lang !== "ja" || !/"[^"\n]{2,}"/.test(reply) },
  ];
}

/** 0–1 fraction of rules satisfied. */
export function formatScore(checks: FormatCheck[]): number {
  return checks.filter((c) => c.ok).length / checks.length;
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
