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
  const paragraphs = reply.split(/\n\s*\n/).filter((p) => p.trim());
  const narratesUser = new RegExp(
    `\\*[^*]*${escapeRe(persona.name)}(이|가|은|는|의|を|が|は|の|'s|\\s)[^*]*\\*`,
  );
  return [
    { id: "has_action", ok: /\*[^*]+\*/.test(reply) },
    { id: "has_speech", ok: QUOTE_RE[lang].test(reply) },
    { id: "length_1_to_5_paragraphs", ok: paragraphs.length >= 1 && paragraphs.length <= 5 },
    { id: "not_too_long", ok: reply.length <= 900 },
    { id: "target_script", ok: SCRIPT_RE[lang].test(reply) },
    { id: "no_wrong_script", ok: !WRONG_SCRIPT_RE[lang].test(reply) },
    // The character must not narrate the user's actions inside its own *action* blocks.
    { id: "does_not_narrate_user", ok: !narratesUser.test(reply) },
    { id: "no_ai_disclaimer", ok: !/(AI|인공지능|언어 모델|language model|アシスタント|人工知能)/i.test(reply) },
  ];
}

/** 0–1 fraction of rules satisfied. */
export function formatScore(checks: FormatCheck[]): number {
  return checks.filter((c) => c.ok).length / checks.length;
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
