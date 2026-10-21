/**
 * Engine-side reply filters: deterministic text fixes for the things a prompt rule cannot enforce.
 *
 * Repeated opener. The card lists a character's openers ("...아,", "음,"); the VARIETY rule says
 * "never the same habit in two consecutive replies", yet in the v9 200-turn runs one Haru reply in
 * five opened with a card opener right after a reply that had (docs/02, `npm run register-stats`).
 * A regenerate would cost a second call; dropping the opener costs nothing and reads naturally,
 * because an opener is a detachable beat ("...아, 그러네요." → "그러네요.").
 */

const OPEN_QUOTES = ['"', "“", "「", "『"]; // " “ 「 『

/** Index and text of the first spoken line (a line that starts with an opening quote). */
export function firstSpeechLine(text: string): { index: number; line: string } | undefined {
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i]!.trimStart();
    if (t && OPEN_QUOTES.includes(t[0]!)) return { index: i, line: lines[i]! };
  }
  return undefined;
}

/** The card opener a spoken line begins with, if any. */
export function openerOf(line: string | undefined, openers: string[]): string | undefined {
  if (!line) return undefined;
  const t = line.trimStart();
  const rest = OPEN_QUOTES.includes(t[0] ?? "") ? t.slice(1).trimStart() : t;
  return openers.find((o) => o && rest.startsWith(o));
}

/**
 * If both the previous reply and this one open with a card opener, remove it from this one.
 * Returns the (possibly) edited text and whether anything changed.
 */
export function stripRepeatedOpener(text: string, prev: string | undefined, openers: string[]): { text: string; stripped: boolean } {
  if (!openers.length || !prev) return { text, stripped: false };
  if (!openerOf(firstSpeechLine(prev)?.line, openers)) return { text, stripped: false };
  const cur = firstSpeechLine(text);
  const opener = openerOf(cur?.line, openers);
  if (!cur || !opener) return { text, stripped: false };

  const lines = text.split("\n");
  const line = lines[cur.index]!;
  const indent = line.length - line.trimStart().length;
  const t = line.trimStart();
  const quote = OPEN_QUOTES.includes(t[0]!) ? t[0]! : "";
  const afterQuote = quote ? t.slice(1).trimStart() : t;
  const rest = afterQuote.slice(opener.length).trimStart();
  // Nothing left to say without the opener: keep the line as it is.
  if (!rest || /^["”」』]/.test(rest)) return { text, stripped: false };
  lines[cur.index] = `${line.slice(0, indent)}${quote}${rest}`;
  return { text: lines.join("\n"), stripped: true };
}

/** True once `partial` holds a complete first spoken line (so a streaming filter can be applied and the prefix released). */
export function firstSpeechComplete(partial: string): boolean {
  const first = firstSpeechLine(partial);
  if (!first) return false;
  return partial.split("\n").length > first.index + 1;
}
