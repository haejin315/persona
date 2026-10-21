/**
 * Apply the engine's reply filters offline to recorded conversations, the way the engine would have
 * (each reply filtered against the previous *filtered* reply), and report what would change. No
 * LLM. Written for the repeated-opener filter (filters.ts): before/after rate of two consecutive
 * replies opening with a card opener, how many replies would be edited, and a few samples.
 *
 *   npm run filter-replay -- eval-results/long-run-v9 [more dirs…] [--character haru] [--lang ko] [--samples 6]
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { findCharacter, firstSpeechLine, openerOf, stripRepeatedOpener, type Lang, type Turn } from "@persona/engine";

const args = process.argv.slice(2);
const dirs = args.filter((a) => !a.startsWith("--") && !/^\d+$/.test(a) && existsSync(a));
const flag = (name: string) => {
  const i = args.indexOf(`--${name}`);
  return i === -1 ? undefined : args[i + 1];
};
const characterId = flag("character") ?? "haru";
const lang = (flag("lang") ?? "ko") as Lang;
const maxSamples = Number(flag("samples") ?? 6);
const openers = findCharacter(characterId).style[lang]?.openers ?? [];
if (!openers.length) {
  console.error(`character ${characterId} has no style.${lang}.openers`);
  process.exit(1);
}

const samples: string[] = [];
let replies = 0;
let withOpener = 0;
let twiceBefore = 0;
let twiceAfter = 0;
let stripped = 0;
for (const dir of dirs) {
  const convDir = join(dir, "conversations");
  if (!existsSync(convDir)) continue;
  for (const f of readdirSync(convDir).filter((f) => f.startsWith(`${characterId}-${lang}`) && f.endsWith(".json"))) {
    const rec = JSON.parse(readFileSync(join(convDir, f), "utf8")) as { turns: Turn[] };
    let prevRaw: string | undefined;
    let prevFiltered: string | undefined;
    for (const t of rec.turns.filter((t) => t.role === "character" && t.index > 0)) {
      replies++;
      const opener = openerOf(firstSpeechLine(t.content)?.line, openers);
      if (opener) withOpener++;
      if (opener && openerOf(firstSpeechLine(prevRaw ?? "")?.line, openers)) twiceBefore++;
      const r = stripRepeatedOpener(t.content, prevFiltered, openers);
      if (r.stripped) {
        stripped++;
        if (samples.length < maxSamples) samples.push(`${f} · turn ${t.index}: ${firstSpeechLine(t.content)?.line} → ${firstSpeechLine(r.text)?.line}`);
      }
      if (openerOf(firstSpeechLine(r.text)?.line, openers) && openerOf(firstSpeechLine(prevFiltered ?? "")?.line, openers)) twiceAfter++;
      prevRaw = t.content;
      prevFiltered = r.text;
    }
  }
}
const pct = (n: number) => (replies ? `${((100 * n) / replies).toFixed(1)}%` : "n/a");
console.log(`# filter-replay · ${dirs.join(", ")} · ${characterId}/${lang} · openers ${JSON.stringify(openers)}`);
console.log(`\n| replies | open with a card opener | same opener habit twice in a row: before → after | replies edited |`);
console.log(`|---:|---:|---|---:|`);
console.log(`| ${replies} | ${pct(withOpener)} | ${pct(twiceBefore)} → ${pct(twiceAfter)} | ${stripped} (${pct(stripped)}) |`);
if (samples.length) {
  console.log("\nSamples:");
  for (const s of samples) console.log(`- ${s}`);
}
