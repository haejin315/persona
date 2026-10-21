/**
 * Register statistics for recorded Korean conversations: how often a polite character's sentences
 * end in plain form, how often a reply ends on a question, and how long actions are. Rule-based,
 * no LLM. Written for the v6 round (docs/09): the judge's residual style notes were about endings.
 *
 *   npm run register-stats -- eval-results/regen-fix-v7 [more dirs…] [--character haru]
 *
 * Plain-form endings are counted on sentences inside "…" that do not end with "?", using the
 * endings the judge actually flagged (~고. ~니까. ~다. ~다니.) plus the usual 반말 finals; "네." (yes)
 * is excluded. A sentence ending in the 요/죠/니다 family is never counted.
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { basename, join } from "node:path";

const dirs = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const charFlag = process.argv.indexOf("--character");
const onlyCharacter = charFlag === -1 ? undefined : process.argv[charFlag + 1];

const SPEECH = /"([^"\n]+)"/g;
const ACTION = /\*([^*\n]+)\*/g;
const PLAIN = /(다|고|니까|지|거야|야|어|아|군|래|자|냐|니|을까|까)[.…!]*$/;
const POLITE = /(요|죠|니다|습니까|세요|네요|게요|까요|잖아요|거예요|예요|에요)[.…!?]*$/;

type Row = { replies: number; sentences: number; plain: number; repliesWithPlain: number; endsOnQuestion: number; actions: number[] };
const rows = new Map<string, Row>();
const median = (xs: number[]) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]! : 0);

for (const dir of dirs) {
  const conv = join(dir, "conversations");
  if (!existsSync(conv)) continue;
  for (const file of readdirSync(conv).filter((f) => f.endsWith(".json"))) {
    const rec = JSON.parse(readFileSync(join(conv, file), "utf8"));
    if (String(rec.arm.model).startsWith("fake")) continue;
    const character = String(rec.scenarioId).split("-")[0];
    if (onlyCharacter && character !== onlyCharacter) continue;
    const key = `${basename(dir)} · ${character} · ${rec.arm.id}`;
    const row = rows.get(key) ?? { replies: 0, sentences: 0, plain: 0, repliesWithPlain: 0, endsOnQuestion: 0, actions: [] };
    for (const t of rec.turns) {
      if (t.role !== "character" || t.index === 0) continue;
      const speech = [...String(t.content).matchAll(SPEECH)].map((m) => m[1]!);
      row.actions.push(...[...String(t.content).matchAll(ACTION)].map((m) => m[1]!.trim().length));
      if (!speech.length) continue;
      row.replies++;
      const sentences = speech
        .flatMap((s) => s.split(/(?<=[.?!…])\s+/))
        .map((s) => s.trim())
        .filter((s) => s && !s.endsWith("?"));
      const plain = sentences.filter((s) => PLAIN.test(s) && !POLITE.test(s));
      row.sentences += sentences.length;
      row.plain += plain.length;
      if (plain.length) row.repliesWithPlain++;
      if (speech[speech.length - 1]!.trim().endsWith("?")) row.endsOnQuestion++;
    }
    rows.set(key, row);
  }
}

const pct = (a: number, b: number) => (b ? `${((100 * a) / b).toFixed(1)}%` : "–");
console.log("| run · character · arm | replies | plain-form sentences | replies with one | ends on a question | action median |");
console.log("|---|---:|---:|---:|---:|---:|");
for (const [k, r] of rows) console.log(`| ${k} | ${r.replies} | ${pct(r.plain, r.sentences)} | ${pct(r.repliesWithPlain, r.replies)} | ${pct(r.endsOnQuestion, r.replies)} | ${median(r.actions)} |`);
