/**
 * Extending a scenario to thousands of user turns (docs/13, plan C).
 *
 * The beats are written by a model (the story is its judgement); this module holds the mechanics
 * around that: the prompt for one chunk of beats, the parser for its JSON, and the deterministic
 * placement of recall probes over the full fact list - distance bands, changed facts asked only
 * for their latest value, one probe per beat window.
 */
import type { Scenario } from "./types.js";

/** One fact the user reveals, with what a later probe should ask and expect. */
export interface FactEntry {
  /** The fact as the beat states it (Korean sentence). */
  fact: string;
  /** Shortest string that answers a probe about it (a proper noun, number, date, title). */
  expected: string;
  /** Instruction for the simulated user to ask about it without hinting at the answer. */
  ask: string;
  /** First user turn of the beat that reveals it. */
  atUserTurn: number;
  /** `expected` of an earlier fact this one supersedes (a changed order, flat, job, clinic). */
  replaces?: string;
}

export interface GeneratedBeat {
  fromUserTurn: number;
  goal: string;
  reveal: { fact: string; expected: string; ask: string; replaces?: string }[];
}

/** Small deterministic PRNG so a scenario can be regenerated to the same probe layout. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Strips a ```json fence if the model wrapped its answer, then parses. */
export function parseBeatsJson(text: string): GeneratedBeat[] {
  let s = text.trim();
  const fence = s.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  if (fence) s = fence[1]!;
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start === -1 || end === -1) throw new Error(`no JSON object in model output: ${s.slice(0, 120)}`);
  const parsed = JSON.parse(s.slice(start, end + 1)) as { beats?: unknown };
  if (!Array.isArray(parsed.beats)) throw new Error("model output has no beats array");
  return parsed.beats.map((b: any, i: number) => {
    if (typeof b.fromUserTurn !== "number" || typeof b.goal !== "string") throw new Error(`beat ${i} malformed: ${JSON.stringify(b).slice(0, 120)}`);
    const reveal = Array.isArray(b.reveal) ? b.reveal : [];
    return {
      fromUserTurn: b.fromUserTurn,
      goal: b.goal.trim(),
      reveal: reveal
        .filter((r: any) => r && typeof r.fact === "string" && typeof r.expected === "string" && typeof r.ask === "string")
        .map((r: any) => ({ fact: r.fact.trim(), expected: r.expected.trim(), ask: r.ask.trim(), ...(typeof r.replaces === "string" && r.replaces.trim() ? { replaces: r.replaces.trim() } : {}) })),
    };
  });
}

export interface ChunkSpec {
  /** First and last user turn of the chunk (inclusive); beats start at `from` and repeat every `every`. */
  from: number;
  to: number;
  every: number;
  /** Target number of facts in the chunk and how many of them change an earlier fact. */
  facts: number;
  changes: number;
  /** How much story time the chunk should cover, for the prompt ("two", "about one and a half"). */
  yearsHint?: string;
}

/** The prompt for one chunk of beats. The model writes the story; the constraints are the mechanics the harness needs. */
export function chunkPrompt(scenario: Scenario, characterBlurb: string, known: FactEntry[], lastBeats: GeneratedBeat[], spec: ChunkSpec): { system: string; user: string } {
  const beatCount = Math.floor((spec.to - spec.from) / spec.every) + 1;
  const turns = Array.from({ length: beatCount }, (_, i) => spec.from + i * spec.every);
  const system = `You extend a long-form roleplay evaluation scenario. The scenario drives a simulated user (${scenario.persona.name}) through a story told in "beats"; a separate character model plays ${scenario.characterId} and is later graded on whether it remembers what the user revealed. You write the next stretch of the story as beats, in the same voice and format as the existing ones.

Respond with JSON only, no prose, of the form
{"beats": [{"fromUserTurn": <int>, "goal": "<Korean, 1-2 sentences>", "reveal": [{"fact": "<Korean sentence>", "expected": "<shortest answer>", "ask": "<Korean instruction>", "replaces": "<expected of an earlier fact, only when this fact changes it>"}]}]}

Rules:
- Exactly ${beatCount} beats, with fromUserTurn exactly these values in this order: ${turns.join(", ")}.
- "goal" is what ${scenario.persona.name} does or talks about that morning, written like the examples (terse, present tense, Korean). Mark time passing in some goals the way the examples do ("봄.", "겨울.", "벌써 n년째", a new year, a season) so that over the chunk about ${spec.yearsHint ?? "two"} years go by. Never state a calendar date in a goal unless the date itself is the fact.
- About ${spec.facts} facts in the chunk, 0-2 per beat, each a concrete thing a listener could be asked about later: a name, a number, a date, a title, a place, a brand, a decision. Use NEW proper nouns (never reuse one from the known facts for a different thing). Every fact must be consistent with the known facts below and with the beats before it.
- "expected" is the shortest string that answers a question about the fact: one proper noun, number, date or title, quoted the way the fact writes it.
- "ask" is an instruction to the simulated user for asking about it much later, in the style: 자기 ~가 뭐였는지 하루가 기억하는지 묻는다 (답을 말하지 않는다). It must make the question specific enough that only "expected" is the right answer.
- Exactly ${spec.changes} of the facts CHANGE something already known (a move, a job, a drink order, a clinic, a phone, a plan): the goal says the change plainly ("이 변화를 분명히 말한다"), the fact states the new value, and "replaces" is the "expected" string of the fact it supersedes (copy it exactly from the known list). For a changed fact the "ask" ends with (정답은 바뀐 뒤의 것이다).
- Some beats (about one in six) have an empty reveal: reflection, asking what ${scenario.characterId} remembers, a hard week, a birthday week where the user waits to see if the date is remembered.
- Facts are about ${scenario.persona.name}'s world (his work, family, pet, friends, habits, plans). Do not decide ${scenario.characterId}'s own life in a fact; the character model plays that.
- Keep the two people where the story has put them: the user visits the café; ${scenario.characterId} works its mornings. Life moves on realistically for someone of the user's age (${characterBlurb ? "see the character note" : ""}); pets age, friends marry, jobs change, family ages, nothing melodramatic every beat.
- Korean throughout, 존댓말 is the user's register but goals are notes, not dialogue.`;

  const knownLines = known.map((f) => `- [turn ${f.atUserTurn}] ${f.fact} (expected: ${f.expected}${f.replaces ? `; replaces: ${f.replaces}` : ""})`).join("\n");
  const examples = lastBeats
    .map((b) => `- fromUserTurn ${b.fromUserTurn}: ${b.goal}${b.reveal.length ? `\n  reveal: ${b.reveal.map((r) => r.fact).join(" / ")}` : "\n  reveal: (none)"}`)
    .join("\n");
  const user = `Scenario: ${scenario.id}. User persona: ${scenario.persona.name}${scenario.persona.description ? ` - ${scenario.persona.description}` : ""}.
User style: ${scenario.userStyle.trim()}
${characterBlurb ? `Character note (${scenario.characterId}): ${characterBlurb}\n` : ""}
KNOWN FACTS so far (turn where revealed):
${knownLines}

THE LAST BEATS (voice and format to match):
${examples}

Write the beats for user turns ${spec.from} to ${spec.to}. JSON only.`;
  return { system, user };
}

export interface PlaceOptions {
  /** Probes go into user turns in [from, to]. */
  from: number;
  to: number;
  /** Beats start every `every` turns from `from`; a probe sits `offset` turns into a beat. */
  every: number;
  offset: number;
  /** Probes per distance band, in band order. */
  bands: { min: number; max: number; count: number }[];
  /** A fact is askable only this many turns after the beat that reveals it starts. */
  margin: number;
  /** How many changed facts (ones with `replaces`) to prefer before filling the bands. */
  preferChanged: number;
  seed: number;
  /** Facts never to ask about (an age that moves with time, "this year's best moment"). */
  exclude?: (f: FactEntry) => boolean;
}

export interface PlacedProbe {
  atUserTurn: number;
  ask: string;
  expected: string;
  /** Turn the fact was revealed, for the report. */
  revealedAt: number;
  distance: number;
  changed: boolean;
}

/**
 * Places probes over the facts. A fact that is later replaced is askable only until the
 * replacing fact's beat; the replacing fact is askable after its own margin. Each fact is asked at
 * most once here, each probe turn is used once, and the turns spread over the bands given.
 */
export function placeProbes(allFacts: FactEntry[], opts: PlaceOptions): PlacedProbe[] {
  const rng = mulberry32(opts.seed);
  // Two facts with the same answer make an ambiguous probe (two "10월"s), unless the earlier one
  // had already been replaced when the later was revealed (아메리카노 → 라떼 → … → 아메리카노).
  const facts = allFacts.filter((f, i) => {
    if (opts.exclude?.(f)) return false;
    return !allFacts.some((e, j) => j < i && e.expected === f.expected && !allFacts.some((g) => g.replaces === e.expected && g.atUserTurn > e.atUserTurn && g.atUserTurn <= f.atUserTurn));
  });
  const shuffle = <T>(a: T[]): T[] => {
    const b = a.slice();
    for (let i = b.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [b[i], b[j]] = [b[j]!, b[i]!];
    }
    return b;
  };
  // Until when is each fact askable? Until a later fact replaces it.
  const until = new Map<FactEntry, number>();
  for (const f of facts) {
    const later = facts.filter((g) => g.replaces === f.expected && g.atUserTurn > f.atUserTurn).map((g) => g.atUserTurn);
    until.set(f, later.length ? Math.min(...later) : Infinity);
  }
  const askable = (f: FactEntry, turn: number) => turn >= f.atUserTurn + opts.margin && turn < until.get(f)!;

  const slots: number[] = [];
  for (let t = opts.from + opts.offset; t <= opts.to; t += opts.every) slots.push(t);
  const freeSlots = new Set(slots);
  const used = new Set<FactEntry>();
  const out: PlacedProbe[] = [];
  const place = (f: FactEntry, turn: number) => {
    out.push({ atUserTurn: turn, ask: f.ask, expected: f.expected, revealedAt: f.atUserTurn, distance: turn - f.atUserTurn, changed: !!f.replaces });
    used.add(f);
    freeSlots.delete(turn);
  };
  const inBand = (d: number, b: { min: number; max: number }) => d >= b.min && d <= b.max;

  // 1. Changed facts first: their latest value is the whole point of asking.
  const changed = shuffle(facts.filter((f) => f.replaces));
  for (const f of changed) {
    if (out.filter((p) => p.changed).length >= opts.preferChanged) break;
    const candidates = shuffle([...freeSlots].filter((t) => askable(f, t) && opts.bands.some((b) => inBand(t - f.atUserTurn, b))));
    if (candidates.length) place(f, candidates[0]!);
  }
  // 2. Fill each band. Long distances are the scarce ones, so fill from the last band backwards.
  const bandOrder = opts.bands.map((b, i) => ({ ...b, i })).sort((a, b) => b.min - a.min);
  for (const band of bandOrder) {
    let have = out.filter((p) => inBand(p.distance, band)).length;
    const pool = shuffle(facts.filter((f) => !used.has(f)));
    for (const f of pool) {
      if (have >= band.count) break;
      const candidates = shuffle([...freeSlots].filter((t) => askable(f, t) && inBand(t - f.atUserTurn, band)));
      if (!candidates.length) continue;
      place(f, candidates[0]!);
      have++;
    }
  }
  return out.sort((a, b) => a.atUserTurn - b.atUserTurn);
}
