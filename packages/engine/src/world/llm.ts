import type { LLMProvider } from "../providers/types.js";
import type { CharacterCard, Lang, MemoryState, TokenUsage, Turn, UserPersona } from "../types.js";
import { SLOTS, type Slot, type WorldEvent, type WorldTask } from "./types.js";
import type { World } from "./world.js";

/**
 * The model-side operations on the world: reading commitments out of a conversation, deciding what
 * happened when a character went somewhere alone, and playing a scene between two characters.
 * Each returns structured JSON that the World records; the character prompt never sees the model's
 * prose except through facts and the optional scene text.
 */

export interface WorldLLM {
  provider: LLMProvider;
  model?: string;
}

export interface WorldCallUsage extends TokenUsage {
  calls: number;
  model?: string;
}

const LANG_NAME: Record<Lang, string> = { ko: "Korean", ja: "Japanese", en: "English" };

export const emptyWorldUsage = (): WorldCallUsage => ({ promptTokens: 0, completionTokens: 0, cachedPromptTokens: 0, calls: 0 });

function add(u: WorldCallUsage, r: { usage?: TokenUsage; model: string }) {
  if (r.usage) {
    u.promptTokens += r.usage.promptTokens;
    u.completionTokens += r.usage.completionTokens;
    u.cachedPromptTokens! += r.usage.cachedPromptTokens ?? 0;
  }
  u.calls++;
  u.model = r.model;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function safeJson(text: string): any | null {
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) return null;
  try {
    return JSON.parse(m[0]);
  } catch {
    return null;
  }
}

const strs = (x: unknown): string[] => (Array.isArray(x) ? x.filter((s): s is string => typeof s === "string" && s.trim().length > 0).map((s) => s.trim()) : []);

const placesList = (world: World) =>
  world.state.places.map((p) => `- ${p.id}: ${p.name}${p.hours ? ` (${p.hours})` : ""}${p.note ? ` - ${p.note}` : ""}${world.condition(p.id) ? ` [RIGHT NOW: ${world.condition(p.id)}]` : ""}`).join("\n");

const placeLine = (world: World, placeId: string) => {
  const p = world.place(placeId);
  return `${p.name}${p.hours ? ` (${p.hours})` : ""}${p.note ? `. ${p.note}` : ""}${world.condition(placeId) ? `\nRIGHT NOW (overrides the usual): ${world.condition(placeId)}` : ""}`;
};

/**
 * Read the commitments the character made in a conversation: what they agreed to do, where, when.
 * Only things the character actually agreed to; a request the character declined is not a task.
 */
export async function extractTasks(
  llm: WorldLLM,
  args: { world: World; card: CharacterCard; lang: Lang; persona: UserPersona; turns: Turn[]; characterId: string },
  usage: WorldCallUsage = emptyWorldUsage(),
): Promise<Omit<WorldTask, "id" | "status" | "createdDay">[]> {
  const { world, card, lang, persona, turns, characterId } = args;
  const name = card.name[lang] ?? card.id;
  const transcript = turns.map((t) => `${t.role === "user" ? persona.name : name}: ${t.content}`).join("\n");
  const pending = world.state.tasks.filter((t) => t.status === "pending" && t.assignee === characterId);
  const system = [
    `You read a roleplay conversation between ${persona.name} (the user) and the character ${name} and list what ${name} AGREED to do later, away from this conversation.`,
    pending.length ? `ALREADY RECORDED (do not list these again, even if the conversation mentions them):\n${pending.map((t) => `- ${t.what} @${t.where ?? "?"} day ${t.dueDay}`).join("\n")}` : "",
    `Today is day ${world.day}. Places in this world (use the id):\n${placesList(world)}`,
    `Output JSON only: {"tasks": [{"what": "one line: what ${name} will do and for whom, with every concrete detail needed to do it (titles, names, what to check), written in ${LANG_NAME[lang]}", "where": "place id or null", "dueDay": <day number>, "dueSlot": "morning|afternoon|evening|night"}]}`,
    `Rules: include only NEW commitments ${name} clearly accepted (a request ${name} declined or left open is not a task). "tomorrow" is day ${world.day + 1}; an unspecified day is day ${world.day + 1}. A task is something ${name} does AWAY from ${persona.name}, where ${persona.name} cannot see it: going somewhere, asking someone, fetching or checking something. Anything ${name} will do the next time ${persona.name} is present (hand something over, tell them a result, make their drink a certain way, bring up a topic, open the shop) is NOT a task - it happens in the conversation itself. Things done inside this same conversation are not tasks. Empty list if none.`,
  ]
    .filter(Boolean)
    .join("\n");
  const res = await llm.provider.generate(
    [
      { role: "system", content: system },
      { role: "user", content: `CONVERSATION:\n${transcript}` },
    ],
    { model: llm.model, temperature: 0.1, maxTokens: 600 },
  );
  add(usage, res);
  const parsed = safeJson(res.text);
  const list: unknown[] = Array.isArray(parsed?.tasks) ? parsed.tasks : [];
  const out: Omit<WorldTask, "id" | "status" | "createdDay">[] = [];
  for (const t of list) {
    if (!t || typeof t !== "object") continue;
    const o = t as Record<string, unknown>;
    if (typeof o.what !== "string" || !o.what.trim()) continue;
    const where = typeof o.where === "string" && world.state.places.some((p) => p.id === o.where) ? o.where : undefined;
    const dueDay = typeof o.dueDay === "number" && o.dueDay >= world.day ? Math.floor(o.dueDay) : world.day + 1;
    const dueSlot = SLOTS.includes(o.dueSlot as Slot) ? (o.dueSlot as Slot) : undefined;
    // Being at one's own routine place is not an errand ("open the café at 7").
    if (where && dueSlot && world.character(characterId).routine[dueSlot] === where) continue;
    out.push({ assignee: characterId, what: o.what.trim(), where, dueDay, dueSlot, from: persona.name });
  }
  return out;
}

export type TaskOutcome = "done" | "partial" | "failed";

export interface ResolvedTask {
  event: Omit<WorldEvent, "id">;
  /** "failed" = could not be attempted (the place was shut, the person was out); the caller retries it the next day, once. */
  outcome: TaskOutcome;
}

/**
 * Decide what happened when a character carried out a task alone. One call, JSON out. The world's
 * rules keep it mundane; the details are what the character will later remember as a witness.
 */
export async function resolveTask(
  llm: WorldLLM,
  args: { world: World; card: CharacterCard; lang: Lang; task: WorldTask; memory?: MemoryState; avoid?: string[] },
  usage: WorldCallUsage = emptyWorldUsage(),
): Promise<ResolvedTask> {
  const { world, card, lang, task } = args;
  const name = card.name[lang] ?? card.id;
  const slot = task.dueSlot ?? "afternoon";
  const where = task.where ?? world.whereIs(task.assignee, task.dueDay, slot) ?? world.state.places[0]!.id;
  const place = world.place(where);
  const system = [
    `You decide what happened, off-screen, when ${name} did an errand. You are the world, not the character: plain, concrete, mundane. Write in ${LANG_NAME[lang]}.`,
    `# WORLD\n${card.world.summary}\nRules:\n${card.world.rules.map((r) => `- ${r}`).join("\n")}`,
    `# WHO\n${name}: ${card.identity.role} ${card.identity.personality.join(", ")}.`,
    `# WHERE\n${placeLine(world, where)}`,
    `# WHEN\nDay ${task.dueDay} (${world.weekdayOf(task.dueDay)}), ${slot}.`,
    args.memory?.facts.length ? `# WHAT ${name} KNOWS\n${args.memory.facts.map((f) => `- ${f.text}`).join("\n")}` : "",
    `Output JSON only: {"outcome": "done|partial|failed", "result": "ONE sentence, past tense: what is now true (what was obtained, learned or not)", "details": ["2 to 4 short concrete things ${name} saw or heard there: what the other person said (brief quote), what the object looked like, a small complication - each one a fact, no feelings"]}`,
    "Rules: nothing dramatic (no accidents, no coincidences, no new love interests). Most errands succeed plainly; at most one small, ordinary complication. \"failed\" only when the errand could not be attempted at all (the place was shut, the person was away): it is tried again the next day. Do not invent facts about the user. Name people at the place only if the place note names them; otherwise call them by role (the owner). Date things by weekday or \"day N\", never by 어제/오늘/내일: these lines are read days later.",
  ]
    .filter(Boolean)
    .join("\n\n");
  const res = await llm.provider.generate(
    [
      { role: "system", content: system },
      { role: "user", content: `TASK: ${task.what}${args.avoid?.length ? `\n\nA previous attempt contradicted the task; do not repeat these:\n${args.avoid.map((p) => `- ${p}`).join("\n")}` : ""}` },
    ],
    { model: llm.model, temperature: 0.7, maxTokens: 500 },
  );
  add(usage, res);
  const parsed = safeJson(res.text) ?? {};
  const details = strs(parsed.details).slice(0, 4);
  const result = typeof parsed.result === "string" && parsed.result.trim() ? parsed.result.trim() : `${name} went to ${place.name}: ${task.what}`;
  const outcome: TaskOutcome = parsed.outcome === "failed" || parsed.outcome === "partial" ? parsed.outcome : "done";
  return {
    outcome,
    event: {
      day: task.dueDay,
      slot,
      where,
      who: [task.assignee],
      scope: "personal",
      result,
      details,
      knownBy: { [task.assignee]: { source: "witnessed" } },
      origin: "errand",
      taskId: task.id,
    },
  };
}

/**
 * Close an errand with what happened. A failed attempt (the shop was shut on its day off) is still
 * remembered, but the task stays pending and moves to the next day, once; a second failure ends it.
 * (T6 first run: the Tuesday errand failed, was marked done, and the span generator then "finished"
 * the user's errand its own way.)
 */
function settle(world: World, group: WorldTask[], r: ResolvedTask): WorldEvent {
  if (r.outcome === "failed" && (group[0]!.attempts ?? 0) < 1) {
    for (const t of group) {
      t.attempts = (t.attempts ?? 0) + 1;
      t.dueDay += 1;
    }
    const attempt = { ...r.event };
    delete attempt.taskId; // not linked: the task is still pending
    return world.record(attempt);
  }
  const ev = world.record(r.event);
  for (const t of group) {
    if (t.status === "pending") t.status = "done";
    t.eventId = ev.id;
  }
  return ev;
}

export interface SceneParty {
  id: string;
  card: CharacterCard;
  memory?: MemoryState;
  task?: WorldTask;
  /** What this party knows and the other does not, since they last met (one line each). */
  news?: string[];
}

export interface SceneResult {
  event: Omit<WorldEvent, "id">;
  /** Per character id: facts they learned from the other, to be recorded as `told`. */
  learned: Record<string, string[]>;
}

/**
 * Two characters at the same place in the same slot. One call writes a short scene and reports it
 * as data: what happened (both witnessed it) and what each told the other (travels as `told`).
 */
export async function runScene(
  llm: WorldLLM,
  args: { world: World; lang: Lang; a: SceneParty; b: SceneParty; where: string; day: number; slot: Slot; persona?: UserPersona; avoid?: string[]; userBackDay?: number },
  usage: WorldCallUsage = emptyWorldUsage(),
): Promise<SceneResult> {
  const { world, lang, a, b, where, day, slot } = args;
  const nameOf = (x: SceneParty) => x.card.name[lang] ?? x.card.id;
  const place = world.place(where);
  const who = (x: SceneParty) =>
    [
      `## ${nameOf(x)}`,
      `${x.card.identity.role} ${x.card.identity.personality.join(", ")}.`,
      `Values: ${x.card.identity.values.join(", ")}. Secrets (never said outright): ${x.card.identity.secrets.join("; ") || "(none)"}`,
      `Speech: ${x.card.style[lang]?.summary ?? ""}`,
      x.task ? `Here because: ${x.task.what}` : `Here as usual (${place.name}).`,
      x.memory?.facts.length
        ? `What ${nameOf(x)} knows (each line says how):\n${x.memory.facts.map((f) => `- ${f.text} [${f.source}${f.by ? `:${f.by}` : ""}]`).join("\n")}`
        : `${nameOf(x)} knows nothing special.`,
      x.news?.length
        ? `What ${nameOf(x)} has to tell ${nameOf(x === a ? b : a)} (new since they last met; ${nameOf(x)} brings these up unless they are private or secret, in their own words):\n${x.news.map((l) => `- ${l}`).join("\n")}`
        : "",
    ]
      .filter(Boolean)
      .join("\n");
  const system = [
    `You write what happened when two characters met, off-screen, and then report it as data. Write in ${LANG_NAME[lang]}.`,
    `# WORLD\n${a.card.world.summary}\nRules:\n${a.card.world.rules.map((r) => `- ${r}`).join("\n")}`,
    `# WHERE / WHEN\n${placeLine(world, where)}. Day ${day} (${world.weekdayOf(day)}), ${slot}.`,
    who(a),
    who(b),
    args.persona
      ? `# THE USER\n${args.persona.name} is not present${args.userBackDay ? ` and will not be back before day ${args.userBackDay}` : ""}. ${args.persona.description ?? ""} Characters may mention ${args.persona.name} only from what they know above. Nothing can be handed to, told to or arranged with ${args.persona.name} before they are back, so do not plan it for a nearer day.`
      : "",
    `Output JSON only: {"scene": "the scene, 6 to 14 short lines in the *action* \\"speech\\" format, in character for both", "result": "ONE sentence, past tense: what is now true after this meeting", "details": ["2 to 4 concrete things both would remember"], "told": {"${a.id}": ["facts ${nameOf(b)} told ${nameOf(a)} that ${nameOf(a)} did not know - one line each, attributed ('${nameOf(b)} said ...'); start the line with [about ${args.persona?.name ?? "the user"}] when it concerns ${args.persona?.name ?? "the user"}'s belongings or a promise to them"], "${b.id}": ["facts ${nameOf(a)} told ${nameOf(b)} that ${nameOf(b)} did not know, same rules"]}}`,
    "Rules: each character says only what they know and would say given their personality and secrets (a reserved character shares little). Nothing dramatic. If a task brought one of them, the scene resolves it plainly. In \"result\", \"details\" and \"told\", date things by weekday or \"day N\", never by 어제/오늘/내일 (yesterday/today/tomorrow): these lines are read days later.",
  ]
    .filter(Boolean)
    .join("\n\n");
  const res = await llm.provider.generate(
    [
      { role: "system", content: system },
      { role: "user", content: `Write the scene and the report.${args.avoid?.length ? `\n\nA previous attempt contradicted the errand; do not repeat these:\n${args.avoid.map((p) => `- ${p}`).join("\n")}` : ""}` },
    ],
    { model: llm.model, temperature: 0.8, maxTokens: 1400 },
  );
  add(usage, res);
  const parsed = safeJson(res.text) ?? {};
  const result = typeof parsed.result === "string" && parsed.result.trim() ? parsed.result.trim() : `${nameOf(a)} and ${nameOf(b)} met at ${place.name}.`;
  const learned: Record<string, string[]> = { [a.id]: strs(parsed.told?.[a.id]), [b.id]: strs(parsed.told?.[b.id]) };
  const task = a.task ?? b.task;
  return {
    event: {
      day,
      slot,
      where,
      who: [a.id, b.id],
      scope: "personal",
      result,
      details: strs(parsed.details).slice(0, 4),
      knownBy: { [a.id]: { source: "witnessed" }, [b.id]: { source: "witnessed" } },
      ...(typeof parsed.scene === "string" ? { scene: parsed.scene } : {}),
      origin: "scene",
      ...(task ? { taskId: task.id } : {}),
    },
    learned,
  };
}

/**
 * Check a decided event against the task that caused it: counts, titles, names and what was asked
 * must not be contradicted (one scene turned a two-volume set into volumes 1 and 3, and both
 * characters then repeated it faithfully - docs/10 T2). One cheap call; the caller retries once.
 */
export async function verifyEvent(
  llm: WorldLLM,
  args: { task: WorldTask; event: Omit<WorldEvent, "id"> },
  usage: WorldCallUsage = emptyWorldUsage(),
): Promise<{ ok: boolean; problems: string[] }> {
  const { task, event } = args;
  const system = [
    "You check a story event against the errand that caused it. Only hard contradictions count: a different count, title, name, object or outcome than the errand states or implies (a two-volume set has no volume 3; a book asked for by name is not a different book). Small complications (not in stock yet, a worn cover) are fine. Missing detail is fine.",
    'Output JSON only: {"ok": true|false, "problems": ["one line each, only for contradictions"]}',
  ].join("\n");
  const user = `ERRAND: ${task.what}\n\nRESULT: ${event.result}\nDETAILS:\n${event.details.map((d) => `- ${d}`).join("\n")}${event.scene ? `\nSCENE:\n${event.scene}` : ""}`;
  const res = await llm.provider.generate([{ role: "system", content: system }, { role: "user", content: user }], { model: llm.model, temperature: 0, maxTokens: 300 });
  add(usage, res);
  const parsed = safeJson(res.text) ?? {};
  const problems = strs(parsed.problems);
  return { ok: parsed.ok !== false && problems.length === 0, problems };
}

/** Shared context for the world's own events: places with their current state, who is where. */
function worldSheet(world: World, deps: Pick<AdvanceDeps, "cards" | "lang">): string {
  const chars = Object.values(world.state.characters)
    .map((c) => {
      const card = deps.cards[c.id];
      const who = card ? `${card.identity.role} ${card.identity.personality.slice(0, 3).join(", ")}` : (c.sketch ?? "");
      const routine = SLOTS.map((s) => `${s}: ${c.routine[s] ? world.place(c.routine[s]!).name : "-"}`).join(", ");
      return `- ${c.name}${c.aka?.length ? ` (also called ${c.aka.join(", ")})` : ""} [id ${c.id}, ${c.tier}]: ${who}. Usually ${routine}.`;
    })
    .join("\n");
  const recent = world.state.ledger
    .filter((e) => e.origin !== "chat")
    .slice(-8)
    .map((e) => `- day ${e.day} ${e.slot} @${e.where}: ${e.result}`)
    .join("\n");
  return `# PLACES (id: name)\n${placesList(world)}\n\n# PEOPLE (name [id, tier]; in result and details write people by NAME, never by id)\n${chars}\n\n# RECENTLY (newest last)\n${recent || "(nothing yet)"}`;
}

/** The user is away. The world may not settle the user's affairs for them (T6: a span had the aunt mail the user's books). */
const userAway = (deps: Pick<AdvanceDeps, "persona" | "userBackDay">): string =>
  deps.persona
    ? `# THE USER\n${deps.persona.name} is away${deps.userBackDay ? ` until day ${deps.userBackDay}` : ""} and cannot appear in these events, not even as "a regular" or "a customer". Do not decide anything about ${deps.persona.name}'s belongings, orders or errands: nothing of theirs is handed over, sent, sold or moved while ${deps.persona.name} is away (T8: a world day handed the user's books to "a regular"); those resolve separately and stand as RECENTLY says.`
    : "";

function parseGenerated(world: World, parsed: unknown, dayDefault: number, maxEvents: number, origin: "world" | "span"): Omit<WorldEvent, "id">[] {
  const list: unknown[] = Array.isArray((parsed as { events?: unknown })?.events) ? ((parsed as { events: unknown[] }).events) : [];
  const out: Omit<WorldEvent, "id">[] = [];
  for (const raw of list) {
    if (!raw || typeof raw !== "object") continue;
    const o = raw as Record<string, unknown>;
    const where = typeof o.where === "string" && world.state.places.some((p) => p.id === o.where) ? o.where : undefined;
    if (!where || typeof o.result !== "string" || !o.result.trim()) continue;
    const slot = SLOTS.includes(o.slot as Slot) ? (o.slot as Slot) : "afternoon";
    const day = typeof o.day === "number" ? Math.floor(o.day) : dayDefault;
    const who = strs(o.who).filter((id) => world.state.characters[id]);
    const scope = o.scope === "regional" || o.scope === "world" ? o.scope : "personal";
    const knownBy: WorldEvent["knownBy"] = {};
    for (const id of who) knownBy[id] = { source: "witnessed" };
    const effects: Record<string, string | null> = {};
    if (o.effects && typeof o.effects === "object") {
      for (const [pid, cond] of Object.entries(o.effects as Record<string, unknown>)) {
        if (!world.state.places.some((p) => p.id === pid)) continue;
        if (cond === null || (typeof cond === "string" && /^(null|none|usual|back to usual|-)?$/i.test(cond.trim()))) effects[pid] = null;
        else if (typeof cond === "string") effects[pid] = cond.trim();
      }
    }
    out.push({ day, slot, where, who, scope, result: o.result.trim(), details: strs(o.details).slice(0, 3), knownBy, origin, ...(Object.keys(effects).length ? { effects } : {}) });
    if (out.length >= maxEvents) break;
  }
  return out;
}

/**
 * The world's own day (docs/10 T6): one call proposes up to K small events for `day`, consistent
 * with the places' current conditions and continuing what is already in the ledger. A quiet day
 * is allowed and costs only this call. Characters present become witnesses.
 */
export async function generateDay(world: World, deps: AdvanceDeps, day: number, usage: WorldCallUsage): Promise<Omit<WorldEvent, "id">[]> {
  const k = deps.eventsPerDay ?? 0;
  if (k <= 0) return [];
  const system = [
    `You are the world of a quiet, realistic slice-of-life story (${LANG_NAME[deps.lang]}). Decide what, if anything, happened on day ${day} (${world.weekdayOf(day)}) that is worth remembering. Nothing dramatic: deliveries, weather, a shop closing early, a regular not showing up, a repair finished, a small favour. Continue threads already in RECENTLY (a closed place reopens when it said it would; set its condition back with null). A regional event has consequences at the places it touches (a water cut: a café cannot make espresso and closes early or serves bottled drinks; a stall stays shut): write them as that day's events with effects, and undo the effects when the cause ends. Consequences, not a chain reaction: stay under the cap. Respect each place's RIGHT NOW condition.`,
    worldSheet(world, deps),
    userAway(deps),
    `Output JSON only: {"events": [{"slot": "morning|afternoon|evening|night", "where": "<place id>", "who": ["<character ids who were there>"], "scope": "personal|regional", "result": "ONE sentence, past tense, dated by weekday or day number (never 어제/오늘/내일), in ${LANG_NAME[deps.lang]}", "details": ["0-3 concrete things a witness would remember"], "effects": {"<place id>": "new condition text, or null when back to usual"}}]}. At most ${k} event(s); an empty list is a fine answer for an ordinary day.`,
  ]
    .filter(Boolean)
    .join("\n\n");
  const res = await deps.llm.provider.generate([{ role: "system", content: system }, { role: "user", content: `Day ${day}.` }], { model: deps.llm.model, temperature: 0.8, maxTokens: 700 });
  add(usage, res);
  return parseGenerated(world, safeJson(res.text), day, k, "world");
}

/**
 * A compressed stretch of days (docs/10 "결정은 즉시, 장면은 나중에"): one call decides the few
 * events of a whole span, each dated. Used for the older part of a long gap so a six-month absence
 * is a handful of calls, not 180.
 */
export async function generateSpan(world: World, deps: AdvanceDeps, from: number, to: number, usage: WorldCallUsage): Promise<Omit<WorldEvent, "id">[]> {
  const k = deps.eventsPerSpan ?? 3;
  const system = [
    `You are the world of a quiet, realistic slice-of-life story (${LANG_NAME[deps.lang]}). Days ${from} to ${to} (${world.weekdayOf(from)} to ${world.weekdayOf(to)}, ${to - from + 1} days) pass with nobody watching. Decide the few things worth remembering from that whole stretch. Nothing dramatic; the kind of change a regular notices after a month away: a menu changed, a shop closed for a week and reopened, someone took a trip, a repair, a new regular. Continue threads in RECENTLY and close them when they would close (set a condition back with null). Do not repeat a kind of event already in RECENTLY (one menu change is a thread, six are a loop): a month brings different things. Respect each place's RIGHT NOW condition.`,
    worldSheet(world, deps),
    userAway(deps),
    `Output JSON only: {"events": [{"day": <${from}..${to}>, "slot": "morning|afternoon|evening|night", "where": "<place id>", "who": ["<character ids who were there>"], "scope": "personal|regional", "result": "ONE sentence, past tense, dated by weekday or day number, in ${LANG_NAME[deps.lang]}", "details": ["0-3 concrete things a witness would remember"], "effects": {"<place id>": "new condition text, or null when back to usual"}}]}. At most ${k} events for the whole span, in day order.`,
  ]
    .filter(Boolean)
    .join("\n\n");
  const res = await deps.llm.provider.generate([{ role: "system", content: system }, { role: "user", content: `Days ${from}-${to}.` }], { model: deps.llm.model, temperature: 0.8, maxTokens: 900 });
  add(usage, res);
  return parseGenerated(world, safeJson(res.text), from, k, "span").map((e) => ({ ...e, day: Math.min(Math.max(e.day, from), to) }));
}

export interface CatchUpResult {
  events: WorldEvent[];
  /** Days lived one by one (with scenes), days folded into spans. */
  dayByDay: number;
  inSpans: number;
}

/**
 * Move the world forward `days` days while the user is away. The most recent `recentDays` are
 * lived day by day (tasks, routine scenes, the world's own events); everything before that is
 * decided in spans of `spanDays`, oldest first, so cost grows with the number of spans, not days.
 */
export async function catchUp(world: World, deps: AdvanceDeps & { recentDays?: number; spanDays?: number }, days: number): Promise<CatchUpResult> {
  const usage = deps.usage ?? emptyWorldUsage();
  const recent = Math.max(0, deps.recentDays ?? 7);
  const spanDays = Math.max(1, deps.spanDays ?? 30);
  const events: WorldEvent[] = [];
  const first = world.day; // the day that is ending now
  const spanUntil = first + days - recent - 1; // last day decided in spans
  let d = first;
  let inSpans = 0;
  while (d <= spanUntil) {
    const to = Math.min(d + spanDays - 1, spanUntil);
    for (const e of await generateSpan(world, deps, d, to, usage)) events.push(world.record(e));
    spreadRumours(world, deps, world.state.ledger.filter((e) => e.day >= d && e.day <= to));
    // tasks due inside the span are lived like any other day's (scene when the other main is there)
    for (let x = d; x <= to; x++) {
      world.setClock(x, "night");
      const met = new Set<string>();
      for (const group of dueGroups(world, x).values()) {
        const ev = await liveTask(world, deps, group, usage, met);
        if (ev) events.push(ev);
      }
    }
    inSpans += to - d + 1;
    d = to + 1;
  }
  world.setClock(d, "morning");
  let dayByDay = 0;
  while (world.day < first + days) {
    events.push(...(await advanceDay(world, { ...deps, usage })));
    dayByDay++;
  }
  return { events, dayByDay, inSpans };
}

export interface AdvanceDeps {
  llm: WorldLLM;
  /** Model for verifyEvent (default: `llm`). A small model is enough. */
  verifier?: WorldLLM | null;
  lang: Lang;
  cards: Record<string, CharacterCard>;
  /** Live session memories; what the ledger says a character knows is added on top. */
  memories?: Record<string, MemoryState>;
  persona?: UserPersona;
  usage?: WorldCallUsage;
  /** Routine scenes per day between co-located main characters who have news for each other (default 1; 0 = off). */
  scenesPerDay?: number;
  /** The world's own events per day (generateDay), default 0 = the world only moves through tasks. */
  eventsPerDay?: number;
  /** Events per compressed span (generateSpan), default 3. */
  eventsPerSpan?: number;
  /** Scenes get a "what to bring up" list: the events each party knows and the other does not (docs/10 T7). Default off. */
  sceneNews?: boolean;
  /** The day the user is back (known when a gap is being lived). Scenes and world days then know nothing of the user's can be settled before it. */
  userBackDay?: number;
  /** The last conversation each character had with the user (for decideAttention): what was said is what connects them. */
  recentChats?: RecentChat[];
}

export interface RecentChat {
  characterId: string;
  day: number;
  /** The conversation's text, "Name: line" per turn, already trimmed. */
  transcript: string;
}

export interface AttentionDecision {
  /** Characters who live in full while the user is away. */
  main: string[];
  why: Record<string, string>;
}

/** Set the tiers: `main` live in full, every other character with a card lives in the background (a few lines, like any extra). */
export function applyAttention(world: World, cards: Record<string, CharacterCard>, main: string[]): void {
  for (const c of Object.values(world.state.characters)) {
    if (!cards[c.id]) continue;
    if (main.includes(c.id)) c.tier = "main";
    else {
      c.tier = "named";
      c.sketch ??= cards[c.id]!.identity.role;
    }
  }
}

/**
 * Who lives in full while the user is away (docs/10 T10). Not a rule: one call reads what the user
 * said lately, the pending errands and the user's things, and names the characters whose days must
 * be lived (meetings written, errands run) because what happens to them can reach the user on
 * return. The rest live in the background at no cost: they still hear what goes around town, and
 * nothing the user could have known is lost. The only rule is the cap.
 */
export async function decideAttention(
  world: World,
  deps: AdvanceDeps,
  args: { daysAway: number; cap: number },
  usage: WorldCallUsage = emptyWorldUsage(),
): Promise<AttentionDecision> {
  const user = deps.persona?.name ?? "the user";
  const backDay = world.day + args.daysAway;
  const chars = Object.values(world.state.characters)
    .filter((c) => deps.cards[c.id])
    .map((c) => {
      const card = deps.cards[c.id]!;
      const last = world.state.lastSeen[c.id];
      const routine = SLOTS.map((s) => `${s}: ${c.routine[s] ? world.place(c.routine[s]!).name : "-"}`).join(", ");
      return `- ${c.name}${c.aka?.length ? ` (also called ${c.aka.join(", ")})` : ""} [id ${c.id}]: ${card.identity.role}. Usually ${routine}. Last spoke with ${user}: ${last === undefined ? "never" : `day ${last} (${world.day - last} days ago)`}.`;
    })
    .join("\n");
  const tasks = world.state.tasks
    .filter((t) => t.status === "pending")
    .map((t) => `- ${world.character(t.assignee).name} [id ${t.assignee}]: ${t.what} @${t.where ? world.place(t.where).name : "?"} day ${t.dueDay}`)
    .join("\n");
  const things = Object.entries(world.state.user)
    .map(([k, v]) => `- ${k}: ${v}`)
    .join("\n");
  const chats = (deps.recentChats ?? [])
    .map((c) => `## ${world.character(c.characterId).name} [id ${c.characterId}], day ${c.day}\n${c.transcript}`)
    .join("\n\n");
  const system = [
    `You run the off-screen world of a quiet slice-of-life story while ${user} is away for ${args.daysAway} days (back on day ${backDay}). Living a character's days in full costs money: their meetings are written and their errands run. Living them in the background costs nothing: they are still heard of, and still hear what goes around town.`,
    `Choose who lives in full: the characters whose days can reach ${user} on return - through an errand ${user} asked for or that passes through them, a plan ${user} made with or about them, something of ${user}'s in their keeping, or because ${user} will plainly be seeing them soon. Leave out people ${user} only passes by. At most ${args.cap}; fewer is better when nothing connects a character to ${user}.`,
    `# CHARACTERS\n${chars}`,
    `# PENDING ERRANDS\n${tasks || "(none)"}`,
    `# ${user.toUpperCase()}'S THINGS IN THE WORLD\n${things || "(nothing recorded)"}`,
    chats ? `# RECENT CONVERSATIONS WITH ${user.toUpperCase()}\n${chats}` : "",
    `Output JSON only: {"main": ["id", ...], "why": {"id": "one short line: what connects them to ${user}"}}`,
  ]
    .filter(Boolean)
    .join("\n\n");
  const res = await deps.llm.provider.generate([{ role: "system", content: system }, { role: "user", content: "Decide." }], { model: deps.llm.model, temperature: 0, maxTokens: 400 });
  add(usage, res);
  const parsed = safeJson(res.text) ?? {};
  const known = new Set(Object.values(world.state.characters).filter((c) => deps.cards[c.id]).map((c) => c.id));
  const main = strs(parsed.main).filter((id) => known.has(id)).slice(0, Math.max(1, args.cap));
  const why: Record<string, string> = {};
  if (parsed.why && typeof parsed.why === "object") for (const [k, v] of Object.entries(parsed.why)) if (known.has(k) && typeof v === "string") why[k] = v;
  // an unreadable answer changes nothing: the tiers stay as they are
  if (main.length) applyAttention(world, deps.cards, main);
  return { main: main.length ? main : Object.values(world.state.characters).filter((c) => c.tier === "main" && deps.cards[c.id]).map((c) => c.id), why };
}

/** What a character knows: live memory facts plus everything the ledger says reached them. */
export function knowledgeOf(world: World, id: string, memories?: Record<string, MemoryState>): MemoryState {
  const base = memories?.[id];
  const have = new Set((base?.facts ?? []).map((f) => f.text));
  const fromLedger = world.factsFor(id, world.state.ledger.filter((e) => e.knownBy[id])).filter((f) => !have.has(f.text));
  return {
    summary: base?.summary ?? "",
    facts: [...(base?.facts ?? []), ...fromLedger],
    relationship: base?.relationship ?? { stage: "initial", notes: [] },
    coveredUpTo: base?.coveredUpTo ?? -1,
  };
}

/**
 * True when one of the two knows a ledger event the other does not, newer than their last meeting
 * and already past by (day, slot): an evening event is not news in that day's afternoon.
 */
/** Events `a` knows and `b` does not, since the two last met, up to (not including) `day`/`slot`. */
export function newsFor(world: World, a: string, b: string, day: number, slot: Slot): WorldEvent[] {
  const ledger = world.state.ledger;
  const before = (e: WorldEvent) => e.day < day || (e.day === day && SLOTS.indexOf(e.slot) < SLOTS.indexOf(slot));
  let lastMet = -1;
  ledger.forEach((e, i) => {
    if (e.who.includes(a) && e.who.includes(b)) lastMet = i;
  });
  // A rumour is not knowing: a witness still has news for someone who only heard around (T9).
  const knows = (e: WorldEvent, id: string) => !!e.knownBy[id] && e.knownBy[id]!.source !== "rumor";
  return ledger.slice(lastMet + 1).filter((e) => before(e) && knows(e, a) && !knows(e, b));
}

function hasNews(world: World, a: string, b: string, day: number, slot: Slot): boolean {
  return newsFor(world, a, b, day, slot).length > 0 || newsFor(world, b, a, day, slot).length > 0;
}

/** The "what to bring up" list for a scene party, when `sceneNews` is on (docs/10 T7). */
function newsLines(world: World, deps: AdvanceDeps, from: string, to: string, day: number, slot: Slot): { news?: string[] } {
  if (!deps.sceneNews) return {};
  return { news: newsFor(world, from, to, day, slot).map((e) => `day ${e.day} ${e.slot}, ${world.place(e.where).name}: ${e.result}`) };
}

/**
 * A regional event is known around its region by the end of the day: every main character whose
 * routine passes a place in that region hears of it as rumour (the result only, no details). A
 * world-scope event reaches everyone. Zero calls (docs/10 T7).
 */
function spreadRumours(world: World, deps: AdvanceDeps, events: WorldEvent[]): void {
  for (const e of events) {
    if (e.scope === "personal" || e.origin === "chat") continue;
    const region = world.place(e.where).region;
    for (const c of Object.values(world.state.characters)) {
      // anyone with a card hears, main or not: a character living in the background (T10) still knows what goes around town
      if (!deps.cards[c.id] || e.knownBy[c.id]) continue;
      const passes = e.scope === "world" || (!!region && Object.values(c.routine).some((pid) => pid && world.place(pid).region === region));
      if (passes) e.knownBy[c.id] = { source: "rumor" };
    }
  }
}

async function recordScene(world: World, deps: AdvanceDeps, scene: SceneResult, aId: string, bId: string, slot: Slot, where: string, day: number, usage: WorldCallUsage, task?: WorldTask): Promise<WorldEvent> {
  let event = scene.event;
  let learned = scene.learned;
  if (task && deps.verifier !== null) {
    const check = await verifyEvent(deps.verifier ?? deps.llm, { task, event }, usage);
    let retried = false;
    if (!check.ok) {
      retried = true;
      const again = await runScene(
        deps.llm,
        {
          world,
          lang: deps.lang,
          a: { id: aId, card: deps.cards[aId]!, memory: knowledgeOf(world, aId, deps.memories), task },
          b: { id: bId, card: deps.cards[bId]!, memory: knowledgeOf(world, bId, deps.memories) },
          where,
          day,
          slot,
          persona: deps.persona,
          userBackDay: deps.userBackDay,
          avoid: check.problems,
        },
        usage,
      );
      event = again.event;
      learned = again.learned;
    }
    event = { ...event, checked: { ...check, retried } };
  }
  const ev = world.record(event);
  for (const [id, lines] of Object.entries(learned)) {
    const teller = id === aId ? bId : aId;
    const by = deps.cards[teller]!.name[deps.lang] ?? teller;
    for (const raw of lines) {
      const m = raw.match(/^\[about [^\]]*\]\s*/);
      const line = m ? raw.slice(m[0].length) : raw;
      if (!line.trim()) continue;
      world.record({ day, slot, where, who: [aId, bId], scope: "personal", result: line, details: [], knownBy: { [id]: { source: "told", by } }, origin: "scene", ...(m ? { aboutUser: true } : {}) });
    }
  }
  return ev;
}

const pairKey = (a: string, b: string) => [a, b].sort().join("+");

/** One outing per (character, place, slot): two errands at the same shop are one visit, not two scenes in the same hour (T3 first run). */
function dueGroups(world: World, day: number): Map<string, WorldTask[]> {
  const groups = new Map<string, WorldTask[]>();
  for (const t of world.dueTasks(day)) {
    const slot = t.dueSlot ?? "afternoon";
    const where = t.where ?? world.whereIs(t.assignee, t.dueDay, slot) ?? "?";
    const key = `${t.assignee}|${where}|${t.dueDay}|${slot}`;
    groups.set(key, [...(groups.get(key) ?? []), t]);
  }
  return groups;
}

/**
 * Live one outing: deferred when the user had the character in that slot (docs/10 T4), a scene when
 * another main character is at the place (both witness it), otherwise alone and checked against the
 * errand. Used for days lived one by one and for days inside a span alike (T6b: a span resolved the
 * handover alone, so Yoon never knew he had given the books away).
 */
async function liveTask(world: World, deps: AdvanceDeps, group: WorldTask[], usage: WorldCallUsage, met: Set<string>): Promise<WorldEvent | undefined> {
  const task: WorldTask = group.length === 1 ? group[0]! : { ...group[0]!, what: group.map((t) => t.what).join(" / ") };
  // The user wins the slot (docs/10 T4): a character who spent it with the user did not go. The
  // errand moves to the same slot next day and the character remembers why.
  const chat = world.withUser(task.assignee, task.dueDay, task.dueSlot ?? "afternoon");
  if (chat) {
    const name = deps.cards[task.assignee]?.name[deps.lang] ?? task.assignee;
    const placeName = task.where ? world.place(task.where).name : "the errand";
    const due = task.dueDay;
    for (const t of group) t.dueDay = due + 1;
    return world.record({
        day: due,
        slot: task.dueSlot ?? "afternoon",
        where: chat.where,
        who: [task.assignee],
        scope: "personal",
        result: `${name} did not go to ${placeName} that ${task.dueSlot ?? "afternoon"}: ${deps.persona?.name ?? "the user"} was there with ${name}, so the errand (${task.what}) moved to the next day`,
        details: [],
        knownBy: { [task.assignee]: { source: "witnessed" } },
        origin: "catchup",
      // not taskId: the task stays pending
    });
  }
  const finish = (ev: WorldEvent) => {
    for (const t of group) {
      if (t.status === "pending") t.status = "done";
      t.eventId = ev.id;
    }
  };
  const slot = task.dueSlot ?? "afternoon";
  const where = task.where ?? world.whereIs(task.assignee, task.dueDay, slot);
  const card = deps.cards[task.assignee];
  if (!card) throw new Error(`advanceDay: no card for ${task.assignee}`);
  const others = where
    ? world.whoElseAt(where, task.dueDay, slot, task.assignee).filter((id) => world.character(id).tier === "main" && deps.cards[id])
    : [];
  if (others.length && where) {
    const otherId = others[0]!;
    const scene = await runScene(
      deps.llm,
      {
        world,
        lang: deps.lang,
        a: { id: task.assignee, card, memory: knowledgeOf(world, task.assignee, deps.memories), task, ...newsLines(world, deps, task.assignee, otherId, task.dueDay, slot) },
        b: { id: otherId, card: deps.cards[otherId]!, memory: knowledgeOf(world, otherId, deps.memories), ...newsLines(world, deps, otherId, task.assignee, task.dueDay, slot) },
        where,
        day: task.dueDay,
        slot,
        persona: deps.persona,
        userBackDay: deps.userBackDay,
      },
      usage,
    );
    const ev = await recordScene(world, deps, scene, task.assignee, otherId, slot, where, task.dueDay, usage, task);
    finish(ev);
    met.add(pairKey(task.assignee, otherId));
    return ev;
  } else {
    let r = await resolveTask(deps.llm, { world, card, lang: deps.lang, task, memory: knowledgeOf(world, task.assignee, deps.memories) }, usage);
    if (deps.verifier !== null) {
      const check = await verifyEvent(deps.verifier ?? deps.llm, { task, event: r.event }, usage);
      let retried = false;
      if (!check.ok) {
        retried = true;
        r = await resolveTask(deps.llm, { world, card, lang: deps.lang, task, memory: knowledgeOf(world, task.assignee, deps.memories), avoid: check.problems }, usage);
      }
      r = { ...r, event: { ...r.event, checked: { ...check, retried } } };
    }
    return settle(world, group, r);
  }
}

/**
 * Move the world forward one day and live through the day that ended. Every task due by then is
 * resolved: as a scene when another main character is at the task's place in its slot, otherwise
 * alone, checked against the task and redone once if it contradicts it. Then, up to `scenesPerDay`
 * routine scenes between main characters who share a place and have news for each other. A quiet
 * day costs no calls.
 */
export async function advanceDay(world: World, deps: AdvanceDeps): Promise<WorldEvent[]> {
  const usage = deps.usage ?? emptyWorldUsage();
  world.setClock(world.day + 1, "morning");
  const day = world.day - 1;
  const events: WorldEvent[] = [];
  const met = new Set<string>();

  // The world's own events for the day that ended (none unless eventsPerDay > 0).
  for (const e of await generateDay(world, deps, day, usage)) events.push(world.record(e));

  for (const group of dueGroups(world, day).values()) {
    const ev = await liveTask(world, deps, group, usage, met);
    if (ev) events.push(ev);
  }

  // Routine scenes: co-located main characters with something to tell, capped per day. Where a
  // character actually was in a slot is the place of an event they took part in, else the routine.
  const locationOn = (id: string, slot: Slot): string | undefined =>
    world.state.ledger.find((e) => e.day === day && e.slot === slot && e.who.includes(id))?.where ?? world.character(id).routine[slot];
  let budget = deps.scenesPerDay ?? 1;
  const mains = Object.keys(world.state.characters).filter((id) => world.character(id).tier === "main" && deps.cards[id]);
  for (const slot of SLOTS) {
    if (budget <= 0) break;
    for (let i = 0; i < mains.length && budget > 0; i++) {
      for (let j = i + 1; j < mains.length && budget > 0; j++) {
        const a = mains[i]!;
        const b = mains[j]!;
        if (met.has(pairKey(a, b))) continue;
        if (world.withUser(a, day, slot) || world.withUser(b, day, slot)) continue; // the user wins the slot
        const where = locationOn(a, slot);
        if (!where || locationOn(b, slot) !== where) continue;
        if (!hasNews(world, a, b, day, slot)) continue;
        const scene = await runScene(
          deps.llm,
          {
            world,
            lang: deps.lang,
            a: { id: a, card: deps.cards[a]!, memory: knowledgeOf(world, a, deps.memories), ...newsLines(world, deps, a, b, day, slot) },
            b: { id: b, card: deps.cards[b]!, memory: knowledgeOf(world, b, deps.memories), ...newsLines(world, deps, b, a, day, slot) },
            where,
            day,
            slot,
            persona: deps.persona,
            userBackDay: deps.userBackDay,
          },
          usage,
        );
        events.push(await recordScene(world, deps, scene, a, b, slot, where, day, usage));
        met.add(pairKey(a, b));
        budget--;
      }
    }
  }
  // By the end of the day the region has heard of its regional events (zero calls).
  spreadRumours(world, deps, world.state.ledger.filter((e) => e.day === day));
  return events;
}
