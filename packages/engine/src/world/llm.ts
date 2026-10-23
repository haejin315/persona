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
  world.state.places.map((p) => `- ${p.id}: ${p.name}${p.hours ? ` (${p.hours})` : ""}${p.note ? ` - ${p.note}` : ""}`).join("\n");

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
  const system = [
    `You read a roleplay conversation between ${persona.name} (the user) and the character ${name} and list what ${name} AGREED to do later, away from this conversation.`,
    `Today is day ${world.day}. Places in this world (use the id):\n${placesList(world)}`,
    `Output JSON only: {"tasks": [{"what": "one line: what ${name} will do and for whom, with every concrete detail needed to do it (titles, names, what to check), written in ${LANG_NAME[lang]}", "where": "place id or null", "dueDay": <day number>, "dueSlot": "morning|afternoon|evening|night"}]}`,
    `Rules: include only commitments ${name} clearly accepted (a request ${name} declined or left open is not a task). "tomorrow" is day ${world.day + 1}; an unspecified day is day ${world.day + 1}. Things done inside this same conversation are not tasks. Empty list if none.`,
  ].join("\n");
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
    out.push({ assignee: characterId, what: o.what.trim(), where, dueDay, dueSlot, from: persona.name });
  }
  return out;
}

/**
 * Decide what happened when a character carried out a task alone. One call, JSON out. The world's
 * rules keep it mundane; the details are what the character will later remember as a witness.
 */
export async function resolveTask(
  llm: WorldLLM,
  args: { world: World; card: CharacterCard; lang: Lang; task: WorldTask; memory?: MemoryState },
  usage: WorldCallUsage = emptyWorldUsage(),
): Promise<Omit<WorldEvent, "id">> {
  const { world, card, lang, task } = args;
  const name = card.name[lang] ?? card.id;
  const slot = task.dueSlot ?? "afternoon";
  const where = task.where ?? world.whereIs(task.assignee, task.dueDay, slot) ?? world.state.places[0]!.id;
  const place = world.place(where);
  const system = [
    `You decide what happened, off-screen, when ${name} did an errand. You are the world, not the character: plain, concrete, mundane. Write in ${LANG_NAME[lang]}.`,
    `# WORLD\n${card.world.summary}\nRules:\n${card.world.rules.map((r) => `- ${r}`).join("\n")}`,
    `# WHO\n${name}: ${card.identity.role} ${card.identity.personality.join(", ")}.`,
    `# WHERE\n${place.name}${place.hours ? ` (${place.hours})` : ""}${place.note ? `. ${place.note}` : ""}`,
    `# WHEN\nDay ${task.dueDay}, ${slot}.`,
    args.memory?.facts.length ? `# WHAT ${name} KNOWS\n${args.memory.facts.map((f) => `- ${f.text}`).join("\n")}` : "",
    `Output JSON only: {"outcome": "done|partial|failed", "result": "ONE sentence, past tense: what is now true (what was obtained, learned or not)", "details": ["2 to 4 short concrete things ${name} saw or heard there: what the other person said (brief quote), what the object looked like, a small complication - each one a fact, no feelings"]}`,
    "Rules: nothing dramatic (no accidents, no coincidences, no new love interests). Most errands succeed plainly; at most one small, ordinary complication. Do not invent facts about the user. Name people at the place only if the place note names them; otherwise call them by role (the owner).",
  ]
    .filter(Boolean)
    .join("\n\n");
  const res = await llm.provider.generate(
    [
      { role: "system", content: system },
      { role: "user", content: `TASK: ${task.what}` },
    ],
    { model: llm.model, temperature: 0.7, maxTokens: 500 },
  );
  add(usage, res);
  const parsed = safeJson(res.text) ?? {};
  const details = strs(parsed.details).slice(0, 4);
  const result = typeof parsed.result === "string" && parsed.result.trim() ? parsed.result.trim() : `${name} went to ${place.name}: ${task.what}`;
  return {
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
  };
}

export interface SceneParty {
  id: string;
  card: CharacterCard;
  memory?: MemoryState;
  task?: WorldTask;
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
  args: { world: World; lang: Lang; a: SceneParty; b: SceneParty; where: string; day: number; slot: Slot; persona?: UserPersona },
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
    ].join("\n");
  const system = [
    `You write what happened when two characters met, off-screen, and then report it as data. Write in ${LANG_NAME[lang]}.`,
    `# WORLD\n${a.card.world.summary}\nRules:\n${a.card.world.rules.map((r) => `- ${r}`).join("\n")}`,
    `# WHERE / WHEN\n${place.name}${place.hours ? ` (${place.hours})` : ""}${place.note ? `. ${place.note}` : ""}. Day ${day}, ${slot}.`,
    who(a),
    who(b),
    args.persona
      ? `# THE USER\n${args.persona.name} is not present. ${args.persona.description ?? ""} Characters may mention ${args.persona.name} only from what they know above.`
      : "",
    `Output JSON only: {"scene": "the scene, 6 to 14 short lines in the *action* \\"speech\\" format, in character for both", "result": "ONE sentence, past tense: what is now true after this meeting", "details": ["2 to 4 concrete things both would remember"], "told": {"${a.id}": ["facts ${nameOf(b)} told ${nameOf(a)} that ${nameOf(a)} did not know - one line each, attributed ('${nameOf(b)} said ...')"], "${b.id}": ["facts ${nameOf(a)} told ${nameOf(b)} that ${nameOf(b)} did not know"]}}`,
    "Rules: each character says only what they know and would say given their personality and secrets (a reserved character shares little). Nothing dramatic. If a task brought one of them, the scene resolves it plainly.",
  ]
    .filter(Boolean)
    .join("\n\n");
  const res = await llm.provider.generate(
    [
      { role: "system", content: system },
      { role: "user", content: "Write the scene and the report." },
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

export interface AdvanceDeps {
  llm: WorldLLM;
  lang: Lang;
  cards: Record<string, CharacterCard>;
  memories?: Record<string, MemoryState>;
  persona?: UserPersona;
  usage?: WorldCallUsage;
}

/**
 * Move the world forward one day and live through every task that fell due. When another main
 * character is at the task's place in its slot, the task resolves as a scene between the two;
 * otherwise alone. A routine day with no tasks costs no calls.
 */
export async function advanceDay(world: World, deps: AdvanceDeps): Promise<WorldEvent[]> {
  const usage = deps.usage ?? emptyWorldUsage();
  world.setClock(world.day + 1, "morning");
  const events: WorldEvent[] = [];
  // The day that just ended is lived through: every task due on or before it is resolved, so a
  // character met on the morning of day d+1 can speak about what they did on day d. A task due
  // later on d+1 stays pending.
  for (const task of world.dueTasks(world.day - 1)) {
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
          a: { id: task.assignee, card, memory: deps.memories?.[task.assignee], task },
          b: { id: otherId, card: deps.cards[otherId]!, memory: deps.memories?.[otherId] },
          where,
          day: task.dueDay,
          slot,
          persona: deps.persona,
        },
        usage,
      );
      events.push(world.record(scene.event));
      // What each was told is its own ledger line with a `told` source, so provenance survives.
      for (const [id, lines] of Object.entries(scene.learned)) {
        const teller = id === task.assignee ? otherId : task.assignee;
        const by = deps.cards[teller]!.name[deps.lang] ?? teller;
        for (const line of lines) {
          world.record({ day: task.dueDay, slot, where, who: [task.assignee, otherId], scope: "personal", result: line, details: [], knownBy: { [id]: { source: "told", by } }, origin: "scene" });
        }
      }
    } else {
      events.push(world.record(await resolveTask(deps.llm, { world, card, lang: deps.lang, task, memory: deps.memories?.[task.assignee] }, usage)));
    }
  }
  return events;
}
