/**
 * World experiments (docs/10 step 2+). A world scenario is a sequence of phases: a chat between the
 * user and one main character, or the world advancing by some days (tasks the characters agreed to
 * are lived through off-screen and land in the ledger). Everything the character later says about
 * the off-screen time is judged against the ledger, which is the one place the truth is written.
 */
import {
  ChatSession,
  World,
  advanceDay,
  catchUp,
  compactWorldFacts,
  emptyWorldUsage,
  extractTasks,
  findCharacter,
  observedProvider,
  startTrace,
  syncCharacter,
  type CharacterCard,
  type Fact,
  type LLMProvider,
  type MemoryState,
  type TokenUsage,
  type Turn,
  type WorldBlock,
  type WorldCallUsage,
  type WorldEvent,
  type WorldState,
} from "@persona/engine";
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";
import { z } from "zod";
import { BudgetTracker, trackedProvider } from "./budget.js";
import { resolveProvider } from "./providers.js";
import { recordFileName } from "./runner.js";
import { ScriptedUser, UserSimulator, type UserSource } from "./simulator.js";
import type { Arm, Scenario, UsageTotals } from "./types.js";

const REPO_ROOT = resolve(fileURLToPath(new URL("../../..", import.meta.url)));
export const WORLD_SCENARIOS_DIR = resolve(fileURLToPath(new URL("../world-scenarios", import.meta.url)));
export const WORLDS_DIR = join(REPO_ROOT, "worlds");

// ---------- scenario ----------

const FactSchema = z.object({
  text: z.string(),
  source: z.enum(["user", "self", "witnessed", "told", "rumor", "unknown"]).default("unknown"),
  by: z.string().optional(),
  /** World day the fact refers to; rendered relative to today in the prompt. Never write "today" into a fact. */
  day: z.number().int().optional(),
});

const Beat = z.object({ fromUserTurn: z.number().int().min(1), goal: z.string(), reveal: z.array(z.string()).default([]) });

export const WorldProbe = z.object({
  atUserTurn: z.number().int().min(1),
  /** recall: the reply must carry the ledger content. negative: the reply must assert nothing the ledger lacks. */
  kind: z.enum(["recall", "negative"]).default("recall"),
  ask: z.string(),
  /** "@ledger.result" | "@ledger.details" | "@ledger.none" | literal text */
  expected: z.string(),
});
export type WorldProbe = z.infer<typeof WorldProbe>;

const ChatPhase = z.object({
  kind: z.literal("chat"),
  characterId: z.string(),
  /** Slot of the day the chat happens in (default: the clock's, i.e. morning after an advance). */
  slot: z.enum(["morning", "afternoon", "evening", "night"]).optional(),
  userTurns: z.number().int().min(1),
  beats: z.array(Beat).min(1),
  probes: z.array(WorldProbe).default([]),
});
const AdvancePhase = z.object({ kind: z.literal("advance"), days: z.number().int().min(1).default(1) });
/** A scripted world event (T5): happens at the given slot of the current day, before the next advance. */
const EventPhase = z.object({
  kind: z.literal("event"),
  slot: z.enum(["morning", "afternoon", "evening", "night"]),
  where: z.string(),
  who: z.array(z.string()).default([]),
  scope: z.enum(["personal", "regional", "world"]).default("regional"),
  result: z.string(),
  details: z.array(z.string()).default([]),
  /** character id -> witnessed | told:<name> | rumor */
  knownBy: z.record(z.string(), z.string()).default({}),
  effects: z.record(z.string(), z.string().nullable()).default({}),
  /** What the event did to the user's things, by label (T9): text = new state, null = back to usual. */
  userEffects: z.record(z.string(), z.string().nullable()).default({}),
});

export const WorldScenario = z.object({
  id: z.string(),
  world: z.string(),
  lang: z.enum(["ko", "ja", "en"]),
  persona: z.object({ name: z.string(), description: z.string().optional() }),
  userStyle: z.string(),
  /** Per character id: what the character already remembers when the scenario starts. */
  seedMemory: z
    .record(
      z.string(),
      z.object({
        summary: z.string().default(""),
        facts: z.array(FactSchema).default([]),
        relationship: z.object({ stage: z.string().default("initial"), notes: z.array(z.string()).default([]) }).default({}),
      }),
    )
    .default({}),
  phases: z.array(z.discriminatedUnion("kind", [ChatPhase, AdvancePhase, EventPhase])).min(1),
});
export type WorldScenario = z.infer<typeof WorldScenario>;
export type ChatPhaseDef = z.infer<typeof ChatPhase>;

export function loadWorldScenarios(dir = WORLD_SCENARIOS_DIR): WorldScenario[] {
  return readdirSync(dir)
    .filter((f) => f.endsWith(".yaml") || f.endsWith(".yml"))
    .map((f) => {
      const parsed = WorldScenario.safeParse(YAML.parse(readFileSync(join(dir, f), "utf8")));
      if (!parsed.success) throw new Error(`Invalid world scenario ${f}:\n${parsed.error.issues.map((i) => `  - ${i.path.join(".")}: ${i.message}`).join("\n")}`);
      return parsed.data;
    });
}

export function findWorldScenario(id: string): WorldScenario {
  const s = loadWorldScenarios().find((x) => x.id === id);
  if (!s) throw new Error(`World scenario "${id}" not found in ${WORLD_SCENARIOS_DIR}`);
  return s;
}

// ---------- config & records ----------

export interface WorldArm extends Arm {
  /** false = the character is never told what happened off-screen (today's 1:1 apps). Default true. */
  worldSync?: boolean;
  /** Scenes get a "what to bring up" list (docs/10 T7). Default false. */
  sceneNews?: boolean;
  /** Which of the world's characters are main for this arm (docs/10 T8); the others become named extras. Default: as the world file says. */
  mains?: string[];
}

export interface WorldRunConfig {
  scenarioId: string;
  arms: WorldArm[];
  repeats: number;
  simulatorModel?: string;
  memoryModel?: string;
  /** Model for task extraction, errand resolution and scenes. Defaults to the arm's model. */
  worldModel?: string;
  /** Model that checks a decided event against its task (default: worldModel). A small one is enough. */
  verifierModel?: string;
  /** The world's own events per day while the user is away (default 0: nothing happens without tasks). */
  eventsPerDay?: number;
  /** For an advance longer than `recentDays`, older days are decided in spans of `spanDays` with at most `eventsPerSpan` events. */
  recentDays?: number;
  spanDays?: number;
  eventsPerSpan?: number;
  /** Fold ledger facts older than 7 days into the summary when a character carries more than this many (default 25). */
  maxFacts?: number;
  judgeModel?: string;
  outDir: string;
  /** Arm whose user turns and world the other arms replay (default: the first arm). "none" = every arm runs its own world. */
  replayArmId?: string;
  maxUsd?: number;
}

export type PhaseRecord =
  | {
      index: number;
      kind: "chat";
      characterId: string;
      day: number;
      worldBlock?: WorldBlock;
      /** Facts folded into the character's memory before this chat (empty when worldSync is off). */
      injected: Fact[];
      /** Dated facts folded into the summary before this chat (long gaps). */
      compacted?: number;
      turns: Turn[];
      memory: MemoryState;
      probes: WorldProbe[];
      /** Tasks extracted from this chat (source arm only). */
      tasks: string[];
    }
  | { index: number; kind: "advance"; fromDay: number; toDay: number; events: WorldEvent[]; usage: WorldCallUsage; worldAfter: WorldState; dayByDay?: number; inSpans?: number }
  | { index: number; kind: "event"; day: number; event: WorldEvent };

export interface WorldRecord {
  id: string;
  scenarioId: string;
  arm: WorldArm;
  lang: WorldScenario["lang"];
  persona: WorldScenario["persona"];
  phases: PhaseRecord[];
  world: WorldState;
  usage: { character: UsageTotals; simulator: UsageTotals; memory: UsageTotals; world: UsageTotals };
  replayOf?: string;
  traceId?: string;
  startedAt: string;
  finishedAt: string;
}

const emptyTotals = (): UsageTotals => ({ promptTokens: 0, completionTokens: 0, cachedPromptTokens: 0, calls: 0 });
function addUsage(t: UsageTotals, u?: TokenUsage, model?: string) {
  if (!u) return;
  t.promptTokens += u.promptTokens;
  t.completionTokens += u.completionTokens;
  t.cachedPromptTokens! += u.cachedPromptTokens ?? 0;
  t.calls++;
  if (model) t.model = model;
}

/** The simulated user asks a world probe without knowing the answer. */
const worldProbeText = (characterName: string) => (p: { ask: string }) =>
  `\nIMPORTANT for this message: ${p.ask} You do not know the answer; ask ${characterName} in character, casually, and do not suggest or guess any answer yourself.`;

function phaseScenario(s: WorldScenario, p: ChatPhaseDef, index: number): Scenario {
  return { id: `${s.id}#${index}`, characterId: p.characterId, lang: s.lang, persona: s.persona, userStyle: s.userStyle, beats: p.beats, probes: p.probes, safetyChecks: [] };
}

// ---------- run ----------

export async function runWorldConversation(
  scenario: WorldScenario,
  arm: WorldArm,
  cfg: WorldRunConfig,
  repeat: number,
  source: WorldRecord | undefined,
  budget = new BudgetTracker(cfg.maxUsd),
  log: (s: string) => void = () => {},
): Promise<WorldRecord> {
  const id = `${scenario.id}__${arm.id}__r${repeat}`;
  const trace = startTrace(`world:${scenario.id}/${arm.id}`, {
    input: { scenarioId: scenario.id, arm, repeat },
    metadata: { scenarioId: scenario.id, armId: arm.id, promptVersion: arm.promptVersion, model: arm.model, repeat: String(repeat) },
    tags: [scenario.id, arm.id, arm.promptVersion, source ? "replay" : "simulated", "world"],
  });
  let phaseNow = 0;
  const track = (p: LLMProvider, role: string): LLMProvider => observedProvider(trackedProvider(p, budget), trace, role, () => ({ phase: phaseNow }));

  const chat = resolveProvider(arm.model);
  const chatProvider = track(chat.provider, "character");
  const mem = chat.provider.id === "fake" || !cfg.memoryModel ? { provider: chat.provider, model: undefined } : resolveProvider(cfg.memoryModel);
  const memoryProvider = track(mem.provider, "memory");
  const wl = chat.provider.id === "fake" || !cfg.worldModel ? chat : resolveProvider(cfg.worldModel);
  const worldLLM = { provider: track(wl.provider, "world"), model: wl.model };
  const vf = chat.provider.id === "fake" || !cfg.verifierModel ? wl : resolveProvider(cfg.verifierModel);
  const verifierLLM = { provider: track(vf.provider, "world"), model: vf.model };
  const sim = chat.provider.id === "fake" || !cfg.simulatorModel ? chat : resolveProvider(cfg.simulatorModel);
  const simProvider = track(sim.provider, "simulator");

  const world = World.fromFile(join(WORLDS_DIR, `${scenario.world}.yaml`));
  if (arm.mains) {
    for (const id of arm.mains) if (!world.state.characters[id]) throw new Error(`arm ${arm.id}: mains names unknown character "${id}"`);
    for (const c of Object.values(world.state.characters)) {
      if (arm.mains.includes(c.id)) c.tier = "main";
      else if (c.tier === "main") {
        c.tier = "named";
        c.sketch ??= findCharacter(c.id).identity.role; // a demoted main is a few lines, like any extra
      }
    }
  }
  const cards: Record<string, CharacterCard> = {};
  for (const c of Object.values(world.state.characters)) if (c.tier === "main") cards[c.id] = findCharacter(c.id);

  const sessions = new Map<string, ChatSession>();
  const usage = { character: emptyTotals(), simulator: emptyTotals(), memory: emptyTotals(), world: emptyTotals() };
  const phases: PhaseRecord[] = [];
  const startedAt = new Date().toISOString();
  const sync = arm.worldSync !== false;

  for (let i = 0; i < scenario.phases.length; i++) {
    phaseNow = i;
    const p = scenario.phases[i]!;
    if (p.kind === "event") {
      const knownBy: WorldEvent["knownBy"] = {};
      for (const [cid, how] of Object.entries(p.knownBy)) {
        if (how === "witnessed" || how === "rumor") knownBy[cid] = { source: how };
        else if (how.startsWith("told:")) knownBy[cid] = { source: "told", by: how.slice(5) };
        else throw new Error(`event phase ${i}: knownBy[${cid}] must be witnessed | told:<name> | rumor`);
      }
      const ev = world.record({ day: world.day, slot: p.slot, where: p.where, who: p.who, scope: p.scope, result: p.result, details: p.details, knownBy, origin: "catchup", ...(Object.keys(p.effects).length ? { effects: p.effects } : {}), ...(Object.keys(p.userEffects).length ? { userEffects: p.userEffects } : {}) });
      phases.push({ index: i, kind: "event", day: world.day, event: ev });
      log(`  ${id}: phase ${i} event day ${world.day} ${p.slot} @${p.where}: ${p.result.slice(0, 60)}`);
      continue;
    }
    if (p.kind === "advance") {
      const fromDay = world.day;
      const wu = emptyWorldUsage();
      let events: WorldEvent[] = [];
      let tiers: { dayByDay?: number; inSpans?: number } = {};
      if (source) {
        // Same world as the source arm: copy its ledger and clock, keep this arm's sync bookkeeping.
        const src = source.phases[i];
        if (!src || src.kind !== "advance") throw new Error(`replay: source record has no advance phase at ${i}`);
        world.state = { ...structuredClone(src.worldAfter), synced: world.state.synced, lastSeen: world.state.lastSeen };
        events = src.events;
      } else {
        const memories: Record<string, MemoryState> = {};
        for (const [cid, s] of sessions) memories[cid] = s.memory;
        const deps = { llm: worldLLM, verifier: verifierLLM, lang: scenario.lang, cards, memories, persona: scenario.persona, usage: wu, eventsPerDay: cfg.eventsPerDay, eventsPerSpan: cfg.eventsPerSpan, sceneNews: arm.sceneNews };
        if (p.days > (cfg.recentDays ?? 7)) {
          const r = await catchUp(world, { ...deps, recentDays: cfg.recentDays, spanDays: cfg.spanDays }, p.days);
          events.push(...r.events);
          tiers = { dayByDay: r.dayByDay, inSpans: r.inSpans };
        } else {
          for (let d = 0; d < p.days; d++) events.push(...(await advanceDay(world, deps)));
          tiers = { dayByDay: p.days, inSpans: 0 };
        }
      }
      addUsage(usage.world, { promptTokens: wu.promptTokens, completionTokens: wu.completionTokens, cachedPromptTokens: wu.cachedPromptTokens }, wu.model);
      usage.world.calls += wu.calls - 1; // addUsage counted one call
      phases.push({ index: i, kind: "advance", fromDay, toDay: world.day, events, usage: wu, worldAfter: world.snapshot(), ...tiers });
      log(`  ${id}: day ${fromDay} → ${world.day}, ${events.length} event(s), ${wu.calls} world call(s)${tiers.inSpans ? ` (${tiers.inSpans} days in spans, ${tiers.dayByDay} day by day)` : ""}`);
      continue;
    }

    if (p.slot) world.setClock(world.day, p.slot);
    const card = cards[p.characterId] ?? findCharacter(p.characterId);
    let session = sessions.get(p.characterId);
    if (!session) {
      session = new ChatSession({
        card,
        lang: scenario.lang,
        provider: chatProvider,
        memoryProvider,
        promptVersion: arm.promptVersion,
        model: chat.model,
        temperature: arm.temperature,
        persona: scenario.persona,
        memory: { model: mem.model, ...(arm.memory ?? {}) },
        filters: arm.filters,
      });
      const seed = scenario.seedMemory[p.characterId];
      if (seed) {
        session.memory = {
          summary: seed.summary,
          facts: seed.facts.map((f) => ({ text: f.text, source: f.source, ...(f.by ? { by: f.by } : {}), ...(f.day !== undefined ? { day: f.day } : {}) })),
          relationship: { stage: seed.relationship.stage, notes: [...seed.relationship.notes] },
          coveredUpTo: -1,
        };
      }
      sessions.set(p.characterId, session);
    }
    const injected = sync ? syncCharacter(session, world, p.characterId) : [];
    let compacted = 0;
    if (sync) {
      const c = await compactWorldFacts(session, world, { provider: memoryProvider, model: mem.model }, { maxFacts: cfg.maxFacts });
      compacted = c.folded;
      if (c.usage) addUsage(usage.memory, c.usage, c.usage.model);
    }
    const worldBlock = sync ? session.world : undefined;

    const name = card.name[scenario.lang] ?? card.id;
    let user: UserSource;
    if (source) {
      const src = source.phases[i];
      if (!src || src.kind !== "chat") throw new Error(`replay: source record has no chat phase at ${i}`);
      user = new ScriptedUser(src.turns.filter((t) => t.role === "user").map((t) => t.content));
    } else {
      user = new UserSimulator(simProvider, phaseScenario(scenario, p, i), sim.model, { probeText: worldProbeText(name) });
    }

    const from = session.turns.length;
    const consolidationsBefore = session.memoryUsage.calls;
    for (let ut = 1; ut <= p.userTurns; ut++) {
      const msg = await user.next(ut, session.turns, name);
      addUsage(usage.simulator, msg.usage, msg.model);
      const reply = await session.send(msg.text);
      if (reply.meta?.usage) addUsage(usage.character, reply.meta.usage, reply.meta.model);
    }
    if (session.memoryUsage.calls > consolidationsBefore) {
      const u = session.memoryUsage;
      usage.memory.promptTokens = u.promptTokens;
      usage.memory.completionTokens = u.completionTokens;
      usage.memory.calls = u.calls;
      usage.memory.model = u.model;
    }
    const phaseTurns = session.turns.slice(from);

    let tasks: string[] = [];
    if (!source) {
      const wu = emptyWorldUsage();
      const found = await extractTasks(worldLLM, { world, card, lang: scenario.lang, persona: scenario.persona, turns: phaseTurns, characterId: p.characterId }, wu);
      for (const t of found) world.addTask(t);
      tasks = found.map((t) => `${t.what} @${t.where ?? "?"} day ${t.dueDay} ${t.dueSlot ?? ""}`.trim());
      addUsage(usage.world, wu, wu.model);
    }
    world.sawUser(p.characterId);
    if (!source) {
      // The user met the character where the routine puts them; a pending errand's place is where
      // they would have been, which is exactly what the chat prevented (T4).
      const at = world.character(p.characterId).routine[world.slot] ?? world.whereIs(p.characterId) ?? world.state.places[0]!.id;
      world.recordChat(p.characterId, scenario.persona.name, at);
    }
    phases.push({ index: i, kind: "chat", characterId: p.characterId, day: world.day, worldBlock, injected, ...(compacted ? { compacted } : {}), turns: phaseTurns, memory: structuredClone(session.memory), probes: p.probes, tasks });
    log(`  ${id}: phase ${i} chat day ${world.day}, ${p.userTurns} user turns, ${injected.length} fact(s) injected${compacted ? `, ${compacted} folded into the summary` : ""}${tasks.length ? `, tasks: ${tasks.join(" | ")}` : ""}`);
  }

  const rec: WorldRecord = {
    id,
    scenarioId: scenario.id,
    arm,
    lang: scenario.lang,
    persona: scenario.persona,
    phases,
    world: world.snapshot(),
    usage,
    ...(source ? { replayOf: source.id } : {}),
    ...(trace.traceId ? { traceId: trace.traceId } : {}),
    startedAt,
    finishedAt: new Date().toISOString(),
  };
  trace.update({ output: { usage, ledger: world.state.ledger.length } });
  trace.end();
  return rec;
}

export async function runWorldMatrix(cfg: WorldRunConfig, log: (s: string) => void = console.log, fresh = false): Promise<WorldRecord[]> {
  const scenario = findWorldScenario(cfg.scenarioId);
  const dir = join(cfg.outDir, "conversations");
  mkdirSync(dir, { recursive: true });
  const budget = new BudgetTracker(cfg.maxUsd);
  const sourceArm = cfg.replayArmId === "none" ? undefined : (cfg.arms.find((a) => a.id === cfg.replayArmId) ?? cfg.arms[0]!);
  const out: WorldRecord[] = [];
  const load = (id: string): WorldRecord | undefined => {
    const f = join(dir, recordFileName(id));
    return !fresh && existsSync(f) ? (JSON.parse(readFileSync(f, "utf8")) as WorldRecord) : undefined;
  };
  for (let r = 0; r < cfg.repeats; r++) {
    let src: WorldRecord | undefined;
    if (sourceArm) {
      const srcId = `${scenario.id}__${sourceArm.id}__r${r}`;
      src = load(srcId);
      if (!src) {
        src = await runWorldConversation(scenario, sourceArm, cfg, r, undefined, budget, log);
        writeFileSync(join(dir, recordFileName(src.id)), JSON.stringify(src, null, 2));
      } else log(`  resuming ${srcId}`);
      out.push(src);
    }
    for (const arm of cfg.arms) {
      if (sourceArm && arm.id === sourceArm.id) continue;
      const id = `${scenario.id}__${arm.id}__r${r}`;
      let rec = load(id);
      if (!rec) {
        rec = await runWorldConversation(scenario, arm, cfg, r, src, budget, log);
        writeFileSync(join(dir, recordFileName(rec.id)), JSON.stringify(rec, null, 2));
      } else log(`  resuming ${id}`);
      out.push(rec);
    }
  }
  return out;
}

// ---------- judge ----------

export type ReplyLabel = "grounded" | "fabricated" | "unaware" | "none";

export interface WorldPhaseScore {
  index: number;
  characterId: string;
  replies: { userTurn: number; label: ReplyLabel; note?: string; snippet: string }[];
  probes: { atUserTurn: number; kind: "recall" | "negative"; ok: boolean; note?: string; answer: string }[];
}

export interface WorldScoreRecord {
  conversationId: string;
  armId: string;
  judgeModel: string;
  phases: WorldPhaseScore[];
  totals: { replies: number; grounded: number; fabricated: number; unaware: number; recallOk: number; recallTotal: number; negativeOk: number; negativeTotal: number };
}

/** What the character knows from the ledger before a chat phase on `day`, as text for the judge. */
function knownText(rec: WorldRecord, characterId: string, day: number, charName: string): string {
  const lines: string[] = [];
  for (const e of rec.world.ledger) {
    const k = e.knownBy[characterId];
    if (!k || e.day >= day) continue;
    const when = `day ${e.day} ${e.slot}`;
    const place = rec.world.places.find((p) => p.id === e.where)?.name ?? e.where;
    if (k.source === "witnessed") {
      lines.push(`- [${when}, ${place}, ${charName} was there] ${e.result}`);
      for (const d of e.details) lines.push(`    - detail: ${d}`);
    } else lines.push(`- [${when}, ${k.source}${k.by ? ` by ${k.by}` : ""}] ${e.result}`);
    for (const [pid, cond] of Object.entries(e.effects ?? {})) lines.push(`    - since then, ${rec.world.places.find((p) => p.id === pid)?.name ?? pid}: ${cond ?? "back to usual"}`);
    for (const [label, state] of Object.entries(e.userEffects ?? {})) lines.push(`    - what this did to ${rec.persona.name}'s things, ${label}: ${state ?? "back to usual"} (${charName} knows this and may raise it unasked)`);
  }
  return lines.length ? lines.join("\n") : "(nothing: no event the character knows of happened in between)";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function askJson(provider: LLMProvider, model: string | undefined, system: string, user: string): Promise<any> {
  const res = await provider.generate([{ role: "system", content: system }, { role: "user", content: user }], { model, temperature: 0, maxTokens: 1800 });
  const m = res.text.match(/\{[\s\S]*\}/);
  if (!m) return {};
  try {
    return JSON.parse(m[0]);
  } catch {
    return {};
  }
}

export async function judgeWorld(rec: WorldRecord, opts: { provider: LLMProvider; model?: string }): Promise<WorldScoreRecord> {
  const phases: WorldPhaseScore[] = [];
  const totals: WorldScoreRecord["totals"] = { replies: 0, grounded: 0, fabricated: 0, unaware: 0, recallOk: 0, recallTotal: 0, negativeOk: 0, negativeTotal: 0 };
  let advanced = false;
  for (const p of rec.phases) {
    if (p.kind !== "chat") {
      if (p.kind === "advance") advanced = true;
      continue;
    }
    if (!advanced) continue; // nothing off-screen has happened yet
    const card = findCharacter(p.characterId);
    const name = card.name[rec.lang] ?? card.id;
    const known = knownText(rec, p.characterId, p.day, name);
    // What the character knew apart from the ledger: seeded/remembered facts and earlier chats with
    // the user. Without these the judge calls a detail the user said on day 1 "fabricated" (T3b).
    const prior = p.memory.facts.filter((f) => f.source === "user" || f.source === "self" || f.source === "unknown").map((f) => `- ${f.text}`);
    const earlier: string[] = [];
    for (const q of rec.phases) {
      if (q.kind !== "chat" || q.index >= p.index || q.characterId !== p.characterId) continue;
      earlier.push(`(day ${q.day})`, ...q.turns.map((t) => `${t.role === "user" ? rec.persona.name : name}: ${t.content}`));
    }
    const transcript: string[] = [];
    let ut = 0;
    const replyByUserTurn = new Map<number, string>();
    for (const t of p.turns) {
      if (t.role === "user") {
        ut++;
        transcript.push(`[user turn ${ut}] ${rec.persona.name}: ${t.content}`);
      } else {
        transcript.push(`[reply ${ut}] ${name}: ${t.content}`);
        replyByUserTurn.set(ut, t.content);
      }
    }
    // One person, several names (선영 / 이모 / 이모님): without this the judge called every 이모님 a fabrication (T6 first run).
    const roster = Object.values(rec.world.characters)
      .map((c) => `- ${c.name}${c.aka?.length ? ` (also called ${c.aka.join(", ")})` : ""}: ${c.tier === "main" ? findCharacter(c.id).identity.role : (c.sketch ?? "")}`)
      .join("\n");
    const probes = p.probes.map((pr) => ({ ...pr, expected: pr.expected.startsWith("@ledger") ? `(${pr.expected.slice(8)} of the ledger above)` : pr.expected }));
    const system = `You check a roleplay character's replies against the world's ledger. ${name} spoke with ${rec.persona.name} on day ${p.day}. Between their previous conversation and this one, this is EVERYTHING that happened to ${name} off-screen, as the world recorded it:

${known}

People in this world (each line is ONE person, whichever of the names is used):
${roster}

${name} also already knew (from before, not off-screen):
${prior.length ? prior.join("\n") : "(nothing recorded)"}
${earlier.length ? `\nEarlier conversations between ${name} and ${rec.persona.name} (what was said there is known to both and is not fabrication):\n${earlier.join("\n")}\n` : ""}
The ledger is the only truth about the off-screen time; the facts and conversations above are what else ${name} legitimately knows. Label every reply of ${name}:
- "grounded": the reply speaks of the off-screen time and everything it states about it is in the ledger (paraphrase is fine; leaving things out is fine).
- "fabricated": the reply states something about the off-screen time or the errand that the ledger does not hold or that contradicts it - a quote, a detail, an outcome, a message from someone. Clearly marked guesses ("아마", "~것 같아요") about things not in the ledger do not count.
- "unaware": ${rec.persona.name} refers to the errand or the off-screen time and the reply acts as if it had not happened, does not know how it went, or asks ${rec.persona.name} what happened, although the ledger says ${name} was there.
- "none": the reply does not touch the off-screen time.
Then judge each probe by the reply to that user turn. "recall": ok when the reply conveys the expected ledger content (the result, or at least one of its details, in substance). "negative": ok when the reply asserts nothing the ledger lacks - saying nothing of the kind happened, or not knowing, is ok; inventing something is not.
Respond with JSON only: {"replies": [{"userTurn": n, "label": "grounded|fabricated|unaware|none", "note": "one short sentence quoting what was fabricated or missed, if anything"}], "probes": [{"atUserTurn": n, "ok": true|false, "note": "one short sentence"}]}`;
    const user = `CONVERSATION (day ${p.day}):\n${transcript.join("\n\n")}\n\nPROBES:\n${probes.map((pr) => `- at user turn ${pr.atUserTurn} (${pr.kind}): ${pr.ask} Expected: ${pr.expected}`).join("\n") || "(none)"}`;
    const out = await askJson(opts.provider, opts.model, system, user);
    const labels = new Map<number, { label: ReplyLabel; note?: string }>();
    for (const r of Array.isArray(out.replies) ? out.replies : []) {
      if (typeof r?.userTurn === "number" && ["grounded", "fabricated", "unaware", "none"].includes(r.label)) labels.set(r.userTurn, { label: r.label, note: typeof r.note === "string" ? r.note : undefined });
    }
    const replies: WorldPhaseScore["replies"] = [];
    for (const [turn, text] of replyByUserTurn) {
      const l = labels.get(turn) ?? { label: "none" as ReplyLabel };
      replies.push({ userTurn: turn, label: l.label, note: l.note, snippet: text.slice(0, 160).replace(/\s+/g, " ") });
      totals.replies++;
      if (l.label === "grounded") totals.grounded++;
      if (l.label === "fabricated") totals.fabricated++;
      if (l.label === "unaware") totals.unaware++;
    }
    const verdicts = new Map<number, { ok: boolean; note?: string }>();
    for (const v of Array.isArray(out.probes) ? out.probes : []) if (typeof v?.atUserTurn === "number") verdicts.set(v.atUserTurn, { ok: !!v.ok, note: typeof v.note === "string" ? v.note : undefined });
    const probeScores: WorldPhaseScore["probes"] = p.probes.map((pr) => {
      const v = verdicts.get(pr.atUserTurn) ?? { ok: false, note: "no verdict" };
      if (pr.kind === "recall") {
        totals.recallTotal++;
        if (v.ok) totals.recallOk++;
      } else {
        totals.negativeTotal++;
        if (v.ok) totals.negativeOk++;
      }
      return { atUserTurn: pr.atUserTurn, kind: pr.kind, ok: v.ok, note: v.note, answer: (replyByUserTurn.get(pr.atUserTurn) ?? "").slice(0, 200).replace(/\s+/g, " ") };
    });
    phases.push({ index: p.index, characterId: p.characterId, replies, probes: probeScores });
  }
  return { conversationId: rec.id, armId: rec.arm.id, judgeModel: `${opts.provider.id}:${opts.model ?? opts.provider.defaultModel}`, phases, totals };
}

// ---------- report ----------

export function worldReport(records: WorldRecord[], scores: WorldScoreRecord[], title: string): string {
  const byArm = new Map<string, { recs: WorldRecord[]; scores: WorldScoreRecord[] }>();
  for (const r of records) {
    const e = byArm.get(r.arm.id) ?? { recs: [], scores: [] };
    e.recs.push(r);
    byArm.set(r.arm.id, e);
  }
  for (const s of scores) byArm.get(s.armId)?.scores.push(s);
  const pct = (a: number, b: number) => (b ? `${a}/${b}` : "–");
  const lines: string[] = [`## ${title}`, "", "| arm | n | recall probes | negative probes | fabricated / replies | unaware | grounded | world calls · tokens / run | char prompt tokens / reply |", "|---|---:|---:|---:|---:|---:|---:|---|---:|"];
  for (const [armId, { recs, scores: sc }] of byArm) {
    const t = sc.reduce(
      (acc, s) => {
        for (const k of Object.keys(acc) as (keyof typeof acc)[]) acc[k] += s.totals[k];
        return acc;
      },
      { replies: 0, grounded: 0, fabricated: 0, unaware: 0, recallOk: 0, recallTotal: 0, negativeOk: 0, negativeTotal: 0 },
    );
    const wcalls = recs.reduce((n, r) => n + r.usage.world.calls, 0) / recs.length;
    const wtok = recs.reduce((n, r) => n + r.usage.world.promptTokens + r.usage.world.completionTokens, 0) / recs.length;
    const creplies = recs.reduce((n, r) => n + r.usage.character.calls, 0);
    const cprompt = recs.reduce((n, r) => n + r.usage.character.promptTokens, 0);
    lines.push(
      `| ${armId} | ${recs.length} | ${pct(t.recallOk, t.recallTotal)} | ${pct(t.negativeOk, t.negativeTotal)} | ${pct(t.fabricated, t.replies)} | ${t.unaware} | ${t.grounded} | ${wcalls.toFixed(1)} · ${Math.round(wtok)} | ${creplies ? Math.round(cprompt / creplies) : "–"} |`,
    );
  }
  lines.push("", "### Ledger per run", "");
  for (const r of records) {
    if (r.replayOf) continue;
    lines.push(`**${r.id}**`);
    for (const p of r.phases) {
      if (p.kind === "chat" && p.tasks.length) lines.push(`- tasks after phase ${p.index}: ${p.tasks.join(" | ")}`);
      if (p.kind === "event") lines.push(`- ${p.event.id} day ${p.event.day} ${p.event.slot} @${p.event.where} (scripted): ${p.event.result}${p.event.effects ? ` → ${JSON.stringify(p.event.effects)}` : ""}`);
      if (p.kind === "advance") {
        const scenes = p.events.filter((x) => x.origin === "scene" && x.details.length).length;
        // told lines are recorded by recordScene straight into the ledger (not returned as events), so count them there
        const told = p.worldAfter.ledger.filter((x) => x.day >= p.fromDay && x.day < p.toDay && Object.values(x.knownBy).some((k) => k.source === "told")).length;
        // rumours spread onto events already in the ledger (a scripted notice), so count on the ledger of those days
        const rumours = p.worldAfter.ledger.filter((x) => x.day >= p.fromDay && x.day < p.toDay && Object.values(x.knownBy).some((k) => k.source === "rumor")).length;
        const conds = Object.entries(p.worldAfter.conditions ?? {}).map(([k, v]) => `${k}: ${v}`).join("; ");
        const ustate = Object.entries(p.worldAfter.user ?? {}).map(([k, v]) => `${k}: ${v}`).join("; ");
        lines.push(
          `- advance day ${p.fromDay} → ${p.toDay}: ${p.events.length} events (${p.events.filter((x) => x.origin === "world" || x.origin === "span").length} world, ${scenes} scenes, ${told} told lines, ${rumours} heard as rumour), ${p.usage.calls} world calls${p.inSpans ? ` (${p.inSpans} days in spans, ${p.dayByDay} day by day)` : ""}; conditions after: ${conds || "all usual"}${ustate ? `; ${r.persona.name}'s things: ${ustate}` : ""}`,
        );
        for (const e of p.events) lines.push(`- ${e.id} day ${e.day} ${e.slot} @${e.where} (${e.origin}): ${e.result}${e.details.length ? `\n  - ${e.details.join("\n  - ")}` : ""}`);
      }
    }
    lines.push("");
  }
  lines.push("### Probes and labels", "");
  for (const s of scores) {
    lines.push(`**${s.conversationId}**`);
    for (const p of s.phases) {
      for (const pr of p.probes) lines.push(`- probe @${pr.atUserTurn} ${pr.kind}: ${pr.ok ? "O" : "X"} ${pr.note ?? ""} ← ${pr.answer}`);
      for (const r of p.replies) if (r.label === "fabricated" || r.label === "unaware") lines.push(`- reply ${r.userTurn} \`${r.label}\`: ${r.note ?? ""} ← ${r.snippet}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

export function readWorldRecords(outDir: string): WorldRecord[] {
  const dir = join(outDir, "conversations");
  return existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".json")).map((f) => JSON.parse(readFileSync(join(dir, f), "utf8")) as WorldRecord) : [];
}

export function readWorldScores(outDir: string): WorldScoreRecord[] {
  const dir = join(outDir, "scores");
  return existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".json")).map((f) => JSON.parse(readFileSync(join(dir, f), "utf8")) as WorldScoreRecord) : [];
}

export const worldReportTitle = (outDir: string) => `World eval · ${basename(outDir)}`;
