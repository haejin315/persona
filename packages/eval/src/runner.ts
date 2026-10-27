import { ChatSession, findCharacter, observedProvider, startTrace, type LLMProvider, type TokenUsage } from "@persona/engine";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";
import { BudgetExceededError, BudgetTracker, trackedProvider } from "./budget.js";
import { costOf } from "./pricing.js";
import { resolveProvider } from "./providers.js";
import { findScenario } from "./scenarios.js";
import { ScriptedUser, UserSimulator, type UserSource } from "./simulator.js";
import type { Arm, ConversationRecord, RunConfig, Scenario, UsageTotals } from "./types.js";

export interface RunHooks {
  onTurn?: (c: { conversationId: string; userTurn: number; of: number }) => void;
  onConversation?: (rec: ConversationRecord) => void;
  onSkip?: (ids: string[]) => void;
}

export interface RunContext {
  budget?: BudgetTracker;
  replay?: ReplayIndex;
  /** Reuse finished recordings already in outDir (default true). */
  resume?: boolean;
}

/** Recorded conversations to replay the user side from. */
export class ReplayIndex {
  constructor(private recs: ConversationRecord[], private armId?: string) {}

  static conversationsDir(dir: string): string {
    return basename(dir) === "conversations" ? dir : join(dir, "conversations");
  }

  /** Loads whatever recordings exist (none is fine: the source arm may be about to produce them). */
  static load(dir: string, armId?: string): ReplayIndex {
    const convDir = ReplayIndex.conversationsDir(dir);
    const recs = existsSync(convDir)
      ? readdirSync(convDir)
          .filter((f) => f.endsWith(".json"))
          .map((f) => JSON.parse(readFileSync(join(convDir, f), "utf8")) as ConversationRecord)
          .filter((r) => !r.truncated)
      : [];
    return new ReplayIndex(recs, armId);
  }

  /** True when `armId` is the arm whose recordings are replayed (it runs with the simulator itself). */
  isSourceArm(armId: string): boolean {
    return this.armId === armId;
  }

  /** The recording for this scenario and repeat: r-th matching recording, wrapping around. */
  pick(scenarioId: string, repeat: number): ConversationRecord | undefined {
    const pool = this.recs
      .filter((r) => r.scenarioId === scenarioId && (!this.armId || r.arm.id === this.armId))
      .sort((a, b) => a.id.localeCompare(b.id));
    return pool.length ? pool[repeat % pool.length] : undefined;
  }
}

/** Conversation id → file name (arm ids like "v2-structured/sonnet" contain path separators). */
export const recordFileName = (id: string) => `${id.replace(/[\/:*?"<>|]/g, "-")}.json`;

const emptyTotals = (): UsageTotals => ({ promptTokens: 0, completionTokens: 0, cachedPromptTokens: 0, calls: 0 });
function addUsage(t: UsageTotals, u?: TokenUsage, model?: string) {
  if (!u) return;
  t.promptTokens += u.promptTokens;
  t.completionTokens += u.completionTokens;
  t.cachedPromptTokens! += u.cachedPromptTokens ?? 0;
  t.calls++;
  if (model) t.model = model;
}

/** Generate one simulated conversation for a (scenario, arm) pair. */
/** Run a streaming reply to completion and return the recorded turn. */
export async function drain<T>(gen: AsyncGenerator<string, T>): Promise<T> {
  let r = await gen.next();
  while (!r.done) r = await gen.next();
  return r.value;
}

export async function runConversation(
  scenario: Scenario,
  arm: Arm,
  cfg: Pick<RunConfig, "userTurns" | "simulatorModel" | "memoryModel">,
  repeat = 0,
  hooks: RunHooks = {},
  ctx: RunContext = {},
): Promise<ConversationRecord> {
  const card = findCharacter(scenario.characterId);
  const budget = ctx.budget ?? new BudgetTracker();
  const id = `${scenario.id}__${arm.id}__r${repeat}`;

  // One Langfuse trace per conversation (no-op without credentials); every provider call below
  // becomes a generation tagged with its role and the current user turn.
  let userTurnNow = 0;
  const trace = startTrace(`eval:${scenario.id}/${arm.id}`, {
    input: { scenarioId: scenario.id, arm, repeat, userTurns: cfg.userTurns },
    metadata: { scenarioId: scenario.id, armId: arm.id, promptVersion: arm.promptVersion, model: arm.model, repeat: String(repeat) },
    tags: [scenario.id, arm.id, arm.promptVersion, ctx.replay ? "replay" : "simulated"],
  });
  const track = (p: LLMProvider, role: string): LLMProvider =>
    observedProvider(trackedProvider(p, budget), trace, role, () => ({ userTurn: userTurnNow }));

  const chat = resolveProvider(arm.model);
  const chatProvider = track(chat.provider, "character");
  // Memory consolidation: its own spec (e.g. "claude-cli:haiku"), else the character's provider.
  const mem = chat.provider.id === "fake" || !cfg.memoryModel ? { provider: chat.provider, model: undefined } : resolveProvider(cfg.memoryModel);
  const memoryProvider = track(mem.provider, "memory");

  const session = new ChatSession({
    card,
    lang: scenario.lang,
    provider: chatProvider,
    memoryProvider,
    promptVersion: arm.promptVersion,
    model: chat.model,
    temperature: arm.temperature,
    thinking: arm.thinking,
    persona: scenario.persona,
    memory: { model: mem.model, ...(arm.memory ?? {}) },
    filters: arm.filters,
  });

  let user: UserSource;
  let replayOf: string | undefined;
  const src = ctx.replay?.pick(scenario.id, repeat);
  if (src) {
    user = new ScriptedUser(src);
    replayOf = src.id;
  } else {
    if (ctx.replay && !ctx.replay.isSourceArm(arm.id)) throw new Error(`replayFrom: no recording for scenario ${scenario.id} to replay into arm ${arm.id}`);
    const sim = cfg.simulatorModel ? resolveProvider(cfg.simulatorModel) : chat;
    user = new UserSimulator(track(sim.provider, "simulator"), scenario, sim.model);
  }

  const name = card.name[scenario.lang] ?? card.id;
  const startedAt = new Date().toISOString();
  const totalTurns = user.length !== undefined ? Math.min(cfg.userTurns, user.length) : cfg.userTurns;

  session.open();
  const usage: TokenUsage & { calls: number } = { promptTokens: 0, completionTokens: 0, cachedPromptTokens: 0, calls: 0 };
  const simUsage = emptyTotals();
  let truncated = false;
  let charModel = chat.model ?? chat.provider.defaultModel;
  const memorySnapshots: ConversationRecord["memorySnapshots"] = [];
  let consolidations = 0;

  for (let userTurn = 1; userTurn <= totalTurns; userTurn++) {
    userTurnNow = userTurn;
    try {
      const msg = await user.next(userTurn, session.turns, name);
      addUsage(simUsage, msg.usage, msg.model);
      const reply = await drain(session.sendStream(msg.text)); // streamed, as the chat CLI serves it: first-token time is recorded
      await session.settle(); // consolidation runs after the reply; the snapshot below wants it finished
      if (reply.meta?.usage) {
        addUsage(usage as UsageTotals, reply.meta.usage);
        charModel = reply.meta.model ?? charModel;
      }
      if (session.memoryUsage.calls > consolidations) {
        consolidations = session.memoryUsage.calls;
        memorySnapshots.push({ afterTurnIndex: session.turns.length - 1, memory: structuredClone(session.memory) });
      }
    } catch (e) {
      if (e instanceof BudgetExceededError) {
        truncated = true;
        break;
      }
      throw e;
    }
    hooks.onTurn?.({ conversationId: id, userTurn, of: totalTurns });
  }

  const memoryUsage: UsageTotals = { ...session.memoryUsage };
  const costUsd =
    costOf(charModel, usage).usd +
    (simUsage.model ? costOf(simUsage.model, simUsage).usd : 0) +
    (memoryUsage.model ? costOf(memoryUsage.model, memoryUsage).usd : 0);

  const rec: ConversationRecord = {
    id,
    scenarioId: scenario.id,
    arm,
    lang: scenario.lang,
    persona: scenario.persona,
    turns: session.turns,
    memory: session.memory,
    memorySnapshots,
    usage,
    simUsage,
    memoryUsage,
    costUsd,
    ...(replayOf ? { replayOf } : {}),
    ...(truncated ? { truncated } : {}),
    ...(trace.traceId ? { traceId: trace.traceId } : {}),
    startedAt,
    finishedAt: new Date().toISOString(),
  };
  trace.update({ output: { turns: session.turns.length, usage, simUsage, memoryUsage, costUsd, truncated, memory: session.memory } });
  trace.end();
  hooks.onConversation?.(rec);
  return rec;
}

/**
 * Run the whole matrix with bounded concurrency, writing each conversation as it finishes.
 * Stops scheduling new conversations once the budget cap is exceeded.
 */
export async function runMatrix(cfg: RunConfig, concurrency = 4, hooks: RunHooks = {}, ctx: RunContext = {}): Promise<ConversationRecord[]> {
  const dir = join(cfg.outDir, "conversations");
  mkdirSync(dir, { recursive: true });
  const skipped: ConversationRecord[] = [];
  const budget = ctx.budget ?? new BudgetTracker(cfg.maxUsd);
  const replaySrcDir = cfg.replayFrom ? ReplayIndex.conversationsDir(cfg.replayFrom) : undefined;

  // Self-replay: if the replay source arm is itself in `arms`, phase 1 runs it with the simulator
  // and phase 2 replays its recordings into the other arms. Otherwise there is a single phase.
  const sourceArm = cfg.replayArmId && cfg.arms.some((a) => a.id === cfg.replayArmId) ? cfg.replayArmId : undefined;
  const phases: Array<Array<() => Promise<ConversationRecord>>> = [[], []];

  for (const sid of cfg.scenarioIds) {
    const scenario = findScenario(sid);
    for (const arm of cfg.arms) {
      for (let r = 0; r < cfg.repeats; r++) {
        const id = `${scenario.id}__${arm.id}__r${r}`;
        const existing = join(dir, recordFileName(id));
        if (ctx.resume !== false) {
          // Resume: a finished recording already on disk is reused (lets a run continue after a
          // subscription usage-window limit without redoing paid work).
          let rec = existsSync(existing) ? (JSON.parse(readFileSync(existing, "utf8")) as ConversationRecord) : undefined;
          // An identical (scenario, arm, repeat) recording in the replay source counts too, so a
          // comparison config can list the source arm without regenerating it.
          const srcFile = replaySrcDir ? join(replaySrcDir, recordFileName(id)) : undefined;
          if (!rec && srcFile && existsSync(srcFile)) {
            const cand = JSON.parse(readFileSync(srcFile, "utf8")) as ConversationRecord;
            if (!cand.truncated && cand.arm.promptVersion === arm.promptVersion && cand.arm.model === arm.model) {
              rec = cand;
              writeFileSync(existing, JSON.stringify(rec, null, 2));
            }
          }
          if (rec && !rec.truncated) {
            skipped.push(rec);
            continue;
          }
        }
        const phase = sourceArm && arm.id !== sourceArm ? 1 : 0;
        phases[phase]!.push(async () => {
          const replay =
            ctx.replay ??
            (cfg.replayFrom ? ReplayIndex.load(phase === 1 ? cfg.outDir : cfg.replayFrom, cfg.replayArmId) : undefined);
          const rec = await runConversation(scenario, arm, cfg, r, hooks, { budget, replay });
          writeFileSync(join(dir, recordFileName(rec.id)), JSON.stringify(rec, null, 2));
          return rec;
        });
      }
    }
  }

  if (skipped.length) hooks.onSkip?.(skipped.map((r) => r.id));
  const results: ConversationRecord[] = [...skipped];
  for (const jobs of phases) {
    if (!jobs.length) continue;
    let next = 0;
    const workers = Array.from({ length: Math.min(concurrency, jobs.length) }, async () => {
      while (next < jobs.length && !budget.exceeded) {
        const job = jobs[next++]!;
        results.push(await job());
      }
    });
    await Promise.all(workers);
  }
  return results;
}
