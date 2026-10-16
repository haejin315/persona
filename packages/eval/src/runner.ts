import { ChatSession, findCharacter, type TokenUsage } from "@persona/engine";
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
}

export interface RunContext {
  budget?: BudgetTracker;
  replay?: ReplayIndex;
}

/** Recorded conversations to replay the user side from. */
export class ReplayIndex {
  constructor(private recs: ConversationRecord[], private armId?: string) {}

  static load(dir: string, armId?: string): ReplayIndex {
    const convDir = basename(dir) === "conversations" ? dir : join(dir, "conversations");
    if (!existsSync(convDir)) throw new Error(`replayFrom: no conversations in ${convDir}`);
    const recs = readdirSync(convDir)
      .filter((f) => f.endsWith(".json"))
      .map((f) => JSON.parse(readFileSync(join(convDir, f), "utf8")) as ConversationRecord)
      .filter((r) => !r.truncated);
    return new ReplayIndex(recs, armId);
  }

  /** The recording for this scenario and repeat: r-th matching recording, wrapping around. */
  pick(scenarioId: string, repeat: number): ConversationRecord {
    const pool = this.recs
      .filter((r) => r.scenarioId === scenarioId && (!this.armId || r.arm.id === this.armId))
      .sort((a, b) => a.id.localeCompare(b.id));
    if (!pool.length) throw new Error(`replayFrom: no recording for scenario ${scenarioId}${this.armId ? ` / arm ${this.armId}` : ""}`);
    return pool[repeat % pool.length]!;
  }
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

/** Generate one simulated conversation for a (scenario, arm) pair. */
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
  const track = <P extends { generate: Function; stream: Function; id: string; defaultModel: string }>(p: P) => trackedProvider(p, budget);

  const chat = resolveProvider(arm.model);
  const chatProvider = track(chat.provider);
  const memoryProvider = chat.provider.id === "fake" ? chatProvider : track(resolveProvider("openai").provider);

  const session = new ChatSession({
    card,
    lang: scenario.lang,
    provider: chatProvider,
    memoryProvider,
    promptVersion: arm.promptVersion,
    model: chat.model,
    temperature: arm.temperature,
    persona: scenario.persona,
    memory: { model: cfg.memoryModel },
  });

  let user: UserSource;
  let replayOf: string | undefined;
  if (ctx.replay) {
    const src = ctx.replay.pick(scenario.id, repeat);
    user = new ScriptedUser(src);
    replayOf = src.id;
  } else {
    const sim = cfg.simulatorModel ? resolveProvider(cfg.simulatorModel) : chat;
    user = new UserSimulator(track(sim.provider), scenario, sim.model);
  }

  const name = card.name[scenario.lang] ?? card.id;
  const id = `${scenario.id}__${arm.id}__r${repeat}`;
  const startedAt = new Date().toISOString();
  const totalTurns = user.length !== undefined ? Math.min(cfg.userTurns, user.length) : cfg.userTurns;

  session.open();
  const usage: TokenUsage & { calls: number } = { promptTokens: 0, completionTokens: 0, cachedPromptTokens: 0, calls: 0 };
  const simUsage = emptyTotals();
  let truncated = false;
  let charModel = chat.model ?? chat.provider.defaultModel;

  for (let userTurn = 1; userTurn <= totalTurns; userTurn++) {
    try {
      const msg = await user.next(userTurn, session.turns, name);
      addUsage(simUsage, msg.usage, msg.model);
      const reply = await session.send(msg.text);
      if (reply.meta?.usage) {
        addUsage(usage as UsageTotals, reply.meta.usage);
        charModel = reply.meta.model ?? charModel;
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
    usage,
    simUsage,
    memoryUsage,
    costUsd,
    ...(replayOf ? { replayOf } : {}),
    ...(truncated ? { truncated } : {}),
    startedAt,
    finishedAt: new Date().toISOString(),
  };
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
  const budget = ctx.budget ?? new BudgetTracker(cfg.maxUsd);
  const replay = ctx.replay ?? (cfg.replayFrom ? ReplayIndex.load(cfg.replayFrom, cfg.replayArmId) : undefined);

  const jobs: Array<() => Promise<ConversationRecord>> = [];
  for (const sid of cfg.scenarioIds) {
    const scenario = findScenario(sid);
    for (const arm of cfg.arms) {
      for (let r = 0; r < cfg.repeats; r++) {
        jobs.push(async () => {
          const rec = await runConversation(scenario, arm, cfg, r, hooks, { budget, replay });
          writeFileSync(join(dir, `${rec.id}.json`), JSON.stringify(rec, null, 2));
          return rec;
        });
      }
    }
  }

  const results: ConversationRecord[] = [];
  let next = 0;
  const workers = Array.from({ length: Math.min(concurrency, jobs.length) }, async () => {
    while (next < jobs.length && !budget.exceeded) {
      const job = jobs[next++]!;
      results.push(await job());
    }
  });
  await Promise.all(workers);
  return results;
}
