import type { Lang, MemoryState, TokenUsage, Turn, UserPersona } from "@persona/engine";
import { z } from "zod";

/**
 * A scenario is a scripted user: who they are, what they reveal and when, and which facts the
 * character is later probed on. It makes 100-turn conversations reproducible enough to compare
 * prompt versions and models against each other.
 */
export const Scenario = z.object({
  id: z.string(),
  characterId: z.string(),
  lang: z.enum(["ko", "ja", "en"]),
  persona: z.object({ name: z.string(), description: z.string().optional() }),
  /** How the simulated user writes: tone, length, how often they use *actions*. */
  userStyle: z.string(),
  /** Ordered beats. The simulator moves to the next beat when its turn range is reached. */
  beats: z.array(
    z.object({
      /** First user turn (1-based, counting user messages) where this beat is active. */
      fromUserTurn: z.number().int().min(1),
      /** What the user is trying to do during this beat. */
      goal: z.string(),
      /** Facts the user must reveal naturally during this beat (ground truth for recall probes). */
      reveal: z.array(z.string()).default([]),
    }),
  ),
  /** Recall probes: at user turn N, the user asks something whose answer is `expected`. */
  probes: z
    .array(
      z.object({
        atUserTurn: z.number().int().min(1),
        ask: z.string(),
        expected: z.string(),
      }),
    )
    .default([]),
});
export type Scenario = z.infer<typeof Scenario>;

/** One (character, prompt version, model) configuration under test. */
export interface Arm {
  id: string;
  promptVersion: string;
  /** "openai:gpt-5-mini" etc. "fake" runs offline for tests. */
  model: string;
  temperature?: number;
}

export interface RunConfig {
  scenarioIds: string[];
  arms: Arm[];
  /** Total user turns per conversation. */
  userTurns: number;
  /** Simulated-user model spec; defaults to the arm's model. */
  simulatorModel?: string;
  memoryModel?: string;
  /** Repeat each (scenario, arm) this many times to average out sampling noise. */
  repeats: number;
  outDir: string;
  /** Judge spec used by `all` / `judge` when --judge is not given, e.g. "claude-cli:sonnet". */
  judgeModel?: string;
  /**
   * Stop the run once *billed* cost (list price of calls that draw on prepaid credit) passes
   * this many USD. Conversations in flight are written out truncated. Subscription (claude-cli)
   * and offline calls count as $0.
   */
  maxUsd?: number;
  /**
   * Replay mode: reuse the user side of previously recorded conversations instead of running the
   * simulator. Points at an eval output dir (or its conversations/ folder). Every arm then sees
   * identical user input, which removes simulator cost and cuts variance between arms.
   */
  replayFrom?: string;
  /** Which arm's recordings to replay (default: the first conversation found per scenario/repeat). */
  replayArmId?: string;
}

export interface UsageTotals extends TokenUsage {
  calls: number;
  /** Model the calls were made with (last seen), for pricing. */
  model?: string;
}

export interface ConversationRecord {
  id: string;
  scenarioId: string;
  arm: Arm;
  lang: Lang;
  persona: UserPersona;
  turns: Turn[];
  memory: MemoryState;
  /** Character-side usage summed over the conversation. */
  usage: TokenUsage & { calls: number };
  /** Simulated-user and memory-consolidation usage, so the whole run can be priced. */
  simUsage?: UsageTotals;
  memoryUsage?: UsageTotals;
  /** List-price cost of this conversation (character + simulator + memory), USD. */
  costUsd?: number;
  /** Set when the user side was replayed from a recording. */
  replayOf?: string;
  /** Set when the run stopped early (budget cap) — fewer user turns than configured. */
  truncated?: boolean;
  startedAt: string;
  finishedAt: string;
}

export const METRICS = ["consistency", "memory", "style", "format"] as const;
export type Metric = (typeof METRICS)[number];

export interface CheckpointScore {
  /** Character turn index (into `turns`) that ends this window. */
  turnIndex: number;
  consistency: number;
  memory: number;
  style: number;
  /** Rule-based: fraction of format rules satisfied, 0–1 → scaled to 1–5. */
  format: number;
  /** One-line judge rationale per metric, for the report and for debugging the prompt. */
  notes: Partial<Record<Metric, string>>;
}

export interface ProbeResult {
  atUserTurn: number;
  ask: string;
  expected: string;
  answer: string;
  /** Judge verdict. */
  recalled: boolean;
}

export interface ScoreRecord {
  conversationId: string;
  scenarioId: string;
  armId: string;
  judgeModel: string;
  checkpoints: CheckpointScore[];
  probes: ProbeResult[];
  /** Mean over checkpoints per metric, 1–5. */
  mean: Record<Metric, number>;
  probeRecall: number;
}
