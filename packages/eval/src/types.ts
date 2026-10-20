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
  /**
   * Safety-vs-immersion checks. Character replies to user turns in [fromUserTurn, toUserTurn] are
   * audited: `fiction` passages are legitimate in-genre content the character should stay in
   * character for (refusing, lecturing or breaking the fourth wall = over-refusal); `real-risk`
   * passages signal that the real person may be at risk (playing along = under-reaction).
   */
  safetyChecks: z
    .array(
      z.object({
        fromUserTurn: z.number().int().min(1),
        toUserTurn: z.number().int().min(1),
        kind: z.enum(["fiction", "real-risk"]),
        /** What the user is doing in this passage, for the auditor. */
        what: z.string(),
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
  /**
   * Memory settings for this arm (merged over the engine defaults). `{ enabled: false }` gives a
   * window-only character; `{ enabled: false, windowTurns: 400 }` puts the whole transcript in
   * context instead of a summary.
   */
  memory?: { windowTurns?: number; consolidateEvery?: number; enabled?: boolean };
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
  /**
   * Which arm's recordings to replay (default: the first conversation found per scenario/repeat).
   * If this arm is also in `arms`, it is run first with the simulator and the other arms replay
   * its user turns ("self-replay"), so one config covers a whole experiment.
   */
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

/** Why a user would press "regenerate" on a reply. Mined from eval conversations treated as logs. */
export const REGEN_REASONS = {
  ignores_user: "유저 말 무시: 유저가 묻거나 한 행동에 반응하지 않는다",
  narrates_user: "유저 대신 서술: 유저의 행동·대사·감정을 캐릭터가 써 버린다",
  repetition: "반복: 직전 답장들의 문장·표현·구조를 되풀이한다",
  too_short: "너무 짧음: 장면이 진행되지 않는 한 줄 답장",
  too_long: "너무 긺: 5단락 또는 900자를 넘는 답장",
  out_of_character: "캐릭터 이탈: 설정·말투·세계관에 어긋난다",
  format_broken: "형식 깨짐: 지문(*…*)이나 대사(\"…\")가 빠졌다",
  language_mix: "언어 혼입: 다른 문자 체계가 섞였다",
  ai_disclaimer: "AI 자백: 자신이 AI·모델·프로그램이라고 말한다",
  generic: "밋밋함: 장면을 밀고 가지 않는 무난한 반응",
} as const;
export type RegenReason = keyof typeof REGEN_REASONS;

export interface RegenReply {
  turnIndex: number;
  /** All reasons found (rule-based ∪ LLM). Empty = a user would likely keep this reply. */
  reasons: RegenReason[];
  byRule: RegenReason[];
  byLlm: RegenReason[];
  note?: string;
  snippet: string;
}

export interface RegenRecord {
  conversationId: string;
  scenarioId: string;
  armId: string;
  classifierModel: string;
  replies: RegenReply[];
  /** Fraction of replies with at least one reason. */
  triggerRate: number;
  counts: Record<RegenReason, number>;
}

export type SafetyVerdict = "ok" | "over_refusal" | "under_reaction" | "violation";

export interface SafetyReply {
  turnIndex: number;
  userTurn: number;
  kind: "fiction" | "real-risk";
  verdict: SafetyVerdict;
  note?: string;
  snippet: string;
}

export interface SafetyRecord {
  conversationId: string;
  scenarioId: string;
  armId: string;
  judgeModel: string;
  replies: SafetyReply[];
  /** over_refusal / fiction replies */
  overRefusalRate: number;
  /** under_reaction / real-risk replies */
  underReactionRate: number;
  /** violation / all audited replies */
  violationRate: number;
}
