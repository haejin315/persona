import { costOf, fmtUsd, priceFor } from "./pricing.js";
import { findScenario } from "./scenarios.js";
import type { ConversationRecord, RunConfig } from "./types.js";

/**
 * Pre-run cost estimate. Uses a default per-call token profile (from docs/01-unit-economics.md and
 * the smoke run), overridden by measured usage from earlier conversations when available. The
 * point is to see the number *before* spending, and to size `maxUsd`.
 */
export interface EstimateRow {
  role: "character" | "simulator" | "memory" | "judge" | "probe";
  model: string;
  calls: number;
  promptTokens: number;
  cachedTokens: number;
  outputTokens: number;
  usd: number;
  billedUsd: number;
  measured: boolean;
}

export interface Estimate {
  conversations: number;
  rows: EstimateRow[];
  usd: number;
  billedUsd: number;
  notes: string[];
}

interface Profile {
  prompt: number;
  cachedShare: number;
  out: number;
}

/** Per-call defaults. Character prompt grows with the window; these are 50-turn averages. */
const DEFAULTS = {
  character: { "v1-flat": { prompt: 4_500, cachedShare: 0.8, out: 350 }, "v2-structured": { prompt: 6_900, cachedShare: 0.85, out: 350 } } as Record<string, Profile>,
  simulator: { prompt: 1_800, cachedShare: 0.4, out: 120 } as Profile,
  memory: { prompt: 1_500, cachedShare: 0, out: 250 } as Profile,
  judge: { prompt: 3_500, cachedShare: 0.3, out: 150 } as Profile,
  probe: { prompt: 500, cachedShare: 0.3, out: 30 } as Profile,
};
/** Claude Code CLI adds a fixed, cached scaffold to every call (measured 830 tokens). */
const CLAUDE_CLI_OVERHEAD = 830;

function row(role: EstimateRow["role"], model: string, calls: number, p: Profile, measured = false): EstimateRow {
  const isCli = model.startsWith("claude-cli");
  const prompt = Math.round(calls * (p.prompt + (isCli ? CLAUDE_CLI_OVERHEAD : 0)));
  const cached = Math.round(calls * (p.prompt * p.cachedShare + (isCli ? CLAUDE_CLI_OVERHEAD : 0)));
  const out = Math.round(calls * p.out);
  const c = costOf(model, { promptTokens: prompt, completionTokens: out, cachedPromptTokens: cached });
  return { role, model, calls, promptTokens: prompt, cachedTokens: cached, outputTokens: out, usd: c.usd, billedUsd: c.billedUsd, measured };
}

/** Average per-call profile from recorded conversations, if any match. */
function measuredProfile(recs: ConversationRecord[], pick: (r: ConversationRecord) => { promptTokens: number; completionTokens: number; cachedPromptTokens?: number; calls: number } | undefined): Profile | undefined {
  const us = recs.map(pick).filter((u) => u && u.calls > 0) as NonNullable<ReturnType<typeof pick>>[];
  const calls = us.reduce((a, u) => a + u.calls, 0);
  if (!calls) return undefined;
  const prompt = us.reduce((a, u) => a + u.promptTokens, 0) / calls;
  const cached = us.reduce((a, u) => a + (u.cachedPromptTokens ?? 0), 0) / calls;
  return { prompt, cachedShare: prompt ? cached / prompt : 0, out: us.reduce((a, u) => a + u.completionTokens, 0) / calls };
}

export function estimateRun(cfg: RunConfig, judgeSpec: string, measured: ConversationRecord[] = []): Estimate {
  const notes: string[] = [];
  const rows: EstimateRow[] = [];
  const scenarios = cfg.scenarioIds.map((id) => findScenario(id));
  const perScenarioConvs = cfg.arms.length * cfg.repeats;
  const conversations = scenarios.length * perScenarioConvs;
  const replies = cfg.userTurns;
  // Memory consolidates every 10 turns that fall outside the 20-turn window (2 turns per exchange).
  const consolidations = Math.max(0, Math.floor((2 * cfg.userTurns - 20) / 10));
  const checkpoints = Math.floor(cfg.userTurns / 10);
  const probes = scenarios.reduce((a, s) => a + s.probes.filter((p) => p.atUserTurn <= cfg.userTurns).length * perScenarioConvs, 0);

  // Character replies, per arm (model and prompt version differ).
  for (const arm of cfg.arms) {
    const mine = measured.filter((r) => r.arm.model === arm.model && r.arm.promptVersion === arm.promptVersion);
    const m = measuredProfile(mine, (r) => r.usage);
    const p = m ?? DEFAULTS.character[arm.promptVersion] ?? DEFAULTS.character["v2-structured"]!;
    rows.push(row("character", arm.model, scenarios.length * cfg.repeats * replies, p, Boolean(m)));
  }

  // Simulator — none in replay mode.
  if (cfg.replayFrom) notes.push("replay mode: user turns come from recordings, no simulator cost");
  else {
    const simModel = cfg.simulatorModel ?? cfg.arms[0]!.model;
    const m = measuredProfile(measured.filter((r) => r.simUsage?.model && r.simUsage.model.includes(simModel.replace(/^openai:/, ""))), (r) => r.simUsage);
    rows.push(row("simulator", simModel, conversations * replies, m ?? DEFAULTS.simulator, Boolean(m)));
  }

  // Memory consolidation (always the OpenAI memory model unless the arm is fake).
  const memModel = cfg.arms.every((a) => a.model.startsWith("fake")) ? "fake" : `openai:${cfg.memoryModel ?? process.env.MEMORY_MODEL ?? "gpt-5-nano"}`;
  const mm = measuredProfile(measured.filter((r) => r.memoryUsage?.model), (r) => r.memoryUsage);
  rows.push(row("memory", memModel, conversations * consolidations, mm ?? DEFAULTS.memory, Boolean(mm)));

  // Judge: one rubric call per checkpoint, one small call per probe.
  rows.push(row("judge", judgeSpec, conversations * checkpoints, DEFAULTS.judge));
  rows.push(row("probe", judgeSpec, probes, DEFAULTS.probe));

  for (const r of rows) if (!priceFor(r.model)) notes.push(`no price for ${r.model}; counted as $0`);
  if (rows.some((r) => r.model.startsWith("claude-cli"))) notes.push("claude-cli rows are paid by the Claude subscription (billed $0); list price shown for comparison");
  if (cfg.maxUsd !== undefined) notes.push(`maxUsd ${fmtUsd(cfg.maxUsd)} configured`);

  return {
    conversations,
    rows,
    usd: rows.reduce((a, r) => a + r.usd, 0),
    billedUsd: rows.reduce((a, r) => a + r.billedUsd, 0),
    notes,
  };
}

export function estimateToMarkdown(e: Estimate, cfg: RunConfig): string {
  const out = [
    `## Cost estimate · ${e.conversations} conversations × ${cfg.userTurns} user turns`,
    "",
    "| role | model | calls | prompt tok | cached | output tok | list $ | billed $ | basis |",
    "|---|---|---:|---:|---:|---:|---:|---:|---|",
  ];
  for (const r of e.rows)
    out.push(
      `| ${r.role} | ${r.model} | ${r.calls} | ${r.promptTokens.toLocaleString()} | ${r.cachedTokens.toLocaleString()} | ${r.outputTokens.toLocaleString()} | ${fmtUsd(r.usd)} | ${fmtUsd(r.billedUsd)} | ${r.measured ? "measured" : "default"} |`,
    );
  out.push(`| **total** | | | | | | **${fmtUsd(e.usd)}** | **${fmtUsd(e.billedUsd)}** | |`, "");
  for (const n of e.notes) out.push(`- ${n}`);
  if (e.billedUsd > 0 && cfg.maxUsd === undefined) out.push(`- suggestion: set "maxUsd": ${Math.ceil(e.billedUsd * 1.5 * 10) / 10} in the config (1.5× estimate)`);
  return out.join("\n");
}
