/**
 *   npm run eval -- estimate experiments/prompt-version.json [--judge spec] [--from <outDir>]
 *   npm run eval -- run      experiments/smoke.json          # generate conversations (resumes; --fresh redoes)
 *   npm run eval -- judge    eval-results/smoke [--judge fake|openai:gpt-5|claude-cli:sonnet] [--format-only]
 *   npm run eval -- report   eval-results/smoke
 *   npm run eval -- regen    eval-results/smoke [--judge spec] [--rules-only]   # regenerate-trigger mining (--rules-only keeps earlier LLM labels)
 *   npm run eval -- safety   eval-results/safety [--judge spec]                 # safety-vs-immersion audit
 *   npm run eval -- all      experiments/smoke.json          # run + judge + report + regen (+ safety if scenarios have checks)
 *   npm run eval -- backfill eval-results/prompt-version [more dirs…] [--include-fake] [--again] [--match <regex>]   # push recorded runs to Langfuse
 *
 * Cost controls: `estimate` prints the expected bill before anything runs; `maxUsd` in the config
 * stops a run once billed cost passes the cap; `replayFrom` reuses recorded user turns; a judge
 * spec of `claude-cli:<model>` grades through the Claude Code subscription instead of API credit.
 */
import "dotenv/config";
import { flushTracing, initTracing, observedProvider, scoreTrace, startTrace, tracingEnabled, type LLMProvider } from "@persona/engine";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { BudgetTracker, trackedProvider } from "./budget.js";
import { estimateRun, estimateToMarkdown } from "./estimate.js";
import { judgeConversation } from "./judge.js";
import { fmtUsd } from "./pricing.js";
import { resolveProvider } from "./providers.js";
import { classifyRegen, regenToMarkdown, summariseRegen } from "./regen.js";
import { summarise, toMarkdown } from "./report.js";
import { recordFileName, runMatrix } from "./runner.js";
import { auditSafety, safetyToMarkdown, summariseSafety } from "./safety.js";
import { findScenario } from "./scenarios.js";
import type { ConversationRecord, RegenRecord, RunConfig, SafetyRecord, ScoreRecord } from "./types.js";

const REPO_ROOT = resolve(new URL("../../..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));

function flag(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 ? process.argv[i + 1] : undefined;
}

function loadConfig(path: string): RunConfig {
  const cfg = JSON.parse(readFileSync(path, "utf8")) as RunConfig;
  cfg.outDir = resolve(REPO_ROOT, cfg.outDir);
  if (cfg.replayFrom) cfg.replayFrom = resolve(REPO_ROOT, cfg.replayFrom);
  return cfg;
}

function readJsonDir<T>(dir: string): T[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => JSON.parse(readFileSync(join(dir, f), "utf8")) as T);
}

function judgeSpec(cfg?: RunConfig): string {
  return flag("judge") ?? cfg?.judgeModel ?? process.env.JUDGE_MODEL ?? "claude-cli:opus";
}

function printBudget(label: string, b: BudgetTracker) {
  const unknown = b.unknownModels.size ? `  (no price for: ${[...b.unknownModels].join(", ")})` : "";
  console.log(`${label}: ${b.calls} calls · list ${fmtUsd(b.listUsd)} · billed ${fmtUsd(b.spentUsd)}${unknown}`);
}

function cmdEstimate(cfgPath: string) {
  const cfg = loadConfig(cfgPath);
  const from = flag("from");
  const measured = readJsonDir<ConversationRecord>(join(from ? resolve(REPO_ROOT, from) : cfg.outDir, "conversations"));
  const md = estimateToMarkdown(estimateRun(cfg, judgeSpec(cfg), measured), cfg);
  console.log(md);
  if (measured.length) console.log(`\n(character/simulator/memory rows marked "measured" use ${measured.length} recorded conversations)`);
}

async function cmdRun(cfgPath: string) {
  const cfg = loadConfig(cfgPath);
  const total = cfg.scenarioIds.length * cfg.arms.length * cfg.repeats;
  const budget = new BudgetTracker(cfg.maxUsd);
  console.log(
    `run: ${total} conversations × ${cfg.userTurns} user turns → ${cfg.outDir}` +
      (cfg.replayFrom ? `  (replaying user turns from ${cfg.replayFrom})` : "") +
      (cfg.maxUsd !== undefined ? `  (cap ${fmtUsd(cfg.maxUsd)})` : ""),
  );
  const t0 = Date.now();
  await runMatrix(
    cfg,
    Number(flag("concurrency") ?? 4),
    {
      onTurn: ({ conversationId, userTurn, of }) => {
        if (userTurn % 10 === 0 || userTurn === of) console.log(`  ${conversationId}: ${userTurn}/${of}  (billed so far ${fmtUsd(budget.spentUsd)})`);
      },
      onConversation: (rec) =>
        console.log(
          `${rec.truncated ? "✗ TRUNCATED" : "✓"} ${rec.id}  prompt ${rec.usage.promptTokens} (cached ${rec.usage.cachedPromptTokens}) / out ${rec.usage.completionTokens} tokens · ${fmtUsd(rec.costUsd ?? 0)}`,
        ),
      onSkip: (ids) => console.log(`  resuming: ${ids.length} finished conversation(s) already on disk (use --fresh to redo)`),
    },
    { budget, resume: !process.argv.includes("--fresh") },
  );
  console.log(`done in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  printBudget("run cost", budget);
  if (budget.exceeded) console.log(`!! budget cap ${fmtUsd(cfg.maxUsd!)} hit — remaining conversations were skipped`);
  return cfg;
}

async function cmdJudge(outDir: string, cfg?: RunConfig) {
  const spec = judgeSpec(cfg);
  const budget = new BudgetTracker(cfg?.maxUsd);
  const { provider: raw, model } = resolveProvider(spec);
  const provider = trackedProvider(raw, budget);
  const convs = readJsonDir<ConversationRecord>(join(outDir, "conversations"));
  const scoreDir = join(outDir, "scores");
  mkdirSync(scoreDir, { recursive: true });
  const concurrency = Number(flag("concurrency") ?? 4);
  const formatOnly = process.argv.includes("--format-only");
  console.log(`judge: ${convs.length} conversations with ${formatOnly ? "earlier LLM scores (format recomputed)" : `${raw.id}:${model ?? raw.defaultModel}`} (concurrency ${concurrency})`);

  let next = 0;
  const workers = Array.from({ length: Math.min(concurrency, convs.length) }, async () => {
    while (next < convs.length) {
      const rec = convs[next++]!;
      const priorFile = join(scoreDir, recordFileName(rec.id));
      const reuse = formatOnly && existsSync(priorFile) ? (JSON.parse(readFileSync(priorFile, "utf8")) as ScoreRecord) : undefined;
      const trace = startTrace(`judge:${rec.id}`, { metadata: { conversationId: rec.id, armId: rec.arm.id, judge: spec }, tags: ["judge", rec.scenarioId, rec.arm.id] });
      const score = await judgeConversation(rec, { provider: observedProvider(provider, trace, "judge"), model, reuse });
      trace.update({ output: { mean: score.mean, probeRecall: score.probeRecall } });
      trace.end();
      for (const m of ["consistency", "memory", "style", "format"] as const) scoreTrace(rec.traceId, m, score.mean[m]);
      scoreTrace(rec.traceId, "probe_recall", score.probeRecall);
      writeFileSync(join(scoreDir, recordFileName(rec.id)), JSON.stringify(score, null, 2));
      const m = score.mean;
      console.log(
        `✓ ${rec.id}  cons ${m.consistency.toFixed(1)}  mem ${m.memory.toFixed(1)}  style ${m.style.toFixed(1)}  fmt ${m.format.toFixed(1)}  recall ${Math.round(score.probeRecall * 100)}%`,
      );
    }
  });
  await Promise.all(workers);
  printBudget("judge cost", budget);
}

/** Run `fn` over every recorded conversation with bounded concurrency, writing one JSON per conversation. */
async function forEachConversation<T extends { conversationId: string }>(
  outDir: string,
  sub: string,
  fn: (rec: ConversationRecord, traced: (p: LLMProvider) => LLMProvider) => Promise<T>,
  onDone: (t: T, rec: ConversationRecord) => void,
): Promise<T[]> {
  const convs = readJsonDir<ConversationRecord>(join(outDir, "conversations"));
  const dir = join(outDir, sub);
  mkdirSync(dir, { recursive: true });
  const concurrency = Number(flag("concurrency") ?? 4);
  const out: T[] = [];
  let next = 0;
  const workers = Array.from({ length: Math.min(concurrency, convs.length) }, async () => {
    while (next < convs.length) {
      const rec = convs[next++]!;
      const trace = startTrace(`${sub}:${rec.id}`, { metadata: { conversationId: rec.id, armId: rec.arm.id }, tags: [sub, rec.scenarioId, rec.arm.id] });
      const t = await fn(rec, (p) => observedProvider(p, trace, sub));
      trace.update({ output: t });
      trace.end();
      writeFileSync(join(dir, recordFileName(t.conversationId)), JSON.stringify(t, null, 2));
      out.push(t);
      onDone(t, rec);
    }
  });
  await Promise.all(workers);
  return out;
}

async function cmdRegen(outDir: string, cfg?: RunConfig) {
  const rulesOnly = process.argv.includes("--rules-only");
  const spec = judgeSpec(cfg);
  const budget = new BudgetTracker(cfg?.maxUsd);
  const { provider: raw, model } = resolveProvider(spec);
  const provider = trackedProvider(raw, budget);
  console.log(`regen: classifying replies with ${rulesOnly ? "rules only" : `rules + ${raw.id}:${model ?? raw.defaultModel}`}`);
  const records = await forEachConversation<RegenRecord>(
    outDir,
    "regen",
    (rec, traced) => {
      const priorFile = join(outDir, "regen", recordFileName(rec.id));
      const reuse = rulesOnly && existsSync(priorFile) ? (JSON.parse(readFileSync(priorFile, "utf8")) as RegenRecord) : undefined;
      return classifyRegen(rec, { provider: traced(provider), model, rulesOnly, reuse });
    },
    (r, rec) => {
      scoreTrace(rec.traceId, "regen_trigger_rate", r.triggerRate);
      for (const [k, n] of Object.entries(r.counts)) if (n) scoreTrace(rec.traceId, `regen_${k}`, (100 * n) / Math.max(1, r.replies.length), "per 100 replies");
      const top = Object.entries(r.counts)
        .filter(([, n]) => n)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([k, n]) => `${k}x${n}`)
        .join(" ");
      console.log(`✓ ${r.conversationId}  trigger ${Math.round(r.triggerRate * 100)}%  ${top}`);
    },
  );
  const md = regenToMarkdown(summariseRegen(records), `Regenerate-trigger mining · ${basename(outDir)}`);
  writeFileSync(join(outDir, "regen.md"), md);
  console.log(md);
  if (!rulesOnly) printBudget("regen cost", budget);
}

async function cmdSafety(outDir: string, cfg?: RunConfig) {
  const spec = judgeSpec(cfg);
  const budget = new BudgetTracker(cfg?.maxUsd);
  const { provider: raw, model } = resolveProvider(spec);
  const provider = trackedProvider(raw, budget);
  console.log(`safety: auditing marked passages with ${raw.id}:${model ?? raw.defaultModel}`);
  const pct = (x: number) => (Number.isNaN(x) ? "–" : `${Math.round(x * 100)}%`);
  const records = await forEachConversation<SafetyRecord>(
    outDir,
    "safety",
    (rec, traced) => auditSafety(rec, { provider: traced(provider), model }),
    (r, rec) => {
      scoreTrace(rec.traceId, "safety_over_refusal", r.overRefusalRate);
      scoreTrace(rec.traceId, "safety_under_reaction", r.underReactionRate);
      scoreTrace(rec.traceId, "safety_violation", r.violationRate);
      console.log(`✓ ${r.conversationId}  over-refusal ${pct(r.overRefusalRate)}  under-reaction ${pct(r.underReactionRate)}  violation ${pct(r.violationRate)}`);
    },
  );
  const md = safetyToMarkdown(summariseSafety(records), `Safety vs immersion · ${basename(outDir)}`);
  writeFileSync(join(outDir, "safety.md"), md);
  console.log(md);
  printBudget("safety cost", budget);
}

/**
 * Push recorded conversations (and their judge / regen / safety results) to Langfuse as traces with
 * historical timestamps. For runs made before tracing existed, or for a new Langfuse project. Each
 * character reply becomes a generation (input = the user's message, output = the reply; the full
 * prompt is not stored in the record). The trace id is written back to the record so later
 * `judge` / `regen` / `safety` runs attach their scores to the same trace.
 */
async function cmdBackfill(dirs: string[]) {
  if (!tracingEnabled()) throw new Error("backfill needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and LANGFUSE_BASE_URL)");
  const includeFake = process.argv.includes("--include-fake");
  const again = process.argv.includes("--again");
  const match = flag("match") ? new RegExp(flag("match")!) : undefined;
  let traces = 0;
  let generations = 0;
  let scores = 0;
  for (const dir of dirs) {
    const experiment = basename(dir);
    const convDir = join(dir, "conversations");
    if (!existsSync(convDir)) continue;
    for (const file of readdirSync(convDir).filter((f) => f.endsWith(".json"))) {
      const path = join(convDir, file);
      const rec = JSON.parse(readFileSync(path, "utf8")) as ConversationRecord;
      if (!includeFake && (rec.arm.model === "fake" || rec.arm.model.startsWith("fake:"))) continue;
      if (match && !match.test(rec.id)) continue;
      if (rec.traceId && !again) continue;
      const started = new Date(rec.startedAt);
      const finished = new Date(rec.finishedAt);
      const trace = startTrace(`eval:${rec.scenarioId}/${rec.arm.id}`, {
        startTime: started,
        input: { scenarioId: rec.scenarioId, arm: rec.arm, replayOf: rec.replayOf },
        metadata: { experiment, scenarioId: rec.scenarioId, armId: rec.arm.id, promptVersion: rec.arm.promptVersion, model: rec.arm.model, backfill: "true" },
        tags: [experiment, rec.scenarioId, rec.arm.id, rec.arm.promptVersion, rec.replayOf ? "replay" : "simulated", "backfill"],
      });
      let userTurn = 0;
      let lastUser = "";
      for (const t of rec.turns) {
        if (t.role === "user") {
          userTurn++;
          lastUser = t.content;
          continue;
        }
        if (t.index === 0 || !t.meta) continue;
        const end = new Date(t.ts);
        const start = new Date(t.ts - (t.meta.latencyMs ?? 0));
        const gen = trace.generation("character", {
          model: t.meta.model,
          input: [{ role: "user", content: lastUser }],
          metadata: { role: "character", userTurn, promptVersion: t.meta.promptVersion, fullPromptStored: false },
          startTime: start,
        });
        gen.end({ output: t.content, usage: t.meta.usage, model: t.meta.model, latencyMs: t.meta.latencyMs, endTime: end });
        generations++;
      }
      trace.update({ output: { turns: rec.turns.length, usage: rec.usage, simUsage: rec.simUsage, memoryUsage: rec.memoryUsage, costUsd: rec.costUsd, truncated: rec.truncated, memory: rec.memory } });
      trace.end(finished);
      traces++;

      const scoreFile = join(dir, "scores", recordFileName(rec.id));
      if (existsSync(scoreFile)) {
        const s = JSON.parse(readFileSync(scoreFile, "utf8")) as ScoreRecord;
        for (const m of ["consistency", "memory", "style", "format"] as const) scoreTrace(trace.traceId, m, s.mean[m], `judge ${s.judgeModel}`, finished);
        scoreTrace(trace.traceId, "probe_recall", s.probeRecall, undefined, finished);
        scores += 5;
      }
      const regenFile = join(dir, "regen", recordFileName(rec.id));
      if (existsSync(regenFile)) {
        const r = JSON.parse(readFileSync(regenFile, "utf8")) as RegenRecord;
        scoreTrace(trace.traceId, "regen_trigger_rate", r.triggerRate, undefined, finished);
        for (const [k, n] of Object.entries(r.counts)) if (n) scoreTrace(trace.traceId, `regen_${k}`, (100 * n) / Math.max(1, r.replies.length), "per 100 replies", finished);
        scores++;
      }
      const safetyFile = join(dir, "safety", recordFileName(rec.id));
      if (existsSync(safetyFile)) {
        const sf = JSON.parse(readFileSync(safetyFile, "utf8")) as SafetyRecord;
        scoreTrace(trace.traceId, "safety_over_refusal", sf.overRefusalRate, undefined, finished);
        scoreTrace(trace.traceId, "safety_under_reaction", sf.underReactionRate, undefined, finished);
        scoreTrace(trace.traceId, "safety_violation", sf.violationRate, undefined, finished);
        scores += 3;
      }
      if (trace.traceId) {
        rec.traceId = trace.traceId;
        writeFileSync(path, JSON.stringify(rec, null, 2));
      }
      console.log(`✓ ${experiment}/${rec.id}`);
    }
  }
  await flushTracing();
  console.log(`backfill: ${traces} traces, ${generations} generations, ~${scores} score groups → ${process.env.LANGFUSE_BASE_URL ?? "http://localhost:3000"}`);
}

function hasSafetyChecks(cfg: RunConfig): boolean {
  return cfg.scenarioIds.some((id) => findScenario(id).safetyChecks.length > 0);
}

function cmdReport(outDir: string) {
  const convs = readJsonDir<ConversationRecord>(join(outDir, "conversations"));
  const scores = readJsonDir<ScoreRecord>(join(outDir, "scores"));
  const md = toMarkdown(summarise(scores, convs), `Eval report · ${basename(outDir)}`);
  writeFileSync(join(outDir, "report.md"), md);
  console.log(md);
}

async function main() {
  const [cmd, target] = process.argv.slice(2);
  if (initTracing({ environment: "eval" })) console.log(`tracing → Langfuse (${process.env.LANGFUSE_BASE_URL ?? "http://localhost:3000"})`);
  if (!cmd || !target) {
    console.log("usage: eval estimate <config.json> | run <config.json> | judge <outDir> | report <outDir> | regen <outDir> | safety <outDir> | all <config.json> | backfill <outDir…>");
    return;
  }
  switch (cmd) {
    case "estimate":
      cmdEstimate(resolve(target));
      break;
    case "run":
      await cmdRun(resolve(target));
      break;
    case "judge":
      await cmdJudge(resolve(target));
      break;
    case "report":
      cmdReport(resolve(target));
      break;
    case "regen":
      await cmdRegen(resolve(target));
      break;
    case "safety":
      await cmdSafety(resolve(target));
      break;
    case "backfill":
      await cmdBackfill(process.argv.slice(3).filter((a) => !a.startsWith("--")).map((d) => resolve(d)));
      break;
    case "all": {
      const cfg = await cmdRun(resolve(target));
      await cmdJudge(cfg.outDir, cfg);
      cmdReport(cfg.outDir);
      await cmdRegen(cfg.outDir, cfg);
      if (hasSafetyChecks(cfg)) await cmdSafety(cfg.outDir, cfg);
      break;
    }
    default:
      throw new Error(`unknown command ${cmd}`);
  }
}

main()
  .then(() => flushTracing())
  .catch(async (e) => {
    console.error(e instanceof Error ? e.stack ?? e.message : e);
    if (tracingEnabled()) await flushTracing();
    process.exit(1);
  });
