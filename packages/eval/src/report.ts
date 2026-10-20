import { METRICS, type ConversationRecord, type Metric, type ScoreRecord } from "./types.js";

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const sd = (xs: number[]) => {
  if (xs.length < 2) return NaN;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
};
const f1 = (x: number) => (Number.isNaN(x) ? "–" : x.toFixed(2));
const pct = (x: number) => (Number.isNaN(x) ? "–" : `${Math.round(x * 100)}%`);

export interface ArmSummary {
  armId: string;
  n: number;
  metric: Record<Metric, { mean: number; sd: number }>;
  probeRecall: number;
  /** Probes whose fact the simulated user never revealed (excluded from recall). */
  unanswerableProbes: number;
  /** Mean per-metric score at each checkpoint turn, to see drift over the conversation. */
  drift: Array<{ turnIndex: number } & Record<Metric, number>>;
  tokensPerReply: { prompt: number; cached: number; completion: number };
}

export function summarise(scores: ScoreRecord[], convs: ConversationRecord[]): ArmSummary[] {
  const byArm = new Map<string, ScoreRecord[]>();
  for (const s of scores) byArm.set(s.armId, [...(byArm.get(s.armId) ?? []), s]);
  const convById = new Map(convs.map((c) => [c.id, c]));

  return [...byArm.entries()].map(([armId, ss]) => {
    const metric = Object.fromEntries(
      METRICS.map((m) => {
        const xs = ss.map((s) => s.mean[m]).filter((x) => !Number.isNaN(x));
        return [m, { mean: mean(xs), sd: sd(xs) }];
      }),
    ) as ArmSummary["metric"];

    const turnIdxs = [...new Set(ss.flatMap((s) => s.checkpoints.map((c) => c.turnIndex)))].sort((a, b) => a - b);
    const drift = turnIdxs.map((turnIndex) => {
      const cps = ss.flatMap((s) => s.checkpoints.filter((c) => c.turnIndex === turnIndex));
      return { turnIndex, ...(Object.fromEntries(METRICS.map((m) => [m, mean(cps.map((c) => c[m]))])) as Record<Metric, number>) };
    });

    const usages = ss.map((s) => convById.get(s.conversationId)?.usage).filter(Boolean) as ConversationRecord["usage"][];
    const calls = usages.reduce((a, u) => a + u.calls, 0) || 1;

    return {
      armId,
      n: ss.length,
      metric,
      probeRecall: mean(ss.map((s) => s.probeRecall).filter((x) => !Number.isNaN(x))),
      unanswerableProbes: ss.reduce((a, s) => a + (s.unanswerableProbes ?? 0), 0),
      drift,
      tokensPerReply: {
        prompt: usages.reduce((a, u) => a + u.promptTokens, 0) / calls,
        cached: usages.reduce((a, u) => a + (u.cachedPromptTokens ?? 0), 0) / calls,
        completion: usages.reduce((a, u) => a + u.completionTokens, 0) / calls,
      },
    };
  });
}

export function toMarkdown(summaries: ArmSummary[], title = "Eval report"): string {
  const out: string[] = [`## ${title}`, ""];
  out.push(`| arm | n | consistency | memory | style | format | probe recall | tokens/reply (prompt · cached · out) |`);
  const unanswerable = summaries.reduce((a, s) => a + s.unanswerableProbes, 0);
  out.push(`|---|---:|---:|---:|---:|---:|---:|---|`);
  for (const s of summaries) {
    const cell = (m: Metric) => `${f1(s.metric[m].mean)} ± ${f1(s.metric[m].sd)}`;
    const t = s.tokensPerReply;
    out.push(
      `| ${s.armId} | ${s.n} | ${cell("consistency")} | ${cell("memory")} | ${cell("style")} | ${cell("format")} | ${pct(s.probeRecall)} | ${Math.round(t.prompt)} · ${Math.round(t.cached)} · ${Math.round(t.completion)} |`,
    );
  }
  if (unanswerable) out.push("", `_${unanswerable} probe(s) excluded from recall: the simulated user never revealed the fact before asking._`);
  out.push("", "### Drift over the conversation (mean score at each checkpoint)", "");
  for (const s of summaries) {
    out.push(`**${s.armId}**`, "");
    out.push(`| turn | consistency | memory | style | format |`);
    out.push(`|---:|---:|---:|---:|---:|`);
    for (const d of s.drift) out.push(`| ${d.turnIndex} | ${f1(d.consistency)} | ${f1(d.memory)} | ${f1(d.style)} | ${f1(d.format)} |`);
    out.push("");
  }
  return out.join("\n");
}
