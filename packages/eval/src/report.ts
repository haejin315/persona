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
  /** Facts in the prompt's list at the end, and in the uncapped archive (memory rules v4, docs/12). */
  facts: { ledger: number; archive: number };
  /**
   * Recall layer (docs/12): mean archive entries shown per reply, and the share of answerable probes
   * whose expected answer was among the entries shown for that reply (the retrieval's own hit rate,
   * separate from whether the character then said it).
   */
  recall: { perReply: number; probeHit: number };
}

/** Content words of a probe's expected answer, as judge.ts matches them. */
const contentWords = (expected: string) =>
  expected
    .replace(/\([^)]*\)/g, " ")
    .split(/[\s,·/]+/)
    .map((w) => w.replace(/(의|을|를|은|는|이|가|에서|으로|로|과|와)$/u, ""))
    .filter((w) => w.length >= 2);

/** The character reply to user turn N (1-based count of user turns). */
function replyToUserTurn(rec: ConversationRecord, userTurn: number) {
  let seen = 0;
  for (let i = 0; i < rec.turns.length; i++) {
    if (rec.turns[i]!.role !== "user") continue;
    if (++seen === userTurn) return rec.turns.slice(i + 1).find((t) => t.role === "character");
  }
  return undefined;
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
    const recs = ss.map((s) => convById.get(s.conversationId)).filter(Boolean) as ConversationRecord[];
    const replies = recs.flatMap((c) => c.turns.filter((t) => t.role === "character"));
    const probeHits = ss.flatMap((s) => {
      const rec = convById.get(s.conversationId);
      if (!rec) return [];
      return s.probes
        .filter((p) => p.answerable)
        .map((p) => {
          const shown = replyToUserTurn(rec, p.atUserTurn)?.meta?.recalled ?? [];
          const words = contentWords(p.expected);
          return shown.some((h) => words.every((w) => h.text.includes(w))) ? 1 : 0;
        });
    });

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
      facts: {
        ledger: mean(recs.map((c) => c.memory.facts.length)),
        archive: mean(recs.map((c) => c.memory.archive?.length ?? 0)),
      },
      recall: {
        perReply: mean(replies.map((t) => t.meta?.recalled?.length ?? 0)),
        probeHit: mean(probeHits),
      },
    };
  });
}

export function toMarkdown(summaries: ArmSummary[], title = "Eval report"): string {
  const out: string[] = [`## ${title}`, ""];
  const archive = summaries.some((s) => s.facts.archive > 0);
  out.push(`| arm | n | consistency | memory | style | format | probe recall | tokens/reply (prompt · cached · out) |${archive ? " facts (list · archive) | recalled/reply · probe in recalled |" : ""}`);
  const unanswerable = summaries.reduce((a, s) => a + s.unanswerableProbes, 0);
  out.push(`|---|---:|---:|---:|---:|---:|---:|---|${archive ? "---:|---:|" : ""}`);
  for (const s of summaries) {
    const cell = (m: Metric) => `${f1(s.metric[m].mean)} ± ${f1(s.metric[m].sd)}`;
    const t = s.tokensPerReply;
    const extra = archive
      ? ` ${Math.round(s.facts.ledger)} · ${Math.round(s.facts.archive)} | ${s.recall.perReply ? `${s.recall.perReply.toFixed(1)} · ${pct(s.recall.probeHit)}` : "–"} |`
      : "";
    out.push(
      `| ${s.armId} | ${s.n} | ${cell("consistency")} | ${cell("memory")} | ${cell("style")} | ${cell("format")} | ${pct(s.probeRecall)} | ${Math.round(t.prompt)} · ${Math.round(t.cached)} · ${Math.round(t.completion)} |${extra}`,
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
