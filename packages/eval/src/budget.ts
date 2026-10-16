import type { TokenUsage } from "@persona/engine";
import { costOf, fmtUsd } from "./pricing.js";

export class BudgetExceededError extends Error {
  constructor(public spentUsd: number, public maxUsd: number) {
    super(`budget exceeded: spent ${fmtUsd(spentUsd)} of ${fmtUsd(maxUsd)} (billed at list price)`);
    this.name = "BudgetExceededError";
  }
}

/**
 * Running total of billed cost across every call in a run (character, simulator, memory, judge).
 * Shared by all concurrent conversations so the cap applies to the run, not to each conversation.
 */
export class BudgetTracker {
  spentUsd = 0;
  listUsd = 0;
  unknownModels = new Set<string>();
  calls = 0;

  constructor(public maxUsd?: number) {}

  add(model: string, usage?: TokenUsage): void {
    if (!usage) return;
    const c = costOf(model, usage);
    this.calls++;
    this.listUsd += c.usd;
    this.spentUsd += c.billedUsd;
    if (c.unknownModel) this.unknownModels.add(model);
  }

  /** Throws once the billed total passes the cap. Call before starting a new paid call. */
  check(): void {
    if (this.maxUsd !== undefined && this.spentUsd > this.maxUsd) throw new BudgetExceededError(this.spentUsd, this.maxUsd);
  }

  get exceeded(): boolean {
    return this.maxUsd !== undefined && this.spentUsd > this.maxUsd;
  }
}

/** Wrap a provider so every call is priced into the tracker and refused once the cap is hit. */
export function trackedProvider<P extends { generate: Function; stream: Function; id: string; defaultModel: string }>(
  provider: P,
  tracker: BudgetTracker,
): P {
  const wrapped = Object.create(provider) as P & { generate: P["generate"]; stream: P["stream"] };
  wrapped.generate = (async (messages: unknown, opts?: { model?: string }) => {
    tracker.check();
    const res = await (provider.generate as (m: unknown, o?: unknown) => Promise<{ model: string; usage?: TokenUsage }>)(messages, opts);
    tracker.add(res.model ?? opts?.model ?? provider.defaultModel, res.usage);
    return res;
  }) as P["generate"];
  wrapped.stream = (function* () {
    throw new Error("streaming is not budget-tracked; use generate()");
  }) as unknown as P["stream"];
  return wrapped;
}
