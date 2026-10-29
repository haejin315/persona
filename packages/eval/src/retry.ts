import { BudgetExceededError } from "./budget.js";

/**
 * A 1000-turn run is five hours of calls; one failed call must not throw it away. Rate-limit errors
 * wait five minutes and retry for up to three hours (the subscription window resets); anything else
 * retries twice after 20 s. `undo` puts the session back as it was before the attempt.
 */
export async function withRetry<T>(fn: () => Promise<T>, undo: () => void, log?: (msg: string) => void): Promise<T> {
  let limited = 0;
  let other = 0;
  for (;;) {
    try {
      return await fn();
    } catch (e) {
      if (e instanceof BudgetExceededError) throw e;
      const msg = String((e as Error)?.message ?? e);
      // An expired OAuth session that the CLI could not refresh (seen 2026-10-29 at turn 300 of two
      // 1000-turn arms) is transient too: it came back by itself, so it waits like a rate limit.
      const rateLimited = Boolean((e as { rateLimited?: boolean })?.rateLimited) || /limit|rate|overloaded|429|529|authenticat|oauth/i.test(msg);
      if (rateLimited ? ++limited > 36 : ++other > 2) throw e;
      undo();
      const waitMs = rateLimited ? 5 * 60_000 : 20_000;
      log?.(`retry in ${waitMs / 1000}s after: ${msg.slice(0, 160)}`);
      await new Promise((res) => setTimeout(res, waitMs));
    }
  }
}
