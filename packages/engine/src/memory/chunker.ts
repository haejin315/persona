import type { Turn } from "../types.js";

/**
 * Transcript chunking for the vector index (docs/14). A chunk is a window of whole exchanges
 * (user turn + character turn), `exchanges` long, stepping by `exchanges - overlap`, so a detail
 * said at the edge of one window is inside the next. Windows sit on a fixed grid from turn 0,
 * which makes chunking idempotent: the same turns always give the same chunk ids, so re-ingesting
 * a range upserts rather than duplicates.
 */

export interface Chunk {
  /** Deterministic id from the turn range. */
  id: string;
  text: string;
  turnFrom: number;
  turnTo: number;
}

export interface ChunkOptions {
  /** Exchanges per window (default 2 = 4 turns). */
  exchanges?: number;
  /** Exchanges shared with the previous window (default 1). */
  overlap?: number;
  /** Speaker labels: what to print before user / character lines. */
  user?: string;
  character?: string;
}

const step = (o: ChunkOptions) => Math.max(1, (o.exchanges ?? 2) - (o.overlap ?? 1)) * 2;
const size = (o: ChunkOptions) => (o.exchanges ?? 2) * 2;

/** The first turn index of the window that ends at or after `turnTo`, on the grid. */
function windowsEndingIn(fromTurn: number, toTurn: number, o: ChunkOptions): [number, number][] {
  const s = step(o);
  const n = size(o);
  const out: [number, number][] = [];
  // Window k covers [k*s, k*s + n - 1]. Emit the windows whose last turn lies in [fromTurn, toTurn].
  const kMin = Math.max(0, Math.ceil((fromTurn - n + 1) / s));
  for (let k = kMin; k * s + n - 1 <= toTurn; k++) {
    const a = k * s;
    const b = a + n - 1;
    if (b >= fromTurn) out.push([a, b]);
  }
  return out;
}

/**
 * Chunks whose last turn falls in [fromTurn, toTurn]. Call it with the range that just left the
 * verbatim window: each chunk is produced exactly once, when its last turn becomes history.
 */
export function chunkRange(turns: Turn[], fromTurn: number, toTurn: number, o: ChunkOptions = {}): Chunk[] {
  const byIndex = new Map(turns.map((t) => [t.index, t]));
  const user = o.user ?? "User";
  const character = o.character ?? "Character";
  return windowsEndingIn(fromTurn, toTurn, o)
    .map(([a, b]) => {
      const lines: string[] = [];
      for (let i = a; i <= b; i++) {
        const t = byIndex.get(i);
        if (!t) continue;
        lines.push(`${t.role === "user" ? user : character}: ${t.content.replace(/\s+/g, " ").trim()}`);
      }
      return { id: `c${a}-${b}`, text: lines.join("\n"), turnFrom: a, turnTo: b };
    })
    .filter((c) => c.text.length > 0);
}

/** Every chunk of a whole transcript (for rebuilding an index). */
export function chunkAll(turns: Turn[], o: ChunkOptions = {}): Chunk[] {
  const last = turns.at(-1)?.index ?? -1;
  return last < 0 ? [] : chunkRange(turns, 0, last, o);
}
