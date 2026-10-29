import type { ArchivedFact, Fact } from "../types.js";

/**
 * Recall layer (docs/12). The fact list in the prompt is capped (25) and the summariser drops what
 * does not fit; everything it ever extracted lives on in `MemoryState.archive`. Before each reply
 * the archive is searched with the user's latest message and the few entries that fit are shown to
 * the character next to MEMORY. Two indexes: an exact one on the proper nouns the summariser tagged
 * on each fact (a name in the message pulls every fact about it), and a vector one for paraphrase.
 */

export interface Embedder {
  id: string;
  /** Unit-length vectors, one per text. `kind` lets asymmetric models prefix queries and passages. */
  embed(texts: string[], kind: "query" | "passage"): Promise<number[][]>;
  /** Load weights ahead of the first call (off the user's path). */
  warm?(): Promise<void>;
}

/**
 * Offline embedder: hashed character uni- and bigrams, unit length. No model, deterministic, enough
 * for tests and for a baseline arm; Korean and Japanese work at the syllable level.
 */
export function bigramEmbedder(dims = 1024): Embedder {
  const hash = (s: string) => {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0) % dims;
  };
  const one = (text: string): number[] => {
    const v = new Array<number>(dims).fill(0);
    const t = text.toLowerCase().replace(/\s+/g, " ");
    for (let i = 0; i < t.length; i++) {
      v[hash(t[i]!)]! += 1;
      if (i + 1 < t.length) v[hash(t.slice(i, i + 2))]! += 2;
    }
    const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
    return v.map((x) => x / n);
  };
  return { id: "bigram", async embed(texts) { return texts.map(one); } };
}

/**
 * Local sentence embeddings through transformers.js (ONNX on the CPU, no API, no key). The model is
 * fetched from the Hugging Face hub on first use and cached. Default: multilingual-e5-small, which
 * covers Korean and Japanese and wants "query: " / "passage: " prefixes.
 */
export function localEmbedder(model = "Xenova/multilingual-e5-small"): Embedder {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let pipe: Promise<any> | undefined;
  const load = () =>
    (pipe ??= import(/* @vite-ignore */ "@huggingface/transformers" as string).then((m) =>
      m.pipeline("feature-extraction", model, { dtype: "q8" }),
    ));
  const e5 = /e5/i.test(model);
  return {
    id: `local:${model}`,
    async warm() {
      await load();
    },
    async embed(texts, kind) {
      const p = await load();
      const out = await p(texts.map((t) => (e5 ? `${kind}: ${t}` : t)), { pooling: "mean", normalize: true });
      return out.tolist() as number[][];
    },
  };
}

export interface RecallHit {
  fact: ArchivedFact;
  /** Cosine similarity to the query; entity hits add 1 so they sort first. */
  score: number;
  why: "entity" | "vector";
}

export interface RecallOptions {
  /** Entries shown per reply (default 5). */
  k?: number;
  /** Vector hits below this cosine are not shown (entity hits always are). */
  minScore?: number;
  /**
   * Vector hits are shown only when they stand this many standard deviations above the mean
   * similarity of the whole archive to the query. Cosines from e5 sit in a 0.03-wide band (mean
   * 0.83-0.84, sd 0.02 on the 1000-turn archive), so an absolute `minScore` does not separate a
   * fact that fits from the rest; a fact that fits stood at z 3.2-4.0, the best of nothing at 2.2-2.7
   * (docs/12 result 2). Absent = every vector hit may be shown.
   */
  minZ?: number;
  /**
   * Entities never used for exact matching. The two speakers' names say nothing about what a message
   * is about: "고마워요, 하루 씨" pulled five restatements of "하루 told him to close the door" (docs/12
   * result 2). The session adds the character's and the user's names.
   */
  ignoreEntities?: string[];
}

/**
 * Two archive texts that are the same fact a word apart: the summariser restates what it extracted
 * across consecutive consolidations ("하루는 문은 바로 닫아 달라고 했다" three ways), and exact-text
 * merging left them to fill the k slots with one fact (docs/12 result 2). Jaccard on character bigrams.
 */
export function nearDuplicate(a: string, b: string, threshold = 0.5): boolean {
  const grams = (s: string) => {
    const t = s.replace(/\s+/g, "");
    const g = new Set<string>();
    for (let i = 0; i + 1 < t.length; i++) g.add(t.slice(i, i + 2));
    return g;
  };
  const A = grams(a), B = grams(b);
  if (!A.size || !B.size) return a === b;
  let both = 0;
  for (const g of A) if (B.has(g)) both++;
  return both / (A.size + B.size - both) >= threshold;
}

const dot = (a: number[], b: number[]) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
};

export class Recall {
  private vectors = new Map<string, number[]>();
  readonly k: number;
  readonly minScore?: number;
  readonly minZ?: number;
  readonly ignoreEntities: string[];

  constructor(
    readonly embedder: Embedder,
    opts: RecallOptions = {},
  ) {
    this.k = opts.k ?? 5;
    this.minScore = opts.minScore;
    this.minZ = opts.minZ;
    this.ignoreEntities = opts.ignoreEntities ?? [];
  }

  async warm(): Promise<void> {
    await this.embedder.warm?.();
  }

  /** Embed archive entries not seen yet. Runs after a consolidation, off the user's path. */
  async index(archive: ArchivedFact[]): Promise<void> {
    const missing = [...new Set(archive.map((f) => f.text))].filter((t) => !this.vectors.has(t));
    if (!missing.length) return;
    const vs = await this.embedder.embed(missing, "passage");
    missing.forEach((t, i) => this.vectors.set(t, vs[i]!));
  }

  /**
   * Entries that fit `query`, newest first on ties, minus what the prompt's fact list already holds
   * (same text, or one contained in the other).
   */
  async query(query: string, archive: ArchivedFact[], ledger: Fact[] = []): Promise<RecallHit[]> {
    if (!archive.length || !query.trim()) return [];
    await this.index(archive);
    const [q] = await this.embedder.embed([query], "query");
    const inLedger = (text: string) => ledger.some((f) => f.text === text || f.text.includes(text) || text.includes(f.text));
    // One entry per text: the latest mention.
    const latest = new Map<string, ArchivedFact>();
    for (const f of archive) {
      const prev = latest.get(f.text);
      if (!prev || f.turnTo > prev.turnTo) latest.set(f.text, f);
    }
    const cands: { f: ArchivedFact; cos: number; entity: boolean }[] = [];
    for (const f of latest.values()) {
      if (inLedger(f.text)) continue;
      const v = this.vectors.get(f.text);
      const cos = v ? dot(q!, v) : 0;
      const entity = (f.entities ?? []).some((e) => e.length >= 2 && !this.ignoreEntities.includes(e) && query.includes(e));
      cands.push({ f, cos, entity });
    }
    // The floor a vector hit must clear: the absolute one, or the archive's own distribution for this query.
    let floor = this.minScore ?? -Infinity;
    if (this.minZ !== undefined && cands.length > 1) {
      const mean = cands.reduce((a, c) => a + c.cos, 0) / cands.length;
      const sd = Math.sqrt(cands.reduce((a, c) => a + (c.cos - mean) ** 2, 0) / cands.length);
      floor = Math.max(floor, mean + this.minZ * sd);
    }
    const hits: RecallHit[] = [];
    for (const c of cands) {
      if (c.entity) hits.push({ fact: c.f, score: 1 + c.cos, why: "entity" });
      else if (c.cos >= floor) hits.push({ fact: c.f, score: c.cos, why: "vector" });
    }
    hits.sort((a, b) => b.score - a.score || b.fact.turnTo - a.fact.turnTo);
    const picked: RecallHit[] = [];
    for (const h of hits) {
      if (picked.length >= this.k) break;
      if (picked.some((p) => nearDuplicate(p.fact.text, h.fact.text))) continue;
      picked.push(h);
    }
    return picked;
  }
}
