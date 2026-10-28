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

  constructor(
    readonly embedder: Embedder,
    opts: RecallOptions = {},
  ) {
    this.k = opts.k ?? 5;
    this.minScore = opts.minScore;
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
    const hits: RecallHit[] = [];
    for (const f of latest.values()) {
      if (inLedger(f.text)) continue;
      const v = this.vectors.get(f.text);
      const cos = v ? dot(q!, v) : 0;
      const entity = (f.entities ?? []).find((e) => e.length >= 2 && query.includes(e));
      if (entity) hits.push({ fact: f, score: 1 + cos, why: "entity" });
      else if (this.minScore === undefined || cos >= this.minScore) hits.push({ fact: f, score: cos, why: "vector" });
    }
    hits.sort((a, b) => b.score - a.score || b.fact.turnTo - a.fact.turnTo);
    return hits.slice(0, this.k);
  }
}
