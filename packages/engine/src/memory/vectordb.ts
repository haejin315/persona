import { createHash } from "node:crypto";
import { QdrantClient } from "@qdrant/js-client-rest";

/**
 * Vector database behind the Graph RAG memory (docs/14): Qdrant over HTTP (HNSW, payload filters).
 * One collection per kind per namespace (`<ns>-chunks`, `<ns>-facts`); a namespace is one
 * conversation. The interface is small so a different store can stand in (and tests use a fake).
 */

export interface VectorPoint {
  /** Any string; hashed to the UUID Qdrant wants. */
  id: string;
  vector: number[];
  payload: Record<string, unknown>;
}

export interface VectorHitRow {
  id: string;
  score: number;
  payload: Record<string, unknown>;
}

export interface VectorFilter {
  /** Only points whose payload.turnTo is at most this (keeps the verbatim window out of recall). */
  turnToMax?: number;
}

export interface VectorIndex {
  id: string;
  ensure(collection: string, dims: number): Promise<void>;
  upsert(collection: string, points: VectorPoint[]): Promise<void>;
  search(collection: string, vector: number[], opts: { limit: number; filter?: VectorFilter }): Promise<VectorHitRow[]>;
  count(collection: string): Promise<number>;
  drop(collection: string): Promise<void>;
}

/** Deterministic UUID (v5-like, SHA-1) from any string id. */
export function uuidFor(id: string): string {
  const h = createHash("sha1").update(id).digest("hex").slice(0, 32).split("");
  h[12] = "5";
  h[16] = ((parseInt(h[16]!, 16) & 0x3) | 0x8).toString(16);
  const s = h.join("");
  return `${s.slice(0, 8)}-${s.slice(8, 12)}-${s.slice(12, 16)}-${s.slice(16, 20)}-${s.slice(20)}`;
}

export class QdrantIndex implements VectorIndex {
  readonly id: string;
  private client: QdrantClient;
  private ensured = new Set<string>();

  constructor(url = process.env.QDRANT_URL ?? "http://localhost:6333") {
    // The pinned server (1.13) is two minors behind the client; the calls used here exist in both.
    this.client = new QdrantClient({ url, checkCompatibility: false });
    this.id = `qdrant:${url}`;
  }

  async ensure(collection: string, dims: number): Promise<void> {
    if (this.ensured.has(collection)) return;
    const exists = await this.client.collectionExists(collection);
    if (!exists.exists) {
      await this.client.createCollection(collection, { vectors: { size: dims, distance: "Cosine" } });
      await this.client.createPayloadIndex(collection, { field_name: "turnTo", field_schema: "integer", wait: true });
    }
    this.ensured.add(collection);
  }

  async upsert(collection: string, points: VectorPoint[]): Promise<void> {
    if (!points.length) return;
    await this.ensure(collection, points[0]!.vector.length);
    await this.client.upsert(collection, {
      wait: true,
      points: points.map((p) => ({ id: uuidFor(p.id), vector: p.vector, payload: { ...p.payload, key: p.id } })),
    });
  }

  async search(collection: string, vector: number[], opts: { limit: number; filter?: VectorFilter }): Promise<VectorHitRow[]> {
    const exists = await this.client.collectionExists(collection);
    if (!exists.exists) return [];
    const filter =
      opts.filter?.turnToMax !== undefined ? { must: [{ key: "turnTo", range: { lte: opts.filter.turnToMax } }] } : undefined;
    const res = await this.client.query(collection, { query: vector, limit: opts.limit, with_payload: true, ...(filter ? { filter } : {}) });
    return res.points.map((r) => ({ id: String((r.payload as { key?: string } | null)?.key ?? r.id), score: r.score, payload: (r.payload ?? {}) as Record<string, unknown> }));
  }

  async count(collection: string): Promise<number> {
    const exists = await this.client.collectionExists(collection);
    if (!exists.exists) return 0;
    return (await this.client.count(collection, { exact: true })).count;
  }

  async drop(collection: string): Promise<void> {
    this.ensured.delete(collection);
    const exists = await this.client.collectionExists(collection);
    if (exists.exists) await this.client.deleteCollection(collection);
  }
}

/** In-memory stand-in with the same interface (tests, offline runs). Exact cosine, no index. */
export class MemoryVectorIndex implements VectorIndex {
  readonly id = "memory";
  private data = new Map<string, Map<string, VectorPoint>>();

  async ensure(): Promise<void> {}

  async upsert(collection: string, points: VectorPoint[]): Promise<void> {
    const c = this.data.get(collection) ?? this.data.set(collection, new Map()).get(collection)!;
    for (const p of points) c.set(p.id, p);
  }

  async search(collection: string, vector: number[], opts: { limit: number; filter?: VectorFilter }): Promise<VectorHitRow[]> {
    const c = this.data.get(collection);
    if (!c) return [];
    const out: VectorHitRow[] = [];
    for (const p of c.values()) {
      const turnTo = p.payload.turnTo as number | undefined;
      if (opts.filter?.turnToMax !== undefined && turnTo !== undefined && turnTo > opts.filter.turnToMax) continue;
      let dot = 0;
      for (let i = 0; i < vector.length; i++) dot += vector[i]! * p.vector[i]!;
      out.push({ id: p.id, score: dot, payload: p.payload });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, opts.limit);
  }

  async count(collection: string): Promise<number> {
    return this.data.get(collection)?.size ?? 0;
  }

  async drop(collection: string): Promise<void> {
    this.data.delete(collection);
  }
}
