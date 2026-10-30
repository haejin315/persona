import type { ArchivedFact, Turn } from "../types.js";
import { chunkRange, type ChunkOptions } from "./chunker.js";
import type { GraphEdge, GraphStore } from "./graphdb.js";
import type { Embedder } from "./recall.js";
import type { VectorIndex } from "./vectordb.js";

/**
 * Graph RAG memory (docs/14): the transcript is chunked and embedded into a vector database, the
 * summariser's facts are embedded too, and every fact with a subject and object is an edge in a
 * knowledge graph. Before a reply, the user's message seeds the graph (entity linking by name),
 * its neighbourhood is read, and the vector database answers the paraphrase side with the chunks
 * and facts nearest the message. All three are rendered into one context block.
 */

export interface GraphRagOptions {
  /** Namespace (one conversation). */
  ns: string;
  embedder: Embedder;
  vectors: VectorIndex;
  graph: GraphStore;
  /** Raw transcript chunks shown per reply (default 3). */
  chunkK?: number;
  /** Facts shown per reply (default 5). */
  factK?: number;
  /** Graph hops from the seed entities (default 2). */
  hops?: number;
  /** Edges shown per reply (default 12). */
  edgeLimit?: number;
  /** Vector hits below this cosine are dropped (default 0, keep all). */
  minScore?: number;
  /**
   * A node touching more than this share of all edges is a hub (the speakers, "오늘" when the
   * summariser lets a time word through) and seeds nothing; it still appears as a neighbour.
   * Default 0.1, with a floor of 10 edges so a young graph excludes nothing.
   */
  maxHubShare?: number;
  chunking?: ChunkOptions;
}

export interface GraphRagResult {
  seeds: string[];
  edges: GraphEdge[];
  chunks: { id: string; text: string; turnFrom: number; turnTo: number; score: number }[];
  facts: { text: string; turnFrom: number; turnTo: number; score: number }[];
}

export class GraphRag {
  private names?: Promise<{ name: string; degree: number }[]>;
  readonly chunksCollection: string;
  readonly factsCollection: string;

  constructor(readonly opts: GraphRagOptions) {
    this.chunksCollection = `${opts.ns}-chunks`;
    this.factsCollection = `${opts.ns}-facts`;
  }

  /** Chunk and embed the transcript turns whose windows end in [fromTurn, toTurn]. */
  async ingestTurns(turns: Turn[], fromTurn: number, toTurn: number): Promise<number> {
    const chunks = chunkRange(turns, fromTurn, toTurn, this.opts.chunking);
    if (!chunks.length) return 0;
    const vectors = await this.opts.embedder.embed(chunks.map((c) => c.text), "passage");
    await this.opts.vectors.upsert(
      this.chunksCollection,
      chunks.map((c, i) => ({ id: c.id, vector: vectors[i]!, payload: { text: c.text, turnFrom: c.turnFrom, turnTo: c.turnTo } })),
    );
    return chunks.length;
  }

  /** Embed facts into the vector database and put their triples into the graph. */
  async ingestFacts(facts: ArchivedFact[]): Promise<void> {
    if (!facts.length) return;
    const vectors = await this.opts.embedder.embed(facts.map((f) => f.text), "passage");
    await this.opts.vectors.upsert(
      this.factsCollection,
      facts.map((f, i) => ({ id: `f${f.turnFrom}-${f.turnTo}:${f.text}`, vector: vectors[i]!, payload: { text: f.text, turnFrom: f.turnFrom, turnTo: f.turnTo } })),
    );
    const edges = facts.flatMap((f) => (f.triple ? [{ ...f.triple, text: f.text, turnFrom: f.turnFrom, turnTo: f.turnTo }] : []));
    if (edges.length) {
      await this.opts.graph.addTriples(edges);
      this.names = undefined;
    }
  }

  /**
   * Everything the stores hold that bears on the user's latest message. `turnToMax` keeps the
   * turns still verbatim in the prompt out of the vector hits; `ignore` are the speakers' names,
   * which seed nothing (they touch every edge).
   */
  async retrieve(message: string, o: { turnToMax?: number; ignore?: string[] } = {}): Promise<GraphRagResult> {
    const ignore = new Set((o.ignore ?? []).map((s) => s.replace(/\s*(씨|님)$/u, "")));
    const nodes = await (this.names ??= this.opts.graph.entities());
    const totalEdges = nodes.reduce((a, n) => a + n.degree, 0) / 2;
    const hub = Math.max(10, (this.opts.maxHubShare ?? 0.1) * totalEdges);
    const seeds = nodes.filter((n) => n.name.length >= 2 && !ignore.has(n.name) && n.degree <= hub && message.includes(n.name)).map((n) => n.name);
    const [edges, [qv]] = await Promise.all([
      seeds.length ? this.opts.graph.neighborhood(seeds, { hops: this.opts.hops ?? 2, limit: this.opts.edgeLimit ?? 12 }) : Promise.resolve([]),
      this.opts.embedder.embed([message], "query"),
    ]);
    const filter = o.turnToMax !== undefined ? { turnToMax: o.turnToMax } : undefined;
    const [chunkHits, factHits] = await Promise.all([
      this.opts.vectors.search(this.chunksCollection, qv!, { limit: this.opts.chunkK ?? 3, ...(filter ? { filter } : {}) }),
      this.opts.vectors.search(this.factsCollection, qv!, { limit: this.opts.factK ?? 5, ...(filter ? { filter } : {}) }),
    ]);
    const min = this.opts.minScore ?? 0;
    const num = (v: unknown) => (typeof v === "number" ? v : Number(v ?? 0));
    return {
      seeds,
      edges,
      chunks: chunkHits
        .filter((h) => h.score >= min)
        .map((h) => ({ id: h.id, text: String(h.payload.text ?? ""), turnFrom: num(h.payload.turnFrom), turnTo: num(h.payload.turnTo), score: h.score })),
      facts: factHits
        .filter((h) => h.score >= min)
        .map((h) => ({ text: String(h.payload.text ?? ""), turnFrom: num(h.payload.turnFrom), turnTo: num(h.payload.turnTo), score: h.score })),
    };
  }

  async counts(): Promise<{ chunks: number; facts: number; nodes: number; edges: number }> {
    const [chunks, facts, g] = await Promise.all([
      this.opts.vectors.count(this.chunksCollection),
      this.opts.vectors.count(this.factsCollection),
      this.opts.graph.count(),
    ]);
    return { chunks, facts, ...g };
  }

  async clear(): Promise<void> {
    await Promise.all([this.opts.vectors.drop(this.chunksCollection), this.opts.vectors.drop(this.factsCollection), this.opts.graph.clear()]);
    this.names = undefined;
  }

  async close(): Promise<void> {
    await this.opts.graph.close();
  }
}

/** The context block the character reads. Empty string when nothing was found. */
export function renderGraphRag(r: GraphRagResult, now: number, user: string): string {
  const age = (turnTo: number) => `[about ${Math.max(1, Math.round((now - turnTo) / 2))} exchanges ago]`;
  const parts: string[] = [];
  if (r.edges.length) {
    parts.push(
      `## KNOWLEDGE GRAPH (what is known about ${r.seeds.join(", ")} and what connects to it; "no longer" marks what was true then and has since changed)\n` +
        r.edges.map((e) => `- ${age(e.turnTo)} ${e.subject} —${e.relation}→ ${e.object}${e.current ? "" : " (no longer)"}: ${e.text}`).join("\n"),
    );
  }
  if (r.facts.length) {
    parts.push(`## RELATED FACTS\n` + r.facts.map((f) => `- ${age(f.turnTo)} ${f.text}`).join("\n"));
  }
  if (r.chunks.length) {
    parts.push(
      `## RELATED PAST EXCHANGES (verbatim)\n` + r.chunks.map((c) => `${age(c.turnTo)}\n${c.text}`).join("\n\n"),
    );
  }
  if (!parts.length) return "";
  return `# MEMORY, retrieved for what ${user} just said (from the full history of this relationship; as true as MEMORY, but each item is from the time it was said - "tomorrow" or "next week" inside one counts from then, not from now; if anything here clashes with MEMORY above, MEMORY is current)\n\n${parts.join("\n\n")}`;
}
