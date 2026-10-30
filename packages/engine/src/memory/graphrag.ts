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
 *
 * Round 2 (docs/14 결과 1 → 2): the graph widens the candidate set and the vectors rank it - the
 * neighbourhood's edges are scored against the message through the fact vectors already in the
 * store, instead of being cut by recency, because a 2-hop walk through a hub (the user, the
 * character) otherwise returns that hub's latest edges whatever the message was. Facts are queried
 * in two pools, about the user and the character's own lines, because the latter took 87% of the
 * vector hits in round 1. A node also seeds on a token of its name (본가 → 대전 본가).
 */

export interface GraphRagOptions {
  /** Namespace (one conversation). */
  ns: string;
  embedder: Embedder;
  vectors: VectorIndex;
  graph: GraphStore;
  /** Raw transcript chunks shown per reply (default 3). */
  chunkK?: number;
  /** Facts about the user (source other than "self") shown per reply (default 5). */
  factK?: number;
  /**
   * The character's own lines (source "self") shown per reply (default 2). Negative = no split:
   * one query of `factK` over all facts, as in round 1.
   */
  selfK?: number;
  /** Graph hops from the seed entities (default 2). */
  hops?: number;
  /** Edges shown per reply (default 12). 0 = the graph is not read. */
  edgeLimit?: number;
  /**
   * Edges read from the graph before ranking by similarity to the message (default 300). At most
   * `edgeLimit` of them are shown. 0 = no ranking: the graph's own order (depth, then recency).
   * The graph cuts the candidates by depth and recency, so a small pool loses old edges when many
   * nodes seed (round 2 at 60).
   */
  edgeCandidates?: number;
  /** Vector hits below this cosine are dropped (default 0, keep all). */
  minScore?: number;
  /**
   * A node touching more than this share of all edges is a hub (the speakers, "오늘" when the
   * summariser lets a time word through) and seeds nothing; it still appears as a neighbour.
   * Default 0.1, with a floor of 10 edges so a young graph excludes nothing.
   */
  maxHubShare?: number;
  /**
   * Seed a node when the message holds any token (2+ characters) of its name, not only the whole
   * name (default true): "본가" wakes "대전 본가".
   */
  tokenLinking?: boolean;
  /**
   * A token links only if it occurs in at most this share of all node names (default 0.02, floor
   * 3 names): a token shared by many names (얘기, 컵) is not a name.
   */
  maxTokenShare?: number;
  chunking?: ChunkOptions;
}

export interface GraphRagResult {
  seeds: string[];
  edges: GraphEdge[];
  chunks: { id: string; text: string; turnFrom: number; turnTo: number; score: number }[];
  facts: { text: string; turnFrom: number; turnTo: number; score: number; self: boolean }[];
}

/** The id a fact has in the facts collection; an edge carries the same fields, so it maps back. */
export const factId = (f: { turnFrom: number; turnTo: number; text: string }) => `f${f.turnFrom}-${f.turnTo}:${f.text}`;

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
      facts.map((f, i) => ({
        id: factId(f),
        vector: vectors[i]!,
        payload: { text: f.text, turnFrom: f.turnFrom, turnTo: f.turnTo, self: f.source === "self", source: f.source },
      })),
    );
    const edges = facts.flatMap((f) => (f.triple ? [{ ...f.triple, text: f.text, turnFrom: f.turnFrom, turnTo: f.turnTo }] : []));
    if (edges.length) {
      await this.opts.graph.addTriples(edges);
      this.names = undefined;
    }
  }

  /**
   * Entity linking: the graph's nodes the message names (whole name, or a token of it), and the
   * hubs - nodes on more than `maxHubShare` of all edges - which seed nothing and are crossed only
   * along one relation.
   */
  private async seeds(message: string, ignore: Set<string>): Promise<{ seeds: string[]; hubs: string[] }> {
    const nodes = await (this.names ??= this.opts.graph.entities());
    const totalEdges = nodes.reduce((a, n) => a + n.degree, 0) / 2;
    const hub = Math.max(10, (this.opts.maxHubShare ?? 0.1) * totalEdges);
    const tokens = this.opts.tokenLinking ?? true;
    // A token links only when it is discriminative: present in few node names. 얘기 (in 만화 얘기,
    // 판교 얘기, 보리 얘기 …) or 손잡이 would otherwise seed on nearly every message (round 2: 37% of
    // all seeds came from such tokens). Statistics over the graph's own names, not a word list.
    const tokenCount = new Map<string, number>();
    for (const n of nodes) for (const tok of new Set(n.name.split(/\s+/))) tokenCount.set(tok, (tokenCount.get(tok) ?? 0) + 1);
    const maxTokenNames = Math.max(3, (this.opts.maxTokenShare ?? 0.02) * nodes.length);
    const named = (name: string) =>
      message.includes(name) ||
      (tokens && name.split(/\s+/).some((tok) => tok.length >= 2 && tok !== name && (tokenCount.get(tok) ?? 0) <= maxTokenNames && message.includes(tok)));
    return {
      seeds: nodes.filter((n) => n.name.length >= 2 && !ignore.has(n.name) && n.degree <= hub && named(n.name)).map((n) => n.name),
      hubs: nodes.filter((n) => n.degree > hub || ignore.has(n.name)).map((n) => n.name),
    };
  }

  /**
   * Everything the stores hold that bears on the user's latest message. `turnToMax` keeps the
   * turns still verbatim in the prompt out of the vector hits; `ignore` are the speakers' names,
   * which seed nothing (they touch every edge).
   */
  async retrieve(message: string, o: { turnToMax?: number; ignore?: string[] } = {}): Promise<GraphRagResult> {
    const ignore = new Set((o.ignore ?? []).map((s) => s.replace(/\s*(씨|님)$/u, "")));
    const edgeLimit = this.opts.edgeLimit ?? 12;
    const candidates = this.opts.edgeCandidates ?? 300;
    const { seeds, hubs } = edgeLimit > 0 ? await this.seeds(message, ignore) : { seeds: [], hubs: [] };
    const [candidateEdges, [qv]] = await Promise.all([
      seeds.length
        ? this.opts.graph.neighborhood(seeds, { hops: this.opts.hops ?? 2, limit: Math.max(edgeLimit, candidates), hubs })
        : Promise.resolve([] as GraphEdge[]),
      this.opts.embedder.embed([message], "query"),
    ]);
    const filter = o.turnToMax !== undefined ? { turnToMax: o.turnToMax } : {};
    const selfK = this.opts.selfK ?? 2;
    const factK = this.opts.factK ?? 5;
    const [chunkHits, userFactHits, selfFactHits, edgeScores] = await Promise.all([
      this.opts.vectors.search(this.chunksCollection, qv!, { limit: this.opts.chunkK ?? 3, filter }),
      selfK < 0
        ? this.opts.vectors.search(this.factsCollection, qv!, { limit: factK, filter })
        : this.opts.vectors.search(this.factsCollection, qv!, { limit: factK, filter: { ...filter, self: false } }),
      selfK > 0 ? this.opts.vectors.search(this.factsCollection, qv!, { limit: selfK, filter: { ...filter, self: true } }) : Promise.resolve([]),
      candidates > 0 && candidateEdges.length > 0
        ? this.opts.vectors.search(this.factsCollection, qv!, { limit: candidateEdges.length, filter: { ids: [...new Set(candidateEdges.map(factId))] } })
        : Promise.resolve([]),
    ]);
    // Rank the neighbourhood by how close each edge's fact is to the message; edges whose fact the
    // vector store does not hold keep the graph's order after the scored ones.
    let edges = candidateEdges;
    if (candidates > 0 && candidateEdges.length) {
      const score = new Map(edgeScores.map((h) => [h.id, h.score]));
      edges = candidateEdges
        .map((e) => ({ ...e, score: score.get(factId(e)) }))
        .sort((a, b) => (b.score ?? -1) - (a.score ?? -1) || a.depth - b.depth || b.turnTo - a.turnTo);
    }
    // One line per triple: the summariser restates a fact every time it comes up (민수 —그렸다→
    // 달빛 고양이 twice, 만화 원서 twice), and each restatement took one of the edgeLimit slots
    // (round 3, t996). The best-ranked instance stays.
    const seen = new Set<string>();
    edges = edges.filter((e) => {
      const k = `${e.subject}\u0000${e.relation}\u0000${e.object}\u0000${e.current}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    edges = edges.slice(0, edgeLimit);
    const min = this.opts.minScore ?? 0;
    const num = (v: unknown) => (typeof v === "number" ? v : Number(v ?? 0));
    const fact = (h: { payload: Record<string, unknown>; score: number }) => ({
      text: String(h.payload.text ?? ""),
      turnFrom: num(h.payload.turnFrom),
      turnTo: num(h.payload.turnTo),
      score: h.score,
      self: Boolean(h.payload.self),
    });
    return {
      seeds,
      edges,
      chunks: chunkHits
        .filter((h) => h.score >= min)
        .map((h) => ({ id: h.id, text: String(h.payload.text ?? ""), turnFrom: num(h.payload.turnFrom), turnTo: num(h.payload.turnTo), score: h.score })),
      facts: [...userFactHits, ...selfFactHits].filter((h) => h.score >= min).map(fact),
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
