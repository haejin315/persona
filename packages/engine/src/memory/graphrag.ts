import type { ArchivedFact, Turn } from "../types.js";
import { chunkRange, type ChunkOptions } from "./chunker.js";
import type { GraphEdge, GraphStore } from "./graphdb.js";
import type { Embedder } from "./recall.js";
import type { VectorFilter, VectorHitRow, VectorIndex } from "./vectordb.js";

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
  /**
   * After ranking, this many more candidate edges with the top edge's subject and relation are
   * kept next to it (default 2): the message "I drew comics as a child" ranks 민수 —그렸다→ 만화
   * first, and the title is 민수 —그렸다→ 달빛 고양이, 15th by cosine (round 3, t996). 0 = off.
   */
  expandTop?: number;
  /**
   * What is embedded as the query (round 5, docs/14). "message": the whole user message (rounds
   * 1-4). "speech": the message without its *stage directions*. "sentences": each spoken sentence
   * on its own, every candidate scored by its best sentence - a message carries several topics
   * ("아, 하나만요." / "어릴 때 만화를 그렸어요." / "내일 또 올게요."), and one vector of the whole
   * message lands between them (t996: the title's edge ranked 29th of 51 by the whole message,
   * 6th by its best sentence).
   */
  query?: "message" | "speech" | "sentences";
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
/** The message without its *stage directions*. */
export const speechOf = (message: string): string => message.replace(/\*[^*\n]*\*/g, " ").replace(/\s+/g, " ").trim();

/** The spoken sentences of a message, quotes stripped, for the "sentences" query mode. */
export function speechSentences(message: string): string[] {
  return speechOf(message)
    .split(/(?<=[.!?。…])\s+|"\s+"|"\s*$/u)
    .map((s) => s.replace(/^["“”]+|["“”]+$/gu, "").trim())
    .filter((s) => s.length >= 2);
}

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
  async retrieve(message: string, o: { turnToMax?: number; ignore?: string[]; context?: string[] } = {}): Promise<GraphRagResult> {
    const ignore = new Set((o.ignore ?? []).map((s) => s.replace(/\s*(씨|님)$/u, "")));
    const edgeLimit = this.opts.edgeLimit ?? 12;
    const candidates = this.opts.edgeCandidates ?? 300;
    // Seeds come from the message and, when given, the context (round 5: the character's own last
    // reply - a name it brought up itself is what the next turn is about, t814/816). The vectors
    // are queried with the message alone.
    const seedText = [message, ...(o.context ?? [])].join("\n");
    const { seeds, hubs } = edgeLimit > 0 ? await this.seeds(seedText, ignore) : { seeds: [], hubs: [] };
    const mode = this.opts.query ?? "message";
    const queries = mode === "sentences" ? speechSentences(message) : mode === "speech" ? [speechOf(message)] : [message];
    if (!queries.length || !queries[0]) queries.splice(0, queries.length, message);
    const [candidateEdges, qvs] = await Promise.all([
      seeds.length
        ? this.opts.graph.neighborhood(seeds, { hops: this.opts.hops ?? 2, limit: Math.max(edgeLimit, candidates), hubs })
        : Promise.resolve([] as GraphEdge[]),
      this.opts.embedder.embed(queries, "query"),
    ]);
    // One search per query vector; a point's score is its best over them (max fusion).
    const search = async (collection: string, opts: { limit: number; filter?: VectorFilter }) => {
      const lists = await Promise.all(qvs.map((qv) => this.opts.vectors.search(collection, qv, opts)));
      if (lists.length === 1) return lists[0]!;
      const best = new Map<string, VectorHitRow>();
      for (const hits of lists) for (const h of hits) if ((best.get(h.id)?.score ?? -Infinity) < h.score) best.set(h.id, h);
      return [...best.values()].sort((a, b) => b.score - a.score).slice(0, opts.limit);
    };
    const filter = o.turnToMax !== undefined ? { turnToMax: o.turnToMax } : {};
    const selfK = this.opts.selfK ?? 2;
    const factK = this.opts.factK ?? 5;
    const [chunkHits, userFactHits, selfFactHits, edgeScores] = await Promise.all([
      search(this.chunksCollection, { limit: this.opts.chunkK ?? 3, filter }),
      selfK < 0 ? search(this.factsCollection, { limit: factK, filter }) : search(this.factsCollection, { limit: factK, filter: { ...filter, self: false } }),
      selfK > 0 ? search(this.factsCollection, { limit: selfK, filter: { ...filter, self: true } }) : Promise.resolve([] as VectorHitRow[]),
      candidates > 0 && candidateEdges.length > 0
        ? search(this.factsCollection, { limit: candidateEdges.length, filter: { ids: [...new Set(candidateEdges.map(factId))] } })
        : Promise.resolve([] as VectorHitRow[]),
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
    // The top edge's siblings (same subject and relation) come next to it, ahead of the cut: what
    // the message is about is often one hop beside what it says.
    const expand = this.opts.expandTop ?? 2;
    if (expand > 0 && edges.length > 1) {
      const top = edges[0]!;
      const siblings = edges.slice(1).filter((e) => e.subject === top.subject && e.relation === top.relation).slice(0, expand);
      if (siblings.length) {
        const rest = edges.slice(1).filter((e) => !siblings.includes(e));
        edges = [top, ...siblings, ...rest];
      }
    }
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

/**
 * The context block the character reads. Empty string when nothing was found. Style "v1" is the
 * round 1-3 block (graph / facts / exchanges). Style "v2" (round 4) separates what the user said
 * from what the character itself said or planned at the time, and states how to read both: a
 * question about the past is answered from here even when MEMORY shows something newer, a detail
 * here is said out loud when asked for, and a plan whose day has passed is over, not re-proposed.
 */
export function renderGraphRag(r: GraphRagResult, now: number, user: string, o: { character?: string; style?: "v1" | "v2" } = {}): string {
  const age = (turnTo: number) => `[about ${Math.max(1, Math.round((now - turnTo) / 2))} exchanges ago]`;
  const parts: string[] = [];
  if (o.style === "v2") {
    const me = (o.character ?? "").replace(/\s*(씨|님)$/u, "");
    const mine = (e: GraphEdge) => !!me && e.subject === me;
    const edgeLine = (e: GraphEdge) => `- ${age(e.turnTo)} ${e.subject} —${e.relation}→ ${e.object}${e.current ? "" : " (no longer)"}: ${e.text}`;
    const factLine = (f: GraphRagResult["facts"][number]) => `- ${age(f.turnTo)} ${f.text}`;
    const about = [...r.edges.filter((e) => !mine(e)).map(edgeLine), ...r.facts.filter((f) => !f.self).map(factLine)];
    const own = [...r.edges.filter(mine).map(edgeLine), ...r.facts.filter((f) => f.self).map(factLine)];
    if (about.length) parts.push(`## ABOUT ${user} (what ${user} told you or you saw, at the time marked; "no longer" = true then, changed since; a plan or a date in an old line may already have come and gone - a later line about the same thing tells you)\n${about.join("\n")}`);
    if (own.length) parts.push(`## WHAT YOU SAID OR PLANNED THEN (your own words at the time marked - a promise or plan here was for that day; if that day is past, it is over and you do not propose it again)\n${own.join("\n")}`);
    if (r.chunks.length) parts.push(`## PAST EXCHANGES, verbatim\n${r.chunks.map((c) => `${age(c.turnTo)}\n${c.text}`).join("\n\n")}`);
    if (!parts.length) return "";
    return `# WHAT YOU REMEMBER ABOUT THIS (from the whole history with ${user}, found for what ${user} just said; as true as MEMORY; each line says how long ago it was said, and "tomorrow" or "next week" inside a line counts from then, not from now)
How to use it: when ${user} asks what was said or what things were like before, answer from here, in your own words, even if MEMORY shows something newer now - the old thing was true then, say so plainly. A name, a title, a number or a place that is here is said out loud when ${user} asks for it; never say you were not told something that is written here. For what is true right now, MEMORY above and this conversation win.

${parts.join("\n\n")}`;
  }
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
