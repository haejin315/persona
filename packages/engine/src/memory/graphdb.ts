import neo4j, { type Driver } from "neo4j-driver";

/**
 * Knowledge graph behind the Graph RAG memory (docs/14): Neo4j over Bolt. Nodes are single
 * entities or concepts the summariser named (a person, a pet, a place, a drink, a habit); every
 * extracted fact that has a subject and an object becomes one relationship carrying the fact's
 * text, turn range and whether it still holds. A new triple with the same subject and relation
 * but a different object marks the old one `current: false` (아메리카노 → 라떼), so change is in
 * the graph, not lost. Namespaced by conversation (`ns`) inside one database.
 */

export interface Triple {
  subject: string;
  relation: string;
  object: string;
  /**
   * The summariser's judgement (memory rules v6) that this triple replaces what held before about
   * the same subject and relation. Without it, an older edge with another object stays current:
   * 민수 그렸다 만화 and 민수 그렸다 달빛 고양이 are both true.
   */
  replaces?: boolean;
}

export interface GraphEdge extends Triple {
  text: string;
  turnFrom: number;
  turnTo: number;
  current: boolean;
  /** Hops from the seed entity the query started at (1 = direct). */
  depth: number;
  /** Cosine of the edge's fact to the user's message, when the retrieval ranked it (GraphRag). */
  score?: number;
}

export interface GraphStore {
  id: string;
  addTriples(edges: Omit<GraphEdge, "current" | "depth">[]): Promise<void>;
  /** Entities in this namespace with how many edges touch each. */
  entities(): Promise<{ name: string; degree: number }[]>;
  /**
   * Edges within `hops` of any seed, nearest then most recent first, at most `limit`. A path may
   * pass through a node in `hubs` (the speakers: on hundreds of edges) only along the relation it
   * arrived by - 만화 ←그렸다— 민수 —그렸다→ 달빛 고양이 - otherwise every edge of the hub is two
   * hops from everything.
   */
  neighborhood(seeds: string[], opts: { hops: number; limit: number; hubs?: string[] }): Promise<GraphEdge[]>;
  count(): Promise<{ nodes: number; edges: number }>;
  clear(): Promise<void>;
  close(): Promise<void>;
}

const norm = (s: string) => s.replace(/\s*(씨|님)$/u, "").replace(/\s+/g, " ").trim();

export class Neo4jGraph implements GraphStore {
  readonly id: string;
  private driver: Driver;
  private ready?: Promise<void>;

  constructor(
    readonly ns: string,
    url = process.env.NEO4J_URL ?? "bolt://localhost:7687",
    user = process.env.NEO4J_USER ?? "neo4j",
    password = process.env.NEO4J_PASSWORD ?? "persona-dev",
  ) {
    this.driver = neo4j.driver(url, neo4j.auth.basic(user, password));
    this.id = `neo4j:${url}/${ns}`;
  }

  private ensure(): Promise<void> {
    return (this.ready ??= this.driver
      .executeQuery("CREATE CONSTRAINT entity_ns_name IF NOT EXISTS FOR (e:Entity) REQUIRE (e.ns, e.name) IS UNIQUE")
      .then(() => undefined));
  }

  async addTriples(edges: Omit<GraphEdge, "current" | "depth">[]): Promise<void> {
    await this.ensure();
    const rows = edges
      .map((e) => ({ ...e, subject: norm(e.subject), object: norm(e.object), relation: e.relation.trim(), replaces: e.replaces === true }))
      .filter((e) => e.subject && e.object && e.relation && e.subject !== e.object);
    if (!rows.length) return;
    await this.driver.executeQuery(
      `UNWIND $rows AS r
       MERGE (s:Entity {ns: $ns, name: r.subject})
       MERGE (o:Entity {ns: $ns, name: r.object})
       WITH s, o, r
       // When the summariser judged this a change, the same subject and relation pointing elsewhere
       // was then, and this is now.
       OPTIONAL MATCH (s)-[old:REL {ns: $ns, rel: r.relation, current: true}]->(other:Entity)
         WHERE other <> o AND r.replaces
       SET old.current = false
       WITH s, o, r
       MERGE (s)-[x:REL {ns: $ns, rel: r.relation, text: r.text}]->(o)
       ON CREATE SET x.turnFrom = r.turnFrom, x.turnTo = r.turnTo, x.current = true`,
      { ns: this.ns, rows },
    );
  }

  async entities(): Promise<{ name: string; degree: number }[]> {
    await this.ensure();
    const res = await this.driver.executeQuery(
      "MATCH (e:Entity {ns: $ns}) OPTIONAL MATCH (e)-[r:REL {ns: $ns}]-() RETURN e.name AS name, count(r) AS degree",
      { ns: this.ns },
    );
    const n = (v: unknown) => (neo4j.isInt(v) ? (v as { toNumber(): number }).toNumber() : Number(v));
    return res.records.map((r) => ({ name: r.get("name") as string, degree: n(r.get("degree")) }));
  }

  async neighborhood(seeds: string[], opts: { hops: number; limit: number; hubs?: string[] }): Promise<GraphEdge[]> {
    await this.ensure();
    if (!seeds.length) return [];
    const names = seeds.map(norm);
    const hubs = (opts.hubs ?? []).map(norm);
    const hops = Math.max(1, Math.min(3, opts.hops));
    // Variable-length paths from each seed; every relationship on a path is an edge, tagged with
    // how far along the path it sits. A hub in the middle of a path must be entered and left by
    // the same relation. Deduplicated by relationship id, nearest then newest first.
    const res = await this.driver.executeQuery(
      `MATCH (seed:Entity {ns: $ns}) WHERE seed.name IN $names
       MATCH p = (seed)-[:REL*1..${hops}]-(x:Entity)
       WITH relationships(p) AS rels, nodes(p) AS ns_
       WHERE all(i IN range(1, size(ns_) - 2) WHERE NOT ns_[i].name IN $hubs OR rels[i - 1].rel = rels[i].rel)
       UNWIND range(0, size(rels) - 1) AS i
       WITH rels[i] AS r, i + 1 AS depth
       WHERE r.ns = $ns
       WITH r, min(depth) AS depth
       RETURN startNode(r).name AS subject, r.rel AS relation, endNode(r).name AS object,
              r.text AS text, r.turnFrom AS turnFrom, r.turnTo AS turnTo, r.current AS current, depth
       ORDER BY depth ASC, turnTo DESC
       LIMIT $limit`,
      { ns: this.ns, names, hubs, limit: neo4j.int(opts.limit) },
    );
    const n = (v: unknown) => (neo4j.isInt(v) ? (v as { toNumber(): number }).toNumber() : Number(v));
    return res.records.map((r) => ({
      subject: r.get("subject"),
      relation: r.get("relation"),
      object: r.get("object"),
      text: r.get("text"),
      turnFrom: n(r.get("turnFrom")),
      turnTo: n(r.get("turnTo")),
      current: Boolean(r.get("current")),
      depth: n(r.get("depth")),
    }));
  }

  async count(): Promise<{ nodes: number; edges: number }> {
    await this.ensure();
    const a = await this.driver.executeQuery("MATCH (e:Entity {ns: $ns}) RETURN count(e) AS n", { ns: this.ns });
    const b = await this.driver.executeQuery("MATCH ()-[r:REL {ns: $ns}]->() RETURN count(r) AS n", { ns: this.ns });
    const n = (v: unknown) => (neo4j.isInt(v) ? (v as { toNumber(): number }).toNumber() : Number(v));
    return { nodes: n(a.records[0]!.get("n")), edges: n(b.records[0]!.get("n")) };
  }

  async clear(): Promise<void> {
    await this.ensure();
    await this.driver.executeQuery("MATCH (e:Entity {ns: $ns}) DETACH DELETE e", { ns: this.ns });
  }

  async close(): Promise<void> {
    await this.driver.close();
  }
}

/** In-memory stand-in with the same interface (tests, offline runs). */
export class MemoryGraph implements GraphStore {
  readonly id = "memory-graph";
  private edges: GraphEdge[] = [];

  async addTriples(edges: Omit<GraphEdge, "current" | "depth">[]): Promise<void> {
    for (const e0 of edges) {
      const e = { ...e0, subject: norm(e0.subject), object: norm(e0.object), relation: e0.relation.trim() };
      if (!e.subject || !e.object || !e.relation || e.subject === e.object) continue;
      if (e.replaces) for (const old of this.edges) if (old.subject === e.subject && old.relation === e.relation && old.object !== e.object) old.current = false;
      const dup = this.edges.find((x) => x.subject === e.subject && x.relation === e.relation && x.object === e.object && x.text === e.text);
      if (!dup) this.edges.push({ ...e, current: true, depth: 0 });
    }
  }

  async entities(): Promise<{ name: string; degree: number }[]> {
    const deg = new Map<string, number>();
    for (const e of this.edges) {
      deg.set(e.subject, (deg.get(e.subject) ?? 0) + 1);
      deg.set(e.object, (deg.get(e.object) ?? 0) + 1);
    }
    return [...deg.entries()].map(([name, degree]) => ({ name, degree }));
  }

  async neighborhood(seeds: string[], opts: { hops: number; limit: number; hubs?: string[] }): Promise<GraphEdge[]> {
    const hubs = new Set((opts.hubs ?? []).map(norm));
    const found = new Map<GraphEdge, number>();
    // Frontier nodes with the relations the walk arrived by (null = a seed, any relation leaves).
    let frontier = new Map<string, Set<string> | null>(seeds.map((s) => [norm(s), null]));
    const seen = new Set(frontier.keys());
    for (let d = 1; d <= opts.hops && frontier.size; d++) {
      const next = new Map<string, Set<string> | null>();
      for (const e of this.edges) {
        for (const [here, there] of [[e.subject, e.object], [e.object, e.subject]] as const) {
          if (!frontier.has(here)) continue;
          const arrived = frontier.get(here);
          // A hub is crossed only along the relation that led into it.
          if (arrived && hubs.has(here) && !arrived.has(e.relation)) continue;
          if (!found.has(e)) found.set(e, d);
          if (seen.has(there)) continue;
          const rels = next.get(there) ?? new Set<string>();
          rels.add(e.relation);
          next.set(there, rels);
        }
      }
      for (const n of next.keys()) seen.add(n);
      frontier = next;
    }
    return [...found.entries()]
      .map(([e, depth]) => ({ ...e, depth }))
      .sort((a, b) => a.depth - b.depth || b.turnTo - a.turnTo)
      .slice(0, opts.limit);
  }

  async count(): Promise<{ nodes: number; edges: number }> {
    return { nodes: (await this.entities()).length, edges: this.edges.length };
  }

  async clear(): Promise<void> {
    this.edges = [];
  }

  async close(): Promise<void> {}
}
