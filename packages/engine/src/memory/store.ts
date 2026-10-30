import { DatabaseSync, type StatementSync } from "node:sqlite";
import type { ArchivedFact, Fact, MemoryState, Turn } from "../types.js";

/**
 * Persistent memory (docs/13). The session object is a cache; this file is the truth. One SQLite
 * database per character-user pair, through Node's built-in `node:sqlite` (no server, no native
 * dependency), holding the transcript, every fact ever extracted with its proper-noun edges, the
 * hierarchical summaries and the ledger snapshots. Nothing here grows the prompt: the prompt reads a
 * fixed-size view (ledger, current chapter and season, one recalled page), and this is where the rest
 * lives so that turn 1 is still answerable at turn 1,000,000.
 *
 * Vectors are stored as float32 blobs and compared in JavaScript over a candidate set the caller has
 * already narrowed (an entity's facts, a turn range). There is no ANN index: a narrowed scan of 100k
 * vectors measured 240 ms, and a product of this shape never asks for the whole store at once. If a
 * query ever does, that is the moment to add one, not before.
 */

export type FactStatus = "current" | "superseded";

export interface StoredFact extends ArchivedFact {
  id: number;
  status: FactStatus;
  /** Id of the fact that replaced this one, when `status` is "superseded". */
  supersededBy?: number;
}

export interface EntityRow {
  name: string;
  nFacts: number;
  firstTurn: number;
  lastTurn: number;
}

export interface StoredSummary {
  /** 0 = consolidation summary, 1 = chapter (~100 turns), 2 = season (~1,000 turns). */
  level: number;
  turnFrom: number;
  turnTo: number;
  text: string;
}

export interface VectorHit {
  fact: StoredFact;
  cos: number;
}

/** "하루 씨" / "민수 님" is the name plus an address suffix; the store compares names bare. */
export const bareName = (e: string): string => e.replace(/\s*(씨|님|선배|군|양)$/u, "");

export class MemoryStore {
  private readonly db: DatabaseSync;
  private readonly q: {
    insertTurn: StatementSync;
    insertFact: StatementSync;
    insertEdge: StatementSync;
    upsertEntity: StatementSync;
    factById: StatementSync;
    factsByEntity: StatementSync;
    factsInRange: StatementSync;
    allFacts: StatementSync;
    supersede: StatementSync;
    entities: StatementSync;
    insertLedger: StatementSync;
    lastLedger: StatementSync;
    insertSummary: StatementSync;
    summaries: StatementSync;
  };
  private dims?: number;
  private entityCache?: EntityRow[];

  constructor(readonly path = ":memory:") {
    this.db = new DatabaseSync(path);
    if (path !== ":memory:") {
      this.db.exec("PRAGMA journal_mode = WAL");
      this.db.exec("PRAGMA synchronous = NORMAL");
    }
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS turns (
        idx INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL, ts INTEGER NOT NULL DEFAULT 0, meta TEXT
      );
      CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL, source TEXT NOT NULL, by TEXT, day INTEGER, about TEXT,
        turn_from INTEGER NOT NULL, turn_to INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'current', superseded_by INTEGER,
        embedding BLOB
      );
      CREATE INDEX IF NOT EXISTS facts_turn_to ON facts(turn_to);
      CREATE TABLE IF NOT EXISTS entities (
        name TEXT PRIMARY KEY, n_facts INTEGER NOT NULL, first_turn INTEGER NOT NULL, last_turn INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS fact_entities (
        fact_id INTEGER NOT NULL REFERENCES facts(id), entity TEXT NOT NULL,
        PRIMARY KEY (fact_id, entity)
      );
      CREATE INDEX IF NOT EXISTS fact_entities_entity ON fact_entities(entity, fact_id);
      CREATE TABLE IF NOT EXISTS ledger (turn_to INTEGER PRIMARY KEY, json TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT, level INTEGER NOT NULL,
        turn_from INTEGER NOT NULL, turn_to INTEGER NOT NULL, text TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS summaries_level ON summaries(level, turn_to);
    `);
    this.q = {
      insertTurn: this.db.prepare("INSERT OR IGNORE INTO turns(idx, role, content, ts, meta) VALUES (?, ?, ?, ?, ?)"),
      insertFact: this.db.prepare(
        "INSERT INTO facts(text, source, by, day, about, turn_from, turn_to, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
      ),
      insertEdge: this.db.prepare("INSERT OR IGNORE INTO fact_entities(fact_id, entity) VALUES (?, ?)"),
      upsertEntity: this.db.prepare(
        `INSERT INTO entities(name, n_facts, first_turn, last_turn) VALUES (?, 1, ?, ?)
         ON CONFLICT(name) DO UPDATE SET n_facts = n_facts + 1,
           first_turn = MIN(first_turn, excluded.first_turn), last_turn = MAX(last_turn, excluded.last_turn)`,
      ),
      factById: this.db.prepare("SELECT * FROM facts WHERE id = ?"),
      factsByEntity: this.db.prepare(
        `SELECT f.* FROM facts f JOIN fact_entities fe ON fe.fact_id = f.id
         WHERE fe.entity = ? ORDER BY f.turn_to, f.id`,
      ),
      factsInRange: this.db.prepare("SELECT * FROM facts WHERE turn_to >= ? AND turn_to <= ? ORDER BY turn_to, id"),
      allFacts: this.db.prepare("SELECT * FROM facts ORDER BY id"),
      supersede: this.db.prepare("UPDATE facts SET status = 'superseded', superseded_by = ? WHERE id = ?"),
      entities: this.db.prepare("SELECT name, n_facts, first_turn, last_turn FROM entities ORDER BY n_facts DESC, name"),
      insertLedger: this.db.prepare("INSERT OR REPLACE INTO ledger(turn_to, json) VALUES (?, ?)"),
      lastLedger: this.db.prepare("SELECT turn_to, json FROM ledger ORDER BY turn_to DESC LIMIT 1"),
      insertSummary: this.db.prepare("INSERT INTO summaries(level, turn_from, turn_to, text) VALUES (?, ?, ?, ?)"),
      summaries: this.db.prepare("SELECT level, turn_from, turn_to, text FROM summaries WHERE level = ? ORDER BY turn_to DESC LIMIT ?"),
    };
    const d = this.db.prepare("SELECT value FROM meta WHERE key = 'dims'").get() as { value: string } | undefined;
    if (d) this.dims = Number(d.value);
  }

  // ---- transcript -------------------------------------------------------------------------------

  appendTurns(turns: Turn[]): void {
    this.tx(() => {
      for (const t of turns) this.q.insertTurn.run(t.index, t.role, t.content, t.ts ?? 0, t.meta ? JSON.stringify(t.meta) : null);
    });
  }

  turnCount(): number {
    return (this.db.prepare("SELECT COUNT(*) AS n FROM turns").get() as { n: number }).n;
  }

  turns(from: number, to: number): Turn[] {
    const rows = this.db.prepare("SELECT idx, role, content, ts, meta FROM turns WHERE idx >= ? AND idx <= ? ORDER BY idx").all(from, to) as {
      idx: number;
      role: Turn["role"];
      content: string;
      ts: number;
      meta: string | null;
    }[];
    return rows.map((r) => ({ index: r.idx, role: r.role, content: r.content, ts: r.ts, ...(r.meta ? { meta: JSON.parse(r.meta) } : {}) }) as Turn);
  }

  // ---- facts and their entities ----------------------------------------------------------------

  /**
   * Append facts as the summariser stated them. `embeddings[i]` (unit length) is stored beside
   * `facts[i]` when given. Entities are indexed bare ("하루 씨" → "하루"). Returns the new ids.
   */
  appendFacts(facts: ArchivedFact[], embeddings?: number[][]): number[] {
    if (embeddings && embeddings.length !== facts.length) throw new Error("one embedding per fact");
    const ids: number[] = [];
    this.entityCache = undefined;
    this.tx(() => {
      facts.forEach((f, i) => {
        const vec = embeddings?.[i];
        if (vec) this.checkDims(vec.length);
        const res = this.q.insertFact.run(
          f.text,
          f.source,
          f.by ?? null,
          f.day ?? null,
          f.about ?? null,
          f.turnFrom,
          f.turnTo,
          vec ? Buffer.from(Float32Array.from(vec).buffer) : null,
        );
        const id = Number(res.lastInsertRowid);
        ids.push(id);
        for (const e of new Set((f.entities ?? []).map(bareName).filter((e) => e.length >= 2))) {
          this.q.insertEdge.run(id, e);
          this.q.upsertEntity.run(e, f.turnFrom, f.turnTo);
        }
      });
    });
    return ids;
  }

  factCount(): number {
    return (this.db.prepare("SELECT COUNT(*) AS n FROM facts").get() as { n: number }).n;
  }

  fact(id: number): StoredFact | undefined {
    const row = this.q.factById.get(id) as unknown as FactRow | undefined;
    return row ? this.rowToFact(row) : undefined;
  }

  /** The summariser said `oldId` no longer holds and `newId` is what holds now. */
  supersede(oldId: number, newId: number): void {
    this.q.supersede.run(newId, oldId);
  }

  /**
   * Everything known about one proper noun, oldest first, superseded entries included unless asked
   * otherwise. This is the unit recall shows (docs/13): one page, not five scattered hits.
   */
  entityPage(name: string, opts: { includeSuperseded?: boolean } = {}): StoredFact[] {
    const rows = this.q.factsByEntity.all(bareName(name)) as unknown as FactRow[];
    const facts = rows.map((r) => this.rowToFact(r));
    return opts.includeSuperseded === false ? facts.filter((f) => f.status === "current") : facts;
  }

  factsInRange(turnFrom: number, turnTo: number): StoredFact[] {
    return (this.q.factsInRange.all(turnFrom, turnTo) as unknown as FactRow[]).map((r) => this.rowToFact(r));
  }

  /** The whole archive in the session's shape (for restoring a session or an old-style replay). */
  loadArchive(): ArchivedFact[] {
    const ents = new Map<number, string[]>();
    for (const r of this.db.prepare("SELECT fact_id, entity FROM fact_entities").all() as { fact_id: number; entity: string }[]) {
      (ents.get(r.fact_id) ?? ents.set(r.fact_id, []).get(r.fact_id)!).push(r.entity);
    }
    const out: ArchivedFact[] = [];
    for (const r of this.q.allFacts.iterate() as Iterable<FactRow>) {
      const f = this.rowToFact(r, ents.get(r.id));
      const { id: _id, status: _s, supersededBy: _b, ...plain } = f;
      out.push(plain);
    }
    return out;
  }

  entities(): EntityRow[] {
    // Read once per batch of appends: the trigger check runs on every reply and must not rescan the table.
    return (this.entityCache ??= (this.q.entities.all() as { name: string; n_facts: number; first_turn: number; last_turn: number }[]).map((r) => ({
      name: r.name,
      nFacts: r.n_facts,
      firstTurn: r.first_turn,
      lastTurn: r.last_turn,
    })));
  }

  /**
   * Entities that can trigger recall: tagged on at least `minFacts` facts and on no more than
   * `maxShare` of all facts. A tag on every sixth fact (the two speakers' names, "커피" in a café)
   * says nothing about what one message is about; this is the statistical filter docs/13 uses in
   * place of a word list (docs/12 result 3: 금요일, 아메리카노, 골목, 카운터 were tagged as proper nouns).
   * On the 1000-turn archive at 10%: the speakers (16% and 19%) fall out, the user's cat (7%) stays.
   * The share cap has a floor (`minCap`, default 10 facts) so a young store excludes nothing by share.
   */
  discriminativeEntities(opts: { maxShare?: number; minFacts?: number; minCap?: number } = {}): EntityRow[] {
    const maxShare = opts.maxShare ?? 0.1;
    const minFacts = opts.minFacts ?? 1;
    const cap = Math.max(maxShare * this.factCount(), opts.minCap ?? 10);
    return this.entities().filter((e) => e.nFacts >= minFacts && e.nFacts <= cap);
  }

  /**
   * Known, discriminative entities that appear in `text` (the user's latest message), with the
   * ones the `ledger` text already carries removed: those are visible already and need no recall.
   * Mechanical on both sides; what counts as an entity was the summariser's judgement.
   */
  entitiesIn(text: string, opts: { ledger?: string; maxShare?: number; minFacts?: number; minCap?: number; ignore?: string[] } = {}): EntityRow[] {
    const ignore = new Set((opts.ignore ?? []).map(bareName));
    return this.discriminativeEntities(opts)
      .filter((e) => !ignore.has(e.name) && text.includes(e.name))
      .filter((e) => !(opts.ledger && opts.ledger.includes(e.name)));
  }

  // ---- vectors ----------------------------------------------------------------------------------

  /**
   * Cosine top-k over a narrowed candidate set: the facts of one entity, a turn range, or explicit
   * ids. Without any narrowing it scans the whole table (kept for measurement; a product query
   * should always narrow, docs/13).
   */
  vectorSearch(
    query: number[],
    opts: { k?: number; entity?: string; turnFrom?: number; turnTo?: number; ids?: number[]; includeSuperseded?: boolean } = {},
  ): VectorHit[] {
    const k = opts.k ?? 5;
    const q = Float32Array.from(query);
    let rows: Iterable<FactRow>;
    if (opts.ids) {
      const marks = opts.ids.map(() => "?").join(",");
      rows = this.db.prepare(`SELECT * FROM facts WHERE id IN (${marks}) AND embedding IS NOT NULL`).all(...opts.ids) as unknown as FactRow[];
    } else if (opts.entity !== undefined) {
      rows = this.db
        .prepare(
          `SELECT f.* FROM facts f JOIN fact_entities fe ON fe.fact_id = f.id
           WHERE fe.entity = ? AND f.embedding IS NOT NULL`,
        )
        .all(bareName(opts.entity)) as unknown as FactRow[];
    } else if (opts.turnFrom !== undefined || opts.turnTo !== undefined) {
      rows = this.db
        .prepare("SELECT * FROM facts WHERE turn_to >= ? AND turn_to <= ? AND embedding IS NOT NULL")
        .iterate(opts.turnFrom ?? 0, opts.turnTo ?? Number.MAX_SAFE_INTEGER) as Iterable<FactRow>;
    } else {
      rows = this.db.prepare("SELECT * FROM facts WHERE embedding IS NOT NULL").iterate() as Iterable<FactRow>;
    }
    const top: { row: FactRow; cos: number }[] = [];
    for (const row of rows) {
      if (opts.includeSuperseded === false && row.status === "superseded") continue;
      const b = row.embedding!;
      const v = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      let dot = 0;
      for (let i = 0; i < v.length; i++) dot += v[i]! * q[i]!;
      if (top.length < k) {
        top.push({ row, cos: dot });
        if (top.length === k) top.sort((a, b) => b.cos - a.cos);
      } else if (dot > top[k - 1]!.cos) {
        top[k - 1] = { row, cos: dot };
        top.sort((a, b) => b.cos - a.cos);
      }
    }
    top.sort((a, b) => b.cos - a.cos);
    return top.map((t) => ({ fact: this.rowToFact(t.row), cos: t.cos }));
  }

  // ---- ledger and summaries ---------------------------------------------------------------------

  /** Snapshot the prompt-side memory (without the archive, which lives here already). */
  saveLedger(turnTo: number, memory: MemoryState): void {
    const { archive: _a, ...rest } = memory;
    this.q.insertLedger.run(turnTo, JSON.stringify(rest));
  }

  loadLedger(): { turnTo: number; memory: Omit<MemoryState, "archive"> } | undefined {
    const row = this.q.lastLedger.get() as { turn_to: number; json: string } | undefined;
    return row ? { turnTo: row.turn_to, memory: JSON.parse(row.json) } : undefined;
  }

  putSummary(s: StoredSummary): void {
    this.q.insertSummary.run(s.level, s.turnFrom, s.turnTo, s.text);
  }

  /** Most recent `limit` summaries of one level, newest first. */
  summaries(level: number, limit = 10): StoredSummary[] {
    return (this.q.summaries.all(level, limit) as { level: number; turn_from: number; turn_to: number; text: string }[]).map((r) => ({
      level: r.level,
      turnFrom: r.turn_from,
      turnTo: r.turn_to,
      text: r.text,
    }));
  }

  // ---- housekeeping -----------------------------------------------------------------------------

  /** Bytes on disk (0 for an in-memory store). */
  sizeBytes(): number {
    if (this.path === ":memory:") return 0;
    const { page_count } = this.db.prepare("PRAGMA page_count").get() as { page_count: number };
    const { page_size } = this.db.prepare("PRAGMA page_size").get() as { page_size: number };
    return page_count * page_size;
  }

  close(): void {
    this.db.close();
  }

  private tx(fn: () => void): void {
    this.db.exec("BEGIN");
    try {
      fn();
      this.db.exec("COMMIT");
    } catch (e) {
      this.db.exec("ROLLBACK");
      throw e;
    }
  }

  private checkDims(n: number): void {
    if (this.dims === undefined) {
      this.dims = n;
      this.db.prepare("INSERT OR REPLACE INTO meta(key, value) VALUES ('dims', ?)").run(String(n));
    } else if (this.dims !== n) {
      throw new Error(`embedding has ${n} dims, store holds ${this.dims}`);
    }
  }

  private rowToFact(r: FactRow, entities?: string[]): StoredFact {
    const ents =
      entities ??
      (this.db.prepare("SELECT entity FROM fact_entities WHERE fact_id = ?").all(r.id) as { entity: string }[]).map((x) => x.entity);
    return {
      id: r.id,
      text: r.text,
      source: r.source as Fact["source"],
      ...(r.by ? { by: r.by } : {}),
      ...(r.day !== null ? { day: r.day } : {}),
      ...(r.about === "user" ? { about: "user" as const } : {}),
      turnFrom: r.turn_from,
      turnTo: r.turn_to,
      ...(ents.length ? { entities: ents } : {}),
      status: r.status as FactStatus,
      ...(r.superseded_by !== null ? { supersededBy: r.superseded_by } : {}),
    };
  }
}

interface FactRow {
  id: number;
  text: string;
  source: string;
  by: string | null;
  day: number | null;
  about: string | null;
  turn_from: number;
  turn_to: number;
  status: string;
  superseded_by: number | null;
  embedding: Uint8Array | null;
}
