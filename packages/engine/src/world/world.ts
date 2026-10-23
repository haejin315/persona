import { readFileSync } from "node:fs";
import YAML from "yaml";
import type { Fact } from "../types.js";
import { SLOTS, type Slot, type WorldBlock, type WorldCharacter, type WorldEvent, type WorldPlace, type WorldState, type WorldTask } from "./types.js";

export interface WorldFile {
  id: string;
  places: WorldPlace[];
  characters: WorldCharacter[];
  /** Starting day (default 1). */
  startDay?: number;
}

const nextSlot = (s: Slot): Slot => SLOTS[(SLOTS.indexOf(s) + 1) % SLOTS.length]!;

/**
 * The world state plus the code-only operations on it: where everyone is, which tasks fall due,
 * what a character has not yet been told. Anything that needs a model lives in llm.ts and only
 * writes to the ledger through `record`.
 */
export class World {
  constructor(public state: WorldState) {}

  static fromFile(path: string): World {
    const raw = YAML.parse(readFileSync(path, "utf8")) as WorldFile;
    return World.create(raw);
  }

  static create(def: WorldFile): World {
    const characters: Record<string, WorldCharacter> = {};
    for (const c of def.characters) {
      for (const [slot, place] of Object.entries(c.routine)) {
        if (!SLOTS.includes(slot as Slot)) throw new Error(`world ${def.id}: character ${c.id} has unknown slot "${slot}"`);
        if (place && !def.places.some((p) => p.id === place)) throw new Error(`world ${def.id}: character ${c.id} routine points at unknown place "${place}"`);
      }
      characters[c.id] = c;
    }
    return new World({
      id: def.id,
      clock: { day: def.startDay ?? 1, slot: "morning" },
      places: def.places,
      characters,
      tasks: [],
      ledger: [],
      synced: {},
      lastSeen: {},
    });
  }

  static restore(state: WorldState): World {
    return new World(structuredClone(state));
  }

  get day() {
    return this.state.clock.day;
  }
  get slot() {
    return this.state.clock.slot;
  }

  place(id: string): WorldPlace {
    const p = this.state.places.find((x) => x.id === id);
    if (!p) throw new Error(`world ${this.state.id}: unknown place "${id}"`);
    return p;
  }

  character(id: string): WorldCharacter {
    const c = this.state.characters[id];
    if (!c) throw new Error(`world ${this.state.id}: unknown character "${id}"`);
    return c;
  }

  /** Where a character is in a slot: a due task's place wins over the routine. */
  whereIs(characterId: string, day = this.day, slot = this.slot): string | undefined {
    const task = this.state.tasks.find(
      (t) => t.assignee === characterId && t.status === "pending" && t.dueDay === day && (t.dueSlot ?? slot) === slot && t.where,
    );
    return task?.where ?? this.character(characterId).routine[slot];
  }

  /** Characters (other than `except`) at the same place in the same slot. */
  whoElseAt(placeId: string, day: number, slot: Slot, except?: string): string[] {
    return Object.keys(this.state.characters).filter((id) => id !== except && this.whereIs(id, day, slot) === placeId);
  }

  addTask(t: Omit<WorldTask, "id" | "status" | "createdDay">): WorldTask {
    const task: WorldTask = { ...t, id: `task-${this.state.tasks.length + 1}`, status: "pending", createdDay: this.day };
    this.state.tasks.push(task);
    return task;
  }

  /** Pending tasks due on or before `day`. */
  dueTasks(day = this.day): WorldTask[] {
    return this.state.tasks.filter((t) => t.status === "pending" && t.dueDay <= day);
  }

  /** The single write point. */
  record(e: Omit<WorldEvent, "id">): WorldEvent {
    const event: WorldEvent = { ...e, id: `ev-${this.state.ledger.length + 1}` };
    this.state.ledger.push(event);
    if (event.taskId) {
      const t = this.state.tasks.find((x) => x.id === event.taskId);
      if (t) {
        if (t.status === "pending") t.status = "done";
        t.eventId = event.id;
      }
    }
    return event;
  }

  tick(): void {
    const s = nextSlot(this.state.clock.slot);
    if (s === "morning") this.state.clock.day++;
    this.state.clock.slot = s;
  }

  /** Move the clock (no events are resolved here; see llm.ts advanceDay). */
  setClock(day: number, slot: Slot = "morning"): void {
    this.state.clock = { day, slot };
  }

  /** Ledger entries this character knows about and has not yet had folded into memory. */
  unsynced(characterId: string): WorldEvent[] {
    const from = this.state.synced[characterId] ?? 0;
    return this.state.ledger.slice(from).filter((e) => e.knownBy[characterId]);
  }

  markSynced(characterId: string): void {
    this.state.synced[characterId] = this.state.ledger.length;
  }

  /** Facts for the character's memory, with the source the ledger says. */
  factsFor(characterId: string, events: WorldEvent[]): Fact[] {
    const out: Fact[] = [];
    for (const e of events) {
      const k = e.knownBy[characterId]!;
      const src = { source: k.source, ...(k.by ? { by: k.by } : {}) } as const;
      out.push({ text: `${this.dayLabel(e.day)}, ${e.slot}, ${this.place(e.where).name}: ${e.result}`, ...src });
      // A witness remembers the details; someone who was told gets the result only.
      if (k.source === "witnessed") for (const d of e.details) out.push({ text: d, ...src });
    }
    return out;
  }

  /** A day relative to today, in the prompt's English (the character answers in the card's language). */
  dayLabel(day: number): string {
    const diff = this.day - day;
    if (diff === 0) return "today";
    if (diff === 1) return "yesterday";
    if (diff > 1) return `${diff} days ago`;
    return diff === -1 ? "tomorrow" : `in ${-diff} days`;
  }

  /** The `# NOW` block for a character about to talk to the user. */
  blockFor(characterId: string): WorldBlock {
    const placeId = this.whereIs(characterId);
    const last = this.state.lastSeen[characterId];
    return {
      day: this.day,
      slot: this.slot,
      place: placeId ? this.place(placeId).name : "(unknown)",
      daysSinceUser: last === undefined ? 0 : this.day - last,
    };
  }

  sawUser(characterId: string): void {
    this.state.lastSeen[characterId] = this.day;
  }

  snapshot(): WorldState {
    return structuredClone(this.state);
  }
}
