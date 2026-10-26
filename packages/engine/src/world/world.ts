import { readFileSync } from "node:fs";
import YAML from "yaml";
import { relativeDay, type Fact } from "../types.js";
import { SLOTS, WEEKDAYS, type Slot, type Weekday, type WorldBlock, type WorldCharacter, type WorldEvent, type WorldPlace, type WorldState, type WorldTask } from "./types.js";

export interface WorldFile {
  id: string;
  places: WorldPlace[];
  characters: WorldCharacter[];
  /** Starting day (default 1). */
  startDay?: number;
  /** Weekday of day 1 (default Monday). */
  startWeekday?: Weekday;
}

const nextSlot = (s: Slot): Slot => SLOTS[(SLOTS.indexOf(s) + 1) % SLOTS.length]!;

/**
 * The world state plus the code-only operations on it: where everyone is, which tasks fall due,
 * what a character has not yet been told. Anything that needs a model lives in llm.ts and only
 * writes to the ledger through `record`.
 */
export class World {
  constructor(public state: WorldState) {}

  /**
   * A witness keeps every detail of errands and of the last `detailDays` days; an ordinary day older
   * than that keeps its result only (T6: a week away injected 51-76 facts, most of them details).
   */
  detailDays = 2;

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
    if (def.startWeekday && !WEEKDAYS.includes(def.startWeekday)) throw new Error(`world ${def.id}: unknown weekday "${def.startWeekday}"`);
    return new World({
      id: def.id,
      clock: { day: def.startDay ?? 1, slot: "morning" },
      startWeekday: def.startWeekday ?? "Monday",
      conditions: {},
      user: {},
      places: def.places,
      characters,
      tasks: [],
      ledger: [],
      synced: {},
      lastSeen: {},
    });
  }

  static restore(state: WorldState): World {
    const s = structuredClone(state);
    s.startWeekday ??= "Monday";
    s.conditions ??= {};
    s.user ??= {};
    return new World(s);
  }

  get day() {
    return this.state.clock.day;
  }
  get slot() {
    return this.state.clock.slot;
  }

  weekdayOf(day: number): Weekday {
    const start = WEEKDAYS.indexOf(this.state.startWeekday);
    return WEEKDAYS[(((start + day - 1) % 7) + 7) % 7]!;
  }
  get weekday(): Weekday {
    return this.weekdayOf(this.day);
  }

  /** A place's current condition, when it is not as usual. */
  condition(placeId: string): string | undefined {
    return this.state.conditions[placeId];
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

  /** True when the character spent that slot with the user (a `chat` ledger entry). */
  withUser(characterId: string, day: number, slot: Slot): WorldEvent | undefined {
    return this.state.ledger.find((e) => e.origin === "chat" && e.day === day && e.slot === slot && e.who.includes(characterId));
  }

  /** Record that the user spent a slot with a character where they are; the character's place for that slot. */
  recordChat(characterId: string, user: string, placeId: string): WorldEvent {
    const c = this.character(characterId);
    return this.record({
      day: this.day,
      slot: this.slot,
      where: placeId,
      who: [characterId],
      scope: "personal",
      result: `${c.name} spent the ${this.slot} with ${user} at ${this.place(placeId).name}`,
      details: [],
      // Nobody "knows" this as a fact: the chat itself is in the character's transcript. The entry
      // only fixes where the character was (occupancy for tasks, location for routine scenes).
      knownBy: {},
      origin: "chat",
    });
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
    // The same outing asked for twice (the user repeats the errand in a later chat) is one task.
    const same = this.state.tasks.find((x) => x.status === "pending" && x.assignee === t.assignee && x.where === t.where && x.dueDay === t.dueDay);
    if (same) {
      if (!same.what.includes(t.what) && !t.what.includes(same.what)) same.what = `${same.what} / ${t.what}`;
      return same;
    }
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
    for (const [placeId, cond] of Object.entries(event.effects ?? {})) {
      this.place(placeId); // throws on an unknown place
      if (cond === null) delete this.state.conditions[placeId];
      else this.state.conditions[placeId] = cond;
    }
    for (const [label, state] of Object.entries(event.userEffects ?? {})) {
      if (state === null) delete this.state.user[label];
      else this.state.user[label] = state;
    }
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
      const src = { source: k.source, ...(k.by ? { by: k.by } : {}), day: e.day } as const;
      out.push({ text: `${e.slot}, ${this.place(e.where).name}: ${e.result}`, ...src, ...(e.aboutUser ? { about: "user" as const } : {}) });
      // A witness remembers the details (of errands always, of ordinary days for a while); someone who was told gets the result only.
      if (k.source === "witnessed" && (e.taskId || this.day - e.day <= this.detailDays)) for (const d of e.details) out.push({ text: d, ...src });
      // What happened to the user's things is a marked fact for those who saw it or were told; a rumour
      // carries no such detail (T9: Haru "heard around" the state of the user's book and then claimed she was there).
      if (k.source !== "rumor") for (const [label, state] of Object.entries(e.userEffects ?? {})) if (state) out.push({ text: `${label}: ${state}`, ...src, about: "user" });
    }
    return out;
  }

  /** A day relative to today, in the prompt's English (the character answers in the card's language). */
  dayLabel(day: number): string {
    return relativeDay(day, this.day);
  }

  /** The `# NOW` block for a character about to talk to the user. */
  blockFor(characterId: string): WorldBlock {
    const placeId = this.whereIs(characterId);
    const last = this.state.lastSeen[characterId];
    const cond = placeId ? this.condition(placeId) : undefined;
    return {
      day: this.day,
      weekday: this.weekday,
      slot: this.slot,
      place: placeId ? this.place(placeId).name : "(unknown)",
      ...(cond ? { placeCondition: cond } : {}),
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
