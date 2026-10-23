import type { FactSource } from "../types.js";

/**
 * The shared world (docs/10). Time is counted in days and four slots; places are a finite list so
 * "same day, same slot, same place" can be computed without a model. Only the ledger is written to.
 */
export type Slot = "morning" | "afternoon" | "evening" | "night";
export const SLOTS: readonly Slot[] = ["morning", "afternoon", "evening", "night"];

export interface WorldPlace {
  id: string;
  name: string;
  region?: string;
  /** Free text the resolver reads ("11시~19시, 화요일 휴무"). */
  hours?: string;
  /** One line of what the place is like, for off-screen scenes. */
  note?: string;
}

export type Tier = "main" | "named";

export interface WorldCharacter {
  /** Character card id (main) or a name (named NPC with no card). */
  id: string;
  name: string;
  tier: Tier;
  /** Where the character is in each slot when nothing else is scheduled. Missing slot = unknown/away. */
  routine: Partial<Record<Slot, string>>;
  /** For named NPCs only: a few lines standing in for a card. */
  sketch?: string;
}

export type TaskStatus = "pending" | "done" | "failed";

/** Something a character agreed to do off-screen, extracted from a conversation. */
export interface WorldTask {
  id: string;
  assignee: string;
  /** What, in one line, past-tense-neutral ("책다락에 가서 민수가 맡겨 둔 책이 들어왔는지 확인하고 받아 온다"). */
  what: string;
  where?: string;
  dueDay: number;
  dueSlot?: Slot;
  /** Who asked (persona name or character id). */
  from: string;
  createdDay: number;
  status: TaskStatus;
  eventId?: string;
}

export type EventScope = "personal" | "regional" | "world";

export interface Knowledge {
  source: Extract<FactSource, "witnessed" | "told" | "rumor">;
  by?: string;
}

/** One ledger entry. Decided when it happens; the scene text is optional (render later). */
export interface WorldEvent {
  id: string;
  day: number;
  slot: Slot;
  where: string;
  who: string[];
  scope: EventScope;
  /** One line, past tense: what is now true because of this. */
  result: string;
  /** 2-4 concrete things a witness would remember (what was said, what it looked like). */
  details: string[];
  /** Per character: how they know. Absent = they do not know. */
  knownBy: Record<string, Knowledge>;
  /** Rendered scene text, when a scene was written. */
  scene?: string;
  origin: "errand" | "scene" | "chat" | "catchup";
  taskId?: string;
}

export interface WorldState {
  id: string;
  clock: { day: number; slot: Slot };
  places: WorldPlace[];
  characters: Record<string, WorldCharacter>;
  tasks: WorldTask[];
  ledger: WorldEvent[];
  /** Per character: how many ledger entries have already been folded into their memory. */
  synced: Record<string, number>;
  /** Per character: the last day they spoke with the user. */
  lastSeen: Record<string, number>;
}

/** What the prompt gets about the present (builder `# NOW` block). */
export interface WorldBlock {
  day: number;
  slot: Slot;
  place: string;
  /** Days since the character last spoke with the user; 0 = same day. */
  daysSinceUser: number;
}
