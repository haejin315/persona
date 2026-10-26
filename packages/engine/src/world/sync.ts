import type { ChatSession } from "../session.js";
import type { Fact } from "../types.js";
import type { World } from "./world.js";

/**
 * Fold what the character came to know off-screen into their memory, as facts carrying the
 * ledger's source, and set the `# NOW` block. Call it before the user speaks to the character on
 * a new day. Returns the facts that were added.
 */
export function syncCharacter(session: ChatSession, world: World, characterId: string): Fact[] {
  const events = world.unsynced(characterId);
  const have = new Set(session.memory.facts.map((f) => f.text));
  const fresh = world.factsFor(characterId, events).filter((f) => !have.has(f.text));
  if (fresh.length) session.memory = { ...session.memory, facts: [...session.memory.facts, ...fresh] };
  world.markSynced(characterId);
  const block = world.blockFor(characterId);
  session.world = block;
  if (block.daysSinceUser > 0 && session.turns.length) {
    session.markBoundary(`${block.daysSinceUser === 1 ? "The next day" : `${block.daysSinceUser} days later`}: day ${block.day}, ${block.slot}, at ${block.place}`);
  }
  return fresh;
}
