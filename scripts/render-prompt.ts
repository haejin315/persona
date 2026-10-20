/**
 * Print the system prompt a prompt version would build for a character, with a few fake recent
 * turns, so prompt changes can be eyeballed without an LLM call.
 *
 *   npx tsx scripts/render-prompt.ts v3-recent haru ko [section]
 */
import { emptyMemory, findCharacter, getPromptVersion, type Lang, type Turn } from "@persona/engine";

const [version = "v3-structured", characterId = "haru", lang = "ko", section] = process.argv.slice(2);
const card = findCharacter(characterId);
const name = card.name[lang as Lang] ?? card.id;
const recent: Turn[] = [
  { role: "user", content: '*우산을 턴다.* "오늘 비 많이 오네요."', index: 1, ts: 0 },
  { role: "character", content: `*행주를 접는다.*\n\n"...네. 커피 식어요. 앉으세요."\n\n*한 박자 쉬고.*\n\n"...아니에요, 농담이에요."`, index: 2, ts: 0 },
  { role: "user", content: '"고마워요. 주말에 뭐 하세요?"', index: 3, ts: 0 },
  { role: "character", content: `*컵 받침을 맞춘다.*\n\n"헌책방, 가 보세요. 주말에."`, index: 4, ts: 0 },
];
const messages = getPromptVersion(version).build({ card, lang: lang as Lang, memory: emptyMemory(), recent, persona: { name: "민수" } });
const system = messages[0]!.content;
if (section) {
  const i = system.indexOf(`# ${section}`);
  const j = system.indexOf("\n# ", i + 1);
  console.log(i === -1 ? `(no section ${section})` : system.slice(i, j === -1 ? undefined : j));
} else {
  console.log(system);
}
console.log(`\n--- ${name} · ${version} · ${system.length} chars, ${messages.length - 1} recent turns`);
