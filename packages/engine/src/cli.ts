/**
 * Interactive terminal chat for quick manual testing.
 *
 *   npm run chat -- --character haru --lang ko --persona "민수: 28살 회사원, 매일 아침 출근길에 카페에 들른다"
 *   npm run chat -- --character kael --lang ja --model openai:gpt-5-nano --prompt v1-flat
 *
 * Write like the apps do:  *행동이나 지문*  "대사"
 * Commands:  /more (character continues)  /regen  /memory  /prompt  /quit
 */
import "dotenv/config";
import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { findCharacter, loadAllCharacters } from "./characters.js";
import { parseModelSpec, providerFromEnv } from "./providers/index.js";
import { ChatSession } from "./session.js";
import type { Lang, Turn, UserPersona } from "./types.js";

function arg(name: string, fallback?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 ? process.argv[i + 1] : fallback;
}

/** "이름: 설명" or just "이름". */
function parsePersona(spec: string | undefined): UserPersona | undefined {
  if (!spec) return undefined;
  const i = spec.indexOf(":");
  if (i === -1) return { name: spec.trim() };
  return { name: spec.slice(0, i).trim(), description: spec.slice(i + 1).trim() };
}

async function main() {
  const characterId = arg("character");
  const lang = (arg("lang", "ko") as Lang) ?? "ko";
  const spec = parseModelSpec(arg("model", "openai")!);
  const promptVersion = arg("prompt");
  const persona = parsePersona(arg("persona"));

  if (!characterId) {
    console.log("Available characters:");
    for (const c of loadAllCharacters()) console.log(`  ${c.id}  (${c.genre})  ${c.name.ko}`);
    console.log(
      '\nUsage: npm run chat -- --character <id> [--lang ko|ja] [--model openai[:model]|ollama[:model]] [--prompt v1-flat|v2-structured] [--persona "이름: 설명"]',
    );
    return;
  }

  const card = findCharacter(characterId);
  const provider = providerFromEnv(spec.provider);
  const session = new ChatSession({
    card,
    lang,
    provider,
    promptVersion,
    model: spec.model,
    persona,
    memory: { model: process.env.MEMORY_MODEL },
  });

  const name = card.name[lang] ?? card.id;
  console.log(`\n[${provider.id}:${spec.model ?? provider.defaultModel}] [${session.promptVersion.id}] ${name} ↔ ${session.persona.name}\n`);
  console.log(`${name}: ${session.open().content}\n`);

  const printTurn = (t: Turn) => {
    const u = t.meta?.usage;
    const tokens = u ? `, ${u.promptTokens} in${u.cachedPromptTokens ? ` (${u.cachedPromptTokens} cached)` : ""} / ${u.completionTokens} out` : "";
    stdout.write(`\n   (turn ${t.index}, ${t.meta?.latencyMs} ms${tokens})\n\n`);
  };

  const streamReply = async (gen: AsyncGenerator<string, Turn>) => {
    stdout.write(`${name}: `);
    let r = await gen.next();
    while (!r.done) {
      stdout.write(r.value);
      r = await gen.next();
    }
    printTurn(r.value);
  };

  const rl = createInterface({ input: stdin, output: stdout });
  for (;;) {
    const line = (await rl.question(`${session.persona.name}: `)).trim();
    if (!line) continue;
    if (line === "/quit") break;
    if (line === "/memory") {
      console.log(JSON.stringify(session.memory, null, 2));
      continue;
    }
    if (line === "/prompt") {
      console.log(session.buildMessages()[0]?.content);
      continue;
    }
    if (line === "/more") {
      const t = await session.continue();
      console.log(`${name}: ${t.content}`);
      printTurn(t);
      continue;
    }
    if (line === "/regen") {
      const t = await session.regenerate();
      console.log(`${name}: ${t.content}`);
      printTurn(t);
      continue;
    }
    await streamReply(session.sendStream(line));
  }
  rl.close();
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
