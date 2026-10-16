import { readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";
import { CharacterCard } from "./types.js";

/** Default location: <repo>/characters/*.yaml */
export const DEFAULT_CHARACTERS_DIR = resolve(
  fileURLToPath(new URL(".", import.meta.url)),
  "../../../characters",
);

export function loadCharacter(path: string): CharacterCard {
  const raw = YAML.parse(readFileSync(path, "utf8"));
  const parsed = CharacterCard.safeParse(raw);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((i) => `  - ${i.path.join(".")}: ${i.message}`)
      .join("\n");
    throw new Error(`Invalid character card ${path}:\n${issues}`);
  }
  return parsed.data;
}

export function loadAllCharacters(dir = DEFAULT_CHARACTERS_DIR): CharacterCard[] {
  return readdirSync(dir)
    .filter((f) => f.endsWith(".yaml") || f.endsWith(".yml"))
    .map((f) => loadCharacter(join(dir, f)));
}

export function findCharacter(id: string, dir = DEFAULT_CHARACTERS_DIR): CharacterCard {
  const card = loadAllCharacters(dir).find((c) => c.id === id);
  if (!card) throw new Error(`Character "${id}" not found in ${dir}`);
  return card;
}
