import { firstSpeechComplete, stripRepeatedOpener } from "./filters.js";
import { DEFAULT_MEMORY_CONFIG, MemoryManager, type MemoryConfig } from "./memory/manager.js";
import { getPromptVersion, type PromptVersion } from "./prompt/builder.js";
import type { GenerateOptions, LLMProvider } from "./providers/types.js";
import {
  DEFAULT_PERSONA,
  emptyMemory,
  fillUser,
  type CharacterCard,
  type ChatMessage,
  type Lang,
  type MemoryState,
  type TokenUsage,
  type Turn,
  type UserPersona,
} from "./types.js";

export interface SessionOptions {
  card: CharacterCard;
  lang: Lang;
  provider: LLMProvider;
  /** Provider used for memory consolidation; defaults to the chat provider. */
  memoryProvider?: LLMProvider;
  promptVersion?: string;
  memory?: Partial<MemoryConfig>;
  model?: string;
  temperature?: number;
  persona?: UserPersona;
  /**
   * Engine-side reply filters (filters.ts). The repeated-opener filter is on by default since
   * 2026-10-22 (it only acts on characters whose card lists `openers`); an eval arm can turn it off.
   */
  filters?: { repeatedOpener?: boolean };
}

export interface SessionSnapshot {
  characterId: string;
  lang: Lang;
  promptVersion: string;
  model?: string;
  persona: UserPersona;
  turns: Turn[];
  memory: MemoryState;
}

/**
 * One conversation between a user (playing a persona) and a character. Owns the transcript,
 * the long-term memory, and the prompt assembly. The server and the eval harness both drive
 * this class, so whatever the eval measures is exactly what users get.
 */
export class ChatSession {
  readonly card: CharacterCard;
  readonly lang: Lang;
  readonly promptVersion: PromptVersion;
  persona: UserPersona;
  turns: Turn[] = [];
  memory: MemoryState = emptyMemory();
  /** Memory-consolidation usage summed over the session (separate from reply usage on turns). */
  memoryUsage: TokenUsage & { calls: number; model?: string } = { promptTokens: 0, completionTokens: 0, cachedPromptTokens: 0, calls: 0 };

  private provider: LLMProvider;
  private memoryManager: MemoryManager;
  private model?: string;
  private temperature?: number;
  private filters: { repeatedOpener?: boolean };

  constructor(opts: SessionOptions) {
    this.card = opts.card;
    this.lang = opts.lang;
    this.provider = opts.provider;
    this.promptVersion = getPromptVersion(opts.promptVersion);
    this.model = opts.model;
    this.temperature = opts.temperature;
    this.persona = opts.persona ?? DEFAULT_PERSONA;
    this.filters = { repeatedOpener: true, ...(opts.filters ?? {}) };
    this.memoryManager = new MemoryManager(opts.memoryProvider ?? opts.provider, {
      ...DEFAULT_MEMORY_CONFIG,
      ...opts.memory,
    });
  }

  /** The character's scripted first message, pushed as a turn. */
  open(): Turn {
    const raw =
      this.card.scenario.firstMessage[this.lang] ?? this.card.scenario.firstMessage.ko ?? "";
    return this.push("character", fillUser(raw, this.persona), {
      promptVersion: this.promptVersion.id,
    });
  }

  /** Build the exact messages that would be sent right now (useful for debugging and eval). */
  buildMessages(): ChatMessage[] {
    return this.promptVersion.build({
      card: this.card,
      lang: this.lang,
      memory: this.memory,
      recent: this.memoryManager.recentWindow(this.turns),
      persona: this.persona,
    });
  }

  /** Non-streaming reply to a new user message. */
  async send(userMessage: string, opts: GenerateOptions = {}): Promise<Turn> {
    this.push("user", userMessage);
    return this.generate(opts);
  }

  /** Streaming reply to a new user message: yields deltas, then returns the recorded turn. */
  async *sendStream(userMessage: string, opts: GenerateOptions = {}): AsyncGenerator<string, Turn> {
    this.push("user", userMessage);
    return yield* this.generateStream(opts);
  }

  /**
   * One more character turn without new user input ("이어서 받기"). The character keeps the
   * scene moving on its own.
   */
  async continue(opts: GenerateOptions = {}): Promise<Turn> {
    return this.generate(opts);
  }

  /** Drop the last character turn so it can be regenerated. Returns the removed turn. */
  undoLastReply(): Turn | undefined {
    const last = this.turns.at(-1);
    if (last?.role !== "character") return undefined;
    return this.turns.pop();
  }

  /** Regenerate the last character reply (the "재생성" button). */
  async regenerate(opts: GenerateOptions = {}): Promise<Turn> {
    this.undoLastReply();
    return this.generate(opts);
  }

  snapshot(): SessionSnapshot {
    return {
      characterId: this.card.id,
      lang: this.lang,
      promptVersion: this.promptVersion.id,
      model: this.model,
      persona: { ...this.persona },
      turns: [...this.turns],
      memory: structuredClone(this.memory),
    };
  }

  restore(snap: Pick<SessionSnapshot, "turns" | "memory" | "persona">): void {
    this.turns = [...snap.turns];
    this.memory = structuredClone(snap.memory);
    this.persona = { ...snap.persona };
  }

  private async generate(opts: GenerateOptions): Promise<Turn> {
    await this.maybeConsolidate();
    const res = await this.provider.generate(this.buildMessages(), {
      model: this.model,
      temperature: this.temperature,
      ...opts,
    });
    const filtered = this.applyFilters(res.text.trim());
    return this.push("character", filtered.text, {
      model: res.model,
      promptVersion: this.promptVersion.id,
      latencyMs: res.latencyMs,
      usage: res.usage,
      ...(filtered.applied.length ? { filters: filtered.applied } : {}),
    });
  }

  /** Card openers for the session language (used by the repeated-opener filter). */
  private openers(): string[] {
    return this.filters.repeatedOpener ? (this.card.style[this.lang]?.openers ?? []) : [];
  }

  /** Run the enabled reply filters against the previous character reply. */
  private applyFilters(text: string): { text: string; applied: string[] } {
    const applied: string[] = [];
    const openers = this.openers();
    if (openers.length) {
      const prev = [...this.turns].reverse().find((t) => t.role === "character")?.content;
      const r = stripRepeatedOpener(text, prev, openers);
      if (r.stripped) applied.push("repeated-opener");
      text = r.text;
    }
    return { text, applied };
  }

  private async *generateStream(opts: GenerateOptions): AsyncGenerator<string, Turn> {
    await this.maybeConsolidate();
    let done: Parameters<NonNullable<GenerateOptions["onDone"]>>[0] | undefined;
    // With a filter on, the prefix up to the end of the first spoken line is held back so the
    // filter can edit it before the user sees it; the rest streams through untouched.
    let held: string | undefined = this.openers().length ? "" : undefined;
    let out = "";
    const applied: string[] = [];
    const release = (prefix: string): string => {
      const r = this.applyFilters(prefix);
      applied.push(...r.applied);
      return r.text;
    };
    for await (const delta of this.provider.stream(this.buildMessages(), {
      model: this.model,
      temperature: this.temperature,
      ...opts,
      onDone: (r) => {
        done = r;
        opts.onDone?.(r);
      },
    })) {
      if (held === undefined) {
        out += delta;
        yield delta;
        continue;
      }
      held += delta;
      if (firstSpeechComplete(held) || held.length > 400) {
        const prefix = release(held);
        held = undefined;
        out += prefix;
        yield prefix;
      }
    }
    if (held !== undefined) {
      const prefix = release(held);
      out += prefix;
      yield prefix;
    }
    return this.push("character", out.trim(), {
      model: done?.model ?? this.model ?? this.provider.defaultModel,
      promptVersion: this.promptVersion.id,
      latencyMs: done?.latencyMs,
      usage: done?.usage,
      ...(applied.length ? { filters: applied } : {}),
    });
  }

  private async maybeConsolidate(): Promise<void> {
    if (!this.memoryManager.needsConsolidation(this.turns, this.memory)) return;
    this.memory = await this.memoryManager.consolidate(this.card, this.lang, this.turns, this.memory);
    const u = this.memoryManager.lastUsage;
    if (u) {
      this.memoryUsage.promptTokens += u.promptTokens;
      this.memoryUsage.completionTokens += u.completionTokens;
      this.memoryUsage.cachedPromptTokens! += u.cachedPromptTokens ?? 0;
      this.memoryUsage.calls++;
      this.memoryUsage.model = u.model;
    }
  }

  private push(role: Turn["role"], content: string, meta?: Turn["meta"]): Turn {
    const turn: Turn = { role, content, index: this.turns.length, ts: Date.now(), meta };
    this.turns.push(turn);
    return turn;
  }
}
