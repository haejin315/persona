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

  constructor(opts: SessionOptions) {
    this.card = opts.card;
    this.lang = opts.lang;
    this.provider = opts.provider;
    this.promptVersion = getPromptVersion(opts.promptVersion);
    this.model = opts.model;
    this.temperature = opts.temperature;
    this.persona = opts.persona ?? DEFAULT_PERSONA;
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
    return this.push("character", res.text.trim(), {
      model: res.model,
      promptVersion: this.promptVersion.id,
      latencyMs: res.latencyMs,
      usage: res.usage,
    });
  }

  private async *generateStream(opts: GenerateOptions): AsyncGenerator<string, Turn> {
    await this.maybeConsolidate();
    let done: Parameters<NonNullable<GenerateOptions["onDone"]>>[0] | undefined;
    let full = "";
    for await (const delta of this.provider.stream(this.buildMessages(), {
      model: this.model,
      temperature: this.temperature,
      ...opts,
      onDone: (r) => {
        done = r;
        opts.onDone?.(r);
      },
    })) {
      full += delta;
      yield delta;
    }
    return this.push("character", full.trim(), {
      model: done?.model ?? this.model ?? this.provider.defaultModel,
      promptVersion: this.promptVersion.id,
      latencyMs: done?.latencyMs,
      usage: done?.usage,
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
