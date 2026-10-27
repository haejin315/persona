import { firstSpeechComplete, firstSpeechLine, stripRepeatedOpener } from "./filters.js";
import { DEFAULT_MEMORY_CONFIG, MemoryManager, type MemoryConfig } from "./memory/manager.js";
import { getPromptVersion, type PromptVersion } from "./prompt/builder.js";
import type { GenerateOptions, LLMProvider } from "./providers/types.js";
import {
  DEFAULT_PERSONA,
  emptyMemory,
  fillUser,
  normalizeMemory,
  type CharacterCard,
  type ChatMessage,
  type Lang,
  type MemoryState,
  type TokenUsage,
  type Turn,
  type UserPersona,
} from "./types.js";
import type { WorldBlock } from "./world/types.js";

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
  /** Passed to the provider on every reply; "off" answers without extended thinking where the model allows (docs/11). */
  thinking?: "off" | "adaptive";
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
  /**
   * The present as the shared world has it (day, slot, place, days since the user was last seen).
   * Unset in plain 1:1 sessions, in which case the prompt is exactly what it was before the world
   * existed (docs/10 step 2). Set by world/sync.ts.
   */
  world?: WorldBlock;
  /** Memory-consolidation usage summed over the session (separate from reply usage on turns). */
  memoryUsage: TokenUsage & { calls: number; model?: string } = { promptTokens: 0, completionTokens: 0, cachedPromptTokens: 0, calls: 0 };

  private provider: LLMProvider;
  private memoryManager: MemoryManager;
  private model?: string;
  private temperature?: number;
  private thinking?: "off" | "adaptive";
  /** Memory consolidation in flight (started after a reply, awaited before the next one). */
  private pending?: Promise<void>;
  private pendingError?: unknown;
  private filters: { repeatedOpener?: boolean };
  private pendingBoundary?: string;

  constructor(opts: SessionOptions) {
    this.card = opts.card;
    this.lang = opts.lang;
    this.provider = opts.provider;
    this.promptVersion = getPromptVersion(opts.promptVersion);
    this.model = opts.model;
    this.temperature = opts.temperature;
    this.thinking = opts.thinking;
    this.persona = opts.persona ?? DEFAULT_PERSONA;
    this.filters = { repeatedOpener: true, ...(opts.filters ?? {}) };
    this.memoryManager = new MemoryManager(opts.memoryProvider ?? opts.provider, {
      ...DEFAULT_MEMORY_CONFIG,
      ...opts.memory,
    });
  }

  /** Mark that time passed before the next user message (world/sync.ts); attached to that turn. */
  markBoundary(label: string): void {
    this.pendingBoundary = label;
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
      ...(this.world ? { world: this.world } : {}),
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
    this.memory = normalizeMemory(structuredClone(snap.memory));
    this.persona = { ...snap.persona };
  }

  private async generate(opts: GenerateOptions): Promise<Turn> {
    await this.settle();
    const res = await this.provider.generate(this.buildMessages(), {
      model: this.model,
      temperature: this.temperature,
      thinking: this.thinking,
      ...opts,
    });
    const filtered = this.applyFilters(res.text.trim());
    const turn = this.push("character", filtered.text, {
      model: res.model,
      promptVersion: this.promptVersion.id,
      latencyMs: res.latencyMs,
      usage: res.usage,
      ...(filtered.applied.length ? { filters: filtered.applied } : {}),
    });
    this.scheduleConsolidation();
    return turn;
  }

  /**
   * Memory consolidation runs after a reply, off the user's path, and is awaited before the next
   * reply (docs/11: it used to run before the reply and cost a whole extra call on those turns).
   * The next reply sees the same memory either way; only the waiting moved.
   */
  private scheduleConsolidation(): void {
    if (this.pending || !this.memoryManager.needsConsolidation(this.turns, this.memory)) return;
    this.pending = this.maybeConsolidate()
      .catch((e) => {
        this.pendingError = e;
      })
      .finally(() => {
        this.pending = undefined;
      });
  }

  /** Wait for any consolidation in flight; rethrows its error once. */
  async settle(): Promise<void> {
    if (this.pending) await this.pending;
    if (this.pendingError) {
      const e = this.pendingError;
      this.pendingError = undefined;
      throw e;
    }
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
    await this.settle();
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
      thinking: this.thinking,
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
        continue;
      }
      // The filter only ever edits the first spoken line; complete lines ahead of it (the opening
      // *action*) go through at once instead of waiting for the speech to finish (docs/11: that
      // wait was ~2.5 s of the time to the first visible character).
      const lines = held.split("\n");
      const first = firstSpeechLine(held);
      const ahead = first ? first.index : lines.length - 1;
      if (ahead > 0) {
        const prefix = `${lines.slice(0, ahead).join("\n")}\n`;
        held = lines.slice(ahead).join("\n");
        out += prefix;
        yield prefix;
      }
    }
    if (held !== undefined) {
      const prefix = release(held);
      out += prefix;
      yield prefix;
    }
    const turn = this.push("character", out.trim(), {
      model: done?.model ?? this.model ?? this.provider.defaultModel,
      promptVersion: this.promptVersion.id,
      latencyMs: done?.latencyMs,
      ...(done?.firstTokenMs !== undefined ? { firstTokenMs: done.firstTokenMs } : {}),
      usage: done?.usage,
      ...(applied.length ? { filters: applied } : {}),
    });
    this.scheduleConsolidation();
    return turn;
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
    if (role === "user" && this.pendingBoundary) {
      meta = { ...(meta ?? {}), boundary: this.pendingBoundary };
      this.pendingBoundary = undefined;
    }
    const turn: Turn = { role, content, index: this.turns.length, ts: Date.now(), meta };
    this.turns.push(turn);
    return turn;
  }
}
