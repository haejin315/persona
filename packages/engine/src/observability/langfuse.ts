/**
 * LLM observability via Langfuse (self-hosted or cloud). Every provider call becomes a
 * `generation` observation under a per-conversation trace, carrying role (character / simulator /
 * memory / judge / regen / safety), prompt version, arm, scenario and token usage. Evaluation
 * results are attached to the same trace as Langfuse scores, so a trace can be filtered by
 * "consistency < 3" or "regen reason = repetition" and the exact prompt that produced it read.
 *
 * Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set; otherwise every call here
 * is a no-op, and the engine has no runtime dependency on a running Langfuse.
 */
import { LangfuseClient } from "@langfuse/client";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { propagateAttributes, startObservation, type LangfuseGeneration, type LangfuseSpan } from "@langfuse/tracing";
import { NodeSDK } from "@opentelemetry/sdk-node";
import type { GenerateOptions, GenerateResult, LLMProvider } from "../providers/types.js";
import type { ChatMessage } from "../types.js";

export interface TracingConfig {
  publicKey?: string;
  secretKey?: string;
  baseUrl?: string;
  /** Tag every trace with the environment (e.g. "eval", "chat"). */
  environment?: string;
}

let sdk: NodeSDK | undefined;
let processor: LangfuseSpanProcessor | undefined;
let client: LangfuseClient | undefined;
let enabled = false;

/** Start tracing if credentials are present. Safe to call more than once. */
export function initTracing(cfg: TracingConfig = {}): boolean {
  if (sdk) return enabled;
  const publicKey = cfg.publicKey ?? process.env.LANGFUSE_PUBLIC_KEY;
  const secretKey = cfg.secretKey ?? process.env.LANGFUSE_SECRET_KEY;
  const baseUrl = cfg.baseUrl ?? process.env.LANGFUSE_BASE_URL ?? "http://localhost:3000";
  if (!publicKey || !secretKey) return false;
  processor = new LangfuseSpanProcessor({ publicKey, secretKey, baseUrl, environment: cfg.environment ?? process.env.LANGFUSE_ENVIRONMENT });
  sdk = new NodeSDK({ spanProcessors: [processor] });
  sdk.start();
  client = new LangfuseClient({ publicKey, secretKey, baseUrl });
  enabled = true;
  return true;
}

export const tracingEnabled = () => enabled;

/** Flush pending spans and scores (call before the process exits). */
export async function flushTracing(): Promise<void> {
  if (!enabled) return;
  await processor?.forceFlush();
  await client?.flush();
}

export async function shutdownTracing(): Promise<void> {
  if (!enabled) return;
  await flushTracing();
  await sdk?.shutdown();
  sdk = undefined;
  enabled = false;
}

export interface TraceHandle {
  /** Langfuse trace id, stored on records so scores can be attached later. */
  traceId?: string;
  /** Child generation for one provider call; no-op when tracing is off. */
  generation(name: string, params: { model?: string; input?: unknown; metadata?: Record<string, unknown>; startTime?: Date }): GenerationHandle;
  update(params: { output?: unknown; metadata?: Record<string, unknown> }): void;
  end(endTime?: Date): void;
}

export interface GenerationHandle {
  id?: string;
  end(result?: {
    output?: unknown;
    usage?: { promptTokens: number; completionTokens: number; cachedPromptTokens?: number };
    model?: string;
    latencyMs?: number;
    error?: string;
    /** Historical end time (backfill). */
    endTime?: Date;
  }): void;
}

const NOOP_GEN: GenerationHandle = { end() {} };
const NOOP_TRACE: TraceHandle = { generation: () => NOOP_GEN, update() {}, end() {} };

/**
 * Open a trace (root span) for one conversation or chat session. `metadata` and `tags` apply to
 * the trace; `sessionId` groups traces in the Langfuse UI.
 */
export function startTrace(
  name: string,
  opts: { input?: unknown; metadata?: Record<string, unknown>; tags?: string[]; sessionId?: string; userId?: string; startTime?: Date } = {},
): TraceHandle {
  if (!enabled) return NOOP_TRACE;
  // Trace-level attributes (name, tags, session) are stamped on every observation started inside
  // the callback, so each generation carries them too and can be filtered on its own in Langfuse v4.
  const propagated = { traceName: name, tags: opts.tags, sessionId: opts.sessionId, userId: opts.userId, metadata: opts.metadata as Record<string, string> | undefined };
  const root: LangfuseSpan = propagateAttributes(propagated, () =>
    startObservation(name, { input: opts.input, metadata: opts.metadata }, opts.startTime ? { startTime: opts.startTime } : undefined),
  );
  return {
    traceId: root.traceId,
    generation(genName, params) {
      const attrs = { model: params.model, input: params.input, metadata: params.metadata };
      // A historical start time (backfill) is only accepted by the top-level factory, so attach to the
      // root through its span context in that case.
      const gen: LangfuseGeneration = propagateAttributes(propagated, () =>
        params.startTime
          ? startObservation(genName, attrs, { asType: "generation", startTime: params.startTime, parentSpanContext: root.otelSpan.spanContext() })
          : root.startObservation(genName, attrs, { asType: "generation" }),
      );
      return {
        id: gen.id,
        end(result = {}) {
          gen.update({
            output: result.output,
            model: result.model,
            usageDetails: result.usage
              ? {
                  input: Math.max(0, result.usage.promptTokens - (result.usage.cachedPromptTokens ?? 0)),
                  cached: result.usage.cachedPromptTokens ?? 0,
                  output: result.usage.completionTokens,
                }
              : undefined,
            metadata: result.latencyMs !== undefined ? { latencyMs: result.latencyMs } : undefined,
            level: result.error ? "ERROR" : undefined,
            statusMessage: result.error,
          });
          gen.end(result.endTime);
        },
      };
    },
    update(params) {
      root.update(params);
    },
    end(endTime) {
      root.end(endTime);
    },
  };
}

/**
 * Wrap a provider so every generate()/stream() call is recorded as a generation under `trace`.
 * `role` names the call's job (character, simulator, memory, judge, regen, safety); `meta()` can
 * add per-call metadata such as the current user turn.
 */
export function observedProvider<P extends LLMProvider>(provider: P, trace: TraceHandle, role: string, meta: () => Record<string, unknown> = () => ({})): P {
  if (!enabled || trace === NOOP_TRACE) return provider;
  const wrapped: LLMProvider = {
    id: provider.id,
    defaultModel: provider.defaultModel,
    async generate(messages: ChatMessage[], opts?: GenerateOptions): Promise<GenerateResult> {
      const gen = trace.generation(role, { model: `${provider.id}:${opts?.model ?? provider.defaultModel}`, input: messages, metadata: { role, ...meta() } });
      try {
        const res = await provider.generate(messages, opts);
        gen.end({ output: res.text, usage: res.usage, model: res.model, latencyMs: res.latencyMs });
        return res;
      } catch (e) {
        gen.end({ error: e instanceof Error ? e.message : String(e) });
        throw e;
      }
    },
    async *stream(messages: ChatMessage[], opts?: GenerateOptions) {
      const gen = trace.generation(role, { model: `${provider.id}:${opts?.model ?? provider.defaultModel}`, input: messages, metadata: { role, ...meta() } });
      let full = "";
      try {
        const it = provider.stream(messages, {
          ...opts,
          onDone: (r) => {
            gen.end({ output: full, usage: r.usage, model: r.model, latencyMs: r.latencyMs });
            opts?.onDone?.(r);
          },
        });
        for await (const delta of it) {
          full += delta;
          yield delta;
        }
      } catch (e) {
        gen.end({ error: e instanceof Error ? e.message : String(e) });
        throw e;
      }
    },
  };
  return wrapped as P;
}

/** Attach a numeric or categorical score to a trace (no-op when tracing is off or traceId is missing). */
export function scoreTrace(traceId: string | undefined, name: string, value: number | string, comment?: string, timestamp?: Date): void {
  if (!enabled || !client || !traceId) return;
  if (typeof value === "number" && Number.isNaN(value)) return;
  client.score.create({
    traceId,
    name,
    value: value as never,
    comment,
    dataType: typeof value === "number" ? "NUMERIC" : "CATEGORICAL",
    ...(timestamp ? { timestamp: timestamp.toISOString() } : {}),
  });
}
