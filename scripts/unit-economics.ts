/**
 * Unit economics: what does one character reply cost on each LLM, and how does that compare
 * with what users actually pay per message on a credit-based AI character chat service?
 *
 *   npx tsx scripts/unit-economics.ts            # markdown tables to stdout
 *   npx tsx scripts/unit-economics.ts --json     # raw numbers
 *
 * Every number below is an assumption or a public price. The token counts are estimates from the
 * v2-structured prompt and will be replaced by measured `usage` from eval runs (see docs/01-unit-economics.md).
 */

// ---------- Market inputs (public, Oct 2026; sources in docs/01-unit-economics.md) ----------

const FX_KRW_PER_USD = 1500; // ~1,500원/달러 (2026-07 Fed H.10: 1,501원). Parameter.

/** WHIF credit ("위프") packages, web price. 1,900원→1,520 ... 99,900원→91,908 (namu wiki, 2026-06-08). */
const WIF_PACKS = [
  { krw: 1_900, wif: 1_520 },
  { krw: 4_900, wif: 3_920 },
  { krw: 9_900, wif: 8_316 },
  { krw: 19_900, wif: 17_034 },
  { krw: 29_900, wif: 25_953 },
  { krw: 49_900, wif: 43_912 },
  { krw: 99_900, wif: 91_908 },
];
/** Typical buyer: the 9,900원 pack. 1.19원/위프. Range 1.09 (biggest) – 1.25 (smallest). */
const KRW_PER_WIF = WIF_PACKS[2]!.krw / WIF_PACKS[2]!.wif;

/** WHIF model tiers: 위프 deducted per generated reply (or reroll). */
const WIF_TIERS = [
  { tier: "프리즘 Prism", wif: 24 },
  { tier: "아쿠아 Aqua", wif: 54 },
  { tier: "아이리스/루비 Iris/Ruby", wif: 64 },
  { tier: "이클립스/벨벳 Eclipse/Velvet", wif: 80 },
  { tier: "시엘 Ciel", wif: 120 },
];

/** Blended payment take: web PG ~3%, app stores 15–30%. WHIF steers users to web. */
const PAYMENT_FEE = 0.15;

/** Target: LLM COGS ≤ 30% of net revenue per message (≥70% gross margin on inference). */
const TARGET_MARGIN = 0.7;

/** Free allowance: 200 위프/day attendance → ~8 Prism replies/day. */
const FREE_WIF_PER_DAY = 200;

// ---------- Model prices (USD per 1M tokens: input / cached input / output) ----------

interface ModelPrice {
  name: string;
  vendor: "OpenAI" | "Anthropic" | "OSS API" | "Self-host";
  in: number;
  cachedIn: number;
  out: number;
  note?: string;
}

const MODELS: ModelPrice[] = [
  // OpenAI — aggregator-sourced (g2, morphllm, metacto), verify on openai.com/api/pricing
  { name: "gpt-5-nano", vendor: "OpenAI", in: 0.05, cachedIn: 0.005, out: 0.4 },
  { name: "gpt-5-mini", vendor: "OpenAI", in: 0.25, cachedIn: 0.025, out: 2.0 },
  { name: "gpt-4.1-mini", vendor: "OpenAI", in: 0.4, cachedIn: 0.1, out: 1.6 },
  { name: "gpt-5", vendor: "OpenAI", in: 1.25, cachedIn: 0.125, out: 10 },
  { name: "gpt-5.4", vendor: "OpenAI", in: 2.5, cachedIn: 0.25, out: 15, note: "cached price assumed 10%" },
  { name: "gpt-5.5", vendor: "OpenAI", in: 5, cachedIn: 0.5, out: 30, note: "cached price assumed 10%" },
  // Anthropic — first-party list prices
  { name: "claude-haiku-5-5", vendor: "Anthropic", in: 0.1, cachedIn: 0.01, out: 0.5 },
  { name: "claude-sonnet-5-5", vendor: "Anthropic", in: 2.0, cachedIn: 0.2, out: 10 },
  { name: "claude-opus-5-5", vendor: "Anthropic", in: 4.0, cachedIn: 0.2, out: 20 },
  // Open-source via hosted API (DeepInfra list, Sep 2026). No prompt-cache discount assumed.
  { name: "Qwen3-32B (DeepInfra)", vendor: "OSS API", in: 0.08, cachedIn: 0.08, out: 0.28 },
  { name: "Llama-3.3-70B Turbo (DeepInfra)", vendor: "OSS API", in: 0.1, cachedIn: 0.1, out: 0.32 },
  { name: "DeepSeek-V3 (DeepInfra)", vendor: "OSS API", in: 0.32, cachedIn: 0.32, out: 0.89 },
  // Self-hosted fine-tuned ~8B: effective cost at good GPU utilisation. Highly utilisation-dependent.
  { name: "자체 8B 모델 (self-host, 가정)", vendor: "Self-host", in: 0.03, cachedIn: 0.03, out: 0.1, note: "assumes ≥60% GPU utilisation" },
];

// ---------- Per-reply token profile (estimate from v2-structured prompt; replace with measured usage) ----------

const TOKENS = {
  systemPrompt: 2_500, // ~3,100 Korean chars of card → ~2.5K tokens
  memoryBlock: 400, // summary + facts + relationship
  window: 4_000, // 20 recent turns × ~200 tok avg
  output: 350, // Korean reply 300–500 chars
  /** Share of input served from prompt cache in steady state (stable prefix: system + memory + older turns). */
  cacheHitRate: 0.85,
  /** Memory consolidation every 10 turns (= 5 replies): ~4K in / 600 out, amortised per reply. */
  memoryInPerReply: 800,
  memoryOutPerReply: 120,
};
const MEMORY_MODEL = MODELS.find((m) => m.name === "gpt-5-nano")!;

/** Daily reply counts per user segment. */
const SEGMENTS = [
  { name: "무료 (일 200위프 소진)", repliesPerDay: Math.floor(FREE_WIF_PER_DAY / WIF_TIERS[0]!.wif) },
  { name: "라이트 유료", repliesPerDay: 30 },
  { name: "헤비 유료", repliesPerDay: 100 },
  { name: "고래 (리롤 다수)", repliesPerDay: 250 },
];

// ---------- Math ----------

const usd = (tok: number, pricePerM: number) => (tok * pricePerM) / 1e6;

function costPerReplyUSD(m: ModelPrice): number {
  const totalIn = TOKENS.systemPrompt + TOKENS.memoryBlock + TOKENS.window;
  const cached = totalIn * TOKENS.cacheHitRate;
  const fresh = totalIn - cached;
  const chat = usd(fresh, m.in) + usd(cached, m.cachedIn) + usd(TOKENS.output, m.out);
  const memory = usd(TOKENS.memoryInPerReply, MEMORY_MODEL.in) + usd(TOKENS.memoryOutPerReply, MEMORY_MODEL.out);
  return chat + memory;
}

const krw = (x: number) => x * FX_KRW_PER_USD;
const netRevenuePerReplyKRW = (wif: number) => wif * KRW_PER_WIF * (1 - PAYMENT_FEE);
const margin = (revenueKRW: number, costKRW: number) => 1 - costKRW / revenueKRW;

/** Minimum 위프 price per reply for this model to hit the target margin. */
const minWifForTarget = (costKRW: number) =>
  Math.ceil(costKRW / (1 - TARGET_MARGIN) / (1 - PAYMENT_FEE) / KRW_PER_WIF);

// ---------- Output ----------

const fmtUSD = (x: number) => `$${x.toFixed(4)}`;
const fmtKRW = (x: number) => `${x.toFixed(1)}원`;
const fmtPct = (x: number) => `${(x * 100).toFixed(0)}%`;

function main() {
  const rows = MODELS.map((m) => {
    const c = costPerReplyUSD(m);
    return { m, costUSD: c, costKRW: krw(c) };
  }).sort((a, b) => a.costUSD - b.costUSD);

  if (process.argv.includes("--json")) {
    console.log(JSON.stringify({ FX_KRW_PER_USD, KRW_PER_WIF, PAYMENT_FEE, TOKENS, rows }, null, 2));
    return;
  }

  console.log(`### 가정\n`);
  console.log(`- 환율 ${FX_KRW_PER_USD}원/$, 위프 단가 ${KRW_PER_WIF.toFixed(2)}원 (9,900원 팩), 결제 수수료 ${fmtPct(PAYMENT_FEE)}, 목표 추론 마진 ${fmtPct(TARGET_MARGIN)}`);
  console.log(`- 답장 1회 입력 ${TOKENS.systemPrompt + TOKENS.memoryBlock + TOKENS.window} tok (캐시 적중 ${fmtPct(TOKENS.cacheHitRate)}), 출력 ${TOKENS.output} tok, 기억 정리 ${TOKENS.memoryInPerReply}/${TOKENS.memoryOutPerReply} tok (${MEMORY_MODEL.name})\n`);

  console.log(`### 1. 모델별 답장 1회 원가\n`);
  console.log(`| 모델 | 벤더 | 원가/답장 | 원화 | 70% 마진 최소 위프가 |`);
  console.log(`|---|---|---:|---:|---:|`);
  for (const r of rows) {
    console.log(`| ${r.m.name} | ${r.m.vendor} | ${fmtUSD(r.costUSD)} | ${fmtKRW(r.costKRW)} | ${minWifForTarget(r.costKRW)} 위프 |`);
  }

  console.log(`\n### 2. 위프 등급별 순매출 vs 모델 원가 → 추론 마진\n`);
  const header = ["모델", ...WIF_TIERS.map((t) => `${t.tier} (${t.wif})`)];
  console.log(`| ${header.join(" | ")} |`);
  console.log(`|---|${WIF_TIERS.map(() => "---:").join("|")}|`);
  console.log(`| 순매출/답장 | ${WIF_TIERS.map((t) => fmtKRW(netRevenuePerReplyKRW(t.wif))).join(" | ")} |`);
  for (const r of rows) {
    const cells = WIF_TIERS.map((t) => {
      const mg = margin(netRevenuePerReplyKRW(t.wif), r.costKRW);
      const flag = mg >= TARGET_MARGIN ? "" : mg >= 0 ? " ⚠" : " ✖";
      return `${fmtPct(mg)}${flag}`;
    });
    console.log(`| ${r.m.name} | ${cells.join(" | ")} |`);
  }

  console.log(`\n### 3. 세그먼트별 월 추론 원가 (유저 1명, 30일)\n`);
  console.log(`| 모델 | ${SEGMENTS.map((s) => `${s.name} ${s.repliesPerDay}/일`).join(" | ")} |`);
  console.log(`|---|${SEGMENTS.map(() => "---:").join("|")}|`);
  for (const r of rows) {
    const cells = SEGMENTS.map((s) => fmtKRW(r.costKRW * s.repliesPerDay * 30));
    console.log(`| ${r.m.name} | ${cells.join(" | ")} |`);
  }

  console.log(`\n### 4. 구독형(제타패스 10,900원/월 무제한 기본모델) 가정 시 헤비 유저 마진\n`);
  const subNet = 10_900 * (1 - PAYMENT_FEE);
  const heavy = SEGMENTS[2]!;
  console.log(`| 모델 | 월 원가 (${heavy.repliesPerDay}/일) | 마진 |`);
  console.log(`|---|---:|---:|`);
  for (const r of rows) {
    const c = r.costKRW * heavy.repliesPerDay * 30;
    const mg = margin(subNet, c);
    console.log(`| ${r.m.name} | ${fmtKRW(c)} | ${fmtPct(mg)}${mg >= TARGET_MARGIN ? "" : mg >= 0 ? " ⚠" : " ✖"} |`);
  }
}

main();
