# persona

AI 캐릭터 채팅 엔진 + 평가 파이프라인. "캐릭터가 천 번째 턴에서도 살아있는가"를 감이 아니라 숫자로 확인하는 것이 목표.

캐릭터 채팅 서비스가 실제로 겪는 문제 여섯 개를 골라, 각각을 **재현할 수 있는 지표로 바꾸고 → 가설을 세우고 → 같은 유저 대본으로 arm을 비교하고 → 결과를 프롬프트/엔진에 적용**했다. 모든 실험은 이 저장소의 명령 한 줄로 다시 돌릴 수 있다.

## 문제 6개와 재현 결과

| # | 문제 | 어떻게 재현했나 | 결과 (sonnet, opus 채점) | 문서 |
|---|---|---|---|---|
| 1 | 긴 대화에서 설정이 무너진다 | 유저 50턴 대화, 10답장마다 일관성·기억·말투 채점, 20~40턴 전 사실을 되묻는 프로브 5개 | 평면 프롬프트(v1)는 기억 점수 5.0→**2.5**, 회상 45%. 56턴에 고양이 "보리"를 "두부"로 바꿔 기억. 구조화(v2)는 4.5, 회상 83%. v3로 200턴(400턴)을 돌리면 일관성 4.7, 프롬프트 토큰은 1만 1천에서 평탄. 같은 200턴을 v5·v9로 재실행: 기억 요약의 화자 뒤바뀜은 사라졌고 v9에서 회상 7/7. 200턴에서 채점기의 유저 발화 상한이 잘려 v3 기억 점수가 0.45 부풀려졌던 결함도 찾아 고침 | [02](docs/02-long-conversation-drift.md) |
| 2 | "나를 기억하네"가 안 된다 | 같은 유저 대본으로 기억 방식 3종: 최근 20턴만 / 전체 트랜스크립트 / 20턴 + 요약 기억 | 창만: 회상 63%, 52턴에 이름을 잊음. 전체: 83%지만 토큰 1.5~1.8배, 없는 과거를 인용. 요약 기억: 83%, 기억 점수 최고(4.7). 200턴에서 드러난 실패: 요약이 유저의 사실을 캐릭터의 사실로 바꿔 저장(화자 뒤바뀜) → 귀속 규칙으로 저장은 고쳤지만 유저가 그 사실을 처음처럼 다시 꺼내면 캐릭터가 유저 쪽을 따라감 → "MEMORY를 믿는" 규칙(v9)으로 400턴 프로브 통과 | [03](docs/03-remembers-me.md) |
| 3 | 유저가 재생성을 누른다 | 평가 대화를 로그로 간주, 답장마다 재생성 사유를 규칙 + LLM으로 라벨 → 빈도표 → 1위 사유를 겨냥한 프롬프트 수정 → 리플레이 전후 비교 | 1위는 **반복**(100답장당 v2 72~76건): few-shot 예시의 문장 뼈대를 매 턴 복제. VARIETY 블록을 넣은 v3로 리플레이하자 유발률 83%→**42%**, verbatim 반복 59회→0회, 말투 3.6→4.3, 회상·비용 동일, 기억 점수 −0.5(다음 라운드 목표). v4(에코 오프너·고정 마무리 금지)로 41%→**33%**, v5(아는 사실 하나씩 꺼내 쓰기)로 기억 점수를 v2 수준(4.6)으로 회복. v6(문장 끝 경어 수준 고정)로 존댓말 캐릭터의 반말 어미 3.9%→**0.2%**, 질문 끝 8→0%. v8(카드 습관 복원)로 하루의 "...아,"/"음," 오프너 2%→47%, 말투 4.35→4.55. v9(MEMORY를 믿는 규칙)로 200턴 회상 6/7→**7/7**, 일관성 +0.2 → 현재 기본값. 일곱 라운드 합계(arm당 2~4대화): 유발률 83→36%, 말투 3.6→4.65, 회상·비용 동일. 한 라운드(v7)는 두 캐릭터에서 반대로 움직여 폐기 | [04](docs/04-regenerate-mining.md) |
| 4 | 안전과 몰입이 충돌한다 | 시나리오 구간을 fiction / real-risk로 표시, 답장을 ok / 과잉 거부 / 과소 반응 / 위반으로 판정. 안전 문구 3종 비교 | 범용 안전 블록: 과잉 거부 **40%**, 맥주 농담에 "저는 AI라서" + 핫라인. 카드 안에 캐릭터로 쓴 경계: **3%**. 실제 위기 신호 대응은 셋 다 100%, 위반 0% | [05](docs/05-safety-vs-immersion.md) |
| 5 | 한국어 캐릭터를 일본어로 옮기면 말투가 깨진다 | 정체성 블록은 공유, `style.ja`만 교체해 50턴 | 하루: 혼입 0/50, 회상 100%, です・ます 유지. 카엘: 「俺」 고정, 정중체 0회. 실패 패턴(지문 길이, 반복)은 한국어와 동일 → 언어가 아니라 프롬프트 문제. 새 발견: 카엘은 유저가 쓴 ASCII 따옴표를 36/50 따라감 → 언어별 형식 규칙 명시 | [06](docs/06-ko-to-ja.md) |
| 6 | 모델을 바꾸면 목소리가 바뀐다 | 같은 v2 프롬프트, 같은 유저 턴으로 sonnet vs haiku | 회상 83% 동일(기억 블록이 흡수). 일관성 4.6→**3.7**: 사실은 기억하되 시간·공간·누가 했는지의 논리가 꼬임. 유저 말 무시 0→10/100답장. 말투는 둘 다 3.5(프롬프트 탓). 정가 1/12 | [07](docs/07-model-portability.md) |

전제가 되는 비용 분석은 [01 · 유닛 이코노믹스](docs/01-unit-economics.md), 실행 경로의 프롬프트 캐시가 어디서 깨지는지는 [08 · 프롬프트 캐시](docs/08-prompt-cache.md). 채점기가 400턴 내내 같은 지적을 반복한 원인이 캐릭터 카드의 자기모순이었던 건과, 그걸 LLM 없이 잡는 `npm run lint-cards`는 [09 · 카드 린트](docs/09-card-lint.md). 요약 기억이 어떤 사실을 떨어뜨리는지는 녹음 위에서 요약기만 다시 돌리는 `npm run memory-replay`로 1시간짜리 재실행 없이 본다(02, 03). 측정 자체의 신뢰도(채점기 노이즈 ±0.4, 시뮬레이터가 말하지 않은 사실의 프로브 제외, 규칙 보정)는 각 문서의 "한계" 절에 적었다.

## 구조

```
characters/        캐릭터 카드 (YAML). 정체성은 언어 중립, 말투는 언어별 블록(ko/ja/en)으로 분리
packages/engine/   채팅 엔진: 프로바이더 추상화, 버전별 프롬프트 빌더(v1/v2/v3, 경계 변형), 장기 기억, 세션
packages/eval/     유저 시뮬레이터 → 대화 생성 → 채점(LLM + 규칙) → 리포트 / 재생성 사유 분류 / 안전 감사
  scenarios/       유저 대본: 비트, 흘릴 사실, 회상 프로브, fiction/real-risk 구간
  experiments/     실험 설정 (arm, 모델, 턴 수, 리플레이 소스)
docs/              실험별 1 Pager
scripts/           재현 가능한 계산 스크립트 (`npm run econ`)
apps/server, web   TypeScript API 서버(SSE) + React 데모                        (예정)
```

### 엔진 설계 원칙

- **정체성과 상태의 분리.** 카드의 `identity`/`world`/`hardConstraints`는 절대 바뀌지 않는 블록, `memory`는 매 턴 갱신되는 블록. 섞어 두면 긴 대화에서 설정이 흐려진다는 가설은 실험 1에서 확인됐다.
- **말투는 언어별 모듈.** `style.ko`, `style.ja`에 존댓말/경어 수준, 호칭, 1인칭, 말버릇, 금지 표현, 예시 대사를 따로 둔다. 언어가 늘 때 정체성은 건드리지 않는다(실험 5).
- **안전은 캐릭터 안에서.** `boundaries`는 "캐릭터를 유지한 채 걱정하고 도움을 권한다"처럼 그 캐릭터의 행동으로 쓴다. 범용 정책 블록과의 차이는 실험 4.
- **프롬프트는 버전이다.** `v1-flat`(평면), `v2-structured`(섹션 + 메모리 + few-shot), `v3-structured`(+ VARIETY: 반복 금지, 지문 길이), `v4-structured`(+ 에코 오프너·고정 마무리 금지), `v5-structured`(+ 아는 사실 하나씩 꺼내 쓰기), `v6-structured`(+ REGISTER: 문장 끝 경어 수준 고정, 질문 1개 이하, 감정은 행동으로), `v7`/`v8-structured`(+ VARIETY가 지워 버린 카드 습관 복원), `v9-structured`(+ MEMORY를 믿는 규칙: 유저가 이미 말한 사실을 처음처럼 꺼내도 기억한다고 답하고, 안 한 말은 귀속하지 않는다; 현재 기본값), `v10-structured`(+ 현재 시점은 유저가 권위: 요일·지금 하는 일·기분은 기억보다 유저 말이 맞다; 200턴에서 v9가 유저의 요일을 네 번 고친 사건에서 나왔고 검증 중), `v3-cached`(상태 블록을 컨텍스트 메시지로 분리, 실험 8), 경계 변형 `v2-no-boundaries` / `v2-strict-safety`. 평가 하네스가 버전 간 차이를 측정한다.
- **프로바이더 중립.** Claude(Claude Code CLI 경유), OpenAI, Ollama가 같은 인터페이스 뒤에 있다. 모델은 `provider:model` 스펙 문자열 하나로 바꾼다. 엔진, 서버, 평가가 모두 같은 `ChatSession`을 쓰므로 평가가 측정하는 것이 곧 유저가 받는 것이다.
- **장기 기억.** 최근 N턴은 그대로, 그 밖은 요약 + 사실 목록(≤25) + 관계 단계로 접어서 프롬프트에 넣는다. 창을 늘리는 것보다 싸고 정확하다(실험 2).

## 캐릭터

| id | 장르 | 한 줄 |
|----|------|------|
| `haru` | 로맨스/일상 | 동네 카페 바리스타. 낯가림, 관찰력, 존댓말에서 천천히 풀리는 거리감 |
| `kael` | 판타지 모험 | 하프엘프 용병. 반말, 비용과 확률로 말함, 숨기는 과거 |

## 시작하기

```bash
npm install
cp .env.example .env     # 기본값은 Claude Code CLI(구독) 경로. `claude`가 PATH에 있고 로그인돼 있으면 키 없이 동작
npm test

# 터미널 채팅
npm run chat -- --character haru --lang ko                              # claude-cli:sonnet, v9-structured
npm run chat -- --character kael --lang ja --model claude-cli:haiku --prompt v2-structured
npm run chat -- --character haru --lang ko --model openai:gpt-5-mini    # OPENAI_API_KEY 설정 시
npm run chat -- --character haru --lang ko --model ollama:qwen2.5:7b    # Ollama 설치 시
```

기본 LLM 경로는 Claude Code CLI(`claude -p`)다. API 키 대신 Claude 구독으로 돌리기 위한 선택이며, 프로세스 1개/호출(약 4~5초), 온도 조절 불가, 구독 사용량 윈도우 적용이라는 제약이 있다. 그래서 프로바이더 추상화가 있고, 스펙 문자열만 바꾸면 OpenAI나 로컬 모델로 같은 실험을 다시 돌릴 수 있다.

입력은 앱들과 같은 형식으로: `*행동과 지문*` `"대사"`. 채팅 중 명령: `/more` (캐릭터가 이어서), `/regen` (재생성), `/prompt`, `/memory`, `/quit`

## 평가

```bash
npm run eval -- all packages/eval/experiments/smoke.json --judge fake      # 오프라인 파이프라인 점검 (가짜 프로바이더)
npm run eval -- estimate packages/eval/experiments/prompt-version.json     # 실행 전 호출 수·토큰·비용

npm run eval -- all packages/eval/experiments/prompt-version.json          # 1  v1 vs v2 (sonnet, 50턴)
npm run eval -- all packages/eval/experiments/memory.json                  # 2  기억 3종 (prompt-version의 유저 턴 리플레이)
npm run eval -- all packages/eval/experiments/regen-fix.json               # 3  v2 vs v3 (리플레이)
npm run eval -- all packages/eval/experiments/safety.json                  # 4  경계 문구 3종 (self-replay)
npm run eval -- all packages/eval/experiments/haru-ja.json                 # 5  일본어 50턴
npm run eval -- all packages/eval/experiments/model-r1.json                # 6  sonnet vs haiku (리플레이)

npm run eval -- regen  eval-results/prompt-version                         # 재생성 사유 분류만 다시 (--rules-only: 규칙만 재계산)
npm run eval -- safety eval-results/safety                                 # 안전 감사만 다시
npm run eval -- judge  eval-results/model --format-only                    # 규칙 점수만 재계산, LLM 점수 재사용
```

- **시나리오** (`packages/eval/scenarios/*.yaml`): 유저 페르소나, 비트별 목표, 비트마다 자연스럽게 흘릴 사실(정답지), 특정 턴에 던지는 회상 질문(프로브), 안전 감사용 fiction/real-risk 구간.
- **시뮬레이터**: LLM이 유저를 연기한다. `replayFrom`으로 이전 실행의 유저 발화를 그대로 재사용하면 모든 arm이 같은 입력을 받는다. `replayArmId`가 arm 중 하나면 그 arm을 먼저 시뮬레이터로 돌리고 나머지가 리플레이한다.
- **채점**: 10답장마다 체크포인트. LLM 채점 3개(설정 일관성, 기억, 말투 준수, 1~5점) + 규칙 기반 포맷 검사 8개. 프로브는 정답과 대조해 회상률을 내되, 시뮬레이터가 그 사실을 실제로 말한 프로브만 분모에 넣는다.
- **재생성 사유 분류** (`regen`): 답장마다 규칙(형식, 길이, 유저 대신 서술, 언어 혼입, 문장 2개 이상 반복) + LLM(유저 말 무시, 캐릭터 이탈, 밋밋함, 구조 반복) 라벨. arm별 100답장당 빈도표가 "무엇을 먼저 고칠지"를 정한다.
- **안전 감사** (`safety`): 시나리오가 표시한 구간의 답장을 ok / over_refusal / under_reaction / violation으로 판정. 과잉 거부와 과소 반응을 따로 센다.
- **리포트**: arm별 평균과 표준편차, 체크포인트별 드리프트 곡선, 답장당 토큰(프롬프트·캐시·출력).

채점은 생성 모델보다 상위 모델(opus)로 둔다. haiku 채점기는 1턴에 밝힌 이름을 두고 "듣기 전에 불렀다"고 깎는 오판을 했다. 같은 대화를 opus가 세 번 채점하면 일관성은 ±0.4, 기억·말투는 ±0.2 안에서 흔들린다.

### 관측성 (Langfuse)

`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`가 있으면 모든 LLM 호출이 [Langfuse](https://langfuse.com) 트레이스로 남는다. 없으면 전부 no-op이라 평가와 채팅은 그대로 돈다. 로컬은 공식 `docker-compose.yml`로 self-host(`http://localhost:3000`), 키는 `.env.example` 참고.

| 단위 | 무엇이 남나 |
|---|---|
| 트레이스 | 대화 1개 (`eval:<scenario>/<arm>`, 채팅은 세션). 태그: 시나리오, arm, 프롬프트 버전, 리플레이/시뮬레이션 |
| generation | 호출 1개. role(character · simulator · memory · judge · regen · safety), 유저 턴, 모델, 입력 메시지, 출력, 토큰(입력 · 캐시 · 출력), 지연 |
| score | 채점 4지표 + 회상률, 재생성 사유별 100답장당 빈도, 안전 과잉 거부 · 과소 반응 · 위반율. 전부 그 대화의 트레이스에 붙는다 |

그래서 "일관성 3점 이하"나 "regen_repetition > 50"으로 트레이스를 걸러 그 답장을 만든 프롬프트 전문을 바로 읽을 수 있다. 평가 결과 JSON(`eval-results/`)은 그대로 남고, Langfuse는 그 위의 탐색 도구다.

### 비용 통제

| 장치 | 무엇 | 어디 |
|---|---|---|
| `estimate` | 역할별(캐릭터·시뮬레이터·기억·채점) 호출 수와 토큰, 정가 비용과 실제 청구 비용을 실행 전에 표로 출력 | `eval estimate <config>` |
| `maxUsd` | 청구 비용 누적이 상한을 넘으면 새 호출을 거부하고 진행 중 대화는 `truncated`로 저장 | 실험 설정 |
| `replayFrom` | 이전 실행의 유저 발화를 재사용. 시뮬레이터 비용 0, arm 간 분산 감소 | 실험 설정 |
| `claude-cli:<model>` | Claude Code CLI(`claude -p`)로 호출해 구독 한도에서 차감(청구 $0). 고정 오버헤드 830토큰(캐시됨) | 모든 모델 스펙 |
| 재시작 | `run`은 outDir에 이미 끝난 대화가 있으면 건너뛴다. 사용량 윈도우에 걸려 멈춰도 같은 명령으로 이어간다 (`--fresh`로 처음부터) | `eval run` |

실측: 50턴 대화 4개 생성(sonnet) 18분·432호출, opus 채점 40호출 100초. 가격표는 `packages/eval/src/pricing.ts` 한 곳.

## 로드맵

1. ~~엔진 코어, 캐릭터 2종, CLI~~
2. ~~유닛 이코노믹스 1 Pager → 모델 선택 기준~~
3. ~~평가 파이프라인: 시뮬레이터, 채점, 리포트, 비용 통제~~
4. ~~문제 6개 재현 실험 + 1 Pager (docs/02~07)~~
5. ~~LLM 관측성: Langfuse 트레이스 + 점수~~
6. 웹 데모 (TS 서버 + React): 현재 프롬프트, 기억 상태, 재생성 사유 판정을 보여주는 디버그 패널
7. 반복 실험: v4 프롬프트 전후, 200턴 장기 실행, 기억 사실 상한의 교체 정책, 카엘 일본어, Ollama 로컬 모델 arm
8. docker-compose, GitHub Actions(typecheck · vitest · 오프라인 스모크), 배포
