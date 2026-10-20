# persona

AI 캐릭터 채팅 엔진 + 평가 파이프라인. "캐릭터가 천 번째 턴에서도 살아있는가"를 감이 아니라 숫자로 확인하는 것이 목표.

## 구조

```
characters/        캐릭터 카드 (YAML). 정체성은 언어 중립, 말투는 언어별 블록(ko/ja/en)으로 분리
packages/engine/   채팅 엔진: 프로바이더 추상화, 버전별 프롬프트 빌더, 장기 기억, 세션
packages/eval/     유저 시뮬레이터 → 100턴 대화 생성 → LLM 채점 + 규칙 기반 포맷 검사 → 리포트
apps/server/       TypeScript API 서버 (SSE 스트리밍)                           (예정)
apps/web/          React 채팅 데모                                              (예정)
docs/              실험 1 Pager (01: 유닛 이코노믹스와 모델 선택)
scripts/           재현 가능한 계산 스크립트 (`npm run econ`)
```

### 엔진 설계 원칙

- **정체성과 상태의 분리.** 카드의 `identity`/`world`/`hardConstraints`는 절대 바뀌지 않는 블록, `memory`는 매 턴 갱신되는 블록. 둘을 섞어 두면 긴 대화에서 설정이 흐려진다는 가설을 검증한다.
- **말투는 언어별 모듈.** `style.ko`, `style.ja`에 존댓말/경어 수준, 호칭, 1인칭, 말버릇, 금지 표현, 예시 대사를 따로 둔다. 언어가 늘 때 정체성은 건드리지 않는다.
- **프롬프트는 버전이다.** `v1-flat`(평면 베이스라인), `v2-structured`(섹션 + 메모리 + few-shot). 평가 하네스가 버전 간 차이를 측정한다.
- **프로바이더 중립.** Claude(Claude Code CLI 경유), OpenAI, Ollama(로컬 오픈소스)가 같은 인터페이스 뒤에 있다. 모델은 `provider:model` 스펙 문자열 하나로 바꾼다. 엔진, 서버, 평가가 모두 같은 `ChatSession`을 쓰므로 평가가 측정하는 것이 곧 유저가 받는 것이다.
- **장기 기억.** 최근 N턴은 그대로, 그 밖은 요약 + 사실 목록 + 관계 단계로 접어서 프롬프트에 넣는다.

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
npm run chat -- --character haru --lang ko                              # claude-cli:sonnet
npm run chat -- --character kael --lang ja --model claude-cli:haiku --prompt v1-flat
npm run chat -- --character haru --lang ko --model openai:gpt-5-mini    # OPENAI_API_KEY 설정 시
npm run chat -- --character haru --lang ko --model ollama:qwen2.5:7b    # Ollama 설치 시
```

기본 LLM 경로는 Claude Code CLI(`claude -p`)다. API 키 대신 Claude 구독으로 돌리기 위한 선택이며, 프로세스 1개/호출(약 1초), 온도 조절 불가, 구독 사용량 윈도우 적용이라는 제약이 있다. 그래서 프로바이더 추상화가 있고, 스펙 문자열만 바꾸면 OpenAI나 로컬 모델로 같은 실험을 다시 돌릴 수 있다.

입력은 앱들과 같은 형식으로: `*행동과 지문*` `"대사"`. 채팅 중 명령: `/more` (캐릭터가 이어서), `/regen` (재생성), `/prompt`, `/memory`, `/quit`

## 평가

```bash
npm run eval -- all packages/eval/experiments/smoke.json --judge fake      # 오프라인 파이프라인 점검 (가짜 프로바이더)
npm run eval -- all packages/eval/experiments/smoke-claude.json            # 실제 모델로 10턴 1개 (구독, 약 1분)
npm run eval -- estimate packages/eval/experiments/prompt-version.json     # 실행 전 호출 수·토큰·비용
npm run eval -- all packages/eval/experiments/prompt-version.json          # v1-flat vs v2-structured (sonnet)
npm run eval -- all packages/eval/experiments/model.json                   # sonnet vs haiku (v2, 유저 턴 리플레이)
```

- **시나리오** (`packages/eval/scenarios/*.yaml`): 유저 페르소나, 비트별 목표, 비트마다 자연스럽게 흘릴 사실(정답지), 특정 턴에 던지는 회상 질문(프로브).
- **시뮬레이터**: LLM이 유저를 연기한다. 같은 대본으로 모든 arm을 돌리므로 프롬프트 버전과 모델만 변수가 된다.
- **채점**: 10답장마다 체크포인트. LLM 채점 3개(설정 일관성, 기억, 말투 준수, 1~5점) + 규칙 기반 포맷 검사 8개(지문/대사 형식, 유저 대신 서술 금지, 언어 혼입, AI 자백, 길이). 프로브는 정답과 대조해 회상률을 낸다.
- **리포트**: arm별 평균과 표준편차, 체크포인트별 드리프트 곡선, 답장당 토큰(프롬프트·캐시·출력). 토큰 수치는 `docs/01-unit-economics.md`의 추정치를 실측으로 바꾸는 데 쓴다.

### 비용 통제

실험은 유료 API나 구독 사용량 위에서 돌기 때문에, 쓰기 전에 숫자를 보고 상한을 거는 장치를 넣었다.

| 장치 | 무엇 | 어디 |
|---|---|---|
| `estimate` | 역할별(캐릭터·시뮬레이터·기억·채점) 호출 수와 토큰, 정가 비용과 실제 청구 비용을 실행 전에 표로 출력. 이전 실행 기록이 있으면 실측 평균으로 대체 | `eval estimate <config>` |
| `maxUsd` | 실행 중 청구 비용 누적이 상한을 넘으면 새 호출을 거부하고 진행 중 대화는 `truncated`로 저장, 남은 대화는 건너뜀 | 실험 설정 |
| `replayFrom` | 이전 실행의 유저 발화를 그대로 재사용. 시뮬레이터 비용 0, 모든 arm이 같은 입력을 받아 분산 감소 | 실험 설정 |
| `claude-cli:<model>` | Claude Code CLI(`claude -p`)로 호출해 구독 한도에서 차감(청구 $0). 고정 오버헤드 830토큰(캐시됨), 호출당 약 1초. 채점은 생성 모델보다 상위(opus)로 두어 자기 채점을 피한다 | 모든 모델 스펙 |
| 재시작 | `run`은 outDir에 이미 끝난 대화가 있으면 건너뛴다. 구독 사용량 윈도우에 걸려 멈춰도 다음 윈도우에 같은 명령으로 이어간다 (`--fresh`로 처음부터) | `eval run` |

가격표는 `packages/eval/src/pricing.ts` 한 곳에 있고, 구독·로컬 경로는 정가를 계산하되 청구액은 0으로 잡는다.

## 로드맵

1. ~~엔진 코어, 캐릭터 2종, CLI~~
1.5. ~~유닛 이코노믹스 1 Pager → 모델 선택 기준 확정~~
2. ~~평가 파이프라인: 시뮬레이터, 채점, 리포트 (오프라인 검증 완료)~~
2.5. ~~비용 통제: 추정기, 예산 상한, 리플레이, 구독 채점 어댑터~~
2.6. OPENAI_API_KEY 연결 후 실험 2개 실행 → 결과를 docs/02에 정리
3. 실험 결과 분석: 장기 대화 드리프트 (v1 vs v2), 모델 교체 전후 비교 (여유 시 haru-ja로 한국어→일본어 이식)
4. 웹 데모 (TS 서버 + React)
5. git 정리, 배포 자동화 (비용 때문에 마지막)
