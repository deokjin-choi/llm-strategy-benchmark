# ESWA 논문 강화 상세 계획서 — workshop_paper(0416) 기준 "현재 → 수정" 가이드

> 사용법: 본 문서는 **집필용 설계도**다. 실제 문장 작성은 외부 LLM(GPT 등)에 맡긴다.
> 구조는 **현재 워크숍 페이퍼(`RnDSeoul/workshop_paper(0416).md`)의 실제 목차·소제목을 그대로** 기준으로 삼고,
> 각 항목마다 [핵심 메시지] → [현재 상태] → [수정 방향]을 명시한다.
> 필요 시 `RnDSeoul/full_paper_rndm.md`(=FP), `references/`, `논문강화.txt`, `실험강화계획.txt`를 참조한다.
>
> - 베이스 원고: `workshop_paper(0416).md` (=WS)
> - 역수입 원고: `full_paper_rndm.md` (=FP)  → 깔끔한 Abstract/RQ/metric 정의/Limitations 차용
> - 산출 파일: `RnDSeoul/eswa_paper.md` (신규)
> - 타깃: **Expert Systems with Applications (ESWA)**
> - MOT 강도: **moderate** (실험 그대로, 이론 옷은 Intro·신규 이론절·Discussion에만 집중)

---

# PART A. 공통 설정 (모든 섹션 집필의 기준점)

## A-1. 한 줄 정체성
> "R&D 전략 의사결정에 LLM을 **알고리즘 대리인(algorithmic agent)** 으로 투입하기 전에, 그 판단이 맥락·브랜드 프레이밍·디코딩 설정에 얼마나 흔들리는지를 진단하는 **scenario-based audit framework**."

모든 섹션은 이 문장과 정합해야 한다. 충돌 시 이 문장이 우선.

## A-2. 3대 핵심 메시지 (논문 전체에서 반복·강화)
1. **Semantics > Numbers** — 의미적 맥락엔 강하게 반응, ±20% 수치 변화엔 거의 무반응.
2. **Brand framing = amplifier, not fixed bias** — 기업명 노출은 고정 편향이 아니라 맥락 방향을 *증폭*하고 *설명 스타일*까지 바꿈.
3. **Aggregate hides local failure** — 모델 평균은 특정 시나리오-모델 셀의 치명적 실패를 가림 → 대표 시나리오 audit 필요.

## A-3. MOT 이론 옷 3종 (moderate 배치)
- **(A) Algorithmic Bounded Rationality**: 인간의 제한된 합리성·프레이밍 편향(Kahneman/Tversky, Thorndike halo)을 LLM이 인간 언어 학습으로 *대물림/증폭*. → 배치: **1. Introduction 도입 + 신규 이론절 + 6. Discussion**.
- **(B) Algorithmic Agency Problem**: agency theory(주인-대리인) 매핑 — 주인=의사결정자, 대리인=LLM, 위임계약=프롬프트. 프레이밍 민감도 = 대리인 문제. → 배치: **신규 이론절 + 6. Discussion**.
- **(C) Algorithmic Auditor**: LLM은 추천자가 아니라 *검증 대상*. 본 프레임워크 = 알고리즘 감사 도구. → 배치: **Abstract 끝 + 6. Discussion + Conclusion**.
> moderate 원칙: 위 용어는 Intro 2~3문단, 신규 이론절 1개, Discussion 2~3문단에만. Methodology·Results 본문은 실험 언어 유지.

## A-4. 전역 STYLE RULES (외부 LLM 의뢰 시 매 섹션 첨부)
```
[STYLE RULES — 반드시 준수]
1. 한 문단에 굵게(**) 강조 최대 1~2개. 숫자·축약어 굵게 도배 금지.
2. hedge 반복 금지: "in this relative display only", "within this panel",
   "under this normalization" 류. caveat은 절당 1회, 괄호/각주로만.
3. em-dash(—) 한 문단 최대 1회.
4. 모든 결과 절은 4단 구조: (1)무엇을 보는가 →(2)그림/표 →(3)관찰 →(4)함의 1~2문장.
5. 용어 통일: conditional strategic agent / decision sensitivity / framing robustness /
   algorithmic agency problem / audit. 동의어 치환 금지.
6. 결과를 가리키기만 하지 말고 본문에서 해석까지 완결(떡밥 회수).
7. 학술적·평탄한 서술체. 과장·마케팅 어조 금지. 수치·표·그림 경로 변경 금지.
```

## A-5. 목차 변경 요약 (현재 WS → ESWA 목표)
| 현재(WS) 목차 | 처리 | 비고 |
|---|---|---|
| Title / Abstract | 재작성 | agency·audit 프레임 |
| 1. Introduction | 보강 | MOT gap + RQ |
| 2.1–2.4 | 유지(경미 정리) | ESWA 강점, 그대로 |
| 2.5 Strategic Archetypes | **이동** | 신규 2.6 하위(2.6.3)로 |
| **(신규) 2.6 Bounded rationality, agency, and strategic archetypes in LLM decision-making** | **신규 삽입** | 이론 골격(3 기둥), 2.1~2.4와 동일하게 내용나열형 제목 |
| 2.6 Research gap | 유지 → **2.7로 번호이동** | FP §2.3과 통합 |
| 3.1–3.5 | 유지(보강) | 전략 인용·서술 정리 |
| 4.1–4.3 | **빈칸 채우기** | 4.2/4.3 본문 신규 |
| 5.1–5.5.4 | **문체 수술 + 유지** | deep-dive 복원이 핵심 |
| 6. Discussion | 재작성 | MOT 옷 |
| **(신규) 7. Limitations** | **신규**(FP §6) | human baseline 등 |
| **(신규) 8. Conclusion** | **신규**(FP §7) | 정체성으로 닫기 |
| References / Appendix A | 확장·유지 | |

## A-6. 권장 집필 순서
1) Title/Abstract → 2) 신규 2.6(Bounded rationality, agency, archetypes) → 3) 1. Introduction → 4) 6. Discussion
→ 5) 3~4 Methodology 빈칸 → 6) 5. Results 문체 수술 → 7) Limitations/Conclusion/References.
(이론 골격·정체성·회수를 먼저 고정 → 나머지가 자동 정렬되어 산만함 재발 방지)

---

# PART B. 목차별 "현재 → 수정" 가이드 (WS 소제목 그대로)

---

## Title : "Context and Framing Sensitivity in LLM-Based Strategic Decision-Making"
- **핵심 메시지**: 제목에서부터 "전략 의사결정 + 민감도" 정체성.
- **현재 상태**: 무난하나 ESWA의 application/audit 색·agency 관점이 없음.
- **수정 방향**: 아래 후보로 교체(또는 부제 추가).
  - A. *Auditing LLMs as Algorithmic Agents for R&D Strategy: A Scenario-Based Framework for Context and Framing Sensitivity*
  - B. *Context and Framing Sensitivity of LLMs in Strategic R&D Decisions: An Algorithmic Agency Perspective*

## Abstract
- **핵심 메시지**: algorithmic agent / 3대 메시지 / audit protocol.
- **현재 상태(WS L3–L6)**: "conditional decision agent + governance" 까지는 있으나, 이론(agency)·audit·temperature·모델 패널 스펙이 약함. FP L9–L13이 더 정제됨.
- **수정 방향**: FP Abstract를 베이스로, ① 첫 문장 "algorithmic agent" 규정 ② bounded rationality+agency로 "algorithmic agency problem" 도입 ③ 방법 스펙(6 scenarios/7 archetypes/5 models/2 temperatures) 유지 ④ 결과 3줄(A-2) ⑤ audit protocol로 마무리("evaluation before deployment"). 200~250단어, Keywords 8개(+algorithmic agency, AI governance).

## 1. Introduction
- **핵심 메시지**: "정확도 평가의 한계 → 전략 판단 민감도 진단 필요" + algorithmic agency 예고 + RQ.
- **현재 상태(WS L8–L13)**: 4문단. 평가 한계(Chang 2024)·conditional decision agent·7전략 언급은 있으나, MOT 이론/리뷰페이퍼 gap·명시적 RQ가 없음. FP L17–L28에 RQ("How stable are LLM strategic judgments...")가 있음.
- **수정 방향**:
  - FP의 RQ 문장 도입.
  - (논문강화 Action3) 혁신경영 리뷰페이퍼의 future-agenda 인용 1문장 추가 → **선행 TODO T1**.
  - algorithmic agency problem 한 문장 예고(이론절과 연결).
  - 톤 통일(STYLE RULES), WS L13의 장황한 부분 압축.
- **인용**: Chang et al. 2024, MOT 리뷰 1~2편, Kahneman/Tversky 예고.

## 2. Background and related work

### 2.1 LLMs as decision support systems
- **핵심 메시지**: LLM이 운영·계획 파이프라인에 투입되나 *고정 옵션 내 선택 분포·민감도*는 안 봄.
- **현재 상태(WS L17–L18)**: 충실함(WANG/DU/XIONG 등 인용). ESWA 강점.
- **수정 방향**: **유지**. 마지막 gap 문장만 1문장으로 압축. 변경 최소.

### 2.2 Reliability, hallucination, and safety in high-stakes decisions
- **핵심 메시지**: 신뢰성 연구는 사실오류·이상탐지·정책준수 수준 → 전략 판단 안정성은 별개.
- **현재 상태(WS L20–L21)**: 충실(KONG/HEO/PRZYSTALSKI/ANTULEY). ESWA 친화(HaluCheck 등 ESWA 게재).
- **수정 방향**: **유지**. 본 연구와의 대비 문장 1개만 또렷이.

### 2.3 Knowledge-grounded decision infrastructures
- **핵심 메시지**: grounding을 강화해도 *고정 옵션 선택의 프레이밍 민감도*는 분석 안 함.
- **현재 상태(WS L23–L24)**: 충실(OJURI/XIONG/SINHA/ALARCON).
- **수정 방향**: **유지**. 변경 최소.

### 2.4 Evaluation, slow thinking, and behavioral profiling of LLMs
- **핵심 메시지**: 평가·추론 연구가 정확도 너머로 갔지만, *통제된 교란 하 전략 선택 분포*는 드묾.
- **현재 상태(WS L26–L29)**: 충실하나 **L29에 중복 문장 존재**("This study complements those efforts..."가 2회 반복). 또 인지편향(halo/anchoring) 언급이 여기 섞여 있음.
- **수정 방향**: ① **중복 문장 1개 삭제**. ② 인지편향(halo/anchoring/Kahneman) 언급은 신규 2.6 이론절로 **이동**. ③ 순수 "evaluation/profiling 선행연구"만 남김.

### 2.5 Strategic Archetypes in R&D and Innovation Management
- **핵심 메시지**: 7전략을 고전 전략·혁신 문헌에 정박.
- **현재 상태(WS L31–L34)**: 7전략 이론 근거(Schilling/Porter/Barney/Chesbrough/Miles) 잘 정리됨.
- **수정 방향**: **신규 2.6의 하위 소절(2.6.3)로 이동**. 내용은 거의 유지. (이론 배경을 한 곳에 모으기 위함)

### (신규) 2.6 Bounded rationality, agency, and strategic archetypes in LLM decision-making  ★신규 삽입★
- **제목 원칙**: 2.1~2.4처럼 "내용 나열형"(군더더기 'background/foundations' 단어 금지). 제목만 봐도 세 소절 내용이 보이게.
- **제목 후보**: (추천) 위 제목 / 대안1 *Cognitive bias, agency, and strategic archetypes for LLM decision agents* / 대안2 *Behavioral biases and agency in LLM strategic decision-making* (이 경우 archetypes는 별도 유지).
- **핵심 메시지**: MOT 이론 골격(3 기둥). "So what?" 방어 핵심.
- **현재 상태**: 없음.
- **수정 방향**: 3개 소절 신규 작성.
  - **2.6.1 From human to algorithmic bounded rationality**: 인간 경영진의 bounded rationality·framing bias → LLM이 인간 언어 학습으로 inherit/amplify(이론A). 인용: Kahneman 2011, Tversky&Kahneman 1974, Thorndike 1920 + `Cognitive Bias in Decision-Making with LLMs.pdf`, `cheung-et-al-2025.pdf`, `Do LLMs Encode Frame Semantics.pdf`.
  - **2.6.2 An agency-theoretic view of LLM delegation**: agency theory 요약 + 주인/대리인/프롬프트 매핑 → algorithmic agency problem(이론B). 인용: Jensen&Meckling 1976, Eisenhardt 1989 → **선행 TODO T2**.
  - **2.6.3 Strategic archetypes**: 기존 2.5 내용 이동.

### 2.6 Research gap  → (번호 이동) 2.7 Research gap
- **핵심 메시지**: 3대 gap(framing robustness / narrative vs magnitude / operational reliability).
- **현재 상태(WS L36–L41)**: 잘 정리됨. FP L43–L49(3 gaps 번호형)이 더 깔끔.
- **수정 방향**: 번호를 **2.7**로 이동. FP의 번호형 3-gap 서술과 통합. algorithmic agency problem과 연결하는 마무리 1문장 추가.

## 3. Methodology
- **핵심 메시지(섹션 도입 WS L43–L48)**: 5단계 파이프라인(Fig.1). **유지**.
- **수정 방향(공통)**: (논문강화 Action2) **Positioning map 그림 추가** — 기존 벤치마크(QA/coding/game) vs 본 연구(R&D high-stakes + framing/context 민감도) 2축 비교도 → **선행 TODO T3**. 3장 서두 또는 2.1에 삽입.

### 3.1 Base Strategic Scenarios and Problem Framing
- **핵심 메시지**: 6개 역사기반 시나리오 = 고정 딜레마 + Generic/Specific 프레이밍.
- **현재 상태(WS L49–L56)**: 충실.
- **수정 방향**: **유지**. "scenario-based stress testing" 용어 1회 정의 추가(선택). Schilling 인용 유지.

### 3.2 Contextual Scenario Variants
- **핵심 메시지**: 4개 변형(competitive_dynamics/count_fact/opp_focus/randomized_numbers), randomized=수치통제군.
- **현재 상태(WS L58–L63)**: 충실.
- **수정 방향**: **유지**. 변경 최소.

### 3.3 Contextual Information Injection
- **핵심 메시지**: context block 모듈 주입/제거, 프레이밍과 독립.
- **현재 상태(WS L65–L72)**: 충실.
- **수정 방향**: **유지**.

### 3.4 Strategy Selection Task Design *[각 전략별, 참조 논문 달기]*
- **핵심 메시지**: 7개 폐쇄형 선택 + rationale.
- **현재 상태(WS L74–L89)**: 7전략 정의 표는 있으나 제목에 **한글 미완 TODO 마커** 존재, 전략별 인용 미완.
- **수정 방향**: ① 제목의 한글 마커 삭제. ② 표 각 행에 이론 인용 1편씩 부여(Schilling/Porter/Barney/Chesbrough/Miles/Lieberman&Montgomery). ③ JSON schema 언급은 4.2로.

### 3.5 Repeated Inference and Aggregation
- **핵심 메시지**: 단발 아닌 반복추론 → 경험적 전략 분포.
- **현재 상태(WS L91–L98)**: 충실(서술형). FP L97–L103에 entropy/JSD/Spearman·temperature{0,0.7} 명시가 더 구체.
- **수정 방향**: **유지** + FP의 지표·temperature 명시 문장 역수입.

## 4. Experimental Setup

### 4.1 Model Selection
- **핵심 메시지**: 5개 open-weight instruct 모델, 재현성·생태계 다양성·7B–14B band.
- **현재 상태(WS L102–L104)**: 충실(한 문단). FP L107도 동일 취지로 더 정돈.
- **수정 방향**: **유지**(FP L107 표현으로 다듬기 선택).

### 4.2 Prompt Design and Bias Control  ★비어 있음(제목만)★
- **핵심 메시지**: 폐쇄형 JSON schema + 옵션순서/라벨 편향 통제.
- **현재 상태(WS L106)**: **본문 없음**.
- **수정 방향**: **신규 작성**. 근거: `실험강화계획.txt` L257–L282(key_signals_used 표준화, strict JSON: chosen_option/standard_mapping/rationale/key_signals_used). 옵션 제시 순서·라벨 편향 통제, 브랜드 토큰 외 동일 프롬프트 유지 서술.

### 4.3 Inference Settings and Repetition Protocol
- **핵심 메시지**: 디코딩 설정·반복·파싱 처리.
- **현재 상태(WS L108–L110)**: 지표 언급 한 문단만 있고 실제 설정(반복수/temperature/max_tokens/파싱)이 비어 있음.
- **수정 방향**: **신규 보강**. temperature{0.0,0.7}, repeats=R, max_tokens, 파싱 실패는 재시도 대신 명시 기록(실패율 보고). 근거: `실험강화계획.txt` L128–L137(temperature 실험), L151–L204(파싱·재시도).

## 5. Key Findings   ★산만함 교정 핵심 구간 — 내용 유지, 문체 수술★
- **섹션 도입(WS L112–L113)**: 흐름 안내. **유지**(STYLE RULES 적용).

### 5.1 Strategy Distribution Across Contextual Variants
- **핵심 메시지**: 단일 default 없음, opp_focus→leadership, count_fact→defensive, 수치엔 무반응(메시지1).
- **현재 상태(WS L115–L119)**: 내용 충실하나 L116이 한 문단에 과밀(해석 장황).
- **수정 방향**: 4단 구조로 분할, 장황한 실무 해설 압축. Fig.2 유지.

### 5.2 Structural Separation and Statistical Dynamics of Strategic Contexts
- **핵심 메시지**: PCA 분리 + entropy/JSD/Spearman로 구조 확증.
- **현재 상태(WS L121–L145)**: 충실(Fig.3 + Table 2). 양호.
- **수정 방향**: **유지**, bullet 해석부 톤 정리만.

### 5.3 Decision Sensitivity to Brand Framing
- **핵심 메시지**: 브랜드=증폭기(고정 편향 아님)(메시지2).
- **현재 상태(WS L147–L153)**: 충실(Fig.4 + associative anchoring 해석).
- **수정 방향**: **유지**, 굵은글씨 정리. agency 함의 1문장 추가(이론B 연결).

### 5.4 Rationale Framing Shift Under Brand Exposure
- **핵심 메시지**: 선택이 같아도 *설명 스타일*이 바뀜(permutation test).
- **현재 상태(WS L155–L174)**: 충실(global separation 0.323 vs 0.223, p=0.003 + 키워드 표). Table 라벨이 "Table X"로 미완.
- **수정 방향**: **유지** + "Table X" → 정식 번호. 톤 정리.

### 5.5 Model-Level Behavioral Profiling and Temperature Robustness
- **핵심 메시지**: 모델별 strategic fingerprint + temperature trade-off.

  #### 5.5.1 Profiling Axes (FR/CR/NS/DS/EFI)
  - **현재 상태(WS L181–L224)**: 5축 정의. 단 **LaTeX 이스케이프 깨짐**(`\\`, `\!`, `\mathbb` 등 다수)·서술 난삽.
  - **수정 방향**: **FP L185–L209의 정제된 정의로 교체**(수식 깔끔, 의미 설명 포함). 값·정의 동일.

  #### 5.5.2 Cross-Model and Cross-Temperature Comparison
  - **현재 상태(WS L226–L253)**: Table 7(5축 점수) 포함. 양호.
  - **수정 방향**: **유지**. 3대 질문(robustness/agility/consistency) 서술 톤 정리.

  #### 5.5.3 Interpretation and Practical Implications
  - **현재 상태(WS L255–L300)**: (1)temperature shift (2)personas (3)deployment. **굵은 숫자 도배·hedge 반복("in this relative display only" 등) 최다 구간** = 산만함 주범.
  - **수정 방향**: ★STYLE RULES 집중 적용★. 페르소나당 굵게 1~2개, caveat 1회. (3) deployment 권고는 유지(WS L296–L300).

  #### 5.5.4 Scenario-Resolved Behavior (FR/CR/DS 국소 분석)
  - **핵심 메시지**: 평균이 가리는 국소 실패(메시지3).
  - **현재 상태(WS L303–L436)**: Fig.5/6 heatmap + Table 3 priority cells + **deep-dive (A)(B)(C) 완비**. WS의 최대 강점.
  - **수정 방향**: **전부 유지·복원**(FP는 이걸 한 문단으로 뭉갰음 → ESWA에선 살림). 각 케이스((A)5_model_3, (B)4_model_x, (C)2_roadster) 4단 구조 유지 + 끝에 agency 함의 1문장. (FP L238 대비 충실도가 차별점).

## 6. Discussion and Implications
- **핵심 메시지**: context-conditional 활용 + algorithmic auditor/agency governance.
- **현재 상태(WS L439–L444)**: 3문단으로 짧음(맥락조정/프레이밍 과민/휴먼인더루프). FP §5(L242–L266)가 4소절로 더 actionable.
- **수정 방향**: FP §5 구조로 재작성 + MOT 옷.
  - 6.1 context-conditional로 다뤄야(유지).
  - 6.2 **Algorithmic agency problem & audit protocol**: anonymization / dual optimistic-constraint framing test / temperature documentation를 번호형 audit 절차로(이론B·C 회수). 인용 `kdd_paper_prompt_bias.pdf`.
  - 6.3 numerical insensitivity & magnitude-sensitive 의사결정 리스크(FP §5.3).
  - 6.4 human-AI delegation(라벨+rationale 검토). 인용 `llm as virtual expert.pdf`.

## (신규) 7. Limitations and Future Work  ★신규★
- **현재 상태**: WS에 독립 섹션 없음.
- **수정 방향**: FP §6(L272–L278) 도입 — scenario scope(Tesla 6), option set(7), **human baseline 부재**(`실험강화계획.txt` L144–L147), grounded tool use + temperature/모델 band.

## (신규) 8. Conclusion  ★신규★
- **현재 상태**: WS에 없음(Discussion으로 끝남).
- **수정 방향**: FP §7(L283) 도입 — 정체성 문장(A-1)으로 닫기.

## References
- **현재 상태(WS L446–L452)**: 수량 부족(Chang/Lieberman/Schilling 정도 + 본문 \cite 키들).
- **수정 방향**: WS 본문 \cite + FP References(L306–L344) 병합. `references/` 폴더 PDF를 `workshop_paper_reference_plan.md` 매핑대로 추가. **신규 확보**: agency theory(T2), MOT 리뷰(T1). 형식 ESWA 저자-연도 통일.

## Appendix A. Data Validation and Categorical Compliance
- **핵심 메시지**: instruction-following 89%, 시나리오별 compliance.
- **현재 상태(WS L454–L467)**: 충실.
- **수정 방향**: **유지**. Section 번호만 본문 변경에 맞춰 갱신.

---

# PART C. 선행 TODO (집필과 별개 준비)
| ID | 내용 | 산출물 | 연결 섹션 |
|---|---|---|---|
| T1 | MOT 리뷰페이퍼(Research Policy/Technovation/R&D Mgmt "AI & strategic decision") 1~2편 확보 | 인용 1~2 | 1. Intro, 2.7 gap |
| T2 | Agency theory 인용 확보(Jensen&Meckling 1976; Eisenhardt 1989) | 인용 2 | 2.6.2, 6.2 |
| T3 | Positioning map 그림 제작(기존 벤치마크 vs 본 연구 2축) | 신규 PNG | 3. 서두 |
| T4 | `references/` PDF → 저자-연도 인용키 정리 | 인용 목록 | 2.x, 5.x, 6.x |
| T5 | (선택) repeats 30회·human baseline 강화 | 추가 데이터 | reviewer 방어 |

# PART D. 외부 LLM 집필 가이드
1. 섹션 1개씩만 의뢰(전체 일괄 금지 → 톤 붕괴 방지).
2. 의뢰 시 첨부: ① 해당 섹션 [수정 방향] ② A-4 [STYLE RULES] ③ 원본 절 텍스트.
3. 산출물은 `RnDSeoul/eswa_paper.md`에 누적, 절마다 용어 일관성 점검.
4. 수치·표·그림 경로 변경 금지(원본 값 보존).
5. 완성 후 전체 STYLE RULES로 1회 통독 교정.

# PART E. 진행 체크리스트
- [ ] Title/Abstract 재작성
- [ ] 신규 2.6 Bounded rationality, agency, and strategic archetypes in LLM decision-making (2.6.1 bounded rationality / 2.6.2 agency / 2.6.3 archetypes 이동)
- [ ] 2.4 중복 문장 삭제 + 인지편향 언급 2.6으로 이동
- [ ] 2.6 Research gap → 2.7 번호 이동·통합
- [ ] 1. Introduction 보강(RQ + MOT gap)
- [ ] 6. Discussion 재작성(audit protocol)
- [ ] 3.4 제목 한글 마커 삭제 + 전략별 인용
- [ ] 4.2 Prompt Design 본문 신규
- [ ] 4.3 Inference Settings 본문 보강
- [ ] 5.5.1 수식 FP 버전으로 교체
- [ ] 5.5.3 문체 수술(굵은글씨·hedge)
- [ ] 5.5.4 deep-dive (A)(B)(C) 유지·복원 + agency 함의
- [ ] 7. Limitations / 8. Conclusion 신규(FP)
- [ ] References 병합·확장·형식 통일
- [ ] 선행 TODO T1~T4
- [ ] 전체 STYLE RULES 통독 교정
