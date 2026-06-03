## SLIDE 1 — Title

**Context and Framing Sensitivity in LLM-Based Strategic Decision-Making**

- Deokjin Choi, PhD student  
- Sungkyunkwan University / Samsung Electronics  
- R&D Management Workshop 2026 — Seoul, May 29–30  
- deokjin.choi@gmail.com  

---

## SLIDE 2 — Contents

1. Motivation: LLMs in R&D workflows
2. Prior work & research gap
3. Methodology
4. Key findings
5. Implications for R&D management

---

## SLIDE 3 — LLMs in R&D-Related Workflows

**LLMs are already embedded in work that feeds strategic and portfolio decisions—not only text generation.**

| Area | Example LLM roles |
|------|-------------------|
| Technology assessment | Emerging-tech readiness and signal synthesis |
| Competitive analysis | Patent/market monitoring and summarization |
| Long-term planning | Scenario generation and strategic foresight |
| R&D portfolio decisions | Resource-allocation recommendations and option narratives |

- Adoption is **real**: models sit in pipelines that inform high-stakes choices.  
- Evaluation and governance still lean heavily on **task performance, coherence, and factual risk**—less on how **stable categorical strategic stances** are when the *same* dilemma is rephrased or reframed.

---

## SLIDE 4 — Prior Work, Gap, and This Study

**Prior research streams**

| Stream | Typical focus | Limitation for our question |
|--------|---------------|-----------------------------|
| **Decision support & operations** | LLM agents in scheduling, manufacturing, digital twins; task efficiency and plan quality (Wang et al., 2025) | Strong on **task outcomes**; weak on how a **fixed menu of strategic options** is allocated under **controlled wording/context** |
| **Reliability & safety** | Hallucination detection, evidence checks, governance layers (Kong et al., 2026) | Targets **factual and compliance risk** more than **shifts in strategic preference** when narrative framing changes |
| **Knowledge-grounded systems** | RAG, text-to-SQL, domain graphs (Ojuri et al., 2025) | Improves **what the model knows**; does not by itself show **choice stability** when only emphasis or identity cues move |
| **Reasoning & behavioral profiling** | Planning traces, multi-criteria benchmarks, bias vs. human panels (Liu et al., 2026) | Richer than accuracy-only metrics; still rarely treats the model as a **categorical strategic decision-maker** over a **held-fixed option set** |

**Gap**

- **Framing robustness:** Brand identity and narrative cues reshuffle strategy choice when underlying facts are unchanged?  
- **Narrative vs. numbers:** Stronger reaction to semantic emphasis than to moderate quantitative shifts?  
- **Operational reliability:** How decoding settings (e.g., temperature) interact with those sensitivities?

**This study**

- Emphasis on **decision sensitivity** and **empirical choice distributions**, not a single “best” answer.  
- LLMs as **conditional decision agents**: strategic stance depends on the prompt environment.

**Research question**

How stable are LLM **strategic judgments**—over established strategy archetypes—under different **contextual** and **framing** conditions?

---

## SLIDE 5 — Methodology Overview

![Research method](./final_results/plots/research_method.PNG)

- **6 base scenarios** (Tesla historical phases: Founder, Roadster, Model S, Model X, Model 3, Energy infrastructure)  
- **2 framing conditions:** Generic (anonymous firm) vs. Specific (Tesla identified)  
- **4 contextual variants:** `competitive_dynamics`, `count_fact`, `opp_focus`, `randomized_numbers`  
- **7 strategic archetypes:** Technology Leadership, Fast Follower, Open Innovation, Niche Focus, Diversification, Retrenchment, Maintain  
- **5 open-weight, instruction-tuned LLMs:** Llama 3.1 8B Instruct, Mistral 7B Instruct v0.3, Qwen 2.5 14B Instruct, DeepSeek LLM 7B Chat, Yi 1.5 9B Chat  
- **Repeated inference** → empirical strategy distributions  

---

## SLIDE 6 — Finding #1: Strategy Distribution Shifts by Context

![Strategy ratio by scenario](./final_results/plots/eval_eda_Strategy_Ratio_by_Scenario.png)

**LLMs do not converge on a single default strategy—choices shift systematically with context.**

| Context variant | Observed shift |
|-----------------|----------------|
| **Base** | **Niche Focus** most frequent; **Technology Leadership** secondary |
| **opp_focus** | **Technology Leadership** dominant |
| **count_fact** | Defensive repositioning (**Niche Focus**, **Fast Follower**) |
| **randomized_numbers** (±20%) | **Minimal** change—semantic context dominates numerical variation |

Moderate quantitative shifts alone weakly reorient **categorical** strategy choice in this benchmark.

---

## SLIDE 7 — Finding #2: Structural Separation of Strategic Contexts

![PCA of strategy ratios](./final_results/plots/eval_eda_PCA_of_Strategy_Ratios_2D_Vectorized_Analysis.png)

**Models distinguish qualitatively different strategic environments; base and numeric-perturbation conditions cluster together.**

| Scenario | Entropy | JSD from base | Spearman vs. base |
|----------|---------|---------------|-------------------|
| base | 1.8422 | 0 | 1 |
| competitive_dynamics | 1.8211 | 0.0107 | 0.8214 |
| count_fact | 1.7818 | 0.0081 | 0.6429 |
| opp_focus | 1.7030 | 0.0471 | 0.6786 |
| randomized_numbers | 1.8328 | 0.0002 | 1 |

- **opp_focus:** lowest entropy → more concentrated choices under optimistic framing.  
- **opp_focus** JSD from base **~5.8×** **count_fact** JSD → stronger shift under opportunity than under unfavorable-fact emphasis.  
- **randomized_numbers:** JSD **0.0002**, Spearman **1** → **numerical insensitivity** at the distributional level.

---

## SLIDE 8 — Finding #3: Brand Framing Amplifies Contextual Sensitivity

![Generic vs. Specific Δ by scenario](./final_results/plots/eval_Generic_and_Specific_Δ_by_Scenario.png)

**Brand identity does not push a single fixed strategy—it amplifies the direction the context already suggests.**

| Condition | Generic (anonymous) | Specific (Tesla) | Effect |
|-----------|---------------------|------------------|--------|
| **opp_focus** | Shift toward Technology Leadership | **Stronger** Technology Leadership | Amplified |
| **count_fact** | Some defensive moves | **Stronger** defensive moves (e.g., Fast Follower, Niche Focus) | Amplified |
| **randomized_numbers** | Minimal change | Minimal change | No meaningful brand effect |

**Associative anchoring:** Learned narratives (e.g., innovative pioneer vs. resilient survivor) interact with contextual cues; brand acts as a **sensitivity modulator**.

---

## SLIDE 9 — Finding #4: Rationales Shift When the Strategy Choice Is Identical

![Rationale permutation global distribution](./final_results/plots/eval_rationale_perm_global_distribution.png)

**The explanation changes with framing—not only the decision.**

Global separation (mean |Δlog-odds|) = **0.323** vs. permutation baseline **0.223**, **p = 0.003** (FDR-controlled; brand and numeric tokens masked in rationales).

| Dimension | Specific (Tesla) | Generic |
|-----------|------------------|---------|
| Leadership | mission lead, identity leader | trust balanced, goals standards |
| Technology | goal technological, position platform | enables integration, optimization scale |
| Market | world transition, leader capturing | rushed quality, delaying significant |
| Execution | prestige demonstrate, mission showcase | funding invest, make feasible |

The **what** can match while the **why** is framed differently.

---

## SLIDE 10 — Finding #5: Five Behavioral Axes

| Axis | Measures | Formula |
|------|----------|---------|
| **FR** (Framing robustness) | Strategy consistency across Generic vs. Specific | \(1 - \mathbb{E}_{s,v}[\mathrm{JSD}(P_{\mathrm{Generic}}, P_{\mathrm{Specific}})]\) |
| **CR** (Context responsiveness) | Shift under semantic context vs. base | \(\mathbb{E}[\mathrm{JSD}(P_{\mathrm{Base}}, P_{\mathrm{variant}})]\) over `competitive_dynamics`, `count_fact`, `opp_focus` |
| **NS** (Numerical sensitivity) | Shift under numeric perturbation | \(\mathbb{E}[\mathrm{JSD}(P_{\mathrm{Base}}, P_{\mathrm{Randomized}})]\) |
| **DS** (Decision stability) | Repeatability across reruns | \(1 - H(p)/\log_2|\mathcal{A}|\) |
| **EFI** (Explanatory framing invariance) | Rationale stability when choice is fixed | \(1/(1 + \mathrm{EFD}_{\mathrm{raw}})\) |

FR, CR, NS, DS ∈ [0, 1]; EFI ∈ (0, 1]. **Radar:** per-axis scaling for display only. **Tables:** raw axis scores.

---

## SLIDE 11 — Finding #5 (continued): Profiling Radar & Temperature

![Model profile radar](./final_results/plots/eval_model_profile_radar.png)

**Decoding temperature**

- **T = 0.0 → 0.7:** Systematic shift in the radar footprint—not random drift.  
- **T = 0.0:** Typically **higher DS, NS, CR** → sharper, more repeatable outputs; may **lock in** training-time framing habits.  
- **T = 0.7:** Often **higher FR and EFI**, **lower DS** → broader sampling, less superficial framing lock-in, less run-to-run consistency.  
- **Trade-off:** **Precision-oriented stability** (low T) vs. **objectivity-oriented flexibility** (higher T).

| Persona | Model | Profile (raw axis scores) |
|---------|-------|---------------------------|
| **Stable functional** | **Qwen2.5-14B-Instruct** | Very high **DS** (**0.8932 → 0.8600**); lower **FR** / **EFI** than some peers. |
| **Precision-sensitive** | **DeepSeek-LLM-7B-Chat** | **T = 0.0:** highest **FR (0.9405)**, highest **NS (0.1249)**, **DS 0.9015**, **CR 0.1138**. **T = 0.7:** **DS 0.6423**, **CR 0.0582**, **NS 0.0435**. |
| **Adaptive resilient** | **Llama-3.1-8B-Instruct** | Highest **CR** at **T = 0.0 (0.1750)**; highest **EFI** at **T = 0.7 (0.7722)** in this panel. |

Choose **model** and **temperature** together: a strong low-T profile need not carry over to higher T.

---

## SLIDE 12 — Scenario-Level Heterogeneity

![Scenario × model overview at T=0.0](./final_results/plots/eval_scenario_model_overview_FR_CR_DS__T0.png)

![Scenario × model overview at T=0.7](./final_results/plots/eval_scenario_model_overview_FR_CR_DS__T0.7.png)

**Priority cells** (mean scaled heatmap score, **T = 0.0** and **T = 0.7** averaged)

| Metric | Model | Scenario | Score | Note |
|--------|-------|----------|-------|------|
| **FR** (lowest) | Qwen2.5-14B-Instruct | `5_model_3_mass_market` | **0.000** | Extreme framing sensitivity |
| **CR** (highest) | Qwen2.5-14B-Instruct | `4_model_x_launch` | **0.935** | Large context-driven redistribution |
| **DS** (lowest) | DeepSeek-LLM-7B-Chat | `2_roadster_launch` | **0.184** | Low repeatability in this cell |

Aggregate scores can **hide** local failure modes—audit **scenario × framing × context load**.

---

## SLIDE 13 — Practical Implications for R&D Management

| Challenge | Recommendation |
|-----------|----------------|
| Brand-induced bias | **Anonymous** descriptions; **Generic vs. Specific** side-by-side |
| Optimistic framing | Test **opportunity vs. constraint** framings before committing |
| Numerical insensitivity | Stress-test **magnitude** narratives explicitly |
| Temperature | Treat **T** as a **governance lever**—document **T** and model id |
| Rationale bias | Compare **rationales** across framings—not only the chosen strategy |

**General principle:** LLM outputs as **exploratory inputs**; **human-in-the-loop** framing checks.

---

## SLIDE 14 — Discussion: Questions for Editors

| Area | Questions |
|------|-----------|
| **Empirical scope** | (1) Additional scenarios or industries? (2) Expand beyond **7B–14B**? |
| **Methodological rigor** | (3) Is the **five-axis** framework sufficiently validated? (4) **Human manager** baseline? |
| **Theoretical framing** | (5) Does **“conditional strategic agent”** fit? (6) Link to **behavioral decision theory**? |
| **Practical contribution** | (7) **Decision matrix** (model × temperature × task)? (8) **R&D problems** to test next? |
| **Next steps** | (9) Single most important extension before submission? (10) **Journal / special issue** fit? |

Thank you—I look forward to your insights.

---

# BACKUP A — Instruction-Following Compliance

| Scenario variant | Compliance (valid) | Non-compliance (error) |
|------------------|-------------------|------------------------|
| Competitive | 91.49% | 8.51% |
| Count Fact | 89.96% | 10.04% |
| Randomized | 89.47% | 10.53% |
| Opportunity | 87.23% | 12.77% |
| Base | 85.95% | 14.05% |

Analyses use normalized distributions of **valid** strategic choices (invalid responses excluded).

---

# BACKUP B — The 7 Strategic Archetypes

| Strategy option | Definition |
|-----------------|------------|
| **Technology Leadership** | Pursue technological superiority, higher risk / differentiation |
| **Fast Follower** | Adopt or refine proven technology; speed and cost efficiency |
| **Open Innovation** | Collaborate with external partners |
| **Niche Focus** | Target a narrow segment; controlled growth |
| **Diversification** | Expand into adjacent businesses |
| **Retrenchment** | Reduce scope, delay expansion, preserve stability |
| **Maintain** | Stabilize and consolidate; no major new move |

---

# BACKUP C — Model-Level Scores (Raw)

| Model | T | FR | CR | NS | DS | EFI |
|-------|---:|-----:|-----:|-----:|-----:|-----:|
| Yi-1.5-9B-Chat | 0.0 | 0.8137 | 0.0796 | 0.0803 | 0.8552 | 0.6366 |
| Yi-1.5-9B-Chat | 0.7 | 0.8298 | 0.0691 | 0.0430 | 0.6943 | 0.7633 |
| Qwen2.5-14B-Instruct | 0.0 | 0.6749 | 0.1485 | 0.0278 | 0.8932 | 0.6385 |
| Qwen2.5-14B-Instruct | 0.7 | 0.7066 | 0.1340 | 0.0151 | 0.8600 | 0.7070 |
| DeepSeek-LLM-7B-Chat | 0.0 | 0.9405 | 0.1138 | 0.1249 | 0.9015 | 0.6462 |
| DeepSeek-LLM-7B-Chat | 0.7 | 0.9560 | 0.0582 | 0.0435 | 0.6423 | 0.7705 |
| Llama-3.1-8B-Instruct | 0.0 | 0.8515 | 0.1750 | 0.0263 | 0.8759 | 0.6514 |
| Llama-3.1-8B-Instruct | 0.7 | 0.9334 | 0.1206 | 0.0232 | 0.7273 | 0.7722 |
| Mistral-7B-Instruct-v0.3 | 0.0 | 0.7594 | 0.1325 | 0.0336 | 0.9178 | 0.6417 |
| Mistral-7B-Instruct-v0.3 | 0.7 | 0.8195 | 0.1078 | 0.0250 | 0.8336 | 0.7630 |

**T = 0.0 → 0.7:** Often **↑ FR, ↑ EFI, ↓ DS** across this panel.

---

# BACKUP D — FR Case Deep Dive (Qwen2.5-14B / Mass Market)

**Problem (scenario text)**  
Facing an explosive increase in pre-orders, a company must rapidly scale production while maintaining product quality, financial stability, and public trust. The core challenge is to overcome production bottlenecks and reduce costs without sacrificing the brand’s reputation in the mass market.

**Strategy menu (7 options)**  

| Mapped strategy | Execution option |
| :---- | :---- |
| Technology Leadership | Prioritize production speed to meet demand as quickly as possible |
| Fast Follower | Adopt proven mass-manufacturing practices from incumbents to catch up quickly |
| Open Innovation | Utilize manufacturing partners (OEM) to scale production |
| Niche Focus | Restrict deliveries to priority regions/customers first (e.g., North America only) |
| Diversification | Expand into related mass-market products (e.g., Model Y crossover) simultaneously |
| Retrenchment | Scale down volume targets to protect financial stability |
| Maintain | Expand production gradually while prioritizing quality and profitability |

**Observed choices (base context, pooled)**  

![FR deep dive Qwen Model 3](./final_results/plots/eval_deepdive_fr_framing_stacks__Qwen2.5-14B__5_model_3_mass_market.png)

| Framing | Strategy | Rationale keywords (examples) |
|---------|----------|--------------------------------|
| **Generic** | **Open Innovation** | manufacturing partners, scale production, quickly scale |
| **Specific (Tesla)** | **Maintain** | quality profitability, maintaining reputation, long term |

**Implication**  
Same baseline narrative; firm-identity framing reallocates the chosen strategy (**Open Innovation → Maintain**). Rationale language also shifts systematically (lexical gap: **p ≈ 0.005** at **T = 0.0** and **T = 0.7**). Minimum audit: run **Generic vs. Specific** side-by-side.

---

# BACKUP E — CR Case Deep Dive (Qwen2.5-14B / Model X Launch)

**Problem (scenario text)**  
A company aims to enter the growing SUV market. However, a complex product design creates high production difficulty and quality risks, which could severely damage the brand's reputation despite a lack of direct competition.

**Strategy menu (7 options)**  

| Mapped strategy | Execution option |
| :---- | :---- |
| Technology Leadership | Launch a luxury SUV with innovative, complex features |
| Fast Follower | Introduce a simpler SUV quickly to capture demand before competitors |
| Open Innovation | Partner with suppliers/OEMs to co-develop SUV platform and reduce complexity |
| Niche Focus | Develop a standard, mid-priced SUV for a specific customer segment |
| Diversification | Expand into related vehicle categories (e.g., crossover, minivan) alongside SUV |
| Retrenchment | Reduce scope of SUV project, scale down features to cut risk |
| Maintain | Postpone the SUV launch, focus on stabilizing Model S production first |

**Observed choices (by framing × semantic variant)**  

![CR deep dive strategy stacks](./final_results/plots/eval_deepdive_cr_strategy_stacks_framing__Qwen2.5-14B__4_model_x_launch.png)

![CR deep dive JSD by variant](./final_results/plots/eval_deepdive_cr_jsd_by_variant__Qwen2.5-14B__4_model_x_launch.png)

| Context variant | Effect |
|-----------------|--------|
| **opp_focus** | Strong shift toward **Technology Leadership** |
| **competitive_dynamics** | **Fast Follower** rises |
| **count_fact** | Small move vs. base in this cell |

**Implication**  
Semantic variants can reallocate probability mass across distinct strategy families. In this cell, the model reacts strongly to **opportunity** and **competitive** cues, while movement under **unfavorable facts** is comparatively small—an asymmetric cue sensitivity that should be tested explicitly.

---

# BACKUP F — DS Case Deep Dive (DeepSeek-LLM-7B / Roadster)

**Problem (scenario text)**  
A company must manage conflicting goals of product quality and timely delivery during its initial product launch. With significant pre-orders already placed, the company faces severe cash flow issues and supply chain delays, jeopardizing brand trust and future investment if not handled correctly.

**Strategy menu (7 options)**  

| Mapped strategy | Execution option |
| :---- | :---- |
| Technology Leadership | Prioritize product quality and performance, accepting launch delays |
| Fast Follower | Accelerate launch to meet demand, accepting potential quality compromises |
| Open Innovation | Expand manufacturing partnerships to share risk |
| Niche Focus | Limit deliveries to early adopters first, delaying mass rollout |
| Diversification | Introduce parallel revenue streams (e.g., licensing tech, consulting) to ease cash flow |
| Retrenchment | Scale back launch volume until supply chain stabilizes |
| Maintain | Delay full-scale launch, focus on stabilizing operations and cash position |

**Observed choices (by context load / Num Context tiers)**  

![DS deep dive strategy stacks](./final_results/plots/eval_deepdive_ds_strategy_stacks_numcontext_framing__deepseek-llm-7b-chat__2_roadster_launch.png)

![DS deep dive entropy](./final_results/plots/eval_deepdive_ds_entropy_numcontext_box__deepseek-llm-7b-chat__2_roadster_launch.png)

| Context load | Behavior |
|--------------|----------|
| Minimal | Mass leans **Open Innovation** |
| Partial | **Retrenchment** / **Niche Focus** |
| Full | **Fast Follower** vs. **Retrenchment**; **T = 0.7** more diffuse |

**Implication**  
Stability depends on **how much context** is in the prompt (and on framing), not on **temperature** alone. Reliability audits should vary **Num Context** tiers explicitly.
