# Workshop presentation script

Slides: `workshop_slide.md` (SLIDES 1–13, BACKUPS A–F). Use **Emphasis** as delivery cues; **Script** is a spoken guide (trim to time).

---

## SLIDE 1 — Title

**Emphasis**
- Clear title; name, affiliations, workshop, contact.

**Script**
Good morning. I’m Deokjin Choi, from Sungkyunkwan University and Samsung Electronics. Today I’ll present work on how large language models respond to **context and framing** when they make **strategic** choices—not just when they summarize text. I’m grateful for the chance to discuss this at the R&D Management Workshop.

---

## SLIDE 2 — LLMs in R&D-Related Workflows

**Emphasis**
- LLMs are already in the **pipeline** for decisions, not only drafting.
- Gap preview: we care about **stability of strategic stance**, not only task accuracy.

**Script**
Let me orient us. LLMs are already used for technology assessment, competitive intelligence, long-term scenarios, and portfolio-style recommendations. Adoption is real. What is less clear is whether their **strategic recommendations stay stable** when we rephrase the same dilemma, change emphasis, or name the firm. Most governance still focuses on correctness and hallucination—not on **choice sensitivity**.

---

## SLIDE 3 — Prior Work, Gap, and This Study

**Emphasis**
- Four streams: **ops DSS**, **safety**, **grounding**, **reasoning benchmarks**—each stops short of **fixed-option strategic choice** under **framing**.
- Three gaps: **framing**, **narrative vs. numbers**, **temperature**.
- Pivot: **decision sensitivity**, **conditional agents**.
- Close with the **research question** cleanly.

**Script**
Prior work clusters into four areas: operational decision support, reliability and safety, knowledge-grounded systems, and richer reasoning benchmarks. Each improves how we use LLMs, but they rarely ask: if I hold the **menu of strategies** fixed and only change **wording, context, or firm identity**, how does the **distribution of choices** move? Our gaps are framing robustness, whether narrative beats moderate numeric change, and how decoding temperature interacts. We shift from seeking one best answer to measuring **decision sensitivity**, and we treat models as **conditional decision agents**. The question is: how stable are strategic judgments over established archetypes under contextual and framing shifts?

---

## SLIDE 4 — Methodology Overview

**Emphasis**
- **Six** historical scenarios, **two** framings, **four** context variants, **seven** fixed strategies, **five** models, **repeated** sampling.
- Point at the figure: **pipeline** from scenarios to distributions.

**Script**
Methodologically, we use six Tesla-anchored phases as base scenarios. Every run varies framing—anonymous firm versus Tesla—and one of four context variants, including opportunity emphasis, unfavorable facts, competitive dynamics, and a numeric perturbation control. The model must pick **one** of seven canonical strategy types, with a short rationale. We run five open instruction-tuned models and repeat inference to build **empirical distributions**, not single answers. The slide figure summarizes that pipeline.

---

## SLIDE 5 — Finding #1: Strategy Distribution Shifts by Context

**Emphasis**
- No single default strategy.
- **Opportunity** pulls toward leadership; **constraints** toward niche and follower.
- **Numbers alone** barely move the distribution—**semantics** dominate.

**Script**
First result: models do not collapse on one default. In the baseline, niche and related conservative postures are common; when we emphasize opportunity, technology leadership surges; when we emphasize hard constraints, we see defensive repositioning. Crucially, perturbing numbers by about twenty percent barely shifts the strategy mix. So in this benchmark, **categorical** strategy choice tracks **semantic** context much more than moderate numeric tweaks.

---

## SLIDE 6 — Finding #2: Structural Separation of Strategic Contexts

**Emphasis**
- PCA: **opportunity** vs **constraint** separate; **base** sits with **randomized numbers**.
- Table: **entropy** drops under opportunity; **JSD** shows asymmetric sensitivity; **Spearman = 1** for numbers.

**Script**
The PCA view reinforces that: opportunity-focused and constraint-focused environments occupy different regions, while the baseline and numeric-perturbation condition cluster together. The metrics table makes this quantitative: lowest entropy under opportunity framing—models look more “decisive”—and the distributional shift from baseline is several times larger for opportunity than for the unfavorable-fact variant here. For randomized numbers, divergence from baseline is essentially zero and rank correlation is one. That is distributional evidence of **numerical insensitivity** relative to semantic framing.

---

## SLIDE 7 — Finding #3: Brand Framing Amplifies Contextual Sensitivity

**Emphasis**
- Brand is **not** a fixed bias toward one strategy—it **amplifies** whatever the context already suggests.
- **Associative anchoring**: pioneer vs survivor narratives.

**Script**
Third finding: naming Tesla does not uniformly push innovation or defense. It **amplifies** the prevailing cue—stronger leadership tilt under opportunity, stronger defensive tilt under constraint. Numeric perturbation still does not interact with brand. We interpret this as associative anchoring: pretrained narratives about the firm interact with the scenario, acting as a **sensitivity modulator**, not a constant strategic prior.

---

## SLIDE 8 — Finding #4: Rationales Shift When the Strategy Choice Is Identical

**Emphasis**
- **What** vs **why**: same strategy, different **story**.
- Stats: global lexical separation **0.323** vs **0.223**, **p = 0.003**; masked tokens.
- Keyword table: **vision** vs **operations** language.

**Script**
Even when the chosen strategy matches across frames, the **rationale language** diverges systematically. After masking brand and number words, we still see a significant shift in word associations—Tesla-side language is more mission- and vision-led; generic language stresses constraints and feasibility. For governance, that means you cannot audit only the label; you must read **how** the model justifies itself.

---

## SLIDE 9 — Finding #5: Five Behavioral Axes

**Emphasis**
- **FR** framing robustness; **CR** context; **NS** numbers; **DS** repeatability; **EFI** rationale invariance.
- Radar is **scaled for display**; reported model numbers are **raw** (full grid on backup if asked).

**Script**
To compare models, we define five axes: robustness to branding, responsiveness to semantic context, sensitivity to numeric perturbation, stability across repeated runs, and invariance of rationales when the choice is fixed. Framing robustness is one minus average Jensen–Shannon divergence across generic and specific runs; the others follow the definitions on the slide. Note that radar plots rescale spokes for readability; the **numeric table** for each model uses raw scores—worth showing from backup if someone wants every cell.

---

## SLIDE 10 — Finding #5 (continued): Profiling Radar & Temperature

**Emphasis**
- **T = 0** vs **0.7**: systematic footprint change.
- Low T: higher **DS**, **CR**, **NS**—sharper, may lock framing.
- Higher T: often **FR**, **EFI** up; **DS** down—flexibility vs repeatability.
- Three **personas**: Qwen stable; DeepSeek precise but brittle at high T; Llama adaptive.

**Script**
Temperature is not a cosmetic knob. Moving from zero to point seven changes the radar shape in structured ways: lower temperature usually means higher decision stability and context responsiveness in this panel, but it may reinforce narrow habits. Higher temperature often raises framing robustness and rationale invariance while hurting repeatability. Among the five models, Qwen shows very stable choice concentration; DeepSeek is extremely strong at zero temperature but its context responsiveness and stability collapse when sampling noise increases; Llama leads context responsiveness at low T and rationale invariance at high T. The practical point: **pick model and temperature together**.

---

## SLIDE 11 — Scenario-Level Heterogeneity

**Emphasis**
- Heatmaps: same model, different **cells**, very different behavior.
- Three **priority** cells: worst FR, highest CR, worst DS—**aggregate profiles hide this**.

**Script**
Aggregate scores smooth over trouble. In these heatmaps, certain scenario-by-model cells stand out: for example, extreme framing sensitivity in the mass-market production scenario for Qwen, extreme context sensitivity in the Model X launch cell, and very low repeatability for DeepSeek in the early Roadster scenario. If you only look at average radar scores, you miss where the system fails. Deployment checks should probe **specific** scenario, framing, and context-load combinations.

---

## SLIDE 12 — Practical Implications for R&D Management

**Emphasis**
- **Anonymize** and **A/B** framing; test **opportunity vs constraint**; don’t trust silent numeric edits.
- Document **T** and **model**.
- Read **rationales**; treat outputs as **exploratory**.

**Script**
For R&D practice: anonymize firm names when you want less narrative priming; always compare generic and specific prompts when brand matters. Stress-test optimistic versus conservative framings. Do not assume that tweaking percentages will change the strategic class the model recommends—check explicitly. Record temperature and model ID like any other experimental parameter. Compare explanations, not only labels. Overall, treat LLM strategy output as input to human judgment, not a prescription.

---

## SLIDE 13 — Discussion: Questions for Editors

**Emphasis**
- Invite feedback on **scope**, **methods**, **theory**, **practice**, **outlets**.
- Thank the panel.

**Script**
I’ll close with questions for the editors and the room. Should we broaden scenarios or model sizes? Is the five-axis profile convincing, and would a human baseline help? Does “conditional strategic agent” capture what we see, and how should we connect to behavioral theory? Would a practitioner matrix of model, temperature, and task type be useful—and what R&D problems should we test next? Finally, what is the single most important extension before submission, and where should we aim to publish? Thank you—I’m looking forward to your comments.

---

## BACKUP A — Instruction-Following Compliance

**Emphasis**
- ~**89%** valid categorical compliance; invalid responses excluded from distributions.

**Script (if asked)**
Models usually follow the closed-choice format; non-compliance is in the low double digits depending on variant, with slightly more drift in the sparsest baseline context. Analyses use only valid picks, so the strategy distributions are normalized accordingly.

---

## BACKUP B — The 7 Strategic Archetypes

**Emphasis**
- **Closed set** grounded in innovation-strategy literature.

**Script (if asked)**
These seven labels are the full option set in every run—from technology leadership through maintain—so we can compare distributions apples-to-apples across scenarios and models.

---

## BACKUP C — Model-Level Scores (Raw)

**Emphasis**
- Full **10 rows**; pattern **↑ FR, ↑ EFI, ↓ DS** from **T=0** to **0.7** on average.

**Script (if asked)**
Here is the complete raw table behind the radar and persona bullets. It is useful when someone wants exact numbers for a specific model or to verify the temperature comparison claim.

---

## BACKUP D — FR Case Deep Dive

**Emphasis**
- Same facts: **Open Innovation** vs **Maintain**; lexical test **p ≈ 0.005**.

**Script (if asked)**
This is the starkest framing cell: identical baseline story, but naming the firm flips the chosen strategy class and systematically shifts rationale keywords, confirmed by permutation testing.

---

## BACKUP E — CR Case Deep Dive

**Emphasis**
- **Asymmetric** cue response: opportunity moves mass strongly; constraint less.

**Script (if asked)**
In the Model X scenario, semantic variants reallocate mass across strategy families unevenly—watch for overreaction to positive cues and underreaction to negative facts.

---

## BACKUP F — DS Case Deep Dive

**Emphasis**
- **Context load** drives instability; not only temperature.

**Script (if asked)**
For DeepSeek in the Roadster scenario, how much context you inject changes the strategy mixture dramatically; audits must vary information depth, not only randomness settings.
