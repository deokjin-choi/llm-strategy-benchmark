# Workshop presentation script

Slides: `workshop_slide.md` (SLIDES 1–14, BACKUPS A–F). Use **Emphasis** as delivery cues; **Script** is a spoken guide (trim to time).

---

## SLIDE 1 — Title

**Emphasis**
- Clear title; name, affiliations, workshop, contact.

**Script**
Good morning. My name is Deokjin Choi, /a Ph.D. student at Sungkyunkwan University,/ and I also work for Samsung Electronics/ as a data scientist. Today,/ I will talk/ how large language models/ behave as strategic decision-makers,/ focusing on /their sensitivity to/ **context and framing**.

---

## SLIDE 2 — Contents

**Emphasis**
- Roadmap: motivation, prior work and gap, method, findings, implications.

**Script**
This talk/ has five parts. First, /I’ll explain /why this topic matters. Then I’ll briefly discuss /the research gap/ and the methodology. After that, /I’ll present the main findings/ and the implications for R&D management.

---

## SLIDE 3 — LLMs in R&D-Related Workflows

**Emphasis**
- LLMs are already in the **pipeline** for decisions, not only drafting.
- Gap preview: we care about **stability of strategic stance**, not only task accuracy.

**Script**
LLMs are used/ in R&D and/ innovation management tasks/such as technology assessment,/ competitive analysis,/ and strategic planning. In practice, LLMs are no longer used/ only to summarize information. They are  asked/ to recommend strategic directions /under uncertainty. However, most existing evaluation studies/ focus on accuracy, coherence,/ or task performance,/ such as factual Q&A tests. What we still do not understand / is this: Do LLMs make stable strategic judgments,/ or/ do their decisions shift /when context or/ framing changes?

---

## SLIDE 4 — Prior Work, Gap, and This Study

**Emphasis**
- Four streams: **ops DSS**, **safety**, **grounding**, **reasoning benchmarks**—each stops short of **fixed-option strategic choice** under **framing**.
- Three gaps: **framing**, **narrative vs. numbers**, **temperature**.
- Pivot: **decision sensitivity**, **conditional agents**.
- Close with the **research question** cleanly.

**Script**
Prior work/ clusters into/ four areas: operational decision support,/ reliability and safety /such as hallucination risk, /knowledge-grounded systems,/ and richer reasoning benchmarks. Each improves /how we use LLMs, /but they rarely ask: How sensitive are/ LLMs’ strategic decisions /to context /and framing? This study /focuses on that question. Especially, we examine three things: First, the effect of framing.
Second, the difference between semantic information /and numerical changes. And third, the effect of temperature settings.

---

## SLIDE 5 — Methodology Overview

**Emphasis**
- **Six** historical scenarios, **two** framings, **four** context variants, **seven** fixed strategies, **five** models, **repeated** sampling.
- Point at the figure: **pipeline** from scenarios to distributions.

**Script**
Let me explain/ the methodology. If you look at the figure/ on the left,/ the experiment followed/ this pipeline. We constructed/ six base scenarios/ grounded in /Tesla’s historical development. Each base scenario/ represented a fixed strategic dilemma; /for example,/ in the Model 3 case, /the firm had to decide /how to scale production rapidly /while maintaining quality and public trust. Then, /we made different contextual variants,/ such as /an opportunity-focused scenario,/ an unfavorable constraints scenario,/ and a numerical perturbation scenario /within a 20% range. Importantly,/ the main strategic problem /stayed the same. Next,/ the model /had to choose /one strategy /from seven predefined options/ and/ provide brief rationales.

We tested /five open-source LLMs, /used two temperature settings,/ and repeated /each condition/ 30 times. We also added/ a framing layer. In some cases,/ it was kept/ anonymous. In other cases, /it was explicitly named /as 'Tesla'. This allowed us /to test/ whether brand identity/ changes strategic decisions.

---

## SLIDE 6 — Finding #1: Strategy Distribution Shifts by Context

**Emphasis**
- No single default strategy.
- **Opportunity** pulls toward leadership; **constraints** toward niche and follower.
- **Numbers alone** barely move the distribution—**semantics** dominate.

**Script**
Now/ I will turn/ to the main findings. The heatmap /on this slide shows/ the strategy selection ratio/ across /all five conditions. LLMs/ do not rely on /a single default strategy. In the base scenarios,/ Niche Focus /is the most frequent choice,/ followed by Technology Leadership. But strategy selection /varies/ systematically/ across contextual conditions. Opportunity-focused contexts/ increase/ leadership-oriented strategies. Unfavorable constraints /induce/ more Niche Focus/ and Fast Follower strategies/. Pure numerical perturbations/ have minimal effect./ This suggests that/ semantic context,/ not numeric noise,/ drives strategic shifts.

---

## SLIDE 7 — Finding #2: Structural Separation of Strategic Contexts

**Emphasis**
- PCA: **opportunity** vs **constraint** separate; **base** sits with **randomized numbers**.
- Table: **entropy** drops under opportunity; **JSD** shows asymmetric sensitivity; **Spearman = 1** for numbers.

**Script**
These shifts are/ not random. /On the left,/ the strategy selection ratio/ are projected /into 2D PCA. It shows/ clear structural separation/ between opportunity-focused scenarios/ and unfavorable-constraint scenarios/. Meanwhile, /base scenarios/ and randomized-number variants/ cluster/ closely together. This indicates that/ LLMs distinguish /different strategic environments,/ especially/qualitative information changes./ The table on the right/ shows that/ LLMs react /much more strongly/ to positive framing /than to negative facts/. While opportunity scenarios/ lead to a large shift/ from the baseline/ and lower entropy, /changing the actual numbers/ has almost no effect. /Therefore,/ managers should be careful, /as /LLMs may overreact/ to optimistic information /and fail/ to notice critical changes/ in numerical data.

---

## SLIDE 8 — Finding #3: Brand Framing Amplifies Contextual Sensitivity

**Emphasis**
- Brand is **not** a fixed bias toward one strategy—it **amplifies** whatever the context already suggests.
- **Associative anchoring**: pioneer vs survivor narratives.

**Script**
Brand framing/ affects/ decision sensitivity, /not absolute rankings. This slide shows/ how strategy selection shifts/ compared to the base scenario/ across the four contextual variants./ Blue bars/ represent the generic frame, /and orange bars/ represent the Tesla-specific frame./ When Tesla is explicitly named,/ LLMs become more defensive /under unfavorable conditions/ and /more aggressive /under opportunity-focused contexts/. In other words,/ brand framing /amplifies /reactions to context/. We interpret this /as/ associative anchoring/: the model connects/ the brand name/ with familiar narratives/, such as an innovative pioneer/ or a survivor/ under severe manufacturing /and financial pressure./ Depending on the context,/ one of these narratives /becomes/ more salient /and pushes the model’s decision /in that direction.

---

## SLIDE 9 — Finding #4: Rationales Shift When the Strategy Choice Is Identical

**Emphasis**
- **What** vs **why**: same strategy, different **story**.
- Stats: global lexical separation **0.323** vs **0.223**, **p = 0.003**; masked tokens.
- Keyword table: **vision** vs **operations** language.

**Script**
Even when the chosen strategy/ matches across frames, the **rationale language** /still changes. This slide shows/ a keyword table/ comparing specific and/ generic-frame rationales./ We collect/ same-choice rationales, mask brand/ and number words,/ and extract keywords/ that are/ more associated /with each frame. The pattern is clear/: specific-frame language/ is more mission-/ and/ vision-led,/ while generic-frame language/ stresses constraints /and feasibility/. For governance/, the issue/ is not only/ what the model chooses, /but what kind of reasoning /it makes salient.

---

## SLIDE 10 — Finding #5: Five Behavioral Axes

**Emphasis**
- **FR** framing robustness; **CR** context; **NS** numbers; **DS** repeatability; **EFI** rationale invariance.
- Higher scores indicate stronger performance on the corresponding behavioral axis.

**Script**
To compare models,/ we define/ five behavioral axes/. FR is framing robustness,/ meaning /whether the choice changes/ when the firm name changes. CR is context responsiveness,/ meaning sensitivity/ to semantic context./ NS is numerical sensitivity,/ DS is decision stability /across repeated runs,/ and/ EFI is explanatory framing invariance/ when the choice is fixed. Across these axes,/ a higher score means/ stronger performance /on that behavior.

---

## SLIDE 11 — Finding #5 (continued): Profiling Radar & Temperature

**Emphasis**
- Radar is **scaled for display**; reported model numbers are **raw** (full grid on backup if asked).
- **T = 0** vs **0.7**: systematic footprint change.
- Low T: higher **DS**, **CR**, **NS**—sharper, may lock framing.
- Higher T: often **FR**, **EFI** up; **DS** down—flexibility vs repeatability.
- Three **personas**: Qwen stable; DeepSeek precise but brittle at high T; Llama adaptive.

**Script**
From here,/ I use/ these behavioral axes./ In each radar chart,/ the blue line represents/ temperature zero,/ and the red line represents/ temperature zero point seven./ Moving from zero /to point seven/ changes the radar shape/ in structured ways: lower temperature means/ higher decision stability/ and context responsiveness./ Higher temperature/ often raises/ framing robustness /and EFI/ while hurting decision stability./ Among the five models/, Qwen shows/ very stable choice concentration; DeepSeek/ is extremely strong/ at zero temperature/ but/ its context responsiveness and stability/ collapse/ when sampling noise increases. This indicates that/ managers should pick/ model and temperature/ together.

---

## SLIDE 12 — Scenario-Level Heterogeneity

**Emphasis**
- Heatmaps: same model, different **cells**, very different behavior.
- Three **priority** cells: worst FR, highest CR, worst DS—**aggregate profiles hide this**.

**Script**
Aggregate scores/ smooth over trouble. This slide shows that/ the same model can behave/ very differently /depending on the scenario. For example, /framing robustness/ breaks down /in the model 3 mass-market scenario/ for Qwen,/ context responsiveness/ is very high/ in the Model X launch scenario,/ and decision stability/ is very low /for DeepSeek/ in the Roadster launch scenario./ If managers /only look at/ average radar scores/, they miss /where the model fails. We also /need to examine /specific scenario-level behaviors.

---

## SLIDE 13 — Practical Implications for R&D Management

**Emphasis**
- **Anonymize** and **A/B** framing; test **opportunity vs constraint**; don’t trust silent numeric edits.
- Document **T** and **model**.
- Read **rationales**; treat outputs as **exploratory**.

**Script**
This study suggests /several practical implications./ First,/ companies should compare/ anonymous and /brand-specific prompts/ side by side. Second,/ because models/ may react strongly/ to optimistic information, /managers should test/ both optimistic and conservative framings./ Third,/ do not assume that/ small changes in numbers, /such as market size/ or cost figures,/ will change/ the model's strategic choice./ Finally, /compare explanations, /not only labels. /Overall,/ LLM outputs /should be treated /as exploratory support tools/, not final decision-makers.

---

## SLIDE 14 — Discussion: Questions for Editors

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
