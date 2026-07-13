# Archive: §5.6 (pre-rewrite draft)

**Source:** `ESWA_paper_harvard_0708.md` (removed 2026-07-13)  
**Original title:** 5.6 Model-Level Behavioral Profiling and Temperature Robustness  
**Note:** Preserved for rewrite into audit-aligned structure (Localized Sensitivity Profiles and Deployment Risk Cases).

---

**5.6 Model-Level Behavioral Profiling and Temperature Robustness**  
This section extends the scenario-level findings by profiling model-specific behavioral signatures under the same strategic benchmark. Rather than relying on pooled trends, we compare models as distinct strategic reasoners and evaluate whether their behavioral profiles remain stable across decoding temperatures.

To support this comparative analysis, we formalize the quantitative framework into three specific profiling axes. These axes transition from general descriptive metrics to formal behavioral indicators, allowing for a multidimensional assessment of how each model navigates the trade-offs between contextual responsiveness, framing stability, and decision stability.

5.6.1 Profiling Axes  
To quantify model behavior, we define a strategy distribution $P(m, \tau, s, n, v, \phi)$ for a given model $m$, temperature $\tau$, scenario $s$, context load $n$ (Num Context), context variant $v$, and framing type $\phi \in \{\text{Generic, Specific}\}$. we employ the Jensen–Shannon Divergence ($JSD$) to measure the statistical distance between distributions.

1\. Framing Robustness (FR)  
FR measures the model's ability to maintain a consistent strategy regardless of arbitrary branding changes (Generic vs. Specific) under identical underlying conditions.

$$FR(m, \tau) = 1 - \mathbb{E}_{s,n,v} \left[ JSD\left( P(m, \tau, s, n, v, \text{Generic}), P(m, \tau, s, n, v, \text{Specific}) \right) \right]$$

* Interpretation: A higher FR indicates that the model's strategic choice is invariant to framing manipulations, focusing on the core problem rather than brand-level biases.

2\. Context Responsiveness (CR)  
CR evaluates how effectively a model updates its strategic distribution when provided with high-value semantic context (e.g., competitive dynamics, factual constraints, or opportunity focus) compared to a baseline scenario.

$$CR(m,\tau)=E_{s,n,\phi,\,v\in V_{sem}}\Big[JSD\big(P(m,\tau,s,n,\mathrm{Base},\phi),P(m,\tau,s,n,v,\phi)\big)\Big]$$
where $V_{sem}$ is the semantic-variant set `{competitive_dynamics, count_fact, opp_focus}`.

* Interpretation: A higher CR reflects the model's "strategic intelligence"—its capacity to parse and integrate task-relevant nuances into its decision-making process.

3\. Decision Stability (DS)  
DS measures how predictable a model’s strategic choices are across repeated runs under fixed conditions. In the current implementation, DS is defined using the concentration of the empirical choice distribution across repeats (normalized entropy).  
Let $\mathcal{A}$ be the set of strategies and let a fixed condition be $c=(s,n,v,\phi)$, where $s$ is the scenario, $n$ is context load, $v$ is the context variant, and $\phi\in\{\text{Generic},\text{Specific}\}$ is framing. For model $m$ at temperature $\tau$, define the empirical distribution over repeats:

$$p_{c}^{m,\tau}(a)=\frac{1}{R}\sum_{r=1}^{R}\mathbf{1}\big[\text{strategy}_{r}=a\big],\quad a\in\mathcal{A}.$$

Then

$$DS(m,\tau)=\mathbb{E}_{c}\left[1-\frac{H\left(p_{c}^{m,\tau}\right)}{\log_{2}|\mathcal{A}|}\right], \qquad H(p)=-\sum_{a\in\mathcal{A}}p(a)\log_{2}p(a).$$

* Interpretation: A higher DS indicates that repeated runs under the same condition concentrate on fewer strategies (higher predictability). DS approaches 1 when the model consistently selects the same strategy, and approaches 0 when choices are spread uniformly across strategies.

5.6.2 Cross-Model and Cross-Temperature Comparison  
We evaluate each model’s "strategic fingerprint" by calculating the three-axis scores at both $T=0.0$ and $T=0.7$. This comparison reveals whether a model’s behavior is rooted in its structural reasoning logic or merely a byproduct of stochastic decoding.

Table 5 lists the three-axis scores for each model $m$ and temperature $\tau$. By construction, FR, CR,and DS lie in $[0,1]$

| Model | $T$ | FR | CR | DS |
| :---- | :--: | :----: | :----: | :----: |
| Yi-1.5-9B-Chat | 0.0 | 0.8137 | 0.0796 | 0.8552 |
| Yi-1.5-9B-Chat | 0.7 | 0.8298 | 0.0691 | 0.6943 |
| Qwen2.5-14B-Instruct | 0.0 | 0.6749 | 0.1485 | 0.8932 |
| Qwen2.5-14B-Instruct | 0.7 | 0.7066 | 0.1340 | 0.8600 |
| DeepSeek-LLM-7B-Chat | 0.0 | 0.9405 | 0.1138 | 0.9015 |
| DeepSeek-LLM-7B-Chat | 0.7 | 0.9560 | 0.0582 | 0.6423 |
| Llama-3.1-8B-Instruct | 0.0 | 0.8515 | 0.1750 | 0.8759 |
| Llama-3.1-8B-Instruct | 0.7 | 0.9334 | 0.1206 | 0.7273 |
| Mistral-7B-Instruct-v0.3 | 0.0 | 0.7594 | 0.1325 | 0.9178 |
| Mistral-7B-Instruct-v0.3 | 0.7 | 0.8195 | 0.1078 | 0.8336 |

Table 5\. model-level profiling scores (three axes)

Three patterns stand out.

First, temperature consistently improves FR but degrades DS. Every model shows higher FR at T=0.7 (e.g., Llama: 0.8515 → 0.9334; DeepSeek: 0.9405 → 0.9560), while DS declines across all models—most sharply for DeepSeek (0.9015 → 0.6423) and Yi (0.8552 → 0.6943). This confirms a systematic trade-off: stochasticity reduces brand sensitivity at the cost of repeatability.

Second, CR is the most variable and model-dependent axis. Llama achieves the highest CR at T=0.0 (0.1750), nearly double that of Yi (0.0796). At T=0.7, DeepSeek's CR collapses by nearly half (0.1138 → 0.0582), while Llama and Qwen maintain moderate responsiveness (0.1206 and 0.1340, respectively). This suggests that context integration is not a stable model property but varies substantially across architectures and temperatures.

Third, models exhibit distinct prioritization patterns. DeepSeek maximizes FR (0.9405 at T=0.0; 0.9560 at T=0.7)—the most "brand-blind" model—making it suitable for impartial audits. Qwen leads on DS (0.8932 at T=0.0; 0.8600 at T=0.7)—the most repeatable—ideal for compliance workflows. Llama maximizes CR (0.1750 at T=0.0; 0.1206 at T=0.7)—the most context-responsive—best for exploratory analysis. No single model dominates all axes; selection depends on organizational priorities.

5.6.3 Scenario-Resolved Behavior: Localizing FR, CR, and DS Across the Experimental Grid

Section 5.5.2 compared models using aggregate profiling scores. While useful for benchmarking, aggregate averages do not reveal where framing sensitivity, context adaptation, or instability emerge within the experimental grid. To address this limitation, we examine scenario-level behavior for three metrics.

Fig. 8 and Fig. 9 summarize scenario × model heatmaps at $T=0.0$ and $T=0.7$. Each panel is independently min–max scaled to $[0,1]$.

![Scenario_model_overview_T0](final_results/plots/eval_scenario_model_overview_FR_CR_DS__T0.png)  
Fig. 8. Scenario × model heatmaps for $FR_{\mathrm{scenario}}$, $CR_{\mathrm{scenario}}$, and $DS_{\mathrm{scenario}}$ at $T=0.0$.

![Scenario_model_overview_T07](final_results/plots/eval_scenario_model_overview_FR_CR_DS__T0.7.png)  
Fig. 9. Scenario × model heatmaps for $FR_{\mathrm{scenario}}$, $CR_{\mathrm{scenario}}$, and $DS_{\mathrm{scenario}}$ at $T=0.7$.

**(1) Heterogeneous local behavior.**  
FR and DS remain relatively high across most cells, while CR is sparse—only a few scenario–model pairs (e.g., 4_model_x_launch) show strong context-driven redistribution. CR's concentration in specific cells (not uniform across all cells) reveals that context responsiveness is situation-dependent. This conditional dependence supports treating models as conditional decision agents: the three axes do not co-move; aggregate profiling scores therefore pool qualitatively different local behaviors. Consequently, a model that appears "moderately context-sensitive" on aggregate may in fact be highly responsive in a few critical scenarios and entirely unresponsive in others—a distinction that only scenario-resolved analysis can reveal.

**(2) Temperature effects.**  
The relative spatial structure of FR and CR is broadly preserved across temperatures (e.g., Qwen2.5-14B's FR dip on 5_model_3_mass_market persists, and its CR hotspot on 4_model_x_launch remains equally prominent), whereas DS weakens at T=0.7. This indicates that stochastic decoding primarily erodes run-to-run repeatability without fundamentally altering which scenarios trigger framing sensitivity or context-driven redistribution.

**Priority cells for case analysis.**

The patterns above motivate a closer look at specific scenario–model pairs where each metric's characteristic behavior is most pronounced. For FR and DS, we select the cells with the lowest scaled scores—where framing robustness or decision stability fails most visibly. For CR, we select the cell with the highest scaled score—where semantic context most strongly reallocates strategy choices. Table 6 lists the resulting three priority cells, averaging across T=0.0 and T=0.7 to reflect both decoding regimes.

| Metric | Model | Scenario | Mean scaled value (Figs. 8–9, avg.) |
| :---- | :---- | :---- | :---- |
| $FR_{\mathrm{scenario}}$ (min) | Qwen2.5-14B-Instruct | 5_model_3_mass_market | 0.000 |
| $CR_{\mathrm{scenario}}$ (max) | Qwen2.5-14B-Instruct | 4_model_x_launch | 0.935 |
| $DS_{\mathrm{scenario}}$ (min) | deepseek-llm-7b-chat | 2_roadster_launch | 0.184 |

Table 6. Priority scenario–model cells.

---

**(A) FR Case: Brand Identity Changes Strategic Preference (5_model_3_mass_market)**

**(1) Problem (scenario text)**: Facing an explosive increase in pre-orders, a company must rapidly scale production while maintaining product quality, financial stability, and public trust. The core challenge is to overcome production bottlenecks and reduce costs without sacrificing the brand’s reputation in the mass market.

**(2) Strategy menu (options).**

| Mapped strategy | Execution option |
| :---- | :---- |
| Technology Leadership | Prioritize production speed to meet demand as quickly as possible |
| Fast Follower | Adopt proven mass-manufacturing practices from incumbents to catch up quickly |
| Open Innovation | Utilize manufacturing partners (OEM) to scale production |
| Niche Focus | Restrict deliveries to priority regions/customers first (e.g., North America only) |
| Diversification | Expand into related mass-market products (e.g., Model Y crossover) simultaneously |
| Retrenchment | Scale down Model 3 volume targets to protect financial stability |
| Maintain | Expand production gradually while prioritizing quality and profitability |

Table 7. Strategy menu for 5_model_3_mass_market.

**(3) Observed behavior**

To isolate firm-identity framing effects from semantic context variation, Fig. 10 reports pooled strategy choices under the neutral base context variant only.

![FR_deepdive_Qwen_Model3](final_results/plots/eval_deepdive_fr_framing_stacks__Qwen2.5-14B__5_model_3_mass_market.png)  
Fig. 10. FR deep-dive for 5_model_3_mass_market.

A paired permutation test on brand-masked token statistics (480 paired runs per cell) confirms that the lexical gap between Generic and Specific rationales is systematic, not a decoding artifact (global separation statistic with p ≈ 0.005 at both T=0.0 and T=0.7).

| Framing | Contexts | Rationale | Strategy choice |
| :---- | :---- | :---- | :---- |
| **Generic** | High demand, production pressure | Keywords (2-grams): manufacturing partners, scale production, quickly scale, utilizing manufacturing, production rapidly. | **Open Innovation** — Utilize manufacturing partners (OEM) to scale production. External partnerships as the main scaling lever.|
| **Specific** | High demand, production pressure | Keywords (2-grams): quality profitability, maintaining reputation, long term, crucial maintaining.  | **Maintain** — Expand production gradually while prioritizing quality and profitability. Internal cadence and safeguards over fastest-possible scale-out. |

Table 8. FR case summary

As shown in Fig. 10, Generic framing leads the model to favor Open Innovation (external partnerships for scaling). Under Specific (Tesla) framing, it shifts to Maintain (gradual expansion with quality focus). This divergence from the aggregate trend in Section 5.3 (where brand framing boosted Technology Leadership) illustrates that local effects can deviate substantially from averages—highlighting the need for scenario-level auditing.

---

**(B) CR Case: Semantic Cues Reallocate Strategy Mix (4_model_x_launch)**

**(1) Problem (scenario text)**: A company aims to enter the growing SUV market. However, a complex product design creates high production difficulty and quality risks, which could severely damage the brand's reputation despite a lack of direct competition.

**(2) Strategy menu (options).**

| Mapped strategy | Execution option |
| :---- | :---- |
| Technology Leadership | Launch a luxury SUV with innovative, complex features |
| Fast Follower | Introduce a simpler SUV quickly to capture demand before competitors |
| Open Innovation | Partner with suppliers/OEMs to co-develop SUV platform and reduce complexity |
| Niche Focus | Develop a standard, mid-priced SUV for a specific customer segment |
| Diversification | Expand into related vehicle categories (e.g., crossover, minivan) alongside SUV |
| Retrenchment | Reduce scope of SUV project, scale down features to cut risk |
| Maintain | Postpone the SUV launch, focus on stabilizing Model S production first |

Table 9. Strategy menu for 4_model_x_launch.

**(3) Observed behavior.**

semantic manipulation is implemented only through the context_variant axis—base, competitive_dynamics, count_fact, and opp_focus—on top of Generic vs. Specific firm-identity framing.

![CR_deepdive_Qwen_ModelX_framing](final_results/plots/eval_deepdive_cr_strategy_stacks_framing__Qwen2.5-14B__4_model_x_launch.png)  
Fig. 11. CR deep-dive for 4_model_x_launch.

The following plot reports the corresponding mean JSD from base for each semantic variant (CR cue strength), shown separately for $T=0.0$ and $T=0.7$.
![CR_deepdive_Qwen_ModelX_jsd](final_results/plots/eval_deepdive_cr_jsd_by_variant__Qwen2.5-14B__4_model_x_launch.png)  
Fig. 12. CR cue strength: mean JSD from base per semantic variant (T=0.0 vs T=0.7).

 Fig. 11 shows that Generic framing maintains Open Innovation as the dominant choice across all context variants. Under Specific framing, opp_focus uniquely activates Technology Leadership, while competitive_dynamics lifts Fast Follower. The JSD hierarchy in the accompanying plot (opp_focus > competitive_dynamics > count_fact) confirms asymmetric cue sensitivity: positive opportunities drive larger reallocations than unfavorable constraints.

---

**(C) DS Case: Context Load Shapes Stability (2_roadster_launch)**

**(1) Problem (scenario text)**: A company must manage conflicting goals of product quality and timely delivery during its initial product launch. With significant pre-orders already placed, the company faces severe cash flow issues and supply chain delays, jeopardizing brand trust and future investment if not handled correctly.

**(2) Strategy menu (options).**

| Mapped strategy | Execution option |
| :---- | :---- |
| Technology Leadership | Prioritize product quality and performance, accepting launch delays |
| Fast Follower | Accelerate launch to meet demand, accepting potential quality compromises |
| Open Innovation | Expand manufacturing partnerships to share risk |
| Niche Focus | Limit deliveries to early adopters first, delaying mass rollout |
| Diversification | Introduce parallel revenue streams (e.g., licensing tech, consulting) to ease cash flow |
| Retrenchment | Scale back launch volume until supply chain stabilizes |
| Maintain | Delay full-scale launch, focus on stabilizing operations and cash position |

Table 10. Strategy menu for 2_roadster_launch.

**(3) Observed behavior.**

![DS_deepdive_DeepSeek_Roadster_stacks_numctx](final_results/plots/eval_deepdive_ds_strategy_stacks_numcontext_framing__deepseek-llm-7b-chat__2_roadster_launch.png)  
Fig. 13. Strategy mix across Num Context tiers.

![DS_deepdive_DeepSeek_Roadster_entropy](final_results/plots/eval_deepdive_ds_entropy_numcontext_box__deepseek-llm-7b-chat__2_roadster_launch.png)  
Fig. 14. Repeat-level entropy across Num Context tiers.

As shown in Figs. 13–14, strategy mass shifts systematically as context blocks accumulate: near-vacuous prompts favor Open Innovation; partial context pivots toward Retrenchment/Niche Focus; full context yields a Fast Follower vs. Retrenchment split. This demonstrates that decision stability depends not only on temperature but also on the amount of contextual information provided. This variation across context loads explains why DS is low in this cell: even small changes in information quantity produce different strategic responses, making repeatability under identical conditions difficult to achieve.
