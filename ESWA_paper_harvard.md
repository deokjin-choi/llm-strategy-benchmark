**Auditing LLMs as Algorithmic Agents for R&D Strategy: A Scenario-Based Framework for Context and Framing Sensitivity**

**Abstract**  
Large Language Models (LLMs) are increasingly deployed as algorithmic agents in R&D and innovation management, yet their behavior as strategic decision-makers remains poorly understood. This study introduces a scenario-based benchmark for auditing how LLMs' strategic judgments respond to controlled variations in context and framing. Using historically grounded R&D scenarios, we systematically manipulate semantic context (opportunity, constraint, competition signals), brand framing (anonymous vs. a high-profile innovator), and numerical inputs across multiple LLMs and decoding temperatures.

Three main findings emerge. First, LLMs exhibit strong context sensitivity: opportunity framing shifts choices toward Technology Leadership, while unfavorable constraints trigger defensive positioning. Second, brand framing asymmetrically amplifies this pattern—identifying the firm as a high-profile innovator boosts proactive strategies while suppressing collaborative and niche approaches. Third, these framing effects persist in model-generated rationales and vary systematically across models and temperature settings, revealing that LLMs operate as conditional strategic agents rather than stable default reasoners. We conclude by outlining an audit protocol for evaluating robustness before deploying LLMs in strategic R&D contexts, contributing to AI governance and human-in-the-loop oversight.


**1\. Introduction**  
Large Language Models (LLMs) are increasingly used as algorithmic agents in R&D and innovation management—supporting technology assessment, competitive intelligence, and strategic planning under uncertainty. Organizations now consult LLMs not merely as information retrievers but as active contributors to strategic decision-making. Yet, despite this rapid adoption, empirical understanding of how these models behave as strategic agents remains critically limited.

When an LLM produces a strategic recommendation, it does not simply retrieve facts. It actively interprets the problem, weighs competing considerations, and generates a course of action—behaviors that qualify it as an algorithmic agent. However, unlike human agents whose reasoning can be probed and audited, LLMs offer no built-in transparency about what drives their judgments. Do they rely on stable strategic logic, or are their recommendations systematically shaped by how the problem is framed, which brand name appears in the prompt, or even the randomness setting of the decoder?

Current evaluative frameworks offer little answers to these questions. They primarily focus on accuracy, coherence, or task-level performance in isolated contexts (Chang et al., 2024). While informative, such metrics provide no insight into the stability of strategic judgments—how decisions shift when the same underlying dilemma is presented with different contextual emphasis, brand identity, or numerical inputs. These are precisely the factors that matter most in R&D strategy, where outcomes are highly sensitive to market signals, competitive framing, and narrative persuasion.

To address this gap, this study shifts the focus from "correctness" to "decision sensitivity". We introduce a scenario-based audit framework that systematically varies contextual and framing conditions to probe how LLMs adjust their strategic stances. Our central question is: How stable are LLMs' strategic judgments across different conditions and model configurations? By answering this question, we aim to characterize LLMs as conditional strategic agents and to provide practical guidance for auditing their behavior before deployment in high-stakes R&D environments.

**2\. Background and related work**

**2.1 LLMs as decision support systems**  
Recent research has increasingly embedded LLMs into real-world decision pipelines, particularly in operations and engineering management. Wang et al. (2025a) propose a multi-agent scheduling chain that leverages LLM-based agents to handle flexible job shop scheduling and real-time rescheduling, demonstrating substantial gains in scheduling efficiency under disruptions. Du et al. (2025) introduce LLM-MANUF, an integrated framework in which multiple fine-tuned LLMs generate alternative decision plans that are subsequently ranked and fused, highlighting that manufacturing decision quality depends as much on comparing and aggregating candidate strategies as on generating a single “best” answer. Beyond manufacturing, Wang et al. (2025b), Xiong et al. (2025), and Gindullina et al. (2026) combine spatiotemporal knowledge graphs, digital twins, and physics-informed neural networks with LLM agents to support berth allocation, aviation design, and epidemic forecasting, respectively. These studies illustrate how LLMs can act as planning and coordination engines for complex operational systems, yet they typically assess success in terms of task performance and system-level efficiency. They rarely examine how, within a fixed set of strategic options, LLMs distribute their choices or how sensitive those choices are to subtle changes in contextual description or framing.

**2.2 Reliability, hallucination, and safety in high-stakes decisions**  
Another line of work focuses on making LLM-supported decisions more reliable in high-stakes environments. Kong et al. (2026) present HaluGNN, which models question–answer pairs as token graphs and uses graph neural networks to detect hallucinated content, thereby improving decision security in domains where factual errors are costly. Heo et al. (2025) develop HaluCheck, a visualization and automation framework that decomposes model responses into sentence-level claims, retrieves external evidence, and highlights likely hallucinations through an interactive interface for expert systems. Przystalski et al. (2026) use stylometric features to distinguish human- from LLM-generated texts, informing governance around authorship, attribution, and authenticity. In the context of smart-city management, Antuley et al. (2026) propose SORA-ATMAS, an adaptive trust and governance framework that aligns multiple LLM agents with cross-domain policies and regulatory constraints. Collectively, these studies treat reliability and safety primarily at the level of factual correctness, anomaly detection, and policy compliance. What remains underexplored is whether LLMs are reliable as strategic decision agents—specifically, how their strategic choices and rationales shift under contextual and framing manipulations when the underlying problem remains fixed.

**2.3 Knowledge-grounded decision infrastructures**  
A third strand of literature builds knowledge-grounded infrastructures that connect LLMs to structured data and domain ontologies. Ojuri et al. (2025) propose a text-to-SQL framework in which LLMs and intelligent agents translate natural language queries into executable SQL, lowering the access barrier for non-technical users and reducing dependence on data specialists in organizational decision-making. Xiong et al. (2025) introduce DR-RAG, a domain-rule-based retrieval-augmented generation framework that ties aviation digital models to knowledge graphs, rule bases, and digital twins, turning complex product design into a loop of retrieval, generation, and simulation feedback. In financial decision-making, Sinha et al. (2026) present FinBloom, a knowledge-grounded financial agent that combines a domain-specialized LLM with real-time news and regulatory filings to answer dynamic financial queries. Alarcón Serrano et al. (2026) evaluate how well general-purpose LLMs can recognize taxonomic relationships in SNOMED CT, clarifying their role in biomedical knowledge-graph workflows that support clinical reasoning. These works substantially improve what LLMs know and how they access relevant information. Yet, even with stronger grounding, they do not systematically analyze how a model’s strategic choices over a fixed option set change when only the narrative emphasis, identity cues, or quantitative inputs are perturbed.

**2.4 Evaluation and Behavioral Profiling of LLMs**  
Evaluation- and reasoning-centric research has sought to understand how LLMs plan, reason, and exhibit preferences across tasks. Liu et al. (2026) propose CART, a traceable planning framework that decomposes goals into subtasks, tracks planning trajectories, and triggers replanning when conditions change, thereby improving the robustness of LLM-based agents in incomplete-information environments. Gjorgjevikj et al. (2025) introduce xLLMBench, a decision-centric benchmarking framework that uses multi-criteria decision-making to rank models along accuracy, scale, energy consumption, and other non-performance factors. Memduhoğlu et al. (2026) treat LLMs as “virtual experts” for multi-criteria spatial planning, comparing their analytic hierarchy process (AHP) weightings with those of human panels and documenting systematic biases in how models prioritize criteria for solar power plant site selection. Torres-Moreno and Hermosillo-Valadez (2026) propose a semantic knowledge abstraction framework that restructures premise–hypothesis relations in natural language inference to improve consistency and reveal latent semantic gaps in LLMs’ reasoning. These contributions collectively move beyond simple accuracy metrics toward richer assessments of reasoning, robustness, and multi-dimensional trade-offs. However, these evaluation frameworks typically focus on task-level performance or general reasoning consistency, leaving open how LLMs' strategic preferences shift under controlled perturbations of brand identity, contextual emphasis, or numerical signals.

**2.5 Cognitive Biases, Framing Effects, and Strategic Heuristics**

Human strategic judgments are susceptible to cognitive shortcuts that bypass deliberative reasoning (Kahneman, 2011). The *halo effect* occurs when a salient positive attribute—such as a firm's reputation—biases overall evaluation (Thorndike, 1920). *Anchoring* describes the tendency for an initial cue to shape subsequent judgments, even when that cue carries no objective decision-relevant information (Tversky & Kahneman, 1974). More broadly, Kahneman (2011) distinguishes fast, intuitive System 1 thinking from slow, analytical System 2 reasoning, with framing effects arising when System 1 dominates.

In LLM-based strategic decision-making, analogous patterns remain underexplored. The presence of a high-status brand identity or the selective emphasis of opportunity versus constraint information could trigger heuristic-like responses—leading models to prioritize narrative consistency over neutral evaluation of quantitative or factual signals. Nonetheless, existing benchmarks have not systematically characterized how such framing effects manifest in LLMs' categorical strategy choices, nor whether they persist in model-generated rationales.

**2.6 Strategic Archetypes in R&D and Innovation Management**  
To rigorously characterize the strategic behavior of LLMs, it is necessary to map their decision outputs onto established theoretical frameworks. This study adopts seven strategic archetypes rooted in the classical literature of competitive strategy and technological innovation. These options—Technology Leadership, Fast Follower, Open Innovation, Niche Focus, Diversification, Retrenchment, and Maintain—represent the fundamental trajectories firms pursue to navigate market transitions and resource constraints.

Specifically, the distinction between 'Technology Leadership' and 'Fast Follower' is grounded in the pioneering work on timing of entry and R&D intensity (Schilling, 2019). The concepts of 'Niche Focus' and 'Diversification' align with Porter’s generic strategies and the resource-based view of the firm (Porter, 1980; Barney, 1991). Furthermore, 'Open Innovation' reflects the modern shift toward collaborative R&D ecosystems (Chesbrough, 2003), while 'Retrenchment' and 'Maintain' represent critical defensive maneuvers under high environmental volatility (Miles et al., 1978). By constraining the LLM’s choice set to these validated archetypes, we transition from observing simple linguistic patterns to analyzing structural strategic reasoning. This methodological grounding ensures that the observed shifts in choice distributions are interpretable within the context of established management science.

**2.7 Research gap**  
Synthesizing these strands, prior work shows that LLMs can (i) drive automated planning in operational systems, (ii) support reliability through hallucination detection, (iii) operate within knowledge-grounded infrastructures, and (iv) be evaluated along general reasoning dimensions. Yet, taken together, this literature provides little systematic evidence on how LLMs behave as categorical strategic decision-makers when subjected to controlled perturbations of the same business dilemma.

In particular, three critical gaps remain. First, while existing studies focus on task performance, they rarely examine framing robustness—specifically, how arbitrary brand identities and narrative cues reshape strategy selection when the underlying economic facts are unchanged. Second, there is a lack of research on the interplay between qualitative context and quantitative data, leaving it unclear whether LLMs prioritize narrative consistency over numerical shifts in strategic settings. Finally, the operational reliability of these strategic choices across different decoding configurations (e.g., temperature) remains largely unexplored.

To address these limitations, this study reconceptualizes LLMs as conditional strategic agents whose decision-making logic is intrinsically linked to environmental framing. Rather than treating model outputs as static responses, we seek to characterize the dynamic boundaries of LLM-based reasoning by evaluating the stability and sensitivity of strategic choices under controlled perturbations. By bridging the gap between cognitive psychology and strategic management, this research provides a foundational framework for understanding how architectural biases in LLMs can be identified and managed in high-stakes corporate environments.

**3\. Methodology**  
This study adopts a scenario-based benchmarking framework to analyze LLMs make strategic decisions under varying contextual and framing conditions. As illustrated in Fig. 1, the methodology consists of five sequential stages. First, we define six historically grounded base strategic scenarios, each representing a fixed strategic dilemma. Across all scenarios, problem framing is systematically controlled through two conditions: a Generic framing (anonymous firm) and a Specific framing (explicitly identified as Tesla), which are applied consistently throughout the experiment. Second, each base scenario is expanded using four contextual variants that selectively modify competitive, factual, opportunity-related, or numerical information. Third, contextual information related to technology, market competition, policy, and financial conditions is automatically injected or removed while preserving the same core problem structure. Fourth, LLMs are required to select a single strategy from a predefined set of strategic options and to articulate a brief rationale for their choice. Finally, each scenario configuration is evaluated through repeated inference under different decoding settings, and the resulting outputs are aggregated and analyzed to assess decision sensitivity across conditions.

![Research Method](final_results/plots/research_method.PNG)

Fig 1\. Scenario-based benchmarking framework for LLM strategic decision-making  
**3.1 Base Strategic Scenarios and Problem Framing**  
To anchor the analysis in realistic and theoretically grounded decision contexts, we construct six base strategic scenarios based on key phases in Tesla’s historical development. These phases—Founder phase, Roadster launch, Model S, Model X, Model 3, and Energy Infrastructure—are derived from widely used case narratives and strategic patterns in the strategic management of technological innovation literature (Schilling, 2019). Each scenario represents a distinct but well-defined strategic dilemma commonly faced by technology-driven firms as they transition across stages of innovation, market expansion, and organizational scaling.

Importantly, each base scenario is designed to capture a fixed strategic dilemma. While contextual information may vary across experimental conditions, the core problem statement—namely, the fundamental strategic challenge faced by the firm at that stage—remains unchanged. This design ensures that observed differences in model behavior can be attributed to contextual and framing variations rather than to changes in the underlying decision problem itself.

Across all base scenarios and their contextual variants, problem framing is systematically controlled through two conditions. In the Generic framing, the firm is described as an anonymous company, allowing assessment of strategy selection without explicit brand cues. In the Specific framing, the firm is explicitly identified as Tesla, introducing firm identity and brand-related associations into the decision context. This framing distinction is applied consistently across all scenarios and variants, serving as a cross-cutting experimental dimension rather than a scenario-specific feature.

By grounding base scenarios in established innovation strategy frameworks while maintaining consistent problem structures and framing controls, this stage establishes a stable foundation for isolating how LLMs respond to contextual variation and firm identity cues in subsequent experimental steps.

**3.2 Contextual Scenario Variants**  
To examine how LLMs’ strategic judgments change under different informational emphases, we introduce controlled contextual variants to each base scenario. These variants do not alter the underlying strategic dilemma; instead, they selectively modify how the same problem is contextualized.

Four contextual variants are used. The competitive\_dynamics variant adds explicit signals of intensified competition to assess responses to competitive pressure. The count\_fact variant emphasizes unfavorable factual constraints, such as financial or operational limitations, to test defensive strategic adjustment. The opp\_focus variant highlights positive opportunity signals, probing whether opportunity framing amplifies proactive or leadership-oriented strategies. Finally, the randomized\_numbers variant applies numerical perturbations (±20%) to quantitative inputs without changing their semantic meaning, serving as a control to distinguish semantic sensitivity from numerical variation.

All variants are applied consistently across base scenarios and framing conditions, enabling direct comparison of how different types of contextual cues influence strategic decision sensitivity.

**3.3 Contextual Information Injection**  
To probe decision sensitivity, contextual information is injected into each scenario in a controlled and modular manner. Across all scenario instances, including the base case and its contextual variants, context blocks are defined to represent market conditions, technological constraints, financial status, policy support, competitive landscape, and customer response.

Each experimental run receives a subset of these context blocks, which are selectively added to or removed from the prompt while keeping the core problem statement unchanged. This design enables systematic combinations of contextual presence and absence, allowing us to observe how LLMs adjust strategic judgments when specific signals are emphasized, suppressed, or jointly presented.

Importantly, contextual injection operates independently of problem framing. Both Generic (anonymous firm) and Specific (Tesla-identified) problem formulations are paired with identical contextual information, ensuring that observed differences can be attributed to contextual cues and framing effects rather than changes in the underlying strategic dilemma.

This approach enables fine-grained analysis of how individual and combined contextual signals influence strategy selection, decision stability, and sensitivity to semantic framing.

**3.4 Strategy Selection Task Design**  
The seven strategic archetypes introduced in Section 2.6 are operationalized as a closed-choice decision task. For each scenario, LLMs are instructed to select exactly one strategy from the options defined in Table 1 and provide a brief rationale. No ranking or weighting is allowed, forcing the model to commit to a single strategic direction.

| Strategy Option | Definition |
| :--- | :--- |
| **Technology Leadership** | Pursue technological superiority at the cost of higher risk or delayed scalability. |
| **Fast Follower** | Rapidly scale by adopting proven technologies, emphasizing speed and cost efficiency. |
| **Open Innovation** | Collaborate with external partners to share risks, resources, and capabilities. |
| **Niche Focus** | Target a narrowly defined customer segment, emphasizing specialization. |
| **Diversification** | Expand into adjacent businesses to spread risk and create synergies. |
| **Retrenchment** | Reduce scope or delay expansion to preserve financial stability. |
| **Maintain** | Stabilize existing operations before pursuing further strategic moves. |

Table 1. Seven strategic archetypes and their definitions.


**3.5 Repeated Inference and Aggregation**  
To examine the stability and variability of LLM-based strategic judgments, this study adopts a repeated inference approach rather than relying on single-shot model outputs. Strategic decision-making is inherently non-deterministic, and a single response may not adequately represent a model’s underlying preference structure.

Repeated inference enables observation of how consistently a model favors particular strategies under identical conditions and how its choices disperse when multiple plausible interpretations exist. By incorporating both deterministic and stochastic decoding regimes, the framework captures a spectrum of decision behaviors ranging from most-probable judgments to probabilistic exploration.

Instead of treating each output as an isolated recommendation, individual inferences are aggregated into empirical strategy distributions. These distributions reflect relative selection tendencies across the predefined strategic options, allowing analysis of dominant patterns as well as decision uncertainty.

This distributional perspective is essential for assessing context and framing sensitivity. It supports comparative analysis across scenarios, structural examination of strategic environments, and sensitivity measurement relative to baseline conditions, thereby providing a more robust characterization of LLMs as strategic decision agents than single-point evaluations.

4\. Experimental Setup

**4.1 Model Selection**

We evaluate five open-weight, instruction-tuned language models that span distinct developer ecosystems and pretraining traditions: Meta Llama 3.1 8B Instruct, Mistral 7B Instruct v0.3, Qwen 2.5 14B Instruct, DeepSeek LLM 7B Chat, and Yi 1.5 9B Chat. This selection is motivated by three considerations. First, architectural and institutional diversity reduces the risk that findings reflect idiosyncrasies of a single model family or geographic training corpus; the panel mixes U.S., European, and Asia-based open models whose alignment and data mixes differ in ways that may plausibly affect strategic framing and narrative priors. Second, all models are openly available instruct variants that can be hosted on local inference stacks, which supports reproducible, high-volume repeated sampling under fixed prompts and decoding regimes—conditions that are difficult to guarantee with proprietary API-only frontiers whose internals may change without notice. Third, parameter counts are confined to a compact scale band (roughly 7B–14B parameters), which keeps compute and latency within a range typical of on-premise or dedicated-GPU deployments in corporate R\&D settings while still allowing meaningful variation in model capacity (e.g., 7B-class versus 14B-class) within a single experimental design. The goal is comparable results across models and evidence that speaks to open weights firms can run on their own hardware.

**4.2 Prompt Design and Bias Control**

Prompts followed a fixed template with five components: (1) a strategic dilemma statement, (2) optional context blocks, (3) candidate execution options, (4) mapping from options to strategic archetypes, and (5) a JSON output schema. Models were required to select exactly one strategy and provide a brief rationale, preventing free-form outputs.

Firm-identity framing was isolated by keeping all content identical except the problem statement: Generic framing described an anonymous firm; Specific framing explicitly named Tesla. Contextual information, injected modularly as described in §3.3, was held identical across framing conditions to ensure that any difference in strategy distributions arises solely from the brand cue.

To mitigate option-order bias, alphabetic labels (A–G) were rotated across scenarios so that no archetype (e.g., Technology Leadership) was consistently tied to the same label. Responses were parsed into a structured JSON object; unparseable responses or selections outside the seven archetypes were excluded from normalized distributions (compliance reported in Appendix A).

**4.3 Inference Settings and Repetition Protocol**

Each experimental condition was repeated 30 times per model at two decoding temperatures. Temperature 0.0 produces deterministic responses, establishing a baseline of each model's stable strategic preferences under identical inputs. Temperature 0.7 introduces controlled randomness to assess whether the observed patterns remain stable when responses are allowed to vary, and to compare models' sensitivity to decoding temperature. Maximum output length was fixed at 256 tokens.

**5\. Key Findings**  
This section synthesizes the central empirical patterns observed across repeated runs. We first characterize how strategy selections shift across contextual variants and whether those shifts form distinct structural regimes (distributional profiles and PCA separation). We then quantify decision uncertainty and rank stability using entropy, Jensen–Shannon divergence, and Spearman correlation, and separately examine the pure effect of brand framing on strategy reallocation. Finally, we test whether brand exposure alters the rationale framing even when the selected strategy is held constant, and we extend the analysis to model-level behavioral profiling and temperature robustness. (Instruction-following compliance and categorical validity checks are reported in Appendix A.)

**5.1 Strategy Distribution Across Contextual Variants**  
Fig. 2 shows that the distribution of selected strategies varies across repeated runs, indicating that LLMs do not rely on a single dominant default strategy but respond systematically to contextual framing. In the base scenario, conservative positioning such as Niche Focus is most frequent, while Technology Leadership remains secondary. When opportunity signals are emphasized (opp\_focus), leadership-oriented strategies surge and become dominant. In contrast, unfavorable factual constraints (count\_fact) lead to defensive repositioning toward niche or follower strategies. Notably, numerical perturbations alone (randomized\_numbers) produce minimal change, indicating that semantic context rather than numeric variation drives strategic reorientation. This pattern carries important implications for strategic decision-making. In many real-world R\&D and innovation contexts, proportional numerical changes—such as shifts in cost, market size, or resource availability—often serve as triggers for strategic adjustment. However, the observed stability under numerical perturbations suggests that LLMs may treat moderate quantitative variation as secondary when the overall semantic structure of the scenario remains unchanged. This does not imply an inability to process numbers, but it suggests that, within narrative-based prompts, quantitative signals may exert less influence on categorical strategic choice than semantic framing. For practitioners, this indicates that LLM-generated recommendations in magnitude-sensitive environments should be interpreted with attention to how numerical thresholds are presented and whether they meaningfully alter the underlying decision narrative.

![Strategy_Ratio](final_results/plots/eval_eda_Strategy_Ratio_by_Scenario.png)  
Fig. 2\. Strategy distribution across contextual scenario variants.

**5.2 Structural Separation and Statistical Dynamics of Strategic Contexts**  
Fig 3\. examines whether these distributional shifts reflect meaningful structural differences. Principal component analysis reveals clear separation between opportunity-focused and unfavorable scenarios, while the base and randomized-number variants cluster closely together. This suggests that LLMs internally distinguish qualitatively different strategic environments rather than responding to random noise or minor input changes.  
![Strategy_Ratio_PCA](final_results/plots/eval_eda_PCA_of_Strategy_Ratios_2D_Vectorized_Analysis.png)  
Fig. 3\. PCA-based structural separation of strategy distributions across scenarios.


To further quantify these shifts and the underlying decision uncertainty, we calculated Shannon entropy, Jensen-Shannon Divergence (JSD), and Spearman rank correlation for each scenario relative to the base (Table 2).

| Scenario | entropy | jsd\_from\_base | spearman\_vs\_base |
| :---: | :---: | :---: | :---: |
| base | 1.8422 | 0 | 1 |
| competitive\_dynamics | 1.8211 | 0.0107 | 0.8214 |
| count\_fact | 1.7818 | 0.0081 | 0.6429 |
| opp\_focus | 1.7030 | 0.0471 | 0.6786 |
| randomized\_numbers | 1.8328 | 0.0002 | 1 |

Table 2\. Quantitative metrics for strategic decision consistency and shift.

The quantitative analysis provides several key insights into the models' decision-making logic:

* Context-Driven Certainty (Entropy): The opp\_focus scenario exhibited the lowest entropy (1.7030), compared to the base scenario (1.8422). This suggests that while LLMs are generally cautious (high entropy) in ambiguous settings, they become significantly more "confident" and decisive when presented with growth-oriented opportunities.

* Sensitivity to Narrative vs. Magnitude (JSD & Spearman): The jsd\_from\_base for opp\_focus (0.0471) was nearly six times higher than that of count\_fact (0.0081), indicating that LLMs are disproportionately sensitive to optimistic framing. Conversely, the randomized\_numbers scenario showed a perfect Spearman correlation (1.0000) and near-zero JSD (0.0002) relative to the base. This statistically confirms "numerical insensitivity," where the models' strategic ranking remains frozen despite quantitative fluctuations.

These metrics validate the PCA results: the structural separation observed in Fig. 3 is not merely visual but is rooted in distinct changes in decision certainty and rank stability across different semantic contexts.

**5.3 Strategic Reallocation Under Brand Framing**  
To isolate the pure effect of brand framing, we compare strategy selections under Specific (Tesla-identified) versus Generic (anonymous) framing while holding all other experimental conditions strictly identical—same model, temperature, scenario, context variant, and number of context blocks. For each matched pair, we compute Δp = p(Specific) − p(Generic), then average across all condition pairs.

Fig. 4 shows the resulting asymmetric reallocation. Technology Leadership increases by +9.5 percentage points—the largest effect by a substantial margin. Open Innovation and Niche Focus decline by -5.7 pp and -4.7 pp, respectively. Fast Follower shows a modest increase (+2.0 pp), while Diversification, Retrenchment, and Maintain remain largely unchanged (|Δp| < 1.5 pp).

![Strategy_Reallocation_By_Frame](final_results/plots/eval_fr_directionality_bars__ALL.png)  
Fig. 4\. Strategic reallocation under brand framing.

This pattern reveals two characteristics of how brand framing operates. First, the effect is concentrated and asymmetric: the 9.5 pp gain for Technology Leadership nearly offsets the combined losses of Open Innovation and Niche Focus (10.4 pp), suggesting a winner-take-most reallocation dynamic rather than a diffuse shift across multiple strategies. Second, brand framing does not simply "nudge" preferences uniformly; it systematically downweights collaborative and focused strategies (Open Innovation, Niche Focus) while favoring a singular, high-visibility archetype (Technology Leadership).

**5.4 Rationale Framing Shift Under Brand Exposure**

This section examines whether explanatory rationales shift under brand exposure even when the strategic choice remains identical. We constructed matched pairs holding all experimental conditions constant—model, temperature, scenario, context variant, number of context blocks, and chosen strategy—while varying only firm identification (Specific vs. Generic). Firm-referential tokens (e.g., Tesla, company) and numeric terms were masked to minimize superficial lexical artifacts. Lexical divergence was quantified using log-odds ratios, with statistical significance assessed via paired permutation tests(146,429 samples; 59,819 terms) and Benjamini–Hochberg false discovery rate correction. Both global separation (mean absolute log-odds difference) and keyword-level effects were evaluated.

The analysis confirms significant lexical divergence between conditions. The observed global separation (mean |Δlog-odds| = 0.323) exceeded the permutation baseline (0.223, p = 0.003), indicating systematic differences beyond random variation.

Table 3 summarizes the semantic structure of this divergence. Specific (Tesla) rationales were characterized by high-agency, vision-oriented language—mission lead, leader capturing, world transition—emphasizing proactive market positioning. Generic rationales were characterized by constraint-oriented, operational language—rushed quality, delaying significant, make feasible—emphasizing execution risks and resource management.

| Analysis Dimension | Specific (Visionary & Strategic Expansion) | Generic (Operational & Constraint Management) |
| :---: | ----- | ----- |
| **Leadership & Identity** | mission lead, identity leader (Focus on mission-driven leadership and brand identity) | trust balanced, goals standards (Focus on maintaining trust and adhering to industry standards) |
| **Technological Positioning** | goal technological, position platform (Emphasis on technological objectives and platform dominance) | enables integration, optimization scale (Emphasis on systems integration and operational efficiency) |
| **Market Dynamics** | world transition, leader capturing (Highlighting global transformation and proactive market capture) | rushed quality, delaying significant (Highlighting quality risks and significant project delays) |
| **Execution & Feasibility** | prestige demonstrate, mission showcase (Demonstrating institutional prestige and showcasing core missions) | funding invest, make feasible (Addressing capital investment and practical feasibility) |

Table 3. Semantic framing differences in rationale keywords

These findings indicate that LLM-generated strategic rationales are not neutral analytical outputs but reflect framing-consistent narrative patterns. Notably, the effect persists after masking brand-referential terms, suggesting that firm identity cues trigger distinct explanatory styles rather than mere lexical priming.

**5.5 Model-Level Behavioral Profiling and Temperature Robustness**  
This section extends the scenario-level findings by profiling model-specific behavioral signatures under the same strategic benchmark. Rather than relying on pooled trends, we compare models as distinct strategic reasoners and evaluate whether their behavioral profiles remain stable across decoding temperatures.

To support this comparative analysis, we formalize the quantitative framework into five specific profiling axes. These axes transition from general descriptive metrics to formal behavioral indicators, allowing for a multidimensional assessment of how each model navigates the trade-offs between contextual responsiveness and framing stability.

5.5.1 Profiling Axes  
To quantify model behavior, we define a strategy distribution $P(m, \tau, s, n, v, \phi)$ for a given model $m$, temperature $\tau$, scenario $s$, context load $n$ (Num Context), context variant $v$, and framing type $\phi \in \{\text{Generic, Specific}\}$. For decision-level metrics (1–4), we employ the Jensen–Shannon Divergence ($JSD$) to measure the statistical distance between distributions. For the explanation-level metric (5), we measure lexical divergence in the generated rationales.

1\. Framing Robustness (FR)  
FR measures the model's ability to maintain a consistent strategy regardless of arbitrary branding changes (Generic vs. Specific) under identical underlying conditions.

$$FR(m, \tau) = 1 - \mathbb{E}_{s,n,v} \left[ JSD\left( P(m, \tau, s, n, v, \text{Generic}), P(m, \tau, s, n, v, \text{Specific}) \right) \right]$$

* Interpretation: A higher FR indicates that the model's strategic choice is invariant to framing manipulations, focusing on the core problem rather than brand-level biases.

2\. Context Responsiveness (CR)  
CR evaluates how effectively a model updates its strategic distribution when provided with high-value semantic context (e.g., competitive dynamics, factual constraints, or opportunity focus) compared to a baseline scenario.

$$CR(m,\tau)=E_{s,n,\phi,\,v\in V_{sem}}\Big[JSD\big(P(m,\tau,s,n,\mathrm{Base},\phi),P(m,\tau,s,n,v,\phi)\big)\Big]$$
where $V_{sem}$ is the semantic-variant set `{competitive_dynamics, count_fact, opp_focus}`.

* Interpretation: A higher CR reflects the model's "strategic intelligence"—its capacity to parse and integrate task-relevant nuances into its decision-making process.

3\. Numerical Sensitivity (NS)  
NS quantifies the degree to which a model's decisions are driven by quantitative data. It measures the distributional shift when numerical inputs are intentionally perturbed (Randomized condition) relative to the baseline.

$$NS(m, \tau) = \mathbb{E}_{s,n,\phi} \left[ JSD\left( P(m, \tau, s, n, \text{Base}, \phi), P(m, \tau, s, n, \text{Randomized}, \phi) \right) \right]$$

* Interpretation: A higher NS indicates that the model is sensitive to quantitative shifts, suggesting a data-driven approach rather than relying solely on qualitative narratives.

4.Decision Stability (DS)  
DS measures how predictable a model’s strategic choices are across repeated runs under fixed conditions. In the current implementation, DS is defined using the concentration of the empirical choice distribution across repeats (normalized entropy).  
Let $\mathcal{A}$ be the set of strategies and let a fixed condition be $c=(s,n,v,\phi)$, where $s$ is the scenario, $n$ is context load, $v$ is the context variant, and $\phi\in\{\text{Generic},\text{Specific}\}$ is framing. For model $m$ at temperature $\tau$, define the empirical distribution over repeats:

$$p_{c}^{m,\tau}(a)=\frac{1}{R}\sum_{r=1}^{R}\mathbf{1}\big[\text{strategy}_{r}=a\big],\quad a\in\mathcal{A}.$$

Then

$$DS(m,\tau)=\mathbb{E}_{c}\left[1-\frac{H\left(p_{c}^{m,\tau}\right)}{\log_{2}|\mathcal{A}|}\right], \qquad H(p)=-\sum_{a\in\mathcal{A}}p(a)\log_{2}p(a).$$

* Interpretation: A higher DS indicates that repeated runs under the same condition concentrate on fewer strategies (higher predictability). DS approaches 1 when the model consistently selects the same strategy, and approaches 0 when choices are spread uniformly across strategies.

5\. Explanatory Framing Invariance (EFI)  
EFI measures the stylistic and lexical stability of the model's rationales when the chosen strategy is held identical across framings. We align rationale pairs on $(s,m,\tau,n,a)$: scenario $s$, model $m$, temperature $\tau$, context load $n$, and strategy label $a$ (the Standard Mapping in the logs), so that only the firm-identity framing differs within a pair. The Generic- and Specific-side texts are brand-masked and tokenized into an $n$-gram vocabulary $\mathcal{V}$ (bigrams in the reported runs). Let $c_w^{\phi}$ be the pooled count of term $w$ across all aligned pairs in frame $\phi \in \{\mathrm{Generic},\mathrm{Specific}\}$. Using a smoothed log-odds construction (pooled counts with a term-wise backing-off denominator shared across $w$), we define a vocabulary-level contrast $\Delta_w$ between the two frames. The raw divergence and the invariance score are then

$$EFD_{raw}(m, \tau) = \frac{1}{|\mathcal{V}|}\sum_{w\in\mathcal{V}} \left|\Delta_{w}\right|, \qquad EFI(m, \tau) = \frac{1}{1 + EFD_{raw}(m, \tau)}$$

* Interpretation: A higher EFI signals that the model's underlying logic remains consistent across different frames, avoiding "post-hoc" justifications tailored to specific branding.

5.5.2 Cross-Model and Cross-Temperature Comparison  
We evaluate each model’s "strategic fingerprint" by calculating the five-axis scores at both $T=0.0$ and $T=0.7$. This comparison reveals whether a model’s behavior is rooted in its structural reasoning logic or merely a byproduct of stochastic decoding.

The comparative analysis seeks to answer the following three core questions regarding strategic reliability:

1. Framing & Logical Integrity (Robustness): Which models are "brand-blind" (High FR) and maintain a "consistent narrative" (High EFI)?  
   * *Objective:* To identify models that remain objective and do not change their decision or justification simply because the company name or branding context changes.  
2. Information Processing Intelligence (Agility): Which models are "context-smart" (High CR) while remaining "data-responsive" (High NS)?  
   * *Objective:* To find models that effectively pivot their strategy when new market intelligence is provided, while ensuring they are sensitive to changes in critical numerical data rather than just following a generic story.  
3. Operational Reliability (Consistency): Which models offer a "predictable output" (High DS) across different runs and randomness levels?  
   * *Objective:* To determine which models are reliable enough for production environments, where strategic advice must remain stable even when the decoding temperature increases.

Table 4 lists the five-axis scores for each model $m$ and temperature $\tau$. By construction, FR, CR, NS, and DS lie in $[0,1]$; EFI maps $EFD_{\mathrm{raw}}\ge 0$ to $(0,1]$.

| Model | $T$ | FR | CR | NS | DS | EFI |
| :---- | :--: | :----: | :----: | :----: | :----: | :----: |
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

Table 4\. model-level profiling scores (five axes)

5.5.3 Interpretation and Practical Implications

The five-axis profiling reveals that LLM behavior in strategic R&D settings is not monolithic but follows distinct patterns. To enable visual comparison — given that CR and NS exhibit compressed empirical spreads — the radar charts apply a display‑only per‑axis min‑max scaling over the ten (m,τ) rows. (Raw absolute scores are in Table 4.)

By comparing individual footprints against the aggregate Average Profile, we identify three key dimensions of strategic behavior:

![Model_PROFILE](final_results/plots/eval_model_profile_radar.png)

Fig. 5 Profiling radar (five-axis): one polar panel per model; solid vs. dashed curves are $T=0.0$ vs. $0.7$ on spokes built from Table 4 after global per-axis min–max.

The min–max scaled radar profiles indicate that decoding temperature meaningfully changes LLM behavior. Rather than operating as fixed systems, the evaluated models exhibit different strategic tendencies depending on sampling conditions. Three implications are especially notable.

#### (1) Behavioral Shift Across Temperature

The radar charts suggest a fundamental topological shift in model behavior as decoding temperature increases from $T=0.0$ to $T=0.7$.

At $T=0.0$, models generally exhibit elevated DS, NS, and CR, ensuring sharp data alignment and predictive stability. This suggests stronger consistency and tighter adherence to dominant probability paths. Such behavior can be beneficial when stable outputs are preferred, but it may also preserve latent framing tendencies inherited from training patterns.

At $T=0.7$, many models show higher FR and EFI, while DS often declines. This implies that moderate stochasticity may help models move beyond narrow response preferences and rely more on broader reasoning structures. In practice, higher temperature can improve logical flexibility and reduce sensitivity to superficial framing cues, although it may lower deterministic consistency.

Overall, the results suggest a trade-off between precision-oriented stability and objectivity-oriented flexibility.

#### (2) Distinct Model Personas

The models do not respond uniformly to temperature changes. Instead, they display distinct behavioral profiles. The vignettes below use **the same per-axis min–max scaling as the radar** (each spoke is $[0,1]$ **relative to the ten** $(m,\tau)$ **rows in Table 4**)

1. **Stable Functional Type**  
   Qwen2.5-14B keeps a **high DS** spoke at both temperatures (**≈0.91 → 0.79** on that scaled axis), i.e., strong run-to-run concentration *within this panel*. **FR** and **EFI** remain **inward** on the radar—e.g., **FR** near the hub at $T=0.0$—so the model is less “brand-blind” and less rationale-invariant than some peers in this display. The footprint matches **functional persistence** over maximal framing or stylistic invariance.

2. **Precision-Sensitive Type**  
   DeepSeek-LLM-7B-Chat at $T=0.0$ pushes **FR**, **NS**, and **DS** toward the outer ring (**FR ≈0.95**, **NS = 1.00**, **DS ≈0.94** on the scaled spokes). At $T=0.7$, **DS** and **CR** **collapse toward 0** while **EFI** moves **outward (≈0.99)**—the dashed trace **deflates** on repeatability and context-response axes. The persona is a **deterministic** specialist more than a stable stochastic partner.

3. **Adaptive Resilient Type**  
   Meta-Llama-3.1-8B reaches the **CR** spoke maximum at $T=0.0$ (**1.00** under this normalization)—the strongest semantic redistribution in the panel when sampling is greedy. **EFI** is **highest at $T=0.7$** (**1.00** scaled), with **FR** also **high (≈0.92)**, consistent with **situational responsiveness** paired with **stronger cross-frame rationale consistency** when temperature is raised—in this **relative** view only.

These differences imply that model selection should consider not only average performance, but also response stability under different decoding environments.

5.5.4 Scenario-Resolved Behavior: Localizing FR, CR, and DS Across the Experimental Grid

Sections 5.5.2–5.5.3 compared models using aggregate profiling scores. While useful for benchmarking, aggregate averages do not reveal where framing sensitivity, context adaptation, or instability emerge within the experimental grid. To address this limitation, we examine scenario-level behavior for three choice-level metrics: FR, CR, and DS.(NS and EFI are omitted here—Section 5.5 shows NS effects are minimal and EFI focuses on rationale text, not choice-level heterogeneity.)

Fig. 6 and Fig. 7 summarize scenario × model heatmaps at $T=0.0$ and $T=0.7$. Each panel is independently min–max scaled to $[0,1]$.

![Scenario_model_overview_T0](final_results/plots/eval_scenario_model_overview_FR_CR_DS__T0.png)  
Fig. 6. Scenario × model heatmaps for $FR_{\mathrm{scenario}}$, $CR_{\mathrm{scenario}}$, and $DS_{\mathrm{scenario}}$ at $T=0.0$.

![Scenario_model_overview_T07](final_results/plots/eval_scenario_model_overview_FR_CR_DS__T0.7.png)  
Fig. 7. Scenario × model heatmaps for $FR_{\mathrm{scenario}}$, $CR_{\mathrm{scenario}}$, and $DS_{\mathrm{scenario}}$ at $T=0.7$.

**(1) Heterogeneous local behavior.**  
FR and DS remain relatively high across most cells, while CR is sparse—only a few scenario–model pairs (e.g., 4_model_x_launch) show strong context-driven redistribution. CR's concentration in specific cells (not uniform across all cells) reveals that context responsiveness is situation-dependent. This conditional dependence supports treating models as conditional decision agents: the three axes do not co-move; aggregate radar scores therefore pool qualitatively different local behaviors. Consequently, a model that appears "moderately context-sensitive" on aggregate may in fact be highly responsive in a few critical scenarios and entirely unresponsive in others—a distinction that only scenario-resolved analysis can reveal.

**(2) Temperature effects.**  
The relative spatial structure of FR and CR is broadly preserved across temperatures (e.g., Qwen2.5-14B's FR dip on 5_model_3_mass_market persists, and its CR hotspot on 4_model_x_launch remains equally prominent), whereas DS weakens at T=0.7. This indicates that stochastic decoding primarily erodes run-to-run repeatability without fundamentally altering which scenarios trigger framing sensitivity or context-driven redistribution.

**Priority cells for case analysis.**

The patterns above motivate a closer look at specific scenario–model pairs where each metric's characteristic behavior is most pronounced. For FR and DS, we select the cells with the lowest scaled scores—where framing robustness or decision stability fails most visibly. For CR, we select the cell with the highest scaled score—where semantic context most strongly reallocates strategy choices. Table 5 lists the resulting three priority cells, averaging across T=0.0 and T=0.7 to reflect both decoding regimes.

| Metric | Model | Scenario | Mean scaled value (Figs. 5–6, avg.) |
| :---- | :---- | :---- | :---- |
| $FR_{\mathrm{scenario}}$ (min) | Qwen2.5-14B-Instruct | 5_model_3_mass_market | 0.000 |
| $CR_{\mathrm{scenario}}$ (max) | Qwen2.5-14B-Instruct | 4_model_x_launch | 0.935 |
| $DS_{\mathrm{scenario}}$ (min) | deepseek-llm-7b-chat | 2_roadster_launch | 0.184 |

Table 5. Priority scenario–model cells.

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

Table 6. Strategy menu for 5_model_3_mass_market.

**(3) Observed behavior**

To isolate firm-identity framing effects from semantic context variation, Fig. 8 reports pooled strategy choices under the neutral base context variant only.

![FR_deepdive_Qwen_Model3](final_results/plots/eval_deepdive_fr_framing_stacks__Qwen2.5-14B__5_model_3_mass_market.png)  
Fig. 8. FR deep-dive for 5_model_3_mass_market.

A paired permutation test on brand-masked token statistics (480 paired runs per cell) confirms that the lexical gap between Generic and Specific rationales is systematic, not a decoding artifact (global separation statistic with p ≈ 0.005 at both T=0.0 and T=0.7).

| Framing | Contexts | Rationale | Strategy choice |
| :---- | :---- | :---- | :---- |
| **Generic** | High demand, production pressure | Keywords (2-grams): manufacturing partners, scale production, quickly scale, utilizing manufacturing, production rapidly. | **Open Innovation** — Utilize manufacturing partners (OEM) to scale production. External partnerships as the main scaling lever.|
| **Specific** | High demand, production pressure | Keywords (2-grams): quality profitability, maintaining reputation, long term, crucial maintaining.  | **Maintain** — Expand production gradually while prioritizing quality and profitability. Internal cadence and safeguards over fastest-possible scale-out. |

Table 7. FR case summary

As shown in Fig. 8, Generic framing leads the model to favor Open Innovation (external partnerships for scaling). Under Specific (Tesla) framing, it shifts to Maintain (gradual expansion with quality focus). This divergence from the aggregate trend in Section 5.3 (where brand framing boosted Technology Leadership) illustrates that local effects can deviate substantially from averages—highlighting the need for scenario-level auditing.

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

Table 8. Strategy menu for 4_model_x_launch.

**(3) Observed behavior.**

semantic manipulation is implemented only through the context_variant axis—base, competitive_dynamics, count_fact, and opp_focus—on top of Generic vs. Specific firm-identity framing.

![CR_deepdive_Qwen_ModelX_framing](final_results/plots/eval_deepdive_cr_strategy_stacks_framing__Qwen2.5-14B__4_model_x_launch.png)  
Fig. 9. CR deep-dive for 4_model_x_launch.

The following plot reports the corresponding mean JSD from base for each semantic variant (CR cue strength), shown separately for $T=0.0$ and $T=0.7$.
![CR_deepdive_Qwen_ModelX_jsd](final_results/plots/eval_deepdive_cr_jsd_by_variant__Qwen2.5-14B__4_model_x_launch.png)  
Fig. 10. CR cue strength: mean JSD from base per semantic variant (T=0.0 vs T=0.7).

 Fig. 9 shows that Generic framing maintains Open Innovation as the dominant choice across all context variants. Under Specific framing, opp_focus uniquely activates Technology Leadership, while competitive_dynamics lifts Fast Follower. The JSD hierarchy in the accompanying plot (opp_focus > competitive_dynamics > count_fact) confirms asymmetric cue sensitivity: positive opportunities drive larger reallocations than unfavorable constraints.

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

Table 9. Strategy menu for 2_roadster_launch.

**(3) Observed behavior.**

![DS_deepdive_DeepSeek_Roadster_stacks_numctx](final_results/plots/eval_deepdive_ds_strategy_stacks_numcontext_framing__deepseek-llm-7b-chat__2_roadster_launch.png)  
Fig. 11. Strategy mix across Num Context tiers.

![DS_deepdive_DeepSeek_Roadster_entropy](final_results/plots/eval_deepdive_ds_entropy_numcontext_box__deepseek-llm-7b-chat__2_roadster_launch.png)  
Fig. 12. Repeat-level entropy across Num Context tiers.

As shown in Figs. 11–12, strategy mass shifts systematically as context blocks accumulate: near-vacuous prompts favor Open Innovation; partial context pivots toward Retrenchment/Niche Focus; full context yields a Fast Follower vs. Retrenchment split. This demonstrates that decision stability depends not only on temperature but also on the amount of contextual information provided. This variation across context loads explains why DS is low in this cell: even small changes in information quantity produce different strategic responses, making repeatability under identical conditions difficult to achieve.

**6. Discussion**

**6.1 Summary of Findings**

This study introduced a scenario-based audit framework to examine how LLMs respond to contextual and framing variations in R&D strategic decisions. Three main findings emerged. First, LLMs exhibit strong context sensitivity: opportunity framing (opp_focus) shifted strategy distributions toward Technology Leadership (lowest entropy = 1.7030), while unfavorable factual constraints (count_fact) triggered defensive repositioning toward niche or follower strategies. Numerical perturbations alone produced minimal change (JSD = 0.0002, Spearman = 1.00), confirming that semantic context—not numeric variation—drives strategic reorientation (Section 5.1–5.2). Second, brand framing asymmetrically amplified these patterns: identifying the firm as Tesla increased Technology Leadership by 9.5 percentage points while suppressing Open Innovation and Niche Focus by 5.7 pp and 4.7 pp, respectively (Section 5.3). Third, these framing effects persisted in model-generated rationales and varied systematically across models and temperature settings, with temperature 0.7 generally improving framing robustness (FR, EFI) at the cost of decision stability (DS) (Section 5.4–5.5). Collectively, these results characterize LLMs as conditional strategic agents rather than stable default reasoners.

**6.2 Interpreting Context Sensitivity: Asymmetric Responsiveness, Competition, and Numerical Insensitivity**

The observed asymmetry in context sensitivity is striking. Opportunity framing (opp_focus) produced nearly six times larger distributional shift (JSD = 0.0471) than constraint framing (count_fact, JSD = 0.0081). Competition signals (competitive_dynamics) fell in between (JSD = 0.0107, Spearman = 0.82). This hierarchy—opportunity > competition > constraint—reveals that LLMs are most responsive to positive market signals, moderately responsive to competitive threats, and least responsive to internal factual constraints.

Why this pattern? One explanation lies in training data priors. LLMs are trained on vast corpora of human-generated text, which tends to exhibit optimistic bias—narratives about opportunities, growth, and breakthroughs are overrepresented relative to balanced discussions of constraints and risks. When prompted with opportunity-focused language, the model activates rich pre-trained associations, shifting toward proactive, leadership-oriented strategies. Constraint-focused language, by contrast, has weaker or more diffuse associations. This asymmetry mirrors well-documented human cognitive biases (Kahneman, 2011), suggesting that LLMs may not only replicate but potentially amplify optimistic overconfidence.

The near-invariance under numerical perturbations (randomized_numbers: JSD = 0.0002, Spearman = 1.00) is equally telling. Moderate quantitative changes—±20% shifts in cost, market size, or resource availability—produced virtually no strategic reallocation. This suggests that within narrative-based prompts, quantitative signals exert less influence on categorical strategic choice than semantic framing. For R&D managers, this implies that LLM-generated recommendations in magnitude-sensitive environments (e.g., budget thresholds, capacity constraints, timing windows) should not be taken at face value. Numerical thresholds must be made explicitly decision-relevant (e.g., "cost exceeds feasible range") rather than presented as neutral figures. Practitioners should therefore verify that internal risk factors (e.g., cash flow, technical debt) are not systematically discounted relative to external market narratives.

**6.3 Interpreting Brand Framing Effects: Halo Effects and Associative Anchoring**

The finding that brand framing (Tesla vs. anonymous) reallocates strategy choices from Open Innovation and Niche Focus toward Technology Leadership (+9.5 pp) reveals two interconnected cognitive mechanisms.

First, the *halo effect* (Thorndike, 1920) occurs when a salient positive attribute—Tesla's reputation for innovation and market disruption—biases overall strategic evaluation. The model does not evaluate the situation de novo; it retrieves pre-trained associations that paint Tesla as a "technology pioneer" and applies this narrative to the strategic choice, systematically downweighting collaborative (Open Innovation) or focused (Niche Focus) alternatives.

Second, *associative anchoring* (Tversky & Kahneman, 1974) amplifies this effect. The brand name serves as an anchor that shapes subsequent reasoning. The scenario-resolved analysis (Section 5.5.4) provides illustrative evidence: in the CR-max cell (Qwen2.5-14B on the Model X launch scenario), opp_focus uniquely activated Technology Leadership under Specific framing, while constraint-focused variants produced smaller shifts. This suggests that brand framing interacts with contextual cues, though the specific interaction patterns require further investigation across the full experimental grid.

**6.4 Algorithmic Bounded Rationality: A Conceptual Framework**

These findings motivate a broader conceptual contribution: *algorithmic bounded rationality*. Simon (1957) proposed that human decision-making is bounded by cognitive limitations—limited memory, attention, and processing capacity—leading to reliance on heuristics and susceptibility to framing. LLMs, while not subject to the same cognitive constraints, exhibit analogous sensitivity patterns due to their training on human-generated text. They inherit not only factual knowledge but also the heuristic patterns and narrative biases embedded in human language.

We define algorithmic bounded rationality as the characteristic sensitivity of LLM-based strategic agents to problem framing, identity cues, and narrative emphasis, even when underlying decision-relevant facts remain constant. This reframes observed framing effects not as model "errors" or "hallucinations" to be eliminated, but as predictable consequences of how LLMs represent and retrieve strategic knowledge from pre-trained distributions. The governance implication is profound: mitigating these effects requires not better models alone, but better *audits* of how models respond to the framing choices of their human users.

**6.5 Practical Implications for R&D Management**

For organizations deploying LLMs in R&D strategy workflows, our results support four concrete practices.

*Anonymized prompting for bias detection.* Because brand framing substantially reallocates strategy choices (Section 5.3), practitioners should run side-by-side Generic (anonymous) and Specific (identified) prompts whenever firm identity is present. If the two conditions produce systematically different recommendations (e.g., Technology Leadership under Specific vs. Open Innovation under Generic), the LLM is likely activating pre-trained brand associations rather than evaluating neutral facts.

*Context variant stress-testing.* Given the hierarchy of responsiveness (opportunity > competition > constraint), teams should not rely solely on optimistic framings. We recommend stress-testing strategic recommendations under all three context variants: upside (opp_focus), competitive (competitive_dynamics), and downside (count_fact). If the model heavily discounts unfavorable information (i.e., produces similar recommendations under baseline and constraint conditions), human reviewers should explicitly flag missing or downweighted risk factors.

*Temperature as a documented decision parameter.* Section 5.5 demonstrates a clear trade-off: temperature 0.0 maximizes repeatability (DS) but may preserve framing biases; temperature 0.7 improves framing robustness (FR) and rationale invariance (EFI) but reduces decision stability. Organizations should document decoding settings in AI-assisted strategic decisions and align them with task purpose—repeatability-critical tasks (e.g., compliance checks) use T=0.0; exploratory or bias-auditing tasks use T=0.7.

*Model selection guidance.* Section 5.5.3 identifies three distinct model personas. For brand invariance (e.g., impartial audits), the Precision-Sensitive Type (DeepSeek-LLM-7B) is most suitable. For context responsiveness (e.g., exploratory analysis), the Adaptive Resilient Type (Meta-Llama-3.1-8B) is preferred. For output repeatability (e.g., compliance documentation), the Stable Functional Type (Qwen2.5-14B) is most reliable. No single persona dominates all axes; selection should reflect organizational priorities. Raising temperature to 0.7 improves framing robustness for all personas but reduces decision stability.

*Human-in-the-loop integration.* The audit protocol assumes human oversight at three critical junctures: (i) before deployment, comparing Generic vs. Specific prompts to detect brand bias; (ii) during stress-testing, reviewing whether the model discounted unfavorable constraints; and (iii) after generation, examining whether rationales reflect brand-consistent narratives rather than neutral evaluation. Organizations should document these audit steps as part of their AI governance framework.

**7. Limitations and Future Research**

**7.1 Limitations**

Four limitations temper the generalizability of our findings.

*Scenario scope.* Our six scenarios are anchored in Tesla's automotive/EV history. While this ensures internal consistency and provides a concrete, recognizable context, findings may not fully extend to other R&D contexts—pharmaceuticals with regulatory approval cycles, semiconductors with fixed fabrication timelines and high capital intensity, or platform ecosystems with network effect dynamics. The strategic archetypes themselves are general, but the specific instantiation of "opportunity" and "constraint" signals may vary across industries.

*Model scale and family.* We tested five open-weight models in the 7B–14B parameter range. This scale band reflects what organizations can run on local infrastructure, but larger models (70B+, e.g., Llama-3-70B, DeepSeek-V3) or proprietary APIs (GPT-4, Claude-3.5) may exhibit different sensitivity patterns. Preliminary evidence suggests that larger models may be more robust to certain framing manipulations but more susceptible to others due to richer pre-trained associations.

*Numerical perturbation range.* Numerical inputs were perturbed by ±20% while preserving semantic meaning, testing moderate quantitative variation. Extreme magnitude shifts (e.g., order-of-magnitude cost increases crossing feasibility thresholds) might trigger different responses. Our finding of "numerical insensitivity" should thus be interpreted as applying to moderate, non-threshold-crossing perturbations.

*Brand framing specificity.* We used only one high-profile innovator brand (Tesla) for Specific framing. Other brand types—incumbents (e.g., GM, Ford), failed innovators, B2B vs. B2C brands, or culturally specific brands—may yield different associative anchoring effects. Our results establish that brand framing *can* significantly alter strategic recommendations; future work must map the boundary conditions of this effect.

**7.2 Future Research Directions**

These limitations suggest four immediate research directions.

*Cross-industry scenario expansion.* Future work should extend the scenario portfolio beyond automotive/EV to diverse R&D contexts: pharmaceutical clinical trial decisions (regulatory milestone framing), semiconductor process technology choices (capital intensity vs. yield trade-offs), green energy project portfolios (policy uncertainty vs. technological learning), and platform ecosystem governance (complementor incentives vs. platform control). Cross-industry comparisons would reveal whether the observed sensitivity patterns are universal or context-dependent.

*Human-manager baseline comparisons.* A critical open question is how LLM sensitivity compares to human expert sensitivity under matched prompts. Do human R&D managers exhibit similar asymmetric responsiveness to optimistic framing (opportunity > competition > constraint)? Are they more or less susceptible to brand halo effects? Human panel studies (e.g., 50–100 R&D professionals responding to the same scenarios) would provide a benchmark for calibrating appropriate trust in LLM-generated recommendations. If LLMs and humans exhibit qualitatively similar bias patterns, the governance challenge becomes managing known heuristics; if they diverge, distinct protocols may be required.

*Debiasing intervention studies.* Can the observed framing sensitivities be reduced through prompt engineering or model configuration? Promising interventions include: (i) chain-of-thought prompting requiring explicit numerical reasoning, (ii) explicit counter-framing instructions ("Please ignore brand associations and focus on the following facts..."), (iii) ensemble methods aggregating predictions across multiple context variants, and (iv) fine-tuning on balanced examples of optimistic and constraint framing. Intervention studies testing these approaches would provide practical tools for immediate deployment.

*Scaling across model families.* Systematic scaling studies across parameter counts (1B, 7B, 13B, 34B, 70B, frontier) and model families (open vs. proprietary, instruction-tuned vs. base, different alignment methods) would reveal how sensitivity patterns evolve with model capacity. Do larger models become more robust to framing, or do they develop richer pre-trained associations that make them *more* sensitive? Longitudinal tracking across model versions (e.g., Llama-3.1 → Llama-4) would support reproducible AI governance in organizational settings.

**8. Conclusion**

**8.1 Summary of Contributions**

This study introduced a scenario-based audit framework for examining how LLMs respond to contextual and framing variations in R&D strategic decisions. Using six historically grounded scenarios and seven canonical strategic archetypes, we systematically manipulated semantic context, brand framing, and numerical inputs across five open-weight LLMs under two decoding temperatures.

Three main contributions emerge. First, we demonstrate that LLMs exhibit strong context sensitivity—opportunity framing shifts choices toward Technology Leadership, while unfavorable constraints induce defensive positioning—with a clear responsiveness hierarchy (opportunity > competition > constraint). Numerical insensitivity (±20% perturbations) confirms that semantic framing dominates quantitative signals in narrative prompts. Second, brand framing (Tesla vs. anonymous) asymmetrically amplifies these patterns, increasing Technology Leadership by 9.5 percentage points while suppressing collaborative and niche strategies. This effect operates as a sensitivity modulator (halo effect + associative anchoring) rather than a fixed bias. Third, these framing effects persist in model-generated rationales and vary systematically across models and temperature settings, with temperature 0.7 generally improving framing robustness at the cost of decision stability.

**8.2 Key Practical Message for R&D Management**

The central message for R&D organizations is this: **better audits, not just better models, are required.** LLMs are not stable default reasoners; they are conditional strategic agents whose decisions shift systematically with framing, brand identity, and decoding settings.

We therefore recommend three concrete practices for deployment. First, **anonymized prompting**: run side-by-side Generic and Specific prompts to detect brand-driven biases. Second, **context variant stress-testing**: evaluate recommendations under opportunity, competition, and constraint framings, as LLMs disproportionately discount unfavorable information (opportunity > competition > constraint). Third, **temperature documentation**: align decoding settings with task purpose—T=0.0 for repeatability-critical tasks (compliance, audits), T=0.7 for bias detection and exploratory analysis.

**8.3 Concluding Remark**

LLMs offer substantial potential as strategic decision-support tools in R&D and innovation management. But realizing that potential requires more than technical performance improvements. It requires organizational governance that treats LLMs as what they are: conditional agents whose behavior must be audited, documented, and interpreted in context. The framework and findings presented here provide a foundation for such governance—enabling organizations to harness LLMs' strategic capabilities while mitigating risks from framing-induced biases. The goal is not to eliminate sensitivity, which can be valuable for exploratory scenario work, but to make it transparent, auditable, and aligned with organizational decision processes. Effective human-LLM collaboration in R&D strategy demands not better models alone, but better *audits* of how models respond to the framing choices of their human users.

**References**

Alarcón Serrano, J.D., Cano-Marin, E. and Sicilia, M.-A. (2026). Assessing open LLMs' ability to identify biomedical taxonomic relationships: a SNOMED CT-based experimental evaluation. Knowledge-Based Systems, 115882.

Antuley, U., Siddiqui, S., Hameed, S., Arif, W. and Shah, S.A. (2026). SORA-ATMAS: adaptive trust management and multi-LLM aligned governance for future smart cities. Knowledge-Based Systems, 337, 115403.

Barney, J. (1991). Firm resources and sustained competitive advantage. Journal of Management, 17(1), pp. 99–120.

Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang, C. and Wang, Y. et al. (2024). A survey on evaluation of large language models. ACM Transactions on Intelligent Systems and Technology, 15(3), Article 39, pp. 1–45.

Chesbrough, H.W. (2003). Open Innovation: The New Imperative for Creating and Profiting from Technology. Boston, MA: Harvard Business School Press.

Du, K., Yang, B., Xie, K., Dong, N., Zhang, Z., Wang, S. and Mo, F. (2025). LLM-MANUF: an integrated framework of fine-tuning large language models for intelligent decision-making in manufacturing. Advanced Engineering Informatics, 65, 103263.

Gindullina, D., Lazutov, M., Stolyarov, K., Danilenko, D. and Leonenko, V. (2026). Expert-guided forecasting of epidemic ARI incidence based on physics-informed neural networks and large language models. Expert Systems with Applications, 315, 131730.

Gjorgjevikj, A., Nikolikj, A., Koroušić Seljak, B. and Eftimov, T. (2025). User-defined trade-offs in LLM benchmarking: balancing accuracy, scale, and sustainability. Knowledge-Based Systems, 330, 114405.

Heo, S., Son, S. and Park, H. (2025). HaluCheck: explainable and verifiable automation for detecting hallucinations in LLM responses. Expert Systems with Applications, 272, 126712.

Kahneman, D. (2011). Thinking, Fast and Slow. New York: Farrar, Straus and Giroux.

Kong, L., Zhang, Y., Zhong, X., Fu, H., Wang, Y. and Liu, H. (2026). HaluGNN: hallucination detection in large language models using graph neural network. Expert Systems with Applications, 306, 130857.

Liu, J., Hao, W., Cheng, K., Chen, G. and Xie, X. (2026). CART: a traceable zero-shot planning framework for large language models with adaptive replanning. Knowledge-Based Systems, 336, 115189.

Memduhoğlu, A., Fulman, N., Polat, N. and Ataş, T. (2026). Large language models as virtual experts? Evaluating AHP-based criteria weighting performance for solar power plant site selection. Expert Systems with Applications, 299, 130171.

Miles, R.E., Snow, C.C., Meyer, A.D. and Coleman, H.J. (1978). Organizational strategy, structure, and process. Academy of Management Review, 3(3), pp. 546–562.

Ojuri, S., Han, T.A., Chiong, R. and Di Stefano, A. (2025). Optimizing text-to-SQL conversion techniques through the integration of intelligent agents and large language models. Information Processing & Management, 62(5), 104136.

Porter, M.E. (1980). Competitive Strategy: Techniques for Analyzing Industries and Competitors. New York: Free Press.

Przystalski, K., Argasiński, J.K., Grabska-Gradzińska, I. and Ochab, J.K. (2026). Stylometry recognizes human and LLM-generated texts in short samples. Expert Systems with Applications, 296, 129001.

Schilling, M.A. (2019). Strategic Management of Technological Innovation (6th ed.). New York: McGraw-Hill Education.

Simon, H.A. (1957). Models of Man: Social and Rational. New York: John Wiley and Sons.

Sinha, A., Agarwal, C. and Malo, P. (2026). FinBloom: knowledge-grounding large language model with real-time financial data. Knowledge-Based Systems, 339, 115559.

Thorndike, E.L. (1920). A constant error in psychological ratings. Journal of Applied Psychology, 4(1), pp. 25–29.

Torres-Moreno, D. and Hermosillo-Valadez, J. (2026). Semantic knowledge abstraction: consistent reasoning in large language models for natural language inference. Knowledge-Based Systems, 332, 114825.

Tversky, A. and Kahneman, D. (1974). Judgment under uncertainty: heuristics and biases. Science, 185(4157), pp. 1124–1131.

Wang, Z., Wan, C., Liu, J., Zhang, X., Wang, H., Hu, Y. and Hu, Z. (2025a). MASC: large language model-based multi-agent scheduling chain for flexible job shop scheduling problem. Advanced Engineering Informatics, 67, 103527.

Wang, P., Hu, Q., Mei, Q., Wang, S., Yang, Y., Guo, D., Liu, X., Hu, W. and Chen, J. (2025b). Intelligent port logistics: a spatiotemporal knowledge graph and AI-agent framework for berth allocation. Advanced Engineering Informatics, 68, 103633.

Xiong, X., Cai, H., Yu, H., Shen, B. and Hu, P. (2025). DR-RAG: domain-rule-based retrieval-augmented generation for aviation digital model design. Advanced Engineering Informatics, 68, 103688.

**Appendix A. Data Validation and Categorical Compliance**  
This appendix reports instruction-following validity checks for the categorical strategy-selection task. The models demonstrated stable instruction-following performance, with an overall compliance rate of approximately 97.3% (396,961 valid responses out of 408,000 total inferences). Table A1 summarizes the compliance and non-compliance (error) rates across the five context variants.

| Scenario | Compliance Rate (Valid) | Non-compliance Rate (Error) |
| :---- | :---- | :---- |
| Opportunity | 97.57% | 2.43% (Lowest) |
| Randomized | 97.49% | 2.51% |
| Competitive | 97.30% | 2.70% |
| Base | 97.16% | 2.84% |
| Count Fact | 96.96% | 3.04% (Highest) |

Table A1. Categorical compliance and error rates by context variant.

As indicated in Table A1, the non-compliance rate remained within a narrow band, with a maximum deviation of only 0.61 percentage points between the lowest-error (Opportunity, 2.43%) and highest-error (Count Fact, 3.04%) variants. This uniformity indicates that categorical compliance was largely insensitive to contextual framing. The overall consistency across all variants confirms that the models are capable of operating within a constrained decision-making framework. To ensure a rigorous comparison of strategic patterns, subsequent analyses in Section 5 are conducted using the normalized distribution of valid strategic choices, excluding the out-of-set responses.