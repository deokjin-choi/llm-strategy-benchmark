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
Recent research has increasingly embedded LLMs into real-world decision pipelines, particularly in operations and engineering management. Wang et al. propose a multi-agent scheduling chain that leverages LLM-based agents to handle flexible job shop scheduling and real-time rescheduling, demonstrating substantial gains in scheduling efficiency under disruptions \cite{WANG2025103527}. Du et al. introduce LLM-MANUF, an integrated framework in which multiple fine-tuned LLMs generate alternative decision plans that are subsequently ranked and fused, highlighting that manufacturing decision quality depends as much on comparing and aggregating candidate strategies as on generating a single “best” answer \cite{DU2025103263}. Beyond manufacturing, Wang et al., Xiong et al., and Gindullina et al. combine spatiotemporal knowledge graphs, digital twins, and physics-informed neural networks with LLM agents to support berth allocation, aviation design, and epidemic forecasting, respectively \cite{WANG2025103633,XIONG2025103688,GINDULLINA2026131730}. These studies illustrate how LLMs can act as planning and coordination engines for complex operational systems, yet they typically assess success in terms of task performance and system-level efficiency. They rarely examine how, within a fixed set of strategic options, LLMs distribute their choices or how sensitive those choices are to subtle changes in contextual description or framing.

**2.2 Reliability, hallucination, and safety in high-stakes decisions**  
Another line of work focuses on making LLM-supported decisions more reliable in high-stakes environments. Kong et al. present HaluGNN, which models question–answer pairs as token graphs and uses graph neural networks to detect hallucinated content, thereby improving decision security in domains where factual errors are costly \cite{KONG2026130857}. Heo et al. develop HaluCheck, a visualization and automation framework that decomposes model responses into sentence-level claims, retrieves external evidence, and highlights likely hallucinations through an interactive interface for expert systems \cite{HEO2025126712}. Przystalski et al. use stylometric features to distinguish human- from LLM-generated texts, informing governance around authorship, attribution, and authenticity \cite{PRZYSTALSKI2026129001}. In the context of smart-city management, Antuley et al. propose SORA-ATMAS, an adaptive trust and governance framework that aligns multiple LLM agents with cross-domain policies and regulatory constraints \cite{ANTULEY2026115403}. Collectively, these studies treat reliability and safety primarily at the level of factual correctness, anomaly detection, and policy compliance. What remains underexplored is whether LLMs are reliable as strategic decision agents—specifically, how their strategic choices and rationales shift under contextual and framing manipulations when the underlying problem remains fixed.

**2.3 Knowledge-grounded decision infrastructures**  
A third strand of literature builds knowledge-grounded infrastructures that connect LLMs to structured data and domain ontologies. Ojuri et al. propose a text-to-SQL framework in which LLMs and intelligent agents translate natural language queries into executable SQL, lowering the access barrier for non-technical users and reducing dependence on data specialists in organizational decision-making \cite{OJURI2025104136}. Xiong et al. introduce DR-RAG, a domain-rule-based retrieval-augmented generation framework that ties aviation digital models to knowledge graphs, rule bases, and digital twins, turning complex product design into a loop of retrieval, generation, and simulation feedback \cite{XIONG2025103688}. In financial decision-making, Sinha et al. present FinBloom, a knowledge-grounded financial agent that combines a domain-specialized LLM with real-time news and regulatory filings to answer dynamic financial queries \cite{SINHA2026115559}. Alarcón Serrano et al. evaluate how well general-purpose LLMs can recognize taxonomic relationships in SNOMED CT, clarifying their role in biomedical knowledge-graph workflows that support clinical reasoning \cite{ALARCONSERRANO2026115882}. These works substantially improve what LLMs know and how they access relevant information. Yet, even with stronger grounding, they do not systematically analyze how a model’s strategic choices over a fixed option set change when only the narrative emphasis, identity cues, or quantitative inputs are perturbed.

**2.4 Evaluation and Behavioral Profiling of LLMs**  
Evaluation- and reasoning-centric research has sought to understand how LLMs plan, reason, and exhibit preferences across tasks. Liu et al. propose CART, a traceable planning framework that decomposes goals into subtasks, tracks planning trajectories, and triggers replanning when conditions change, thereby improving the robustness of LLM-based agents in incomplete-information environments \cite{LIU2026115189}. Gjorgjevikj et al. introduce xLLMBench, a decision-centric benchmarking framework that uses multi-criteria decision-making to rank models along accuracy, scale, energy consumption, and other non-performance factors \cite{GJORGJEVIKJ2025114405}. Memduhoğlu et al. treat LLMs as “virtual experts” for multi-criteria spatial planning, comparing their analytic hierarchy process (AHP) weightings with those of human panels and documenting systematic biases in how models prioritize criteria for solar power plant site selection \cite{MEMDUHOGLU2026130171}. Torres-Moreno and Hermosillo-Valadez propose a semantic knowledge abstraction framework that restructures premise–hypothesis relations in natural language inference to improve consistency and reveal latent semantic gaps in LLMs’ reasoning \cite{TORRESMORENO2026114825}. These contributions collectively move beyond simple accuracy metrics toward richer assessments of reasoning, robustness, and multi-dimensional trade-offs. However, these evaluation frameworks typically focus on task-level performance or general reasoning consistency, leaving open how LLMs' strategic preferences shift under controlled perturbations of brand identity, contextual emphasis, or numerical signals.

**2.5 Cognitive Biases, Framing Effects, and Strategic Heuristics**

Human strategic judgments are susceptible to cognitive shortcuts that bypass deliberative reasoning \cite{KAHNEMAN2011}. The *halo effect* occurs when a salient positive attribute—such as a firm's reputation—biases overall evaluation \cite{THORNDIKE1920}. *Anchoring* describes the tendency for an initial cue to shape subsequent judgments, even when that cue carries no objective decision-relevant information \cite{TVERSKY1974}. More broadly, Kahneman (2011) distinguishes fast, intuitive System 1 thinking from slow, analytical System 2 reasoning, with framing effects arising when System 1 dominates \cite{KAHNEMAN2011}.

In LLM-based strategic decision-making, analogous patterns remain underexplored. The presence of a high-status brand identity or the selective emphasis of opportunity versus constraint information could trigger heuristic-like responses—leading models to prioritize narrative consistency over neutral evaluation of quantitative or factual signals. Nonetheless, existing benchmarks have not systematically characterized how such framing effects manifest in LLMs' categorical strategy choices, nor whether they persist in model-generated rationales.

**2.6 Strategic Archetypes in R&D and Innovation Management**  
To rigorously characterize the strategic behavior of LLMs, it is necessary to map their decision outputs onto established theoretical frameworks. This study adopts seven strategic archetypes rooted in the classical literature of competitive strategy and technological innovation. These options—Technology Leadership, Fast Follower, Open Innovation, Niche Focus, Diversification, Retrenchment, and Maintain—represent the fundamental trajectories firms pursue to navigate market transitions and resource constraints.

Specifically, the distinction between 'Technology Leadership' and 'Fast Follower' is grounded in the pioneering work on timing of entry and R&D intensity \cite{SCHILLING2019}. The concepts of 'Niche Focus' and 'Diversification' align with Porter’s generic strategies and the resource-based view of the firm \cite{PORTER1980, BARNEY1991}. Furthermore, 'Open Innovation' reflects the modern shift toward collaborative R&D ecosystems \cite{CHESBROUGH2003}, while 'Retrenchment' and 'Maintain' represent critical defensive maneuvers under high environmental volatility \cite{MILES1978}. By constraining the LLM’s choice set to these validated archetypes, we transition from observing simple linguistic patterns to analyzing structural strategic reasoning. This methodological grounding ensures that the observed shifts in choice distributions are interpretable within the context of established management science.

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

$$CR(m, \tau) = \mathbb{E}_{s,n,\phi,v} \left[ JSD\left( P(m, \tau, s, n, \text{Base}, \phi), P(m, \tau, s, n, v, \phi) \right) \right]$$  
where $v \in \mathcal{V}_{sem}$ and $\mathcal{V}_{sem} = \{\mathrm{competitive\_dynamics},\ \mathrm{count\_fact},\ \mathrm{opp\_focus}\}$.

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

The models do not respond uniformly to temperature changes. Instead, they display distinct behavioral profiles. The vignettes below use **the same per-axis min–max scaling as the radar** (each spoke is $[0,1]$ **relative to the ten** $(m,\tau)$ **rows in Table 4**), not the raw axis entries in Table 4 itself.

1. **Stable Functional Type**  
   Qwen2.5-14B keeps a **high DS** spoke at both temperatures (**≈0.91 → 0.79** on that scaled axis), i.e., strong run-to-run concentration *within this panel*. **FR** and **EFI** remain **inward** on the radar—e.g., **FR** near the hub at $T=0.0$—so the model is less “brand-blind” and less rationale-invariant than some peers in this display. The footprint matches **functional persistence** over maximal framing or stylistic invariance.

2. **Precision-Sensitive Type**  
   DeepSeek-LLM-7B-Chat at $T=0.0$ pushes **FR**, **NS**, and **DS** toward the outer ring (**FR ≈0.95**, **NS = 1.00**, **DS ≈0.94** on the scaled spokes). At $T=0.7$, **DS** and **CR** **collapse toward 0** while **EFI** moves **outward (≈0.99)**—the dashed trace **deflates** on repeatability and context-response axes. The persona is a **deterministic** specialist more than a stable stochastic partner.

3. **Adaptive Resilient Type**  
   Meta-Llama-3.1-8B reaches the **CR** spoke maximum at $T=0.0$ (**1.00** under this normalization)—the strongest semantic redistribution in the panel when sampling is greedy. **EFI** is **highest at $T=0.7$** (**1.00** scaled), with **FR** also **high (≈0.92)**, consistent with **situational responsiveness** paired with **stronger cross-frame rationale consistency** when temperature is raised—in this **relative** view only.

These differences imply that model selection should consider not only average performance, but also response stability under different decoding environments.

#### (3) Practical Deployment Implications

The findings indicate that temperature should be selected according to task objectives rather than treated as a universal default parameter.

- **Low temperature ($T=0.0$)** is suitable for factual verification, rule-based workflows, structured extraction, and cases where repeatability is critical.
- **Mid temperature ($T=0.3$–$0.5$)** may be suitable when organizations require both stable outputs and moderate reasoning flexibility.
- **Higher temperature ($T=0.7$)** is more suitable for exploratory analysis, alternative generation, or tasks where reducing framing bias is important.

More broadly, organizations should view temperature as a controllable strategic lever. Performance may improve when decoding settings are aligned with task goals such as precision, robustness, or objectivity, rather than relying on a single fixed configuration for all use cases.


### 5.5.4 Scenario-Resolved Behavior: Localizing FR, CR, and DS Across the Experimental Grid

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

### (A) FR Case: Brand Identity Changes Strategic Preference (5_model_3_mass_market)

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

**(3) Observed choices.**

To isolate firm-identity framing effects from semantic context variation, Fig. 8 reports pooled strategy choices under the neutral base context variant only.

![FR_deepdive_Qwen_Model3](final_results/plots/eval_deepdive_fr_framing_stacks__Qwen2.5-14B__5_model_3_mass_market.png)  
Fig. 8. FR deep-dive for 5_model_3_mass_market.

A paired permutation test on brand-masked token statistics (480 paired runs per cell) confirms that the lexical gap between Generic and Specific rationales is systematic, not a decoding artifact (global separation statistic with p ≈ 0.005 at both T=0.0 and T=0.7).

| Framing | Contexts | Rationale | Strategy choice |
| :---- | :---- | :---- | :---- |
| **Generic** | High demand, production pressure | Keywords (2-grams): manufacturing partners, scale production, quickly scale, utilizing manufacturing, production rapidly. | **Open Innovation** — Utilize manufacturing partners (OEM) to scale production. External partnerships as the main scaling lever.|
| **Specific** | High demand, production pressure | Keywords (2-grams): quality profitability, maintaining reputation, long term, crucial maintaining.  | **Maintain** — Expand production gradually while prioritizing quality and profitability. Internal cadence and safeguards over fastest-possible scale-out. |

Table 7. FR case summary

**(4) Inferred rationale and implications (core).** The same baseline narrative yields a large reallocation from Open Innovation (Generic) to Maintain (Specific), stable across temperatures in this priority cell. This aligns with a framing-conditional explanatory shift (partner-led scaling vs. controlled ramp-up), and a paired permutation test confirms that the lexical gap between framings is systematic. These results suggest that firm naming can change both the “default” strategic choice and its accompanying rationale, even when embedded facts are held fixed. For deployment, FR-type cells motivate side-by-side Generic/Specific prompting as a minimum audit.

---

### (B) CR Case: Semantic Cues Reallocate Strategy Mix (4_model_x_launch)

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

**(3) Observed choices.**

semantic manipulation is implemented only through the context_variant axis—base, competitive_dynamics, count_fact, and opp_focus—on top of Generic vs. Specific firm-identity framing.

![CR_deepdive_Qwen_ModelX_framing](final_results/plots/eval_deepdive_cr_strategy_stacks_framing__Qwen2.5-14B__4_model_x_launch.png)  
Fig. 9. CR deep-dive for 4_model_x_launch.

The following plot reports the corresponding mean JSD from base for each semantic variant (CR cue strength), shown separately for $T=0.0$ and $T=0.7$.
![CR_deepdive_Qwen_ModelX_jsd](final_results/plots/eval_deepdive_cr_jsd_by_variant__Qwen2.5-14B__4_model_x_launch.png)  
Fig. 10. CR cue strength: mean JSD from base per semantic variant (T=0.0 vs T=0.7).

**(4) Inferred rationale and implications (core).** In this CR-max cell, semantic variants do not merely “nudge” choices: they reallocate mass across distinct strategic families. Generic framing remains dominated by Open Innovation, while Specific framing becomes more dispersed—competitive_dynamics lifts Fast Follower, and opp_focus uniquely activates Technology Leadership—consistent with a variant-dependent gating of “innovation-forward” recommendations. Quantitatively, the JSD between Base and each variant confirms a clear hierarchy across temperatures: opp_focus produces the largest shift, followed by competitive_dynamics, while count_fact yields minimal movement. The implication is that the model shows asymmetric cue sensitivity: it over-responds to positive opportunities and competitive pressure (activating new strategic modes) while under-responding to unfavorable factual constraints. Practitioners should therefore verify whether their model discounts critical but negative information, not just whether it reacts to salient cues.

---

### (C) DS Case: Context Load Shapes Stability (2_roadster_launch)

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

**(3) Observed choices.**

![DS_deepdive_DeepSeek_Roadster_stacks_numctx](final_results/plots/eval_deepdive_ds_strategy_stacks_numcontext_framing__deepseek-llm-7b-chat__2_roadster_launch.png)  
Fig. 11. Strategy mix across Num Context tiers.

![DS_deepdive_DeepSeek_Roadster_entropy](final_results/plots/eval_deepdive_ds_entropy_numcontext_box__deepseek-llm-7b-chat__2_roadster_launch.png)  
Fig. 12. Repeat-level entropy across Num Context tiers.

**(4) Inferred rationale and implications (core).** Strategy mass shifts as context blocks accumulate: near-vacuous prompts favor **Open Innovation**, partial context pivots toward **Retrenchment**/**Niche Focus**, and full context yields a **Fast Follower vs. Retrenchment** split, with $T{=}0.7$ producing more diffuse mixtures. This pattern is consistent with the scenario’s explicitly conflicting pressures (speed vs. quality vs. cash vs. supply reliability), where adding non-redundant facts need not select a single modal strategy under sampling. For deployment, DS-type cells indicate that reliability auditing should vary **Num Context** and **framing**, not temperature alone.

**6\. Discussion and Implications**  
The findings indicate that LLMs adjust strategic recommendations systematically in response to contextual cues, reflecting an ability to distinguish between qualitatively different strategic environments rather than producing rigid outputs.

However, these adjustments are highly sensitive to framing. Brand identification and selective emphasis on contextual information can amplify or dampen strategic shifts beyond what factual changes alone suggest, indicating potential overreaction to semantic framing.

For R\&D and innovation management, this suggests that LLM outputs should be treated as context-conditional. Effective use therefore requires human-in-the-loop processes that compare alternative framings, examine embedded assumptions, and interpret LLM-generated strategies as exploratory inputs rather than prescriptions.

**References**

Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang, C. and Wang, Y. et al. (2024). A survey on evaluation of large language models. ACM Transactions on Intelligent Systems and Technology, 15(3), Article 39, pp. 1–45. 

Lieberman, M.B. and Montgomery, D.B. (1988). First-mover advantages. Strategic Management Journal, 9(S1), pp. 41–58.

Schilling, M.A. (2019). Strategic Management of Technological Innovation (6th ed.). New York: McGraw-Hill Education.

**Appendix A. Data Validation and Categorical Compliance**  
This appendix reports instruction-following validity checks for the categorical strategy-selection task. The models demonstrated stable instruction-following performance, with an overall compliance rate of approximately 89%. Table A1 summarizes the compliance and non-compliance (error) rates across the five experimental scenarios.

| Scenario | Compliance Rate (Valid) | Non-compliance Rate (Error) |
| :---- | :---- | :---- |
| Competitive | 91.49% | 8.51% (Lowest) |
| Count Fact | 89.96% | 10.04% |
| Randomized | 89.47% | 10.53% |
| Opportunity | 87.23% | 12.77% |
| Base | 85.95% | 14.05% (Highest) |

As indicated in Table A1, the non-compliance rate remained within a relatively narrow range, with a maximum deviation of only 5.54 percentage points between the Competitive (8.51%) and Base (14.05%) scenarios. The slightly higher error rate in the Base scenario suggests that in the absence of explicit contextual anchors, LLMs may exhibit a greater tendency for 'strategic drift'—providing descriptive justifications instead of selecting a single category.

However, the overall consistency across all scenarios confirms that the models are capable of operating within a constrained decision-making framework. To ensure a rigorous comparison of strategic patterns, subsequent analyses in Section 5 are conducted using the normalized distribution of valid strategic choices, excluding the out-of-set responses.