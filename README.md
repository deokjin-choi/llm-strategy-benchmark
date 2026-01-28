# llm-strategy-benchmark

### Project Overview

This project is a comprehensive benchmark designed to systematically evaluate how **Large Language Models (LLMs) make strategic decisions** in complex, real-world business scenarios. Our goal is to move beyond simple question-and-answer tests to understand the nuances of an LLM's analytical and reasoning capabilities, with a specific focus on diagnosing their cognitive biases and flexibility.

---

### Motivation and Core Hypothesis

Our primary motivation is to answer a fundamental question: "How do LLMs reason when given a strategic problem, and what cognitive biases do they exhibit?"

We aim to verify two main hypotheses:

1. **Context Dependency and Flexibility**: An LLM's strategic recommendations will change based on the specific contextual information it receives (e.g., market conditions, financial data), and the degree of this change will vary across models.
2. **Framing and Brand Bias**: An LLM's strategic choices might differ when a problem is presented as a specific company case (e.g., **Tesla**) versus an anonymous, generic case, indicating a potential brand or role bias.

---

### Key Diagnostic Metrics

This benchmark uses five key metrics to quantitatively diagnose the strategic decision-making profile of LLMs:

* **Technology Leadership Preference Index**: Measures a model's tendency to favor specific strategic options.
* **Brand Bias Index**: Quantifies the influence of a brand name (e.g., Tesla) on a model's decision-making.
* **Context Dependency Index**: Measures the degree to which a decision changes when additional contextual information is added or removed.
* **Numerical Insensitivity Index**: Evaluates a model's insensitivity to changes in numerical data within the problem statement.
* **Rationale-Choice Alignment Score**: Assesses the logical consistency between a model's chosen strategy and its provided reasoning.

---

### The Six Business Scenarios

The experiment is built around **six historical business scenarios based on Tesla's development**, each with a fixed core problem and various strategic options.

1. **Founder Period**: A new company with limited resources must choose a market entry strategy to secure funding and establish a brand.
2. **Roadster Launch**: The company faces a difficult balancing act between product quality and timely delivery during its first major product launch.
3. **Model S Launch**: The challenge is to transition from a niche, high-end automaker to a mass-market manufacturer by scaling production infrastructure.
4. **Model X Launch**: The company seeks to enter the growing SUV market, but a highly complex product design creates significant manufacturing risks.
5. **Model 3 Mass Market**: With an overwhelming number of pre-orders, the company must rapidly scale production while navigating financial and reputational risks.
6. **Energy Infrastructure**: The company must strategically diversify its business by addressing key bottlenecks in EV adoption, such as battery costs and charging infrastructure.

---

### Experimental Setup and Methodology

To rigorously test our hypotheses and evaluate our diagnostic metrics, we designed the experiment with the following key variables and parameters:

* **Problem Framing**: Each scenario is tested with two distinct problem types:
  * **Generic**: The problem is framed as a challenge for an "anonymous company," which helps to identify a model's pure, unbiased reasoning.
  * **Specific**: The problem explicitly names **Tesla**, allowing us to test for any brand or name-based biases.

* **Dynamic Context**: The core problem statement remains fixed, but additional data—such as market conditions, technology limitations, or financial details—is **dynamically added or removed**. This allows us to measure how an LLM's decision changes as the amount of available information varies.

* **Models**: The benchmark now supports **six distinct LLMs** to compare performance across different architectures:
  * `mistralai/Mistral-7B-Instruct-v0.3`
  * `Qwen/Qwen2.5-14B-Instruct`
  * `meta-llama/Meta-Llama-3.1-8B-Instruct`
  * `deepseek-ai/deepseek-7b-instruct`
  * `01-ai/Yi-9B-Chat`

* **Temperature Settings**: Each experiment is conducted with **two decoding strategies**:
  * `temperature=0.0` (deterministic reasoning)
  * `temperature=0.7` (creative reasoning)

* **Repeats**: Each unique combination of the above variables is repeated **30 times** to ensure statistical robustness.

* **Scenario Variants**: Multiple `.json` files (e.g., `scenarios.json`, `scenarios_randomized_numbers.json`, etc.) are used. Results are automatically saved under:
  * `scenario_infer_results/base` (for `scenarios.json`)
  * `scenario_infer_results/<filename>` (for other scenario files)

---

## Strategy Distribution Analysis

### Figure 1. Strategy Ratio by Scenario
![Strategy Ratio by Scenario](final_results/plots/eval_eda_Strategy_Ratio_by_Scenario.png)

Across scenarios, the strategy distribution shows a clear **scenario-dependent shift** rather than a single dominant default. In the **base** setting, **Niche Focus** is the most frequent choice (0.28), followed by **Open Innovation** and **Technology Leadership** (both 0.15). Several scenario-specific patterns stand out:

- **competitive_dynamics**: **Technology Leadership** increases (0.15 → 0.23), while **Niche Focus** slightly decreases (0.28 → 0.24). At the same time, **Open Innovation** shows a modest rise (0.15 → 0.17). This pattern suggests that competitive pressure encourages consideration of leadership-oriented strategies, but without the strong commitment observed under opportunity-focused conditions.
- **count_fact**: **Technology Leadership** drops markedly (0.15 → 0.09), while **Niche Focus** rises (0.28 → 0.33) and **Fast Follower** increases (0.07 → 0.10). Under unfavorable facts, the model shifts away from leadership strategies toward more conservative positioning.
- **opp_focus**: **Technology Leadership** surges strongly (0.15 → 0.39), becoming the dominant strategy. Opportunity-focused context sharply amplifies leadership-oriented decisions.
- **randomized_numbers**: Nearly identical to the base scenario (e.g., **Niche Focus** 0.28 → 0.29; **Technology Leadership** 0.15 → 0.14), indicating **numerical perturbations have limited impact** on overall strategy choice.

Overall, the results indicate that **context framing (opportunity vs. unfavorable facts)** drives large shifts in strategy preference, while **pure numerical variation** produces minimal change.

---
### Figure 2. PCA of Strategy Ratios (2D Projection)
![PCA of Strategy Ratios](final_results/plots/eval_eda_PCA_of_Strategy_Ratios_2D_Vectorized_Analysis.png)

We applied Singular Value Decomposition (SVD) to the scenario–strategy ratio matrix (after mean-centering) and projected it into two dimensions. This visualization highlights the relative similarity of strategic distributions across scenarios.

Results show:
- **base** and **randomized_numbers** cluster closely together.  
- **opp_focus** and **count_fact** diverge strongly, indicating that **positive opportunity framing** and **negative fact framing** distinctly reshape strategic choices.  

This demonstrates that **LLM strategy distributions are conditionally separable**, and PCA effectively captures these structural shifts.

---
### Figure 3. Delta from Base (Generic vs. Specific)
![Delta from Base](final_results/plots/eval_Generic_and_Specific_Δ_by_Scenario.png)

Figure 3 compares **strategy shifts relative to the base scenario** under **Generic (anonymous company)** and **Specific (brand-framed)** conditions, highlighting how **problem framing alters decision sensitivity rather than absolute strategy choice**.

Several consistent patterns emerge:

- **Overall pattern**: Brand framing does not uniformly increase or decrease aggressiveness. Instead, it **modulates the direction and magnitude of strategy shifts depending on context**.
- **competitive_dynamics**: Under competitive pressure, **Specific framing dampens extreme shifts** observed in the Generic case. Technology Leadership increases modestly, while alternative strategies such as Open Innovation also gain share, indicating a stabilizing effect rather than full commitment.
- **count_fact**: When unfavorable facts are emphasized, **Specific framing amplifies defensive reactions**. Technology Leadership declines more strongly, accompanied by increased movement toward Fast Follower and Open Innovation strategies.
- **opp_focus**: In opportunity-focused contexts, **Specific framing amplifies aggressive responses**. Technology Leadership increases more strongly than in the Generic case, suggesting that brand identity acts as a catalyst rather than a constraint when upside potential is salient.
- **randomized_numbers**: Both Generic and Specific framings remain close to zero across strategies, indicating that **brand effects are largely inactive under purely numerical perturbations**.

Overall, Figure 3 demonstrates that **brand framing functions as a context-dependent amplifier or stabilizer**, not a fixed bias. This highlights the importance of considering **decision sensitivity and framing effects** when deploying LLMs for strategic decision-making in R&D contexts.

