# llm-strategy-benchmark

### Project Overview

This project is a comprehensive benchmark designed to systematically evaluate how **Large Language Models (LLMs) make strategic decisions** in complex, real-world business scenarios. Our goal is to move beyond simple question-and-answer tests to understand the nuances of an LLM's analytical and reasoning capabilities, with a specific focus on diagnosing their cognitive biases and flexibility.

---

### Motivation and Core Hypothesis

Our primary motivation is to answer a fundamental question: "How do LLMs reason when given a strategic problem, and what cognitive biases do they exhibit?"

We aim to verify two main hypotheses:

1.  **Context Dependency and Flexibility**: An LLM's strategic recommendations will change based on the specific contextual information it receives (e.g., market conditions, financial data), and the degree of this change will vary across models.
2.  **Framing and Brand Bias**: An LLM's strategic choices might differ when a problem is presented as a specific company case (e.g., **Tesla**) versus an anonymous, generic case, indicating a potential brand or role bias.

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

1.  **Founder Period**: A new company with limited resources must choose a market entry strategy to secure funding and establish a brand.
2.  **Roadster Launch**: The company faces a difficult balancing act between product quality and timely delivery during its first major product launch.
3.  **Model S Launch**: The challenge is to transition from a niche, high-end automaker to a mass-market manufacturer by scaling production infrastructure.
4.  **Model X Launch**: The company seeks to enter the growing SUV market, but a highly complex product design creates significant manufacturing risks.
5.  **Model 3 Mass Market**: With an overwhelming number of pre-orders, the company must rapidly scale production while navigating financial and reputational risks.
6.  **Energy Infrastructure**: The company must strategically diversify its business by addressing key bottlenecks in EV adoption, such as battery costs and charging infrastructure.

---

### Experimental Setup and Methodology

To rigorously test our hypotheses and evaluate our diagnostic metrics, we designed the experiment with the following key variables and parameters:

* **Problem Framing**: Each scenario is tested with two distinct problem types:
    * **Generic**: The problem is framed as a challenge for an "anonymous company," which helps to identify a model's pure, unbiased reasoning.
    * **Specific**: The problem explicitly names **Tesla**, allowing us to test for any brand or name-based biases.
* **Dynamic Context**: The core problem statement remains fixed, but additional data—such as market conditions, technology limitations, or financial details—is **dynamically added or removed**. This allows us to measure how an LLM's decision changes as the amount of available information varies.
* **Models**: The benchmark uses three distinct LLMs to compare performance across different architectures:
    * `mistralai/Mistral-7B-Instruct-v0.3`
    * `Qwen/Qwen2.5-14B-Instruct`
    * `meta-llama/Meta-Llama-3.1-8B-Instruct`
* **Repeats**: Each unique combination of the above variables is repeated **5 times** to ensure statistical reliability of the results.

This extensive setup allows us to generate a large, detailed dataset that will be used to analyze and interpret the models' strategic decision-making process.