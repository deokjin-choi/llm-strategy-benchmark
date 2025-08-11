# llm-strategy-benchmark

### 🌟 Project Overview

This project is a comprehensive benchmark designed to systematically evaluate how **Large Language Models (LLMs) make strategic decisions** in complex, real-world business scenarios. Our goal is to move beyond simple question-and-answer tests to understand the nuances of an LLM's analytical and reasoning capabilities when faced with ambiguous, high-stakes problems.

---

### Motivation and Core Hypothesis

The primary motivation for this project is to answer a fundamental question: **How do LLMs reason when given a strategic problem, and how much does external context influence their decision-making?**

We aim to verify two main hypotheses:

1.  **Context Dependency**: An LLM's strategic recommendations will change based on the specific contextual information it receives, and these changes should be logically sound.
2.  **Role Bias**: An LLM's strategic choices might differ when a problem is presented as a specific company case (e.g., **Tesla**) versus an anonymous, generic case, indicating a potential brand or role bias.

---

### The Six Business Scenarios

The experiment is built around **six historical business scenarios** based on Tesla's development, with a fixed core problem at each stage. This provides a rich, real-world foundation for testing.

1.  **Founder Period**: A new company with limited resources must choose a market entry strategy to secure funding and establish a brand.
2.  **Roadster Launch**: The company faces a difficult balancing act between product quality and timely delivery during its first major product launch.
3.  **Model S Launch**: The challenge is to transition from a niche, high-end automaker to a mass-market manufacturer by scaling production infrastructure.
4.  **Model X Launch**: The company seeks to enter the growing SUV market, but a highly complex product design creates significant manufacturing risks.
5.  **Model 3 Mass Market**: With an overwhelming number of pre-orders, the company must rapidly scale production while navigating financial and reputational risks.
6.  **Energy Infrastructure**: The company must strategically diversify its business by addressing key bottlenecks in EV adoption, such as battery costs and charging infrastructure.

---

### Experimental Setup

To rigorously test our hypotheses, we designed the experiment with the following key variables and parameters:

* **Problem Framing**: Each scenario is tested with two distinct problem types:
    * **Generic**: The problem is framed as a challenge for an "anonymous company," which helps to identify a model's unbiased, pure reasoning.
    * **Specific**: The problem explicitly names **Tesla**, allowing us to test for any brand or name-based biases.
* **Dynamic Context**: The problem statement remains fixed, but additional data—such as market conditions, technology limitations, or financial details—is **dynamically added or removed**. This allows us to measure how an LLM's decision changes as the amount of available information varies.
* **Models**: The benchmark uses three distinct LLMs to compare performance across different architectures:
    * `mistralai/Mistral-7B-Instruct-v0.3`
    * `Qwen/Qwen2.5-14B-Instruct`
    * `meta-llama/Meta-Llama-3.1-8B-Instruct`
* **Model Parameters**: To observe how a model's behavior changes, we vary two key parameters:
    * **Temperature**: `0.0`, `0.3`, `0.7` (to test the balance between determinism and creativity).
    * **Max Tokens**: `150`, `200`, `256` (to test the impact of response length constraints).
* **Repeats**: Each unique combination of the above variables is repeated **15 times** to ensure statistical reliability of the results.

This extensive setup allows us to generate a large, detailed dataset that will be used to analyze and interpret the models' strategic decision-making process.
