# Context Collapse in Long-Horizon Agents

### Benchmarking Hierarchical Memory against RAG and Summarization

> Controlled benchmark for evaluating memory architectures in long-horizon LLM conversations.

---

## Overview

Large Language Model (LLM) agents increasingly operate across long conversations, but maintaining factual consistency over time remains difficult.

This project investigates **context collapse** — the gradual degradation of factual recall as conversations become longer and memory systems compress, retrieve, or restructure context.

We benchmark four memory architectures under controlled conditions:

* **Baseline Memory** — Full conversation history
* **Hierarchical Memory** — Working + Episodic + Semantic memory
* **RAG Memory** — Retrieval-Augmented Generation using vector search
* **Rolling Summary Memory** — Incremental conversation summarization

The benchmark evaluates:

* Factual Retention Rate (FRR)
* Context efficiency
* Memory compression tradeoffs
* LLM call overhead
* Domain sensitivity
* Long-horizon conversational robustness

The implementation accompanies the research paper:

**Context Collapse in Long-Horizon Agents: Benchmarking Hierarchical Memory against RAG and Summarization**

---

## Key Findings

* Hierarchical memory achieved **41.6% FRR**
* Baseline retained **40.6% FRR**
* Hierarchical memory reduced context usage by **~34.8%**
* Rolling Summary reduced context size substantially but introduced additional degradation and higher API usage
* Medical conversations were significantly more fragile than Legal and Technical domains
* Strong evidence of **primacy effects** appeared across architectures

---

# Project Structure

```bash
context_collapse/
│
├── analysis/                  # Evaluation and statistical analysis
│   ├── compute_efficiency.py
│   ├── get_plot_data.py
│   ├── graphical_view.py
│   ├── judge.py
│   └── p_value.py
│
├── graphs/                    # Generated figures
│   ├── plot_1_primacy_effect.png
│   ├── plot_2_efficiency_frontier.png
│   ├── plot_3_domain_heatmap.png
│   └── plot_4_memory_cost.png
│
├── llm/                        # Azure OpenAI integration
│   ├── llm_azure.py
│   └── utils.py
│
├── memory_strategies/
│   ├── base.py
│   ├── baseline.py
│   ├── hierarchical.py
│   ├── rag.py
│   └── rolling_summary.py
│
├── results/
│   ├── histories/
│   ├── efficiency_table_*.csv
│   └── scored_results.csv
│
├── scripts/                    # Conversation benchmark datasets
│   ├── all_legal.json
│   ├── all_medical.json
│   ├── all_tech.json
│   └── all_travel.json
│
├── run_benchmark.py
├── requirements.txt
└── paper.pdf
```

---

# Memory Architectures

## 1. Baseline Memory

Stores the entire conversation history.

### Characteristics

* No compression
* Maximum context growth
* Reference upper-bound for recall

---

## 2. Hierarchical Memory

Multi-layer memory architecture.

### Layers

* Working Memory
* Episodic Memory
* Semantic Memory

### Features

* Controlled compression
* Fact preservation
* Structured memory updates
* Lower token usage

---

## 3. RAG Memory

Retrieval-based memory using:

* ChromaDB
* Sentence Transformers

### Retrieval Scoring

Combines:

* Semantic similarity
* Recency weighting

---

## 4. Rolling Summary Memory

Continuously compresses dialogue into structured summaries.

### Tradeoffs

Pros:

* Lower context size

Cons:

* Additional LLM calls
* Information loss accumulation

---

# Experimental Design

## Domains

Benchmark scripts span four domains:

| Domain  | Focus                        |
| ------- | ---------------------------- |
| Legal   | Contracts, compliance        |
| Medical | Healthcare facts             |
| Tech    | Technical configurations     |
| Travel  | Long conversational planning |

---

## Benchmark Procedure

1. Inject factual information early
2. Continue conversation
3. Ask delayed recall questions
4. Retrieve memory context
5. Evaluate correctness
6. Repeat across strategies

---

## Metrics

### Factual Retention Rate (FRR)

Measures recall accuracy.

[
FRR = Correct\ Recall / Total\ Recall
]

---

### AUC-FRR

Average factual retention across recall distances.

---

### Efficiency Metrics

* Average context tokens
* LLM calls
* Compression overhead

---

# Installation

Clone repository:

```bash
git clone https://github.com/ebaadraheem/Context-Collapse_Research.git

cd context_collapse
```

Create environment:

```bash
python -m venv venv
```

Activate:

Linux / macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create `.env`

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
AZURE_DEPLOYMENT_NAME=
```

---

# Running Benchmarks

Run default benchmark:

```bash
python run_benchmark.py
```

Run custom script folder:

```bash
python run_benchmark.py \
    --scripts_dir scripts \
    --output_suffix experiment_1
```

Outputs:

```bash
results/
├── histories/
├── raw_results.csv
└── scored_results.csv
```

---

# Running Analysis

Compute efficiency:

```bash
python analysis/compute_efficiency.py \
    --scored results/scored_results.csv \
    --history results/histories/
```

Generate plots:

```bash
python analysis/graphical_view.py
```

Calculate significance:

```bash
python analysis/p_value.py
```

Judge responses:

```bash
python analysis/judge.py
```

---

# Example Research Questions

* Does compression improve factual retention?
* Does retrieval outperform summarization?
* How expensive is memory compression?
* Which domains collapse first?
* Can hierarchical memory replace full context?

---

# Results

Generated outputs include:

```bash
results/
graphs/
histories/
```

Artifacts:

* Efficiency tables
* Recall statistics
* Token usage
* Visualization plots

---

# Reproducibility

The benchmark is designed for reproducible experimentation.

Includes:

* Fixed conversation scripts
* Controlled recall positions
* Consistent evaluation pipeline
* Deterministic benchmarking structure

---

---

# Future Directions

* Multi-agent memory systems
* Agentic long-term planning
* Adaptive compression
* Memory-aware orchestration
* Production deployment benchmarks

---

# License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---
