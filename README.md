# Retention, Tokens, and Calls: Four Conversation-Memory Strategies

Code, synthetic scripts, and logs for the paper *Token Savings Are Not Free: Retention and Call Costs of Four Memory Strategies* (pilot study).

We compare four memory strategies with one model (GPT-4.1-mini) on 40 synthetic 25-turn scripts (10 each in Legal, Medical, Tech, Travel; 5 repetitions per script and strategy; 4,000 scored answers):

| Strategy | Context supplied to the model |
|---|---|
| Full history (`baseline`) | All previous messages |
| Rolling summary (`rolling_summary`) | Bullet-point fact list plus unsummarized messages (re-summarized above 400 estimated tokens; summary output capped at 400 tokens) |
| RAG (`rag`) | Retrieved sentence chunks plus the last 6 messages (all-MiniLM-L6-v2, Chroma, top 8 of 16 candidates, score 0.7·similarity + 0.3·1/(1+age)) |
| Hierarchical (`hierarchical`) | Semantic fact list, episodic entries, recent raw messages (compression above 300 / 600 estimated tokens; both steps use the model) |

## Important limitation: probe answerability

Each script has five recall probes (turns 5, 10, 15, 20, 25), 200 in total. For many probes, the target fact is **not stated in the user turns that the model sees** (only user turns are replayed; scripted assistant turns are not). A rule-based audit finds that this holds for only **40 of 200 probes** (Legal 24, Medical 3, Tech 8, Travel 5). The paper's main comparison uses these 40 probes; the other 160 mostly produce refusals under every strategy.

* The per-probe audit is in `results/probe_validity_check.csv` (column `auto_label` is the rule's output; `author_label (present/absent)` is for manual checking).
* Results by domain or recall distance are therefore **not** meaningful with the scripts as released.
* If you reuse these scripts, regenerate the Medical, Tech, and Travel scripts so that every probed fact is stated in a user turn, and run the audit first.

## Reproducing the paper's tables (no API access needed)

```bash
pip install -r requirements.txt
python analysis/paper_tables.py
```

This reads `scripts/all_*.json` and `results/scored_*.csv`, writes `results/probe_validity_check.csv`, and prints the numbers behind the audit and the results tables (script-level means, 95% bootstrap intervals from resampling scripts, seed 0, 4,000 resamples). Cost columns (context tokens, LLM calls) come from `results/efficiency_table_*.csv`, which are computed from the saved histories and are pooled over all domains (the four files are identical).

## Re-running the experiments (needs Azure OpenAI access)

```bash
cp .env.example .env   # then fill in the values below
```

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
AZURE_DEPLOYMENT_NAME=      # model that answers and compresses (GPT-4.1-mini in the paper)
```

```bash
# 1. Run all strategies on a script folder (results go to results/)
python run_benchmark.py --scripts_dir scripts --output_suffix run1

# 2. Score the answers: exact match, then LLM judge
#    NOTE: analysis/score_results.py reads the same AZURE_DEPLOYMENT_NAME variable.
#    Set it to the judge deployment before this step.  <FILL IN: judge model and version used for the paper>
python analysis/score_results.py results/benchmark_results_<timestamp>_run1.csv

# 3. Cost columns from the saved histories
python analysis/compute_efficiency.py --scored results/scored_<timestamp>.csv --history results/histories/
```

Answers are generated with the model's default sampling settings, so re-running will not reproduce the released logs exactly.

## Data notes

* `results/scored_<domain>.csv`: one row per answer (script, strategy, repetition, probe turn, question, response, ground truth, judgment). 800 runs → 4,000 rows.
* `results/histories/`: saved conversation histories. 795 files are included (5 of the 800 runs have no saved history); cost figures are computed from these.
* Judgments: `CORRECT`, `PARTIAL` (part of a compound fact recalled; see `analysis/judge.py`), `INCORRECT`. Strict retention counts `CORRECT`; lenient counts `CORRECT` + `PARTIAL`. "I don't know" answers are counted as incorrect and reported as refusals.
* Context tokens are estimated as characters / 4 at the five probe turns; they are not billed token counts.
* `analysis/p_value.py` is an earlier run-level paired t-test and is **not** used in the paper. The paper's intervals come from `analysis/paper_tables.py`.

## Layout

```
analysis/        scoring, judge, efficiency, paper_tables.py (audit + paper numbers)
llm/             Azure OpenAI client
memory_strategies/   baseline, rolling_summary, rag, hierarchical
scripts/         synthetic conversation scripts (all_<domain>.json)
results/         logs, scored answers, efficiency tables, probe audit
run_benchmark.py experiment runner
```

## License

MIT (see `LICENSE`).
