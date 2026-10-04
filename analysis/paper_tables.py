"""Reproduce the probe-validity audit and the tables reported in the paper.

Run from the repository root:
    python analysis/paper_tables.py

Inputs : scripts/all_<domain>.json, results/scored_<domain>.csv,
         results/efficiency_table_legal.csv (cost columns, pooled over domains)
Outputs: results/probe_validity_check.csv  (one row per probe, with an empty
         'author_label' column for manual checking), and printed tables.

Conventions
-----------
* Unit of analysis: the script. Repetitions are averaged within a script, and
  scripts are weighted equally. Intervals are 95% percentile bootstrap
  intervals from resampling scripts (N_BOOT resamples, fixed seed).
* Lenient retention counts CORRECT + PARTIAL; strict counts CORRECT only.
* A "refusal" is an answer containing "I don't know".
* A probe is "answerable" if >= THRESHOLD of the key tokens of its shortest
  accepted answer appear in the script's user turns before the probe, and all
  numeric tokens appear.
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

DOMAINS = ["legal", "medical", "tech", "travel"]
STRATS = ["baseline", "hierarchical", "rag", "rolling_summary"]
NAMES = {"baseline": "Full history", "hierarchical": "Hierarchical",
         "rag": "RAG", "rolling_summary": "Rolling summary"}
THRESHOLD = 0.6
N_BOOT = 4000
SEED = 0
STOP = set("the and that with from this were was have has had for are its their "
           "they them been will would which about into also than then when where "
           "while after before over under more most such only some each other "
           "these those".split())


def key_tokens(text):
    toks = re.findall(r"[a-z0-9$%.:/_-]+", text.lower())
    toks = [w.strip(".:,") for w in toks]
    return [w for w in toks if w and ((len(w) > 3 and w not in STOP) or any(c.isdigit() for c in w))]


def audit_probes():
    rows = []
    for dom in DOMAINS:
        for sc in json.load(open(f"scripts/all_{dom}.json")):
            for i, t in enumerate(sc["turns"]):
                if not t.get("is_recall"):
                    continue
                prev = sc["turns"][:i]
                user_text = " ".join(x["content"] for x in prev if x["role"] == "user").lower()
                asst_text = " ".join(x["content"] for x in prev if x["role"] == "assistant").lower()
                gt = min(t["ground_truth"], key=len)
                kt = key_tokens(gt)
                nums = [w for w in kt if any(c.isdigit() for c in w)]
                user_cov = sum(w in user_text for w in kt) / max(1, len(kt))
                asst_cov = sum(w in asst_text for w in kt) / max(1, len(kt))
                sents = [s for x in prev if x["role"] == "user"
                         for s in re.split(r"(?<=[.?!])\s+", x["content"])]
                best = max(sents, key=lambda s: sum(w in s.lower() for w in kt), default="")
                rows.append(dict(
                    domain=dom, script_id=sc["script_id"], turn=t["turn"],
                    fact=t["target_fact_id"], ground_truth=gt,
                    user_cov=round(user_cov, 2), asst_cov=round(asst_cov, 2),
                    numbers_ok_user=all(w in user_text for w in nums),
                    numbers_ok_asst=all(w in asst_text for w in nums),
                    best_matching_user_sentence=best[:200]))
    a = pd.DataFrame(rows)
    for th in (0.5, 0.6, 0.8):
        a[f"answerable@{th}"] = (a.user_cov >= th) & a.numbers_ok_user
    a["in_assistant_turns_only"] = (~a["answerable@0.6"]) & (a.asst_cov >= 0.6) & a.numbers_ok_asst
    a["auto_label"] = np.where(a["answerable@0.6"], "present", "absent")
    a["author_label (present/absent)"] = ""
    return a


def load_scored(audit):
    df = pd.concat([pd.read_csv(f"results/scored_{d}.csv").assign(domain=d) for d in DOMAINS],
                   ignore_index=True)
    df = df.merge(audit[["script_id", "turn"] + [c for c in audit if c.startswith("answerable@")]],
                  on=["script_id", "turn"], how="left")
    df["refusal"] = df.agent_response.astype(str).str.lower().str.contains(r"i don'?t know")
    cat = np.where(df.refusal, "Refusal", df.judgment.str.capitalize())
    for c in ("Correct", "Partial", "Incorrect", "Refusal"):
        df[c] = (cat == c).astype(int)
    df["lenient"] = df.judgment.isin(["CORRECT", "PARTIAL"]).astype(int)
    df["strict"] = (df.judgment == "CORRECT").astype(int)
    df["refusals"] = df.refusal.astype(int)
    return df


def script_means(d, col):
    """script x strategy matrix of means (repetitions and probes averaged within script)."""
    return d.groupby(["script_id", "strategy"])[col].mean().unstack()[STRATS] * 100


def boot_ci(x, rng):
    x = np.asarray(x.dropna())
    b = [rng.choice(x, len(x)).mean() for _ in range(N_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def fmt(x, ci):
    return f"{x:5.1f} [{ci[0]:5.1f}, {ci[1]:5.1f}]"


def report(d, label, cols=("lenient", "strict", "refusals"), diff_col="lenient"):
    rng = np.random.default_rng(SEED)
    n_probes = d.drop_duplicates(["script_id", "turn"]).shape[0]
    print(f"\n=== {label}: {n_probes} probes, {d.script_id.nunique()} scripts ===")
    for c in cols:
        m = script_means(d, c)
        print(f"  {c:10s}", " | ".join(f"{NAMES[s]}: {fmt(m[s].mean(), boot_ci(m[s], rng))}" for s in STRATS))
    for c in (diff_col, "strict"):
        m = script_means(d, c)
        for s in STRATS[1:]:
            diff = (m[s] - m["baseline"]).dropna()
            ci = boot_ci(diff, rng)
            try:
                p = stats.wilcoxon(diff).pvalue
            except ValueError:
                p = float("nan")
            print(f"  diff {c:8s} {NAMES[s]:15s} - full history: {diff.mean():6.1f} "
                  f"[{ci[0]:5.1f}, {ci[1]:5.1f}]  Wilcoxon p={p:.3f}")


def main():
    audit = audit_probes()
    audit.to_csv("results/probe_validity_check.csv", index=False)
    print("answerable probes by domain (threshold 0.6):")
    print(audit.groupby("domain")["answerable@0.6"].agg(["sum", "count"]).T.to_string())
    n_other = (~audit["answerable@0.6"]).sum()
    print(f"unanswerable: {n_other}; of these, found only in scripted assistant turns: "
          f"{audit.in_assistant_turns_only.sum()}; found in neither: {n_other - audit.in_assistant_turns_only.sum()}")

    df = load_scored(audit)
    vis = df[df["answerable@0.6"]]
    oth = df[~df["answerable@0.6"]]

    report(vis, "ANSWERABLE (Table 2)")
    print("\n  Outcome mix, answerable probes, script-averaged (Table 3)")
    print((df[df['answerable@0.6']].groupby(["script_id", "strategy"])[["Correct", "Partial", "Incorrect", "Refusal"]]
           .mean().groupby("strategy").mean().loc[STRATS] * 100).round(1).to_string())
    report(oth, "UNANSWERABLE / OTHER (Table 4)", cols=("lenient", "refusals"))
    report(df, "ALL 200 PROBES (pooled)", cols=("lenient", "strict"))

    for th in (0.5, 0.8):
        sub = df[df[f"answerable@{th}"]]
        m = script_means(sub, "lenient").mean().round(1)
        print(f"\nSensitivity, threshold {th}: {sub.drop_duplicates(['script_id','turn']).shape[0]} probes, "
              f"lenient (script-averaged): " + " / ".join(f"{m[s]}" for s in STRATS))

    eff = pd.read_csv("results/efficiency_table_legal.csv")
    print("\nCost columns (pooled over domains; identical in all four efficiency tables):")
    print(eff.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
