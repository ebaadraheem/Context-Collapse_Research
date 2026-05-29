import pandas as pd
from scipy import stats

CSV_FILES = ["results/scored_legal.csv", "results/scored_medical.csv", "results/scored_tech.csv", "results/scored_travel.csv"]

# Load and prepare data
df = pd.concat([pd.read_csv(f) for f in CSV_FILES], ignore_index=True)
df['is_success'] = df['judgment'].isin(['CORRECT', 'PARTIAL']).astype(int)

# Group by script_id and repetition to get paired scores
paired_data = df.groupby(['script_id', 'repetition', 'strategy'])['is_success'].mean().unstack()

# Run Paired T-Test between Hierarchical and Baseline
t_stat, p_val = stats.ttest_rel(paired_data['hierarchical'], paired_data['baseline'])

print(f"P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Result is STATISTICALLY SIGNIFICANT!")
else:
    print("Result is NOT statistically significant.")