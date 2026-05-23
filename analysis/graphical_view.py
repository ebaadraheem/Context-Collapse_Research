import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Set style for academic aesthetic
plt.style.use('ggplot')
domains = ['legal', 'medical', 'tech', 'travel']

# --- Data Loading and Prep ---

# 1. Scored Data Analysis (Overall FRR, Domain FRR)
scored_dfs = []
for d in domains:
    df = pd.read_csv(f'results/scored_{d}.csv')
    df['domain'] = d
    scored_dfs.append(df)
scored_all = pd.concat(scored_dfs, ignore_index=True)
scored_all['is_correct_lenient'] = scored_all['judgment'].isin(['CORRECT', 'PARTIAL']).astype(int)

frr_overall = scored_all.groupby('strategy')['is_correct_lenient'].mean().reset_index()
frr_domain = scored_all.groupby(['strategy', 'domain'])['is_correct_lenient'].mean().unstack()

# 2. Efficiency Analysis
eff_dfs = []
for d in domains:
    df = pd.read_csv(f'results/efficiency_table_{d}.csv')
    df['domain'] = d
    eff_dfs.append(df)
eff_all = pd.concat(eff_dfs, ignore_index=True)
eff_summary = eff_all.groupby('strategy')[['avg_context_tokens', 'llm_calls_per_run', 'context_reduction_pct']].mean().reset_index()
eff_summary = pd.merge(eff_summary, frr_overall, on='strategy')

# 3. FRR by Distance
frr_dist_dfs = []
for d in domains:
    df = pd.read_csv(f'results/frr_by_distance_{d}.csv')
    df['domain'] = d
    frr_dist_dfs.append(df)
frr_dist_all = pd.concat(frr_dist_dfs, ignore_index=True)
dist_summary = frr_dist_all.groupby(['strategy', 'distance'])[['hits', 'total']].sum().reset_index()
dist_summary['frr'] = dist_summary['hits'] / dist_summary['total']

# --- Plotting ---

# Plot 1: The Primacy Effect (Line Plot)
fig1, ax1 = plt.subplots(figsize=(10, 6))
for strategy in dist_summary['strategy'].unique():
    subset = dist_summary[dist_summary['strategy'] == strategy]
    ax1.plot(subset['distance'], subset['frr'], marker='o', label=strategy, linewidth=2)
ax1.set_title('The Primacy Effect: Factual Retention Rate Over Conversational Distance')
ax1.set_xlabel('Distance from Recall (Turns)')
ax1.set_ylabel('Factual Retention Rate (FRR)')
ax1.set_xticks([4, 9, 14, 19, 24])
ax1.legend()
plt.tight_layout()
plt.savefig('plot_1_primacy_effect.png')
plt.close(fig1)

# Plot 2: Efficiency Frontier (Scatter Plot / Bubble Chart)
# Note: Bubble size represents the number of LLM calls
fig2, ax2 = plt.subplots(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'red']
for i, strategy in enumerate(eff_summary['strategy']):
    ax2.scatter(eff_summary.loc[i, 'context_reduction_pct'], eff_summary.loc[i, 'is_correct_lenient'], 
                s=eff_summary.loc[i, 'llm_calls_per_run']*20, label=strategy, alpha=0.7)
    ax2.annotate(strategy, (eff_summary.loc[i, 'context_reduction_pct'], eff_summary.loc[i, 'is_correct_lenient']), 
                 xytext=(10, 5), textcoords='offset points')
ax2.set_title('Efficiency Frontier: Context Reduction vs. Factual Retention')
ax2.set_xlabel('Context Reduction % (Higher is Better)')
ax2.set_ylabel('Overall Factual Retention Rate (FRR)')
ax2.grid(True)
plt.tight_layout()
plt.savefig('plot_2_efficiency_frontier.png')
plt.close(fig2)

# Plot 3: Strategy vs. Domain Performance (Heatmap)
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(frr_domain * 100, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax3, cbar_kws={'label': 'FRR %'})
ax3.set_title('Domain Brittleness: Factual Retention Rate (%) by Domain')
ax3.set_xlabel('Domain')
ax3.set_ylabel('Memory Strategy')
plt.tight_layout()
plt.savefig('plot_3_domain_heatmap.png')
plt.close(fig3)

# Plot 4: The Cost of Memory (Grouped Bar Chart)
fig4, ax4a = plt.subplots(figsize=(10, 6))
x = np.arange(len(eff_summary['strategy']))
width = 0.35

ax4a.bar(x - width/2, eff_summary['avg_context_tokens'], width, label='Avg Context Tokens', color='skyblue')
ax4a.set_ylabel('Average Context Tokens', color='skyblue')
ax4a.tick_params(axis='y', labelcolor='skyblue')
ax4a.set_xticks(x)
ax4a.set_xticklabels(eff_summary['strategy'])

ax4b = ax4a.twinx()
ax4b.bar(x + width/2, eff_summary['llm_calls_per_run'], width, label='LLM Calls per Run', color='salmon')
ax4b.set_ylabel('LLM Calls per Run', color='salmon')
ax4b.tick_params(axis='y', labelcolor='salmon')

plt.title('The Cost of Memory: Context Tokens vs. LLM Calls')
fig4.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax4a.transAxes)
plt.tight_layout()
plt.savefig('plot_4_memory_cost.png')
plt.close(fig4)