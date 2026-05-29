import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ==========================================
# GLOBAL ACADEMIC STYLING
# ==========================================
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1
})

# Academic Color Palette
color_base = '#c0392b'  # Crimson Red
color_hier = '#2980b9'  # Navy Blue
color_rag = '#27ae60'   # Forest Green
color_roll = '#7f8c8d'  # Neutral Gray

# ==========================================
# PLOT 1: THE PRIMACY EFFECT
# ==========================================
def plot_primacy_effect():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # X-axis distances
    distances = [4, 9, 14, 19, 24]
    
    # Y-axis data 
    baseline = [32.0, 33.5, 43.5, 39.0, 55.0]
    hierarchical = [32.0, 36.5, 42.5, 44.0, 53.0]
    rag = [33.5, 33.5, 37.5, 40.5, 44.5]
    rolling = [24.0, 30.0, 33.0, 27.5, 40.5]

    ax.plot(distances, baseline, marker='o', markersize=8, linewidth=2.5, color=color_base, label='Baseline')
    ax.plot(distances, hierarchical, marker='s', markersize=8, linewidth=2.5, color=color_hier, label='Hierarchical')
    ax.plot(distances, rag, marker='^', markersize=8, linewidth=2.5, color=color_rag, label='RAG')
    ax.plot(distances, rolling, marker='d', markersize=8, linewidth=2.5, color=color_roll, label='Rolling Summary')

    # ax.set_title('The Primacy Effect: Factual Retention Rate vs. Distance', pad=15)
    ax.set_xlabel('Distance from Recall (Turns)')
    ax.set_ylabel('Factual Retention Rate (FRR %)')
    ax.set_xticks(distances)
    ax.legend(frameon=True, edgecolor='black')
    
    plt.savefig('plot_1_primacy_effect.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# PLOT 2: EFFICIENCY FRONTIER
# ==========================================
def plot_efficiency_frontier():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    strategies = ['Baseline', 'Hierarchical', 'RAG', 'Rolling Summary']
    reduction = [0, 34.8, 55.6, 60.5]
    frr = [40.6, 41.6, 37.9, 31.0]
    api_calls = [15.1, 25.9, 15.1, 46.0]
    colors = [color_base, color_hier, color_rag, color_roll]

    # Bubble size scale factor 
    sizes = [calls * 12 for calls in api_calls]

    scatter = ax.scatter(reduction, frr, s=sizes, c=colors, alpha=0.8, edgecolors='black', linewidth=1.5)

    # Add labels with better positioning
    for i, txt in enumerate(strategies):
        if txt == 'Rolling Summary':
            ax.annotate(txt, (reduction[i] - 1.5, frr[i] + 0.4), fontsize=10, horizontalalignment='right')
        else:
            ax.annotate(txt, (reduction[i] + 1.8, frr[i]), fontsize=10, verticalalignment='center')

    # ax.set_title('Efficiency Frontier: Context Reduction vs. Retention', pad=15)
    ax.set_xlabel('Context Token Reduction (%)')
    ax.set_ylabel('Overall Factual Retention Rate (FRR %)')
    
    ax.text(0.03, 0.05, 'Bubble Size = API Calls/Run', transform=ax.transAxes, 
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.9), fontsize=10)

    plt.savefig('plot_2_efficiency_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# PLOT 3: DOMAIN HEATMAP
# ==========================================
def plot_domain_heatmap():
    data = {
        'Legal': [88.8, 84.8, 79.6, 70.4],
        'Travel': [31.2, 35.2, 29.2, 21.2],
        'Tech': [30.4, 29.2, 27.2, 19.6],
        'Medical': [12.0, 17.2, 15.6, 12.8]
    }
    df = pd.DataFrame(data, index=['Baseline', 'Hierarchical', 'RAG', 'Rolling\nSummary'])

    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'FRR %'}, 
                linewidths=2, linecolor='white', square=True, ax=ax, annot_kws={"size": 11})

    ax.grid(False)
    ax.tick_params(axis='both', which='both', length=0)

    # ax.set_title('Domain Brittleness: Factual Retention Rate (%)', pad=15)
    # ax.set_ylabel('Memory Strategy')
    # ax.set_xlabel('Domain')
    
    plt.yticks(rotation=0)

    plt.savefig('plot_3_domain_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# PLOT 4: THE COST OF MEMORY
# ==========================================
def plot_memory_cost():
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    strategies = ['Baseline', 'Hierarchical', 'RAG', 'Rolling Summary']
    tokens = [1242, 810, 551, 491]
    api_calls = [15.1, 25.9, 15.1, 46.0]

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax1.bar(x - width/2, tokens, width, label='Avg Context Tokens', color='#3498db', edgecolor='black')
    # ax1.set_ylabel('Average Context Tokens', color='#2980b9', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.grid(False) 

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, api_calls, width, label='LLM Calls per Run', color='#e74c3c', edgecolor='black')
    # ax2.set_ylabel('LLM API Calls per Run', color='#c0392b', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#c0392b')

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
               ncol=2, frameon=False)

    # plt.title('The Cost of Memory: Tokens vs. API Calls', pad=35)
    
    plt.savefig('plot_4_memory_cost.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# EXECUTE ALL FUNCTIONS
# ==========================================
if __name__ == "__main__":
    print("Generating Plot 1...")
    plot_primacy_effect()
    print("Generating Plot 2...")
    plot_efficiency_frontier()
    print("Generating Plot 3...")
    plot_domain_heatmap()
    print("Generating Plot 4...")
    plot_memory_cost()
    print("All plots successfully generated in Elsevier-compliant format!")