import pandas as pd
from pathlib import Path

def process_raw_scored_data():
    print("\n" + "="*60)
    print(" 🛠️ RECALCULATING PLOT 1 DATA FROM RAW SCORES (LENIENT)")
    print("="*60)
    
    # Search for raw scored files
    search_dirs = [Path('.'), Path('results'), Path('analysis/results')]
    valid_files = []
    
    for directory in search_dirs:
        if not directory.exists():
            continue
        for f in directory.glob('*scored*.csv'):
            valid_files.append(f)
            
    valid_files = list(set(valid_files))
    
    if not valid_files:
        print("❌ No files matching '*scored*.csv' found.")
        return

    print(f"✅ Found {len(valid_files)} raw scored CSV files. Processing...\n")
    
    dfs = []
    for f in valid_files:
        temp_df = pd.read_csv(f)
        temp_df.columns = [str(c).lower().strip() for c in temp_df.columns]
        dfs.append(temp_df)
        
    df = pd.concat(dfs, ignore_index=True)
    
    df['standard_distance'] = df['turn'] - 1 

    # --- 2. APPLY LENIENT SCORING (THE BUG FIX) ---
    def calculate_score(val):
        val = str(val).upper()
        if 'INCORRECT' in val:
            return 0  # Fail it FIRST so it doesn't trigger the 'CORRECT' check
        elif 'CORRECT' in val or 'PARTIAL' in val:
            return 1
        return 0
        
    df['is_success'] = df['judgment'].apply(calculate_score)

    # --- 3. CALCULATE AVERAGES ---
    grouped = df.groupby(['strategy', 'standard_distance'])['is_success'].mean().unstack() * 100
    distances = sorted([int(d) for d in grouped.columns.tolist()])
    
    print("--- PASTE THESE ARRAYS INTO plot_1_primacy_effect.py ---\n")
    print(f"distances = {distances}\n")
    
    for strategy in grouped.index:
        # float() removes the ugly np.float64 text
        values = [round(float(grouped.loc[strategy, d]), 1) for d in distances]
        var_name = str(strategy).lower().replace(" ", "_")
        print(f"{var_name} = {values}")

    # --- 4. PROOF CHECK ---
    print("\n" + "-"*60)
    print("🎯 PROOF CHECK: OVERALL AVERAGES (Must match Table 1)")
    print("-"*60)
    overall_mean = df.groupby('strategy')['is_success'].mean() * 100
    for strat, val in overall_mean.items():
        print(f"{strat.capitalize()}: {val:.1f}%")

if __name__ == "__main__":
    process_raw_scored_data()