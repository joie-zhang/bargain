#!/usr/bin/env python3
"""
Analyze Claude 3-7 Sonnet name experiment results for gender and race-based payoff differences.

This script:
1. Loads all Claude 3-7 Sonnet experiment results
2. Categorizes names by gender and race
3. Calculates payoff differences accounting for order effects
4. Creates visualizations showing gender and race-based differences
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set style
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.facecolor'] = 'white'

# Name mappings and categorizations
NAME_MAPPINGS = {
    "DaShawn Washington": "Deshaun Washington",
    "Dawei Li": "Li Dawei",
    "Taohua Li": "Li Taohua",
}

# Gender categorization
GIRL_NAMES = {"Claire Becker", "Tanisha Washington", "Li Taohua", "Taohua Li"}
BOY_NAMES = {"Deshaun Washington", "DaShawn Washington", "Hunter Becker", "Li Dawei", "Dawei Li"}

# Race categorization
AFRICAN_AMERICAN_NAMES = {"Deshaun Washington", "DaShawn Washington", "Tanisha Washington"}
WHITE_NAMES = {"Claire Becker", "Hunter Becker"}
CHINESE_NAMES = {"Li Dawei", "Dawei Li", "Li Taohua", "Taohua Li"}

def normalize_name(name: str) -> str:
    """Normalize name variations."""
    return NAME_MAPPINGS.get(name, name)

def get_gender(name: str) -> Optional[str]:
    """Get gender category for a name."""
    normalized = normalize_name(name)
    if normalized in GIRL_NAMES or name in GIRL_NAMES:
        return "Girl"
    elif normalized in BOY_NAMES or name in BOY_NAMES:
        return "Boy"
    return None

def get_race(name: str) -> Optional[str]:
    """Get race category for a name."""
    normalized = normalize_name(name)
    if normalized in AFRICAN_AMERICAN_NAMES or name in AFRICAN_AMERICAN_NAMES:
        return "African-American"
    elif normalized in WHITE_NAMES or name in WHITE_NAMES:
        return "White"
    elif normalized in CHINESE_NAMES or name in CHINESE_NAMES:
        return "Chinese"
    return None

def load_claude_sonnet_results(results_dir: Path) -> List[Dict]:
    """Load all Claude 3-7 Sonnet name experiment results."""
    experiments = []
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Only process Claude 3-7 Sonnet experiments
        if "claude-3-7-sonnet" not in exp_dir.name:
            continue
        
        # Skip company name experiments
        if "Anthropic" in exp_dir.name or "OpenAI" in exp_dir.name:
            continue
            
        results_file = exp_dir / "experiment_results.json"
        if not results_file.exists():
            continue
            
        try:
            with open(results_file) as f:
                data = json.load(f)
                
            config = data.get("config", {})
            agent_names = config.get("agent_names", [])
            
            if len(agent_names) != 2:
                continue
            
            # Skip if not consensus reached
            if not data.get("consensus_reached", False):
                continue
            
            final_utilities = data.get("final_utilities", {})
            if not final_utilities or len(final_utilities) != 2:
                continue
            
            # Extract competition level from directory name
            comp_level = None
            if "comp0_5" in exp_dir.name:
                comp_level = 0.5
            elif "comp1_0" in exp_dir.name:
                comp_level = 1.0
            
            experiments.append({
                "name1": agent_names[0],
                "name2": agent_names[1],
                "utility1": final_utilities.get(agent_names[0], 0),
                "utility2": final_utilities.get(agent_names[1], 0),
                "competition_level": comp_level,
                "experiment_id": data.get("experiment_id", ""),
                "directory": exp_dir.name
            })
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
            continue
    
    return experiments

def process_experiments(experiments: List[Dict]) -> pd.DataFrame:
    """Process experiments into a DataFrame with gender and race categories."""
    rows = []
    
    for exp in experiments:
        name1 = exp["name1"]
        name2 = exp["name2"]
        util1 = exp["utility1"]
        util2 = exp["utility2"]
        
        gender1 = get_gender(name1)
        gender2 = get_gender(name2)
        race1 = get_race(name1)
        race2 = get_race(name2)
        
        # Skip if we can't categorize both names
        if not gender1 or not gender2 or not race1 or not race2:
            continue
        
        # Calculate payoff difference (name1 - name2)
        payoff_diff = util1 - util2
        
        rows.append({
            "name1": name1,
            "name2": name2,
            "gender1": gender1,
            "gender2": gender2,
            "race1": race1,
            "race2": race2,
            "utility1": util1,
            "utility2": util2,
            "payoff_diff": payoff_diff,
            "competition_level": exp["competition_level"],
            "order": "name1_first"  # Track which name appears first
        })
    
    return pd.DataFrame(rows)

def create_gender_plot(df: pd.DataFrame, output_dir: Path):
    """Create plot showing gender-based payoff differences, disaggregated by competition level."""
    # Filter to cross-gender comparisons
    gender_df = df[df["gender1"] != df["gender2"]].copy()
    
    if len(gender_df) == 0:
        print("No cross-gender comparisons found!")
        return
    
    # Create a column for the gender of the first agent
    gender_df["first_agent_gender"] = gender_df["gender1"]
    
    # Create 2x2 subplot: rows = competition level, cols = plot type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    competition_levels = [0.5, 1.0]
    gender_order = ["Boy", "Girl"]
    
    for row_idx, comp_level in enumerate(competition_levels):
        comp_df = gender_df[gender_df["competition_level"] == comp_level].copy()
        
        if len(comp_df) == 0:
            continue
        
        # Plot 1: Payoff difference by gender (first agent)
        ax1 = axes[row_idx, 0]
        data_to_plot = []
        labels = []
        
        for gender in gender_order:
            subset = comp_df[comp_df["first_agent_gender"] == gender]
            if len(subset) > 0:
                data_to_plot.append(subset["payoff_diff"].values)
                labels.append(f"{gender}\n(n={len(subset)})")
        
        if data_to_plot:
            bp1 = ax1.boxplot(data_to_plot, tick_labels=[l.split("\n")[0] for l in labels], 
                             patch_artist=True, showmeans=True)
            colors = ['lightblue', 'lightpink']
            for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
                patch.set_facecolor(color)
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Equal Payoff')
            ax1.set_ylabel('Payoff Difference', fontsize=10)
            ax1.set_xlabel('Gender of First Agent', fontsize=10)
            ax1.set_title(f'Competition Level {comp_level}\nPayoff by Gender (First Agent)', 
                         fontsize=11, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Payoff difference by gender pair type, accounting for order
        ax2 = axes[row_idx, 1]
        
        # Create comparison labels
        comp_df["comparison"] = comp_df.apply(
            lambda row: f"{row['gender1']} vs {row['gender2']}", axis=1
        )
        
        comparison_order = ["Boy vs Girl", "Girl vs Boy"]
        data_to_plot2 = []
        labels2 = []
        
        for comp in comparison_order:
            subset = comp_df[comp_df["comparison"] == comp]
            if len(subset) > 0:
                data_to_plot2.append(subset["payoff_diff"].values)
                labels2.append(f"{comp}\n(n={len(subset)})")
        
        if data_to_plot2:
            bp2 = ax2.boxplot(data_to_plot2, tick_labels=[l.split("\n")[0] for l in labels2],
                             patch_artist=True, showmeans=True)
            colors = ['lightcoral', 'lightgreen']
            for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
                patch.set_facecolor(color)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Equal Payoff')
            ax2.set_ylabel('Payoff Difference', fontsize=10)
            ax2.set_xlabel('Gender Comparison', fontsize=10)
            ax2.set_title(f'Competition Level {comp_level}\nPayoff by Gender Pair (Order)', 
                         fontsize=11, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Gender-Based Payoff Differences by Competition Level', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "gender_payoff_differences.png", dpi=300, bbox_inches='tight')
    print(f"Saved gender plot to {output_dir / 'gender_payoff_differences.png'}")
    plt.close()

def create_race_plot(df: pd.DataFrame, output_dir: Path):
    """Create plot showing race-based payoff differences, disaggregated by competition level."""
    # Filter to cross-race comparisons
    race_df = df[df["race1"] != df["race2"]].copy()
    
    if len(race_df) == 0:
        print("No cross-race comparisons found!")
        return
    
    # Create comparison labels
    race_df["comparison"] = race_df.apply(
        lambda row: f"{row['race1']} vs {row['race2']}", axis=1
    )
    
    # Create 2x2 subplot: rows = competition level, cols = plot type
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    competition_levels = [0.5, 1.0]
    race_order = ["White", "African-American", "Chinese"]
    
    for row_idx, comp_level in enumerate(competition_levels):
        comp_df = race_df[race_df["competition_level"] == comp_level].copy()
        
        if len(comp_df) == 0:
            continue
        
        # Plot 1: Payoff difference by race (first agent)
        ax1 = axes[row_idx, 0]
        data_to_plot = []
        labels = []
        
        for race in race_order:
            subset = comp_df[comp_df["race1"] == race]
            if len(subset) > 0:
                data_to_plot.append(subset["payoff_diff"].values)
                labels.append(f"{race}\n(n={len(subset)})")
        
        if data_to_plot:
            bp1 = ax1.boxplot(data_to_plot, tick_labels=[l.split("\n")[0] for l in labels],
                             patch_artist=True, showmeans=True)
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
                patch.set_facecolor(color)
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Equal Payoff')
            ax1.set_ylabel('Payoff Difference', fontsize=10)
            ax1.set_xlabel('Race of First Agent', fontsize=10)
            ax1.set_title(f'Competition Level {comp_level}\nPayoff by Race (First Agent)', 
                         fontsize=11, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=15)
        
        # Plot 2: All race pair comparisons
        ax2 = axes[row_idx, 1]
        
        # Get all unique comparisons for this competition level
        comparisons = sorted(comp_df["comparison"].unique())
        data_to_plot2 = []
        labels2 = []
        
        for comp in comparisons:
            subset = comp_df[comp_df["comparison"] == comp]
            if len(subset) > 0:
                data_to_plot2.append(subset["payoff_diff"].values)
                labels2.append(f"{comp}\n(n={len(subset)})")
        
        if data_to_plot2:
            bp2 = ax2.boxplot(data_to_plot2, tick_labels=[l.split("\n")[0] for l in labels2],
                             patch_artist=True, showmeans=True)
            # Use different colors for different comparisons
            num_boxes = len(bp2['boxes'])
            colors = plt.cm.Set3(np.linspace(0, 1, num_boxes))
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Equal Payoff')
            ax2.set_ylabel('Payoff Difference', fontsize=10)
            ax2.set_xlabel('Race Comparison', fontsize=10)
            ax2.set_title(f'Competition Level {comp_level}\nPayoff by Race Pair (Order)', 
                         fontsize=11, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Race-Based Payoff Differences by Competition Level', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "race_payoff_differences.png", dpi=300, bbox_inches='tight')
    print(f"Saved race plot to {output_dir / 'race_payoff_differences.png'}")
    plt.close()

def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics, disaggregated by competition level."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Gender statistics by competition level
    print("\n--- GENDER ANALYSIS ---")
    gender_df = df[df["gender1"] != df["gender2"]].copy()
    if len(gender_df) > 0:
        gender_df["first_agent_gender"] = gender_df["gender1"]
        print(f"\nTotal cross-gender comparisons: {len(gender_df)}")
        
        for comp_level in [0.5, 1.0]:
            comp_gender_df = gender_df[gender_df["competition_level"] == comp_level]
            if len(comp_gender_df) > 0:
                print(f"\n  Competition Level {comp_level}:")
                for gender in ["Boy", "Girl"]:
                    subset = comp_gender_df[comp_gender_df["first_agent_gender"] == gender]
                    if len(subset) > 0:
                        mean_diff = subset["payoff_diff"].mean()
                        std_diff = subset["payoff_diff"].std()
                        print(f"    {gender} as first agent:")
                        print(f"      Mean payoff difference: {mean_diff:.2f}")
                        print(f"      Std payoff difference: {std_diff:.2f}")
                        print(f"      N: {len(subset)}")
    
    # Race statistics by competition level
    print("\n--- RACE ANALYSIS ---")
    race_df = df[df["race1"] != df["race2"]].copy()
    if len(race_df) > 0:
        print(f"\nTotal cross-race comparisons: {len(race_df)}")
        
        for comp_level in [0.5, 1.0]:
            comp_race_df = race_df[race_df["competition_level"] == comp_level]
            if len(comp_race_df) > 0:
                print(f"\n  Competition Level {comp_level}:")
                for race in ["White", "African-American", "Chinese"]:
                    subset = comp_race_df[comp_race_df["race1"] == race]
                    if len(subset) > 0:
                        mean_diff = subset["payoff_diff"].mean()
                        std_diff = subset["payoff_diff"].std()
                        print(f"    {race} as first agent:")
                        print(f"      Mean payoff difference: {mean_diff:.2f}")
                        print(f"      Std payoff difference: {std_diff:.2f}")
                        print(f"      N: {len(subset)}")

def main():
    """Main analysis function."""
    results_dir = Path("experiments/results/name_results")
    output_dir = Path("experiments/results/name_results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print("Loading Claude 3-7 Sonnet name experiment results...")
    experiments = load_claude_sonnet_results(results_dir)
    
    if not experiments:
        print("No experiment results found!")
        return 1
    
    print(f"Loaded {len(experiments)} experiments\n")
    
    # Process into DataFrame
    print("Processing experiments...")
    df = process_experiments(experiments)
    print(f"Processed {len(df)} data points\n")
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create plots
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)
    
    create_gender_plot(df, output_dir)
    create_race_plot(df, output_dir)
    
    # Save processed data
    df.to_csv(output_dir / "processed_data.csv", index=False)
    print(f"\nSaved processed data to {output_dir / 'processed_data.csv'}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

