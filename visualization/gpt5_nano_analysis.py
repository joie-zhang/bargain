#!/usr/bin/env python
# coding: utf-8

# # GPT-5-nano Negotiation Analysis
# 
# This notebook analyzes negotiation experiments where GPT-5-nano is paired against various models across different tiers.
# 
# ## Model Tiers (from generate_configs_both_orders.sh)
# - **STRONG TIER** (Elo >= 1415): gemini-3-pro, gemini-3-flash, claude-opus-4-5-thinking-32k, claude-opus-4-5, 
#   claude-sonnet-4-5, glm-4.7, gpt-5.2-high, qwen3-max, deepseek-r1-0528, grok-4
# - **MEDIUM TIER** (1290 <= Elo < 1415): claude-haiku-4-5, deepseek-r1, claude-sonnet-4, claude-3.5-sonnet,
#   gemma-3-27b-it, o3-mini-high, deepseek-v3, gpt-4o, QwQ-32B, llama-3.3-70b-instruct, Qwen2.5-72B-Instruct,
#   gemma-2-27b-it, Meta-Llama-3-70B-Instruct, claude-3-haiku, phi-4
# - **WEAK TIER** (Elo < 1290): amazon-nova-micro, mixtral-8x22b-instruct-v0.1, gpt-3.5-turbo-0125,
#   llama-3.1-8b-instruct, mixtral-8x7b-instruct-v0.1, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.2,
#   Phi-3-mini-128k-instruct, Llama-3.2-1B-Instruct
# 
# ## Competition Levels
# Experiments use competition levels: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

# In[ ]:


import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


# In[ ]:


# Baseline model (GPT-5-nano) Elo rating
# Source: Hugging Face Chatbot Arena Leaderboard (as of January 16, 2026)
# gpt-5-nano-high has Elo 1338
BASELINE_ELO = 1338
BASELINE_MODEL = "gpt-5-nano"

# Adversary model tier definitions with Elo ratings
# Updated to match experiments from generate_configs_both_orders.sh
# Source: Hugging Face Chatbot Arena Leaderboard (as of January 16, 2026)
MODEL_INFO = {
    # Strong tier (Elo >= 1415)
    "gemini-3-pro": {"tier": "Strong", "elo": 1490, "source": "Closed", "reasoning": False},
    "gemini-3-flash": {"tier": "Strong", "elo": 1472, "source": "Closed", "reasoning": False},
    "claude-opus-4-5-thinking-32k": {"tier": "Strong", "elo": 1470, "source": "Closed", "reasoning": True},
    "claude-opus-4-5": {"tier": "Strong", "elo": 1467, "source": "Closed", "reasoning": False},
    "claude-sonnet-4-5": {"tier": "Strong", "elo": 1450, "source": "Closed", "reasoning": False},
    "glm-4.7": {"tier": "Strong", "elo": 1441, "source": "Open", "reasoning": False},
    "gpt-5.2-high": {"tier": "Strong", "elo": 1436, "source": "Closed", "reasoning": True},
    "qwen3-max": {"tier": "Strong", "elo": 1434, "source": "Open", "reasoning": False},
    "deepseek-r1-0528": {"tier": "Strong", "elo": 1418, "source": "Open", "reasoning": True},
    "grok-4": {"tier": "Strong", "elo": 1409, "source": "Closed", "reasoning": False},
    
    # Medium tier (1290 <= Elo < 1415)
    "claude-haiku-4-5": {"tier": "Medium", "elo": 1403, "source": "Closed", "reasoning": False},
    "deepseek-r1": {"tier": "Medium", "elo": 1397, "source": "Open", "reasoning": True},
    "claude-sonnet-4": {"tier": "Medium", "elo": 1390, "source": "Closed", "reasoning": False},
    "claude-3.5-sonnet": {"tier": "Medium", "elo": 1373, "source": "Closed", "reasoning": False},
    "gemma-3-27b-it": {"tier": "Medium", "elo": 1365, "source": "Open", "reasoning": False},
    "o3-mini-high": {"tier": "Medium", "elo": 1364, "source": "Closed", "reasoning": True},
    "deepseek-v3": {"tier": "Medium", "elo": 1358, "source": "Open", "reasoning": False},
    "gpt-4o": {"tier": "Medium", "elo": 1346, "source": "Closed", "reasoning": False},
    "QwQ-32B": {"tier": "Medium", "elo": 1336, "source": "Open", "reasoning": False},
    "llama-3.3-70b-instruct": {"tier": "Medium", "elo": 1320, "source": "Open", "reasoning": False},
    "Qwen2.5-72B-Instruct": {"tier": "Medium", "elo": 1303, "source": "Open", "reasoning": False},
    "gemma-2-27b-it": {"tier": "Medium", "elo": 1288, "source": "Open", "reasoning": False},
    "Meta-Llama-3-70B-Instruct": {"tier": "Medium", "elo": 1277, "source": "Open", "reasoning": False},
    "claude-3-haiku": {"tier": "Medium", "elo": 1262, "source": "Closed", "reasoning": False},
    "phi-4": {"tier": "Medium", "elo": 1256, "source": "Open", "reasoning": False},
    
    # Weak tier (Elo < 1290)
    "amazon-nova-micro": {"tier": "Weak", "elo": 1241, "source": "Closed", "reasoning": False},
    "mixtral-8x22b-instruct-v0.1": {"tier": "Weak", "elo": 1231, "source": "Open", "reasoning": False},
    "gpt-3.5-turbo-0125": {"tier": "Weak", "elo": 1225, "source": "Closed", "reasoning": False},
    "llama-3.1-8b-instruct": {"tier": "Weak", "elo": 1212, "source": "Open", "reasoning": False},
    "mixtral-8x7b-instruct-v0.1": {"tier": "Weak", "elo": 1198, "source": "Open", "reasoning": False},
    "Llama-3.2-3B-Instruct": {"tier": "Weak", "elo": 1167, "source": "Open", "reasoning": False},
    "Mistral-7B-Instruct-v0.2": {"tier": "Weak", "elo": 1151, "source": "Open", "reasoning": False},
    "Phi-3-mini-128k-instruct": {"tier": "Weak", "elo": 1130, "source": "Open", "reasoning": False},
    "Llama-3.2-1B-Instruct": {"tier": "Weak", "elo": 1112, "source": "Open", "reasoning": False},
    
    # Legacy models (kept for backward compatibility with older experiments)
    "claude-4-5-haiku": {"tier": "Medium", "elo": 1378, "source": "Closed", "reasoning": False},  # Alias for claude-haiku-4-5
    "o4-mini-2025-04-16": {"tier": "Medium", "elo": 1362, "source": "Closed", "reasoning": True},
    "gpt-oss-20b": {"tier": "Medium", "elo": 1315, "source": "Open", "reasoning": False},
    "kimi-k2-thinking": {"tier": "Strong", "elo": 1438, "source": "Open", "reasoning": True},
    "qwen3-235b-a22b-instruct-2507": {"tier": "Strong", "elo": 1418, "source": "Open", "reasoning": False},
}

TIER_ORDER = ["Strong", "Medium", "Weak"]
TIER_COLORS = {"Strong": "#e74c3c", "Medium": "#f39c12", "Weak": "#27ae60"}


# In[ ]:


# Define experiment directories to search
RESULTS_DIR = Path("/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results")

# Define figures directory relative to script location
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SCALING_EXPERIMENTS = [
    # "scaling_experiment_20260117_185910",  # Most recent
    # "scaling_experiment_20260117_172734",
    # "scaling_experiment_20260116_052234",
    # "scaling_experiment_old_20260116_052119",
    "scaling_experiment_20260121_070359",
]


# ## Data Loading and Filtering

# In[ ]:


def load_experiment_result(result_path: Path) -> Optional[Dict]:
    """Load a single experiment result file."""
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return None


def is_successful_experiment(result: Dict) -> bool:
    """Check if an experiment completed successfully.
    
    Filters out:
    - Experiments that didn't reach consensus (optional - we may want to include these)
    - Experiments with missing/invalid data
    - API failures (usually indicated by empty or malformed results)
    """
    if result is None:
        return False
    
    # Must have key fields
    required_fields = ['final_utilities', 'final_round', 'config', 'agent_preferences']
    for field in required_fields:
        if field not in result:
            return False
    
    # Must have valid utilities (not None or empty)
    utilities = result.get('final_utilities', {})
    if not utilities or len(utilities) < 2:
        return False
    
    # Utilities should be numeric
    for agent, util in utilities.items():
        if util is None or not isinstance(util, (int, float)):
            return False
    
    # Must have valid preferences
    prefs = result.get('agent_preferences', {})
    if not prefs or len(prefs) < 2:
        return False
    
    return True


def parse_model_pair_from_path(path: Path) -> Tuple[str, str]:
    """Extract model pair from experiment path.
    
    Path format: .../gpt-5-nano_vs_gemini-3-pro/weak_first/comp_0.5/run_1/
    Returns: (gpt-5-nano, gemini-3-pro)
    """
    parts = path.parts
    for part in parts:
        if '_vs_' in part:
            models = part.split('_vs_')
            if len(models) == 2:
                return tuple(models)
    return (None, None)


def parse_experiment_config_from_path(path: Path) -> Dict:
    """Extract experiment configuration from path.
    
    Returns: {model_order, competition_level, run_number, scaling_experiment}
    """
    parts = path.parts
    config = {
        'model_order': None,
        'competition_level': None,
        'run_number': None,
        'scaling_experiment': None
    }
    
    for i, part in enumerate(parts):
        if part.startswith('scaling_experiment'):
            config['scaling_experiment'] = part
        elif part in ['weak_first', 'strong_first']:
            config['model_order'] = part
        elif part.startswith('comp_'):
            try:
                config['competition_level'] = float(part.replace('comp_', ''))
            except ValueError:
                pass
        elif part.startswith('run_'):
            try:
                config['run_number'] = int(part.replace('run_', ''))
            except ValueError:
                pass
    
    return config


# In[ ]:


def discover_gpt5_nano_experiments() -> List[Dict]:
    """Discover all GPT-5-nano experiments across scaling experiment directories."""
    all_experiments = []
    
    for scaling_exp in SCALING_EXPERIMENTS:
        exp_dir = RESULTS_DIR / scaling_exp
        if not exp_dir.exists():
            print(f"Skipping missing directory: {scaling_exp}")
            continue
        
        # Find all gpt-5-nano model pair directories
        for model_pair_dir in exp_dir.glob("gpt-5-nano_vs_*"):
            if not model_pair_dir.is_dir():
                continue
            
            weak_model, strong_model = parse_model_pair_from_path(model_pair_dir)
            if weak_model != 'gpt-5-nano':
                continue
            
            # Find all experiment result files
            for result_file in model_pair_dir.glob("**/experiment_results.json"):
                result = load_experiment_result(result_file)
                
                if not is_successful_experiment(result):
                    continue
                
                path_config = parse_experiment_config_from_path(result_file)
                
                experiment_data = {
                    'file_path': str(result_file),
                    'weak_model': weak_model,
                    'strong_model': strong_model,
                    'adversary_model': strong_model,  # Since weak_model is always gpt-5-nano
                    **path_config,
                    **result
                }
                
                all_experiments.append(experiment_data)
    
    return all_experiments


# Load all experiments
print("Discovering GPT-5-nano experiments...")
experiments = discover_gpt5_nano_experiments()
print(f"Found {len(experiments)} successful experiments")


# In[ ]:


# Convert to DataFrame for easier analysis
def extract_experiment_features(exp: Dict) -> Dict:
    """Extract key features from experiment data for DataFrame.

    Terminology:
    - Baseline model: GPT-5-nano (the fixed model in all experiments)
    - Adversary model: The adversary model GPT-5-nano is paired against
    """
    utilities = exp.get('final_utilities', {})
    strategic = exp.get('strategic_behaviors', {})

    # Determine which agent is the baseline (GPT-5-nano) based on model_order
    model_order = exp.get('model_order', 'weak_first')
    if model_order == 'weak_first':
        baseline_agent = 'Agent_Alpha'
        adversary_agent = 'Agent_Beta'
    else:
        baseline_agent = 'Agent_Beta'
        adversary_agent = 'Agent_Alpha'

    baseline_utility = utilities.get(baseline_agent, 0)
    adversary_utility = utilities.get(adversary_agent, 0)

    adversary_model = exp.get('adversary_model', 'unknown')
    adversary_info = MODEL_INFO.get(adversary_model, {'tier': 'Unknown', 'elo': 0})
    adversary_elo = adversary_info.get('elo', 0)

    # Calculate Elo difference (adversary - baseline)
    elo_diff = adversary_elo - BASELINE_ELO

    return {
        'adversary_model': adversary_model,
        'adversary_tier': adversary_info.get('tier', 'Unknown'),
        'adversary_elo': adversary_elo,
        'adversary_source': adversary_info.get('source', 'Unknown'),
        'adversary_reasoning': adversary_info.get('reasoning', False),
        'elo_diff': elo_diff,  # Adversary Elo - Baseline Elo
        'model_order': model_order,
        'competition_level': exp.get('competition_level', 0),
        'consensus_reached': exp.get('consensus_reached', False),
        'final_round': exp.get('final_round', 10),
        'baseline_utility': baseline_utility,
        'adversary_utility': adversary_utility,
        'total_utility': baseline_utility + adversary_utility,
        'utility_share': baseline_utility / (baseline_utility + adversary_utility) if (baseline_utility + adversary_utility) > 0 else 0.5,
        'payoff_diff': adversary_utility - baseline_utility,  # Adversary payoff - Baseline payoff
        'manipulation_attempts': strategic.get('manipulation_attempts', 0),
        'anger_expressions': strategic.get('anger_expressions', 0),
        'gaslighting_attempts': strategic.get('gaslighting_attempts', 0),
        'cooperation_signals': strategic.get('cooperation_signals', 0),
        'scaling_experiment': exp.get('scaling_experiment', ''),
        'run_number': exp.get('run_number', 0),
    }


# Create DataFrame
df = pd.DataFrame([extract_experiment_features(exp) for exp in experiments])
print(f"\nDataFrame shape: {df.shape}")
df.head()


# ## Data Overview

# In[ ]:


# Summary statistics
print("="*60)
print("GPT-5-NANO EXPERIMENT SUMMARY")
print("="*60)

print(f"\nTotal successful experiments: {len(df)}")
print(f"\nExperiments by adversary model:")
print(df['adversary_model'].value_counts().to_string())

print(f"\nExperiments by adversary tier:")
print(df['adversary_tier'].value_counts().to_string())

print(f"\nExperiments by competition level:")
print(df['competition_level'].value_counts().sort_index().to_string())

print(f"\nExperiments by model order:")
print(df['model_order'].value_counts().to_string())


# In[ ]:


# Coverage matrix: which model pairs have data at which competition levels
coverage = df.pivot_table(
    index='adversary_model',
    columns='competition_level',
    values='baseline_utility',
    aggfunc='count',
    fill_value=0
)

# Add tier info
coverage['tier'] = coverage.index.map(lambda x: MODEL_INFO.get(x, {}).get('tier', 'Unknown'))
coverage = coverage.sort_values(['tier', 'adversary_model'])

print("\nData Coverage Matrix (count of experiments):")
print(coverage)


# ## Aggregate Statistics

# In[ ]:


# Aggregate stats by adversary model
agg_by_model = df.groupby('adversary_model').agg({
    'baseline_utility': ['mean', 'std', 'count'],
    'adversary_utility': ['mean', 'std'],
    'utility_share': ['mean', 'std'],
    'consensus_reached': 'mean',
    'final_round': 'mean',
    'manipulation_attempts': 'mean',
    'gaslighting_attempts': 'mean',
    'cooperation_signals': 'mean',
}).round(2)

# Flatten column names
agg_by_model.columns = ['_'.join(col).strip() for col in agg_by_model.columns.values]

# Add tier and elo
agg_by_model['tier'] = agg_by_model.index.map(lambda x: MODEL_INFO.get(x, {}).get('tier', 'Unknown'))
agg_by_model['elo'] = agg_by_model.index.map(lambda x: MODEL_INFO.get(x, {}).get('elo', 0))

# Sort by tier and elo
tier_order = {'Strong': 0, 'Medium': 1, 'Weak': 2, 'Unknown': 3}
agg_by_model['tier_order'] = agg_by_model['tier'].map(tier_order)
agg_by_model = agg_by_model.sort_values(['tier_order', 'elo'], ascending=[True, False])
agg_by_model = agg_by_model.drop('tier_order', axis=1)

print("\nAggregate Statistics by Adversary Model:")
print(agg_by_model.to_string())


# In[ ]:


# Aggregate stats by tier
agg_by_tier = df.groupby('adversary_tier').agg({
    'baseline_utility': ['mean', 'std', 'count'],
    'adversary_utility': ['mean', 'std'],
    'utility_share': ['mean', 'std'],
    'consensus_reached': 'mean',
    'final_round': 'mean',
}).round(2)

agg_by_tier.columns = ['_'.join(col).strip() for col in agg_by_tier.columns.values]
agg_by_tier = agg_by_tier.reindex(['Strong', 'Medium', 'Weak'])

print("\nAggregate Statistics by Adversary Tier:")
print(agg_by_tier.to_string())


# In[ ]:


# Stats by competition level
agg_by_comp = df.groupby('competition_level').agg({
    'baseline_utility': ['mean', 'std', 'count'],
    'utility_share': ['mean', 'std'],
    'consensus_reached': 'mean',
    'final_round': 'mean',
    'total_utility': 'mean',
}).round(2)

agg_by_comp.columns = ['_'.join(col).strip() for col in agg_by_comp.columns.values]

print("\nAggregate Statistics by Competition Level:")
print(agg_by_comp.to_string())


# ## Visualizations

# In[ ]:


# Plot 1: Baseline model payoff by adversary model (sorted by Elo)
fig, ax = plt.subplots(figsize=(14, 8))

# Sort by elo
model_order = df.groupby('adversary_model')['adversary_elo'].first().sort_values(ascending=False).index.tolist()

# Create box plot
plot_df = df[df['adversary_model'].isin(model_order)].copy()
plot_df['adversary_model'] = pd.Categorical(plot_df['adversary_model'], categories=model_order, ordered=True)

colors = [TIER_COLORS.get(MODEL_INFO.get(m, {}).get('tier', 'Unknown'), '#95a5a6') for m in model_order]

bp = sns.boxplot(data=plot_df, x='adversary_model', y='baseline_utility', ax=ax, palette=colors)
sns.stripplot(data=plot_df, x='adversary_model', y='baseline_utility', ax=ax, 
              color='black', alpha=0.3, size=4)

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Equal Split')
ax.set_xlabel('Adversary Model (sorted by Elo, high to low)', fontsize=12)
ax.set_ylabel('Baseline Model Payoff', fontsize=12)
ax.set_title('Baseline Model Payoff Distribution by Adversary Model', fontsize=14)
ax.tick_params(axis='x', rotation=45)

# Add tier legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=TIER_COLORS[t], label=f'{t} Tier') for t in TIER_ORDER]
legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', label='Equal Split'))
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'gpt5_nano_utility_by_opponent.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 2: Utility share by adversary tier
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Box plot of utility share by tier
tier_df = df[df['adversary_tier'].isin(TIER_ORDER)].copy()
tier_df['adversary_tier'] = pd.Categorical(tier_df['adversary_tier'], categories=TIER_ORDER, ordered=True)

sns.boxplot(data=tier_df, x='adversary_tier', y='utility_share', ax=axes[0],
            palette=[TIER_COLORS[t] for t in TIER_ORDER])
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
axes[0].set_xlabel('Adversary Tier', fontsize=12)
axes[0].set_ylabel('Baseline Model Payoff Share', fontsize=12)
axes[0].set_title('Utility Share by Adversary Tier', fontsize=14)
axes[0].set_ylim(0, 1)

# Right: Bar plot of mean utility share with error bars
tier_means = tier_df.groupby('adversary_tier')['utility_share'].agg(['mean', 'std', 'count'])
tier_means = tier_means.reindex(TIER_ORDER)
tier_means['se'] = tier_means['std'] / np.sqrt(tier_means['count'])

bars = axes[1].bar(range(len(TIER_ORDER)), tier_means['mean'], 
                   yerr=tier_means['se'] * 1.96,  # 95% CI
                   color=[TIER_COLORS[t] for t in TIER_ORDER],
                   capsize=5, edgecolor='black', linewidth=1.5)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
axes[1].set_xticks(range(len(TIER_ORDER)))
axes[1].set_xticklabels(TIER_ORDER)
axes[1].set_xlabel('Adversary Tier', fontsize=12)
axes[1].set_ylabel('Mean Utility Share (95% CI)', fontsize=12)
axes[1].set_title('GPT-5-nano Mean Utility Share by Adversary Tier', fontsize=14)
axes[1].set_ylim(0, 1)

# Add count annotations
for i, (tier, row) in enumerate(tier_means.iterrows()):
    axes[1].annotate(f'n={int(row["count"])}', xy=(i, row['mean'] + row['se']*1.96 + 0.03),
                     ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'gpt5_nano_utility_share_by_tier.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 3: Utility vs Elo scatter with regression
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot colored by tier
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier]
    ax.scatter(tier_data['adversary_elo'], tier_data['baseline_utility'], 
               c=TIER_COLORS[tier], alpha=0.6, s=60, label=f'{tier} Tier', edgecolors='white')

# Add regression line
valid_df = df[df['adversary_elo'] > 0]
if len(valid_df) > 5:
    z = np.polyfit(valid_df['adversary_elo'], valid_df['baseline_utility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['adversary_elo'].min(), valid_df['adversary_elo'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.3f})')

ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Adversary Model Elo Rating', fontsize=12)
ax.set_ylabel('Baseline Model Payoff', fontsize=12)
ax.set_title('Baseline Model Payoff vs Opponent Elo Rating', fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'gpt5_nano_utility_vs_elo.png', dpi=150, bbox_inches='tight')
plt.show()

# Print correlation
corr = valid_df['adversary_elo'].corr(valid_df['baseline_utility'])
print(f"\nCorrelation between adversary Elo and Baseline model payoff: {corr:.3f}")


# In[ ]:


# Plot 4: Heatmap of utility by adversary model and competition level
fig, ax = plt.subplots(figsize=(12, 10))

# Pivot table for heatmap
heatmap_data = df.pivot_table(
    index='adversary_model',
    columns='competition_level',
    values='baseline_utility',
    aggfunc='mean'
)

# Sort by elo
elo_order = df.groupby('adversary_model')['adversary_elo'].first().sort_values(ascending=False).index
heatmap_data = heatmap_data.reindex(elo_order)

# Add tier labels to index
new_index = [f"{m} ({MODEL_INFO.get(m, {}).get('tier', '?')[0]})" for m in heatmap_data.index]
heatmap_data.index = new_index

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
            cbar_kws={'label': 'GPT-5-nano Mean Utility'}, ax=ax)
ax.set_xlabel('Competition Level', fontsize=12)
ax.set_ylabel('Adversary Model (S=Strong, M=Medium, W=Weak)', fontsize=12)
ax.set_title('Baseline Model Payoff Heatmap\n(Models sorted by Elo, high to low)', fontsize=14)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'gpt5_nano_utility_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 5: Competition level effect
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Utility by competition level
comp_levels = sorted(df['competition_level'].unique())
colors = plt.cm.coolwarm(np.linspace(0, 1, len(comp_levels)))

for i, comp in enumerate(comp_levels):
    comp_data = df[df['competition_level'] == comp]
    axes[0].scatter([comp] * len(comp_data), comp_data['baseline_utility'], 
                    c=[colors[i]], alpha=0.5, s=50)

# Add mean line
comp_means = df.groupby('competition_level')['baseline_utility'].mean()
axes[0].plot(comp_means.index, comp_means.values, 'ko-', linewidth=2, markersize=10, label='Mean')
axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Competition Level', fontsize=12)
axes[0].set_ylabel('Baseline Model Payoff', fontsize=12)
axes[0].set_title('Utility vs Competition Level', fontsize=14)
axes[0].legend()

# Right: Total utility (efficiency) by competition level
total_means = df.groupby('competition_level')['total_utility'].mean()
total_std = df.groupby('competition_level')['total_utility'].std()

axes[1].bar(total_means.index, total_means.values, width=0.15, 
            yerr=total_std.values, capsize=5, color='steelblue', edgecolor='black')
axes[1].axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Max possible (200)')
axes[1].set_xlabel('Competition Level', fontsize=12)
axes[1].set_ylabel('Total Utility (Both Agents)', fontsize=12)
axes[1].set_title('Negotiation Efficiency by Competition Level', fontsize=14)
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'gpt5_nano_competition_effects.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 6: Rounds to consensus
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Histogram of final rounds
consensus_df = df[df['consensus_reached'] == True]
axes[0].hist(consensus_df['final_round'], bins=range(1, 12), edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(x=consensus_df['final_round'].mean(), color='red', linestyle='--', 
                label=f'Mean: {consensus_df["final_round"].mean():.1f}')
axes[0].set_xlabel('Final Round', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Rounds to Consensus', fontsize=14)
axes[0].legend()

# Right: Rounds by adversary tier
tier_rounds = consensus_df.groupby('adversary_tier')['final_round'].agg(['mean', 'std', 'count'])
tier_rounds = tier_rounds.reindex(TIER_ORDER)
tier_rounds['se'] = tier_rounds['std'] / np.sqrt(tier_rounds['count'])

bars = axes[1].bar(range(len(TIER_ORDER)), tier_rounds['mean'],
                   yerr=tier_rounds['se'] * 1.96,
                   color=[TIER_COLORS[t] for t in TIER_ORDER],
                   capsize=5, edgecolor='black')
axes[1].set_xticks(range(len(TIER_ORDER)))
axes[1].set_xticklabels(TIER_ORDER)
axes[1].set_xlabel('Adversary Tier', fontsize=12)
axes[1].set_ylabel('Mean Rounds to Consensus', fontsize=12)
axes[1].set_title('Negotiation Length by Adversary Tier', fontsize=14)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'gpt5_nano_rounds_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 7: Strategic behavior comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

behaviors = ['manipulation_attempts', 'gaslighting_attempts', 'anger_expressions', 'cooperation_signals']
titles = ['Manipulation Attempts', 'Gaslighting Attempts', 'Anger Expressions', 'Cooperation Signals']

for ax, behavior, title in zip(axes.flat, behaviors, titles):
    # Mean by adversary model
    model_means = df.groupby('adversary_model')[behavior].mean().sort_values(ascending=False)
    colors = [TIER_COLORS.get(MODEL_INFO.get(m, {}).get('tier', 'Unknown'), '#95a5a6') for m in model_means.index]
    
    bars = ax.barh(range(len(model_means)), model_means.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(model_means)))
    ax.set_yticklabels(model_means.index)
    ax.set_xlabel(f'Mean {title}')
    ax.set_title(title)
    ax.invert_yaxis()

# Add legend
legend_elements = [Patch(facecolor=TIER_COLORS[t], label=f'{t} Tier') for t in TIER_ORDER]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))

plt.suptitle('Strategic Behaviors by Adversary Model', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'gpt5_nano_strategic_behaviors.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 8: Model order effects (weak_first vs strong_first)
if 'model_order' in df.columns and df['model_order'].nunique() > 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Utility by model order
    order_df = df[df['model_order'].isin(['weak_first', 'strong_first'])]
    
    sns.boxplot(data=order_df, x='model_order', y='baseline_utility', ax=axes[0],
                palette=['#3498db', '#e74c3c'])
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Model Order', fontsize=12)
    axes[0].set_ylabel('Baseline Model Payoff', fontsize=12)
    axes[0].set_title('Utility by Speaking Order', fontsize=14)
    
    # Right: Utility by model order and tier
    order_tier_means = order_df.groupby(['model_order', 'adversary_tier'])['baseline_utility'].mean().unstack()
    order_tier_means = order_tier_means[TIER_ORDER] if all(t in order_tier_means.columns for t in TIER_ORDER) else order_tier_means
    
    x = np.arange(len(TIER_ORDER))
    width = 0.35
    
    if 'weak_first' in order_tier_means.index:
        axes[1].bar(x - width/2, order_tier_means.loc['weak_first'], width, label='Weak First', color='#3498db')
    if 'strong_first' in order_tier_means.index:
        axes[1].bar(x + width/2, order_tier_means.loc['strong_first'], width, label='Strong First', color='#e74c3c')
    
    axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(TIER_ORDER)
    axes[1].set_xlabel('Adversary Tier', fontsize=12)
    axes[1].set_ylabel('Mean Baseline Model Payoff', fontsize=12)
    axes[1].set_title('Order Effect by Adversary Tier', fontsize=14)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gpt5_nano_order_effects.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("Insufficient model order variation for analysis")


# In[ ]:


# Plot 9: Reasoning vs Non-reasoning models
fig, ax = plt.subplots(figsize=(10, 6))

reasoning_df = df[df['adversary_reasoning'].notna()].copy()
reasoning_df['reasoning_label'] = reasoning_df['adversary_reasoning'].map({True: 'Reasoning', False: 'Non-Reasoning'})

sns.boxplot(data=reasoning_df, x='reasoning_label', y='baseline_utility', ax=ax,
            palette=['#9b59b6', '#1abc9c'])
sns.stripplot(data=reasoning_df, x='reasoning_label', y='baseline_utility', ax=ax,
              color='black', alpha=0.3, size=4)

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Adversary Model Type', fontsize=12)
ax.set_ylabel('Baseline Model Payoff', fontsize=12)
ax.set_title('GPT-5-nano Performance vs Reasoning/Non-Reasoning Models', fontsize=14)

# Add stats
for i, reason_type in enumerate(['Reasoning', 'Non-Reasoning']):
    type_data = reasoning_df[reasoning_df['reasoning_label'] == reason_type]['baseline_utility']
    ax.annotate(f'n={len(type_data)}\nmean={type_data.mean():.1f}', 
                xy=(i, type_data.max() + 5), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'gpt5_nano_vs_reasoning.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Summary Statistics Table

# In[ ]:


# Create a comprehensive summary table
summary_table = df.groupby('adversary_model').agg({
    'baseline_utility': ['mean', 'std', 'min', 'max', 'count'],
    'utility_share': ['mean', 'std'],
    'consensus_reached': ['mean'],
    'final_round': ['mean'],
}).round(2)

summary_table.columns = ['Mean Util', 'Std Util', 'Min Util', 'Max Util', 'N',
                          'Mean Share', 'Std Share', 'Consensus %', 'Avg Rounds']

# Add model info
summary_table['Tier'] = summary_table.index.map(lambda x: MODEL_INFO.get(x, {}).get('tier', 'Unknown'))
summary_table['Elo'] = summary_table.index.map(lambda x: MODEL_INFO.get(x, {}).get('elo', 0))
summary_table['Source'] = summary_table.index.map(lambda x: MODEL_INFO.get(x, {}).get('source', 'Unknown'))
summary_table['Reasoning'] = summary_table.index.map(lambda x: MODEL_INFO.get(x, {}).get('reasoning', False))

# Reorder columns
summary_table = summary_table[['Tier', 'Elo', 'Source', 'Reasoning', 'N', 
                                'Mean Util', 'Std Util', 'Min Util', 'Max Util',
                                'Mean Share', 'Consensus %', 'Avg Rounds']]

# Sort by tier and elo
tier_order = {'Strong': 0, 'Medium': 1, 'Weak': 2, 'Unknown': 3}
summary_table['tier_order'] = summary_table['Tier'].map(tier_order)
summary_table = summary_table.sort_values(['tier_order', 'Elo'], ascending=[True, False])
summary_table = summary_table.drop('tier_order', axis=1)

print("\n" + "="*100)
print("COMPREHENSIVE SUMMARY: GPT-5-nano vs All Opponents")
print("="*100)
print(summary_table.to_string())

# Save to CSV
summary_table.to_csv(FIGURES_DIR / 'gpt5_nano_summary.csv')
print(f"\nSaved to {FIGURES_DIR / 'gpt5_nano_summary.csv'}")


# In[ ]:


# Key findings summary
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

overall_mean = df['baseline_utility'].mean()
overall_share = df['utility_share'].mean()
print(f"\n1. Overall GPT-5-nano Performance:")
print(f"   - Mean utility: {overall_mean:.1f}")
print(f"   - Mean utility share: {overall_share:.1%}")
print(f"   - Total experiments: {len(df)}")

print(f"\n2. Performance by Adversary Tier:")
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier]
    if len(tier_data) > 0:
        print(f"   - vs {tier}: {tier_data['baseline_utility'].mean():.1f} utility (n={len(tier_data)})")

# Best and worst matchups
model_perf = df.groupby('adversary_model')['baseline_utility'].mean().sort_values()
print(f"\n3. Best matchups for GPT-5-nano:")
for model in model_perf.tail(3).index:
    print(f"   - vs {model}: {model_perf[model]:.1f} utility")

print(f"\n4. Hardest matchups for GPT-5-nano:")
for model in model_perf.head(3).index:
    print(f"   - vs {model}: {model_perf[model]:.1f} utility")

# Correlation with Elo
valid_df = df[df['adversary_elo'] > 0]
if len(valid_df) > 5:
    corr = valid_df['adversary_elo'].corr(valid_df['baseline_utility'])
    print(f"\n5. Correlation with adversary Elo: {corr:.3f}")
    if corr < -0.1:
        print("   -> GPT-5-nano performs worse against stronger models")
    elif corr > 0.1:
        print("   -> GPT-5-nano performs better against stronger models (unexpected!)")
    else:
        print("   -> No strong relationship between opponent strength and performance")


# In[ ]:


# Save full dataframe for further analysis
df.to_csv(FIGURES_DIR / 'gpt5_nano_full_data.csv', index=False)
print(f"Full dataset saved to {FIGURES_DIR / 'gpt5_nano_full_data.csv'} ({len(df)} rows)")


# In[ ]:


# =============================================================================
# NEW FIGURES: Payoff Analysis by Elo and Competition Level
# =============================================================================

# Get unique competition levels for subplot creation
comp_levels = sorted(df['competition_level'].dropna().unique())
n_comp_levels = len(comp_levels)

# Calculate subplot grid dimensions (aim for roughly square)
n_cols = min(4, n_comp_levels)
n_rows = (n_comp_levels + n_cols - 1) // n_cols


# In[ ]:


# FIGURE A1: Baseline Model (GPT-5-nano) Payoff vs Adversary Model Elo - OVERALL
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot colored by tier
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier]
    ax.scatter(tier_data['adversary_elo'], tier_data['baseline_utility'],
               c=TIER_COLORS[tier], alpha=0.6, s=60, label=f'{tier} Tier', edgecolors='white')

# Add regression line with confidence band
valid_df = df[df['adversary_elo'] > 0]
if len(valid_df) > 5:
    z = np.polyfit(valid_df['adversary_elo'], valid_df['baseline_utility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['adversary_elo'].min(), valid_df['adversary_elo'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    # Add correlation
    corr = valid_df['adversary_elo'].corr(valid_df['baseline_utility'])
    ax.annotate(f'r = {corr:.3f}\nn = {len(valid_df)}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Equal Split')
ax.set_xlabel('Adversary Model Elo Rating', fontsize=12)
ax.set_ylabel('GPT-5-nano (Baseline Model) Payoff', fontsize=12)
ax.set_title('Baseline Model Payoff vs Adversary Model Capability (All Competition Levels)', fontsize=14)
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_a1_weak_payoff_vs_elo_overall.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE A1-SUB: Baseline Model Payoff vs Adversary Model Elo - BY COMPETITION LEVEL
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = np.atleast_2d(axes)
axes_flat = axes.flatten()

for idx, comp_level in enumerate(comp_levels):
    ax = axes_flat[idx]
    comp_df = df[df['competition_level'] == comp_level]

    # Scatter plot colored by tier
    for tier in TIER_ORDER:
        tier_data = comp_df[comp_df['adversary_tier'] == tier]
        if len(tier_data) > 0:
            ax.scatter(tier_data['adversary_elo'], tier_data['baseline_utility'],
                       c=TIER_COLORS[tier], alpha=0.6, s=40, label=f'{tier}', edgecolors='white')

    # Add regression line if enough data
    valid_comp_df = comp_df[comp_df['adversary_elo'] > 0]
    if len(valid_comp_df) > 3:
        z = np.polyfit(valid_comp_df['adversary_elo'], valid_comp_df['baseline_utility'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_comp_df['adversary_elo'].min(), valid_comp_df['adversary_elo'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.5)

        corr = valid_comp_df['adversary_elo'].corr(valid_comp_df['baseline_utility'])
        ax.annotate(f'r={corr:.2f}\nn={len(valid_comp_df)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Adversary Model Elo', fontsize=10)
    ax.set_ylabel('Baseline Model Payoff', fontsize=10)
    ax.set_title(f'Competition = {comp_level}', fontsize=11)

    if idx == 0:
        ax.legend(loc='lower left', fontsize=8)

# Hide unused subplots
for idx in range(len(comp_levels), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Baseline Model (GPT-5-nano) Payoff vs Adversary Model Elo by Competition Level', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_a1_weak_payoff_vs_elo_by_comp.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE A2: Baseline Model (GPT-5-nano) Payoff vs Competition Level - OVERALL
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot with jitter
jitter = 0.02
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier].copy()
    tier_data['comp_jittered'] = tier_data['competition_level'] + np.random.uniform(-jitter, jitter, len(tier_data))
    ax.scatter(tier_data['comp_jittered'], tier_data['baseline_utility'],
               c=TIER_COLORS[tier], alpha=0.5, s=50, label=f'{tier} Tier', edgecolors='white')

# Add mean line with error bars
comp_stats = df.groupby('competition_level')['baseline_utility'].agg(['mean', 'std', 'count'])
comp_stats['se'] = comp_stats['std'] / np.sqrt(comp_stats['count'])

ax.errorbar(comp_stats.index, comp_stats['mean'], yerr=comp_stats['se']*1.96,
            fmt='ko-', linewidth=2, markersize=10, capsize=5, label='Mean (95% CI)')

# Add regression line
if len(df) > 5:
    z = np.polyfit(df['competition_level'], df['baseline_utility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.2f})')

    corr = df['competition_level'].corr(df['baseline_utility'])
    ax.annotate(f'r = {corr:.3f}\nn = {len(df)}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Competition Level', fontsize=12)
ax.set_ylabel('GPT-5-nano (Baseline Model) Payoff', fontsize=12)
ax.set_title('Baseline Model Payoff vs Competition Level (All Opponents)', fontsize=14)
ax.set_xlim(-0.05, 1.05)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_a2_weak_payoff_vs_comp_overall.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE A2-SUB: Baseline Model Payoff vs Competition Level - BY COMPETITION LEVEL (boxplot per level)
# Since we can't filter competition by competition, we'll do it by OPPONENT TIER instead
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, tier in enumerate(TIER_ORDER):
    ax = axes[idx]
    tier_data = df[df['adversary_tier'] == tier]

    if len(tier_data) > 0:
        # Box plot by competition level
        tier_data_sorted = tier_data.copy()
        tier_data_sorted['competition_level'] = tier_data_sorted['competition_level'].astype(str)

        comp_order = [str(c) for c in sorted(tier_data['competition_level'].unique())]

        sns.boxplot(data=tier_data_sorted, x='competition_level', y='baseline_utility',
                    ax=ax, color=TIER_COLORS[tier], order=comp_order)

        # Add mean line
        tier_comp_means = tier_data.groupby('competition_level')['baseline_utility'].mean()
        ax.plot(range(len(tier_comp_means)), tier_comp_means.values, 'ko-', linewidth=2, markersize=6)

        # Correlation annotation
        corr = tier_data['competition_level'].corr(tier_data['baseline_utility'])
        ax.annotate(f'r={corr:.2f}\nn={len(tier_data)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Competition Level', fontsize=11)
    ax.set_ylabel('Baseline Model Payoff' if idx == 0 else '', fontsize=11)
    ax.set_title(f'{tier} Tier Opponents', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Baseline Model (GPT-5-nano) Payoff vs Competition Level by Adversary Tier', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_a2_weak_payoff_vs_comp_by_tier.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE A3: Adversary Model (Non-GPT-5-nano) Payoff vs Adversary Model Elo - OVERALL
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot colored by tier
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier]
    ax.scatter(tier_data['adversary_elo'], tier_data['adversary_utility'],
               c=TIER_COLORS[tier], alpha=0.6, s=60, label=f'{tier} Tier', edgecolors='white')

# Add regression line
valid_df = df[df['adversary_elo'] > 0]
if len(valid_df) > 5:
    z = np.polyfit(valid_df['adversary_elo'], valid_df['adversary_utility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['adversary_elo'].min(), valid_df['adversary_elo'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    corr = valid_df['adversary_elo'].corr(valid_df['adversary_utility'])
    ax.annotate(f'r = {corr:.3f}\nn = {len(valid_df)}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Equal Split')
ax.set_xlabel('Adversary Model Elo Rating', fontsize=12)
ax.set_ylabel('Adversary Model Payoff', fontsize=12)
ax.set_title('Adversary Model Payoff vs Own Capability (All Competition Levels)', fontsize=14)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_a3_strong_payoff_vs_elo_overall.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE A3-SUB: Adversary Model Payoff vs Adversary Model Elo - BY COMPETITION LEVEL
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = np.atleast_2d(axes)
axes_flat = axes.flatten()

for idx, comp_level in enumerate(comp_levels):
    ax = axes_flat[idx]
    comp_df = df[df['competition_level'] == comp_level]

    # Scatter plot colored by tier
    for tier in TIER_ORDER:
        tier_data = comp_df[comp_df['adversary_tier'] == tier]
        if len(tier_data) > 0:
            ax.scatter(tier_data['adversary_elo'], tier_data['adversary_utility'],
                       c=TIER_COLORS[tier], alpha=0.6, s=40, label=f'{tier}', edgecolors='white')

    # Add regression line if enough data
    valid_comp_df = comp_df[comp_df['adversary_elo'] > 0]
    if len(valid_comp_df) > 3:
        z = np.polyfit(valid_comp_df['adversary_elo'], valid_comp_df['adversary_utility'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_comp_df['adversary_elo'].min(), valid_comp_df['adversary_elo'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.5)

        corr = valid_comp_df['adversary_elo'].corr(valid_comp_df['adversary_utility'])
        ax.annotate(f'r={corr:.2f}\nn={len(valid_comp_df)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Adversary Model Elo', fontsize=10)
    ax.set_ylabel('Adversary Model Payoff', fontsize=10)
    ax.set_title(f'Competition = {comp_level}', fontsize=11)

    if idx == 0:
        ax.legend(loc='lower right', fontsize=8)

# Hide unused subplots
for idx in range(len(comp_levels), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Adversary Model Payoff vs Own Elo by Competition Level', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_a3_strong_payoff_vs_elo_by_comp.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE A4: Adversary Model Payoff vs Competition Level - OVERALL
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot with jitter
jitter = 0.02
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier].copy()
    tier_data['comp_jittered'] = tier_data['competition_level'] + np.random.uniform(-jitter, jitter, len(tier_data))
    ax.scatter(tier_data['comp_jittered'], tier_data['adversary_utility'],
               c=TIER_COLORS[tier], alpha=0.5, s=50, label=f'{tier} Tier', edgecolors='white')

# Add mean line with error bars
comp_stats = df.groupby('competition_level')['adversary_utility'].agg(['mean', 'std', 'count'])
comp_stats['se'] = comp_stats['std'] / np.sqrt(comp_stats['count'])

ax.errorbar(comp_stats.index, comp_stats['mean'], yerr=comp_stats['se']*1.96,
            fmt='ko-', linewidth=2, markersize=10, capsize=5, label='Mean (95% CI)')

# Add regression line
if len(df) > 5:
    z = np.polyfit(df['competition_level'], df['adversary_utility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.2f})')

    corr = df['competition_level'].corr(df['adversary_utility'])
    ax.annotate(f'r = {corr:.3f}\nn = {len(df)}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Competition Level', fontsize=12)
ax.set_ylabel('Adversary Model Payoff', fontsize=12)
ax.set_title('Adversary Model Payoff vs Competition Level (All Opponents)', fontsize=14)
ax.set_xlim(-0.05, 1.05)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_a4_strong_payoff_vs_comp_overall.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE A4-SUB: Adversary Model Payoff vs Competition Level - BY OPPONENT TIER
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, tier in enumerate(TIER_ORDER):
    ax = axes[idx]
    tier_data = df[df['adversary_tier'] == tier]

    if len(tier_data) > 0:
        # Box plot by competition level
        tier_data_sorted = tier_data.copy()
        tier_data_sorted['competition_level'] = tier_data_sorted['competition_level'].astype(str)

        comp_order = [str(c) for c in sorted(tier_data['competition_level'].unique())]

        sns.boxplot(data=tier_data_sorted, x='competition_level', y='adversary_utility',
                    ax=ax, color=TIER_COLORS[tier], order=comp_order)

        # Add mean line
        tier_comp_means = tier_data.groupby('competition_level')['adversary_utility'].mean()
        ax.plot(range(len(tier_comp_means)), tier_comp_means.values, 'ko-', linewidth=2, markersize=6)

        # Correlation annotation
        corr = tier_data['competition_level'].corr(tier_data['adversary_utility'])
        ax.annotate(f'r={corr:.2f}\nn={len(tier_data)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Competition Level', fontsize=11)
    ax.set_ylabel('Adversary Model Payoff' if idx == 0 else '', fontsize=11)
    ax.set_title(f'{tier} Tier Opponents', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Adversary Model Payoff vs Competition Level by Adversary Tier', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_a4_strong_payoff_vs_comp_by_tier.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# =============================================================================
# FIGURES B1-B4: Payoff DIFFERENCE (Adversary - Baseline) Analysis
# =============================================================================

# Create payoff difference column
df['payoff_diff'] = df['adversary_utility'] - df['baseline_utility']


# In[ ]:


# FIGURE B1: Payoff Difference vs Adversary Model Elo - OVERALL
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot colored by tier
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier]
    ax.scatter(tier_data['adversary_elo'], tier_data['payoff_diff'],
               c=TIER_COLORS[tier], alpha=0.6, s=60, label=f'{tier} Tier', edgecolors='white')

# Add regression line
valid_df = df[df['adversary_elo'] > 0]
if len(valid_df) > 5:
    z = np.polyfit(valid_df['adversary_elo'], valid_df['payoff_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['adversary_elo'].min(), valid_df['adversary_elo'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    corr = valid_df['adversary_elo'].corr(valid_df['payoff_diff'])
    ax.annotate(f'r = {corr:.3f}\nn = {len(valid_df)}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Equal Payoff')
ax.set_xlabel('Adversary Model Elo Rating', fontsize=12)
ax.set_ylabel('Payoff Difference (Adversary - Baseline)', fontsize=12)
ax.set_title('Payoff Difference vs Adversary Model Capability (All Competition Levels)', fontsize=14)
ax.legend(loc='upper left')

# Add shaded regions
ax.axhspan(0, ax.get_ylim()[1], alpha=0.1, color='red', label='Adversary wins')
ax.axhspan(ax.get_ylim()[0], 0, alpha=0.1, color='green', label='Baseline wins')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_b1_payoff_diff_vs_elo_overall.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE B1-SUB: Payoff Difference vs Adversary Model Elo - BY COMPETITION LEVEL
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = np.atleast_2d(axes)
axes_flat = axes.flatten()

for idx, comp_level in enumerate(comp_levels):
    ax = axes_flat[idx]
    comp_df = df[df['competition_level'] == comp_level]

    # Scatter plot colored by tier
    for tier in TIER_ORDER:
        tier_data = comp_df[comp_df['adversary_tier'] == tier]
        if len(tier_data) > 0:
            ax.scatter(tier_data['adversary_elo'], tier_data['payoff_diff'],
                       c=TIER_COLORS[tier], alpha=0.6, s=40, label=f'{tier}', edgecolors='white')

    # Add regression line if enough data
    valid_comp_df = comp_df[comp_df['adversary_elo'] > 0]
    if len(valid_comp_df) > 3:
        z = np.polyfit(valid_comp_df['adversary_elo'], valid_comp_df['payoff_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_comp_df['adversary_elo'].min(), valid_comp_df['adversary_elo'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.5)

        corr = valid_comp_df['adversary_elo'].corr(valid_comp_df['payoff_diff'])
        ax.annotate(f'r={corr:.2f}\nn={len(valid_comp_df)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Adversary Model Elo', fontsize=10)
    ax.set_ylabel('Payoff Diff (A-B)', fontsize=10)
    ax.set_title(f'Competition = {comp_level}', fontsize=11)

    if idx == 0:
        ax.legend(loc='upper left', fontsize=8)

# Hide unused subplots
for idx in range(len(comp_levels), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Payoff Difference (Adversary - Baseline) vs Adversary Model Elo by Competition Level', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_b1_payoff_diff_vs_elo_by_comp.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE B2: Payoff Difference vs Competition Level - OVERALL
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot with jitter
jitter = 0.02
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier].copy()
    tier_data['comp_jittered'] = tier_data['competition_level'] + np.random.uniform(-jitter, jitter, len(tier_data))
    ax.scatter(tier_data['comp_jittered'], tier_data['payoff_diff'],
               c=TIER_COLORS[tier], alpha=0.5, s=50, label=f'{tier} Tier', edgecolors='white')

# Add mean line with error bars
comp_stats = df.groupby('competition_level')['payoff_diff'].agg(['mean', 'std', 'count'])
comp_stats['se'] = comp_stats['std'] / np.sqrt(comp_stats['count'])

ax.errorbar(comp_stats.index, comp_stats['mean'], yerr=comp_stats['se']*1.96,
            fmt='ko-', linewidth=2, markersize=10, capsize=5, label='Mean (95% CI)')

# Add regression line
if len(df) > 5:
    z = np.polyfit(df['competition_level'], df['payoff_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.2f})')

    corr = df['competition_level'].corr(df['payoff_diff'])
    ax.annotate(f'r = {corr:.3f}\nn = {len(df)}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=2)
ax.set_xlabel('Competition Level', fontsize=12)
ax.set_ylabel('Payoff Difference (Adversary - Baseline)', fontsize=12)
ax.set_title('Payoff Difference vs Competition Level (All Opponents)', fontsize=14)
ax.set_xlim(-0.05, 1.05)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_b2_payoff_diff_vs_comp_overall.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE B2-SUB: Payoff Difference vs Competition Level - BY OPPONENT TIER
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, tier in enumerate(TIER_ORDER):
    ax = axes[idx]
    tier_data = df[df['adversary_tier'] == tier]

    if len(tier_data) > 0:
        # Box plot by competition level
        tier_data_sorted = tier_data.copy()
        tier_data_sorted['competition_level'] = tier_data_sorted['competition_level'].astype(str)

        comp_order = [str(c) for c in sorted(tier_data['competition_level'].unique())]

        sns.boxplot(data=tier_data_sorted, x='competition_level', y='payoff_diff',
                    ax=ax, color=TIER_COLORS[tier], order=comp_order)

        # Add mean line
        tier_comp_means = tier_data.groupby('competition_level')['payoff_diff'].mean()
        ax.plot(range(len(tier_comp_means)), tier_comp_means.values, 'ko-', linewidth=2, markersize=6)

        # Correlation annotation
        corr = tier_data['competition_level'].corr(tier_data['payoff_diff'])
        ax.annotate(f'r={corr:.2f}\nn={len(tier_data)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Competition Level', fontsize=11)
    ax.set_ylabel('Payoff Diff (A-B)' if idx == 0 else '', fontsize=11)
    ax.set_title(f'{tier} Tier Opponents', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Payoff Difference (Adversary - Baseline) vs Competition Level by Adversary Tier', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_b2_payoff_diff_vs_comp_by_tier.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE B3: Payoff Difference vs Adversary Model Elo - FACETED BY COMPETITION LEVEL (alternative view)
# This shows the same data as B1-SUB but with a different visualization approach
fig, ax = plt.subplots(figsize=(14, 8))

# Use color gradient for competition level
cmap = plt.cm.coolwarm
norm = plt.Normalize(0, 1)

scatter = ax.scatter(df['adversary_elo'], df['payoff_diff'],
                     c=df['competition_level'], cmap=cmap, norm=norm,
                     alpha=0.6, s=60, edgecolors='white')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Competition Level', fontsize=12)

# Add regression lines for low, medium, high competition
for comp_range, color, label in [((0, 0.3), 'blue', 'Low (0-0.3)'),
                                   ((0.4, 0.6), 'gray', 'Mid (0.4-0.6)'),
                                   ((0.7, 1.0), 'red', 'High (0.7-1.0)')]:
    subset = df[(df['competition_level'] >= comp_range[0]) & (df['competition_level'] <= comp_range[1])]
    if len(subset) > 5:
        z = np.polyfit(subset['adversary_elo'], subset['payoff_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset['adversary_elo'].min(), subset['adversary_elo'].max(), 100)
        ax.plot(x_line, p(x_line), '--', color=color, alpha=0.8, linewidth=2, label=f'{label}: slope={z[0]:.4f}')

ax.axhline(y=0, color='black', linestyle=':', alpha=0.7, linewidth=2)
ax.set_xlabel('Adversary Model Elo Rating', fontsize=12)
ax.set_ylabel('Payoff Difference (Adversary - Baseline)', fontsize=12)
ax.set_title('Payoff Difference vs Elo, Colored by Competition Level', fontsize=14)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_b3_payoff_diff_vs_elo_colored_by_comp.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE B4: Summary - Mean Payoff Difference by Elo Bin and Competition Level (Heatmap)
fig, ax = plt.subplots(figsize=(14, 10))

# Create Elo bins
df['elo_bin'] = pd.cut(df['adversary_elo'], bins=[1100, 1200, 1300, 1400, 1500],
                       labels=['1100-1200', '1200-1300', '1300-1400', '1400-1500'])

# Pivot table for heatmap - mean values
heatmap_data = df.pivot_table(
    index='elo_bin',
    columns='competition_level',
    values='payoff_diff',
    aggfunc='mean'
)

# Pivot table for counts
heatmap_counts = df.pivot_table(
    index='elo_bin',
    columns='competition_level',
    values='payoff_diff',
    aggfunc='count'
)

# Sort by elo bin (reversed so higher Elo is at top)
heatmap_data = heatmap_data.iloc[::-1]
heatmap_counts = heatmap_counts.iloc[::-1]

# Create custom annotations with mean and n
annot_labels = heatmap_data.copy().astype(str)
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        mean_val = heatmap_data.iloc[i, j]
        count_val = heatmap_counts.iloc[i, j]
        if pd.notna(mean_val) and pd.notna(count_val):
            annot_labels.iloc[i, j] = f'{mean_val:.1f}\n(n={int(count_val)})'
        else:
            annot_labels.iloc[i, j] = ''

sns.heatmap(heatmap_data, annot=annot_labels, fmt='', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Mean Payoff Difference (Adversary - Baseline)'}, ax=ax,
            annot_kws={'fontsize': 9})
ax.set_xlabel('Competition Level', fontsize=12)
ax.set_ylabel('Adversary Model Elo Range', fontsize=12)
ax.set_title('Payoff Difference Heatmap\n(Positive = Adversary wins more, Negative = Baseline wins more)', fontsize=14)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_b4_payoff_diff_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Clean up temporary column
df = df.drop('elo_bin', axis=1)


# In[ ]:


# =============================================================================
# FIGURES C & D: Using Elo DIFFERENCE (Adversary Elo - Baseline Elo) on x-axis
# =============================================================================

# Note: elo_diff was already computed in extract_experiment_features()
# elo_diff = adversary_elo - BASELINE_ELO (where BASELINE_ELO = 1338 for GPT-5-nano)


# In[ ]:


# FIGURE C1: Baseline Model Payoff vs Elo Difference - OVERALL
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot colored by tier
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier]
    ax.scatter(tier_data['elo_diff'], tier_data['baseline_utility'],
               c=TIER_COLORS[tier], alpha=0.6, s=60, label=f'{tier} Tier', edgecolors='white')

# Add regression line
valid_df = df[df['elo_diff'].notna()]
if len(valid_df) > 5:
    z = np.polyfit(valid_df['elo_diff'], valid_df['baseline_utility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['elo_diff'].min(), valid_df['elo_diff'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    corr = valid_df['elo_diff'].corr(valid_df['baseline_utility'])
    ax.annotate(f'r = {corr:.3f}\nn = {len(valid_df)}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Equal Split')
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(f'Elo Difference (Adversary - Baseline)\n[Baseline Elo = {BASELINE_ELO}]', fontsize=12)
ax.set_ylabel('Baseline Model Payoff', fontsize=12)
ax.set_title('Baseline Model Payoff vs Elo Difference (All Competition Levels)', fontsize=14)
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_c1_baseline_payoff_vs_elo_diff_overall.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE C1-SUB: Baseline Model Payoff vs Elo Difference - BY COMPETITION LEVEL
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = np.atleast_2d(axes)
axes_flat = axes.flatten()

for idx, comp_level in enumerate(comp_levels):
    ax = axes_flat[idx]
    comp_df = df[df['competition_level'] == comp_level]

    # Scatter plot colored by tier
    for tier in TIER_ORDER:
        tier_data = comp_df[comp_df['adversary_tier'] == tier]
        if len(tier_data) > 0:
            ax.scatter(tier_data['elo_diff'], tier_data['baseline_utility'],
                       c=TIER_COLORS[tier], alpha=0.6, s=40, label=f'{tier}', edgecolors='white')

    # Add regression line if enough data
    valid_comp_df = comp_df[comp_df['elo_diff'].notna()]
    if len(valid_comp_df) > 3:
        z = np.polyfit(valid_comp_df['elo_diff'], valid_comp_df['baseline_utility'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_comp_df['elo_diff'].min(), valid_comp_df['elo_diff'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.5)

        corr = valid_comp_df['elo_diff'].corr(valid_comp_df['baseline_utility'])
        ax.annotate(f'r={corr:.2f}\nn={len(valid_comp_df)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Elo Diff (A-B)', fontsize=10)
    ax.set_ylabel('Baseline Payoff', fontsize=10)
    ax.set_title(f'Competition = {comp_level}', fontsize=11)

    if idx == 0:
        ax.legend(loc='lower left', fontsize=8)

# Hide unused subplots
for idx in range(len(comp_levels), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle(f'Baseline Model Payoff vs Elo Difference by Competition Level\n[Baseline Elo = {BASELINE_ELO}]', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_c1_baseline_payoff_vs_elo_diff_by_comp.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE C2: Adversary Model Payoff vs Elo Difference - OVERALL
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot colored by tier
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier]
    ax.scatter(tier_data['elo_diff'], tier_data['adversary_utility'],
               c=TIER_COLORS[tier], alpha=0.6, s=60, label=f'{tier} Tier', edgecolors='white')

# Add regression line
valid_df = df[df['elo_diff'].notna()]
if len(valid_df) > 5:
    z = np.polyfit(valid_df['elo_diff'], valid_df['adversary_utility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['elo_diff'].min(), valid_df['elo_diff'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    corr = valid_df['elo_diff'].corr(valid_df['adversary_utility'])
    ax.annotate(f'r = {corr:.3f}\nn = {len(valid_df)}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Equal Split')
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(f'Elo Difference (Adversary - Baseline)\n[Baseline Elo = {BASELINE_ELO}]', fontsize=12)
ax.set_ylabel('Adversary Model Payoff', fontsize=12)
ax.set_title('Adversary Model Payoff vs Elo Difference (All Competition Levels)', fontsize=14)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_c2_adversary_payoff_vs_elo_diff_overall.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE C2-SUB: Adversary Model Payoff vs Elo Difference - BY COMPETITION LEVEL
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = np.atleast_2d(axes)
axes_flat = axes.flatten()

for idx, comp_level in enumerate(comp_levels):
    ax = axes_flat[idx]
    comp_df = df[df['competition_level'] == comp_level]

    # Scatter plot colored by tier
    for tier in TIER_ORDER:
        tier_data = comp_df[comp_df['adversary_tier'] == tier]
        if len(tier_data) > 0:
            ax.scatter(tier_data['elo_diff'], tier_data['adversary_utility'],
                       c=TIER_COLORS[tier], alpha=0.6, s=40, label=f'{tier}', edgecolors='white')

    # Add regression line if enough data
    valid_comp_df = comp_df[comp_df['elo_diff'].notna()]
    if len(valid_comp_df) > 3:
        z = np.polyfit(valid_comp_df['elo_diff'], valid_comp_df['adversary_utility'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_comp_df['elo_diff'].min(), valid_comp_df['elo_diff'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.5)

        corr = valid_comp_df['elo_diff'].corr(valid_comp_df['adversary_utility'])
        ax.annotate(f'r={corr:.2f}\nn={len(valid_comp_df)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Elo Diff (A-B)', fontsize=10)
    ax.set_ylabel('Adversary Payoff', fontsize=10)
    ax.set_title(f'Competition = {comp_level}', fontsize=11)

    if idx == 0:
        ax.legend(loc='lower right', fontsize=8)

# Hide unused subplots
for idx in range(len(comp_levels), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle(f'Adversary Model Payoff vs Elo Difference by Competition Level\n[Baseline Elo = {BASELINE_ELO}]', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_c2_adversary_payoff_vs_elo_diff_by_comp.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE D1: Payoff Difference vs Elo Difference - OVERALL
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot colored by tier
for tier in TIER_ORDER:
    tier_data = df[df['adversary_tier'] == tier]
    ax.scatter(tier_data['elo_diff'], tier_data['payoff_diff'],
               c=TIER_COLORS[tier], alpha=0.6, s=60, label=f'{tier} Tier', edgecolors='white')

# Add regression line
valid_df = df[df['elo_diff'].notna()]
if len(valid_df) > 5:
    z = np.polyfit(valid_df['elo_diff'], valid_df['payoff_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['elo_diff'].min(), valid_df['elo_diff'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    corr = valid_df['elo_diff'].corr(valid_df['payoff_diff'])
    ax.annotate(f'r = {corr:.3f}\nn = {len(valid_df)}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Equal Payoff')
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(f'Elo Difference (Adversary - Baseline)\n[Baseline Elo = {BASELINE_ELO}]', fontsize=12)
ax.set_ylabel('Payoff Difference (Adversary - Baseline)', fontsize=12)
ax.set_title('Payoff Difference vs Elo Difference (All Competition Levels)', fontsize=14)
ax.legend(loc='upper left')

# Add shaded regions
ylim = ax.get_ylim()
ax.axhspan(0, ylim[1], alpha=0.1, color='red')
ax.axhspan(ylim[0], 0, alpha=0.1, color='green')
ax.set_ylim(ylim)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_d1_payoff_diff_vs_elo_diff_overall.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE D1-SUB: Payoff Difference vs Elo Difference - BY COMPETITION LEVEL
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = np.atleast_2d(axes)
axes_flat = axes.flatten()

for idx, comp_level in enumerate(comp_levels):
    ax = axes_flat[idx]
    comp_df = df[df['competition_level'] == comp_level]

    # Scatter plot colored by tier
    for tier in TIER_ORDER:
        tier_data = comp_df[comp_df['adversary_tier'] == tier]
        if len(tier_data) > 0:
            ax.scatter(tier_data['elo_diff'], tier_data['payoff_diff'],
                       c=TIER_COLORS[tier], alpha=0.6, s=40, label=f'{tier}', edgecolors='white')

    # Add regression line if enough data
    valid_comp_df = comp_df[comp_df['elo_diff'].notna()]
    if len(valid_comp_df) > 3:
        z = np.polyfit(valid_comp_df['elo_diff'], valid_comp_df['payoff_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_comp_df['elo_diff'].min(), valid_comp_df['elo_diff'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.5)

        corr = valid_comp_df['elo_diff'].corr(valid_comp_df['payoff_diff'])
        ax.annotate(f'r={corr:.2f}\nn={len(valid_comp_df)}',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Elo Diff (A-B)', fontsize=10)
    ax.set_ylabel('Payoff Diff (A-B)', fontsize=10)
    ax.set_title(f'Competition = {comp_level}', fontsize=11)

    if idx == 0:
        ax.legend(loc='upper left', fontsize=8)

# Hide unused subplots
for idx in range(len(comp_levels), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle(f'Payoff Difference vs Elo Difference by Competition Level\n[Baseline Elo = {BASELINE_ELO}]', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_d1_payoff_diff_vs_elo_diff_by_comp.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# FIGURE D2: Payoff Difference vs Elo Difference - Colored by Competition Level
fig, ax = plt.subplots(figsize=(14, 8))

# Use color gradient for competition level
cmap = plt.cm.coolwarm
norm = plt.Normalize(0, 1)

scatter = ax.scatter(df['elo_diff'], df['payoff_diff'],
                     c=df['competition_level'], cmap=cmap, norm=norm,
                     alpha=0.6, s=60, edgecolors='white')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Competition Level', fontsize=12)

# Add regression lines for low, medium, high competition
for comp_range, color, label in [((0, 0.3), 'blue', 'Low (0-0.3)'),
                                   ((0.4, 0.6), 'gray', 'Mid (0.4-0.6)'),
                                   ((0.7, 1.0), 'red', 'High (0.7-1.0)')]:
    subset = df[(df['competition_level'] >= comp_range[0]) & (df['competition_level'] <= comp_range[1])]
    if len(subset) > 5:
        z = np.polyfit(subset['elo_diff'], subset['payoff_diff'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset['elo_diff'].min(), subset['elo_diff'].max(), 100)
        ax.plot(x_line, p(x_line), '--', color=color, alpha=0.8, linewidth=2, label=f'{label}: slope={z[0]:.4f}')

ax.axhline(y=0, color='black', linestyle=':', alpha=0.7, linewidth=2)
ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
ax.set_xlabel(f'Elo Difference (Adversary - Baseline)\n[Baseline Elo = {BASELINE_ELO}]', fontsize=12)
ax.set_ylabel('Payoff Difference (Adversary - Baseline)', fontsize=12)
ax.set_title('Payoff Difference vs Elo Difference, Colored by Competition Level', fontsize=14)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_d2_payoff_diff_vs_elo_diff_colored_by_comp.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Summary of new figures
print("\n" + "="*60)
print("NEW FIGURES GENERATED")
print("="*60)
print("\nFigure A1: Baseline Model (GPT-5-nano) Payoff vs Adversary Model Elo")
print("  - fig_a1_weak_payoff_vs_elo_overall.png (overall)")
print("  - fig_a1_weak_payoff_vs_elo_by_comp.png (subplots by competition level)")

print("\nFigure A2: Baseline Model (GPT-5-nano) Payoff vs Competition Level")
print("  - fig_a2_weak_payoff_vs_comp_overall.png (overall)")
print("  - fig_a2_weak_payoff_vs_comp_by_tier.png (subplots by adversary tier)")

print("\nFigure A3: Adversary Model Payoff vs Adversary Model Elo")
print("  - fig_a3_strong_payoff_vs_elo_overall.png (overall)")
print("  - fig_a3_strong_payoff_vs_elo_by_comp.png (subplots by competition level)")

print("\nFigure A4: Adversary Model Payoff vs Competition Level")
print("  - fig_a4_strong_payoff_vs_comp_overall.png (overall)")
print("  - fig_a4_strong_payoff_vs_comp_by_tier.png (subplots by adversary tier)")

print("\nFigure B1: Payoff Difference (Adversary - Baseline) vs Adversary Model Elo")
print("  - fig_b1_payoff_diff_vs_elo_overall.png (overall)")
print("  - fig_b1_payoff_diff_vs_elo_by_comp.png (subplots by competition level)")

print("\nFigure B2: Payoff Difference (Adversary - Baseline) vs Competition Level")
print("  - fig_b2_payoff_diff_vs_comp_overall.png (overall)")
print("  - fig_b2_payoff_diff_vs_comp_by_tier.png (subplots by adversary tier)")

print("\nFigure B3: Payoff Difference vs Elo (colored by competition)")
print("  - fig_b3_payoff_diff_vs_elo_colored_by_comp.png")

print("\nFigure B4: Payoff Difference Heatmap (Elo bins x Competition)")
print("  - fig_b4_payoff_diff_heatmap.png")

print("\nFigure C1: Baseline Model Payoff vs Elo Difference")
print("  - fig_c1_baseline_payoff_vs_elo_diff_overall.png (overall)")
print("  - fig_c1_baseline_payoff_vs_elo_diff_by_comp.png (subplots by competition level)")

print("\nFigure C2: Adversary Model Payoff vs Elo Difference")
print("  - fig_c2_adversary_payoff_vs_elo_diff_overall.png (overall)")
print("  - fig_c2_adversary_payoff_vs_elo_diff_by_comp.png (subplots by competition level)")

print("\nFigure D1: Payoff Difference vs Elo Difference")
print("  - fig_d1_payoff_diff_vs_elo_diff_overall.png (overall)")
print("  - fig_d1_payoff_diff_vs_elo_diff_by_comp.png (subplots by competition level)")

print("\nFigure D2: Payoff Difference vs Elo Difference (colored by competition)")
print("  - fig_d2_payoff_diff_vs_elo_diff_colored_by_comp.png")

