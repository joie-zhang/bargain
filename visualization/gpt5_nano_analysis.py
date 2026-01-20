#!/usr/bin/env python
# coding: utf-8

# # GPT-5-nano Negotiation Analysis
# 
# This notebook analyzes negotiation experiments where GPT-5-nano is paired against various models across different tiers.
# 
# ## Model Tiers
# - **STRONG TIER** (Elo >= 1415): gemini-3-pro, gpt-5.2-high, claude-opus-4-5, kimi-k2-thinking, deepseek-r1-0528, qwen3-235b-a22b-instruct-2507
# - **MEDIUM TIER** (1290 <= Elo < 1415): claude-4-5-haiku, o4-mini-2025-04-16, gpt-oss-20b
# - **WEAK TIER** (Elo < 1290): llama-3.3-70b-instruct, llama-3.1-8b-instruct

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


# Model tier definitions with Elo ratings
MODEL_INFO = {
    # Strong tier (Elo >= 1415)
    "gemini-3-pro": {"tier": "Strong", "elo": 1492, "source": "Closed", "reasoning": True},
    "gpt-5.2-high": {"tier": "Strong", "elo": 1465, "source": "Closed", "reasoning": True},
    "claude-opus-4-5": {"tier": "Strong", "elo": 1462, "source": "Closed", "reasoning": False},
    "kimi-k2-thinking": {"tier": "Strong", "elo": 1438, "source": "Open", "reasoning": True},
    "deepseek-r1-0528": {"tier": "Strong", "elo": 1426, "source": "Open", "reasoning": True},
    "qwen3-235b-a22b-instruct-2507": {"tier": "Strong", "elo": 1418, "source": "Open", "reasoning": False},
    
    # Medium tier (1290 <= Elo < 1415)
    "claude-4-5-haiku": {"tier": "Medium", "elo": 1378, "source": "Closed", "reasoning": False},
    "o4-mini-2025-04-16": {"tier": "Medium", "elo": 1362, "source": "Closed", "reasoning": True},
    "gpt-oss-20b": {"tier": "Medium", "elo": 1315, "source": "Open", "reasoning": False},
    
    # Weak tier (Elo < 1290)
    "llama-3.3-70b-instruct": {"tier": "Weak", "elo": 1276, "source": "Open", "reasoning": False},
    "llama-3.1-8b-instruct": {"tier": "Weak", "elo": 1193, "source": "Open", "reasoning": False},
}

TIER_ORDER = ["Strong", "Medium", "Weak"]
TIER_COLORS = {"Strong": "#e74c3c", "Medium": "#f39c12", "Weak": "#27ae60"}


# In[ ]:


# Define experiment directories to search
RESULTS_DIR = Path("/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results")

SCALING_EXPERIMENTS = [
    "scaling_experiment_20260117_185910",  # Most recent
    "scaling_experiment_20260117_172734",
    "scaling_experiment_20260116_052234",
    "scaling_experiment_old_20260116_052119",
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
                    'opponent_model': strong_model,  # Since weak_model is always gpt-5-nano
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
    """Extract key features from experiment data for DataFrame."""
    utilities = exp.get('final_utilities', {})
    strategic = exp.get('strategic_behaviors', {})
    
    # Determine which agent is GPT-5-nano based on model_order
    model_order = exp.get('model_order', 'weak_first')
    if model_order == 'weak_first':
        nano_agent = 'Agent_Alpha'
        opponent_agent = 'Agent_Beta'
    else:
        nano_agent = 'Agent_Beta'
        opponent_agent = 'Agent_Alpha'
    
    nano_utility = utilities.get(nano_agent, 0)
    opponent_utility = utilities.get(opponent_agent, 0)
    
    opponent_model = exp.get('opponent_model', 'unknown')
    opponent_info = MODEL_INFO.get(opponent_model, {'tier': 'Unknown', 'elo': 0})
    
    return {
        'opponent_model': opponent_model,
        'opponent_tier': opponent_info.get('tier', 'Unknown'),
        'opponent_elo': opponent_info.get('elo', 0),
        'opponent_source': opponent_info.get('source', 'Unknown'),
        'opponent_reasoning': opponent_info.get('reasoning', False),
        'model_order': model_order,
        'competition_level': exp.get('competition_level', 0),
        'consensus_reached': exp.get('consensus_reached', False),
        'final_round': exp.get('final_round', 10),
        'nano_utility': nano_utility,
        'opponent_utility': opponent_utility,
        'total_utility': nano_utility + opponent_utility,
        'utility_share': nano_utility / (nano_utility + opponent_utility) if (nano_utility + opponent_utility) > 0 else 0.5,
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
print(f"\nExperiments by opponent model:")
print(df['opponent_model'].value_counts().to_string())

print(f"\nExperiments by opponent tier:")
print(df['opponent_tier'].value_counts().to_string())

print(f"\nExperiments by competition level:")
print(df['competition_level'].value_counts().sort_index().to_string())

print(f"\nExperiments by model order:")
print(df['model_order'].value_counts().to_string())


# In[ ]:


# Coverage matrix: which model pairs have data at which competition levels
coverage = df.pivot_table(
    index='opponent_model',
    columns='competition_level',
    values='nano_utility',
    aggfunc='count',
    fill_value=0
)

# Add tier info
coverage['tier'] = coverage.index.map(lambda x: MODEL_INFO.get(x, {}).get('tier', 'Unknown'))
coverage = coverage.sort_values(['tier', 'opponent_model'])

print("\nData Coverage Matrix (count of experiments):")
print(coverage)


# ## Aggregate Statistics

# In[ ]:


# Aggregate stats by opponent model
agg_by_model = df.groupby('opponent_model').agg({
    'nano_utility': ['mean', 'std', 'count'],
    'opponent_utility': ['mean', 'std'],
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

print("\nAggregate Statistics by Opponent Model:")
print(agg_by_model.to_string())


# In[ ]:


# Aggregate stats by tier
agg_by_tier = df.groupby('opponent_tier').agg({
    'nano_utility': ['mean', 'std', 'count'],
    'opponent_utility': ['mean', 'std'],
    'utility_share': ['mean', 'std'],
    'consensus_reached': 'mean',
    'final_round': 'mean',
}).round(2)

agg_by_tier.columns = ['_'.join(col).strip() for col in agg_by_tier.columns.values]
agg_by_tier = agg_by_tier.reindex(['Strong', 'Medium', 'Weak'])

print("\nAggregate Statistics by Opponent Tier:")
print(agg_by_tier.to_string())


# In[ ]:


# Stats by competition level
agg_by_comp = df.groupby('competition_level').agg({
    'nano_utility': ['mean', 'std', 'count'],
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


# Plot 1: GPT-5-nano utility by opponent model (sorted by Elo)
fig, ax = plt.subplots(figsize=(14, 8))

# Sort by elo
model_order = df.groupby('opponent_model')['opponent_elo'].first().sort_values(ascending=False).index.tolist()

# Create box plot
plot_df = df[df['opponent_model'].isin(model_order)].copy()
plot_df['opponent_model'] = pd.Categorical(plot_df['opponent_model'], categories=model_order, ordered=True)

colors = [TIER_COLORS.get(MODEL_INFO.get(m, {}).get('tier', 'Unknown'), '#95a5a6') for m in model_order]

bp = sns.boxplot(data=plot_df, x='opponent_model', y='nano_utility', ax=ax, palette=colors)
sns.stripplot(data=plot_df, x='opponent_model', y='nano_utility', ax=ax, 
              color='black', alpha=0.3, size=4)

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Equal Split')
ax.set_xlabel('Opponent Model (sorted by Elo, high to low)', fontsize=12)
ax.set_ylabel('GPT-5-nano Utility', fontsize=12)
ax.set_title('GPT-5-nano Utility Distribution by Opponent Model', fontsize=14)
ax.tick_params(axis='x', rotation=45)

# Add tier legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=TIER_COLORS[t], label=f'{t} Tier') for t in TIER_ORDER]
legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', label='Equal Split'))
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('figures/gpt5_nano_utility_by_opponent.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 2: Utility share by opponent tier
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Box plot of utility share by tier
tier_df = df[df['opponent_tier'].isin(TIER_ORDER)].copy()
tier_df['opponent_tier'] = pd.Categorical(tier_df['opponent_tier'], categories=TIER_ORDER, ordered=True)

sns.boxplot(data=tier_df, x='opponent_tier', y='utility_share', ax=axes[0],
            palette=[TIER_COLORS[t] for t in TIER_ORDER])
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
axes[0].set_xlabel('Opponent Tier', fontsize=12)
axes[0].set_ylabel('GPT-5-nano Utility Share', fontsize=12)
axes[0].set_title('Utility Share by Opponent Tier', fontsize=14)
axes[0].set_ylim(0, 1)

# Right: Bar plot of mean utility share with error bars
tier_means = tier_df.groupby('opponent_tier')['utility_share'].agg(['mean', 'std', 'count'])
tier_means = tier_means.reindex(TIER_ORDER)
tier_means['se'] = tier_means['std'] / np.sqrt(tier_means['count'])

bars = axes[1].bar(range(len(TIER_ORDER)), tier_means['mean'], 
                   yerr=tier_means['se'] * 1.96,  # 95% CI
                   color=[TIER_COLORS[t] for t in TIER_ORDER],
                   capsize=5, edgecolor='black', linewidth=1.5)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
axes[1].set_xticks(range(len(TIER_ORDER)))
axes[1].set_xticklabels(TIER_ORDER)
axes[1].set_xlabel('Opponent Tier', fontsize=12)
axes[1].set_ylabel('Mean Utility Share (95% CI)', fontsize=12)
axes[1].set_title('GPT-5-nano Mean Utility Share by Opponent Tier', fontsize=14)
axes[1].set_ylim(0, 1)

# Add count annotations
for i, (tier, row) in enumerate(tier_means.iterrows()):
    axes[1].annotate(f'n={int(row["count"])}', xy=(i, row['mean'] + row['se']*1.96 + 0.03),
                     ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/gpt5_nano_utility_share_by_tier.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 3: Utility vs Elo scatter with regression
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot colored by tier
for tier in TIER_ORDER:
    tier_data = df[df['opponent_tier'] == tier]
    ax.scatter(tier_data['opponent_elo'], tier_data['nano_utility'], 
               c=TIER_COLORS[tier], alpha=0.6, s=60, label=f'{tier} Tier', edgecolors='white')

# Add regression line
valid_df = df[df['opponent_elo'] > 0]
if len(valid_df) > 5:
    z = np.polyfit(valid_df['opponent_elo'], valid_df['nano_utility'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['opponent_elo'].min(), valid_df['opponent_elo'].max(), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.3f})')

ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Opponent Model Elo Rating', fontsize=12)
ax.set_ylabel('GPT-5-nano Utility', fontsize=12)
ax.set_title('GPT-5-nano Utility vs Opponent Elo Rating', fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig('figures/gpt5_nano_utility_vs_elo.png', dpi=150, bbox_inches='tight')
plt.show()

# Print correlation
corr = valid_df['opponent_elo'].corr(valid_df['nano_utility'])
print(f"\nCorrelation between opponent Elo and GPT-5-nano utility: {corr:.3f}")


# In[ ]:


# Plot 4: Heatmap of utility by opponent model and competition level
fig, ax = plt.subplots(figsize=(12, 10))

# Pivot table for heatmap
heatmap_data = df.pivot_table(
    index='opponent_model',
    columns='competition_level',
    values='nano_utility',
    aggfunc='mean'
)

# Sort by elo
elo_order = df.groupby('opponent_model')['opponent_elo'].first().sort_values(ascending=False).index
heatmap_data = heatmap_data.reindex(elo_order)

# Add tier labels to index
new_index = [f"{m} ({MODEL_INFO.get(m, {}).get('tier', '?')[0]})" for m in heatmap_data.index]
heatmap_data.index = new_index

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
            cbar_kws={'label': 'GPT-5-nano Mean Utility'}, ax=ax)
ax.set_xlabel('Competition Level', fontsize=12)
ax.set_ylabel('Opponent Model (S=Strong, M=Medium, W=Weak)', fontsize=12)
ax.set_title('GPT-5-nano Utility Heatmap\n(Models sorted by Elo, high to low)', fontsize=14)

plt.tight_layout()
plt.savefig('figures/gpt5_nano_utility_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 5: Competition level effect
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Utility by competition level
comp_levels = sorted(df['competition_level'].unique())
colors = plt.cm.coolwarm(np.linspace(0, 1, len(comp_levels)))

for i, comp in enumerate(comp_levels):
    comp_data = df[df['competition_level'] == comp]
    axes[0].scatter([comp] * len(comp_data), comp_data['nano_utility'], 
                    c=[colors[i]], alpha=0.5, s=50)

# Add mean line
comp_means = df.groupby('competition_level')['nano_utility'].mean()
axes[0].plot(comp_means.index, comp_means.values, 'ko-', linewidth=2, markersize=10, label='Mean')
axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Competition Level', fontsize=12)
axes[0].set_ylabel('GPT-5-nano Utility', fontsize=12)
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
plt.savefig('figures/gpt5_nano_competition_effects.png', dpi=150, bbox_inches='tight')
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

# Right: Rounds by opponent tier
tier_rounds = consensus_df.groupby('opponent_tier')['final_round'].agg(['mean', 'std', 'count'])
tier_rounds = tier_rounds.reindex(TIER_ORDER)
tier_rounds['se'] = tier_rounds['std'] / np.sqrt(tier_rounds['count'])

bars = axes[1].bar(range(len(TIER_ORDER)), tier_rounds['mean'],
                   yerr=tier_rounds['se'] * 1.96,
                   color=[TIER_COLORS[t] for t in TIER_ORDER],
                   capsize=5, edgecolor='black')
axes[1].set_xticks(range(len(TIER_ORDER)))
axes[1].set_xticklabels(TIER_ORDER)
axes[1].set_xlabel('Opponent Tier', fontsize=12)
axes[1].set_ylabel('Mean Rounds to Consensus', fontsize=12)
axes[1].set_title('Negotiation Length by Opponent Tier', fontsize=14)

plt.tight_layout()
plt.savefig('figures/gpt5_nano_rounds_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 7: Strategic behavior comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

behaviors = ['manipulation_attempts', 'gaslighting_attempts', 'anger_expressions', 'cooperation_signals']
titles = ['Manipulation Attempts', 'Gaslighting Attempts', 'Anger Expressions', 'Cooperation Signals']

for ax, behavior, title in zip(axes.flat, behaviors, titles):
    # Mean by opponent model
    model_means = df.groupby('opponent_model')[behavior].mean().sort_values(ascending=False)
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

plt.suptitle('Strategic Behaviors by Opponent Model', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/gpt5_nano_strategic_behaviors.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


# Plot 8: Model order effects (weak_first vs strong_first)
if 'model_order' in df.columns and df['model_order'].nunique() > 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Utility by model order
    order_df = df[df['model_order'].isin(['weak_first', 'strong_first'])]
    
    sns.boxplot(data=order_df, x='model_order', y='nano_utility', ax=axes[0],
                palette=['#3498db', '#e74c3c'])
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Model Order', fontsize=12)
    axes[0].set_ylabel('GPT-5-nano Utility', fontsize=12)
    axes[0].set_title('Utility by Speaking Order', fontsize=14)
    
    # Right: Utility by model order and tier
    order_tier_means = order_df.groupby(['model_order', 'opponent_tier'])['nano_utility'].mean().unstack()
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
    axes[1].set_xlabel('Opponent Tier', fontsize=12)
    axes[1].set_ylabel('Mean GPT-5-nano Utility', fontsize=12)
    axes[1].set_title('Order Effect by Opponent Tier', fontsize=14)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('figures/gpt5_nano_order_effects.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("Insufficient model order variation for analysis")


# In[ ]:


# Plot 9: Reasoning vs Non-reasoning models
fig, ax = plt.subplots(figsize=(10, 6))

reasoning_df = df[df['opponent_reasoning'].notna()].copy()
reasoning_df['reasoning_label'] = reasoning_df['opponent_reasoning'].map({True: 'Reasoning', False: 'Non-Reasoning'})

sns.boxplot(data=reasoning_df, x='reasoning_label', y='nano_utility', ax=ax,
            palette=['#9b59b6', '#1abc9c'])
sns.stripplot(data=reasoning_df, x='reasoning_label', y='nano_utility', ax=ax,
              color='black', alpha=0.3, size=4)

ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Opponent Model Type', fontsize=12)
ax.set_ylabel('GPT-5-nano Utility', fontsize=12)
ax.set_title('GPT-5-nano Performance vs Reasoning/Non-Reasoning Models', fontsize=14)

# Add stats
for i, reason_type in enumerate(['Reasoning', 'Non-Reasoning']):
    type_data = reasoning_df[reasoning_df['reasoning_label'] == reason_type]['nano_utility']
    ax.annotate(f'n={len(type_data)}\nmean={type_data.mean():.1f}', 
                xy=(i, type_data.max() + 5), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/gpt5_nano_vs_reasoning.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Summary Statistics Table

# In[ ]:


# Create a comprehensive summary table
summary_table = df.groupby('opponent_model').agg({
    'nano_utility': ['mean', 'std', 'min', 'max', 'count'],
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
summary_table.to_csv('figures/gpt5_nano_summary.csv')
print("\nSaved to figures/gpt5_nano_summary.csv")


# In[ ]:


# Key findings summary
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

overall_mean = df['nano_utility'].mean()
overall_share = df['utility_share'].mean()
print(f"\n1. Overall GPT-5-nano Performance:")
print(f"   - Mean utility: {overall_mean:.1f}")
print(f"   - Mean utility share: {overall_share:.1%}")
print(f"   - Total experiments: {len(df)}")

print(f"\n2. Performance by Opponent Tier:")
for tier in TIER_ORDER:
    tier_data = df[df['opponent_tier'] == tier]
    if len(tier_data) > 0:
        print(f"   - vs {tier}: {tier_data['nano_utility'].mean():.1f} utility (n={len(tier_data)})")

# Best and worst matchups
model_perf = df.groupby('opponent_model')['nano_utility'].mean().sort_values()
print(f"\n3. Best matchups for GPT-5-nano:")
for model in model_perf.tail(3).index:
    print(f"   - vs {model}: {model_perf[model]:.1f} utility")

print(f"\n4. Hardest matchups for GPT-5-nano:")
for model in model_perf.head(3).index:
    print(f"   - vs {model}: {model_perf[model]:.1f} utility")

# Correlation with Elo
valid_df = df[df['opponent_elo'] > 0]
if len(valid_df) > 5:
    corr = valid_df['opponent_elo'].corr(valid_df['nano_utility'])
    print(f"\n5. Correlation with opponent Elo: {corr:.3f}")
    if corr < -0.1:
        print("   -> GPT-5-nano performs worse against stronger models")
    elif corr > 0.1:
        print("   -> GPT-5-nano performs better against stronger models (unexpected!)")
    else:
        print("   -> No strong relationship between opponent strength and performance")


# In[ ]:


# Save full dataframe for further analysis
df.to_csv('figures/gpt5_nano_full_data.csv', index=False)
print(f"Full dataset saved to figures/gpt5_nano_full_data.csv ({len(df)} rows)")

