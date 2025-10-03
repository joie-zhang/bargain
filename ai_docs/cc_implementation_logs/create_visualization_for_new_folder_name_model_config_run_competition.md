# Negotiation Results Analysis Report

## Executive Summary

Analysis of negotiation experiments between GPT-4o and three Grok models (Grok-3, Grok-3-mini, and Grok-4-0709) across different competition levels (0.0 to 1.0). Each configuration was run 5 times to establish confidence intervals.

## Key Findings

### 1. Model Performance by Competition Level

#### GPT-4o vs Grok-3
- **Low Competition (0.0)**: Perfect cooperation achieved (100/100 split)
- **Medium Competition (0.25-0.75)**: Grok-3 consistently outperforms GPT-4o
  - At 0.5 competition: Grok-3 gets 71.8±12.3 vs GPT-4o's 51.8±10.6 (p<0.001)
  - At 0.75 competition: Grok-3 gets 65.0±17.6 vs GPT-4o's 45.8±15.1 (p=0.005)
- **High Competition (1.0)**: Results flip - GPT-4o gets 61.2±28.9 vs Grok-3's 38.8±28.9

#### GPT-4o vs Grok-3-mini
- **Low Competition (0.0)**: Perfect cooperation (100/100)
- **Medium Competition (0.25-0.75)**: No significant differences between models
  - Performance remains relatively balanced
  - Total utility declines gradually from 144.8 to 118.0
- **High Competition (1.0)**: Even split (58.7±46.0 vs 41.3±46.0)

#### GPT-4o vs Grok-4-0709
- **Low Competition (0.0)**: Near-perfect cooperation (92.0±22.2 vs 100.0±0.0)
- **Medium-High Competition (0.25-0.75)**: Grok-4-0709 significantly dominates
  - At 0.25: Grok-4-0709 gets 89.6±6.6 vs GPT-4o's 71.0±11.4 (p=0.007)
  - At 0.5: Grok-4-0709 gets 83.2±8.8 vs GPT-4o's 62.6±5.6 (p<0.001)
  - At 0.75: Grok-4-0709 gets 66.2±8.9 vs GPT-4o's 56.4±4.5 (p=0.044)
- **High Competition (1.0)**: Returns to balance (53.5±39.5 vs 46.5±39.5)

### 2. Efficiency Analysis (Total Utility)

Competition level strongly affects negotiation efficiency:
- **0.0 (Cooperative)**: Near-maximum efficiency (192-200 total utility)
- **0.25**: Moderate efficiency loss (131-161 total utility)
- **0.5**: Significant efficiency loss (124-146 total utility)
- **0.75**: Severe efficiency loss (111-123 total utility)
- **1.0 (Fully Competitive)**: Maximum efficiency loss (100 total utility)

### 3. Statistical Significance

**Significant Findings (p < 0.05):**
- Grok-3 outperforms GPT-4o at competition levels 0.5 and 0.75
- Grok-4-0709 outperforms GPT-4o at competition levels 0.25, 0.5, and 0.75
- Grok-3-mini shows no significant advantage over GPT-4o at any level

**Effect Sizes (Cohen's d):**
- Largest effects seen in GPT-4o vs Grok-4-0709 matchups (d > 3.0 at 0.5 competition)
- Medium to large effects in GPT-4o vs Grok-3 matchups
- Small effects in GPT-4o vs Grok-3-mini matchups

### 4. Patterns and Insights

1. **Model Strength Correlation**: Stronger Grok models (4-0709 > 3 > 3-mini) show greater ability to exploit GPT-4o in semi-competitive settings

2. **Competition Sweet Spot**: The 0.5-0.75 competition range reveals the largest performance differences between models

3. **Cooperation vs Competition**: All model pairs achieve perfect or near-perfect cooperation at 0.0 competition, suggesting good cooperative capabilities

4. **High Competition Paradox**: At maximum competition (1.0), results become more balanced or even reverse, possibly due to:
   - Deadlock situations
   - Random strategies becoming optimal
   - Failure to reach consensus (some NaN values)

## Recommendations

1. **For Scaling Law Research**: Focus on 0.25-0.75 competition range where model strength differences are most apparent

2. **For Safety Research**: Investigate why stronger models (Grok-4-0709) consistently exploit weaker ones in semi-competitive settings

3. **For Future Experiments**:
   - Increase sample size for high competition scenarios (many NaN values)
   - Add intermediate model strengths to better characterize scaling laws
   - Analyze conversation transcripts to understand exploitation mechanisms

## Data Quality Notes

- All competition levels 0.0-0.75 have complete data (5 runs each)
- Competition level 1.0 has some missing data (NaN values) indicating failed negotiations
- Confidence intervals calculated at 95% level using t-distribution
- Total of 75 experiments analyzed across 3 model pairings and 5 competition levels