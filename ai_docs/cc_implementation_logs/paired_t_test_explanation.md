# Paired t-test Analysis in Negotiation Results

## Overview
The code uses `scipy.stats.ttest_rel()` to perform a **paired samples t-test** (also called dependent samples t-test) to compare the utilities achieved by two different models in the same negotiation scenarios.

## Exact Implementation

### Code Location
File: `visualize_negotiation_results.py`, lines 224-225
```python
if len(alpha_utils) == len(beta_utils):
    t_stat, p_value = stats.ttest_rel(alpha_utils, beta_utils)
```

### Function Used
`scipy.stats.ttest_rel(a, b, axis=0, nan_policy='propagate', alternative='two-sided')`

## Parameters and Data Structure

### Input Data
- **alpha_utils**: Array of final utilities for Model A (e.g., gpt-4o) across 5 runs
- **beta_utils**: Array of final utilities for Model B (e.g., grok-3) across the same 5 runs

Example data structure:
```
Run 1: gpt-4o gets 51.0, grok-3 gets 65.0
Run 2: gpt-4o gets 43.0, grok-3 gets 62.0
Run 3: gpt-4o gets 60.0, grok-3 gets 83.0
Run 4: gpt-4o gets 61.0, grok-3 gets 82.0
Run 5: gpt-4o gets 44.0, grok-3 gets 67.0
```

### Key Assumption: Paired Data
The test assumes the data is **paired** - meaning:
- Run 1 of Model A corresponds to Run 1 of Model B
- They negotiated against each other in the same game instance
- They faced the same random seed, items, and preference distributions

## What the Test is Testing

### Null Hypothesis (H₀)
**μ_d = 0**: The mean difference between paired utilities is zero
- In other words: There is no systematic difference in performance between the two models
- Any observed differences are due to random chance

### Alternative Hypothesis (H₁)
**μ_d ≠ 0**: The mean difference between paired utilities is not zero
- One model systematically achieves different utilities than the other
- The test is **two-tailed** (default in scipy)

### Mathematical Definition
For each pair i:
- d_i = utility_alpha[i] - utility_beta[i]
- The test examines whether the mean of all d_i values is significantly different from 0

### Test Statistic
```
t = (mean(d) - 0) / (std(d) / sqrt(n))
```
Where:
- mean(d) = average difference across all pairs
- std(d) = standard deviation of differences
- n = number of pairs (typically 5 in our experiments)

## Parametric Assumptions

### 1. **Normality of Differences**
- **Assumption**: The differences (d_i = utility_alpha[i] - utility_beta[i]) should follow a normal distribution
- **Why it matters**: The t-distribution assumes normality for valid p-values
- **In our case**: With only n=5 pairs, this is difficult to verify and likely violated
- **Robustness**: The t-test is somewhat robust to non-normality with larger samples, but n=5 is quite small

### 2. **Continuous Scale**
- **Assumption**: The measured variable (utility) should be continuous
- **In our case**: ✓ Utilities are continuous values (0-100)

### 3. **Independence of Pairs**
- **Assumption**: Each pair (negotiation run) should be independent of others
- **In our case**: ✓ Different random seeds and independent runs

### 4. **No Significant Outliers**
- **Assumption**: The differences shouldn't contain extreme outliers
- **Why it matters**: Outliers can heavily influence the mean and standard deviation
- **In our case**: Should check with only 5 data points

## How the P-value is Calculated

### Step 1: Calculate the Test Statistic
```python
# For paired data (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)
differences = [x[i] - y[i] for i in range(n)]
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)  # sample standard deviation
n = len(differences)

t_statistic = mean_diff / (std_diff / np.sqrt(n))
```

### Step 2: Determine Degrees of Freedom
```python
df = n - 1  # For our case with 5 pairs: df = 4
```

### Step 3: Calculate P-value from t-distribution
The p-value is the probability of observing a test statistic as extreme or more extreme than the calculated value, assuming the null hypothesis is true.

```python
from scipy import stats

# For two-tailed test (default in scipy.stats.ttest_rel)
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

# Or using scipy's survival function
p_value = 2 * stats.t.sf(abs(t_statistic), df)
```

### Mathematical Formula
The p-value represents:
```
P(|T| ≥ |t_observed| | H₀ is true)
```
Where T follows a t-distribution with (n-1) degrees of freedom.

## Example Calculation

Let's work through a real example from the data:

```python
# GPT-4o vs Grok-3 at competition 0.5
alpha_utils = [51.0, 43.0, 60.0, 61.0, 44.0]
beta_utils = [65.0, 62.0, 83.0, 82.0, 67.0]

# Calculate differences
differences = [51-65, 43-62, 60-83, 61-82, 44-67]
            = [-14, -19, -23, -21, -23]

# Statistics
mean_diff = -20.0
std_diff = 3.74 (sample std)
n = 5

# t-statistic
t = -20.0 / (3.74 / sqrt(5))
t = -20.0 / 1.67
t = -11.95

# Degrees of freedom
df = 4

# P-value (two-tailed)
# Looking up |t|=11.95 with df=4 in t-distribution
p_value ≈ 0.0003
```

## Interpretation

### P-value Thresholds
- **p < 0.05**: Reject null hypothesis (significant difference at 95% confidence)
- **p ≥ 0.05**: Fail to reject null hypothesis (no significant difference)

### T-statistic Sign
- **t > 0**: Model A (alpha) tends to get higher utilities than Model B (beta)
- **t < 0**: Model B (beta) tends to get higher utilities than Model A (alpha)
- **|t| magnitude**: Larger absolute values indicate stronger evidence against H₀

## Why Paired t-test is Appropriate Here

### Advantages of Paired Test
1. **Controls for game-specific variation**: Each negotiation setup might be easier/harder
2. **Reduces variance**: By looking at differences within pairs, we eliminate between-game variance
3. **More powerful**: Can detect smaller differences than unpaired tests
4. **Natural pairing**: The models literally negotiated against each other

### Alternative (Incorrect) Approach
An **unpaired t-test** (`ttest_ind`) would:
- Treat the utilities as independent samples
- Ignore that models negotiated against each other
- Have less statistical power
- Be inappropriate for this data structure

## Example Interpretation

From actual results:
```
Competition 0.50:
  Paired t-test: t=-11.952, p=0.0003
  → grok-3 significantly outperforms gpt-4o
```

This means:
1. **t = -11.952**: Large negative value indicates grok-3 consistently scored higher
2. **p = 0.0003**: Extremely strong evidence against the null hypothesis
3. **Conclusion**: At 0.5 competition level, grok-3 systematically achieves higher utilities than gpt-4o
4. **Confidence**: We can be 99.97% confident this isn't due to random chance

## Violations and Alternatives

### When Assumptions are Violated

With n=5, the normality assumption is particularly concerning. Here are alternatives:

### 1. **Wilcoxon Signed-Rank Test** (Non-parametric alternative)
```python
from scipy import stats
statistic, p_value = stats.wilcoxon(alpha_utils, beta_utils)
```
- Doesn't assume normality
- Uses ranks instead of raw values
- More appropriate for small samples

### 2. **Bootstrap Methods**
```python
# Resample with replacement to estimate distribution
bootstrap_diffs = []
for _ in range(10000):
    sample_indices = np.random.choice(5, 5, replace=True)
    sample_diff = np.mean([differences[i] for i in sample_indices])
    bootstrap_diffs.append(sample_diff)

# Calculate confidence interval
ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)
```

### 3. **Permutation Test**
- Randomly shuffle the pairing
- Calculate test statistic for each permutation
- P-value = proportion of permutations with |t| ≥ observed |t|

## Checking Assumptions in Practice

```python
# Check normality of differences
from scipy import stats

differences = alpha_utils - beta_utils

# Shapiro-Wilk test for normality (though unreliable with n=5)
statistic, p_value = stats.shapiro(differences)
if p_value < 0.05:
    print("Differences may not be normally distributed")

# Visual check (though limited with n=5)
import matplotlib.pyplot as plt
stats.probplot(differences, dist="norm", plot=plt)
plt.show()

# Check for outliers using IQR method
Q1 = np.percentile(differences, 25)
Q3 = np.percentile(differences, 75)
IQR = Q3 - Q1
outliers = [x for x in differences if x < Q1-1.5*IQR or x > Q3+1.5*IQR]
```

## Cohen's d Effect Size

The code also calculates Cohen's d:
```python
mean_diff = alpha_utils.mean() - beta_utils.mean()
pooled_std = np.sqrt((alpha_utils.std()**2 + beta_utils.std()**2) / 2)
cohens_d = mean_diff / pooled_std
```

This measures the **standardized mean difference**:
- Small effect: |d| < 0.5
- Medium effect: 0.5 ≤ |d| < 0.8
- Large effect: |d| ≥ 0.8

Effect size complements p-values by indicating practical significance, not just statistical significance.

## Limitations and Recommendations

### Current Limitations
1. **Small Sample Size**: With only 5 runs, the test has limited power
2. **Normality Assumption**: Difficult to verify with n=5
3. **Outlier Sensitivity**: Single outliers can heavily influence results

### Recommendations
1. **Increase Sample Size**: Consider running 10-30 negotiations per condition for more reliable results
2. **Use Non-parametric Tests**: Given n=5, Wilcoxon signed-rank test may be more appropriate
3. **Report Multiple Metrics**: Include both parametric and non-parametric results
4. **Check Assumptions**: Always verify test assumptions before interpreting results
5. **Focus on Effect Sizes**: With small samples, effect sizes may be more informative than p-values

## Summary

The paired t-test in this code is testing whether one model systematically achieves different utilities than another when they negotiate against each other under identical conditions. It's the appropriate test because:
1. The data is naturally paired (same negotiation instance)
2. We want to control for negotiation-specific factors
3. We're interested in within-pair differences, not absolute utilities

The test answers: **"Does Model A consistently get better/worse outcomes than Model B when they negotiate against each other?"**

However, with only 5 samples, the parametric assumptions (especially normality) are difficult to verify, and non-parametric alternatives should be considered for more robust conclusions.