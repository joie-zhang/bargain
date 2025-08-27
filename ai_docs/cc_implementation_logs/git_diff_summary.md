# Git Diff Summary: Preference Vector Normalization

## Overview
This session focused on fixing the preference vectors to ensure equal maximum attainable utilities for both agents while maintaining precise cosine similarities for different competition levels.

## Key Changes

### 1. **vector_pairs.py** - Core preference vector updates
**Problem Fixed:** 
- Previous vectors had unequal maximum utilities (e.g., one agent could earn max 10 points while another could earn max 40 points)
- Cosine similarity errors were too large for intermediate competition levels

**Solution Implemented:**
- All vectors now sum to exactly 30.0 (equal maximum utilities)
- All cosine similarity errors are now < 0.01 from target values
- Used scipy optimization to find precise vector pairs

**Specific Changes:**
```python
# OLD vectors (unequal max utilities, large errors):
# Pair 1: v1_a sum=10, v1_b sum=40
v1_a_nn = np.array([10, 0, 0, 0, 0])       # sum = 10
v1_b_nn = np.array([0, 10, 10, 10, 10])    # sum = 40

# NEW vectors (equal max utilities of 30, precise cosine similarities):
# Pair 1: Cosine similarity = 0.00 (exact)
v1_a_nn = np.array([15.0, 15.0, 0.0, 0.0, 0.0])  # sum = 30
v1_b_nn = np.array([0.0, 0.0, 10.0, 10.0, 10.0]) # sum = 30
```

**Results Achieved:**
| Pair | Target Cosine | Actual Cosine | Error | Sum A | Sum B |
|------|---------------|---------------|-------|-------|-------|
| 1    | 0.00          | 0.0000        | 0.0000| 30.00 | 30.00 |
| 2    | 0.25          | 0.2495        | 0.0005| 30.00 | 30.00 |
| 3    | 0.50          | 0.4940        | 0.0060| 30.00 | 30.00 |
| 4    | 0.75          | 0.7494        | 0.0006| 30.00 | 30.00 |
| 5    | 1.00          | 1.0000        | 0.0000| 30.00 | 30.00 |

### 2. **New Helper Scripts Created**
- `vector_pairs_equal_max.py` - Initial attempt at creating equal max utility vectors
- `create_better_equal_vectors.py` - Improved version with scipy optimization
- `create_precise_equal_vectors.py` - Final version achieving < 0.01 error tolerance

### 3. **Dependencies Added**
- Added `scipy` to the project for optimization capabilities
- Installed via: `uv pip install scipy`

## Technical Approach

1. **Optimization Method:**
   - Used `scipy.optimize.minimize` with SLSQP method
   - Objective function: minimize cosine similarity error
   - Constraints: Both vectors must sum to 30.0
   - Bounds: All values non-negative

2. **Multiple Starting Points:**
   - Tried up to 10 different initial guesses per target
   - Selected best result with lowest error

3. **Rounding Strategy:**
   - Rounded to 1 decimal place where possible
   - Maintained exact sum constraint

## Impact on Experiments

These changes ensure:
1. **Fair comparison between agents** - Both agents have equal maximum possible utilities
2. **Precise control over competition levels** - Cosine similarities accurately represent intended competition (0.00, 0.25, 0.50, 0.75, 1.00)
3. **Reproducible results** - Hard-coded vectors eliminate randomness in preference generation

## Files Modified Summary
```
vector_pairs.py                        | 47 lines changed
create_win_rate_heatmap_a1.py          | 161 lines changed  
run_strong_models_experiment.py        | 12 lines changed
scripts/run_single_experiment.sh       | 5 lines changed
strong_models_experiment/experiment.py | 12 lines changed
win_rate_heatmap.png                   | Binary file updated
```

## Verification
All changes have been verified to:
- ✅ Maintain exact sum of 30.0 for all vectors
- ✅ Achieve cosine similarity errors < 0.01 
- ✅ Use only non-negative values
- ✅ Maintain numerical stability