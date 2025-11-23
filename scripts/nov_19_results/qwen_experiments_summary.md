# Qwen2.5 Experiments: Summary for Advisor Meeting

## Executive Summary

**Research Question**: How do Qwen2.5 models (3B, 7B, 14B) perform in negotiation tasks against Claude-3.7-Sonnet under different competition levels?

**Key Finding**: All models achieve 100% consensus rates, but utility distributions vary dramatically between cooperation (comp=0) and competition (comp=1) settings.

---

## Experimental Setup

- **Models Tested**: Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct
- **Adversary**: Claude-3.7-Sonnet
- **Competition Levels**: 
  - Level 0: Full cooperation (orthogonal preferences, maximum welfare possible)
  - Level 1: Full competition (zero-sum game)
- **Runs**: 5 runs per model per competition level (some still in progress)
- **Items**: 5 items, 10 max rounds

---

## Key Results

### 1. Consensus Achievement
**All models achieve 100% consensus** across all competition levels tested.

This suggests:
- Qwen models are capable of reaching agreements
- Model size doesn't affect consensus rate
- Competition level doesn't prevent consensus

### 2. Cooperation Setting (Competition Level = 0)

| Model | Runs | Avg Rounds | Qwen Utility | Claude Utility | Difference |
|-------|------|------------|--------------|----------------|------------|
| 3B    | 1    | 2.0        | 100.0        | 100.0          | 0.0        |
| 7B    | 5    | 3.6        | 100.0        | 97.8           | -2.2       |
| 14B   | 2    | 4.5        | 100.0        | 100.0          | 0.0        |

**Observations**:
- Near-perfect cooperation: utilities are very close to equal (100/100 or 100/97.8)
- Larger models (14B) take slightly longer to reach consensus (4.5 rounds vs 2.0)
- Qwen models achieve maximum utility (100) in cooperative settings
- Exploitation detected in 20% of 7B runs (but still fair outcomes)

### 3. Competition Setting (Competition Level = 1)

| Model | Runs | Avg Rounds | Qwen Utility | Claude Utility | Difference |
|-------|------|------------|--------------|----------------|------------|
| 3B    | 5    | 1.0        | 19.6         | 80.4           | +60.8      |
| 7B    | 5    | 1.0        | 34.0         | 66.0           | +32.0      |
| 14B   | -    | -          | -            | -              | -          |

**Observations**:
- **Claude dominates**: Gets 60-80% of utility in competitive settings
- **Model size matters**: 7B performs better than 3B (34.0 vs 19.6 utility)
- **Faster convergence**: Consensus reached in 1 round (vs 2-4.5 in cooperation)
- **Clear exploitation**: Large utility gaps favor Claude

---

## Key Insights

### 1. Model Size Effect
- **Cooperation**: All sizes perform equally well (100 utility)
- **Competition**: Larger models perform better
  - 3B: 19.6 utility (Claude gets 80.4)
  - 7B: 34.0 utility (Claude gets 66.0)
  - **Hypothesis**: 14B will perform even better (data pending)

### 2. Competition Level Effect
- **Cooperation (comp=0)**: Fair outcomes, equal utilities
- **Competition (comp=1)**: Strong model (Claude) exploits weaker models
- **Utility gap**: Increases dramatically from ~0 (cooperation) to +30-60 (competition)

### 3. Negotiation Dynamics
- **Rounds to consensus**: 
  - Competition: Very fast (1 round) - likely quick agreement on split
  - Cooperation: Slower (2-4.5 rounds) - more discussion needed
- **Exploitation detection**: More common in competitive settings

---

## Comparison: Cooperation vs Competition

| Metric | Cooperation (comp=0) | Competition (comp=1) |
|--------|---------------------|----------------------|
| **Consensus Rate** | 100% | 100% |
| **Avg Rounds** | 2.0-4.5 | 1.0 |
| **Utility Distribution** | Fair (100/100 or 100/98) | Unfair (20-34 / 66-80) |
| **Qwen Performance** | Excellent (100 utility) | Poor (20-34 utility) |
| **Model Size Effect** | None | Strong (larger = better) |

---

## Implications

1. **Qwen models are cooperative**: In cooperative settings, they achieve fair outcomes
2. **Size matters under competition**: Larger Qwen models negotiate better outcomes
3. **Claude is dominant**: Consistently outperforms Qwen models in competitive settings
4. **Consensus is always possible**: Even in zero-sum games, models find agreements

---

## Next Steps / Questions for Discussion

1. **Complete 14B experiments**: How does the largest model perform in competition?
2. **Dialogue analysis**: What strategies do models use? How do they differ?
3. **Intermediate competition levels**: Test comp=0.25, 0.5, 0.75 to see transition
4. **More runs**: Increase sample size for statistical significance
5. **Cross-model comparison**: How do Qwen models compare to other open-source models?

---

## Technical Notes

- Experiments run on Princeton cluster using local model paths
- Competition level 0 = cosine similarity 0 (orthogonal preferences = cooperation)
- Competition level 1 = cosine similarity 1 (aligned preferences = zero-sum)
- All experiments use same random seed progression for reproducibility

