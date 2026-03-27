# Co-Funding Game Parameter Calibration

## C_MIN and C_MAX: Principled Calibration

### What they control

Project costs `c_j` are drawn i.i.d. from `Uniform(c_min, c_max)` at game
initialization. They set the **absolute scale** of the cost/budget system.
The relative tension between agents is governed by `sigma`, but `c_min`/`c_max`
determine whether the numbers are in a cognitively legible range and whether
the burden ratio is well-calibrated.

### The key invariant: burden ratio

Define the **burden ratio** as the fair-share cost per agent relative to that
agent's average per-project valuation:

```
burden_ratio = E[c_j] / n  ÷  (100 / m)
             = m × (c_min + c_max) / (2 × n × 100)
```

where:
- `n` = number of agents
- `m` = number of projects (`m_projects`)
- `100` = valuation scale (each agent's valuation vector sums to 100, fixed)

**Target: burden_ratio ≈ 0.5.**

At 0.5, funding a project at equal split costs exactly half your valuation for
it — you net the other half. Projects are worth doing jointly but expensive to
fund unilaterally. This creates genuine strategic tension without making the
game trivially easy (burden → 0) or trivially hopeless (burden → 1+).

### Current defaults: n=2, m=5

With `c_min=10`, `c_max=30`:

| Quantity | Value |
|---|---|
| Average project cost `E[c_j]` | 20 |
| Average per-project valuation `100/m` | 20 |
| Fair-share cost per agent `E[c_j]/n` | 10 |
| Net utility per project at equal split | 10 |
| Per-agent budget at σ=0.3 | 32.5 |
| Per-agent budget at σ=0.5 | 37.5 |
| Per-agent budget at σ=1.0 | 50.0 |
| Burden ratio | **0.50** |

The clean identity `E[c_j] = 100/m` (average project cost = average
per-project valuation) is not a coincidence — it's what gives burden_ratio = 0.5
at n=2.

Each agent can afford roughly 1.6–2.5 full projects alone (depending on σ),
so there is real choice about which projects to champion and which to free-ride on.

### Rescaling for different n

The burden ratio breaks if you add agents without adjusting costs. To maintain
burden_ratio = 0.5:

```
E[c_j]  =  n × (100/m) × 0.5  =  50n/m
```

Suggested ranges (keeping a similar width, c_max ≈ 3 × c_min):

| n agents | m=5, target E[c_j] | Suggested [c_min, c_max] |
|---|---|---|
| 2 | 10 | [10, 30] ← current default |
| 3 | 15 | [10, 20] |
| 4 | 20 | [15, 25] |
| 5 | 25 | [20, 30] |

For a different `m`, the formula is the same: `E[c_j] = 50n/m`.

### Range width: c_max / c_min

The current 3× ratio (`c_max/c_min = 30/10 = 3`) creates meaningful
project-cost heterogeneity:

- Cheap projects (~10) can be funded by one agent alone
- Expensive projects (~30) require genuine cooperation
- This produces richer strategy than a uniform cost

A tighter range (e.g., [15, 25]) would reduce variance and make project
outcomes more uniform. A wider range (e.g., [5, 45]) would increase it.
The current default is a reasonable middle ground.

### Budget formula reminder

```python
budget_ratio    = 0.5 + 0.5 * sigma      # in [0.5, 1.0]
total_budget    = budget_ratio * total_cost
per_agent_budget = total_budget / n
```

So `sigma=1` means agents collectively have exactly enough to fund everything;
`sigma=0` (limit) means they have only half the total cost collectively.
At `sigma=0.3` (hard condition in current configs): `budget_ratio = 0.65`,
meaning 65% of total cost is available — below what's needed to fund all projects.
