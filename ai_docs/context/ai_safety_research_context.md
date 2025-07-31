# AI Safety & Alignment Research Context

## Research Domain
**Focus Area**: AI Safety & Alignment research with emphasis on:
- Neural network interpretability and mechanistic understanding
- Training dynamics and optimization safety
- Alignment techniques and evaluation methods
- Safety properties of large language models
- Robustness and failure mode analysis

## Technical Stack
- **Primary Framework**: PyTorch for deep learning experiments
- **Development Environment**: Jupyter notebooks for exploration and analysis
- **Compute Resources**: 
  - Princeton Della cluster (high-memory nodes, A100 GPUs)
  - Princeton PLI cluster (specialized AI/ML hardware)
  - SLURM job scheduling for distributed training
- **Languages**: Python (primary), with shell scripting for cluster jobs

## Common Research Patterns
1. **Exploratory Analysis**: Jupyter notebooks for initial investigation
2. **Experiment Implementation**: PyTorch models with proper logging
3. **Cluster Training**: SLURM jobs for large-scale experiments
4. **Results Analysis**: Post-processing and visualization
5. **Paper Writing**: LaTeX with experimental results integration

## Verification Protocols for AI Safety Research
- **Reproducibility**: All experiments must include random seeds and environment specs
- **Safety Checks**: Never expose model weights or training data in logs
- **Baseline Comparisons**: Always compare against established safety baselines
- **Statistical Significance**: Use proper statistical tests for claims
- **Ablation Studies**: Systematic component analysis for safety claims

## Cluster Computing Guidelines
- Use job arrays for hyperparameter sweeps
- Monitor GPU memory usage to prevent OOM failures
- Implement checkpointing for long-running training jobs
- Use shared storage efficiently (avoid excessive small file I/O)
- Follow Princeton HPC best practices for job submission

## Common Failure Modes to Avoid
- **Training Instability**: Gradient explosion, NaN losses, mode collapse
- **Data Leakage**: Improper train/val/test splits in safety evaluations
- **Overfitting to Safety Metrics**: Gaming evaluation without real safety improvement
- **Resource Waste**: Inefficient cluster usage, unnecessary large batch sizes
- **Reproducibility Issues**: Missing seeds, environment dependencies, version mismatches