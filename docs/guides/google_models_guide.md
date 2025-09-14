# Google Models Guide: Gemini & Gemma

This guide covers the comprehensive Google model support in the negotiation framework, including both the Gemini API models and open-source Gemma models.

## 🧠 Gemini Models Overview

Google's Gemini models are available through the Google AI API and offer advanced reasoning, vision, and function-calling capabilities.

### Gemini 2.5 Series (Latest)

The newest generation with enhanced capabilities and competitive pricing.

| Model | Display Name | Context | Vision | Functions | Cost (Input/Output per 1M) |
|-------|-------------|---------|--------|-----------|----------------------------|
| `gemini-2.5-pro` | Gemini 2.5 Pro | 2M tokens | ✅ | ✅ | $1.25 / $2.50 |
| `gemini-2.5-flash` | Gemini 2.5 Flash | 1M tokens | ✅ | ✅ | $0.075 / $0.30 |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | 1M tokens | ✅ | ❌ | $0.037 / $0.15 |

**Recommended for**: High-quality reasoning tasks, complex negotiations requiring advanced planning.

### Gemini 2.0 Series

Balanced performance and capabilities.

| Model | Display Name | Context | Vision | Functions | Cost (Input/Output per 1M) |
|-------|-------------|---------|--------|-----------|----------------------------|
| `gemini-2.0-flash` | Gemini 2.0 Flash | 1M tokens | ✅ | ✅ | $0.075 / $0.30 |
| `gemini-2.0-flash-lite` | Gemini 2.0 Flash Lite | 1M tokens | ✅ | ❌ | $0.037 / $0.15 |

**Recommended for**: Fast iteration, cost-effective experiments with good quality.

### Gemini 1.5 Series (Legacy)

Proven models with large context windows.

| Model | Display Name | Context | Vision | Functions | Cost (Input/Output per 1M) |
|-------|-------------|---------|--------|-----------|----------------------------|
| `gemini-1.5-pro` | Gemini 1.5 Pro | 2M tokens | ✅ | ❌ | $1.25 / $5.00 |
| `gemini-1.5-flash` | Gemini 1.5 Flash | 1M tokens | ✅ | ❌ | $0.075 / $0.30 |
| `gemini-1.5-flash-8b` | Gemini 1.5 Flash 8B | 1M tokens | ✅ | ❌ | $0.037 / $0.15 |

**Recommended for**: Large context requirements, established workflows.

## 🔓 Gemma Models Overview

Open-source models from Google that can be run via OpenRouter or locally on Princeton cluster.

### Gemma 2 Series (2B, 9B, 27B)

The established generation with solid performance.

| Model | Display Name | Parameters | VRAM | GPU Required | Reasoning | Speed |
|-------|-------------|------------|------|--------------|-----------|-------|
| `gemma-2-27b` | Gemma 2 27B | 27B | 54GB | Yes | High | Medium |
| `gemma-2-9b` | Gemma 2 9B | 9B | 18GB | Yes | Medium | Fast |
| `gemma-2-2b` | Gemma 2 2B | 2B | 4GB | Yes | Medium | Fast |

**OpenRouter API Names**:
- `google/gemma-2-27b-it`
- `google/gemma-2-9b-it`
- `google/gemma-2-2b-it`

### Gemma 3 Series (1B, 4B, 12B, 27B) - Latest

The newest generation with improved capabilities and efficiency.

| Model | Display Name | Parameters | VRAM | GPU Required | Reasoning | Speed |
|-------|-------------|------------|------|--------------|-----------|-------|
| `gemma-3-27b` | Gemma 3 27B | 27B | 54GB | Yes | High | Medium |
| `gemma-3-12b` | Gemma 3 12B | 12B | 24GB | Yes | Medium | Fast |
| `gemma-3-4b` | Gemma 3 4B | 4B | 8GB | Yes | Medium | Fast |
| `gemma-3-1b` | Gemma 3 1B | 1B | 2GB | No | Low | Very Fast |

**OpenRouter API Names**:
- `google/gemma-3-27b-it`
- `google/gemma-3-12b-it`
- `google/gemma-3-4b-it`
- `google/gemma-3-1b-it`

## 🚀 Usage Examples

### Basic Gemini Configuration

```yaml
# Using latest Gemini 2.5 models
providers:
  google:
    provider: "google"
    # API key will be read from GOOGLE_API_KEY environment variable

available_models:
  gemini-2.5-pro:
    # Model spec automatically loaded from registry
  gemini-2.5-flash:
    # Model spec automatically loaded from registry

agents:
  - agent_id: "strategic_gemini_pro"
    model_id: "gemini-2.5-pro"
    temperature: 0.7
    strategic_level: "balanced"

  - agent_id: "quick_gemini_flash"
    model_id: "gemini-2.5-flash" 
    temperature: 0.8
    strategic_level: "cooperative"
```

### Mixed Gemini/Gemma Configuration

```yaml
# Combining cloud Gemini with open-source Gemma
providers:
  google:
    provider: "google"
  openrouter:
    provider: "openrouter"

available_models:
  gemini-2.5-flash:
    # Cloud-based fast reasoning
  gemma-3-12b:
    # Latest open-source alternative
  gemma-2-9b:
    # Established open-source model

agents:
  - agent_id: "cloud_agent"
    model_id: "gemini-2.5-flash"
    system_prompt: "You are a cloud-powered negotiator with advanced capabilities."

  - agent_id: "open_source_latest"
    model_id: "gemma-3-12b"
    system_prompt: "You represent the latest in open-source AI technology."
    
  - agent_id: "open_source_stable"
    model_id: "gemma-2-9b" 
    system_prompt: "You bring proven open-source reliability to this negotiation."
```

### Princeton Cluster Gemma

```yaml
# Running Gemma models on Princeton cluster
providers:
  princeton_cluster:
    provider: "princeton_cluster"

available_models:
  gemma-2-12b-local:
    display_name: "Gemma 2 12B (Local)"
    family: "gemma"
    provider: "princeton_cluster"
    local_path: "/scratch/gpfs/DANQIC/models/gemma-2-12b-it"
    context_window: 8192
    requires_gpu: true
    estimated_vram_gb: 24

cluster_config:
  slurm_partition: "gpu"
  slurm_gpus_per_node: 1
  slurm_mem: "32GB"
```

## 🎯 Model Selection Guidelines

### For Research Quality
- **High reasoning**: `gemini-2.5-pro`, `gemma-2-27b`
- **Balanced performance**: `gemini-2.5-flash`, `gemma-2-12b`
- **Fast iteration**: `gemini-2.5-flash-lite`, `gemma-2-4b`

### For Cost Optimization
1. **Most cost-effective**: `gemini-2.5-flash-lite` ($0.037/$0.15 per 1M tokens)
2. **Best value**: `gemini-2.5-flash` ($0.075/$0.30 per 1M tokens)
3. **Open source**: `gemma-2-*` (via OpenRouter or local deployment)

### For Specific Use Cases
- **Large context needs**: `gemini-2.5-pro`, `gemini-1.5-pro` (2M tokens)
- **Vision capabilities**: Any Gemini model (all support vision)
- **Function calling**: Gemini 2.5/2.0 series
- **Local deployment**: Gemma series (open source)
- **Resource-constrained**: `gemma-2-1b` (CPU-compatible)

## 🔑 API Key Setup

### Google AI API
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

Get your key at: https://aistudio.google.com/apikey

### OpenRouter (for Gemma models)
```bash
export OPENROUTER_API_KEY="your-openrouter-key" 
```

Get your key at: https://openrouter.ai/keys

## ⚡ Performance Characteristics

### Reasoning Capability Hierarchy
1. **High**: `gemini-2.5-pro`, `gemini-1.5-pro`, `gemma-2-27b`
2. **Medium**: `gemini-2.5-flash`, `gemini-1.5-flash`, `gemma-2-12b`, `gemma-2-4b`
3. **Low**: `gemma-2-1b`

### Speed Rankings
1. **Very Fast**: `gemini-2.5-flash-lite`, `gemini-1.5-flash-8b`, `gemma-2-1b`
2. **Fast**: `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemma-2-12b`, `gemma-2-4b`
3. **Medium**: `gemini-2.5-pro`, `gemini-1.5-pro`, `gemma-2-27b`

## 🧪 Experimental Recommendations

### O3 vs Google Models
Compare OpenAI's reasoning model with Google's offerings:

```yaml
agents:
  - agent_id: "openai_o3"
    model_id: "o3"
  - agent_id: "gemini_pro"
    model_id: "gemini-2.5-pro"  
  - agent_id: "gemma_large"
    model_id: "gemma-2-27b"
```

### Multi-Tier Google Experiment
Test different capability levels within Google's ecosystem:

```yaml
agents:
  - agent_id: "premium_tier"
    model_id: "gemini-2.5-pro"
  - agent_id: "standard_tier"
    model_id: "gemini-2.5-flash"
  - agent_id: "budget_tier"  
    model_id: "gemini-2.5-flash-lite"
  - agent_id: "open_source_tier"
    model_id: "gemma-2-12b"
```

## 💡 Tips and Best Practices

1. **Start with Flash models**: `gemini-2.5-flash` offers the best balance of quality and cost
2. **Use Pro for final runs**: Reserve `gemini-2.5-pro` for important experiments
3. **Consider Gemma for baselines**: Open-source models provide good baselines without API costs
4. **Leverage huge context**: Gemini models can handle very long negotiation histories
5. **Mix cloud and local**: Use cloud for main agents, local Gemma for control conditions

## 🔍 Troubleshooting

### Common Issues
- **API key errors**: Ensure environment variables are set correctly
- **Rate limits**: Use the configured rate limiting in provider settings
- **Context length**: Gemini models have very large context windows but check your usage
- **OpenRouter availability**: Some Gemma models may have availability windows

### Model-Specific Notes
- **Gemini vision**: All models support image input (useful for diagram-based negotiations)
- **Function calling**: Only available in 2.5 and 2.0 series
- **Gemma deployment**: Larger models require significant GPU memory
- **Cost monitoring**: Set up billing alerts for Google AI API usage

This comprehensive model support enables sophisticated multi-agent experiments across the full spectrum of Google's AI offerings, from cutting-edge cloud models to open-source alternatives.