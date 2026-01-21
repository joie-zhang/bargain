# OpenRouter API Pricing

API pricing information for OpenRouter models used in the 36-model experiment.

---

## Platform Overview

OpenRouter applies a **5.5% platform fee** on top of provider pricing for pay-as-you-go users. All prices shown below are per 1M tokens (MTok) in USD and represent base provider prices before the platform fee.

### Platform Details

- **Platform Fee:** 5.5% markup for pay-as-you-go users
- **Billing:** Based on successful requests only
- **Reasoning Tokens:** For reasoning-enabled models, reasoning tokens are tracked as part of output tokens
- **Caching:** Some models support prompt caching with reduced input token costs

---

## Models Used in 36-Model Experiment

| Model | Provider | Input | Output | Context | Notes |
|-------|----------|-------|--------|---------|-------|
| GLM-4.7 | Z.AI | $0.40 | $1.50 | ~203K | Rank 8, Elo: 1441, Open-source |
| Qwen3 Max | Qwen | $1.20 | $6.00 | 256K | Rank 10, Elo: 1434, Open-source |
| DeepSeek R1 0528 | DeepSeek | $0.40 | $1.75 | - | Rank 11, Elo: 1418, Open-source, Reasoning |
| Grok 4 | xAI | $3.00* | $15.00* | 256K | Rank 12, Elo: 1409, Closed-source (doubles after 128K tokens) |
| DeepSeek R1 | DeepSeek | $0.70 | $2.50 | 64K | Rank 14, Elo: 1397, Open-source, Reasoning |
| Claude 3.5 Sonnet | Anthropic | $6.00 | $30.00 | - | Rank 16, Elo: 1373, Closed-source |
| DeepSeek V3 (Chat) | DeepSeek | $0.14 | $0.28 | - | Rank 19, Elo: 1358, Open-source (cache miss pricing) |
| Amazon Nova Micro | Amazon | $0.035 | $0.14 | 128K | Rank 28, Elo: 1241, Closed-source |
| Mixtral 8x22B Instruct v0.1 | Mistral AI | $2.00 | $6.00 | - | Rank 29, Elo: 1231, Open-source |
| Mixtral 8x7B Instruct v0.1 | Mistral AI | $0.08-$0.54 | $0.24-$0.54 | - | Rank 32, Elo: 1198, Open-source (varies by provider) |

**Model IDs (from configs.py):**
- `z-ai/glm-4.7` → GLM-4.7
- `qwen/qwen3-max` → Qwen3 Max
- `deepseek/deepseek-r1-0528` → DeepSeek R1 0528
- `x-ai/grok-4` → Grok 4
- `deepseek/deepseek-r1` → DeepSeek R1
- `anthropic/claude-3.5-sonnet` → Claude 3.5 Sonnet
- `deepseek/deepseek-chat` → DeepSeek V3
- `amazon/nova-micro-v1` → Amazon Nova Micro
- `mistralai/mixtral-8x22b-instruct` → Mixtral 8x22B Instruct v0.1
- `mistralai/mixtral-8x7b-instruct` → Mixtral 8x7B Instruct v0.1

---

## Pricing Notes

### Special Pricing Considerations

- **Grok 4:** Prices double when total tokens exceed 128,000 in a single request
  - Standard: $3.00/$15.00 per MTok
  - After 128K tokens: $6.00/$30.00 per MTok

- **Mixtral 8x7B:** Pricing varies significantly by provider route (range shown above)
  - Lowest: $0.08/$0.24 per MTok
  - Highest: $0.54/$0.54 per MTok
  - Check OpenRouter for current provider options

- **DeepSeek V3:** Cache hit pricing may be lower; shown prices are for cache misses
  - Cache miss: $0.14/$0.28 per MTok
  - Cache hits typically reduce input token costs

- **Amazon Nova Micro:** 128K context window, very low pricing
  - Input: $0.035 per MTok
  - Output: $0.14 per MTok

---

## Cost Estimation Notes

1. **Token Counting:** 
   - Input tokens include system prompts, user messages, and context
   - Output tokens include model responses
   - For reasoning models, reasoning tokens are billed as output tokens

2. **OpenRouter Pricing Calculation:**
   - Prices shown are base provider prices (before platform fee)
   - Add 5.5% platform fee for final cost
   - Example: If provider price is $1.00/MTok input, OpenRouter cost = $1.00 × 1.055 = $1.055/MTok
   - Some models may have multiple provider routes with different pricing
   - Reasoning tokens for reasoning-enabled models are billed as output tokens

3. **Pricing Variability:**
   - Provider pricing changes
   - Model availability
   - Regional differences
   - Volume discounts
   - Provider route selection

4. **Cost Optimization:**
   - Use prompt caching when available (reduces input token costs)
   - Consider batch processing for non-time-sensitive requests
   - Monitor token usage to identify optimization opportunities
   - Compare provider routes for models with multiple options (e.g., Mixtral 8x7B)

---

## Quick Reference: Models by Rank

| Rank | Model | Provider | Input $/MTok | Output $/MTok |
|------|-------|----------|--------------|---------------|
| 8 | GLM-4.7 | Z.AI | $0.40 | $1.50 |
| 10 | Qwen3 Max | Qwen | $1.20 | $6.00 |
| 11 | DeepSeek R1 0528 | DeepSeek | $0.40 | $1.75 |
| 12 | Grok 4 | xAI | $3.00* | $15.00* |
| 14 | DeepSeek R1 | DeepSeek | $0.70 | $2.50 |
| 16 | Claude 3.5 Sonnet | Anthropic | $6.00 | $30.00 |
| 19 | DeepSeek V3 | DeepSeek | $0.14 | $0.28 |
| 28 | Amazon Nova Micro | Amazon | $0.035 | $0.14 |
| 29 | Mixtral 8x22B Instruct | Mistral AI | $2.00 | $6.00 |
| 32 | Mixtral 8x7B Instruct | Mistral AI | $0.08-$0.54 | $0.24-$0.54 |

*Grok 4 prices double after 128K tokens in a single request.

---

## References

- [OpenRouter Pricing](https://openrouter.ai/pricing)
- [OpenRouter Models](https://openrouter.ai/models)

---

**Last Updated:** January 2026. Pricing information should be verified from official sources as rates may change frequently.
