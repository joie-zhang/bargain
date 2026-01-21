# Google Gemini API Pricing

API pricing information for Google Gemini models used in the 36-model experiment.

---

## Model Pricing

Pricing for Google Gemini models accessed via Google's API. All prices are per 1M tokens (MTok) in USD.

### Text Tokens

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| Gemini 3 Pro (≤200K tokens) | $2.00 | $12.00 | Rank 1, Elo: 1490, Standard context |
| Gemini 3 Pro (>200K tokens) | $4.00 | $18.00 | Rank 1, Elo: 1490, Long context (2M token window) |
| Gemini 3 Flash | $0.50 | $3.00 | Rank 3, Elo: 1472, Flat pricing (1M token window) |

**Model IDs (from configs.py):**
- `gemini-3-pro` → Gemini 3 Pro
- `gemini-3-flash` → Gemini 3 Flash

---

## Additional Features and Pricing

### Batch Processing
- **Batch API:** 50% discount available for batch processing
- Reduces costs for non-time-sensitive requests

### Context Caching
- **Context Caching:** Can reduce costs for repeated content by up to 90%
- Useful for applications with repeated context across requests

### Additional Services
- **Grounding with Google Search:** ~$35 per 1,000 requests
- **Audio Input:** Gemini 3 Flash supports audio at $1.00 per 1M tokens
- **Thinking Mode:** Additional output tokens consumed for reasoning chains

---

## Cost Estimation Notes

1. **Token Counting:** 
   - Input tokens include system prompts, user messages, and context
   - Output tokens include model responses
   - For reasoning models, reasoning tokens are billed as output tokens

2. **Context Window Pricing:**
   - Gemini 3 Pro uses tiered pricing based on context length
   - Standard pricing applies for requests ≤200K tokens
   - Long context pricing (2x input, 1.5x output) applies for requests >200K tokens
   - Gemini 3 Flash uses flat pricing regardless of context length

3. **Cost Optimization:**
   - Use context caching for repeated content (up to 90% savings)
   - Consider batch API for non-time-sensitive requests (50% discount)
   - Monitor token usage to identify optimization opportunities

---

## References

- [Google AI Studio Pricing](https://aistudio.google.com/pricing)
- [Google Cloud Vertex AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)

---

**Last Updated:** January 2026. Pricing information should be verified from official sources as rates may change frequently.
