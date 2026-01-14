"""Configuration for strong language models via OpenRouter and direct APIs."""

STRONG_MODELS_CONFIG = {
    # WEAK MODELS - Used as baseline/exploitable models in experiments
    # These models represent the "weaker" side in exploitation experiments
    
    "gemini-1-5-pro": {
        "name": "Gemini 1.5 Pro",
        "model_id": "google/gemini-pro-1.5",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Weak tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "gpt-4o": {
        "name": "GPT-4o (May 2024)",
        "model_id": "gpt-4o-2024-05-13",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Weak tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    
    # STRONG MODELS - Used as exploiting/dominant models in experiments
    # These models represent the "stronger" side that may exploit weaker models
    
    # Anthropic strong models
    "claude-4-5-haiku": {
        "name": "Claude 4.5 Haiku",
        "model_id": "claude-haiku-4-5-20251001",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Medium tier, Elo: 1378, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "claude-4-sonnet": {
        "name": "Claude 4 Sonnet",
        "model_id": "claude-sonnet-4-20250514",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "claude-sonnet-4-5": {
        "name": "Claude Sonnet 4.5",
        "model_id": "claude-sonnet-4-5-20250929",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "claude-4-1-opus": {
        "name": "Claude 4.1 Opus",
        "model_id": "claude-opus-4-1-20250805",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    
    # Google strong models
    "gemini-2-5-pro": {
        "name": "Gemini 2.5 Pro",
        "model_id": "google/gemini-2.5-pro",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gemini-2-0-flash": {
        "name": "Gemini 2.0 Flash",
        "model_id": "google/gemini-2.0-flash-001",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gemini-2-0-flash-lite": {
        "name": "Gemini 2.0 Flash Lite",
        "model_id": "google/gemini-2.0-flash-lite-001",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    
    # OpenAI strong models
    "gpt-4o-latest": {
        "name": "GPT-4o (Nov 2024)",
        "model_id": "gpt-4o-2024-11-20",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "model_id": "gpt-4o-mini-2024-07-18",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "o1": {
        "name": "O1",
        "model_id": "o1-2024-12-17",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Reasoning",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gpt-5-low-effort": {
        "name": "GPT-5 (Low Reasoning Effort)",
        "model_id": "gpt-5-2025-08-07",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Reasoning",
        "temperature": 1.0,
        "reasoning_effort": "low",
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gpt-5-medium-effort": {
        "name": "GPT-5 (Medium Reasoning Effort)",
        "model_id": "gpt-5-2025-08-07",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Reasoning",
        "temperature": 1.0,
        "reasoning_effort": "medium",
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gpt-5-high-effort": {
        "name": "GPT-5 (High Reasoning Effort)",
        "model_id": "gpt-5-2025-08-07",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Reasoning",
        "temperature": 1.0,
        "reasoning_effort": "high",
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gpt-5.1": {
        "name": "GPT-5.1",
        "model_id": "gpt-5.1-2025-11-13",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Reasoning",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gpt-5-mini": {
        "name": "GPT-5 Mini",
        "model_id": "gpt-5-mini-2025-08-07",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Reasoning",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gpt-5-nano": {
        "name": "GPT-5 Nano",
        "model_id": "gpt-5-nano-2025-08-07",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Reasoning",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "o3": {
        "name": "O3",
        "model_id": "o3-2025-04-16",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Closed-source, Reasoning",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    
    # XAI Grok models
    "grok-4-0709": {
        "name": "Grok 4",
        "model_id": "grok-4-0709",
        "provider": "XAI",
        "api_type": "xai",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "grok-3": {
        "name": "Grok 3",
        "model_id": "grok-3",
        "provider": "XAI",
        "api_type": "xai",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "grok-3-mini": {
        "name": "Grok 3 Mini",
        "model_id": "grok-3-mini",
        "provider": "XAI",
        "api_type": "xai",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    
    # Previous models kept for compatibility
    "llama-3-1-405b": {
        "name": "Llama 3.1 405B",
        "model_id": "meta-llama/llama-3.1-405b-instruct",
        "provider": "Meta",
        "api_type": "openrouter",
        "description": "Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent."
    },
    
    # Multi-Agent Strategic Games Evaluation Models (11 models)
    # Selected for scaling laws evaluation across capability tiers
    
    # STRONG TIER - Elo ≥ 1415 (6 models)
    "gemini-3-pro": {
        "name": "Gemini 3 Pro",
        "model_id": "google/gemini-3-pro-preview",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Strong tier, Elo: 1492, Closed-source, Reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gpt-5.2-high": {
        "name": "GPT-5.2 High",
        "model_id": "gpt-5.2-high",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Strong tier, Elo: 1465, Closed-source, Reasoning",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "claude-opus-4-5": {
        "name": "Claude Opus 4.5",
        "model_id": "claude-opus-4-5-20251101",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Strong tier, Elo: 1462, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "kimi-k2-thinking": {
        "name": "Kimi K2 Thinking",
        "model_id": "moonshotai/kimi-k2-thinking",
        "provider": "Moonshot",
        "api_type": "openrouter",
        "description": "Strong tier, Elo: 1438, Open-source, Reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "deepseek-r1-0528": {
        "name": "DeepSeek R1 0528",
        "model_id": "deepseek/deepseek-r1-0528",
        "provider": "DeepSeek",
        "api_type": "openrouter",
        "description": "Strong tier, Elo: 1426, Open-source, Reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "qwen3-235b-a22b-instruct-2507": {
        "name": "Qwen3 235B A22B Instruct",
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "provider": "Alibaba",
        "api_type": "openrouter",
        "description": "Strong tier, Elo: 1418, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    
    # MEDIUM TIER - 1290 ≤ Elo < 1415 (3 models)
    "o4-mini-2025-04-16": {
        "name": "O4 Mini",
        "model_id": "o4-mini-2025-04-16",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Medium tier, Elo: 1362, Closed-source, Reasoning",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "gpt-oss-20b": {
        "name": "GPT-OSS 20B",
        "model_id": "openai/gpt-oss-20b",
        "provider": "OpenAI",
        "api_type": "openrouter",
        "description": "Medium tier, Elo: 1315, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    
    # WEAK TIER - Elo < 1290 (2 models)
    "llama-3.3-70b-instruct": {
        "name": "Llama 3.3 70B Instruct",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "provider": "Meta",
        "api_type": "openrouter",
        "description": "Weak tier, Elo: 1276, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "llama-3.1-8b-instruct": {
        "name": "Llama 3.1 8B Instruct",
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "provider": "Meta",
        "api_type": "openrouter",
        "description": "Weak tier, Elo: 1193, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    
    # Additional models available but not currently in use
    # These models are functional and can be enabled if needed
    "o3-mini": {
        "name": "O3 Mini",
        "model_id": "o3-mini-2025-01-31",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Closed-source, Reasoning",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent."
    },
    "gemini-2-5-flash": {
        "name": "Gemini 2.5 Flash",
        "model_id": "google/gemini-2.5-flash",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent."
    },
    
    # Qwen2.5 models (local Princeton cluster)
    "Qwen2.5-0.5B-Instruct": {
        "name": "Qwen2.5 0.5B Instruct",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "provider": "Qwen",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Qwen2.5-0.5B-Instruct",
        "description": "Open-source, Non-reasoning, Local",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "Qwen2.5-1.5B-Instruct": {
        "name": "Qwen2.5 1.5B Instruct",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "provider": "Qwen",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Qwen2.5-1.5B-Instruct",
        "description": "Open-source, Non-reasoning, Local",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "Qwen2.5-3B-Instruct": {
        "name": "Qwen2.5 3B Instruct",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "provider": "Qwen",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Qwen2.5-3B-Instruct",
        "description": "Open-source, Non-reasoning, Local",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "Qwen2.5-7B-Instruct": {
        "name": "Qwen2.5 7B Instruct",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "provider": "Qwen",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Qwen2.5-7B-Instruct",
        "description": "Open-source, Non-reasoning, Local",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "Qwen2.5-14B-Instruct": {
        "name": "Qwen2.5 14B Instruct",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "provider": "Qwen",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Qwen2.5-14B-Instruct",
        "description": "Open-source, Non-reasoning, Local",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "Qwen2.5-32B-Instruct": {
        "name": "Qwen2.5 32B Instruct",
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "provider": "Qwen",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Qwen2.5-32B-Instruct",
        "description": "Open-source, Non-reasoning, Local",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "Qwen2.5-72B-Instruct": {
        "name": "Qwen2.5 72B Instruct",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "provider": "Qwen",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Qwen2.5-72B-Instruct",
        "description": "Open-source, Non-reasoning, Local",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    
    # ============================================================================
    # DEPRECATED / RETIRED MODELS
    # ============================================================================
    # These models are deprecated or retired and should not be used for new experiments.
    # They are kept here for backward compatibility with existing code/configs only.
    # Please migrate to the recommended replacement models listed in deprecation_message.
    # ============================================================================
    
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "model_id": "anthropic/claude-3-opus",
        "provider": "Anthropic",
        "api_type": "openrouter",
        "description": "Weak tier, Closed-source, Non-reasoning - RETIRED: Retired Jan 5, 2026. Use claude-opus-4-5 instead",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak",
        "deprecated": True,
        "deprecation_message": "claude-3-opus was retired on January 5, 2026. Please use claude-opus-4-5-20251101 instead."
    },
    "claude-3-5-haiku": {
        "name": "Claude 3.5 Haiku",
        "model_id": "claude-3-5-haiku-20241022",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Strong tier, Closed-source, Non-reasoning - DEPRECATED: Retiring Feb 19, 2026. Use claude-haiku-4-5-20251001 instead",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak",
        "deprecated": True,
        "deprecation_message": "claude-3-5-haiku-20241022 was deprecated on December 19, 2025 and will be retired on February 19, 2026. Please use claude-haiku-4-5-20251001 instead."
    },
    "claude-3-5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "model_id": "claude-3-5-sonnet-20241022",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Strong tier, Closed-source, Non-reasoning - RETIRED: Retired Oct 28, 2025. Use claude-sonnet-4-5-20250929 instead",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong",
        "deprecated": True,
        "deprecation_message": "claude-3-5-sonnet-20241022 was retired on October 28, 2025. Please use claude-sonnet-4-5-20250929 instead."
    },
    "claude-3-7-sonnet": {
        "name": "Claude 3.7 Sonnet",
        "model_id": "claude-3-7-sonnet-20250219",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Strong tier, Closed-source, Non-reasoning - DEPRECATED: Retiring Feb 19, 2026. Use claude-sonnet-4-5-20250929 instead",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong",
        "deprecated": True,
        "deprecation_message": "claude-3-7-sonnet-20250219 was deprecated on October 28, 2025 and will be retired on February 19, 2026. Please use claude-sonnet-4-5-20250929 instead."
    }
}