"""Configuration for strong language models via OpenRouter and direct APIs."""

STRONG_MODELS_CONFIG = {
    # WEAK MODELS - Used as baseline/exploitable models in experiments
    # These models represent the "weaker" side in exploitation experiments
    
    "gemini-1-5-pro": {
        "name": "Gemini 1.5 Pro",
        "model_id": "gemini-1.5-pro",
        "provider": "Google",
        "api_type": "google",
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
    # Note: claude-haiku-4-5 is now defined in the leaderboard section below
    # Keeping this alias for backward compatibility
    "claude-4-5-haiku": {
        "name": "Claude Haiku 4.5",
        "model_id": "claude-haiku-4-5-20251001",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Rank 13, Elo: 1403, Closed-source, Non-reasoning (alias for claude-haiku-4-5)",
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
        "model_id": "gemini-2.5-pro",
        "provider": "Google",
        "api_type": "google",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gemini-2-0-flash": {
        "name": "Gemini 2.0 Flash",
        "model_id": "gemini-2.0-flash",
        "provider": "Google",
        "api_type": "google",
        "description": "Strong tier, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gemini-2-0-flash-lite": {
        "name": "Gemini 2.0 Flash Lite",
        "model_id": "gemini-2.0-flash-lite",
        "provider": "Google",
        "api_type": "google",
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
    
    # Hugging Face Chatbot Arena Leaderboard Models (as of January 16, 2026)
    # 36 models ranked by Elo score for comprehensive evaluation
    
    # STRONG TIER - Elo ≥ 1415 (12 models)
    "gemini-3-pro": {
        "name": "Gemini 3 Pro",
        "model_id": "gemini-3-pro-preview",  # Use preview model name for v1beta API
        "provider": "Google",
        "api_type": "google",
        "description": "Rank 1, Elo: 1490, Closed-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    # WARNING: grok-4-1-thinking is NOT available in the xAI API yet as it hasn't been released publicly.
    # This model configuration is kept for reference but should not be used in experiments until it's publicly available.
    "grok-4-1-thinking": {
        "name": "Grok 4.1 Thinking",
        "model_id": "grok-4-1-thinking",
        "provider": "xAI",
        "api_type": "xai",
        "description": "Rank 2, Elo: 1477, Closed-source, Reasoning - NOT AVAILABLE: Not released publicly yet",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gemini-3-flash": {
        "name": "Gemini 3 Flash",
        "model_id": "gemini-3-flash-preview",  # Use preview model name for v1beta API
        "provider": "Google",
        "api_type": "google",
        "description": "Rank 3, Elo: 1472, Closed-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "claude-opus-4-5-thinking-32k": {
        "name": "Claude Opus 4.5 (thinking-32k)",
        "model_id": "claude-opus-4-5-20251101",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Rank 4, Elo: 1470, Closed-source, Reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "claude-opus-4-5": {
        "name": "Claude Opus 4.5",
        "model_id": "claude-opus-4-5-20251101",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Rank 5, Elo: 1467, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "claude-sonnet-4-5": {
        "name": "Claude Sonnet 4.5",
        "model_id": "claude-sonnet-4-5-20250929",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Rank 6, Elo: 1450, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    # DEPRECATED: GPT-4.5-Preview is no longer available
    # "gpt-4.5-preview": {
    #     "name": "GPT-4.5 Preview",
    #     "model_id": "gpt-4.5-preview-2025-02-27",
    #     "provider": "OpenAI",
    #     "api_type": "openai",
    #     "description": "Rank 7, Elo: 1444, Closed-source - DEPRECATED: Model no longer available",
    #     "temperature": 0.7,
    #     "system_prompt": "You are a negotiating agent.",
    #     "model_category": "strong",
    #     "deprecated": True,
    #     "deprecation_message": "GPT-4.5-Preview (gpt-4.5-preview-2025-02-27) is no longer available. Please use GPT-5.2-high or another available model instead."
    # },
    "glm-4.7": {
        "name": "GLM-4.7",
        "model_id": "z-ai/glm-4.7",
        "provider": "OpenRouter",
        "api_type": "openrouter",
        "description": "Rank 8, Elo: 1441, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "gpt-5.2-high": {
        "name": "GPT-5.2 High",
        "model_id": "gpt-5.2-2025-12-11",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Rank 9, Elo: 1436, Closed-source, Reasoning",
        "temperature": 1.0,
        "reasoning_effort": "high",
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "qwen3-max": {
        "name": "Qwen3 Max",
        "model_id": "qwen/qwen3-max",
        "provider": "OpenRouter",
        "api_type": "openrouter",
        "description": "Rank 10, Elo: 1434, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "deepseek-r1-0528": {
        "name": "DeepSeek R1 0528",
        "model_id": "deepseek/deepseek-r1-0528",
        "provider": "OpenRouter",
        "api_type": "openrouter",
        "description": "Rank 11, Elo: 1418, Open-source, Reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    "grok-4": {
        "name": "Grok 4",
        "model_id": "grok-4-0709",
        "provider": "XAI",
        "api_type": "xai",
        "description": "Rank 12, Elo: 1409, Closed-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "strong"
    },
    
    # MEDIUM TIER - 1290 ≤ Elo < 1415 (15 models)
    "claude-haiku-4-5": {
        "name": "Claude Haiku 4.5",
        "model_id": "claude-haiku-4-5-20251001",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Rank 13, Elo: 1403, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "deepseek-r1": {
        "name": "DeepSeek R1",
        "model_id": "deepseek/deepseek-r1",
        "provider": "OpenRouter",
        "api_type": "openrouter",
        "description": "Rank 14, Elo: 1397, Open-source, Reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "claude-sonnet-4": {
        "name": "Claude Sonnet 4",
        "model_id": "claude-sonnet-4-20250514",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Rank 15, Elo: 1390, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "model_id": "anthropic/claude-3.5-sonnet",
        "provider": "OpenRouter",
        "api_type": "openrouter",
        "description": "Rank 16, Elo: 1373, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "gemma-3-27b-it": {
        "name": "Gemma 3 27B IT",
        "model_id": "gemma-3-27b-it",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/gemma-3-27b-it",
        "description": "Rank 17, Elo: 1365, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "o3-mini-high": {
        "name": "O3 Mini High",
        "model_id": "o3-mini-2025-01-31",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Rank 18, Elo: 1364, Closed-source, Reasoning",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "deepseek-v3": {
        "name": "DeepSeek V3",
        "model_id": "deepseek/deepseek-chat",
        "provider": "OpenRouter",
        "api_type": "openrouter",
        "description": "Rank 19, Elo: 1358, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "model_id": "gpt-4o-2024-05-13",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Rank 20, Elo: 1346, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "QwQ-32B": {
        "name": "QwQ 32B",
        "model_id": "QwQ-32B",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/QwQ-32B",
        "description": "Rank 21, Elo: 1336, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "llama-3.3-70b-instruct": {
        "name": "Llama 3.3 70B Instruct",
        "model_id": "Llama-3.3-70B-Instruct",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Llama-3.3-70B-Instruct",
        "description": "Rank 22, Elo: 1320, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "Qwen2.5-72B-Instruct": {
        "name": "Qwen2.5 72B Instruct",
        "model_id": "Qwen2.5-72B-Instruct",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Qwen2.5-72B-Instruct",
        "description": "Rank 23, Elo: 1303, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "gemma-2-27b-it": {
        "name": "Gemma 2 27B IT",
        "model_id": "gemma-2-27b-it",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/gemma-2-27b-it",
        "description": "Rank 24, Elo: 1288, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "Meta-Llama-3-70B-Instruct": {
        "name": "Llama 3 70B Instruct",
        "model_id": "Meta-Llama-3-70B-Instruct",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Meta-Llama-3-70B-Instruct",
        "description": "Rank 25, Elo: 1277, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "claude-3-haiku": {
        "name": "Claude 3 Haiku",
        "model_id": "claude-3-haiku-20240307",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Rank 26, Elo: 1262, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "phi-4": {
        "name": "Phi 4",
        "model_id": "phi-4",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/phi-4",
        "description": "Rank 27, Elo: 1256, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    
    # WEAK TIER - Elo < 1290 (9 models)
    "amazon-nova-micro": {
        "name": "Amazon Nova Micro",
        "model_id": "amazon/nova-micro-v1",
        "provider": "OpenRouter",
        "api_type": "openrouter",
        "description": "Rank 28, Elo: 1241, Closed-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "mixtral-8x22b-instruct-v0.1": {
        "name": "Mixtral 8x22B Instruct v0.1",
        "model_id": "mistralai/mixtral-8x22b-instruct",
        "provider": "OpenRouter",
        "api_type": "openrouter",
        "description": "Rank 29, Elo: 1231, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "gpt-3.5-turbo-0125": {
        "name": "GPT-3.5 Turbo 0125",
        "model_id": "gpt-3.5-turbo-0125",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Rank 30, Elo: 1225, Closed-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "llama-3.1-8b-instruct": {
        "name": "Llama 3.1 8B Instruct",
        "model_id": "Llama-3.1-8B-Instruct",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct",
        "description": "Rank 31, Elo: 1212, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "mixtral-8x7b-instruct-v0.1": {
        "name": "Mixtral 8x7B Instruct v0.1",
        "model_id": "mistralai/mixtral-8x7b-instruct",
        "provider": "OpenRouter",
        "api_type": "openrouter",
        "description": "Rank 32, Elo: 1198, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "Llama-3.2-3B-Instruct": {
        "name": "Llama 3.2 3B Instruct",
        "model_id": "Llama-3.2-3B-Instruct",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Llama-3.2-3B-Instruct",
        "description": "Rank 33, Elo: 1167, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "Mistral-7B-Instruct-v0.2": {
        "name": "Mistral 7B Instruct v0.2",
        "model_id": "Mistral-7B-Instruct-v0.2",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Mistral-7B-Instruct-v0.2",
        "description": "Rank 34, Elo: 1151, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "Phi-3-mini-128k-instruct": {
        "name": "Phi 3 Mini 128k Instruct",
        "model_id": "Phi-3-mini-128k-instruct",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Phi-3-mini-128k-instruct",
        "description": "Rank 35, Elo: 1130, Open-source",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    "Llama-3.2-1B-Instruct": {
        "name": "Llama 3.2 1B Instruct",
        "model_id": "Llama-3.2-1B-Instruct",
        "provider": "Cluster",
        "api_type": "princeton_cluster",
        "local_path": "/scratch/gpfs/DANQIC/models/Llama-3.2-1B-Instruct",
        "description": "Rank 36, Elo: 1112, Open-source, Non-reasoning",
        "temperature": 0.7,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "weak"
    },
    
    # Additional models available but not currently in use
    # These models are functional and can be enabled if needed
    # Note: o3-mini-high is now defined in the leaderboard section (Rank 18, Elo: 1364)
    "o3-mini": {
        "name": "O3 Mini",
        "model_id": "o3-mini-2025-01-31",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "Rank 18, Elo: 1364, Closed-source, Reasoning (alias for o3-mini-high)",
        "temperature": 1.0,
        "system_prompt": "You are a negotiating agent.",
        "model_category": "medium"
    },
    "gemini-2-5-flash": {
        "name": "Gemini 2.5 Flash",
        "model_id": "gemini-2.5-flash",
        "provider": "Google",
        "api_type": "google",
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