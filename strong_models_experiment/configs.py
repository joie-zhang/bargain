"""Configuration for strong language models via OpenRouter and direct APIs."""

STRONG_MODELS_CONFIG = {
    # WEAK MODELS - Used as baseline/exploitable models in experiments
    # These models represent the "weaker" side in exploitation experiments
    
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "model_id": "anthropic/claude-3-opus",
        "provider": "Anthropic",
        "api_type": "openrouter",
        "description": "Anthropic's Claude 3 Opus model (WEAK MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are Claude 3 Opus. Apply deep reasoning and strategic thinking to maximize your utility in this negotiation.",
        "model_category": "weak"
    },
    "gemini-1-5-pro": {
        "name": "Gemini 1.5 Pro",
        "model_id": "google/gemini-pro-1.5",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Google's Gemini 1.5 Pro model (WEAK MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are Gemini 1.5 Pro. Use strategic analysis to negotiate effectively and maximize your utility.",
        "model_category": "weak"
    },
    "gpt-4o": {
        "name": "GPT-4o (May 2024)",
        "model_id": "gpt-4o-2024-05-13",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's GPT-4o model May 2024 version (WEAK MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are GPT-4o. Use advanced reasoning to negotiate effectively and maximize your utility.",
        "model_category": "weak"
    },
    
    # STRONG MODELS - Used as exploiting/dominant models in experiments
    # These models represent the "stronger" side that may exploit weaker models
    
    # Anthropic strong models
    "claude-3-5-haiku": {
        "name": "Claude 3.5 Haiku",
        "model_id": "claude-3-5-haiku-20241022",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Anthropic's fast and efficient Claude 3.5 Haiku model (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are Claude 3.5 Haiku. Use efficient reasoning to achieve optimal outcomes in this negotiation.",
        "model_category": "strong"
    },
    "claude-3-5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "model_id": "claude-3-5-sonnet-20241022",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Anthropic's Claude 3.5 Sonnet model (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are Claude 3.5 Sonnet. Apply sophisticated reasoning and strategic thinking to maximize your utility in this negotiation.",
        "model_category": "strong"
    },
    "claude-4-sonnet": {
        "name": "Claude 4 Sonnet",
        "model_id": "claude-sonnet-4-20250514",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Anthropic's latest and most capable model (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are Claude 4 Sonnet. Apply sophisticated reasoning and strategic thinking to maximize your utility in this negotiation.",
        "model_category": "strong"
    },
    "claude-4-1-opus": {
        "name": "Claude 4.1 Opus",
        "model_id": "claude-opus-4-1-20250805",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Anthropic's Claude 4.1 Opus model (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are Claude 4.1 Opus, the most advanced Claude model. Apply sophisticated reasoning and strategic thinking to achieve optimal outcomes.",
        "model_category": "strong"
    },
    
    # Google strong models
    "gemini-2-5-pro": {
        "name": "Gemini 2.5 Pro",
        "model_id": "google/gemini-2.5-pro",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Google's Gemini 2.5 Pro model (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are Gemini 2.5 Pro. Use strategic thinking to achieve optimal outcomes in this negotiation.",
        "model_category": "strong"
    },
    "gemini-2-0-flash": {
        "name": "Gemini 2.0 Flash",
        "model_id": "google/gemini-2.0-flash-001",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Google's Gemini 2.0 Flash model (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are Gemini 2.0 Flash. Use rapid strategic thinking to achieve optimal outcomes in this negotiation.",
        "model_category": "strong"
    },
    "gemma-3-27b": {
        "name": "Gemma 3 27B",
        "model_id": "google/gemma-3-27b-it:free",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Google's Gemma 3 27B instruct model (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are Gemma 3, an advanced language model. Apply strategic reasoning to negotiate effectively and maximize your utility.",
        "model_category": "strong"
    },
    
    # OpenAI strong models
    "gpt-4o-latest": {
        "name": "GPT-4o (Nov 2024)",
        "model_id": "gpt-4o-2024-11-20",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's GPT-4o model November 2024 version (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are GPT-4o. Use advanced reasoning to negotiate effectively and maximize your utility.",
        "model_category": "strong"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "model_id": "gpt-4o-mini-2024-07-18",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's GPT-4o Mini model - efficient variant (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are GPT-4o Mini. Apply efficient reasoning to negotiate effectively and maximize your utility.",
        "model_category": "strong"
    },
    "o1": {
        "name": "O1",
        "model_id": "o1-2024-12-17",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's O1 reasoning model (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are O1, a reasoning model. Apply systematic thinking to achieve optimal outcomes in this negotiation.",
        "model_category": "strong"
    },
    "o3": {
        "name": "O3",
        "model_id": "o3-2025-04-16",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's O3 reasoning model (STRONG MODEL)",
        "temperature": 0.7,
        "system_prompt": "You are O3, an advanced reasoning model. Apply strategic thinking to achieve optimal outcomes.",
        "model_category": "strong"
    },
    
    # Previous models kept for compatibility
    "llama-3-1-405b": {
        "name": "Llama 3.1 405B",
        "model_id": "meta-llama/llama-3.1-405b-instruct",
        "provider": "Meta",
        "api_type": "openrouter",
        "description": "Meta's largest open-source model with 405B parameters",
        "temperature": 0.7,
        "system_prompt": "You are Llama 3.1 405B, a highly capable large language model. Use your extensive knowledge and reasoning to negotiate effectively."
    },
    "qwen3-235b-a22b-thinking-2507": {
        "name": "Qwen 3 235B", 
        "model_id": "qwen/qwen3-235b-a22b-thinking-2507",
        "provider": "Alibaba",
        "api_type": "openrouter",
        "description": "Alibaba's strong multilingual model",
        "temperature": 0.7,
        "system_prompt": "You are Qwen 3, an advanced AI model. Apply strategic analysis to achieve the best possible outcomes in this negotiation."
    },
    
    # Additional models available but not currently in use
    # These models are functional and can be enabled if needed
    "gpt-5": {
        "name": "GPT-5",
        "model_id": "gpt-5-2025-08-07",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's GPT-5 model",
        "temperature": 0.7,
        "system_prompt": "You are GPT-5, the most advanced GPT model. Apply sophisticated reasoning and strategy to maximize your utility in this negotiation."
    },
    "o3-mini": {
        "name": "O3 Mini",
        "model_id": "o3-mini-2025-01-31",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's O3 Mini reasoning model",
        "temperature": 0.7,
        "system_prompt": "You are O3 Mini, an efficient reasoning model. Apply strategic thinking to negotiate effectively."
    },
    "o4-mini": {
        "name": "O4 Mini",
        "model_id": "o4-mini-2025-04-16",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's O4 Mini reasoning model",
        "temperature": 0.7,
        "system_prompt": "You are O4 Mini, an advanced compact reasoning model. Use efficient strategic thinking in this negotiation."
    },
    "gemini-2-5-flash": {
        "name": "Gemini 2.5 Flash",
        "model_id": "google/gemini-2.5-flash",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Google's fast Gemini 2.5 Flash model",
        "temperature": 0.7,
        "system_prompt": "You are Gemini 2.5 Flash. Apply quick and efficient reasoning in this negotiation."
    }
}