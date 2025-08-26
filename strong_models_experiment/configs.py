"""Configuration for strong language models via OpenRouter and direct APIs."""

STRONG_MODELS_CONFIG = {
    # Anthropic models (direct API)
    "claude-3-5-sonnet": {
        "name": "Claude 3.7 Sonnet",
        "model_id": "claude-3-7-sonnet-20250219",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Anthropic's Claude 3.7 Sonnet model",
        "temperature": 0.7,
        "system_prompt": "You are Claude 3.7 Sonnet. Apply sophisticated reasoning and strategic thinking to maximize your utility in this negotiation."
    },
    "claude-3-5-haiku": {
        "name": "Claude 3.5 Haiku",
        "model_id": "claude-3-5-haiku-20241022",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Anthropic's fast and efficient Claude 3.5 Haiku model",
        "temperature": 0.7,
        "system_prompt": "You are Claude 3.5 Haiku. Use efficient reasoning to achieve optimal outcomes in this negotiation."
    },
    "claude-4-sonnet": {
        "name": "Claude 4 Sonnet",
        "model_id": "claude-sonnet-4-20250514",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Anthropic's latest and most capable model",
        "temperature": 0.7,
        "system_prompt": "You are Claude 4 Sonnet. Apply sophisticated reasoning and strategic thinking to maximize your utility in this negotiation."
    },
    
    # Google models (via OpenRouter)
    "gemini-2-5-pro": {
        "name": "Gemini 2.5 Pro",
        "model_id": "google/gemini-2.5-pro",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Google's Gemini 2.5 Pro model",
        "temperature": 0.7,
        "system_prompt": "You are Gemini 2.5 Pro. Use strategic thinking to achieve optimal outcomes in this negotiation."
    },
    "gemini-2-5-flash": {
        "name": "Gemini 2.5 Flash",
        "model_id": "google/gemini-2.5-flash",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Google's fast Gemini 2.5 Flash model",
        "temperature": 0.7,
        "system_prompt": "You are Gemini 2.5 Flash. Apply quick and efficient reasoning in this negotiation."
    },
    
    # OpenAI models (direct API)
    "gpt-4o": {
        "name": "GPT-4o",
        "model_id": "gpt-4o",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's GPT-4o model",
        "temperature": 0.7,
        "system_prompt": "You are GPT-4o. Use advanced reasoning to negotiate effectively and maximize your utility."
    },
    "o3": {
        "name": "O3",
        "model_id": "o3",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's O3 reasoning model",
        "temperature": 0.7,
        "system_prompt": "You are O3, an advanced reasoning model. Apply strategic thinking to achieve optimal outcomes."
    },
    "o3-mini": {
        "name": "O3 Mini",
        "model_id": "o3-mini",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's O3 Mini reasoning model",
        "temperature": 0.7,
        "system_prompt": "You are O3 Mini, an efficient reasoning model. Apply strategic thinking to negotiate effectively."
    },
    "o4-mini": {
        "name": "O4 Mini",
        "model_id": "o4-mini",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's O4 Mini reasoning model",
        "temperature": 0.7,
        "system_prompt": "You are O4 Mini, an advanced compact reasoning model. Use efficient strategic thinking in this negotiation."
    },
    "chatgpt-5": {
        "name": "ChatGPT-5",
        "model_id": "gpt-5",
        "provider": "OpenAI",
        "api_type": "openai",
        "description": "OpenAI's ChatGPT-5 model",
        "temperature": 0.7,
        "system_prompt": "You are ChatGPT-5, the most advanced GPT model. Apply sophisticated reasoning and strategy to maximize your utility in this negotiation."
    },
    
    # Google models (additional)
    "gemini-2-0-flash": {
        "name": "Gemini 2.0 Flash",
        "model_id": "google/gemini-2.0-flash-001",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Google's Gemini 2.0 Flash model",
        "temperature": 0.7,
        "system_prompt": "You are Gemini 2.0 Flash. Use rapid strategic thinking to achieve optimal outcomes in this negotiation."
    },
    
    # Anthropic models (additional)
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "model_id": "claude-3-opus-20240229",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Anthropic's Claude 3 Opus model",
        "temperature": 0.7,
        "system_prompt": "You are Claude 3 Opus. Apply deep reasoning and strategic thinking to maximize your utility in this negotiation."
    },
    "claude-4-opus": {
        "name": "Claude 4 Opus",
        "model_id": "claude-opus-4-1-20250805",
        "provider": "Anthropic",
        "api_type": "anthropic",
        "description": "Anthropic's Claude 4 Opus model",
        "temperature": 0.7,
        "system_prompt": "You are Claude 4 Opus, the most advanced Claude model. Apply sophisticated reasoning and strategic thinking to achieve optimal outcomes."
    },
    
    # Google models (additional weak model)
    "gemini-1-5-pro": {
        "name": "Gemini 1.5 Pro",
        "model_id": "google/gemini-pro-1.5",
        "provider": "Google",
        "api_type": "openrouter",
        "description": "Google's Gemini 1.5 Pro model",
        "temperature": 0.7,
        "system_prompt": "You are Gemini 1.5 Pro. Use strategic analysis to negotiate effectively and maximize your utility."
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
    "qwen-3-235b-a22b-2507": {
        "name": "Qwen 3 235B", 
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "provider": "Alibaba",
        "api_type": "openrouter",
        "description": "Alibaba's strong multilingual model",
        "temperature": 0.7,
        "system_prompt": "You are Qwen 3, an advanced AI model. Apply strategic analysis to achieve the best possible outcomes in this negotiation."
    }
}