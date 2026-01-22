"""
Unified Model Client Factory for Multi-Provider LLM Integration

This module implements client classes that can communicate with different
LLM providers through their native APIs, OpenRouter, or local deployment.
"""

from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import json
import time
import logging
from pathlib import Path
import os

from .model_config import (
    ModelSpec, 
    ModelProvider, 
    ProviderConfig, 
    AgentModelConfig
)

# Optional imports based on availability
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None

try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user, system
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    XAIClient = None
    user = None
    system = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    torch = None


@dataclass
class ModelResponse:
    """Standard response format from any model client."""
    content: str
    model_id: str
    provider: str
    timestamp: float
    
    # Usage information
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Performance metrics
    response_time: float = 0.0
    cost_estimate: Optional[float] = None
    
    # Metadata
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModelClient(ABC):
    """Abstract base class for all model clients."""
    
    def __init__(self, config: ProviderConfig, model_spec: ModelSpec):
        self.config = config
        self.model_spec = model_spec
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    async def generate_stream(self, 
                             messages: List[Dict[str, str]], 
                             **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the model."""
        pass
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Estimate the cost of a request based on token counts."""
        if self.model_spec.input_cost_per_1m and self.model_spec.output_cost_per_1m:
            input_cost = (input_tokens / 1_000_000) * self.model_spec.input_cost_per_1m
            output_cost = (output_tokens / 1_000_000) * self.model_spec.output_cost_per_1m
            return input_cost + output_cost
        return None


class OpenAIClient(BaseModelClient):
    """Client for OpenAI models including GPT-4, O3, etc."""
    
    def __init__(self, config: ProviderConfig, model_spec: ModelSpec):
        super().__init__(config, model_spec)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key or os.getenv('OPENAI_API_KEY'),
            base_url=config.api_base_url,
            organization=config.organization,
            timeout=config.read_timeout,
            max_retries=config.max_retries
        )
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        # Prepare request parameters
        params = {
            'model': self.model_spec.api_model_name or self.model_spec.model_id,
            'messages': messages,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', self.model_spec.max_tokens),
        }
        
        # Add optional parameters
        if 'top_p' in kwargs and kwargs['top_p'] is not None:
            params['top_p'] = kwargs['top_p']
        if 'frequency_penalty' in kwargs and kwargs['frequency_penalty'] is not None:
            params['frequency_penalty'] = kwargs['frequency_penalty']
        if 'presence_penalty' in kwargs and kwargs['presence_penalty'] is not None:
            params['presence_penalty'] = kwargs['presence_penalty']
        if 'stop' in kwargs and kwargs['stop']:
            params['stop'] = kwargs['stop']
        
        try:
            response = await self.client.chat.completions.create(**params)
            response_time = time.time() - start_time
            
            # Extract response details
            choice = response.choices[0]
            usage = response.usage
            
            return ModelResponse(
                content=choice.message.content,
                model_id=self.model_spec.model_id,
                provider="openai",
                timestamp=start_time,
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                response_time=response_time,
                cost_estimate=self.estimate_cost(
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0
                ),
                finish_reason=choice.finish_reason,
                metadata={'response_id': response.id}
            )
        
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        params = {
            'model': self.model_spec.api_model_name or self.model_spec.model_id,
            'messages': messages,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', self.model_spec.max_tokens),
            'stream': True
        }
        
        try:
            async for chunk in await self.client.chat.completions.create(**params):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {e}")
            raise


class AnthropicClient(BaseModelClient):
    """Client for Claude models."""
    
    def __init__(self, config: ProviderConfig, model_spec: ModelSpec):
        super().__init__(config, model_spec)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available. Install with: pip install anthropic")
        
        self.client = anthropic.AsyncAnthropic(
            api_key=config.api_key or os.getenv('ANTHROPIC_API_KEY'),
            base_url=config.api_base_url,
            timeout=config.read_timeout,
            max_retries=config.max_retries
        )
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        # Convert messages format for Anthropic
        system_message = None
        claude_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                claude_messages.append(msg)
        
        # Prepare request parameters
        params = {
            'model': self.model_spec.api_model_name or self.model_spec.model_id,
            'messages': claude_messages,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 4096)  # Anthropic requires max_tokens
        }
        
        if system_message:
            params['system'] = system_message
        
        if 'top_p' in kwargs and kwargs['top_p'] is not None:
            params['top_p'] = kwargs['top_p']
        if 'stop_sequences' in kwargs and kwargs['stop_sequences']:
            params['stop_sequences'] = kwargs['stop_sequences']
        
        try:
            response = await self.client.messages.create(**params)
            response_time = time.time() - start_time
            
            return ModelResponse(
                content=response.content[0].text,
                model_id=self.model_spec.model_id,
                provider="anthropic",
                timestamp=start_time,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                response_time=response_time,
                cost_estimate=self.estimate_cost(
                    response.usage.input_tokens,
                    response.usage.output_tokens
                ),
                finish_reason=response.stop_reason,
                metadata={'response_id': response.id}
            )
        
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
    
    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        # Similar to generate but with stream=True
        # Implementation would follow similar pattern to generate method
        yield "Streaming not implemented for Anthropic client"


class GoogleClient(BaseModelClient):
    """Client for Google Gemini models."""
    
    def __init__(self, config: ProviderConfig, model_spec: ModelSpec):
        super().__init__(config, model_spec)
        
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google AI package not available. Install with: pip install google-generativeai")
        
        genai.configure(api_key=config.api_key or os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(
            model_name=self.model_spec.api_model_name or self.model_spec.model_id
        )
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using Google Gemini API."""
        start_time = time.time()
        
        # Convert messages to Gemini format
        prompt_text = self._messages_to_prompt(messages)
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.model.generate_content(
                    prompt_text,
                    generation_config=genai.types.GenerationConfig(
                        temperature=kwargs.get('temperature', 0.7),
                        max_output_tokens=kwargs.get('max_tokens', 8192),
                        top_p=kwargs.get('top_p'),
                        top_k=kwargs.get('top_k')
                    )
                )
            )
            
            response_time = time.time() - start_time
            
            return ModelResponse(
                content=response.text,
                model_id=self.model_spec.model_id,
                provider="google",
                timestamp=start_time,
                response_time=response_time,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                metadata={'safety_ratings': response.candidates[0].safety_ratings if response.candidates else []}
            )
        
        except Exception as e:
            self.logger.error(f"Google API error: {e}")
            raise
    
    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        yield "Streaming not implemented for Google client"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Gemini prompt format."""
        prompt_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"


class GrokClient(BaseModelClient):
    """Client for XAI Grok models via file-based proxy."""

    POLL_DIR = Path("/home/jz4391/xai_proxy")
    PROCESSED_DIR = POLL_DIR / "processed"

    def __init__(self, config: ProviderConfig, model_spec: ModelSpec):
        super().__init__(config, model_spec)
        self.api_key = config.api_key or os.getenv('XAI_API_KEY')
        self.model_name = model_spec.api_model_name or model_spec.model_id

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response via xAI proxy."""
        import shutil
        start_time = time.time()

        timestamp = f"{time.time():.6f}".replace('.', '-')
        request_path = self.POLL_DIR / f"request_{timestamp}.json"
        response_path = self.POLL_DIR / f"response_{timestamp}.json"

        with open(request_path, 'w') as f:
            json.dump({
                "api_key": self.api_key,
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get('temperature', 0.7),
                "timeout": 300
            }, f)

        # Poll for response (10 min timeout)
        poll_start = time.time()
        while not response_path.exists():
            if time.time() - poll_start > 600:
                if request_path.exists():
                    request_path.unlink()
                raise TimeoutError(f"No response after 600s")
            await asyncio.sleep(0.5)

        with open(response_path) as f:
            resp = json.load(f)
        shutil.move(response_path, self.PROCESSED_DIR / response_path.name)

        if resp["error"]:
            raise Exception(resp["error"])

        return ModelResponse(
            content=resp["result"],
            model_id=self.model_spec.model_id,
            provider="xai",
            timestamp=start_time,
            response_time=time.time() - start_time,
            metadata={'model': self.model_name}
        )

        # --- OLD DIRECT XAI SDK CODE (commented out) ---
        # if not XAI_AVAILABLE:
        #     raise ImportError("XAI SDK not available. Install with: pip install xai-sdk")
        # self.client = XAIClient(api_key=config.api_key or os.getenv('XAI_API_KEY'))
        # try:
        #     chat = self.client.chat.create(
        #         model=self.model_name,
        #         temperature=kwargs.get('temperature', 0.7)
        #     )
        #     for msg in messages:
        #         if msg['role'] == 'system':
        #             chat.append(system(msg['content']))
        #         elif msg['role'] == 'user':
        #             chat.append(user(msg['content']))
        #         elif msg['role'] == 'assistant':
        #             chat.append(user(f"Assistant previously said: {msg['content']}"))
        #     response = await asyncio.get_event_loop().run_in_executor(None, chat.sample)
        #     response_time = time.time() - start_time
        #     return ModelResponse(
        #         content=response.content,
        #         model_id=self.model_spec.model_id,
        #         provider="xai",
        #         timestamp=start_time,
        #         response_time=response_time,
        #         metadata={'model': self.model_name}
        #     )
        # except Exception as e:
        #     self.logger.error(f"XAI Grok API error: {e}")
        #     raise
    
    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        # XAI SDK might not support streaming yet
        yield "Streaming not implemented for Grok client"


class OpenRouterClient(BaseModelClient):
    """Client for models accessed via OpenRouter API."""
    
    def __init__(self, config: ProviderConfig, model_spec: ModelSpec):
        super().__init__(config, model_spec)
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests package not available. Install with: pip install requests")
        
        self.api_key = config.api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = config.api_base_url or "https://openrouter.ai/api/v1"
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using OpenRouter API."""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_spec.api_model_name or self.model_spec.model_id,
            "messages": messages,
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', self.model_spec.max_tokens),
        }
        
        try:
            async with asyncio.create_task(asyncio.to_thread(
                requests.post,
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.config.read_timeout
            )) as response:
                response.raise_for_status()
                result = response.json()
                
                response_time = time.time() - start_time
                choice = result['choices'][0]
                usage = result.get('usage', {})
                
                return ModelResponse(
                    content=choice['message']['content'],
                    model_id=self.model_spec.model_id,
                    provider="openrouter",
                    timestamp=start_time,
                    input_tokens=usage.get('prompt_tokens'),
                    output_tokens=usage.get('completion_tokens'),
                    total_tokens=usage.get('total_tokens'),
                    response_time=response_time,
                    finish_reason=choice.get('finish_reason'),
                    metadata={'openrouter_id': result.get('id')}
                )
        
        except Exception as e:
            self.logger.error(f"OpenRouter API error: {e}")
            raise
    
    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        yield "Streaming not implemented for OpenRouter client"


class PrincetonClusterClient(BaseModelClient):
    """Client for models running on Princeton cluster."""
    
    def __init__(self, config: ProviderConfig, model_spec: ModelSpec):
        super().__init__(config, model_spec)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package not available. Install with: pip install transformers torch")
        
        self.model_path = model_spec.local_path
        if not self.model_path:
            raise ValueError("local_path must be specified for Princeton cluster models")
        
        # Initialize model and tokenizer (this would be done lazily in practice)
        self.tokenizer = None
        self.model = None
        self._initialized = False
    
    async def _initialize_model(self):
        """Initialize the model and tokenizer lazily."""
        if self._initialized:
            return
        
        try:
            # Load in a separate thread to avoid blocking
            self.tokenizer, self.model = await asyncio.get_event_loop().run_in_executor(
                None,
                self._load_model
            )
            self._initialized = True
            self.logger.info(f"Initialized Princeton cluster model from {self.model_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize cluster model: {e}")
            raise
    
    def _load_model(self):
        """Load model and tokenizer (runs in executor)."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Set pad token if not already set (for Qwen models, pad_token_id should be different from eos_token_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        return tokenizer, model
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using local model."""
        await self._initialize_model()
        start_time = time.time()
        
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        
        try:
            # Generate response in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._generate_text,
                prompt,
                kwargs
            )
            
            response_time = time.time() - start_time
            
            return ModelResponse(
                content=result['generated_text'],
                model_id=self.model_spec.model_id,
                provider="princeton_cluster",
                timestamp=start_time,
                response_time=response_time,
                input_tokens=result.get('input_tokens'),
                output_tokens=result.get('output_tokens'),
                metadata={'model_path': self.model_path}
            )
        
        except Exception as e:
            self.logger.error(f"Princeton cluster model error: {e}")
            raise
    
    async def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        yield "Streaming not implemented for Princeton cluster client"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to model-specific prompt format."""
        # Try to use tokenizer's chat template if available (for Qwen models)
        if self.tokenizer is not None and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # Convert messages format for chat template
                chat_messages = []
                for msg in messages:
                    if msg['role'] == 'system':
                        chat_messages.append({"role": "system", "content": msg['content']})
                    elif msg['role'] == 'user':
                        chat_messages.append({"role": "user", "content": msg['content']})
                    elif msg['role'] == 'assistant':
                        chat_messages.append({"role": "assistant", "content": msg['content']})
                
                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                self.logger.warning(f"Failed to use chat template, falling back to generic format: {e}")
        
        # Fallback to generic format
        prompt_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                prompt_parts.append(f"<|system|>\n{content}")
            elif role == 'user':
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == 'assistant':
                prompt_parts.append(f"<|assistant|>\n{content}")
        
        return "\n".join(prompt_parts) + "\n<|assistant|>\n"
    
    def _generate_text(self, prompt: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text using the local model (runs in executor)."""
        # Use tokenizer properly to get attention mask
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        input_length = input_ids.shape[1]
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            generate_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": kwargs.get('max_tokens', 512),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 50),
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
                
            outputs = self.model.generate(**generate_kwargs)
        
        # Extract only the newly generated tokens
        new_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return {
            'generated_text': generated_text,
            'input_tokens': input_length,
            'output_tokens': len(new_tokens)
        }


class ModelClientFactory:
    """Factory for creating appropriate model clients."""
    
    @staticmethod
    def create_client(config: ProviderConfig, model_spec: ModelSpec) -> BaseModelClient:
        """Create the appropriate client for the given model specification."""
        
        if model_spec.provider == ModelProvider.OPENAI:
            return OpenAIClient(config, model_spec)
        
        elif model_spec.provider == ModelProvider.ANTHROPIC:
            return AnthropicClient(config, model_spec)
        
        elif model_spec.provider == ModelProvider.GOOGLE:
            return GoogleClient(config, model_spec)
        
        elif model_spec.provider == ModelProvider.XAI:
            return GrokClient(config, model_spec)
        
        elif model_spec.provider == ModelProvider.OPENROUTER:
            return OpenRouterClient(config, model_spec)
        
        elif model_spec.provider == ModelProvider.PRINCETON_CLUSTER:
            return PrincetonClusterClient(config, model_spec)
        
        else:
            raise ValueError(f"Unsupported provider: {model_spec.provider}")


class UnifiedModelManager:
    """High-level manager for multiple model clients."""
    
    def __init__(self):
        self.clients: Dict[str, BaseModelClient] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_agent(self, agent_config: AgentModelConfig, provider_config: ProviderConfig):
        """Register an agent with its model configuration."""
        client = ModelClientFactory.create_client(provider_config, agent_config.model_spec)
        self.clients[agent_config.agent_id] = client
        self.logger.info(f"Registered agent {agent_config.agent_id} with model {agent_config.model_spec.model_id}")
    
    async def generate(self, agent_id: str, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response for a specific agent."""
        if agent_id not in self.clients:
            raise ValueError(f"Agent {agent_id} not registered")
        
        client = self.clients[agent_id]
        return await client.generate(messages, **kwargs)
    
    async def generate_batch(self, 
                            requests: List[tuple[str, List[Dict[str, str]], Dict[str, Any]]]) -> List[ModelResponse]:
        """Generate responses for multiple agents in parallel."""
        tasks = []
        for agent_id, messages, kwargs in requests:
            task = self.generate(agent_id, messages, **kwargs)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a registered agent."""
        if agent_id not in self.clients:
            raise ValueError(f"Agent {agent_id} not registered")
        
        client = self.clients[agent_id]
        return {
            'agent_id': agent_id,
            'model_id': client.model_spec.model_id,
            'provider': client.model_spec.provider.value,
            'display_name': client.model_spec.display_name,
            'capabilities': {
                'context_window': client.model_spec.context_window,
                'supports_system_prompt': client.model_spec.supports_system_prompt,
                'supports_vision': client.model_spec.supports_vision,
                'reasoning_capability': client.model_spec.reasoning_capability
            }
        }
    
    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self.clients.keys())