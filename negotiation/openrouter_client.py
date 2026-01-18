"""
OpenRouter client for accessing various LLM models including Gemma.

OpenRouter provides a unified API for multiple model providers.
"""

import os
import json
import asyncio
import aiohttp
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import sys
import shutil
from pathlib import Path


from .llm_agents import BaseLLMAgent, LLMConfig, ModelType


# OpenRouter model mappings
OPENROUTER_MODELS = {
    # Gemma models
    "gemma-2b": "google/gemma-2-9b-it",  # Using 9B as smallest available
    "gemma-7b": "google/gemma-2-9b-it",  # Using 9B instead of 7B
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2-27b": "google/gemma-2-27b-it",
    
    # Other models available on OpenRouter
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "gpt-4": "openai/gpt-4",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
    "llama-3-70b": "meta-llama/llama-3-70b-instruct",
    "llama-3-8b": "meta-llama/llama-3-8b-instruct"
}


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    site_url: Optional[str] = "https://github.com/negotiation-research"
    site_name: Optional[str] = "Negotiation Research"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class OpenRouterAgent(BaseLLMAgent):
    """Agent that uses OpenRouter API for various models."""
    
    def __init__(self, 
                 agent_id: str,
                 llm_config: LLMConfig,
                 api_key: str,
                 model_id: Optional[str] = None):
        """
        Initialize OpenRouter agent.
        
        Args:
            agent_id: Unique identifier for the agent
            llm_config: LLM configuration
            api_key: OpenRouter API key
            model_id: Optional direct model ID (e.g., "google/gemma-7b-it")
        """
        super().__init__(agent_id, llm_config)
        self.llm_config = llm_config
        
        self.openrouter_config = OpenRouterConfig(api_key=api_key)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize message history (required by BaseLLMAgent)
        self.message_history = []
        
        # Determine model ID
        if model_id:
            self.model_id = model_id
        elif llm_config.model_type.value in OPENROUTER_MODELS:
            self.model_id = OPENROUTER_MODELS[llm_config.model_type.value]
        else:
            # Default to a Gemma model based on the type
            if "gemma" in llm_config.model_type.value.lower():
                if "27b" in llm_config.model_type.value:
                    self.model_id = "google/gemma-2-27b-it"
                else:
                    # Use 9B for smaller models since 2B and 7B aren't available
                    self.model_id = "google/gemma-2-9b-it"
            else:
                raise ValueError(f"Unknown model type for OpenRouter: {llm_config.model_type}")
        
        self.logger = logging.getLogger(f"OpenRouterAgent-{agent_id}")
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None:
            connector = aiohttp.TCPConnector(force_close=True)
            self.session = aiohttp.ClientSession(
                connector=connector,
                json_serialize=lambda x: json.dumps(x, ensure_ascii=False)
            )
    

    async def send_request(self, url: str, headers: dict, payload: dict, timeout: float) -> tuple[str | None, str | None]:
        """
        Write request file, poll for response, return (result, error).
        Raises TimeoutError if no response within 10 minutes.
        """
        POLL_DIR = Path("/home/jz4391/openrouter_proxy")
        PROCESSED_DIR = POLL_DIR / "processed"
        POLL_INTERVAL = 0.5
        CLIENT_TIMEOUT = 6000  # 10 minutes

        timestamp = f"{time.time():.6f}".replace('.', '-')
        request_path = POLL_DIR / f"request_{timestamp}.json"
        response_path = POLL_DIR / f"response_{timestamp}.json"

        request_data = {
            "url": url,
            "headers": headers,
            "payload": payload,
            "timeout": timeout,
        }
        with open(request_path, 'w') as f:
            json.dump(request_data, f)
        print(f"[client] Wrote {request_path.name}")

        start = time.time()
        while not response_path.exists():
            if time.time() - start > CLIENT_TIMEOUT:
                # Clean up orphaned request if still there
                if request_path.exists():
                    request_path.unlink()
                raise TimeoutError(f"No response after {CLIENT_TIMEOUT}s for {timestamp}")
            await asyncio.sleep(POLL_INTERVAL)

        with open(response_path, 'r') as f:
            response_data = json.load(f)

        result, error = response_data["result"], response_data["error"]
        print(f"[client] Got response for {timestamp}: {'error' if error else 'success'}")

        shutil.move(response_path, PROCESSED_DIR / response_path.name)

        return result, error
    
    async def _make_request(self, messages: List[Dict[str, str]]) -> str:
        """Make a request to OpenRouter API."""
        await self._ensure_session()
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.openrouter_config.site_url or "",
            "X-Title": self.openrouter_config.site_name or ""
        }
        
        # Prepare request payload
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.llm_config.temperature,
            "max_tokens": min(self.llm_config.max_tokens, 4000),  # Cap at 4000 for safety
        }
        
        # Add any custom parameters
        if self.llm_config.custom_parameters:
            payload.update(self.llm_config.custom_parameters)
        
        url = f"{self.openrouter_config.base_url}/chat/completions"
        
        for attempt in range(self.openrouter_config.max_retries):
            try:
                
                result, error = await self.send_request(url, headers, payload, self.openrouter_config.timeout)
                if error:
                    raise Exception(error)
                return result
                # async with self.session.post(
                #     url,
                #     headers=headers,
                #     json=payload,
                #     timeout=aiohttp.ClientTimeout(total=self.openrouter_config.timeout)
                # ) as response:
                #     if response.status == 200:
                #         # Ensure proper UTF-8 decoding
                #         data = await response.json(encoding='utf-8')
                #         content = data["choices"][0]["message"]["content"]
                #         # Ensure content is properly encoded
                #         if isinstance(content, bytes):
                #             content = content.decode('utf-8')
                #         return content
                #     else:
                #         error_text = await response.text()
                #         self.logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                        
                #         if response.status == 429:  # Rate limit
                #             await asyncio.sleep(self.openrouter_config.retry_delay * (attempt + 1))
                #             continue
                #         elif response.status >= 500:  # Server error
                #             await asyncio.sleep(self.openrouter_config.retry_delay)
                #             continue
                #         else:
                #             raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
                            
            except asyncio.TimeoutError:
                self.logger.warning(f"Request timeout (attempt {attempt + 1}/{self.openrouter_config.max_retries})")
                if attempt < self.openrouter_config.max_retries - 1:
                    await asyncio.sleep(self.openrouter_config.retry_delay)
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Request failed: {e}")
                if attempt < self.openrouter_config.max_retries - 1:
                    await asyncio.sleep(self.openrouter_config.retry_delay)
                else:
                    raise
        
        raise Exception(f"Failed after {self.openrouter_config.max_retries} attempts")
    
    async def discuss(self, context: Any, prompt: str) -> str:
        """Participate in discussion phase."""
        return await self.generate_message(context, "discussion", prompt)
    
    async def propose(self, context: Any, prompt: str) -> str:
        """Make a proposal."""
        return await self.generate_message(context, "proposal", prompt)
    
    async def vote(self, context: Any, prompt: str) -> str:
        """Vote on proposals."""
        return await self.generate_message(context, "vote", prompt)
    
    async def reflect(self, context: Any, prompt: str) -> str:
        """Reflect on the negotiation."""
        return await self.generate_message(context, "reflection", prompt)
    
    async def generate_message(self, 
                              context: Any,
                              message_type: str,
                              prompt: str = None,
                              previous_messages: List[Dict[str, Any]] = None) -> str:
        """Generate a message using OpenRouter API."""
        
        # Build conversation history
        messages = []
        
        # Add system prompt if available
        if self.llm_config.system_prompt:
            messages.append({
                "role": "system",
                "content": str(self.llm_config.system_prompt)
            })
        
        # Add previous messages if available
        if previous_messages:
            for msg in previous_messages[-10:]:  # Keep last 10 messages for context
                role = "assistant" if msg.get("from") == self.agent_id else "user"
                messages.append({
                    "role": role,
                    "content": str(msg.get("content", ""))
                })
        
        # Add current context
        if prompt:
            # Use provided prompt directly
            messages.append({
                "role": "user",
                "content": str(prompt)
            })
        else:
            # Build prompt from context
            prompt = self._build_prompt(context, message_type)
            messages.append({
                "role": "user",
                "content": str(prompt)
            })
        
        # Make request
        response = await self._make_request(messages)
        
        # Update history
        self.message_history.append({
            "type": message_type,
            "prompt": prompt,
            "response": response
        })
        
        return response
    
    def _build_prompt(self, context: Any, message_type: str) -> str:
        """Build prompt based on message type and context."""
        
        if message_type == "discussion":
            return self._build_discussion_prompt(context)
        elif message_type == "proposal":
            return self._build_proposal_prompt(context)
        elif message_type == "vote":
            return self._build_vote_prompt(context)
        elif message_type == "reflection":
            return self._build_reflection_prompt(context)
        else:
            return f"Generate a {message_type} message based on the context."
    
    def _build_discussion_prompt(self, context: Any) -> str:
        """Build discussion phase prompt."""
        prompt = f"""You are agent {self.agent_id} in a negotiation over {context.m_items} items.

Items to allocate: {', '.join([f"{i}:{context.item_names[i]}" for i in range(context.m_items)])}

Current round: {context.current_round}/{context.t_rounds}
Discount factor: {context.gamma_discount} (payoffs decrease each round)

Your task: Discuss your preferences and explore potential allocations with other agents.
- Share information strategically
- Try to understand others' preferences
- Build towards a consensus

Generate a discussion message (2-3 sentences):"""
        
        return prompt
    
    def _build_proposal_prompt(self, context: Any) -> str:
        """Build proposal phase prompt."""
        prompt = f"""You are agent {self.agent_id} proposing an allocation.

Items: {', '.join([f"{i}:{context.item_names[i]}" for i in range(context.m_items)])}
Agents: {', '.join(context.agent_ids)}

Create a proposal in this EXACT format:
I propose this allocation: {{'agent_id1': [item_indices], 'agent_id2': [item_indices], ...}} - [Your reasoning here]

Example:
I propose this allocation: {{'agent_0': [0, 2], 'agent_1': [1], 'agent_2': [3, 4]}} - This gives each agent valuable items.

Your proposal:"""
        
        return prompt
    
    def _build_vote_prompt(self, context: Any) -> str:
        """Build voting phase prompt."""
        prompt = f"""You are agent {self.agent_id} voting on proposals.

Review the proposals and vote 'yes' for acceptable ones, 'no' for others.

Proposals:
{context.proposals_text}

For EACH proposal, respond in this format:
Proposal 1: [yes/no] - [brief reason]
Proposal 2: [yes/no] - [brief reason]
etc.

Your votes:"""
        
        return prompt
    
    def _build_reflection_prompt(self, context: Any) -> str:
        """Build reflection phase prompt."""
        prompt = f"""Reflect on the negotiation so far.

What strategies are working? What should you try differently?
Be concise (1-2 sentences).

Your reflection:"""
        
        return prompt
    
    async def _call_llm_api(self, messages: List[Dict[str, str]], 
                           **kwargs) -> 'AgentResponse':
        """Call the underlying LLM API. Required by BaseLLMAgent."""
        from .llm_agents import AgentResponse
        
        start_time = time.time()
        response_content = await self._make_request(messages)
        response_time = time.time() - start_time
        
        return AgentResponse(
            content=response_content,
            model_used=self.model_id,
            response_time=response_time,
            tokens_used=None,  # OpenRouter doesn't provide token counts in basic API
            cost_estimate=None
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model. Required by BaseLLMAgent."""
        return {
            "model_id": self.model_id,
            "provider": "openrouter",
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens
        }
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.session:
            try:
                asyncio.create_task(self.close())
            except:
                pass