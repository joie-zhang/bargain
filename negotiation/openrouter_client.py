"""
OpenRouter client for accessing various LLM models including Gemma.

On Della, the default transport is the shared file-based proxy monitor first,
with direct HTTPS as automatic fallback when the proxy path is unavailable.
"""

import os
import json
import asyncio
import aiohttp
import time
import uuid
import random
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import sys
import shutil
from pathlib import Path


from .llm_agents import BaseLLMAgent, LLMConfig, ModelType, NonRetryableLLMError
from .provider_key_rotation import (
    ProviderKey,
    ProviderKeyPool,
    call_with_key_rotation,
)


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
    timeout: float = 300.0
    max_retries: int = 12
    retry_delay: float = 2.0
    transport: str = "auto"  # direct | proxy | auto (proxy-first)
    proxy_poll_dir: str = "/home/jz4391/openrouter_proxy"
    proxy_poll_interval: float = 0.1
    proxy_timeout: float = 6000.0
    proxy_probe_timeout: float = 30.0


class ProxyMonitorUnavailableError(RuntimeError):
    """Raised when the shared OpenRouter proxy monitor appears unavailable."""


class OpenRouterAgent(BaseLLMAgent):
    """Agent that uses OpenRouter API for various models."""
    
    def __init__(self, 
                 agent_id: str,
                 llm_config: LLMConfig,
                 api_key: Optional[str],
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
        
        self.key_pool = ProviderKeyPool("openrouter", fallback_key=api_key)
        initial_key = self.key_pool.current()
        if not initial_key.value.startswith("sk-or-v1-"):
            self.logger.warning(
                f"OpenRouter API key {initial_key.label} for agent {agent_id} doesn't start with 'sk-or-v1-'. "
                f"This might indicate an invalid key format."
            )
        
        requested_transport = os.getenv("OPENROUTER_TRANSPORT", "auto").strip().lower()
        if requested_transport not in {"direct", "proxy", "auto"}:
            self.logger.warning(
                "Invalid OPENROUTER_TRANSPORT=%r; defaulting to 'auto'",
                requested_transport,
            )
            requested_transport = "auto"

        self.running_in_slurm = bool(os.getenv("SLURM_JOB_ID"))
        if requested_transport == "auto":
            resolved_transport = "proxy" if self.running_in_slurm else "direct"
        else:
            resolved_transport = requested_transport
        self.requested_transport = requested_transport

        proxy_poll_dir = os.getenv("OPENROUTER_PROXY_POLL_DIR", "/home/jz4391/openrouter_proxy")

        poll_env = os.getenv("OPENROUTER_PROXY_CLIENT_POLL_INTERVAL", "0.1")
        try:
            proxy_poll_interval = max(0.01, float(poll_env))
        except ValueError:
            self.logger.warning(
                "Invalid OPENROUTER_PROXY_CLIENT_POLL_INTERVAL=%r; using 0.1",
                poll_env,
            )
            proxy_poll_interval = 0.1

        timeout_env = os.getenv("OPENROUTER_PROXY_CLIENT_TIMEOUT", "6000")
        try:
            proxy_timeout = max(1.0, float(timeout_env))
        except ValueError:
            self.logger.warning(
                "Invalid OPENROUTER_PROXY_CLIENT_TIMEOUT=%r; using 6000",
                timeout_env,
            )
            proxy_timeout = 6000.0

        probe_timeout_env = os.getenv("OPENROUTER_PROXY_PROBE_TIMEOUT", "30")
        try:
            proxy_probe_timeout = max(1.0, float(probe_timeout_env))
        except ValueError:
            self.logger.warning(
                "Invalid OPENROUTER_PROXY_PROBE_TIMEOUT=%r; using 30",
                probe_timeout_env,
            )
            proxy_probe_timeout = 30.0

        api_timeout_env = os.getenv("OPENROUTER_API_TIMEOUT", "300")
        try:
            api_timeout = max(1.0, float(api_timeout_env))
        except ValueError:
            self.logger.warning(
                "Invalid OPENROUTER_API_TIMEOUT=%r; using 300",
                api_timeout_env,
            )
            api_timeout = 300.0

        self.openrouter_config = OpenRouterConfig(
            api_key=initial_key.value.strip(),
            timeout=api_timeout,
            transport=resolved_transport,
            proxy_poll_dir=proxy_poll_dir,
            proxy_poll_interval=proxy_poll_interval,
            proxy_timeout=proxy_timeout,
            proxy_probe_timeout=proxy_probe_timeout,
        )
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize message history (required by BaseLLMAgent)
        self.message_history = []
        self._last_usage = None  # Store last usage data for access
        
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

        # Keep parity with the other agent classes so downstream analyzers can
        # recover the configured model name from the live agent object.
        self.model_name = self.model_id
        
        self.logger = logging.getLogger(f"OpenRouterAgent-{agent_id}")
        self.logger.debug(
            "Resolved OpenRouter transport requested=%s resolved=%s slurm=%s",
            self.requested_transport,
            self.openrouter_config.transport,
            self.running_in_slurm,
        )
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None:
            connector = aiohttp.TCPConnector(force_close=False)
            self.session = aiohttp.ClientSession(
                connector=connector,
                json_serialize=lambda x: json.dumps(x, ensure_ascii=False)
            )

    def _proxy_monitor_hint(self) -> str:
        host = os.getenv("OPENROUTER_PROXY_HOST", "della-vis1.princeton.edu")
        return (
            f"Ensure negotiation/openrouter_proxy_monitor.py is running on {host} "
            f"and watching {self.openrouter_config.proxy_poll_dir}."
        )

    def _validate_request_result(self, result: Any, error: Any) -> None:
        if error:
            if self._is_non_retryable_error_message(str(error)):
                raise NonRetryableLLMError(str(error))
            raise Exception(error)
        if result is None:
            raise Exception("OpenRouter API returned None result")
        if not isinstance(result, str) or len(result.strip()) == 0:
            raise Exception("OpenRouter API returned empty response content")

    @staticmethod
    def _is_non_retryable_http_status(status: int) -> bool:
        return 400 <= status < 500 and status not in {408, 409, 429}

    @classmethod
    def _is_non_retryable_error_message(cls, message: str) -> bool:
        lowered = message.lower()
        match = re.search(r"http\s+(\d{3})", lowered)
        if match:
            status = int(match.group(1))
            if cls._is_non_retryable_http_status(status):
                return True
        return any(
            marker in lowered
            for marker in (
                "invalid model identifier",
                "no endpoints found",
                "not_found_error",
                "provider returned error",
            )
        )
    
    def _extract_content_from_openrouter_response(self, data: Dict[str, Any]) -> str:
        """Extract message content from OpenRouter response payload."""
        if "error" in data:
            error_msg = data.get("error", {})
            if isinstance(error_msg, dict):
                raise Exception(f"API Error: {error_msg.get('message', error_msg)}")
            raise Exception(f"API Error: {error_msg}")

        choices = data.get("choices") or []
        if not choices:
            raise Exception(f"No choices in response: {json.dumps(data)[:500]}")

        message = choices[0].get("message", {})
        content = message.get("content")

        if (content is None or content == "") and isinstance(message.get("reasoning_content"), str):
            reasoning = message.get("reasoning_content", "").strip()
            if reasoning:
                content = reasoning

        if content is None or content == "":
            reasoning_details = message.get("reasoning_details")
            if isinstance(reasoning_details, list):
                summary_parts = []
                for entry in reasoning_details:
                    if not isinstance(entry, dict):
                        continue
                    summary = entry.get("summary")
                    if isinstance(summary, str) and summary.strip():
                        summary_parts.append(summary.strip())
                        continue
                    if isinstance(summary, list):
                        for chunk in summary:
                            if not isinstance(chunk, dict):
                                continue
                            text = chunk.get("text")
                            if isinstance(text, str) and text.strip():
                                summary_parts.append(text.strip())
                if summary_parts:
                    content = "\n".join(summary_parts)

        if not isinstance(content, str) or not content.strip():
            finish_reason = choices[0].get("finish_reason")
            raise Exception(
                f"OpenRouter returned empty content (finish_reason={finish_reason})"
            )

        return content

    async def _send_request_direct(self, url: str, headers: dict, payload: dict, timeout: float) -> tuple:
        """Query OpenRouter directly over HTTPS."""
        await self._ensure_session()
        assert self.session is not None

        async with self.session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                error_message = f"HTTP {response.status}: {error_text[:500]}"
                if self._is_non_retryable_http_status(response.status):
                    raise NonRetryableLLMError(error_message)
                raise Exception(error_message)

            data = await response.json(encoding="utf-8")
            result = self._extract_content_from_openrouter_response(data)
            usage = data.get("usage", {}) or {}
            return result, None, usage

    async def _send_request_via_proxy(self, url: str, headers: dict, payload: dict, timeout: float) -> tuple:
        """Route request through the shared file-based proxy queue."""
        poll_dir = Path(self.openrouter_config.proxy_poll_dir)
        processed_dir = poll_dir / "processed"
        poll_interval = self.openrouter_config.proxy_poll_interval
        client_timeout = self.openrouter_config.proxy_timeout
        proxy_hint = self._proxy_monitor_hint()

        poll_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        effective_timeout = client_timeout

        # Use nanoseconds + uuid to avoid filename collisions across many workers.
        timestamp = f"{time.time_ns()}_{uuid.uuid4().hex}"
        request_path = poll_dir / f"request_{timestamp}.json"
        response_path = poll_dir / f"response_{timestamp}.json"

        request_data = {
            "url": url,
            "headers": headers,
            "payload": payload,
            "timeout": timeout,
        }
        with open(request_path, "w") as f:
            json.dump(request_data, f)
        self.logger.debug(f"[client] Wrote {request_path.name}")

        start = time.time()
        response_data = None
        while response_data is None:
            if time.time() - start > effective_timeout:
                if request_path.exists():
                    request_path.unlink()
                raise ProxyMonitorUnavailableError(
                    f"No proxy response after {effective_timeout}s for request {timestamp}. {proxy_hint}"
                )
            if not response_path.exists():
                await asyncio.sleep(poll_interval)
                continue
            try:
                with open(response_path, "r") as f:
                    response_data = json.load(f)
            except json.JSONDecodeError:
                # The proxy monitor writes response files from another process.
                # If we observe the file before the JSON payload is fully visible
                # on the shared filesystem, keep polling instead of failing.
                await asyncio.sleep(poll_interval)

        result, error = response_data["result"], response_data["error"]
        usage = response_data.get("usage", {}) or {}
        self.logger.debug(
            f"[client] Got response for {timestamp}: {'error' if error else 'success'}"
        )

        processed_dir.mkdir(exist_ok=True)
        shutil.move(response_path, processed_dir / response_path.name)
        return result, error, usage

    async def _make_request(self, messages: List[Dict[str, str]]) -> tuple:
        """Make a request to OpenRouter API. Returns (content, usage)."""
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.llm_config.temperature,
            "max_tokens": min(self.llm_config.max_tokens, 4000),
        }
        if self.llm_config.custom_parameters:
            payload.update(self.llm_config.custom_parameters)

        url = f"{self.openrouter_config.base_url}/chat/completions"

        async def request_with_key(key: ProviderKey) -> tuple:
            self.openrouter_config.api_key = key.value.strip()
            headers = {
                "Authorization": f"Bearer {self.openrouter_config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.openrouter_config.site_url or "",
                "X-Title": self.openrouter_config.site_name or ""
            }

            if self.openrouter_config.transport == "proxy":
                result, error, usage = await self._send_request_via_proxy(
                    url, headers, payload, self.openrouter_config.timeout
                )
                self._validate_request_result(result, error)
            elif self.openrouter_config.transport == "direct":
                result, error, usage = await self._send_request_direct(
                    url, headers, payload, self.openrouter_config.timeout
                )
                self._validate_request_result(result, error)
            else:
                raise RuntimeError(
                    f"Unsupported OpenRouter transport: {self.openrouter_config.transport}"
                )
            return result, usage

        return await call_with_key_rotation(
            provider="openrouter",
            model=self.model_id,
            key_pool=self.key_pool,
            request_coro_factory=request_with_key,
            logger=self.logger,
        )
    
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
        
        # Make request (returns tuple of content, usage)
        response_content, usage = await self._make_request(messages)
        self._last_usage = usage  # Store for potential later access

        # Update history
        self.message_history.append({
            "type": message_type,
            "prompt": prompt,
            "response": response_content
        })

        return response_content
    
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
        response_content, usage = await self._make_request(messages)
        response_time = time.time() - start_time

        # Extract token counts from usage
        tokens_used = usage.get("total_tokens") if usage else None
        reasoning_tokens = usage.get("reasoning_tokens") if usage else None

        return AgentResponse(
            content=response_content,
            model_used=self.model_id,
            response_time=response_time,
            tokens_used=tokens_used,
            cost_estimate=None,
            metadata={
                "usage": usage,
                "reasoning_tokens": reasoning_tokens
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model. Required by BaseLLMAgent."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_id,
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
