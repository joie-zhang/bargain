"""
Multi-LLM agent system for negotiation environment.

This module implements LLM-powered agents that can participate in negotiations,
with support for different models (O3, Claude Haiku, GPT-4, etc.) and strategic
behavior analysis.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import json
import time
import random
from pathlib import Path
import logging

# API client imports (will be conditionally imported)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ModelProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"  # OpenRouter for multiple models
    TEST = "test"  # For testing with real minimal models


class ModelType(Enum):
    """Specific model types."""
    # Anthropic models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    
    # OpenAI models
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    O3_MINI = "o3-mini"
    O3 = "o3"
    
    # Gemma models (via OpenRouter)
    GEMMA_2B = "gemma-2b"
    GEMMA_7B = "gemma-7b"
    GEMMA_2_9B = "gemma-2-9b"
    GEMMA_2_27B = "gemma-2-27b"
    
    # Test models (using simulated agents for testing)
    TEST_STRONG = "test-strong-model"  # Simulated strong agent
    TEST_WEAK = "test-weak-model"  # Simulated weak agent


@dataclass
class LLMConfig:
    """Configuration for an LLM agent."""
    model_type: ModelType
    temperature: float = 0.7
    max_tokens: int = 999999  # Effectively unlimited - will be ignored in API calls
    timeout: float = 30.0
    
    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 10000
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Model-specific parameters
    system_prompt: Optional[str] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")


@dataclass
class AgentResponse:
    """Response from an LLM agent."""
    content: str
    model_used: str
    response_time: float
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class NegotiationContext:
    """Context information for LLM agents during negotiation."""
    # Environment state
    current_round: int
    max_rounds: int
    items: List[Dict[str, Any]]
    agents: List[str]
    
    # Agent-specific information
    agent_id: str
    preferences: Any  # Agent's preferences (vector or matrix)
    known_preferences: Optional[Dict[str, Any]] = None  # Common knowledge preferences
    
    # Round history
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    previous_proposals: List[Dict[str, Any]] = field(default_factory=list)
    previous_votes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Current state
    current_proposals: List[Dict[str, Any]] = field(default_factory=list)
    turn_type: str = "discussion"  # discussion, proposal, voting
    
    # Strategic context
    strategic_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "items": self.items,
            "agents": self.agents,
            "agent_id": self.agent_id,
            "preferences": self.preferences,
            "known_preferences": self.known_preferences,
            "conversation_history": self.conversation_history,
            "previous_proposals": self.previous_proposals,
            "previous_votes": self.previous_votes,
            "current_proposals": self.current_proposals,
            "turn_type": self.turn_type,
            "strategic_notes": self.strategic_notes
        }


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        self.request_times: List[float] = []
        self.token_usage: List[Tuple[float, int]] = []  # (timestamp, tokens)
        
    async def wait_if_needed(self, estimated_tokens: int = 100) -> None:
        """Wait if rate limits would be exceeded."""
        now = time.time()
        
        # Clean old entries (older than 1 minute)
        cutoff = now - 60
        self.request_times = [t for t in self.request_times if t > cutoff]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff]
        
        # Check request rate
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Check token rate
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            # Find when we can make the request
            for timestamp, tokens in sorted(self.token_usage):
                if current_tokens + estimated_tokens <= self.tokens_per_minute:
                    break
                current_tokens -= tokens
                wait_time = 60 - (now - timestamp)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    break
        
        # Record this request
        self.request_times.append(now)
        self.token_usage.append((now, estimated_tokens))


class BaseLLMAgent(ABC):
    """Abstract base class for LLM-powered negotiation agents."""
    
    def __init__(self, agent_id: str, config: LLMConfig):
        self.agent_id = agent_id
        self.config = config
        self.rate_limiter = RateLimiter(
            config.requests_per_minute,
            config.tokens_per_minute
        )
        
        # Conversation memory
        self.conversation_memory: List[Dict[str, Any]] = []
        self.strategic_memory: List[str] = []
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.response_times: List[float] = []
        
        self.logger = logging.getLogger(f"LLMAgent.{agent_id}")
    
    @abstractmethod
    async def _call_llm_api(self, messages: List[Dict[str, str]], 
                           **kwargs) -> AgentResponse:
        """Call the underlying LLM API. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model. Must be implemented by subclasses."""
        pass
    
    def _build_system_prompt(self, context: NegotiationContext) -> str:
        """Build system prompt for the agent."""
        base_prompt = f"""You are {context.agent_id}, a negotiating agent in a multi-agent negotiation.

NEGOTIATION SETUP:
- Round {context.current_round} of {context.max_rounds}
- Items to negotiate: {len(context.items)} items
- Participants: {', '.join(context.agents)}

YOUR PREFERENCES:
{self._format_preferences(context.preferences, context.items)}

CURRENT SITUATION:
- Turn type: {context.turn_type}
- Items on the table: {[item['name'] for item in context.items]}

INSTRUCTIONS:
- Be strategic and try to maximize your utility
- Consider both cooperation and competition as tactics
- Pay attention to other agents' behavior patterns
- Remember previous rounds and adapt your strategy
- Be concise but persuasive in your communications

IMPORTANT: Your goal is to get the items you value most highly. Act in your own best interest while engaging authentically with other agents."""

        if self.config.system_prompt:
            base_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{self.config.system_prompt}"
        
        return base_prompt
    
    def _format_preferences(self, preferences: Any, items: List[Dict[str, Any]]) -> str:
        """Format preferences for display in prompt."""
        if isinstance(preferences, list):
            # Vector preferences
            pref_str = "Your item valuations:\n"
            for i, (item, value) in enumerate(zip(items, preferences)):
                pref_str += f"- {item['name']}: {value:.1f}/10\n"
            return pref_str
        elif isinstance(preferences, list) and isinstance(preferences[0], list):
            # Matrix preferences  
            pref_str = "Your preference matrix (how much you value each agent getting each item):\n"
            agent_names = [f"agent_{i}" for i in range(len(preferences[0]))]
            for i, (item, row) in enumerate(zip(items, preferences)):
                pref_str += f"- {item['name']}: "
                for j, value in enumerate(row):
                    pref_str += f"{agent_names[j]}={value:.1f} "
                pref_str += "\n"
            return pref_str
        else:
            return f"Your preferences: {preferences}"
    
    async def think_strategy(self, prompt: str, context: NegotiationContext) -> Dict[str, Any]:
        """
        Generate private strategic thinking for proposal planning.
        
        This method allows agents to privately analyze the situation and plan
        their strategy before making public proposals.
        """
        # Wait for rate limits
        await self.rate_limiter.wait_if_needed()
        
        # Build messages for thinking prompt
        thinking_messages = self._build_thinking_messages(context, prompt)
        
        # Call LLM for strategic thinking
        for attempt in range(self.config.max_retries):
            try:
                response = await self._call_llm_api(thinking_messages)
                
                # Update tracking
                self.total_requests += 1
                if response.tokens_used:
                    self.total_tokens += response.tokens_used
                if response.cost_estimate:
                    self.total_cost += response.cost_estimate
                self.response_times.append(response.response_time)
                
                # Parse strategic thinking response
                thinking_result = self._parse_thinking_response(response.content)
                
                # Store in strategic memory
                self.strategic_memory.append(f"Round {context.current_round} thinking: {thinking_result.get('strategy', '')}")
                
                return thinking_result
                
            except Exception as e:
                self.logger.warning(f"Thinking attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    # Return fallback thinking
                    return {
                        "reasoning": "Unable to complete strategic analysis due to technical issues",
                        "strategy": "Will proceed with preference-based approach",
                        "target_items": [],
                        "anticipated_resistance": []
                    }
        
        # Should not reach here
        return {}
    
    def _build_thinking_messages(self, context: NegotiationContext, prompt: str) -> List[Dict[str, str]]:
        """Build messages for private thinking."""
        system_prompt = f"""You are {context.agent_id}, a strategic negotiation agent engaged in private strategic planning.

PRIVATE ANALYSIS SESSION:
- This is completely private - other agents cannot see your thoughts
- Analyze the situation strategically to plan your next proposal
- Consider what you learned from discussions and previous rounds
- Think about other agents' likely preferences and resistance points

YOUR PREFERENCES:
{self._format_preferences(context.preferences, context.items)}

STRATEGIC MINDSET:
- Maximize your utility while ensuring proposals can get unanimous support
- Identify items others seem to value less but you value highly
- Plan how to frame proposals to appear fair to all agents
- Consider what concessions you might need to make

Response format: Provide your analysis as structured strategic thinking."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Add conversation history context if available
        if hasattr(context, 'conversation_history') and context.conversation_history:
            # Add recent context
            recent_context = "RECENT DISCUSSION HIGHLIGHTS:\n"
            for msg in context.conversation_history[-4:]:  # Last 4 messages
                if isinstance(msg, dict) and msg.get('content'):
                    agent = msg.get('agent_id', 'unknown')
                    content = msg.get('content', '')[:150]  # Truncate for context
                    recent_context += f"- {agent}: {content}\n"
            
            # Insert context before the main prompt
            messages[1]["content"] = recent_context + "\n" + messages[1]["content"]
        
        return messages
    
    def _parse_thinking_response(self, response_content: str) -> Dict[str, Any]:
        """Parse the strategic thinking response with enhanced O3 support."""
        try:
            # Try to parse as JSON first
            if response_content.strip().startswith('{'):
                return json.loads(response_content)
            
            # Check if it's O3 format (often uses bullet points and structured text)
            if self._is_o3_thinking_format(response_content):
                return self._parse_o3_thinking_format(response_content)
            
            # Otherwise, extract structured information from text
            result = {
                "reasoning": "",
                "strategy": "",
                "target_items": [],
                "anticipated_resistance": []
            }
            
            # Simple text parsing for structured elements
            lines = response_content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for section headers
                if 'reasoning' in line.lower() and ':' in line:
                    current_section = 'reasoning'
                    result['reasoning'] = line.split(':', 1)[1].strip() if ':' in line else ""
                elif 'strategy' in line.lower() and ':' in line:
                    current_section = 'strategy'
                    result['strategy'] = line.split(':', 1)[1].strip() if ':' in line else ""
                elif 'target' in line.lower() and 'items' in line.lower():
                    current_section = 'target_items'
                elif 'resistance' in line.lower() or 'opposition' in line.lower():
                    current_section = 'anticipated_resistance'
                else:
                    # Add content to current section
                    if current_section == 'reasoning':
                        result['reasoning'] += " " + line
                    elif current_section == 'strategy':
                        result['strategy'] += " " + line
                    elif current_section == 'target_items' and line.startswith('-'):
                        result['target_items'].append(line[1:].strip())
                    elif current_section == 'anticipated_resistance' and line.startswith('-'):
                        result['anticipated_resistance'].append(line[1:].strip())
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to parse thinking response: {e}")
            return {
                "reasoning": response_content[:200] + "..." if len(response_content) > 200 else response_content,
                "strategy": "Basic preference-driven approach",
                "target_items": [],
                "anticipated_resistance": []
            }
    
    def _is_o3_thinking_format(self, response_content: str) -> bool:
        """Check if the response is in O3's typical thinking format."""
        content_lower = response_content.lower()
        
        # O3 often uses bullet points, numbered lists, and analytical structure
        o3_indicators = [
            "- " in response_content or "• " in response_content,  # Bullet points
            "1." in response_content or "2." in response_content,  # Numbered lists
            "analysis:" in content_lower,
            "strategic" in content_lower,
            "quill" in content_lower or "stone" in content_lower,  # Specific to our items
            "utility" in content_lower,
            "proposal" in content_lower and "strategy" in content_lower
        ]
        
        return sum(o3_indicators) >= 2  # Must match at least 2 indicators
    
    def _parse_o3_thinking_format(self, response_content: str) -> Dict[str, Any]:
        """Parse O3's specific thinking format."""
        result = {
            "reasoning": "",
            "strategy": "",
            "target_items": [],
            "anticipated_resistance": []
        }
        
        # O3 tends to provide comprehensive analysis in paragraph or bullet form
        lines = response_content.split('\n')
        full_text = response_content
        
        # Extract reasoning (usually the bulk of O3's analysis)
        reasoning_parts = []
        strategy_parts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for strategy-related content
            if any(keyword in line.lower() for keyword in ['strategy', 'approach', 'plan', 'propose']):
                strategy_parts.append(line)
            # Look for analysis content
            elif any(keyword in line.lower() for keyword in ['analysis', 'prefer', 'value', 'utility', 'because']):
                reasoning_parts.append(line)
            # Bullet points and lists are often reasoning
            elif line.startswith(('- ', '• ', '1.', '2.', '3.')):
                reasoning_parts.append(line)
            else:
                # Default to reasoning for O3's comprehensive analysis
                reasoning_parts.append(line)
        
        # Combine reasoning
        if reasoning_parts:
            result['reasoning'] = ' '.join(reasoning_parts)
        else:
            # Fallback: use first half of content as reasoning
            result['reasoning'] = full_text[:len(full_text)//2]
        
        # Combine strategy
        if strategy_parts:
            result['strategy'] = ' '.join(strategy_parts) 
        else:
            # Fallback: extract strategy from reasoning or use second half
            result['strategy'] = full_text[len(full_text)//2:]
        
        # Extract target items (O3 often mentions specific items)
        items_mentioned = []
        content_lower = full_text.lower()
        item_names = ['apple', 'jewel', 'stone', 'quill', 'pencil']
        for item in item_names:
            if item in content_lower:
                items_mentioned.append(item.title())
        result['target_items'] = items_mentioned[:3]  # Limit to top 3
        
        # Extract anticipated resistance (O3 often mentions other agents)
        resistance_mentions = []
        if 'haiku' in content_lower or 'other' in content_lower:
            resistance_mentions.append("May face resistance from other agents")
        if 'unanimous' in content_lower:
            resistance_mentions.append("Need unanimous approval")
        result['anticipated_resistance'] = resistance_mentions
        
        return result
    
    def _build_context_messages(self, context: NegotiationContext, 
                               prompt: str) -> List[Dict[str, str]]:
        """Build conversation context for the LLM."""
        messages = []
        
        # Add system message
        system_prompt = self._build_system_prompt(context)
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history (last N messages to stay within limits)
        max_history = 20  # Configurable
        recent_history = context.conversation_history[-max_history:]
        
        for msg in recent_history:
            if msg.get("from") == self.agent_id:
                messages.append({"role": "assistant", "content": msg["content"]})
            else:
                sender = msg.get("from", "other")
                content = f"[{sender}]: {msg['content']}"
                messages.append({"role": "user", "content": content})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def generate_response(self, context: NegotiationContext, 
                               prompt: str) -> AgentResponse:
        """Generate a response given the negotiation context and prompt."""
        # Wait for rate limits
        await self.rate_limiter.wait_if_needed()
        
        # Build messages
        messages = self._build_context_messages(context, prompt)
        
        # Call LLM with retries
        for attempt in range(self.config.max_retries):
            try:
                response = await self._call_llm_api(messages)
                
                # Update tracking
                self.total_requests += 1
                if response.tokens_used:
                    self.total_tokens += response.tokens_used
                if response.cost_estimate:
                    self.total_cost += response.cost_estimate
                self.response_times.append(response.response_time)
                
                # Update memory
                self.conversation_memory.append({
                    "context": context.to_dict(),
                    "prompt": prompt,
                    "response": response.content,
                    "timestamp": time.time()
                })
                
                return response
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise
    
    async def propose_allocation(self, context: NegotiationContext) -> Dict[str, Any]:
        """Generate a proposal for item allocation."""
        # Build agent list for the prompt
        if context.agents:
            # Handle both cases: agents as strings or agent objects
            agent_list = []
            for agent in context.agents:
                if isinstance(agent, str):
                    agent_list.append(agent)
                else:
                    agent_list.append(agent.agent_id)
        else:
            agent_list = []
        
        agent_examples = {}
        for i, agent_id in enumerate(agent_list):
            agent_examples[agent_id] = f"[item_indices_for_{agent_id}]"
        
        prompt = f"""Please propose an allocation of items among all agents. 

Current Context:
- Items: {context.items} (indices 0-{len(context.items)-1})
- Agents: {agent_list}
- Round: {context.current_round}

Consider:
- Your preferences and utility maximization
- What you know about other agents' preferences
- Previous proposals and their reception
- Strategic positioning for this round

Respond with ONLY a JSON object in this exact format:
{{
    "allocation": {{
        "{agent_list[0] if agent_list else 'agent_0'}": [0, 2],
        "{agent_list[1] if len(agent_list) > 1 else 'agent_1'}": [1, 3],
        "{agent_list[2] if len(agent_list) > 2 else 'agent_2'}": [4]
    }},
    "reasoning": "Brief explanation of your proposal strategy"
}}

Use actual agent IDs as keys and item indices (0-{len(context.items)-1}) as values."""
        
        response = await self.generate_response(context, prompt)
        
        try:
            # Try to parse the JSON directly first
            proposal = json.loads(response.content)
            # Ensure required fields are present
            if "allocation" not in proposal:
                raise ValueError("Missing allocation field")
            if "reasoning" not in proposal:
                proposal["reasoning"] = "No reasoning provided"
            proposal["proposed_by"] = self.agent_id
            proposal["round"] = context.current_round
            return proposal
        except (json.JSONDecodeError, ValueError) as e:
            # Log the parsing issue for debugging
            self.logger.debug(f"Direct JSON parsing failed for {self.agent_id}: {e}")
            self.logger.debug(f"Raw response content: {response.content[:300]}...")
            
            # Try to extract JSON from text response
            try:
                import re
                
                # Look for JSON block in the response
                json_match = re.search(r'\{[\s\S]*\}', response.content)
                if json_match:
                    json_str = json_match.group(0)
                    self.logger.debug(f"Extracted JSON string: {json_str[:200]}...")
                    proposal = json.loads(json_str)
                    
                    # Convert the allocation format if needed
                    if "allocation" in proposal:
                        allocation = proposal["allocation"]
                        # Convert agent names to indices if needed
                        converted_allocation = {}
                        
                        # Map agent names to actual IDs
                        agent_mapping = {}
                        if context.agents:
                            for i, agent in enumerate(context.agents):
                                if isinstance(agent, str):
                                    agent_mapping[f"agent_{i}"] = agent
                                else:
                                    agent_mapping[f"agent_{i}"] = agent.agent_id
                                
                        for agent_key, items in allocation.items():
                            # Convert generic agent names to actual agent IDs
                            if agent_key in agent_mapping:
                                actual_agent_key = agent_mapping[agent_key]
                            else:
                                actual_agent_key = agent_key
                                
                            # Handle both item names and indices
                            if isinstance(items, list) and len(items) > 0:
                                if isinstance(items[0], str):
                                    # Convert item names to indices
                                    item_indices = []
                                    for item_name in items:
                                        for i, context_item in enumerate(context.items):
                                            if context_item == item_name:
                                                item_indices.append(i)
                                                break
                                    converted_allocation[actual_agent_key] = item_indices
                                else:
                                    converted_allocation[actual_agent_key] = items
                            else:
                                converted_allocation[actual_agent_key] = items
                        
                        proposal["allocation"] = converted_allocation
                    
                    # Ensure required fields are present
                    if "reasoning" not in proposal:
                        proposal["reasoning"] = "No reasoning provided"
                    
                    proposal["proposed_by"] = self.agent_id
                    proposal["round"] = context.current_round
                    return proposal
            except (json.JSONDecodeError, AttributeError):
                pass
            
            # Fallback: create a simple proposal
            self.logger.warning(f"Failed to parse proposal JSON: {response.content[:200]}...")
            return {
                "allocation": {self.agent_id: list(range(len(context.items)))},
                "reasoning": "Failed to parse structured response",
                "proposed_by": self.agent_id,
                "round": context.current_round
            }
    
    async def vote_on_proposal(self, context: NegotiationContext, 
                              proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Vote on a proposed allocation."""
        prompt = f"""A proposal has been made for item allocation:

PROPOSAL: {json.dumps(proposal['allocation'], indent=2)}
REASONING: {proposal.get('reasoning', 'No reasoning provided')}
PROPOSED BY: {proposal.get('proposed_by', 'Unknown')}

Please vote on this proposal. Consider:
- How this allocation affects your utility
- Whether you might get a better deal by continuing negotiation
- The strategic implications of accepting vs. rejecting

Respond with ONLY a JSON object in this exact format:
{{
    "vote": "accept",
    "reasoning": "Brief explanation of your vote"
}}

Vote must be either "accept" or "reject"."""
        
        response = await self.generate_response(context, prompt)
        
        try:
            # Try to parse the JSON directly first
            vote = json.loads(response.content)
            vote["voter"] = self.agent_id
            vote["round"] = context.current_round
            return vote
        except json.JSONDecodeError:
            # Try to extract JSON from text response
            try:
                import re
                
                # Look for JSON block in the response
                json_match = re.search(r'\{[\s\S]*\}', response.content)
                if json_match:
                    json_str = json_match.group(0)
                    vote = json.loads(json_str)
                    vote["voter"] = self.agent_id
                    vote["round"] = context.current_round
                    return vote
            except (json.JSONDecodeError, AttributeError):
                pass
            
            # Fallback: reject by default
            self.logger.warning(f"Failed to parse vote JSON: {response.content[:200]}...")
            return {
                "vote": "reject",
                "reasoning": "Failed to parse structured response",
                "voter": self.agent_id,
                "round": context.current_round
            }
    
    async def discuss(self, context: NegotiationContext, topic: str = "") -> str:
        """Participate in open discussion."""
        prompt = f"""This is an open discussion phase. 

{f'Discussion topic: {topic}' if topic else 'Feel free to discuss strategy, preferences, or anything relevant to the negotiation.'}

Please contribute to the discussion. You might:
- Share (possibly strategic) information about your preferences
- Ask questions about others' preferences
- Propose informal deals or arrangements
- Express frustration, satisfaction, or other emotions
- Try to influence other agents' strategies

Keep your response conversational and authentic. Respond as you would in a real negotiation."""
        
        response = await self.generate_response(context, prompt)
        return response.content
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "agent_id": self.agent_id,
            "model_type": self.config.model_type.value,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_response_time": avg_response_time,
            "conversation_length": len(self.conversation_memory)
        }
    
    async def think_privately(self, context: NegotiationContext, reflection_prompt: str) -> Any:
        """
        Generate private reflection thoughts for post-round analysis.
        
        This method allows agents to privately reflect on round outcomes and
        extract strategic insights for future negotiations.
        """
        # Wait for rate limits
        await self.rate_limiter.wait_if_needed()
        
        # Build messages for reflection prompt
        reflection_messages = self._build_reflection_messages(context, reflection_prompt)
        
        # Call LLM for private reflection
        for attempt in range(self.config.max_retries):
            try:
                response = await self._call_llm_api(reflection_messages)
                
                # Update tracking
                self.total_requests += 1
                if response.tokens_used:
                    self.total_tokens += response.tokens_used
                if response.cost_estimate:
                    self.total_cost += response.cost_estimate
                self.response_times.append(response.response_time)
                
                # Store reflection in conversation memory
                self.conversation_memory.append({
                    "type": "private_reflection",
                    "round": context.current_round,
                    "content": response.content[:500] if response.content else "",  # Truncated for memory
                    "timestamp": time.time()
                })
                
                return response
                
            except Exception as e:
                self.logger.error(f"Private reflection attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        return None
    
    def _build_reflection_messages(self, context: NegotiationContext, reflection_prompt: str) -> List[Dict[str, str]]:
        """Build messages for private reflection."""
        messages = [
            {"role": "system", "content": f"You are {context.agent_id}, a strategic negotiation agent reflecting privately on round outcomes."},
            {"role": "user", "content": reflection_prompt}
        ]
        
        # Add relevant strategic memory context if available
        if self.strategic_memory:
            recent_memory = self.strategic_memory[-5:]  # Last 5 strategic memories
            memory_context = "\n".join(recent_memory)
            messages.append({
                "role": "user", 
                "content": f"Your recent strategic memory:\n{memory_context}\n\nNow provide your reflection on this round:"
            })
        
        return messages
    
    def export_conversation_memory(self, filepath: str) -> None:
        """Export conversation memory to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                "agent_id": self.agent_id,
                "model_type": self.config.model_type.value,
                "conversation_memory": self.conversation_memory,
                "strategic_memory": self.strategic_memory,
                "performance_stats": self.get_performance_stats()
            }, f, indent=2)


class AnthropicAgent(BaseLLMAgent):
    """LLM agent using Anthropic's Claude models."""
    
    def __init__(self, agent_id: str, config: LLMConfig, api_key: str):
        super().__init__(agent_id, config)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not available. Install with: pip install anthropic")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Model name mapping
        if config.model_type == ModelType.CLAUDE_3_OPUS:
            self.model_name = "claude-3-opus-20240229"
        elif config.model_type == ModelType.CLAUDE_3_SONNET:
            self.model_name = "claude-3-sonnet-20240229"
        elif config.model_type == ModelType.CLAUDE_3_HAIKU:
            self.model_name = "claude-3-haiku-20240307"
        elif config.model_type == ModelType.CLAUDE_3_5_SONNET:
            self.model_name = "claude-3-5-sonnet-20241022"
        else:
            raise ValueError(f"Unsupported Anthropic model: {config.model_type}")
    
    async def _call_llm_api(self, messages: List[Dict[str, str]], **kwargs) -> AgentResponse:
        """Call Anthropic API."""
        start_time = time.time()
        
        # Extract system message
        system_message = None
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        
        # Call Anthropic API (max_tokens is required)
        api_params = {
            "model": self.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,  # Required by Anthropic API
            "system": system_message,
            "messages": user_messages,
            **self.config.custom_parameters,
            **kwargs
        }
        
        response = await self.client.messages.create(**api_params)
        
        response_time = time.time() - start_time
        
        return AgentResponse(
            content=response.content[0].text,
            model_used=self.model_name,
            response_time=response_time,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            cost_estimate=self._estimate_cost(response.usage.input_tokens, response.usage.output_tokens),
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "stop_reason": response.stop_reason
            }
        )
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on Anthropic pricing (as of 2024)."""
        # Current pricing estimates 
        if "opus" in self.model_name:
            return (input_tokens * 15 + output_tokens * 75) / 1_000_000
        elif "sonnet" in self.model_name:
            return (input_tokens * 3 + output_tokens * 15) / 1_000_000
        elif "haiku" in self.model_name:
            return (input_tokens * 0.25 + output_tokens * 1.25) / 1_000_000
        else:
            return 0.0  # Unknown model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "anthropic",
            "model_type": self.config.model_type.value,
            "model_name": self.model_name,
            "context_window": self._get_context_window(),
            "capabilities": ["text_generation", "json_mode", "reasoning"]
        }
    
    def _get_context_window(self) -> int:
        """Get context window size for the model."""
        if "opus" in self.model_name or "sonnet" in self.model_name:
            return 200_000
        elif "haiku" in self.model_name:
            return 200_000
        else:
            return 100_000  # Default


class OpenAIAgent(BaseLLMAgent):
    """LLM agent using OpenAI models."""
    
    def __init__(self, agent_id: str, config: LLMConfig, api_key: str):
        super().__init__(agent_id, config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not available. Install with: pip install openai")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # Model name mapping
        if config.model_type == ModelType.GPT_4:
            self.model_name = "gpt-4"
        elif config.model_type == ModelType.GPT_4_TURBO:
            self.model_name = "gpt-4-turbo"
        elif config.model_type == ModelType.GPT_4O:
            self.model_name = "gpt-4o"
        elif config.model_type == ModelType.O3_MINI:
            self.model_name = "o3-mini"
        elif config.model_type == ModelType.O3:
            self.model_name = "o3"
        else:
            raise ValueError(f"Unsupported OpenAI model: {config.model_type}")
    
    async def _call_llm_api(self, messages: List[Dict[str, str]], **kwargs) -> AgentResponse:
        """Call OpenAI API."""
        start_time = time.time()
        
        # O3 models have specific parameter requirements
        api_params = {
            "model": self.model_name,
            "messages": messages,
        }
        
        if "o3" in self.model_name.lower():
            # O3 models use max_completion_tokens and only support temperature=1
            # Remove max_completion_tokens to allow unlimited generation
            api_params["temperature"] = 1  # O3 only supports temperature=1
        else:
            # Standard models - remove max_tokens for unlimited generation
            api_params["temperature"] = self.config.temperature
        
        # Add custom parameters but exclude any max_tokens variants
        custom_params = {k: v for k, v in self.config.custom_parameters.items() 
                        if k not in ['max_tokens', 'max_completion_tokens']}
        
        response = await self.client.chat.completions.create(
            **api_params,
            **custom_params,
            **kwargs
        )
        
        response_time = time.time() - start_time
        
        # DEBUG: Log O3 API responses only if they fail
        if "o3" in self.model_name.lower() and not response.choices[0].message.content:
            print(f"\n⚠️  O3 API Response Issue:")
            print(f"Model: {self.model_name}")
            print(f"Finish reason: {response.choices[0].finish_reason}")
            print(f"Content length: {len(response.choices[0].message.content or '')}")
            if response.usage:
                print(f"Tokens used: {response.usage.total_tokens}")
            print("=" * 80)
        
        return AgentResponse(
            content=response.choices[0].message.content,  # Keep original - no masking
            model_used=self.model_name,
            response_time=response_time,
            tokens_used=response.usage.total_tokens if response.usage else None,
            cost_estimate=self._estimate_cost(response.usage) if response.usage else None,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.dict() if response.usage else None
            }
        )
    
    def _estimate_cost(self, usage) -> float:
        """Estimate cost based on OpenAI pricing."""
        # Current pricing estimates
        if "gpt-4" in self.model_name:
            return (usage.prompt_tokens * 30 + usage.completion_tokens * 60) / 1_000_000
        elif "o3" in self.model_name:
            # O3 pricing (high-end model)
            return (usage.prompt_tokens * 100 + usage.completion_tokens * 200) / 1_000_000
        elif "gpt-3.5" in self.model_name:
            return (usage.prompt_tokens * 1.5 + usage.completion_tokens * 2) / 1_000_000
        else:
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model_type": self.config.model_type.value,
            "model_name": self.model_name,
            "context_window": self._get_context_window(),
            "capabilities": ["text_generation", "json_mode", "reasoning"]
        }
    
    def _get_context_window(self) -> int:
        """Get context window size for the model."""
        if "gpt-4" in self.model_name:
            return 32_000 if "turbo" not in self.model_name else 128_000
        elif "o3" in self.model_name:
            return 128_000  # Estimated
        elif "gpt-3.5" in self.model_name:
            return 16_000
        else:
            return 16_000  # Default


class SimulatedAgent(BaseLLMAgent):
    """
    Simulated LLM agent for testing and development.
    
    Uses deterministic response patterns rather than external APIs,
    enabling rapid testing and development without API costs.
    """
    
    def __init__(self, agent_id: str, config: LLMConfig, 
                 strategic_level: str = "balanced"):
        super().__init__(agent_id, config)
        
        # Configure strategic behavior
        if strategic_level == "aggressive":
            self.cooperation_tendency = 0.2
            self.strategic_sophistication = 0.9
            self.response_style = "competitive"
        elif strategic_level == "cooperative":
            self.cooperation_tendency = 0.8
            self.strategic_sophistication = 0.6
            self.response_style = "collaborative"
        else:  # balanced
            self.cooperation_tendency = 0.5
            self.strategic_sophistication = 0.7
            self.response_style = "strategic"
        
        # Response templates for different scenarios
        self.response_templates = self._initialize_response_templates()
    
    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates based on strategic level."""
        if self.response_style == "competitive":
            return {
                "proposal": [
                    "I believe this allocation maximizes efficiency given the current market conditions.",
                    "Based on my analysis, this distribution creates the optimal outcome for all parties.",
                    "This proposal reflects careful consideration of everyone's stated positions."
                ],
                "vote_accept": [
                    "This proposal aligns with my strategic objectives.",
                    "I can support this allocation given the current circumstances.",
                    "This represents a reasonable compromise."
                ],
                "vote_reject": [
                    "I believe we can achieve a more balanced outcome with further discussion.",
                    "This allocation doesn't fully address the concerns raised earlier.",
                    "I think there's room for improvement in this proposal."
                ],
                "discussion": [
                    "We should consider the long-term implications of our decisions here.",
                    "Has anyone analyzed how this might affect future negotiations?",
                    "I'm curious about the reasoning behind some of the earlier proposals."
                ]
            }
        elif self.response_style == "collaborative":
            return {
                "proposal": [
                    "I'd like to propose an allocation that I think benefits everyone fairly.",
                    "Here's a distribution that tries to balance all our interests.",
                    "This proposal aims to create value for the entire group."
                ],
                "vote_accept": [
                    "I think this is a fair proposal that works for everyone.",
                    "This allocation seems to address most of our concerns.",
                    "I'm happy to support this collaborative solution."
                ],
                "vote_reject": [
                    "I wonder if we could adjust this slightly to better serve everyone.",
                    "Perhaps we could explore some alternatives that work better for all of us.",
                    "I'd like to discuss how we might improve this proposal."
                ],
                "discussion": [
                    "How can we make sure everyone's needs are being met?",
                    "I want to understand what's most important to each of you.",
                    "Let's work together to find a solution that benefits us all."
                ]
            }
        else:  # strategic
            return {
                "proposal": [
                    "This allocation reflects my assessment of the current negotiation dynamics.",
                    "I've structured this proposal to create mutual benefits while addressing key concerns.",
                    "This distribution balances immediate needs with strategic positioning."
                ],
                "vote_accept": [
                    "This proposal offers a good balance of risk and reward.",
                    "I can see the strategic value in this allocation.",
                    "This aligns well with my objectives for this negotiation."
                ],
                "vote_reject": [
                    "I think we have an opportunity to create more value with a different approach.",
                    "While this has merit, I believe we can do better.",
                    "I'd like to explore some alternative structures."
                ],
                "discussion": [
                    "What are the key factors driving everyone's decision-making here?",
                    "I'm interested in understanding the strategic considerations at play.",
                    "How do we balance individual interests with collective outcomes?"
                ]
            }
    
    async def _call_llm_api(self, messages: List[Dict[str, str]], **kwargs) -> AgentResponse:
        """Simulate API call with structured responses."""
        start_time = time.time()
        
        # Add realistic delay
        import asyncio
        await asyncio.sleep(random.uniform(0.3, 1.5))
        
        # Analyze the prompt to determine response type
        last_message = messages[-1]["content"].lower()
        
        if "vote" in last_message and "proposal" in last_message:
            content = self._generate_vote_response(messages)
        elif "propose" in last_message and "allocation" in last_message:
            content = self._generate_proposal_response(messages)
        else:
            content = self._generate_discussion_response(messages)
        
        response_time = time.time() - start_time
        
        return AgentResponse(
            content=content,
            model_used=f"simulated-{self.response_style}",
            response_time=response_time,
            tokens_used=len(content.split()),
            cost_estimate=0.0,  # Simulated is free
            metadata={
                "strategic_level": self.response_style,
                "cooperation_tendency": self.cooperation_tendency,
                "strategic_sophistication": self.strategic_sophistication
            }
        )
    
    def _generate_proposal_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a structured proposal response."""
        # Simple allocation strategy based on agent behavior
        if self.response_style == "competitive":
            # Try to get high-value items for self
            allocation = {self.agent_id: [0, 1]}  # Take first items
            reasoning = random.choice(self.response_templates["proposal"])
        elif self.response_style == "collaborative":
            # Try to distribute fairly
            allocation = {self.agent_id: [0]}  # Take one item
            reasoning = random.choice(self.response_templates["proposal"])
        else:  # strategic
            # Mixed approach
            allocation = {self.agent_id: [0, 2]}  # Strategic selection
            reasoning = random.choice(self.response_templates["proposal"])
        
        return json.dumps({
            "allocation": allocation,
            "reasoning": reasoning
        })
    
    def _generate_vote_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a structured vote response."""
        # Decision based on cooperation tendency
        if random.random() < self.cooperation_tendency:
            vote = "accept"
            reasoning = random.choice(self.response_templates["vote_accept"])
        else:
            vote = "reject"
            reasoning = random.choice(self.response_templates["vote_reject"])
        
        return json.dumps({
            "vote": vote,
            "reasoning": reasoning
        })
    
    def _generate_discussion_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a discussion response."""
        return random.choice(self.response_templates["discussion"])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get simulated model information."""
        return {
            "provider": "simulated",
            "model_type": self.config.model_type.value,
            "model_name": f"simulated-{self.response_style}",
            "context_window": 100_000,  # Unlimited for simulated
            "capabilities": ["text_generation", "json_mode", "strategic_simulation"],
            "strategic_level": self.response_style,
            "cooperation_tendency": self.cooperation_tendency,
            "strategic_sophistication": self.strategic_sophistication
        }