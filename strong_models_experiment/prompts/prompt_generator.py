"""Prompt generation utilities for negotiation phases."""

from typing import List, Dict, Any


class PromptGenerator:
    """Generate prompts for different negotiation phases."""
    
    @staticmethod
    def create_game_rules_prompt(items: List[Dict], num_agents: int, config: Dict[str, Any]) -> str:
        """Create the standardized game rules explanation prompt."""
        items_list = [f"  {i}: {item['name']}" for i, item in enumerate(items)]
        items_text = "\n".join(items_list)
        
        return f"""Welcome to the Multi-Agent Negotiation Game!

You are participating in a strategic negotiation with {num_agents} agents over {len(items)} valuable items. Here are the complete rules:

**ITEMS BEING NEGOTIATED:**
{items_text}

**GAME STRUCTURE:**
- There are {num_agents} agents participating (including you)
- The negotiation will last up to {config["t_rounds"]} rounds
- Each round follows a structured sequence of phases

**YOUR PRIVATE PREFERENCES:**
You have been assigned private preferences for each item (values 0-10). These preferences are SECRET.

**VOTING RULES:**
- You vote "accept" or "reject" on each proposal
- A proposal needs UNANIMOUS acceptance to pass
- If no proposal gets unanimous support, we continue to the next round

**WINNING CONDITIONS:**
- Your goal is to maximize your total utility
- No deal means everyone gets zero utility
- Consider both immediate gains and the likelihood of proposals being accepted

Please acknowledge that you understand these rules and are ready to participate!"""
    
    @staticmethod
    def create_preference_assignment_prompt(items: List[Dict], agent_preferences: List[float], agent_id: str) -> str:
        """Create the private preference assignment prompt for a specific agent."""
        preference_details = []
        for i, item in enumerate(items):
            preference_value = agent_preferences[i] if isinstance(agent_preferences, list) else agent_preferences.get(i, 0)
            priority_level = PromptGenerator._get_priority_level(preference_value)
            preference_details.append(f"  {i}: {item['name']} → {preference_value:.2f}/10 ({priority_level})")
        
        preferences_text = "\n".join(preference_details)
        
        if isinstance(agent_preferences, list):
            max_utility = sum(agent_preferences)
        else:
            max_utility = sum(agent_preferences.values())
        
        return f"""🔒 CONFIDENTIAL: Your Private Preferences Assignment

{agent_id}, you have been assigned the following SECRET preference values for each item:

**YOUR PRIVATE ITEM PREFERENCES:**
{preferences_text}

**STRATEGIC ANALYSIS:**
- Your maximum possible utility: {max_utility:.2f} points (if you get ALL items)
- Your top priorities: {PromptGenerator._get_top_items(items, agent_preferences, 2)}
- Your lower priorities: {PromptGenerator._get_bottom_items(items, agent_preferences, 2)}

**STRATEGIC CONSIDERATIONS:**
1. Other agents don't know your exact preferences
2. You may choose to reveal some preferences truthfully or misleadingly
3. Consider which agents might have complementary preferences
4. Remember: you need ALL agents to accept a proposal

Please acknowledge that you understand your private preferences."""
    
    @staticmethod
    def create_initial_discussion_prompt(items: List[Dict], round_num: int, max_rounds: int) -> str:
        """Create the discussion prompt for the first round."""
        items_list = [f"  {i}: {item['name']}" for i, item in enumerate(items)]
        items_text = "\n".join(items_list)
        
        return f"""🗣️ PUBLIC DISCUSSION PHASE - Round {round_num}/{max_rounds}

This is the open discussion phase where all agents can share information about their preferences.

**ITEMS AVAILABLE:**
{items_text}

**DISCUSSION OBJECTIVES:**
- Share strategic information about your preferences
- Learn about other agents' priorities
- Explore potential coalition opportunities
- Identify mutually beneficial trade possibilities

Please share your thoughts on the items and any initial ideas for how we might structure a deal."""
    
    @staticmethod
    def create_ongoing_discussion_prompt(items: List[Dict], round_num: int, max_rounds: int) -> str:
        """Create the discussion prompt for subsequent rounds."""
        items_list = [f"  {i}: {item['name']}" for i, item in enumerate(items)]
        items_text = "\n".join(items_list)
        
        urgency_note = ""
        if round_num >= max_rounds - 1:
            urgency_note = "\n⏰ **URGENT**: This is one of the final rounds!"
        
        return f"""🗣️ PUBLIC DISCUSSION PHASE - Round {round_num}/{max_rounds}

Previous proposals didn't reach consensus. Adjust your approach based on what you learned.

**ITEMS AVAILABLE:**
{items_text}{urgency_note}

**REFLECTION & STRATEGY:**
- What did you learn from previous proposals and votes?
- Which agents have conflicting vs. compatible preferences?
- How can you adjust to build consensus?

Given what happened in previous rounds, what's your updated strategy?"""
    
    @staticmethod
    def create_contextual_discussion_prompt(base_prompt: str, agent_id: str, discussion_history: List[str], 
                                           speaker_order: int, total_speakers: int) -> str:
        """Create discussion prompt with context from current round's discussion."""
        context_section = ""
        
        if discussion_history:
            context_section = f"""
**DISCUSSION SO FAR THIS ROUND:**
{len(discussion_history)} agent(s) have already spoken.

**YOUR TURN ({speaker_order}/{total_speakers})**:
Consider what others have said and respond strategically.

"""
        else:
            context_section = f"""**YOU'RE SPEAKING FIRST ({speaker_order}/{total_speakers})**:
Set the tone for this round's discussion.

"""
        
        return base_prompt + "\n\n" + context_section
    
    @staticmethod
    def create_thinking_prompt(items: List[Dict], round_num: int, max_rounds: int, discussion_messages: List[Dict]) -> str:
        """Create the private thinking prompt for strategic planning."""
        items_list = [f"  {i}: {item['name']}" for i, item in enumerate(items)]
        items_text = "\n".join(items_list)
        
        urgency_note = ""
        if round_num >= max_rounds - 1:
            urgency_note = "\n⚠️ **CRITICAL**: This is one of your final opportunities!"
        
        return f"""🧠 PRIVATE THINKING PHASE - Round {round_num}/{max_rounds}

This is your private strategic planning time.

**ITEMS AVAILABLE:**
{items_text}{urgency_note}

**STRATEGIC ANALYSIS TASKS:**
1. What did you learn about other agents' preferences?
2. Which items do others value less that you value highly?
3. What allocation would maximize your utility while achieving consensus?
4. What concessions might be necessary?

**OUTPUT REQUIRED:**
- **Reasoning**: Your analysis of the situation
- **Strategy**: Your overall approach for this round
- **Target Items**: Items you most want to secure

Remember: This thinking is completely private."""
    
    @staticmethod
    def _get_priority_level(value: float) -> str:
        """Convert numeric preference to priority description."""
        if value >= 7.0:
            return "HIGH PRIORITY"
        elif value >= 4.0:
            return "Medium Priority"
        else:
            return "Low Priority"
    
    @staticmethod
    def _get_top_items(items: List[Dict], preferences: Any, count: int = 2) -> str:
        """Get the top N items by preference value."""
        if isinstance(preferences, list):
            item_values = [(i, items[i]['name'], preferences[i]) for i in range(len(items))]
        else:
            item_values = [(i, items[i]['name'], preferences.get(i, 0)) for i in range(len(items))]
        
        top_items = sorted(item_values, key=lambda x: x[2], reverse=True)[:count]
        return ", ".join([f"{item[1]} ({item[2]:.1f})" for item in top_items])
    
    @staticmethod
    def _get_bottom_items(items: List[Dict], preferences: Any, count: int = 2) -> str:
        """Get the bottom N items by preference value."""
        if isinstance(preferences, list):
            item_values = [(i, items[i]['name'], preferences[i]) for i in range(len(items))]
        else:
            item_values = [(i, items[i]['name'], preferences.get(i, 0)) for i in range(len(items))]
        
        bottom_items = sorted(item_values, key=lambda x: x[2])[:count]
        return ", ".join([f"{item[1]} ({item[2]:.1f})" for item in bottom_items])