#!/usr/bin/env python3
"""
Generate docs/reference/all_prompts.md from the live prompt methods.

The reference is intentionally example-driven: it renders representative prompt
instances from the current code paths using fixed sample game states so the
examples remain stable and readable.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from textwrap import dedent

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from game_environments.base import (
    CoFundingConfig,
    DiplomaticTreatyConfig,
    ItemAllocationConfig,
)
from game_environments.co_funding import CoFundingGame
from game_environments.diplomatic_treaty import DiplomaticTreatyGame
from game_environments.item_allocation import ItemAllocationGame


OUTPUT_PATH = BASE_DIR / "docs/reference/all_prompts.md"


def code_block(text: str) -> str:
    return f"```\n{text.rstrip()}\n```"


def normalize_section(text: str) -> str:
    lines = text.strip().splitlines()
    normalized = []
    for line in lines:
        if line.startswith("        "):
            normalized.append(line[8:])
        else:
            normalized.append(line)
    return "\n".join(normalized).strip()


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("-" * len(h) for h in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header_line, divider, body])


def render_asset_tables() -> str:
    item_rows = [[str(i), name] for i, name in enumerate(ItemAllocationGame.ITEM_NAMES)]
    issue_rows = [
        [
            str(i + 1),
            DiplomaticTreatyGame.ISSUE_NAMES[i],
            DiplomaticTreatyGame.ISSUE_PROPOSITIONS[i].replace("|", "\\|"),
            DiplomaticTreatyGame.ISSUE_INTERP_TEMPLATES[i],
        ]
        for i in range(len(DiplomaticTreatyGame.ISSUE_NAMES))
    ]
    project_rows = [[str(i), name] for i, name in enumerate(CoFundingGame.PROJECT_NAMES)]

    return normalize_section(
        f"""
        ## Full Asset Lists

        ### Game 1 - All Possible Items (`ITEM_NAMES`)

        Up to 10 items; the game uses the first `m_items` from this list.

        {md_table(["Index", "Name"], item_rows)}

        ---

        ### Game 2 - All Possible Issues (`ISSUE_NAMES`, `ISSUE_PROPOSITIONS`, `ISSUE_INTERP_TEMPLATES`)

        Up to 10 issues; the game uses the first `n_issues` from this list.

        Each issue is a continuous policy rate shown to agents as an integer percentage from 0% to 100%.

        {md_table(["#", "Issue Name", "Scale", "Plain-English interpretation template"], issue_rows)}

        ---

        ### Game 3 - All Possible Projects (`PROJECT_NAMES`)

        Up to 10 projects; the game uses the first `m_projects` from this list.

        {md_table(["Index", "Name"], project_rows)}
        """
    )


def build_item_allocation_state() -> dict:
    items = [{"name": name} for name in ItemAllocationGame.ITEM_NAMES[:5]]
    return {
        "items": items,
        "agent_preferences": {
            "Agent_1": [9, 5, 3, 7, 6],
            "Agent_2": [4, 8, 6, 2, 5],
        },
        "cosine_similarities": {},
        "game_type": "item_allocation",
    }


def build_diplomatic_state() -> dict:
    issues = DiplomaticTreatyGame.ISSUE_NAMES[:5]
    return {
        "issues": issues,
        "n_issues": len(issues),
        "agent_positions": {
            "Agent_1": [0.82, 0.14, 0.65, 0.49, 0.77],
            "Agent_2": [0.28, 0.61, 0.38, 0.56, 0.33],
        },
        "agent_weights": {
            "Agent_1": [0.31, 0.04, 0.23, 0.20, 0.22],
            "Agent_2": [0.18, 0.24, 0.21, 0.17, 0.20],
        },
        "parameters": {"rho": 0.0, "theta": 0.5},
        "game_type": "diplomatic_treaty",
    }


def build_cofunding_base_state(pledge_mode: str = "individual") -> dict:
    projects = [{"name": name, "cost": cost} for name, cost in zip(CoFundingGame.PROJECT_NAMES[:5], [28, 35, 22, 41, 20])]
    return {
        "projects": projects,
        "m_projects": len(projects),
        "project_costs": [28, 35, 22, 41, 20],
        "total_cost": 146,
        "agent_budgets": {"Agent_1": 37, "Agent_2": 37},
        "total_budget": 74,
        "agent_valuations": {
            "Agent_1": [42, 28, 13, 10, 7],
            "Agent_2": [18, 35, 20, 15, 12],
        },
        "parameters": {
            "alpha": 0.5,
            "sigma": 0.5,
            "budget_ratio": 0.5,
            "discussion_transparency": "own",
            "enable_commit_vote": True,
            "enable_time_discount": True,
            "gamma_discount": 0.9,
        },
        "game_type": "co_funding",
        "pledge_mode": pledge_mode,
        "round_pledges": [],
        "current_pledges": {},
        "aggregate_totals": [0.0] * len(projects),
        "funded_projects": [],
        "joint_plans": {},
        "accepted_proposal": None,
    }


def build_common_reflection_result() -> dict:
    return {
        "consensus_reached": False,
        "votes_by_proposal": {
            1: {"accept": 1, "reject": 1},
            2: {"accept": 0, "reject": 2},
        },
    }


def render_game1() -> str:
    game = ItemAllocationGame(
        ItemAllocationConfig(
            n_agents=2,
            t_rounds=3,
            gamma_discount=0.9,
            m_items=5,
            random_seed=0,
        )
    )
    state = build_item_allocation_state()
    voting_proposals = [
        {
            "proposal_number": 1,
            "allocation": {"Agent_1": [0, 3], "Agent_2": [1, 2, 4]},
            "proposed_by": "Agent_1",
        },
        {
            "proposal_number": 2,
            "allocation": {"Agent_1": [1, 4], "Agent_2": [0, 2, 3]},
            "proposed_by": "Agent_2",
        },
    ]

    return normalize_section(
        f"""
        ## Game 1: Item Allocation

        Protocol: `propose_and_vote`
        Runtime structure: one-time setup (`rules + private preferences`), then each round:
        `Discussion -> Private Thinking -> Proposal -> Voting -> Reflection`

        ### 1.1 Setup Prompt (combined rules + private preferences)

        Source: `ItemAllocationGame.get_combined_setup_prompt()`

        {code_block(game.get_combined_setup_prompt("Agent_1", deepcopy(state)))}

        ### 1.2 Preference Assignment Prompt (merged into 1.1)

        Game 1 still implements `get_game_rules_prompt()` and `get_preference_assignment_prompt()`,
        but the runtime uses `uses_combined_setup_phase() == True`, so the setup prompt above is the
        actual live prompt path.

        ### 1.3 Discussion Prompt

        #### Case A - Round 1, first speaker

        {code_block(game.get_discussion_prompt("Agent_1", deepcopy(state), 1, 3, []))}

        #### Case B - Round 1, responding after another agent has spoken

        {code_block(game.get_discussion_prompt("Agent_1", deepcopy(state), 1, 3, ['**Agent_2**: I value Jewel most and can be flexible on Quill.']))}

        #### Case C - Later round, first speaker

        {code_block(game.get_discussion_prompt("Agent_1", deepcopy(state), 2, 3, []))}

        #### Case D - Later round, responding after another agent has spoken

        {code_block(game.get_discussion_prompt("Agent_1", deepcopy(state), 2, 3, ['**Agent_2**: I still want Jewel most, but I might compromise on Pencil.']))}

        ### 1.4 Private Thinking Prompt

        Source: `ItemAllocationGame.get_thinking_prompt()`

        {code_block(game.get_thinking_prompt("Agent_1", deepcopy(state), 1, 3, []))}

        ### 1.5 Proposal Prompt

        Source: `ItemAllocationGame.get_proposal_prompt()`

        {code_block(game.get_proposal_prompt("Agent_1", deepcopy(state), 1, ["Agent_1", "Agent_2"]))}

        ### 1.6 Voting Prompt

        Source: `ItemAllocationGame.get_batch_voting_prompt()`

        {code_block(game.get_batch_voting_prompt("Agent_1", voting_proposals, deepcopy(state), 2))}

        ### 1.7 Reflection Prompt

        Source: default `GameEnvironment.get_reflection_prompt()` in `game_environments/base.py`

        {code_block(game.get_reflection_prompt("Agent_1", deepcopy(state), 2, 3, build_common_reflection_result()))}
        """
    )


def render_game2() -> str:
    game = DiplomaticTreatyGame(
        DiplomaticTreatyConfig(
            n_agents=2,
            t_rounds=3,
            gamma_discount=0.9,
            n_issues=5,
            rho=0.0,
            theta=0.5,
            random_seed=0,
        )
    )
    state = build_diplomatic_state()
    voting_proposals = [
        {
            "proposal_number": 1,
            "agreement": [0.65, 0.20, 0.55, 0.50, 0.70],
            "proposed_by": "Agent_1",
        },
        {
            "proposal_number": 2,
            "agreement": [0.45, 0.60, 0.35, 0.40, 0.50],
            "proposed_by": "Agent_2",
        },
    ]

    return normalize_section(
        f"""
        ## Game 2: Diplomatic Treaty

        Protocol: `propose_and_vote`
        Runtime structure: one-time setup (`rules + private preferences`), then each round:
        `Discussion -> Private Thinking -> Proposal -> Voting -> Reflection`

        ### 2.1 Setup Prompt (combined rules + private preferences)

        Source: `DiplomaticTreatyGame.get_combined_setup_prompt()`

        {code_block(game.get_combined_setup_prompt("Agent_1", deepcopy(state)))}

        ### 2.2 Preference Assignment Prompt (merged into 2.1)

        Game 2 still implements `get_game_rules_prompt()` and `get_preference_assignment_prompt()`,
        but the runtime uses `uses_combined_setup_phase() == True`, so the combined setup prompt above
        is the actual live prompt path.

        ### 2.3 Discussion Prompt

        #### Case A - Round 1, first speaker

        {code_block(game.get_discussion_prompt("Agent_1", deepcopy(state), 1, 3, []))}

        #### Case B - Round 1, responding after another delegation has spoken

        {code_block(game.get_discussion_prompt("Agent_1", deepcopy(state), 1, 3, ['**Agent_2**: AI chip export controls and warhead reduction are my core concerns. I can be flexible on the emergency stockpile.']))}

        #### Case C - Later round, first speaker

        {code_block(game.get_discussion_prompt("Agent_1", deepcopy(state), 2, 3, []))}

        #### Case D - Later round, responding after another delegation has spoken

        {code_block(game.get_discussion_prompt("Agent_1", deepcopy(state), 2, 3, ['**Agent_2**: AI chip export controls still matter most to me, but I may have some flexibility on the emergency stockpile.']))}

        ### 2.4 Private Thinking Prompt

        Source: `DiplomaticTreatyGame.get_thinking_prompt()`

        {code_block(game.get_thinking_prompt("Agent_1", deepcopy(state), 1, 3, ['**Agent_2**: AI chip export controls are non-negotiable for me.']))}

        ### 2.5 Proposal Prompt

        Source: `DiplomaticTreatyGame.get_proposal_prompt()`

        {code_block(game.get_proposal_prompt("Agent_1", deepcopy(state), 1, ["Agent_1", "Agent_2"]))}

        ### 2.6 Voting Prompt

        Source: `DiplomaticTreatyGame.get_batch_voting_prompt()`

        {code_block(game.get_batch_voting_prompt("Agent_1", voting_proposals, deepcopy(state), 2))}

        ### 2.7 Reflection Prompt

        Source: default `GameEnvironment.get_reflection_prompt()` in `game_environments/base.py`

        {code_block(game.get_reflection_prompt("Agent_1", deepcopy(state), 2, 3, build_common_reflection_result()))}
        """
    )


def build_cofunding_joint_proposal() -> tuple[CoFundingGame, dict, dict]:
    game = CoFundingGame(
        CoFundingConfig(
            n_agents=2,
            t_rounds=5,
            gamma_discount=0.9,
            m_projects=5,
            alpha=0.5,
            sigma=0.5,
            pledge_mode="individual",
            discussion_transparency="own",
            enable_commit_vote=True,
            enable_time_discount=True,
            random_seed=0,
        )
    )
    state = build_cofunding_base_state(pledge_mode="individual")
    proposal = game.prepare_proposals_for_voting(
        [
            {
                "contributions": [18.0, 9.0, 0.0, 0.0, 10.0],
                "reasoning": "Focus on the two strongest projects for me.",
                "proposed_by": "Agent_1",
            },
            {
                "contributions": [10.0, 15.0, 0.0, 0.0, 6.0],
                "reasoning": "Concentrate support where the group is closest.",
                "proposed_by": "Agent_2",
            },
        ],
        state,
        round_num=2,
    )[0]
    return game, state, proposal


def render_game3() -> str:
    game_aggregate = CoFundingGame(
        CoFundingConfig(
            n_agents=2,
            t_rounds=5,
            gamma_discount=0.9,
            m_projects=5,
            alpha=0.5,
            sigma=0.5,
            pledge_mode="individual",
            discussion_transparency="aggregate",
            enable_commit_vote=True,
            enable_time_discount=True,
            random_seed=0,
        )
    )
    aggregate_state = build_cofunding_base_state(pledge_mode="individual")

    game_own, live_state, joint_proposal = build_cofunding_joint_proposal()

    game_joint = CoFundingGame(
        CoFundingConfig(
            n_agents=2,
            t_rounds=5,
            gamma_discount=0.9,
            m_projects=5,
            alpha=0.5,
            sigma=0.5,
            pledge_mode="joint",
            discussion_transparency="own",
            enable_commit_vote=True,
            enable_time_discount=True,
            random_seed=0,
        )
    )
    joint_state = build_cofunding_base_state(pledge_mode="joint")
    game_joint.update_game_state_with_pledges(
        joint_state,
        deepcopy(live_state["current_pledges"]),
    )

    game_full = CoFundingGame(
        CoFundingConfig(
            n_agents=2,
            t_rounds=5,
            gamma_discount=0.9,
            m_projects=5,
            alpha=0.5,
            sigma=0.5,
            pledge_mode="individual",
            discussion_transparency="full",
            enable_commit_vote=True,
            enable_time_discount=True,
            random_seed=0,
        )
    )
    full_state = build_cofunding_base_state(pledge_mode="individual")
    game_full.update_game_state_with_pledges(
        full_state,
        deepcopy(live_state["current_pledges"]),
    )

    reflection_result = {"consensus_reached": False}
    case_b_history = ["**Agent_2**: Let's focus on Market Street Protected Bike Lane and Parkside Adventure Playground."]
    thinking_history = ["**Agent_2**: Let's concentrate on the projects that are already close."]

    return normalize_section(
        f"""
        ## Game 3: Co-Funding / Participatory Budgeting

        Protocol: `propose_and_vote`
        Runtime structure: one-time setup (`rules + private preferences`), then each round:
        `Discussion -> Private Thinking -> Proposal -> Voting -> Reflection`

        Current runtime note: `CoFundingGame.get_protocol_type()` returns `propose_and_vote`.
        The legacy feedback / commit-vote helper prompts still exist in `co_funding.py`; those are included in an appendix below.

        ### 3.1 Setup Prompt (combined rules + private preferences)

        Source: `CoFundingGame.get_combined_setup_prompt()`

        {code_block(game_own.get_combined_setup_prompt("Agent_1", build_cofunding_base_state(pledge_mode="individual")))}

        ### 3.2 Preference Assignment Prompt (merged into 3.1)

        Game 3 still implements `get_game_rules_prompt()` and `get_preference_assignment_prompt()`,
        but the runtime uses `uses_combined_setup_phase() == True`, so the combined setup prompt above
        is the actual live prompt path.

        ### 3.3 Discussion Prompt

        #### Case A - Round 1, first speaker (`discussion_transparency="aggregate"`)

        {code_block(game_aggregate.get_discussion_prompt("Agent_1", deepcopy(aggregate_state), 1, 5, []))}

        #### Case B - Round 2, responding after another participant has spoken (`discussion_transparency="own"`)

        {code_block(game_own.get_discussion_prompt("Agent_1", deepcopy(live_state), 2, 5, case_b_history))}

        #### Case C - Later round, first speaker (`discussion_transparency="own"`)

        {code_block(game_own.get_discussion_prompt("Agent_1", deepcopy(live_state), 3, 5, []))}

        #### Full transparency addendum (`discussion_transparency="full"`)

        {code_block(game_full.get_discussion_prompt("Agent_1", deepcopy(full_state), 2, 5, case_b_history))}

        ### 3.4 Private Thinking Prompt

        Source: `CoFundingGame.get_thinking_prompt()`

        {code_block(game_own.get_thinking_prompt("Agent_1", deepcopy(live_state), 2, 5, thinking_history))}

        ### 3.5 Proposal Prompt

        #### Individual mode (current default)

        {code_block(game_own.get_proposal_prompt("Agent_1", deepcopy(live_state), 2, ["Agent_1", "Agent_2"]))}

        #### Joint mode (legacy helper retained in code)

        {code_block(game_joint.get_proposal_prompt("Agent_1", deepcopy(joint_state), 2, ["Agent_1", "Agent_2"]))}

        ### 3.6 Voting Prompt

        Source: `CoFundingGame.get_voting_prompt()`

        {code_block(game_own.get_voting_prompt("Agent_1", deepcopy(joint_proposal), deepcopy(live_state), 2))}

        ### 3.7 Reflection Prompt

        Source: `CoFundingGame.get_reflection_prompt()`

        {code_block(game_own.get_reflection_prompt("Agent_1", deepcopy(live_state), 2, 5, reflection_result))}

        ## Appendix: Legacy Co-Funding Helper Prompts

        These helpers remain in `game_environments/co_funding.py` for the legacy `talk_pledge_revise`
        flow wired in `strong_models_experiment/experiment.py`, but they are not used by the current
        `propose_and_vote` Game 3 runtime.

        ### A.1 Feedback Prompt

        Source: `CoFundingGame.get_feedback_prompt()`

        {code_block(game_own.get_feedback_prompt("Agent_1", deepcopy(live_state)))}

        ### A.2 Commit Vote Prompt

        Source: `CoFundingGame.get_commit_vote_prompt()`

        {code_block(game_own.get_commit_vote_prompt("Agent_1", deepcopy(live_state), 2, 5))}
        """
    )


def render_reasoning_addendum() -> str:
    generic = dedent(
        """
        **REASONING DEPTH:** Please use approximately 2000 tokens in your internal reasoning before outputting your response for this stage.
        """
    ).strip()
    thinking = dedent(
        """
        **REASONING DEPTH:**
        Please use approximately 2000 tokens in your internal reasoning before outputting your response for this stage.
        """
    ).strip()
    return normalize_section(
        f"""
        ## Reasoning Token Budget Addendum

        Many prompt methods accept an optional `reasoning_token_budget`.
        When that argument is provided, the current code appends one of these two suffix styles:

        Generic inline style (discussion / proposal / voting / most reflection prompts):

        {code_block(generic)}

        Thinking-prompt style:

        {code_block(thinking)}
        """
    )


def render_summary_table() -> str:
    return normalize_section(
        f"""
        ## Summary Table

        {md_table(
            ["Game", "Setup path", "Round phases", "Reflection source", "Notes"],
            [
                [
                    "Game 1: Item Allocation",
                    "`get_combined_setup_prompt()`",
                    "`Discussion -> Private Thinking -> Proposal -> Voting -> Reflection`",
                    "`base.py` default",
                    "Separate setup / preference helpers remain in code but are not used at runtime.",
                ],
                [
                    "Game 2: Diplomatic Treaty",
                    "`get_combined_setup_prompt()`",
                    "`Discussion -> Private Thinking -> Proposal -> Voting -> Reflection`",
                    "`base.py` default",
                    "Percent displays are integer percentages throughout the prompt-facing interface.",
                ],
                [
                    "Game 3: Co-Funding",
                    "`get_combined_setup_prompt()`",
                    "`Discussion -> Private Thinking -> Proposal -> Voting -> Reflection`",
                    "`co_funding.py` custom",
                    "Current runtime uses `propose_and_vote`; legacy feedback / commit-vote helpers are documented in the appendix.",
                ],
            ],
        )}
        """
    )


def build_document() -> str:
    sections = [
        "\n".join(
            [
                "# All Negotiation Game Prompts - Reference",
                "",
                "> Current rendered prompt reference for Games 1, 2, and 3.",
                "> Prompt source files: `game_environments/item_allocation.py`, `game_environments/diplomatic_treaty.py`, `game_environments/co_funding.py`, `game_environments/base.py`",
                "> This file is generated from the live prompt methods plus fixed sample game states so the examples stay readable and stable.",
            ]
        ),
        render_asset_tables(),
        render_game1(),
        render_game2(),
        render_game3(),
        render_reasoning_addendum(),
        render_summary_table(),
    ]
    return "\n\n---\n\n".join(section.strip() for section in sections) + "\n"


def main() -> None:
    OUTPUT_PATH.write_text(build_document(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
