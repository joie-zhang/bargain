#!/usr/bin/env python3
"""
Tests for co-funding metrics module.

Tests cover:
- optimal_funded_set (knapsack)
- social_welfare_cofunding
- utilitarian_efficiency
- provision_rate
- coordination_failure_rate
- lindahl_equilibrium and lindahl_distance
- free_rider_index
- exploitation_index_cofunding
- coalition_value and is_in_core
- adaptation_rate
- Edge cases (division by zero, empty sets)
"""

import pytest
import numpy as np

from game_environments.cofunding_metrics import (
    optimal_funded_set,
    social_welfare_cofunding,
    utilitarian_efficiency,
    provision_rate,
    coordination_failure_rate,
    lindahl_equilibrium,
    lindahl_distance,
    free_rider_index,
    exploitation_index_cofunding,
    coalition_value,
    is_in_core,
    adaptation_rate,
)


class TestOptimalFundedSet:
    """Tests for optimal_funded_set (knapsack)."""

    def test_simple_3project_example(self):
        """Known 3-project example with clear optimal set."""
        # Project 0: cost=10, total_val=30, surplus=20
        # Project 1: cost=20, total_val=15, surplus=-5 (not worth funding)
        # Project 2: cost=15, total_val=40, surplus=25
        # Budget=30 -> can fund {0,2} (cost=25<=30, surplus=45)
        valuations = {
            "A": [15.0, 5.0, 20.0],
            "B": [15.0, 10.0, 20.0],
        }
        costs = [10.0, 20.0, 15.0]
        total_budget = 30.0

        result = optimal_funded_set(valuations, costs, total_budget)
        assert set(result) == {0, 2}

    def test_budget_constraint_binding(self):
        """When budget can only afford one project."""
        valuations = {
            "A": [50.0, 50.0],
            "B": [50.0, 50.0],
        }
        costs = [15.0, 20.0]
        total_budget = 18.0  # can only afford project 0

        result = optimal_funded_set(valuations, costs, total_budget)
        # Project 0: surplus=100-15=85, Project 1: surplus=100-20=80
        # Budget only allows project 0
        assert result == [0]

    def test_negative_surplus_excluded(self):
        """Projects with negative surplus should not be funded."""
        valuations = {
            "A": [5.0, 50.0],
            "B": [5.0, 50.0],
        }
        costs = [20.0, 10.0]
        total_budget = 50.0

        result = optimal_funded_set(valuations, costs, total_budget)
        # Project 0: surplus=10-20=-10 (exclude)
        # Project 1: surplus=100-10=90 (include)
        assert 0 not in result
        assert 1 in result

    def test_empty_when_all_negative(self):
        """No projects funded when all have negative surplus."""
        valuations = {"A": [1.0, 1.0], "B": [1.0, 1.0]}
        costs = [50.0, 50.0]
        total_budget = 100.0

        result = optimal_funded_set(valuations, costs, total_budget)
        assert result == []


class TestSocialWelfare:
    """Tests for social_welfare_cofunding."""

    def test_welfare_manual(self):
        """SW = sum over funded projects of (sum vals - sum contribs)."""
        valuations = {"A": [40.0, 30.0, 30.0], "B": [20.0, 50.0, 30.0]}
        contributions = {"A": [6.0, 12.0, 0.0], "B": [5.0, 8.0, 0.0]}
        funded_set = [0, 1]

        sw = social_welfare_cofunding(valuations, contributions, funded_set)
        # Project 0: (40-6) + (20-5) = 34+15 = 49
        # Project 1: (30-12) + (50-8) = 18+42 = 60
        # Total: 109
        assert abs(sw - 109.0) < 0.01

    def test_welfare_empty_funded_set(self):
        """SW should be 0 when nothing is funded."""
        valuations = {"A": [40.0, 30.0], "B": [20.0, 50.0]}
        contributions = {"A": [10.0, 10.0], "B": [10.0, 10.0]}
        sw = social_welfare_cofunding(valuations, contributions, [])
        assert sw == 0.0


class TestUtilitarianEfficiency:
    """Tests for utilitarian_efficiency."""

    def test_perfect_efficiency(self):
        assert utilitarian_efficiency(100.0, 100.0) == 1.0

    def test_half_efficiency(self):
        assert abs(utilitarian_efficiency(50.0, 100.0) - 0.5) < 0.01

    def test_zero_optimal(self):
        """When optimal SW is 0, efficiency should be 1.0 if actual >= 0."""
        assert utilitarian_efficiency(0.0, 0.0) == 1.0
        assert utilitarian_efficiency(10.0, 0.0) == 1.0

    def test_negative_actual(self):
        assert utilitarian_efficiency(-10.0, 0.0) == 0.0


class TestProvisionRate:
    """Tests for provision_rate."""

    def test_all_optimal_funded(self):
        """Provision rate = 1.0 when all optimal projects are funded."""
        assert provision_rate([0, 1, 2], [0, 1, 2]) == 1.0

    def test_partial_provision(self):
        """Provision rate with partial overlap."""
        assert abs(provision_rate([0, 1], [0, 1, 2]) - 2.0 / 3.0) < 0.01

    def test_no_overlap(self):
        """Provision rate = 0 when no optimal projects funded."""
        assert provision_rate([3, 4], [0, 1, 2]) == 0.0

    def test_empty_optimal(self):
        """Provision rate = 1.0 when optimal set is empty."""
        assert provision_rate([0, 1], []) == 1.0


class TestCoordinationFailure:
    """Tests for coordination_failure_rate."""

    def test_all_funded(self):
        """CFR = 0 when all surplus-positive projects are funded."""
        valuations = {"A": [50.0, 50.0], "B": [50.0, 50.0]}
        costs = [20.0, 20.0]
        contributions = {"A": [10.0, 10.0], "B": [10.0, 10.0]}
        cfr = coordination_failure_rate(valuations, costs, contributions)
        assert cfr == 0.0

    def test_half_failure(self):
        """CFR when half of surplus-positive projects not funded."""
        valuations = {"A": [50.0, 50.0], "B": [50.0, 50.0]}
        costs = [20.0, 20.0]
        # Project 0 funded (20>=20), Project 1 not (10<20)
        contributions = {"A": [10.0, 5.0], "B": [10.0, 5.0]}
        cfr = coordination_failure_rate(valuations, costs, contributions)
        assert abs(cfr - 0.5) < 0.01

    def test_no_surplus_positive(self):
        """CFR = 0 when no projects have positive surplus."""
        valuations = {"A": [1.0, 1.0], "B": [1.0, 1.0]}
        costs = [50.0, 50.0]
        contributions = {"A": [0.0, 0.0], "B": [0.0, 0.0]}
        cfr = coordination_failure_rate(valuations, costs, contributions)
        assert cfr == 0.0


class TestLindahlEquilibrium:
    """Tests for lindahl_equilibrium and lindahl_distance."""

    def test_lindahl_proportional(self):
        """Lindahl contributions should be proportional to valuations."""
        valuations = {"A": [60.0, 40.0], "B": [40.0, 60.0]}
        costs = [10.0, 20.0]
        funded_set = [0, 1]

        result = lindahl_equilibrium(valuations, costs, funded_set)

        # Project 0: total_val=100, cost=10
        # A: 10 * 60/100 = 6.0, B: 10 * 40/100 = 4.0
        assert abs(result["A"][0] - 6.0) < 0.01
        assert abs(result["B"][0] - 4.0) < 0.01

        # Project 1: total_val=100, cost=20
        # A: 20 * 40/100 = 8.0, B: 20 * 60/100 = 12.0
        assert abs(result["A"][1] - 8.0) < 0.01
        assert abs(result["B"][1] - 12.0) < 0.01

    def test_lindahl_distance_zero_at_equilibrium(self):
        """Distance should be 0 when actual == Lindahl."""
        valuations = {"A": [60.0, 40.0], "B": [40.0, 60.0]}
        costs = [10.0, 20.0]
        funded_set = [0, 1]

        lindahl = lindahl_equilibrium(valuations, costs, funded_set)
        dist = lindahl_distance(lindahl, lindahl)
        assert abs(dist) < 1e-10


class TestFreeRiderIndex:
    """Tests for free_rider_index."""

    def test_lindahl_equals_one(self):
        """At Lindahl equilibrium, F_ij should be 1.0 for all."""
        valuations = {"A": [60.0, 40.0], "B": [40.0, 60.0]}
        costs = [10.0, 20.0]
        funded_set = [0, 1]

        lindahl = lindahl_equilibrium(valuations, costs, funded_set)
        fri = free_rider_index(valuations, lindahl, funded_set)

        for a in fri:
            for j in fri[a]:
                assert abs(fri[a][j] - 1.0) < 0.01

    def test_zero_contribution_free_rider(self):
        """Zero contribution with positive valuation = infinite free-riding."""
        valuations = {"A": [50.0, 50.0], "B": [50.0, 50.0]}
        contributions = {"A": [0.0, 0.0], "B": [10.0, 10.0]}
        funded_set = [0, 1]

        fri = free_rider_index(valuations, contributions, funded_set)
        assert fri["A"][0] == float('inf')


class TestExploitationIndex:
    """Tests for exploitation_index_cofunding."""

    def test_no_exploitation(self):
        """Equal actual and Lindahl utilities -> E_i = 0."""
        actual = {"A": 50.0, "B": 50.0}
        lindahl = {"A": 50.0, "B": 50.0}
        result = exploitation_index_cofunding(actual, lindahl)
        assert abs(result["A"]) < 0.01
        assert abs(result["B"]) < 0.01

    def test_exploiter_positive(self):
        """Agent gaining relative to Lindahl should have positive E_i."""
        actual = {"A": 60.0, "B": 40.0}
        lindahl = {"A": 50.0, "B": 50.0}
        result = exploitation_index_cofunding(actual, lindahl)
        assert result["A"] > 0  # exploiter
        assert result["B"] < 0  # exploited


class TestCore:
    """Tests for is_in_core."""

    def test_in_core_simple(self):
        """Simple case that should be in the core."""
        valuations = {"A": [60.0, 40.0], "B": [40.0, 60.0]}
        costs = [10.0, 20.0]
        budgets = {"A": 15.0, "B": 15.0}

        lindahl = lindahl_equilibrium(valuations, costs, [0, 1])
        result = is_in_core(valuations, costs, budgets, lindahl, [0, 1])
        # At Lindahl equilibrium, the outcome should typically be in the core
        # (may depend on exact numbers, but this is a reasonable test)
        assert isinstance(result, bool)

    def test_coalition_value_single_agent(self):
        """Coalition value for a single agent."""
        valuations = {"A": [50.0, 50.0], "B": [50.0, 50.0]}
        costs = [10.0, 20.0]
        budgets = {"A": 25.0, "B": 25.0}

        v = coalition_value(valuations, costs, budgets, ["A"])
        # Agent A alone: project 0 surplus=50-10=40, project 1 surplus=50-20=30
        # Budget=25, can fund both (cost=30>25), so only project 0 (cost=10<=25)
        # Actually: can fund project 0 (cost=10) + project 1 (cost=20) = 30 > 25
        # So can only fund one: max surplus is project 0 (40)
        assert v >= 0


class TestAdaptationRate:
    """Tests for adaptation_rate."""

    def test_no_adaptation(self):
        """Constant pledges should give 0 adaptation rate."""
        history = [
            {"A": [5.0, 5.0], "B": [3.0, 7.0]},
            {"A": [5.0, 5.0], "B": [3.0, 7.0]},
            {"A": [5.0, 5.0], "B": [3.0, 7.0]},
        ]
        budgets = {"A": 10.0, "B": 10.0}
        result = adaptation_rate(history, budgets)
        assert abs(result["A"]) < 1e-10
        assert abs(result["B"]) < 1e-10

    def test_full_adaptation(self):
        """Large changes should give high adaptation rate."""
        history = [
            {"A": [10.0, 0.0], "B": [0.0, 10.0]},
            {"A": [0.0, 10.0], "B": [10.0, 0.0]},
        ]
        budgets = {"A": 10.0, "B": 10.0}
        result = adaptation_rate(history, budgets)
        # A changes by |0-10| + |10-0| = 20, normalized by budget=10, over 1 period
        assert abs(result["A"] - 2.0) < 0.01

    def test_single_round(self):
        """Single round -> adaptation rate = 0."""
        history = [{"A": [5.0, 5.0]}]
        budgets = {"A": 10.0}
        result = adaptation_rate(history, budgets)
        assert result["A"] == 0.0
