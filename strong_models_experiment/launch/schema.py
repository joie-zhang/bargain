"""Declared inputs and defaults for the local hosted-API protocol."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

PROTOCOL = "public-hosted-v1"
TEAM_PROTOCOL = "public-binding-team-v1"
FAMILIES = ("two-player", "two-player-llama", "homogeneous", "heterogeneous",
            "homogeneous-adversary", "ttc", "team")
GAMES = {"game1": "item_allocation", "game2": "diplomacy", "game3": "co_funding"}
PHASES = ("default", "discussion", "thinking", "proposal", "voting", "reflection")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


@dataclass(frozen=True)
class LaunchRequest:
    preset: str
    game: str = "game1"
    adversary: str | None = None
    model: str | None = None
    agents: int | None = None
    family: str | None = None
    seed: int = 42
    rounds: int = 10
    discussion_turns: int = 2
    discount: float = 0.9
    position: str = "first"
    competition: float | None = None
    rho: float | None = None
    theta: float | None = None
    alpha: float | None = None
    sigma: float | None = None
    stratum: int | None = None
    max_tokens: int | None = None
    timeout: float = 300.0

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.preset not in FAMILIES or self.game not in GAMES:
            raise ValueError("Unknown experiment preset or game")
        for name, lo, hi in (("seed", 0, 2**32 - 1), ("rounds", 1, 100),
                             ("discussion_turns", 1, 20)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        for name in ("discount", "timeout"):
            if not finite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.discount > 1 or self.timeout > 3600:
            raise ValueError("discount must be <= 1 and timeout must be <= 3600")
        if self.position not in ("first", "last"):
            raise ValueError("position must be first or last")
        if self.agents is not None and (type(self.agents) is not int or self.agents not in (2, 4, 6, 8, 10)):
            raise ValueError("agents must be one of 2, 4, 6, 8, 10")
        if self.preset in ("two-player", "two-player-llama", "ttc") and self.agents not in (None, 2):
            raise ValueError("This preset requires two agents")
        needs_adversary = self.preset in ("two-player", "two-player-llama", "homogeneous-adversary")
        if needs_adversary and not self.adversary:
            raise ValueError("--adversary is required")
        if self.adversary is not None and not needs_adversary:
            raise ValueError("--adversary does not apply to this preset")
        if (self.model is not None) != (self.preset == "homogeneous"):
            raise ValueError("--model is required only for homogeneous runs")
        if self.preset == "ttc":
            if self.family not in ("gpt5", "claude", "gemini"):
                raise ValueError("TTC requires --family gpt5, claude, or gemini")
        elif self.family is not None:
            raise ValueError("--family applies only to TTC")
        if self.preset == "team" and self.game != "game1":
            raise ValueError("The team protocol supports Game 1 only")
        if self.preset in ("homogeneous", "heterogeneous") and self.position != "first":
            raise ValueError("--position applies only to runs with an adversary role")
        if self.stratum is not None and (self.preset != "heterogeneous" or type(self.stratum) is not int or self.stratum not in range(5)):
            raise ValueError("--stratum must be 0..4 and applies only to heterogeneous runs")
        for name, game, lo, hi in (("competition", "game1", 0, 1), ("rho", "game2", -1, 1),
                                   ("theta", "game2", 0, 1), ("alpha", "game3", 0, 1),
                                   ("sigma", "game3", 0, 1)):
            value = getattr(self, name)
            if value is not None and (self.game != game or not finite(value) or not lo <= value <= hi):
                raise ValueError(f"{name} must be in [{lo}, {hi}] and applies only to {game}")
        if self.sigma == 0:
            raise ValueError("sigma must be positive")
        if self.max_tokens is not None and (type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 65536):
            raise ValueError("max_tokens must be an integer in [1, 65536]")
