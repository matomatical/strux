"""
Common example test structs shared across test modules (other bespoke ones
are defined inline where they are used).
"""

from jaxtyping import Array, Bool, Float, Int

import strux


@strux.struct
class Point:
    x: Float[Array, ""]
    y: Float[Array, ""]


@strux.struct
class Environment:
    hero_pos: Int[Array, "2"]
    goal_pos: Int[Array, "2"]
    walls: Bool[Array, "h w"]


@strux.struct
class World:
    env: Environment
    score: Float[Array, ""]
