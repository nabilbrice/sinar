from typing import NamedTuple, Callable
from jax import Array
import jax
import jax.numpy as jnp
from functools import partial

import equinox as eqx

from .shapes import Shape
from ..rays import normalize

# For an entity collection, the distance field is the combination of all with
# a minimum of some kind.
# This approach loses information.
# Holding a Iterable Tuple of Entity as a scene actually allows
# for generating the minimum distance array and then from that array call
# the appropriate function for the color.
class Entity(NamedTuple):
    shape: Shape
    color: Callable

    def surface_color(self, phase):
        direction = phase[3:6]
        uv = self.shape.uv(phase)
        sn = self.shape.sn(phase)[0:3]
        mu = jnp.vecdot(normalize(direction), normalize(sn))
        return self.color(uv, mu)

class Scene(eqx.Module):
    """A collection of entities over which the rays are cast.
    """
    entities: tuple

    def __init__(self, *entities):
        self.entities = tuple(entities)
    
    def sdf(self, phase):
        return jnp.min(jnp.array(
        [entity.shape.sdf(phase) for entity in self.entities]
        ))

    def sdargmin(self, phase):
        return jnp.argmin(jnp.array(
        [entity.shape.sdf(phase) for entity in self.entities]
    ), axis=-1)

    def surface_color(self, phase):
        return jax.lax.switch(
            self.sdargmin(phase),
            tuple(entity.surface_color for entity in self.entities),
            phase
        )

# The signed distance minimum still must compute the sd function
# for each of the entities in the list.
# By taking the minimum of the distance as the key, the color can be obtained.
def sdmin_scene(entities: list, position: Array):
    return jnp.min(jnp.array(
        [entity.shape.sdf(position) for entity in entities]
        ))

def sdargmin_scene(entities: list, position: Array):
    return jnp.argmin(jnp.array(
        [entity.shape.sdf(position) for entity in entities]
    ), axis=-1)

# For a smoother blending of objects, but it is slower
def sdsmin_scene(shapes: list, position: Array):
    sdistances = jnp.array(
        [entity.sdf(position) for entity in shapes]
        )
    return -jax.nn.logsumexp(-sdistances*16.0)/16
