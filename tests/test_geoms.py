from ..geoms import new_sphere
import jax
import jax.numpy as jnp

class TestSpheres():
    sd_origin_sphere = new_sphere()

    def test_sphere_sdf(self):
        distance = self.sd_origin_sphere(jnp.array([3,4,0]))
        assert distance == 4.0