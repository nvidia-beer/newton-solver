# SPDX-License-Identifier: Apache-2.0
"""Particle integration for the template's minimal runnable example."""

import warp as wp

from .collision import project_plane


@wp.kernel
def integrate(
    q: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    gravity: wp.array[wp.vec3],
    world: wp.array[int],
    radius: wp.array[float],
    collide: bool,
    plane_height: float,
    restitution: float,
    dt: float,
    q_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        velocity = v[i] + dt * (gravity[wp.max(world[i], 0)] + inv_mass[i] * force[i])
        position = q[i] + dt * velocity
        if collide:
            position, velocity = project_plane(position, velocity, radius[i], plane_height, restitution)
        q_out[i] = position
        v_out[i] = velocity
    else:
        q_out[i] = q[i]
        v_out[i] = v[i]
