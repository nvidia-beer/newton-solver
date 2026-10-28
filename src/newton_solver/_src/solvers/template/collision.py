# SPDX-License-Identifier: Apache-2.0
"""An example of collision behavior owned by the external solver package."""

import math

import warp as wp


@wp.func
def project_plane(
    position: wp.vec3,
    velocity: wp.vec3,
    radius: float,
    height: float,
    restitution: float,
) -> tuple[wp.vec3, wp.vec3]:
    if position[2] < height + radius:
        position = wp.vec3(position[0], position[1], height + radius)
        velocity = wp.vec3(velocity[0], velocity[1], -restitution * wp.min(velocity[2], 0.0))
    return position, velocity


class PlaneCollider:
    """Project particles onto a horizontal plane, with normal restitution.

    This discrete contact law is not CCD or a soft-body contact solver. Warp
    differentiates the selected branch; gradients at impact/contact transitions
    are not defined by this example.

    Args:
        height: Plane elevation [m] along world Z.
        restitution: Normal coefficient of restitution, in [0, 1].
    """

    def __init__(self, *, height: float = 0.0, restitution: float = 0.25):
        if not math.isfinite(height):
            raise ValueError("height must be finite")
        if not 0.0 <= restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        self.height = float(height)
        self.restitution = float(restitution)
