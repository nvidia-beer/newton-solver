# SPDX-License-Identifier: Apache-2.0
"""Particles bouncing on a package-owned plane collider."""

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver import PlaneCollider, Solver


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.time = 0.0
        self.dt = 1.0 / 120.0
        builder = newton.ModelBuilder()
        for i in range(6):
            builder.add_particle(pos=(0.12 * i, 0.0, 0.5 + 0.08 * i), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.04)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.solver = Solver(self.model, collider=PlaneCollider(restitution=0.5))
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(wp.vec3(1.2, -2.0, 1.0), pitch=-15, yaw=110)

    def step(self):
        for _ in range(2):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.time += self.dt

    def render(self):
        self.viewer.begin_frame(self.time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        if not np.isfinite(q).all() or q[:, 2].min() < 0.03999:
            raise AssertionError("Invalid particle-plane contact state")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
