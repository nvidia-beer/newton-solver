# SPDX-License-Identifier: Apache-2.0
"""Fit initial velocity with Warp autodiff through a small free-flight rollout.

Run ``python -m newton_solver.examples diffsim_optimize --device cpu``.
This headless example differentiates explicit particle steps, not an implicit
solver or collision-event times.
"""

import argparse
import json

import newton
import numpy as np
import warp as wp

from newton_solver import Solver

HEADLESS = True


@wp.kernel
def _target_loss(q: wp.array[wp.vec3], target: wp.vec3, loss: wp.array[float]):
    delta = q[0] - target
    loss[0] = wp.dot(delta, delta)


def optimize_velocity(*, iterations: int = 20, device: str = "cpu") -> dict:
    """Fit a particle's initial velocity [m/s] to a terminal position [m]."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    with wp.ScopedDevice(device):
        steps, dt = 20, 0.025
        target = np.array([0.6, 0.2, 1.0], dtype=np.float32)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_particle(pos=(0.0, 0.0, 1.0), vel=(-0.2, 0.1, 0.3), mass=1.0)
        model = builder.finalize(requires_grad=True)
        solver = Solver(model)
        # Every taped output has its own storage; two-state ping-pong is only
        # appropriate for a forward-only example such as example_drop.
        states = [model.state(requires_grad=True) for _ in range(steps + 1)]
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        history = []
        for iteration in range(iterations + 1):
            loss.zero_()
            tape = wp.Tape()
            with tape:
                for index in range(steps):
                    solver.step(states[index], states[index + 1], None, None, dt)
                wp.launch(_target_loss, 1, inputs=[states[-1].particle_q, wp.vec3(*target)], outputs=[loss])
            value = float(loss.numpy()[0])
            history.append(value)
            if iteration == iterations:
                break
            tape.backward(loss)
            gradient = states[0].particle_qd.grad.numpy().copy()
            velocity = states[0].particle_qd.numpy() - gradient
            tape.zero()
            states[0].particle_qd.assign(velocity)
        return {
            "initial_loss_m2": history[0],
            "final_loss_m2": history[-1],
            "initial_velocity_m_s": states[0].particle_qd.numpy()[0].tolist(),
            "terminal_position_m": states[-1].particle_q.numpy()[0].tolist(),
            "target_m": target.tolist(),
            "history_m2": history,
        }


def create_parser():
    """Use the same optimization options for terminal runs and recorded demonstrations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Warp device, such as cpu or cuda:0.")
    parser.add_argument("--iterations", type=int, default=20)
    return parser


def main():
    args = create_parser().parse_args()
    result = optimize_velocity(iterations=args.iterations, device=args.device)
    print(json.dumps(result, indent=2))
    if not np.isfinite(result["final_loss_m2"]) or result["final_loss_m2"] >= result["initial_loss_m2"]:
        raise RuntimeError("The velocity fit did not reduce its loss")


if __name__ == "__main__":
    main()
