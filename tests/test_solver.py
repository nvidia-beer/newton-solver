# SPDX-License-Identifier: Apache-2.0
"""Forward mechanics and multi-step derivatives against independent references."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver import PlaneCollider, Solver
from newton_solver.examples.example_diffsim_optimize import optimize_velocity


@wp.kernel
def _loss(q: wp.array[wp.vec3], v: wp.array[wp.vec3], result: wp.array[float]):
    delta = q[0] - wp.vec3(0.2, -0.1, 0.3)
    result[0] = wp.dot(delta, delta) + wp.dot(v[0], wp.vec3(0.1, -0.2, 0.3))


class TestSolver(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        self.scope = wp.ScopedDevice(self.device)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def model(self, *, mass=2.0, gravity=(0.0, 0.0, -9.81)):
        builder = newton.ModelBuilder(gravity=gravity)
        builder.add_particle(pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), mass=mass, radius=0.05)
        return builder.finalize(requires_grad=True)

    def rollout(self, solver, q, velocity, forces, *, dt=0.04, gradients=False):
        states = [solver.model.state(requires_grad=gradients) for _ in range(len(forces) + 1)]
        states[0].particle_q.assign(np.asarray([q], dtype=np.float32))
        states[0].particle_qd.assign(np.asarray([velocity], dtype=np.float32))
        for state, force in zip(states[:-1], forces, strict=True):
            state.particle_f.assign(np.asarray([force], dtype=np.float32))
        loss = wp.zeros(1, dtype=float, requires_grad=gradients)
        tape = wp.Tape()
        with tape:
            for index in range(len(forces)):
                solver.step(states[index], states[index + 1], None, None, dt)
            wp.launch(_loss, 1, inputs=[states[-1].particle_q, states[-1].particle_qd], outputs=[loss])
        value = float(loss.numpy()[0])
        derivative = None
        if gradients:
            tape.backward(loss)
            derivative = np.concatenate(
                [states[0].particle_q.grad.numpy().ravel(), states[0].particle_qd.grad.numpy().ravel()]
                + [state.particle_f.grad.numpy().ravel() for state in states[:-1]]
            )
        return value, derivative, states

    def test_constant_acceleration_and_input_preservation(self):
        model = self.model()
        q, v, f = np.array([0.1, -0.2, 1.0]), np.array([0.3, 0.1, -0.2]), np.array([2.0, -4.0, 0.6])
        count, dt = 5, 0.04
        _, _, states = self.rollout(Solver(model), q, v, np.tile(f, (count, 1)), dt=dt)
        acceleration = np.array([0.0, 0.0, -9.81]) + f / 2.0
        for index, state in enumerate(states):
            expected_q = q + index * dt * v + dt**2 * index * (index + 1) / 2 * acceleration
            np.testing.assert_allclose(state.particle_q.numpy()[0], expected_q, atol=2e-7)
            np.testing.assert_allclose(state.particle_qd.numpy()[0], v + index * dt * acceleration, atol=2e-7)
        self.assertEqual(len({state.particle_q.ptr for state in states}), count + 1)

    def test_pinned_particle_and_plane_restitution(self):
        for mass in (0.0, 2.0):
            with self.subTest(mass=mass):
                model = self.model(mass=mass)
                _, _, states = self.rollout(
                    Solver(model, collider=PlaneCollider(restitution=0.5)),
                    [0.0, 0.0, 0.06],
                    [0.0, 0.0, -1.0],
                    [[0.0, 0.0, 0.0]],
                    dt=0.1,
                )
                expected_q = 0.05 if mass else 0.06
                expected_v = 0.9905 if mass else -1.0
                self.assertAlmostEqual(float(states[-1].particle_q.numpy()[0, 2]), expected_q, places=6)
                self.assertAlmostEqual(float(states[-1].particle_qd.numpy()[0, 2]), expected_v, places=6)

    def check_gradient(self, *, contact):
        solver = Solver(self.model(), collider=PlaneCollider(restitution=0.5) if contact else None)
        q = np.array([0.1, -0.15, 0.06 if contact else 1.0])
        velocity = np.array([0.2, -0.1, -1.0 if contact else 0.3])
        forces = np.array([[0.2, 0.1, -0.3], [0.0, -0.2, 0.4], [-0.1, 0.3, 0.2]])
        parameters = np.concatenate([q, velocity, forces.ravel()])

        def evaluate(values, gradients=False):
            return self.rollout(solver, values[:3], values[3:6], values[6:].reshape(-1, 3), gradients=gradients)

        value, analytical, states = evaluate(parameters, True)
        epsilon = 2e-3
        numerical = np.empty_like(parameters)
        for index in range(len(parameters)):
            delta = np.zeros_like(parameters)
            delta[index] = epsilon
            numerical[index] = (evaluate(parameters + delta)[0] - evaluate(parameters - delta)[0]) / (2 * epsilon)
        np.testing.assert_allclose(analytical, numerical, rtol=4e-3, atol=2e-5)
        if contact:
            # First impact remains strictly active under every perturbation;
            # this does not test derivatives through a changing event schedule.
            self.assertAlmostEqual(float(states[1].particle_q.numpy()[0, 2]), 0.05, places=6)
            self.assertGreater(float(states[2].particle_q.numpy()[0, 2]), 0.055)
        snapshots = [state.particle_q.numpy().copy() for state in states]
        repeated_value, repeated_gradient, _ = evaluate(parameters, True)
        self.assertEqual(value, repeated_value)
        np.testing.assert_array_equal(analytical, repeated_gradient)
        for state, snapshot in zip(states, snapshots, strict=True):
            np.testing.assert_array_equal(state.particle_q.numpy(), snapshot)

    def test_multistep_state_and_force_gradients(self):
        self.check_gradient(contact=False)

    def test_multistep_active_contact_gradient_and_fresh_tape(self):
        self.check_gradient(contact=True)

    def test_initial_velocity_optimization(self):
        result = optimize_velocity(iterations=20, device=self.device)
        self.assertLess(result["final_loss_m2"], 1e-10)
        self.assertLess(result["final_loss_m2"], result["initial_loss_m2"] * 1e-8)
        # Semi-implicit Euler's independent closed-form terminal velocity fit.
        expected_velocity = np.array([1.2, 0.4, 9.81 * 0.025 * 21 / 2])
        np.testing.assert_allclose(result["initial_velocity_m_s"], expected_velocity, atol=8e-6)

    def test_invalid_step_and_public_observables(self):
        model = self.model()
        solver = Solver(model)
        a, b = model.state(), model.state()
        for dt in (0.0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                solver.step(a, b, None, None, dt)
        with self.assertRaises(ValueError):
            solver.step(a, a, None, None, 0.01)
        solver.step(a, b, None, None, 0.01, observables=solver.observables([]))
        self.assertEqual(solver.supported_observables, frozenset())


if __name__ == "__main__":
    unittest.main()
