# SPDX-License-Identifier: Apache-2.0
"""Public Newton solver interface, with package-owned integration and contact."""

import math

import newton
import warp as wp
from newton.solvers import SolverBase

from .collision import PlaneCollider
from .kernels import integrate


class Solver(SolverBase):
    """Minimal semi-implicit particle solver to replace with your own method.

    Uses native Newton Model/State buffers. Only particles and an optional
    package-owned horizontal plane are supported; no rigid or elastic dynamics.
    Native Newton Contacts are rejected because this solver does not consume them.
    Record rollouts using a different output State for every step. The optional
    contact projection is differentiable only within an unchanged contact branch.
    """

    def __init__(self, model: newton.Model, *, collider: PlaneCollider | None = None):
        super().__init__(model)
        if model.body_count or model.tet_count or model.tri_count or model.spring_count:
            raise ValueError("Solver supports particles only")
        if model.up_axis != newton.Axis.Z:
            raise ValueError("The template plane example requires Z-up")
        self.collider = collider

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance particle positions and velocities by dt [s]."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if state_in is state_out:
            raise ValueError("Use distinct input/output states")
        self.validate_observables(observables, contacts)
        if contacts is not None:
            raise NotImplementedError("Native Contacts are not implemented by this template")
        wp.launch(
            integrate,
            self.model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                self.model.particle_inv_mass,
                self.model.gravity,
                self.model.particle_world,
                self.model.particle_radius,
                self.collider is not None,
                self.collider.height if self.collider is not None else 0.0,
                self.collider.restitution if self.collider is not None else 0.0,
                dt,
                state_out.particle_q,
                state_out.particle_qd,
            ],
            device=self.model.device,
        )
