## Mathematical method

This is a particle integrator starter, not an implementation of a particular
research paper. A generated solver must replace this section with its own
method and primary references.

For particle position $x_n$, velocity $v_n$, mass $m>0$, force $f_n$ and
timestep $h$, the semi-implicit Euler update is

$$
v_{n+1}=v_n+h(g+f_n/m),\qquad x_{n+1}=x_n+h v_{n+1}.
$$

With a horizontal plane, an intersecting particle is projected to the plane
plus its radius; an incoming normal velocity is multiplied by $-e$, where
$e\in[0,1]$ is restitution. The implemented branch is differentiated using
Warp's reverse mode over distinct saved states. Initial state and applied
forces may be differentiated; mass, timestep, plane position and restitution
are fixed. Contact creation and branch switching are nonsmooth.

## Papers

No paper reproduction is claimed for this starter. The implementation uses
the public [Newton SolverBase interface](https://newton-physics.github.io/newton/1.7.0/api/_generated/newton.solvers.SolverBase.html).
Add the implemented paper's DOI or author-hosted manuscript when replacing
the starter method.
