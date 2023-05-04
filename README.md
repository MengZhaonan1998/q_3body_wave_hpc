# QRes3b 
## A Parallel Nonlinear Eigensolver for Computing Resonant States of Quantum Mechanical Three-body Problems
### What is QRes3b?
QRes3b aims at computing resonant states of 1-dimensional quantum mechanical three-body problems. A quantum two/three-body system can be described by a very well-known equation in the quantum world - the Schrödinger equation. In a two-body system, the interaction between two quantum particles conforms to the stationary Schrödinger equation:Cancel changes
$$H\psi=E\psi,$$
where the Hamiltonian operator $H$ is given by 
$$H = -\frac{1}{2}\Delta(x) + V(x).$$
For simplicity we omit the physical constants such as the Planck's constant in the equation. To extract the resonant states from the above eigenvalue problem, one needs to impose some proper boundary conditions (for more details about resonances please refer to https://www.cs.cornell.edu/~bindel/cims/resonant1d/theo2.html):  
$$(\partial_x-ik)\psi(x)=0 \ at \ x=-L,$$ 
$$(\partial_x+ik)\psi(x)=0 \ at \ x=L.$$
The pseudo-spectral discretization of the eigenvalue problem with above boundary conditions leads to a nonlinear(quadratic) eigenvalue problem represented by matrix
$$(K+kC+k^2M)\psi=0.$$
QRes3b implements a parallel iterative eigensolver based on the Jacobi-Davidson algorithm for solving the above quadratic eigenvalue problem. The algorithm follows the "MPI+OpenMP" programming model, where MPI takes care of data communication between processes (distributed memory), and OpenMP leverages multi-core computation within shared memory.

### Trying it out
