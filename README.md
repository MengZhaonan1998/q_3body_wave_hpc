# QRes3b 
## A Parallel Nonlinear Eigensolver for Computing Resonant States of Quantum Mechanical Three-body Problems
### What is QRes3b?
QRes3b aims at computing resonant states of a quantum mechanical three-body problems. A quantum two/three-body system can be described by a very well-known equation in the quantum world - the Schrödinger equation. In a two-body system, the interaction between two quantum particles conforms to the stationary Schrödinger equation:
$$H\psi=E\psi.$$
The Hamiltonian operator $H$ is given by 
$$H = -\frac{1}{2}\Delta(x) + V(x),$$
where $x$ denotes the distance between two particles. For simplicity we omit the physical constants such as the Planck's constant in the equation. To extract the resonant states from the above eigenvalue problem, one needs to impose some proper boundary conditions (for more details about resonances please refer to https://www.cs.cornell.edu/~bindel/cims/resonant1d/theo2.html):
$$E=\frac{1}{2}k^2,$$
$$(\partial_x-ik)\psi(x)=0 \ at \ x=-L,$$ 
$$(\partial_x+ik)\psi(x)=0 \ at \ x=L.$$
The pseudo-spectral discretization of the eigenvalue problem with above boundary conditions leads to a nonlinear(quadratic) eigenvalue problem represented by matrix
$$(K+kC+k^2M)\psi=0.$$
For a three-body (two heavy + one light) problem, we assume there is no interaction between two heavy particles and introduce another coordinate $y$ to represent the relative coordinate between the two heavy particles:
$$[-\frac{\alpha_x}{2}\Delta(x) - \frac{\alpha_y}{2}\Delta(y) + V(x,y)]\psi(x,y)=E\psi(x,y),$$
where x denotes the coordinate of the light particle with respect to the center of mass of the two heavy particles. The discretization of this two-dimensional eigenvalue problem with above outgoing boundary conditions is similar with the case of the two-body system. The only thing which involves more efforts is to compute the Kronecker product:
$$K_x\otimes I_y + I_x\otimes K_y,$$
$$C_x\otimes I_y + I_x\otimes C_y,$$
$$M_x\otimes I_y + I_x\otimes M_y.$$
QRes3b implements a parallel iterative eigensolver based on the Jacobi-Davidson algorithm for solving the above quadratic eigenvalue problem. The algorithm follows the "MPI+OpenMP" programming model, where MPI takes care of data communication between processes (distributed memory), and OpenMP leverages multi-core computation within shared memory.

The parallelization follows the pattern of SIMD(Single Instruction Multiple Data), distributing the workload of linear operations such as axpby, dot, matrix-vector multiplication among processes and threads. The key point of the project is how to compute the multiplication of the tensor structure and vectors efficiently. To compute the linear operation in the form of:
$$T \cdot w = [a_1(C_1 \otimes I_2)+a_2(I_1 \otimes C_2)]\cdot w,$$
we take advantage of dense matrix-matrix products. Let $C_1\in \mathbb{C}^{N_1\times N_1},\ C_2\in \mathbb{C}^{N_2\times N_2},\ w\in \mathbb{C}^{N_1 N_2},$ and $W=reshape(w,N_2,N_1)$ denote the interpretation of $w$ as an $N_2\times N_1$ matrix. Then we have
$$T\cdot w = reshape(a_2 C_2\cdot W+a_1 W\cdot C_1^T, N_1 N_2, 1),  $$
where the reshape operation is used to interpret the resulting $N_2\times N_1$-matrix as a vector of length $N_1 N_2$. Reshape does not incur any data movement, namely it is just a re-interpretation of a vector as a matrix stored in column-major ordering, and vice versa. To implement the operation in parallel, we use a column-wise distribution of $W$ among the MPI processes, while the dense matrices, such as $C_1, \ C_2$, constituting the Hamiltonian are replicated on all processes. The communication involved in transposing the tensors $W$ and $C_1 W^T$, can be overlapped with computations as follows:
1. Transpose the local columns of $W$; 
2. For each column of $W^T$, dispatch a non-blocking ‘gather’ operation;
3. Whenever a gather operation is finished for a local column of $W^T$, apply $C_1$ to that column;
4. The back transpose is then overlapped with the computation of $C_2 W$.


### Trying it out
