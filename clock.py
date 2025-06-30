import numpy as np
import cirq
import scipy
def count_clock_dimension(number_of_clock_state):
    if number_of_clock_state == 0:
        raise ValueError("The number_of_clock_state must be non zero")
    n_steps = number_of_clock_state+1
    if np.log2(n_steps)%1 == 0:
        n_qubit = np.log2(n_steps)
    else:
        n_qubit = np.floor(np.log2(n_steps))+1
    return n_qubit

def clock_basis(n_qubit,t):
    clock_basis = np.zeros(2**n_qubit)
    clock_basis[t] = 1
    return clock_basis

def H_prop(unitaries):
    T = len(unitaries) #we need log2(T+1) number of qubit for clock |T>
    data_qubit = int(np.log2(len(unitaries[0][0])))
    n_qubit_clock_state = int(count_clock_dimension(T))
    #print(data_qubit)
    #print(n_qubit_clock_state)
    H_prop = np.zeros((2**(data_qubit+n_qubit_clock_state),2**(data_qubit+n_qubit_clock_state)),dtype='complex128')
    for i in range(1,T+1):
        H_prop += np.kron((np.outer(clock_basis(n_qubit_clock_state,i-1),clock_basis(n_qubit_clock_state,i-1))+np.outer(clock_basis(n_qubit_clock_state,i),clock_basis(n_qubit_clock_state,i))),np.eye(2**data_qubit)) #(|t-1><t-1|-|t><t|) otimes I
        # dg2 = np.kron((np.outer(clock_basis(n_qubit_clock_state,i-1),clock_basis(n_qubit_clock_state,i-1))+np.outer(clock_basis(n_qubit_clock_state,i),clock_basis(n_qubit_clock_state,i))),np.eye(2**data_qubit))
        # dg = np.kron(np.outer(clock_basis(n_qubit_clock_state,i),clock_basis(n_qubit_clock_state,i-1)),unitaries[i-1])
        # print(dg.dtype)
        # print(H_prop.dtype)
        
        H_prop += -np.kron(np.outer(clock_basis(n_qubit_clock_state,i),clock_basis(n_qubit_clock_state,i-1)),unitaries[i-1]) #|t><t-1| otimes U_t
        H_prop += -np.kron(np.outer(clock_basis(n_qubit_clock_state,i-1),clock_basis(n_qubit_clock_state,i)),unitaries[i-1].conj().T) #|t-1><t| otimes U_t\dagger
    return 0.5*H_prop
# def H_initial(unitaries):
#     # assuming always starts from |0000000>
#     T = len(unitaries) #we need log2(T+1) number of qubit for clock |T>
#     data_qubit = int(np.log2(len(unitaries[0][0])))
#     n_qubit_clock_state = int(count_clock_dimension(T))
#     zeros_clock_proj = np.outer(clock_basis(n_qubit_clock_state,0),clock_basis(n_qubit_clock_state, 0))
#     zero_data_proj = np.eye(2**data_qubit)-np.outer(clock_basis(data_qubit,0),clock_basis(data_qubit,0))
#     H_in = np.kron(zeros_clock_proj,zero_data_proj)
#     return H_in
Z = np.array([[1, 0],
              [0, -1]])
def H_initial(unitaries):
    # assuming always starts from |0000000>
    T = len(unitaries) #we need log2(T+1) number of qubit for clock |T>
    data_qubit = int(np.log2(len(unitaries[0][0])))
    n_qubit_clock_state = int(count_clock_dimension(T))
    zero_state_projector = 1/2*(np.eye(2)-Z)
    zeros_clock_proj = np.outer(clock_basis(n_qubit_clock_state,0),clock_basis(n_qubit_clock_state, 0))
    #zero_data_proj = np.eye(2**data_qubit)-np.outer(clock_basis(data_qubit,0),clock_basis(data_qubit,0))
    zero_data_proj = sum_of_single_penalty(zero_state_projector,data_qubit)
    H_in = np.kron(zeros_clock_proj,zero_data_proj)
    return H_in
def laplacian(clock_d):
    laplacian = np.eye(clock_d)
    laplacian[0][0] = 0.5
    laplacian[-1][-1] = 0.5
    for i in range(clock_d-1):
        laplacian[i][i+1] = -0.5
        laplacian[i+1][i] = -0.5

    return laplacian
def H_prop_conj(clock_d,data_d):
    H_prop_conj = np.kron(laplacian(clock_d),np.eye(data_d))
    return H_prop_conj
def H_tot_conj(unitaries):
    clock_d = len(unitaries)+1
    data_d = len(unitaries[0][0])
    return H_prop_conj(clock_d,data_d)+ H_initial(unitaries)
def product_unitaries_loop(unitaries):
    """
    Multiply a list of unitary matrices using a loop.

    Parameters:
        unitaries (list of np.ndarray): List of square unitary matrices.
    
    Returns:
        np.ndarray: The product u0 * u1 * ... * un.
    """
    # Assume all matrices are of shape (d, d)
    d = unitaries[0].shape[0]
    result = np.eye(d, dtype=complex)  # Use the identity matrix as the starting value
    for u in unitaries:
        #print(result)
        result = result@u
        #result = result.dot(u)  # or result @ u in Python 3.5+
    return result
from math import ceil
from copy import deepcopy
def clock_append_to_po2(unitary_list):
    u_list = deepcopy(unitary_list)
    if len(u_list) == 0: 
        raise ValueError('you need a non empty list')
    dim = u_list[0].shape[0]
    if np.log2(len(u_list)+1) ==0:
        return u_list
    else:
        total_qubits = ceil(np.log2(len(u_list)+1))
        n_identities = 2**total_qubits - (len(u_list)+1)
        for _ in range(n_identities):
            u_list += [np.eye(dim)]
        return u_list
# def H_out(unitaries):
#     T = len(unitaries) #we need log2(T+1) number of qubit for clock |T>
#     data_qubit = int(np.log2(len(unitaries[0][0])))
#     n_qubit_clock_state = int(count_clock_dimension(T))
#     final_clock_proj = np.outer(clock_basis(n_qubit_clock_state,T),clock_basis(n_qubit_clock_state, T))
#     final_state = np.dot(product_unitaries_loop(unitaries[::-1]),clock_basis(data_qubit,0))
#     final_data_proj = np.eye(2**data_qubit)- np.outer(final_state,final_state)
#     H_out = np.kron(final_clock_proj,final_data_proj)
#     return H_out
# def H_tot(unitaries):
#     return H_initial(unitaries)+H_out(unitaries)+H_prop(unitaries)
def H_tot(unitaries):
    return H_initial(unitaries)+H_prop(unitaries)


def is_normalized(vec, tol=1e-8):
    """
    Check if a vector is normalized (has L2 norm equal to 1) within a given tolerance.
    
    Parameters:
        vec (np.ndarray): The input vector.
        tol (float): The tolerance for comparing the norm to 1.
    
    Returns:
        bool: True if the vector is normalized, False otherwise.
    """
    norm = np.linalg.norm(vec)
    return np.abs(norm - 1) < tol
def normalize(state_vector):
    """
    Normalizes the input state vector using L2 normalization.
    
    Parameters:
        state_vector (array-like): The input vector to normalize.
        
    Returns:
        np.ndarray: The normalized vector. If the norm of the vector is zero,
                    the original vector is returned to avoid division by zero.
    """
    # Convert the input to a numpy array (if it isn't one already)
    state_vector = np.array(state_vector, dtype=float)
    
    # Calculate the L2 norm (Euclidean norm) of the vector
    norm = np.linalg.norm(state_vector)
    
    # If norm is zero, return the vector as is to avoid division by zero.
    if norm == 0:
        return state_vector
    
    # Return the vector divided by its norm
    return state_vector / norm
class USequential(cirq.Gate):
    def __init__(self, matrix):
        super(USequential, self)
        self.matrix = matrix

    def _num_qubits_(self):
        return int(np.log2(self.matrix.shape[0]))

    def _unitary_(self):
        USequential = self.matrix.T.conj()
        return USequential

    def _circuit_diagram_info_(self, args):
        return ["USequential"] * self.num_qubits()
def embed_operator(op, target_indices, num_qubits):
    """
    Embed an operator acting on a subset of qubits into the full Hilbert space.
    
    Parameters:
        op (np.ndarray): A 2**n x 2**n matrix representing an operator on n qubits.
        target_indices (list of int): The qubit indices (0-indexed) on which op acts.
                                      The ordering of target_indices determines the correspondence
                                      between the bits of the full state and the row/column indices of op.
        num_qubits (int): Total number of qubits in the full system.
        
    Returns:
        np.ndarray: A 2**num_qubits x 2**num_qubits matrix representing the operator embedded into the full system.
    
    Example:
        >>> # Define a single-qubit Pauli-X operator.
        >>> X = np.array([[0, 1], [1, 0]])
        >>> # Embed it on a 3-qubit system acting on qubit 1.
        >>> full_op = embed_operator(X, target_indices=[1], num_qubits=3)
        >>> full_op.shape
        (8, 8)
    """
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    Useq = USequential(op).on(*[qubits[i] for i in target_indices])
    
    circuit = cirq.Circuit()
    circuit.append(Useq)
    full_op = circuit.unitary(qubits_that_should_be_present=qubits)
    return full_op

def measure_qubits_outcome(state, measured_qubits, outcome):
    """
    Projects the input state onto the subspace where the measured qubits
    (specified by their indices) yield the specified measurement outcomes,
    and returns a reduced state vector defined only on the unmeasured qubits.
    
    Parameters:
      state (np.ndarray): A 1D numpy array of length 2^n representing the full wave function.
      measured_qubits (list): List of qubit indices (0-indexed, with 0 as the most significant bit)
                              to measure.
      outcome (list or tuple): Desired measurement outcomes (0 or 1) for the corresponding qubits.
    
    Returns:
      new_state (np.ndarray): The normalized state vector of the remaining unmeasured qubits 
                              (dimension 2^(n - len(measured_qubits))).
      prob (float): The probability of obtaining the specified measurement outcome.
    """
    # Determine total number of qubits
    n = int(np.log2(len(state)))
    if len(measured_qubits) != len(outcome):
        raise ValueError("The number of measured qubits must equal the number of outcome bits provided.")

    # Identify full-space indices consistent with the desired measurement outcomes.
    indices_to_keep = []
    for i in range(len(state)):
        keep = True
        for q, bit in zip(measured_qubits, outcome):
            # Extract the bit for qubit q (with 0 as the most significant bit)
            if ((i >> (n - 1 - q)) & 1) != bit:
                keep = False
                break
        if keep:
            indices_to_keep.append(i)
    indices_to_keep = np.array(indices_to_keep)
    
    # Calculate the probability of the measurement outcome via the Born rule.
    prob = np.sum(np.abs(state[indices_to_keep])**2)
    if prob == 0:
        raise ValueError("The measurement outcome {} for qubits {} has zero probability.".format(outcome, measured_qubits))
    
    # Determine which qubits remain (i.e. not measured)
    remaining_qubits = sorted(set(range(n)) - set(measured_qubits))
    new_dim = 2**len(remaining_qubits)
    
    # Initialize the reduced state vector.
    reduced_state = np.zeros(new_dim, dtype=state.dtype)
    
    # Map each index in the full state (consistent with the outcome)
    # to an index in the reduced state vector.
    for i in indices_to_keep:
        reduced_index = 0
        # For each remaining qubit, extract its bit and build the new index.
        for q in remaining_qubits:
            bit = (i >> (n - 1 - q)) & 1
            reduced_index = (reduced_index << 1) | bit
        # Since the mapping is one-to-one, assign the amplitude.
        reduced_state[reduced_index] = state[i]
    
    # Renormalize the reduced state vector.
    reduced_state /= np.sqrt(prob)
    
    return reduced_state, prob

import numpy as np
def sum_of_single_penalty(operator, n_qubits):
    if operator.shape != (2, 2):
        raise ValueError("A must be a 2x2 matrix.")
    dim = 2**(n_qubits)
    sop = np.zeros((dim,dim))
    for i in range(n_qubits):
        sop += np.kron(np.eye(2**i), np.kron(operator, np.eye(2**(n_qubits - i - 1))))

    return sop
def convert_if_real(matrix):
    """
    Convert a complex matrix to a float64 matrix if its imaginary parts are all (approximately) zero.
    
    Parameters:
    matrix (np.ndarray): A NumPy array with dtype np.complex128.
    
    Returns:
    np.ndarray: A float64 matrix if no significant imaginary parts are present,
                otherwise the original complex matrix.
    """
    # Check if all imaginary parts are close to 0.
    if np.allclose(matrix.imag, 0):
        return matrix.real.astype(np.float64)
    else:
        return matrix


def cumulative_unitaries(unitary_list):
    result = []
    cumulative_product = np.eye(unitary_list[0].shape[0], dtype=complex)  # Identity matrix of appropriate size
    
    for U in unitary_list:
        cumulative_product = np.dot(U,cumulative_product)  # Multiply matrices
        result.append(cumulative_product.copy())  # Store the result
    
    return result

def uwu_basis_change(H_problem,unitaries_list):
    accumulated_unitaries = cumulative_unitaries(unitaries_list)
    H_wconj = []
    for i in range(len(accumulated_unitaries)):
        H_wconj.append(accumulated_unitaries[i].conj().T @ H_problem @ accumulated_unitaries[i])
    return H_wconj

def block_diag_matrix(matrices):
    """
    Construct a block diagonal matrix from a list of square matrices.
    Parameters:
        matrices (list of np.ndarray): List of square numpy arrays.
    
    Returns:
        np.ndarray: The block diagonal matrix.
    """
    # Assume all matrices have the same shape (m, m)
    m = matrices[0].shape[0]
    k = len(matrices)
    
    # Create an empty array of shape (k*m, k*m)
    block_diag = np.zeros((k * m, k * m), dtype=matrices[0].dtype)
    
    # Place each matrix in its corresponding block
    for i, matrix in enumerate(matrices):
        block_diag[i*m:(i+1)*m, i*m:(i+1)*m] = matrix
    
    return block_diag

def get_bandwidth(matrix):
    """
    Determines the minimal bandwidth of a square matrix.
    
    A matrix is said to be band diagonal if all entries for which |i - j| > k are zero,
    where k is the bandwidth. This function returns the minimal bandwidth k such that every 
    nonzero element of the matrix satisfies |i - j| <= k.
    
    Parameters:
        matrix (list of lists): The square matrix to check.
        
    Returns:
        int: The minimal bandwidth k.
        
    Raises:
        ValueError: If the matrix is not square.
    """
    n = len(matrix)
    
    # Ensure the matrix is square
    for row in matrix:
        if len(row) != n:
            raise ValueError("Matrix must be square.")
    
    # Compute the minimal required bandwidth by checking every element.
    required_bandwidth = 0
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                required_bandwidth = max(required_bandwidth, abs(i - j))
    
    # By definition, all nonzero elements are within |i - j| <= required_bandwidth.
    # (Elements outside this band are zero, since required_bandwidth is computed as the max |i-j| 
    # for a nonzero entry.)
    
    return required_bandwidth


###############
########
#####
####
##


paulix = np.array([[0, 1],
                   [1, 0]], dtype=complex)   # X gate
pauliz = np.array([[1, 0],
                   [0, -1]], dtype=complex)  # Z gate
hadamard = 1/np.sqrt(2) * np.array([[1, 1],
                                    [1, -1]], dtype=complex)  # Hadamard gate
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# Define the CZ gate
CZ = np.array([
    [1,  0,  0,  0],
    [0,  1,  0,  0],
    [0,  0,  1,  0],
    [0,  0,  0, -1]
], dtype=complex)
def A(g):
    """Compute the tensor A in form of d=2 matrices with shape (D, D) = (2, 2) each."""
    eta = 1 / np.sqrt(1 - g)
    A_0 = np.array([[1, 0], [np.sqrt(-g), 0]]) * eta
    A_1 = np.array([[0, -np.sqrt(-g)], [0, 1]]) * eta
    return (A_0, A_1)
def sequential_prep_unitary(g, n_qubits):
    global np
    n_qubits = n_qubits+2
    A_0, A_1 = A(g)
    E_00 = np.array([[1, 0], [0, 0]])  # |0><0|
    E_10 = np.array([[0, 0], [1, 0]])  # |1><0|
    U_first_term = np.kron(A_0, E_00) + np.kron(A_1, E_10)
    E_01 = np.array([[0, 1], [0, 0]])  # |0><1|
    E_11 = np.array([[0, 0], [0, 1]])  # |1><1|
    C_perp = np.kron(A_1, E_01) + np.kron(A_0, E_11)
    U = U_first_term + C_perp
    eta = 1 / np.sqrt(1 - g)
    unitaries = []
    h_big = np.kron(hadamard,np.eye(2**(n_qubits-1)))
    #print(len(h_big))
    unitaries.append(h_big)
    cnot_big = embed_operator(CNOT, [0, n_qubits-1], n_qubits)
    unitaries.append(cnot_big)
    #print(cnot_big)
    for i in range(1, n_qubits-1):
        #u = embed_operator(U,[i+1,i],n_qubits)
        #print(i)
        #print(U)
        #print(np.allclose(CNOT,np.round(U, 5)))
        u = embed_operator(U,[n_qubits-1,i],n_qubits)
        a = embed_operator(CNOT,[n_qubits-1,i],n_qubits)
        #print(np.allclose(a,u))
        # here U is the preparation unitary
        unitaries.append(u)
    #print(a)
    unitaries.append(cnot_big)
    unitaries.append(h_big)
    zeros_state = np.zeros(2**n_qubits)
    zeros_state[0] = 1  
    # for i in unitaries:
    #     print(np.allclose(i.conj().T @ i, np.eye(len(i[0]))))
    circU = product_unitaries_loop(unitaries[::-1])
    # print(np.allclose(circU.conj().T @ circU, np.eye(len(circU[0]))))
    final = np.dot(product_unitaries_loop(unitaries[::-1]),zeros_state)
    #print(product_unitaries_loop(unitaries[::-1][0:1])[:,0])
    new_state, prob = measure_qubits_outcome(final, [0,n_qubits-1], [0,0])
    return new_state,unitaries

    import numpy as np
from functools import reduce
from typing import List

def kron_list(mats: List[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, mats)

def ghz(n: int,
        X: np.ndarray,
        Z: np.ndarray,
        I: np.ndarray) -> np.ndarray:
    """
    Build the n-qubit GHZ‐type Hamiltonian
      H = - ( sum_{i=0..n-2} Z_i Z_{i+1} + X_0 X_1 ... X_{n-1} ).
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    # ZZ couplings along the chain
    ZZ_terms = []
    for i in range(n-1):
        ops = [I]*n
        ops[i]   = Z
        ops[i+1] = Z
        ZZ_terms.append(kron_list(ops))

    # global X⊗n
    Xn = kron_list([X]*n)

    return -( sum(ZZ_terms) + Xn )




def plus_penalty(n_qubits):
    dim = 2**n_qubits
    plus_state = np.ones(dim)
    plus_state = normalize(plus_state)
    penalty = np.eye(dim)- np.outer(plus_state,plus_state)
    return penalty



def permute_qubits(matrix: np.ndarray, perm: list) -> np.ndarray:
    """
    Permute the qubit ordering of a 2^n x 2^n matrix.

    Args:
        matrix: The input operator matrix of shape (2^n, 2^n).
        perm: A list of length n specifying the new ordering of the qubits.
              For example, perm=[2,0,1,3] means:
                qubit 0 -> position 2,
                qubit 1 -> position 0,
                qubit 2 -> position 1,
                qubit 3 -> position 3.

    Returns:
        The permuted matrix of the same shape.
    """
    n = int(np.log2(matrix.shape[0]))
    assert matrix.shape == (2**n, 2**n), "Matrix must be 2^n x 2^n"
    assert sorted(perm) == list(range(n)), "perm must be a permutation of [0..n-1]"

    # Reshape into a tensor with separate row and column qubit axes
    tensor = matrix.reshape([2]*n + [2]*n)

    # Build transpose axes: first the row-qubit axes in new order, then the col-qubit axes
    axes = perm + [p + n for p in perm]

    # Transpose and collapse back to 2^n x 2^n
    permuted = tensor.transpose(axes).reshape(2**n, 2**n)
    return permuted

def reflector_1(n_qubits,index):
    label_qubits = len(index)
    qubit_indices = list(range(n_qubits))
    for i in index:
        qubit_indices.remove(i)
    qubit_indices = qubit_indices[::-1] + index
    #qubit_indices = [4,2,1,0,3]
    # print(qubit_indices)
    ref_label = np.eye(2**label_qubits)
    ref_label[0][0] = -1
    ref_label_full = np.kron(np.eye(2**(n_qubits-label_qubits)),ref_label)
    ref_label_full = permute_qubits(ref_label_full, qubit_indices)
    return ref_label_full

def aa(index,U_prep,repetition):
    n_qubits = int(np.log2(U_prep.shape[0]))
    aa_circuit = np.eye(2**n_qubits)
    state = np.zeros(2**n_qubits)
    state[0] = 1
    prep_state = U_prep@state
    proj = np.outer(prep_state,prep_state.conj())
    reflector = np.eye(proj.shape[0]) - 2*proj
    # print("Unitarity test for reflector ", -np.allclose(reflector@reflector.T.conj(), np.eye(reflector.shape[0])))
    for i in range(repetition):
        aa_circuit = -reflector@reflector_1(n_qubits,index)@aa_circuit
        # print("Unitarity test for reflector ", np.allclose(reflector_1(n_qubits,index)@reflector_1(n_qubits,index).T.conj(), np.eye(reflector_1(n_qubits,index).shape[0])))
    return aa_circuit

def dummy_state_prep(state):
    """
    Using householder matrix to construct a unitary that prepares a given state.
    input:
    state: np.ndarray, shape (2**n,), the target state to prepare
    output:
    unitary: np.ndarray, shape (2**n, 2**n), the unitary matrix that prepares the state
    """
    eye = np.eye(len(state), dtype=np.complex128)
    k = np.zeros(len(state), dtype=np.complex128)
    
    k[0] = 1
    overlap = k.conj().T@ state/np.abs(k.conj().T@ state)
    print(overlap)
    w = overlap*k - state
    w = w / np.linalg.norm(w)  # Normalize w
    u = eye - 2 * np.outer(w,w.conj()) # Householder reflection
    print(np.allclose(u@u.conj().T,np.eye(len(state), dtype=np.complex128)))
    u = overlap * u
    print(np.allclose(u@u.conj().T,np.eye(len(state), dtype=np.complex128)))
    return u # problem