import pennylane as qml
import numpy as np
class Kernel_QSVM():
    def __init__(self, n_qubits, encoding, repetition):
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.repetition = repetition
        self.projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
        self.projector[0, 0] = 1
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = self.create_qnode()

    
    def create_qnode(self) -> qml.QNode:
        """Creates the quantum node for the hybrid model."""

        @qml.qnode(self.dev)
        def qnode(x: np.array, y: np.array) -> float:
            for _ in range(self.repetition):
                self.encoding_circuit(x)
                qml.adjoint(self.encoding_circuit)(y)
            

            # Measurement
            return qml.expval(qml.Hermitian(self.projector, wires=range(self.n_qubits)))

        return qnode
    
    def ZZ_feature_mapping(self, x: np.array):
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)

        for i in range(self.n_qubits - 1):
            for j in range(i + 1, self.n_qubits):
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])


    def ZZ_feature_mapping_qiskit(self, x: np.array):
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)

        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[i+1]), wires=i+1)
            qml.CNOT(wires=[i, i+1])

        

    def Z_feature_mapping(self, x: np.array):
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)


    def encoding_circuit(self, x: np.array):
        if self.encoding == 'ZZ':
            self.ZZ_feature_mapping(x)
        elif self.encoding == 'Angle':
            qml.AngleEmbedding(x, wires=range(self.n_qubits))
        elif self.encoding == 'Z':
            self.Z_feature_mapping(x)
        elif self.encoding == 'ZZ-qiskit':
            self.ZZ_feature_mapping_qiskit(x)
        else:
            raise ValueError("Invalid encoding type. Choose from 'ZZ', 'Angle', 'Z'")


    def kernel_matrix(self, A: np.array, B: np.array) -> np.array:
        """Compute the matrix whose entries are the kernel
           evaluated on pairwise data from sets A and B."""
        return np.array([[self.qnode(a, b) for b in B] for a in A])
    
    

    
    
 
    

