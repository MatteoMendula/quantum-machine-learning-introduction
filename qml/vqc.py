from __future__ import annotations

import math
from typing import Any

import pennylane as qml
import torch
from torch import nn
import numpy as np

device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .ansatz import Ansatz

class VQC(nn.Module):
    def __init__(
        self,
        num_wires: int,
        num_outputs: int,
        num_layers: int,
        encoding: str = "angle",
        reuploading: bool = False,
        hadamard: bool = False,
        gate_used: str = "Y",
        name_ansatz: str = "strongly",
    ) -> None:
        super().__init__()
        """
        Constructor
        @encoding: String which represents the gates used for the Angle encoding
        @Ansatz: String which represents the ansatz used for quantum circuit
        @Reuploading: Boolean indicating whether or not to use reuploading
        @hadamard: Boolean indicating whether or not to use Hadamard gates
        @num_layers: Integer representing the number of layers in the quantum circuit
        @num_wires: Integer representing the number of wires in the quantum circuit
        @num_outputs: Integer representing the number of output qubits
        @gate_used: String representing the encoding gate used
        @name_ansatz: String representing the ansatz used
        """
        self.hadamard = hadamard
        self.encoding = encoding
        self.reuploading = reuploading
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.num_outputs = num_outputs
        self.gate_used = gate_used
        self.name_ansatz = name_ansatz
        self.ansatz = Ansatz()

        # PennyLane device
        '''if torch.cuda.is_available():
            
            self.dev = qml.device("lightning.gpu", wires=self.num_wires)
        else:
        '''
        self.dev = qml.device("default.qubit", wires=self.num_wires)
        if self.encoding == "angle":
            # Validate Encoding
            valid_encodings = {"X", "Y", "Z"}
            for letter in gate_used:
                if letter not in valid_encodings:
                    msg = f"Invalid encoding gate: {letter}. Choose from 'X', 'Y', 'Z'."
                    raise ValueError(msg)
        """
        if name_ansatz not in {"strongly", "basic", 'circuit_2', 'circuit_3', 'circuit_4', 'circuit_5', 'circuit_6'}:
            msg = f"Invalid ansatz name: {name_ansatz}. Choose from 'strongly', 'basic'."
            raise ValueError(msg)
        """

        
        self.weight_shapes = {"weights": self.ansatz.shape_weights(self.num_wires,
                                                                   self.num_layers,
                                                                   self.name_ansatz)}
        '''
        if self.name_ansatz == "strongly":
            self.weight_shapes = {"weights": (self.num_layers, self.num_wires, 3)}
        elif self.name_ansatz == "basic":
            self.weight_shapes = {"weights": (self.num_layers, self.num_wires)}
        '''
        # Set device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the quantum node
        self.qnode = self.create_qnode()
        # Define the quantum layer in PyTorch
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes).to(self.device)

    def create_qnode(self) -> qml.QNode:
        """Creates the quantum node for the hybrid model."""

        @qml.qnode(self.dev)
        def qnode(inputs: torch.Tensor, weights: torch.nn.parameter.Parameter) -> list[Any]:
            # Apply Hadamard if specified
            if self.hadamard:
                for i in range(self.num_wires):
                    qml.Hadamard(wires=i)

            # Encoding and Ansatz logic
            if self.reuploading:
                if self.encoding != "amplitude":
                    for w in weights:
                        self.encoding_circuit(inputs)
                        self.apply_ansatz(w.unsqueeze(0), name_ansatz=self.name_ansatz)
                elif self.encoding == "amplitude":
                    msg = "Amplitude encoding is not supported with re-uploading."
                    raise ValueError(msg)
            else:
                self.encoding_circuit(inputs)
                self.apply_ansatz(weights, name_ansatz=self.name_ansatz)

            # Measurement
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.num_outputs)]

        return qnode

    def apply_ansatz(self, weights: torch.nn.parameter.Parameter, name_ansatz: str = "strongly") -> None:
        """Apply the variational circuit."""
        '''if name_ansatz == "strongly":
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_wires))
        elif name_ansatz == "basic":
            qml.BasicEntanglerLayers(weights, wires=range(self.num_wires))
        else:
            msg = f"Invalid ansatz name: {name_ansatz}. Choose from 'strongly', 'basic'."
            raise ValueError(msg)
        '''

        self.ansatz.circuit_single(self.name_ansatz, weights, list(range(self.num_wires)))

    def ZZ_feature_mapping(self, x: torch.Tensor) -> None:
        for i in range(self.num_wires):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)

        for i in range(self.num_wires - 1):
            for j in range(i + 1, self.num_wires):
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])


    def ZZ_feature_mapping_qiskit(self, x: torch.Tensor) -> None:
        for i in range(self.num_wires):
            qml.Hadamard(wires=i)
            qml.RZ(2 * x[i], wires=i)

        for i in range(self.num_wires - 1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[i+1]), wires=i+1)
            qml.CNOT(wires=[i, i+1])

    def encoding_circuit(self, inputs: torch.Tensor) -> None:
        """
        Apply encoding circuit based on the specified encoding method.
        @ inputs: array of input values in range [-1, 1]
        """
        if self.encoding == "angle":
            for e in self.gate_used:
                qml.AngleEmbedding(math.pi / 2 * inputs, wires=range(self.num_wires), rotation=e)
        elif self.encoding == 'ZZ':
            inputs = inputs * np.pi / 4
            self.ZZ_feature_mapping(inputs)
        elif self.encoding == 'ZZ-qiskit':
            inputs = inputs * np.pi / 4
            self.ZZ_feature_mapping_qiskit(inputs)
        elif self.encoding == "amplitude":
            norms = torch.norm(inputs, dim=1, keepdim=True)
            inputs /= norms
            qml.StatePrep(inputs, wires=range(self.num_wires))
            # qml.MottonenStatePreparation(inputs, wires=range(self.num_wires))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid model."""
        
        return self.qlayer(inputs)
    

    def get_name(self)->str:
        return "VQC_encoding_" + self.encoding + "_ansatz_" + self.name_ansatz + "_gateUsed_" + self.gate_used \
            + "_reuploading_" + str(self.reuploading) + "_hadamard_" + str(self.hadamard) + "_numLayers_" \
                + str(self.num_layers) + "_numWires_" + str(self.num_wires) + "_numOutputs_" + str(self.num_outputs)
    

    def get_dict_model(self)->dict:
        return {
            "num_wires": self.num_wires,
            "num_outputs": self.num_outputs,
            "num_layers": self.num_layers,
            "encoding": self.encoding,
            "reuploading": self.reuploading,
            "hadamard": self.hadamard,
            "gate_used": self.gate_used,
            "name_ansatz": self.name_ansatz,
        }
