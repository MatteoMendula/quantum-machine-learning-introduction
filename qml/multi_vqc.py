from __future__ import annotations

import torch
from torch import nn
from .vqc import VQC


class MultiVQC(nn.Module):
    def __init__(
        self,
        num_vqc: int,
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
        self.model = nn.ModuleList()
        self.num_vqc = num_vqc
        self.num_wires = num_wires
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.encoding = encoding
        self.reuploading = reuploading
        self.hadamard = hadamard
        self.gate_used = gate_used
        self.name_ansatz = name_ansatz
        if self.encoding == "amplitude" and self.num_vqc > 1:
            msg = "Encoding 'amplitude' is not supported for MultiVQC"
            raise ValueError(msg)

        if self.encoding == "angle" or self.encoding == "ZZ" or self.encoding == "ZZ-qiskit":
            for _i in range(num_vqc - 1):
                self.model.append(
                    VQC(
                        num_wires=num_wires,
                        num_outputs=num_wires,
                        num_layers=num_layers,
                        encoding=encoding,
                        reuploading=reuploading,
                        hadamard=hadamard,
                        gate_used=gate_used,
                        name_ansatz=name_ansatz,
                    )
                )

            self.model.append(
                VQC(
                    num_wires=num_wires,
                    num_outputs=num_outputs,
                    num_layers=num_layers,
                    encoding=encoding,
                    reuploading=reuploading,
                    hadamard=hadamard,
                    gate_used=gate_used,
                    name_ansatz=name_ansatz,
                )
            )
        elif self.encoding == "amplitude":
            self.model.append(
                VQC(
                    num_wires=num_wires,
                    num_outputs=num_outputs,
                    num_layers=num_layers,
                    encoding="amplitude",
                    reuploading=reuploading,
                    hadamard=hadamard,
                    gate_used=gate_used,
                    name_ansatz=name_ansatz,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.model)):
            x = self.model[i](x)
        if self.num_outputs > 1:
            self.softmax = nn.Softmax(dim=1)
            x = self.softmax(x)
        return x

    def get_name(self)->str:
        return "MultiVQC_numVQC_" + str(self.num_vqc) +'_encoding_' + self.encoding + "_ansatz_" + self.name_ansatz + "_gateUsed_" + self.gate_used \
            + "_reuploading_" + str(self.reuploading) + "_hadamard_" + str(self.hadamard) + "_numLayers_" \
                + str(self.num_layers) + "_numWires_" + str(self.num_wires) + "_numOutputs_" + str(self.num_outputs)
    
    def get_dict_model(self)->dict:

        return {
            "num_vqc": self.num_vqc,
            "num_wires": self.num_wires,
            "num_outputs": self.num_outputs,
            "num_layers": self.num_layers,
            "encoding": self.encoding,
            "reuploading": self.reuploading,
            "hadamard": self.hadamard,
            "gate_used": self.gate_used,
            "name_ansatz": self.name_ansatz,
        }
