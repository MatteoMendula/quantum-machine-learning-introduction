from __future__ import annotations

import torch
from torch import nn
from .vqc import VQC
"""
TO DO: expand what is done from in_channel = 1 to in_channel > 1
"""
class QConv(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channel_out: int,
        kernel_size: int = 4,
        stride: int = 1,
        num_layers_per_kernel: int = 1,
        encoding: str = "angle",
        reuploading: bool = False,
        hadamard: bool = False,
        gate_used: str = "Y",
        name_ansatz: str = "strongly",
    ) -> None:
        
        super().__init__()
        self.model = nn.ModuleList()
        self.input_size = input_size
        self.num_channel_out = num_channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers_per_kernel = num_layers_per_kernel
        self.encoding = encoding
        self.reuploading = reuploading
        self.hadamard = hadamard
        self.gate_used = gate_used
        self.name_ansatz = name_ansatz
        if self.encoding == "amplitude" and self.num_channel > 1:
            msg = "Encoding 'amplitude' is not supported for QConv1D"
            raise ValueError(msg)
        # limit of this kind of circuit
        if self. num_channel_out > self.kernel_size:
            msg = "The number of output channels should be less than or equal to the kernel size"
            raise ValueError(msg)
        
        if self.encoding == "angle":
            for _i in range(0, input_size - kernel_size + 1, stride):
                
                self.model.append(
                    VQC(
                        num_wires=kernel_size,
                        num_outputs=num_channel_out,
                        num_layers=num_layers_per_kernel,
                        encoding="angle",
                        reuploading=reuploading,
                        hadamard=hadamard,
                        gate_used=gate_used,
                        name_ansatz=name_ansatz
                    )
                )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for each input channel
        # do the convolution and save the result in the structure
        output = torch.zeros(x.size(0), self.num_channel_out, (self.input_size - self.kernel_size + 1) // self.stride)
        for i in range(x.size(0)):
            for j in range((self.input_size - self.kernel_size)// self.stride + 1):
                # get the input for the VQC
                input_vqc = x[i, j * self.stride :j * self.stride +self.kernel_size]
                # get the output of the VQC
                output_vqc = self.model[j](input_vqc)
                # save the output in the structure
                output[i, :, j] = output_vqc
        return output
        

                
    
        
            