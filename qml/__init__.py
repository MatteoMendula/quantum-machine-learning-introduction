from __future__ import annotations

import logging

# isort: off
from .ansatz import Ansatz
from .vqc import VQC
from .multi_vqc import MultiVQC
from .utils import Utils
from .qconv import QConv
from .qsvm import Kernel_QSVM
# isort: on

__all__ = ["Ansatz", "VQC", "MultiVQC", "Utils", "QConv", 'Kernel_QSVM']

logger = logging.getLogger("quaptor-Quantum-machine-learning")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
