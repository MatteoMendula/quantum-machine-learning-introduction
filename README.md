# Quantum ML (quaptor) — Local QML examples

Brief overview
- This repository contains implementations of quantum machine-learning building blocks using PennyLane and PyTorch: variational quantum circuits (`VQC`), a multi-block composition (`MultiVQC`), a simple quantum convolution `QConv`, and a quantum-kernel SVM (`Kernel_QSVM`). There's also utilities for training, evaluation and cross-validation.

Primary files
- `testDiabetes.py`: example script that loads a diabetes dataset and runs training / cross-validation using the `VQC`/`MultiVQC` modules.
- `qml/ansatz.py`: ansatz templates and entanglement patterns.
- `qml/vqc.py`: core `VQC` PyTorch module wrapping a PennyLane `QNode`.
- `qml/multi_vqc.py`: sequential composition of multiple `VQC` modules.
- `qml/qconv.py`: 1D quantum convolution built from `VQC` kernels.
- `qml/qsvm.py`: quantum kernel SVM helper.
- `qml/utils.py`: training, evaluation and cross-validation helpers.

Dependencies
- See `requirements.txt` for the packages used. Install with:

```bash
python -m pip install -r requirements.txt
```

Running the example `testDiabetes.py`
- The script expects a CSV at `./test/qml_applications/diabetes.csv` (see the `pd.read_csv(...)` call). Make sure the dataset is placed there or update the path in `testDiabetes.py`.
- The example imports `src.quaptor.*` modules and — in the present copy — uses a hard-coded `sys.path.append` to point to a different location. You have two simple options to run the example locally:

  1) Install the `quaptor` package (if you have it in `src/quaptor`) in editable mode from its project root:

```bash
cd /path/to/quaptor/project_root
python -m pip install -e .
# then run the example in this repo
cd /home/matteo/Documents/postDoc/QML/qml
python testDiabetes.py
```

  2) Adjust the import lines in `testDiabetes.py` to use the local `qml` package instead of `src.quaptor.qml` (quick local fix). For example, replace:

```python
from src.quaptor.qml import VQC, MultiVQC, Utils
```

with:

```python
from qml.qml import VQC, MultiVQC, Utils
```

or modify `sys.path` at the top of `testDiabetes.py` to include this repo root:

```python
import sys
sys.path.append(".")
```

Notes and tips
- PennyLane simulations can be slow; consider using a faster backend (e.g., `lightning` plugin) if available and compatible with your hardware.
- If you plan to run many experiments, create a `venv` or use `conda` and pin exact package versions to avoid incompatibilities.

Next steps I can do for you
- (A) Fix the imports in `testDiabetes.py` so it runs without the hard-coded `sys.path.append`.
- (B) Add a tiny smoke-test script that runs a single forward pass with random input.
- (C) Pin version numbers in `requirements.txt`.

If you want one of the above, tell me which and I'll implement it next.
