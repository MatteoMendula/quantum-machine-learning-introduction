# Quantum ML — Local QML examples

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

Running the example `testDiabetes.py
- The script expects a CSV at `./test/qml_applications/diabetes.csv` (see the `pd.read_csv(...)` call). Make sure the dataset is placed there or update the path in `testDiabetes.py`.

```bash
python testDiabetes.py
```