from __future__ import annotations

# Local imports and standard libs
import numpy as np
import pandas as pd
import torch

# Import local modules from the bundled `qml` package
from qml.preprocessing import PreprocessingQml
from qml.vqc import VQC
from qml.multi_vqc import MultiVQC
from qml.utils import Utils
from matplotlib import pyplot as plt
# Load the data
data = pd.read_csv("./diabetes.csv")
# Define the input and output
x = data.drop(columns="Outcome")
y = data["Outcome"].values

preprocessing = PreprocessingQml()
x_train, x_valid, x_test, y_train, y_valid, y_test = preprocessing.preprocessing(
    x, y, test_size=0.2, random_state=42, standardization=True
)  # , pca=True, pca_n_components=4)
# Define the model

'''
model = MultiVQC(
    num_vqc=2,
    num_wires=8,
    num_outputs=1,
    num_layers=8,
    encoding="ZZ",
    reuploading=False,
    hadamard=False,
    name_ansatz="basic",
)'''
model = VQC(num_wires=8,
    num_outputs=1,
    num_layers=2,
    encoding="angle",
    reuploading=False,
    hadamard=False,
    name_ansatz="strongly")

# Train the model
utils = Utils()

y_train = y_train.unsqueeze(1)
y_valid = y_valid.unsqueeze(1)
y_test = y_test.unsqueeze(1)


"""x_train = utils.padding(x_train)
x_valid = utils.padding(x_valid, 2 ** 4)
x_test = utils.padding(x_test, 2 ** 4)
"""
print("Training the model")
'''
utils.train(
    model,
    x_train,
    y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    epochs=10,
    batch_size=32,
    metrics=["loss", "accuracy", "f1", "roc_auc", "precision", "recall", "confusion_matrix"],
    verbose=True,
)
# Evaluate the model
dict_metrics = utils.evaluation(
    model, x_test, y_test, metrics=["accuracy", "f1", "roc_auc", "precision", "recall", "confusion_matrix"]
)
for i in dict_metrics:
    print(i, dict_metrics[i], type(dict_metrics[i]))
print(dict_metrics)
'''
model(x_train)

preprocessing.standardization(x)
x = torch.Tensor(x.to_numpy())
y = torch.Tensor(y).unsqueeze(1)
dict_metrics =\
utils.crossValidation(
    model,
    x,
    y,
    cv=5,
    epochs=10,
    stratified=True,
    batch_size=64,
    metrics=["accuracy", "f1", "roc_auc", "precision", "recall", "confusion_matrix"],
    verbose=True,
)

for i in dict_metrics:
    dict_metrics[i]['confusion_matrix'] = dict_metrics[i]["confusion_matrix"].tolist()
#dict_metrics['confusion_matrix'] = dict_metrics["confusion_matrix"].tolist()

dictRes = model.get_dict_model()
dictRes['results'] = dict_metrics
for i in dictRes:
    print(i, dictRes[i], type(dictRes[i]))
#save dict_metrics in a json file
import json
with open('dict_metrics_diabetes.json', 'w') as fp:
    json.dump(dictRes, fp)