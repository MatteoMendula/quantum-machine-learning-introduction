from __future__ import annotations

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from torch import nn
from typing import Any

class Utils:
    @staticmethod
    def padding(x: torch.Tensor, max_len: int, value: float = 0) -> torch.Tensor:
        """
        Pad the data
        @x: torch.Tensor, data, MANDATORY
        @max_len: int, maximum length
        @value: float, value to pad
        """
        if x.shape[1] < max_len:
            pad = torch.ones(x.shape[0], max_len - x.shape[1]) * value
            x = torch.cat([x, pad], dim=1)
        return x

    @staticmethod
    def evaluation(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        threshold_classification: float = 0.5,
        criterion: nn.Module = None,
        metrics: None | list[str] = None,
    ) -> dict:
        """
        Evaluate the model
        @model: nn.Module, model to evaluate, MANDATORY
        @x: torch.Tensor, data, MANDATORY
        @y: torch.Tensor, labels, MANDATORY
        @criterion: None|nn.Module, loss function
        @threshold_classification: float, threshold for classification
        @metrics: None|list[str], list of metrics to evaluate
        """
        model.eval()
        with torch.no_grad():
            if model.num_outputs == 1:
                y_pred = (1 - model(x)) / 2
            else:
                y_pred = model(x)
                y = y.squeeze(1).long()
                #y_pred = y_pred.float()
        dict_metrics = {}
        if metrics is None:
            metrics = ["loss", "accuracy"]
        for m in metrics:
            if m == "loss":
                if criterion is None:
                    criterion = nn.BCELoss() if model.num_outputs == 1 else nn.CrossEntropyLoss()
                loss = criterion(y_pred, y)
                dict_metrics["loss"] = loss

            if m == "accuracy":
                if model.num_outputs == 1:
                    y_pred = y_pred > threshold_classification
                    acc = (y_pred == y).sum().item() / len(y)
                else:
                    acc = (torch.argmax(y_pred, dim=1) == y).sum().item() / len(y)
                dict_metrics["accuracy"] = acc
            if m == "f1":
                if model.num_outputs == 1:
                    y_pred = y_pred > threshold_classification
                    f1 = f1_score(y, y_pred)
                else:
                    f1 = f1_score(y, torch.argmax(y_pred, dim=1))
                dict_metrics["f1"] = f1
            if m == "roc_auc":
                if model.num_outputs == 1:
                    roc_auc = roc_auc_score(y, y_pred)
                else:
                    roc_auc = roc_auc_score(y, torch.argmax(y_pred, dim=1))
                dict_metrics["roc_auc"] = roc_auc
            if m == "precision":
                if model.num_outputs == 1:
                    y_pred = y_pred > threshold_classification
                    precision = precision_score(y, y_pred)
                else:
                    precision = precision_score(y, torch.argmax(y_pred, dim=1))
                dict_metrics["precision"] = precision
            if m == "recall":
                if model.num_outputs == 1:
                    y_pred = y_pred > threshold_classification
                    recall = recall_score(y, y_pred)
                else:
                    recall = recall_score(y, torch.argmax(y_pred, dim=1))
                dict_metrics["recall"] = recall
            if m == "confusion_matrix":
                if model.num_outputs == 1:
                    y_pred = y_pred > threshold_classification
                    cm = confusion_matrix(y, y_pred)
                else:
                    cm = confusion_matrix(y, torch.argmax(y_pred, dim=1))
                dict_metrics["confusion_matrix"] = cm
        return dict_metrics

    def train(
        self,
        model: nn.Module,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: None | torch.Tensor = None,
        y_valid: None | torch.Tensor = None,
        x_test: None | torch.Tensor = None,
        y_test: None | torch.Tensor = None,
        optimizer: torch.optim.Optimizer = None,
        criterion: None | nn.Module = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping: bool = True,
        patience: int = 5,
        threshold_classification: float = 0.5,
        verbose: bool = True,
        metrics: None | list[str] = None,
    ) -> nn.Module:
        """
        Train the model
        @model: nn.Module, model to train, MANDATORY, The output model should be in the interval [-1, 1]
        @x_train: torch.Tensor, training data, MANDATORY
        @y_train: torch.Tensor, training labels, MANDATORY, The labels should be 0 or 1
        @x_valid: None|torch.Tensor, validation data
        @y_valid: None|torch.Tensor, validation labels
        @x_test: None|torch.Tensor, testing data
        @y_test: None|torch.Tensor, testing labels
        @optimizer: torch.optim.Optimizer, optimizer to use
        @criterion: nn.Module, loss function
        @epochs: int, number of epochs
        @batch_size: int, batch size
        @early_stopping: bool, apply early stopping or not
        @patience: int, number of epochs without improvement
        @threshold_classification: float, threshold for classification
        @verbose: bool, print the training process
        """
        if metrics is None:
            metrics = ["loss", "accuracy"]
        if criterion is None:
            criterion = nn.BCELoss() if model.num_outputs == 1 else nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters())

        min_loss = float("inf")
        patience_counter = 0
        x_train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            model.train()
            loss_res = 0
            for x_batch, y_batch in x_train_dataloader:
                optimizer.zero_grad()
                if model.num_outputs == 1:
                    y_pred = (1 - model(x_batch)) / 2
                else:
                    y_pred = model(x_batch)
                    # Convert labels to correct shape and type
                    y_batch = y_batch.squeeze(1).long()
                    #y_pred = torch.argmax(y_pred, dim=1)
                    #y_pred = y_pred.float()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                loss_res += loss.item()
            loss_res /= len(x_train_dataloader)
            if early_stopping and x_valid is not None and y_valid is not None:
                dict_metrics = self.evaluation(model, x_valid, y_valid, criterion=criterion, metrics=["loss"])
                if dict_metrics["loss"] < min_loss:
                    min_loss = dict_metrics["loss"]
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    model.load_state_dict(best_model_state)  # Restore best model
                    break
                
            if verbose:
                dict_metrics = self.evaluation(
                    model=model,
                    x=x_train,
                    y=y_train,
                    threshold_classification=threshold_classification,
                    criterion=criterion,
                    metrics=metrics,
                )
                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss_res} - {dict_metrics}")
                if x_valid is not None and y_valid is not None:
                    dict_metrics = self.evaluation(
                        model=model,
                        x=x_valid,
                        y=y_valid,
                        threshold_classification=threshold_classification,
                        criterion=criterion,
                        metrics=metrics,
                    )
                    print(f"Validation - {dict_metrics}")
                if x_test is not None and y_test is not None:
                    dict_metrics = self.evaluation(
                        model=model,
                        x=x_test,
                        y=y_test,
                        threshold_classification=threshold_classification,
                        criterion=criterion,
                        metrics=metrics,
                    )
                    print(f"Test - {dict_metrics}")
        return model
    

    def crossValidation(
            self, 
            model: nn.Module,
            x: torch.Tensor,
            y: torch.Tensor,
            stratified: bool = False,
            cv: int = 5,
            epochs: int = 100,
            batch_size: int = 32,
            metrics: list[str] = ["loss", "accuracy"],
            **kwargs: Any
    )->dict:
        """
        Cross validation
        @model: nn.Module, model to train, MANDATORY, The output model should be in the interval [-1, 1]
        @x: torch.Tensor, data, MANDATORY
        @y: torch.Tensor, labels, MANDATORY, The labels should be 0 or 1
        @stratified: bool, apply stratified cross validation or not
        @cv: int, number of folds
        @epochs: int, number of epochs
        @batch_size: int, batch size
        @metrics: list[str], list of metrics to evaluate
        """
        if stratified:
            kfold = StratifiedKFold(n_splits=cv, shuffle=True)
        else:
            kfold = KFold(n_splits=cv, shuffle=True)

        dict_metrics = {}

        for i, (train_index, test_index) in enumerate(kfold.split(x, y)):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = self.train(model, x_train, y_train, x_test=x_test, y_test=y_test, epochs=epochs, batch_size=batch_size, **kwargs)
            dict_metrics[f"fold_{i}"] = self.evaluation(model, x_test, y_test, metrics=metrics)

        return dict_metrics

    
    


   