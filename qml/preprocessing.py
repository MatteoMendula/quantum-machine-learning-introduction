from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class PreprocessingQml:
    def preprocessing(
        self,
        x: pd.DataFrame,
        y: pd.Series | list | np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        standardization: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the dataset into train/valid/test and optionally standardize.

        Returns torch.Tensor objects suitable for the rest of the code.
        """
        # First split off the test set
        x_temp, x_test, y_temp, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Split the remaining into train and validation so that validation size equals `test_size` of original
        valid_fraction = test_size / (1 - test_size)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_temp, y_temp, test_size=valid_fraction, random_state=random_state, stratify=y_temp
        )

        if standardization:
            scaler = StandardScaler()
            x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
            x_valid = pd.DataFrame(scaler.transform(x_valid), columns=x_valid.columns, index=x_valid.index)
            x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

        # Convert to torch tensors
        x_train_t = torch.Tensor(x_train.to_numpy())
        x_valid_t = torch.Tensor(x_valid.to_numpy())
        x_test_t = torch.Tensor(x_test.to_numpy())

        y_train_t = torch.Tensor(y_train).unsqueeze(1)
        y_valid_t = torch.Tensor(y_valid).unsqueeze(1)
        y_test_t = torch.Tensor(y_test).unsqueeze(1)

        return x_train_t, x_valid_t, x_test_t, y_train_t, y_valid_t, y_test_t

    def standardization(self, x: pd.DataFrame) -> pd.DataFrame:
        """Standardize a pandas DataFrame and return the transformed DataFrame."""
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)
        return df
