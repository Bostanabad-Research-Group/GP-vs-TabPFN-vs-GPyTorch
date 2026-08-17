"""
Multi-fidelity standardization utilities.

This module provides functions for standardizing multi-fidelity data using different methods:
- Method 0: Standardize all data together (original method)
- Method 1: Standardize all data based on HF data only
- Method 2: Standardize each data source independently
"""

import torch

from .standard_scaler import StandardScaler, LogScaler


def standardize_mf_data(
    X_train,
    X_test,
    y_train,
    cont_cols,
    source_cols,
    standardize_X=True,
    standardize_y=True,
    standardization_method=0,
    standardize_y_log_scale=False,
):
    """
    Standardize multi-fidelity training and test data using the specified method.

    Args:
        X_train (torch.Tensor): Training input data of shape [N_train, D]
        y_train (torch.Tensor): Training output data of shape [N_train, 1] or [N_train]
        source_indices_train (torch.Tensor): Source indices for training data of shape [N_train]
        cont_cols (list or torch.Tensor): Indices of continuous columns to standardize
        standardize_X (bool): Whether to standardize X data. Default: True
        standardize_y (bool): Whether to standardize y data. Default: True
        standardization_method (int): Standardization method to use:
            0: Standardize all data together (original method)
            1: Standardize all data based on HF data only (source 0)
            2: Standardize each data source independently
            Default: 0
        X_test (torch.Tensor, optional): Test input data of shape [N_test, D]
        source_indices_test (torch.Tensor, optional): Source indices for test data of shape [N_test]

    Returns:
        tuple:
            - If X_test provided: (X_train, X_test, y_train_normal)
            - If X_test not provided: (X_train, y_train_normal)
    """

    if standardization_method == 0:
        if standardize_X:
            Xscaler = StandardScaler()
            Xscaler.fit(X_train[:, cont_cols])
            X_train[:, cont_cols] = Xscaler.transform(X_train[:, cont_cols])
            X_test[:, cont_cols] = Xscaler.transform(X_test[:, cont_cols])
        if standardize_y:
            if standardize_y_log_scale:
                yscaler = LogScaler()
            else:
                yscaler = StandardScaler()
            yscaler.fit(y_train)
            y_train = yscaler.transform(y_train)
            y_train_mean = yscaler.mean
            y_train_std = yscaler.std
            y_train_min = yscaler.data_min if standardize_y_log_scale else None

        return X_train, X_test, y_train, y_train_mean, y_train_std, y_train_min

    elif standardization_method == 1:
        # Handle both one-hot encoded (list) and single column (int) cases
        is_onehot = isinstance(source_cols, (list, tuple)) and len(source_cols) > 1
        if is_onehot:
            # One-hot encoded: source 0 is when the first column in source_cols equals 1
            hf_mask = X_train[:, source_cols[0]] == 1
        else:
            # Single column: source 0 is when that column equals 0
            source_col = source_cols[0] if isinstance(source_cols, (list, tuple)) else source_cols
            hf_mask = X_train[:, source_col] == 0
        if standardize_X:
            X_train_hf = X_train[hf_mask]
            Xscaler = StandardScaler()
            Xscaler.fit(X_train_hf[:, cont_cols])
            X_train[:, cont_cols] = Xscaler.transform(X_train[:, cont_cols])
            X_test[:, cont_cols] = Xscaler.transform(X_test[:, cont_cols])
        if standardize_y:
            if standardize_y_log_scale:
                yscaler = LogScaler()
            else:
                yscaler = StandardScaler()
            yscaler.fit(y_train[hf_mask])
            y_train = yscaler.transform(y_train)
            y_train_mean = yscaler.mean
            y_train_std = yscaler.std
            y_train_min = yscaler.data_min if standardize_y_log_scale else None

        return X_train, X_test, y_train, y_train_mean, y_train_std, y_train_min

    elif standardization_method == 2:
        # Extract source indices from X_train (used for both X and y standardization)
        is_onehot = isinstance(source_cols, (list, tuple)) and len(source_cols) > 1
        if is_onehot:
            onehot_cols_train = X_train[:, source_cols]
            onehot_cols_test = X_test[:, source_cols]
            source_indices_train = torch.argmax(onehot_cols_train, dim=1)
            source_indices_test = torch.argmax(onehot_cols_test, dim=1)
        else:
            source_col = source_cols[0] if isinstance(source_cols, (list, tuple)) else source_cols
            source_indices_train = X_train[:, source_col].long()
            source_indices_test = X_test[:, source_col].long()

        if standardize_X:
            unique_sources_train = torch.unique(source_indices_train)
            for source_idx in unique_sources_train:
                source_mask = source_indices_train == source_idx
                source_mask_test = source_indices_test == source_idx
                X_train_source = X_train[source_mask]
                if len(X_train_source) > 0:
                    X_scaler = StandardScaler()
                    X_scaler.fit(X_train_source[:, cont_cols])
                    # Transform and assign back - need to do this carefully to avoid indexing error
                    X_train_transformed = X_train_source.clone()
                    X_train_transformed[:, cont_cols] = X_scaler.transform(X_train_source[:, cont_cols])
                    X_train[source_mask] = X_train_transformed

                    X_test_source = X_test[source_mask_test]
                    if len(X_test_source) > 0:
                        X_test_transformed = X_test_source.clone()
                        X_test_transformed[:, cont_cols] = X_scaler.transform(X_test_source[:, cont_cols])
                        X_test[source_mask_test] = X_test_transformed

        if standardize_y:
            # Use already extracted source_indices_train
            unique_sources = torch.unique(source_indices_train)
            y_train_normal = torch.zeros_like(y_train)
            y_train_means = {}
            y_train_stds = {}
            y_train_mins = {}
            for source_idx in unique_sources:
                source_mask = source_indices_train == source_idx
                y_train_source = y_train[source_mask]
                
                if standardize_y_log_scale:
                    y_scaler = LogScaler()
                else:
                    y_scaler = StandardScaler()
                
                y_scaler.fit(y_train_source)
                y_train_normal[source_mask] = y_scaler.transform(y_train_source)
                
                # Store mean, std, and min for this source
                y_train_means[source_idx.item()] = y_scaler.mean.squeeze() if y_scaler.mean is not None else None
                y_train_stds[source_idx.item()] = y_scaler.std.squeeze() if y_scaler.std is not None else None
                if standardize_y_log_scale:
                    y_train_mins[source_idx.item()] = y_scaler.data_min.squeeze() if y_scaler.data_min is not None else None
                else:
                    y_train_mins[source_idx.item()] = None

            # For method 2, return dictionary of means/stds/mins for each source
            y_train_mean = y_train_means
            y_train_std = y_train_stds
            y_train_min = y_train_mins if standardize_y_log_scale else None
        else:
            y_train_normal = y_train
            y_train_mean = None
            y_train_std = None
            y_train_min = None

        return X_train, X_test, y_train_normal, y_train_mean, y_train_std, y_train_min
