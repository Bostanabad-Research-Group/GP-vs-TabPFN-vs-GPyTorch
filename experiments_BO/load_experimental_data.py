import torch
import os
import pandas as pd
import numpy as np
from math import sqrt
import torch.nn.functional as F


def wing_mixed_variables(X: torch.Tensor, source: str = "s0") -> torch.Tensor:
    """
    Local copy of the wing function with source-specific variants.
    X shape: (n, 10)
    """
    Sw = X[..., 0]
    Wfw = X[..., 1]
    A = X[..., 2]
    Gama = X[..., 3] * (torch.pi / 180.0)
    q = X[..., 4]
    lamb = X[..., 5]
    tc = X[..., 6]
    Nz = X[..., 7]
    Wdg = X[..., 8]
    Wp = X[..., 9]
    cos_Gama = torch.cos(Gama)

    if source == "s0":
        return (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + Sw * Wp
        )
    elif source == "s1":
        return (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 1 * Wp
        )
    elif source == "s2":
        return (
            0.036
            * Sw**0.8
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 1 * Wp
        )
    elif source == "s3":
        return (
            0.036
            * Sw**0.9
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 0 * Wp
        )
    else:
        raise ValueError(f"Unknown source: {source}")


def generate_mf_wing_data(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                         seed: int = None, train_noise: list[float] = None, test_noise: list[float] = None, 
                         noise_type: str = 'gaussian') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Wing data by drawing a single Sobol batch (no repeats),
    then splitting into test/train per source. Compute test std after the split
    and scale both train and test noise by that std.

    Returns:
      - X_train, y_train: Training data with 11 features (10 continuous + 1 source class column in {0,1,2,3})
      - X_test, y_test: Test data with 11 features (10 continuous + 1 source class column in {0,1,2,3})
    """
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = 42
        torch.manual_seed(seed)

    # Defaults and validation for per-source noise (4 sources)
    if train_noise is None:
        train_noise = [0.0, 0.0, 0.0, 0.0]
    if test_noise is None:
        test_noise = [0.0, 0.0, 0.0, 0.0]
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * 4
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * 4
    if len(train_noise) != 4 or len(test_noise) != 4:
        raise ValueError("train_noise and test_noise must be length-4 (one scalar per source)")

    sources = ["s0", "s1", "s2", "s3"]

    # Bounds for the 10 continuous features
    l_bound = torch.tensor([150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025], dtype=torch.float64)
    u_bound = torch.tensor([200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08], dtype=torch.float64)

    # Total samples per source and overall
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    # Draw all Sobol samples at once (scrambled => randomized QMC) and scale to bounds
    sobol = torch.quasirandom.SobolEngine(dimension=10, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Assign contiguous blocks per source (no repeats globally)
    src_indices = []
    start = 0
    for idx, n in enumerate(total_per_source):
        src_indices.extend([idx] * n)
        start += n
    src_indices_tensor = torch.tensor(src_indices, dtype=torch.long)

    # Compute clean targets once per source
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    offset = 0
    for idx, (src, n) in enumerate(zip(sources, total_per_source)):
        if n == 0:
            continue
        x_block = X_raw_all[offset:offset + n]
        y_clean_all[offset:offset + n] = wing_mixed_variables(x_block, source=src)
        offset += n

    # Split per source into test then train; get test std after split
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]

        # Split: first n_test -> test, remaining -> train
        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 10), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 10), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)

        # Test std after split (per source) as a Python float
        test_std_value: float
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_train_block) * (train_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_train_block) - 0.5) * 2 * (train_noise[idx] * test_std_value) * sqrt(3)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_test_block) * (test_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_test_block) - 0.5) * 2 * (test_noise[idx] * test_std_value) * sqrt(3)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_test_block = y_test_block + noise

        # Append source id as 11th feature
        if n_train > 0:
            src_col_train = torch.full((n_train, 1), float(idx), dtype=torch.float64)
            X_train_list.append(torch.cat([x_train_block, src_col_train], dim=1))
            y_train_list.append(y_train_block)

        if n_test > 0:
            src_col_test = torch.full((n_test, 1), float(idx), dtype=torch.float64)
            X_test_list.append(torch.cat([x_test_block, src_col_test], dim=1))
            y_test_list.append(y_test_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 11), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 11), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


def buckling_mixed_variables(X: torch.Tensor, source: str = "s0") -> torch.Tensor:
    """
    Compute buckling load given input variables.
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, 4] with columns:
            0: L (length of the beam, m)
            1: E (Young's modulus, Pa) 
            2: K (shear modulus, Pa)
            3: I (moment of inertia, m^4)
        source (str): Source of the data ('s0' or 's1')
    Returns:
        torch.Tensor: Buckling load values for each input sample
    """
    L = X[..., 0]
    E = X[..., 1]
    K = X[..., 2]
    I = X[..., 3]

    # Buckling load calculation
    if source == "s0":
        P = torch.pi * E * I / (L * K) ** 2
    elif source == "s1":
        P = ((torch.pi * E * I / (L * K) ** 2) + L) ** 1.1
    else:
        raise ValueError(f"Unknown source: {source}. Only 's0' and 's1' are supported for buckling.")

    return P


def generate_mf_buckling_data_with_folds(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                                         num_runs: int = 4, seed: int = None, train_noise: list[float] = None, 
                                         test_noise: list[float] = None, noise_type: str = 'gaussian', 
                                         return_categorical: bool = True) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Buckling data with pre-stratified folds:
      - Use Sobol sequences to produce EVEN amounts of E, I, and K categorical inputs
      - Generate train data directly as num_runs with even categorical distributions
      - Generate test data with even categorical distributions
      - Each fold has perfectly balanced categorical distributions
    """
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = torch.randint(0, 1000000)
        torch.manual_seed(seed)
    
    # Default noise values
    if train_noise is None:
        train_noise = [0.0] * len(train_samples_per_source)
    if test_noise is None:
        test_noise = [0.0] * len(test_samples_per_source)
    
    # Validate inputs
    if len(train_samples_per_source) != len(test_samples_per_source):
        raise ValueError("train_samples_per_source and test_samples_per_source must have same length")
    if len(train_noise) != len(train_samples_per_source):
        raise ValueError("train_noise must be length-2 (one scalar per source)")
    if len(test_noise) != len(test_samples_per_source):
        raise ValueError("test_noise must be length-2 (one scalar per source)")
    
    sources = ['s0', 's1']  # Two sources
    
    # Categorical index values (0-based) and actual physical values
    # Use strictly positive physical values to avoid 0/0 or division-by-zero in buckling formula
    E_values = torch.tensor([0, 1], dtype=torch.long)  # category indices
    K_values = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # category indices
    I_values = torch.tensor([0, 1, 2], dtype=torch.long)  # category indices
    # Actual physical values for buckling problem
    E_phys = torch.tensor([73.1, 200.0], dtype=torch.float64)  # Young's modulus values
    K_phys = torch.tensor([0.5, 0.7, 1.0, 2.0], dtype=torch.float64)  # Shear modulus values
    I_phys = torch.tensor([9.49, 12.1, 29.5], dtype=torch.float64)  # Moment of inertia values
    
    # Generate all continuous L values per source at once (test + train) using single seed per source
    # This matches the pattern used in other problems: draw all samples for a source at once, then split
    # Use seed offset per source to ensure each source gets a unique sequence
    L_vals_per_source = {}
    for src_idx, (n_test, n_train) in enumerate(zip(test_samples_per_source, train_samples_per_source)):
        total_n = n_test + n_train
        if total_n == 0:
            continue
        # Use seed offset per source to get unique sequences (but consistent within source)
        # This ensures test and train are contiguous within each source's sequence
        sobol_seed = (seed + src_idx * 1000) if seed is not None else None
        sobol = torch.quasirandom.SobolEngine(1, scramble=True, seed=sobol_seed)
        # Draw all samples for this source at once (test + train together)
        L_vals_all = sobol.draw(total_n).squeeze() + 0.5  # L in [0.5, 1.5]
        L_vals_per_source[src_idx] = L_vals_all
    
    # Generate all data (test + train) per source, compute targets once, then split
    X_test_list = []
    y_test_list = []
    X_train_folds = []
    y_train_folds = []
    test_std_per_source = {}  # Store test std for each source
    
    total_train_samples = sum(train_samples_per_source)
    if total_train_samples > 0:
        # Pre-allocate lists for all folds
        for _ in range(num_runs):
            X_train_folds.append([])
            y_train_folds.append([])
    
    for src_idx, (src, n_test, n_train) in enumerate(zip(sources, test_samples_per_source, train_samples_per_source)):
        total_n = n_test + n_train
        if total_n == 0:
            continue
        
        L_vals_all = L_vals_per_source[src_idx]
        all_cat_assignments = []
        
        # Generate test categorical assignments (even distribution)
        if n_test > 0:
            # E values (2 options)
            n_per_E_test = n_test // len(E_values)
            remaining_E_test = n_test % len(E_values)
            E_indices_test = []
            for i in range(len(E_values)):
                count = n_per_E_test + (1 if i < remaining_E_test else 0)
                E_indices_test.append(torch.full((count,), i))
            E_indices_test = torch.cat(E_indices_test)
            E_indices_test = E_indices_test[torch.randperm(n_test)]
            
            # K values (4 options)
            n_per_K_test = n_test // len(K_values)
            remaining_K_test = n_test % len(K_values)
            K_indices_test = []
            for i in range(len(K_values)):
                count = n_per_K_test + (1 if i < remaining_K_test else 0)
                K_indices_test.append(torch.full((count,), i))
            K_indices_test = torch.cat(K_indices_test)
            K_indices_test = K_indices_test[torch.randperm(n_test)]
            
            # I values (3 options)
            n_per_I_test = n_test // len(I_values)
            remaining_I_test = n_test % len(I_values)
            I_indices_test = []
            for i in range(len(I_values)):
                count = n_per_I_test + (1 if i < remaining_I_test else 0)
                I_indices_test.append(torch.full((count,), i))
            I_indices_test = torch.cat(I_indices_test)
            I_indices_test = I_indices_test[torch.randperm(n_test)]
            
            # Store test assignments
            for i in range(n_test):
                all_cat_assignments.append({
                    'e': int(E_indices_test[i].item()),
                    'k': int(K_indices_test[i].item()),
                    'i': int(I_indices_test[i].item())
                })
        
        # Generate train categorical assignments (per fold with exact distributions)
        if n_train > 0:
            target_per_fold = n_train // num_runs
            remainder = n_train % num_runs
            num_E = len(E_values)
            num_K = len(K_values)
            num_I = len(I_values)
            
            for fold in range(num_runs):
                fold_target = target_per_fold + (1 if fold < remainder else 0)
                
                # Calculate EXACT counts for each categorical value in this fold
                E_base = fold_target // num_E
                E_rem = fold_target % num_E
                E_counts = [E_base + (1 if i < E_rem else 0) for i in range(num_E)]
                
                K_base = fold_target // num_K
                K_rem = fold_target % num_K
                K_counts = [K_base + (1 if i < K_rem else 0) for i in range(num_K)]
                
                I_base = fold_target // num_I
                I_rem = fold_target % num_I
                rotation_offset = (fold + src_idx * num_runs) % num_I
                I_counts = [I_base + (1 if (i + rotation_offset) % num_I < I_rem else 0) for i in range(num_I)]
                
                # Build assignments for this fold
                cat_assignments = []
                for e_idx in range(num_E):
                    for _ in range(E_counts[e_idx]):
                        cat_assignments.append({'e': e_idx})
                
                k_list = []
                for k_idx in range(num_K):
                    for _ in range(K_counts[k_idx]):
                        k_list.append(k_idx)
                if seed is not None:
                    torch.manual_seed(seed + src_idx * 2000 + fold * 100 + 1)
                k_perm = torch.randperm(len(k_list))
                k_list = [k_list[i] for i in k_perm.tolist()]
                
                i_list = []
                for i_idx in range(num_I):
                    for _ in range(I_counts[i_idx]):
                        i_list.append(i_idx)
                if seed is not None:
                    torch.manual_seed(seed + src_idx * 2000 + fold * 100 + 2)
                i_perm = torch.randperm(len(i_list))
                i_list = [i_list[i] for i in i_perm.tolist()]
                
                for i in range(fold_target):
                    cat_assignments[i]['k'] = k_list[i]
                    cat_assignments[i]['i'] = i_list[i]
                
                if seed is not None:
                    torch.manual_seed(seed + src_idx * 2000 + fold * 100 + 3)
                perm = torch.randperm(len(cat_assignments))
                cat_assignments = [cat_assignments[i] for i in perm.tolist()]
                
                all_cat_assignments.extend(cat_assignments)
        
        # Build all data (test + train) for this source at once
        x_all = torch.zeros((total_n, 4), dtype=torch.float64)
        for i, assignment in enumerate(all_cat_assignments):
            x_all[i, 0] = L_vals_all[i]
            x_all[i, 1] = E_phys[assignment['e']]
            x_all[i, 2] = K_phys[assignment['k']]
            x_all[i, 3] = I_phys[assignment['i']]
        
        # Compute targets ONCE for all data (test + train)
        y_all = buckling_mixed_variables(x_all, source=src)
        
        # Convert to categorical indices if requested (AFTER computing y)
        if return_categorical:
            for i, assignment in enumerate(all_cat_assignments):
                x_all[i, 1] = float(assignment['e'])
                x_all[i, 2] = float(assignment['k'])
                x_all[i, 3] = float(assignment['i'])
        
        # Split into test and train
        if n_test > 0:
            x_test_block = x_all[:n_test]
            y_test_clean = y_all[:n_test]
            
            # Compute test std for noise scaling
            if y_test_clean.numel() > 1:
                test_std_value = float(y_test_clean.std().item())
            else:
                test_std_value = 0.0
            test_std_per_source[src_idx] = test_std_value
            
            # Add noise to test data
            y_test_block = y_test_clean.clone()
            if test_noise[src_idx] > 0 and test_std_value > 0.0:
                if noise_type == 'gaussian':
                    noise = torch.randn_like(y_test_block) * (test_noise[src_idx] * test_std_value)
                elif noise_type == 'uniform':
                    noise = (torch.rand_like(y_test_block) - 0.5) * 2 * (test_noise[src_idx] * test_std_value) * sqrt(3)
                else:
                    raise ValueError(f"Unknown noise_type: {noise_type}")
                y_test_block = y_test_block + noise
            
            source_column = torch.full((x_test_block.shape[0], 1), src_idx, dtype=torch.float64)
            X_test_list.append(torch.cat([x_test_block, source_column], dim=1))
            y_test_list.append(y_test_block)
        
        if n_train > 0:
            x_train_all = x_all[n_test:]
            y_train_all = y_all[n_test:]
            
            # Add noise to train data
            test_std_value = test_std_per_source.get(src_idx, 0.0)
            if train_noise[src_idx] > 0 and test_std_value > 0.0:
                if noise_type == 'gaussian':
                    noise = torch.randn_like(y_train_all) * (train_noise[src_idx] * test_std_value)
                elif noise_type == 'uniform':
                    noise = (torch.rand_like(y_train_all) - 0.5) * 2 * (train_noise[src_idx] * test_std_value) * sqrt(3)
                else:
                    raise ValueError(f"Unknown noise_type: {noise_type}")
                y_train_all = y_train_all + noise
            
            # Split train into folds
            target_per_fold = n_train // num_runs
            remainder = n_train % num_runs
            fold_start = 0
            for fold in range(num_runs):
                fold_target = target_per_fold + (1 if fold < remainder else 0)
                fold_end = fold_start + fold_target
                
                x_fold = x_train_all[fold_start:fold_end]
                y_fold = y_train_all[fold_start:fold_end]
                
                source_column = torch.full((x_fold.shape[0], 1), src_idx, dtype=torch.float64)
                x_fold_with_source = torch.cat([x_fold, source_column], dim=1)
                
                X_train_folds[fold].append(x_fold_with_source)
                y_train_folds[fold].append(y_fold)
                
                fold_start = fold_end
    
    # Combine test data
    X_test_all = torch.cat(X_test_list, dim=0)
    y_test_all = torch.cat(y_test_list, dim=0)
    
    # Concatenate folds from all sources
    for fold in range(num_runs):
        X_train_folds[fold] = torch.cat(X_train_folds[fold], dim=0)
        y_train_folds[fold] = torch.cat(y_train_folds[fold], dim=0)
    
    return X_train_folds, y_train_folds, X_test_all, y_test_all


def generate_mf_buckling_data(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                              seed: int = None, train_noise: list[float] = None, test_noise: list[float] = None, 
                              noise_type: str = 'gaussian', return_categorical: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Buckling data following the data_gen.py method:
      - Use Sobol sequences to produce EVEN amounts of E, I, and K categorical inputs
      - Draw a single Sobol batch per source (no repeats globally)
      - Split into test then train per source
      - Compute per-source test std after the split
      - Scale both train and test additive noise by that test std

    Returns:
      - X_train, y_train: Training data with 5 features (4 continuous + 1 source class column in {0,1})
        Where columns are [L (cont), E, K, I (categorical or values per return_categorical), source]
      - X_test, y_test: Test data with 5 features (same schema as X_train)
    """
    if seed is not None:
        torch.manual_seed(seed)
    # else:
    #     seed = 42
    #     torch.manual_seed(seed)

    # Defaults and validation for per-source noise (2 sources)
    if train_noise is None:
        train_noise = [0.0, 0.0]
    if test_noise is None:
        test_noise = [0.0, 0.0]
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * 2
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * 2
    if len(train_noise) != 2 or len(test_noise) != 2:
        raise ValueError("train_noise and test_noise must be length-2 (one scalar per source)")

    sources = ["s0", "s1"]

    # Bounds for the 4 continuous features (L, E, K, I)
    l_bound = torch.tensor([0.5, 73.1, 0.5, 9.49], dtype=torch.float64)
    u_bound = torch.tensor([1.5, 200.0, 2.0, 29.5], dtype=torch.float64)

    # Define specific categorical values (same as data_gen.py)
    E_values = torch.tensor([73.1, 200.0], dtype=torch.float64)  # Column 1: E can only be 73.1 or 200
    K_values = torch.tensor([0.5, 0.7, 1.0, 2.0], dtype=torch.float64)  # Column 2: K can only be 0.5, 0.7, 1, or 2
    I_values = torch.tensor([9.49, 12.1, 29.5], dtype=torch.float64)  # Column 3: I can only be 9.49, 12.1, or 29.5

    # Total samples per source and overall
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    # Draw all Sobol samples at once and scale to bounds
    sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Compute clean targets once per source in contiguous blocks
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    X_src_col_all = torch.empty((total_n, 1), dtype=torch.float64)
    cursor = 0
    
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        
        # Generate TEST data with even distribution
        if n_test > 0:
            x_test_block = X_raw_all[cursor:cursor + n_test].clone()
            
            # Column 1: E values (2 options) - ensure even distribution for TEST
            n_per_E_test = n_test // len(E_values)
            remaining_E_test = n_test % len(E_values)
            E_indices_test = []
            for i in range(len(E_values)):
                count = n_per_E_test + (1 if i < remaining_E_test else 0)
                E_indices_test.append(torch.full((count,), i))
            E_indices_test = torch.cat(E_indices_test)
            E_indices_test = E_indices_test[torch.randperm(n_test)]
            x_test_block[:, 1] = E_values[E_indices_test]

            # Column 2: K values (4 options) - ensure even distribution for TEST
            n_per_K_test = n_test // len(K_values)
            remaining_K_test = n_test % len(K_values)
            K_indices_test = []
            for i in range(len(K_values)):
                count = n_per_K_test + (1 if i < remaining_K_test else 0)
                K_indices_test.append(torch.full((count,), i))
            K_indices_test = torch.cat(K_indices_test)
            K_indices_test = K_indices_test[torch.randperm(n_test)]
            x_test_block[:, 2] = K_values[K_indices_test]

            # Column 3: I values (3 options) - ensure even distribution for TEST
            n_per_I_test = n_test // len(I_values)
            remaining_I_test = n_test % len(I_values)
            I_indices_test = []
            for i in range(len(I_values)):
                count = n_per_I_test + (1 if i < remaining_I_test else 0)
                I_indices_test.append(torch.full((count,), i))
            I_indices_test = torch.cat(I_indices_test)
            I_indices_test = I_indices_test[torch.randperm(n_test)]
            x_test_block[:, 3] = I_values[I_indices_test]

            # Compute targets for test data
            y_test_clean = buckling_mixed_variables(x_test_block, source=src)
            
            # Store categorical indices if requested
            if return_categorical:
                x_test_block[:, 1] = E_indices_test.to(torch.float64)
                x_test_block[:, 2] = K_indices_test.to(torch.float64)
                x_test_block[:, 3] = I_indices_test.to(torch.float64)
            
            # Store test data
            y_clean_all[cursor:cursor + n_test] = y_test_clean
            X_raw_all[cursor:cursor + n_test] = x_test_block
            X_src_col_all[cursor:cursor + n_test, 0] = float(idx)

        # Generate TRAIN data with even distribution
        if n_train > 0:
            x_train_block = X_raw_all[cursor + n_test:cursor + n_total].clone()
            
            # Column 1: E values (2 options) - ensure even distribution for TRAIN
            n_per_E_train = n_train // len(E_values)
            remaining_E_train = n_train % len(E_values)
            E_indices_train = []
            for i in range(len(E_values)):
                count = n_per_E_train + (1 if i < remaining_E_train else 0)
                E_indices_train.append(torch.full((count,), i))
            E_indices_train = torch.cat(E_indices_train)
            E_indices_train = E_indices_train[torch.randperm(n_train)]
            x_train_block[:, 1] = E_values[E_indices_train]

            # Column 2: K values (4 options) - ensure even distribution for TRAIN
            n_per_K_train = n_train // len(K_values)
            remaining_K_train = n_train % len(K_values)
            K_indices_train = []
            for i in range(len(K_values)):
                count = n_per_K_train + (1 if i < remaining_K_train else 0)
                K_indices_train.append(torch.full((count,), i))
            K_indices_train = torch.cat(K_indices_train)
            K_indices_train = K_indices_train[torch.randperm(n_train)]
            x_train_block[:, 2] = K_values[K_indices_train]

            # Column 3: I values (3 options) - ensure even distribution for TRAIN
            n_per_I_train = n_train // len(I_values)
            remaining_I_train = n_train % len(I_values)
            I_indices_train = []
            for i in range(len(I_values)):
                count = n_per_I_train + (1 if i < remaining_I_train else 0)
                I_indices_train.append(torch.full((count,), i))
            I_indices_train = torch.cat(I_indices_train)
            I_indices_train = I_indices_train[torch.randperm(n_train)]
            x_train_block[:, 3] = I_values[I_indices_train]

            # Compute targets for train data
            y_train_clean = buckling_mixed_variables(x_train_block, source=src)
            
            # Store categorical indices if requested
            if return_categorical:
                x_train_block[:, 1] = E_indices_train.to(torch.float64)
                x_train_block[:, 2] = K_indices_train.to(torch.float64)
                x_train_block[:, 3] = I_indices_train.to(torch.float64)
            
            # Store train data
            y_clean_all[cursor + n_test:cursor + n_total] = y_train_clean
            X_raw_all[cursor + n_test:cursor + n_total] = x_train_block
            X_src_col_all[cursor + n_test:cursor + n_total, 0] = float(idx)

        cursor += n_total

    # Split per source into test then train; get test std after split and add noise scaled by it
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]
        src_block = X_src_col_all[cursor:cursor + n_total]

        # Split: first n_test -> test, remaining -> train
        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 4), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        src_test_block = src_block[:n_test] if n_test > 0 else torch.empty((0, 1), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 4), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)
        src_train_block = src_block[n_test:] if n_train > 0 else torch.empty((0, 1), dtype=torch.float64)

        # Test std after split (per source)
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_train_block) * (train_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_train_block) - 0.5) * 2 * (train_noise[idx] * test_std_value) * sqrt(3)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_test_block) * (test_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_test_block) - 0.5) * 2 * (test_noise[idx] * test_std_value) * sqrt(3)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_test_block = y_test_block + noise

        # Append source column and collect
        X_test_list.append(torch.cat([x_test_block, src_test_block], dim=1))
        y_test_list.append(y_test_block)
        X_train_list.append(torch.cat([x_train_block, src_train_block], dim=1))
        y_train_list.append(y_train_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 5), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 5), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


def borehole_mixed_variables(X: torch.Tensor, source: str = "s0") -> torch.Tensor:
    """
    Compute borehole water flow rate given input variables (torch implementation).

    Args:
        X (torch.Tensor): Input array of shape [n_samples, 8] with columns:
            0: rw (radius of borehole, m)
            1: r (radius of influence, m)
            2: Tu (transmissivity of upper aquifer, m^2/yr)
            3: Hu (potentiometric head of upper aquifer, m)
            4: Tl (transmissivity of lower aquifer, m^2/yr)
            5: Hl (potentiometric head of lower aquifer, m)
            6: L (length of borehole, m)
            7: Kw (hydraulic conductivity of borehole, m/yr)
        source (str): Source/fidelity level ("s0".."s4")
    Returns:
        torch.Tensor: Flow rate values for each input sample
    """
    rw = X[..., 0]
    r = X[..., 1]
    Tu = X[..., 2]
    Hu = X[..., 3]
    Tl = X[..., 4]
    Hl = X[..., 5]
    L = X[..., 6]
    Kw = X[..., 7]

    if source == "s0":
        numerator = 2 * torch.pi * Tu * (Hu - Hl)
        denominator = torch.log(r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == "s1":
        numerator = 2 * torch.pi * Tu * (Hu - 0.8 * Hl)
        denominator = torch.log(r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == "s2":
        numerator = 2 * torch.pi * Tu * (Hu - 3 * Hl)
        denominator = torch.log(r / rw) * (1 + 8 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + 0.75 * Tu / Tl)
        result = numerator / denominator
    elif source == "s3":
        numerator = 2 * torch.pi * Tu * (1.1 * Hu - Hl)
        denominator = torch.log(4 * r / rw) * (1 + 3 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == "s4":
        numerator = 2 * torch.pi * Tu * (1.05 * Hu - Hl)
        denominator = torch.log(2 * r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    else:
        raise ValueError(f"Unknown source: {source}. Only s0..s4 are supported for borehole.")

    return result


def generate_mf_borehole_data(
    train_samples_per_source: list[int],
    test_samples_per_source: list[int],
    *,
    seed: int | None = None,
    train_noise: list[float] | float | None = None,
    test_noise: list[float] | float | None = None,
    noise_type: str = 'gaussian',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Borehole data consistent with wing/buckling helpers.

    - Draw a single Sobol batch across all sources (no repeats globally)
    - Split per source: first test, remaining train
    - Scale additive noise by per-source test std (post-split), like wing/buckling

    Returns:
      - X_train, y_train with 9 features (8 continuous + 1 numeric source column)
      - X_test, y_test with same schema
    """
    if seed is not None:
        torch.manual_seed(seed)
    # else:
    #     seed = 42
    #     torch.manual_seed(seed)

    # Defaults and validation for per-source noise (5 sources)
    num_sources = 5
    if train_noise is None:
        train_noise = [0.0] * num_sources
    if test_noise is None:
        test_noise = [0.0] * num_sources
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * num_sources
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * num_sources
    if len(train_noise) != num_sources or len(test_noise) != num_sources:
        raise ValueError(f"train_noise and test_noise must be length-{num_sources} (one scalar per source)")

    sources = ["s0", "s1", "s2", "s3", "s4"]

    # Bounds for the 8 continuous features
    l_bound = torch.tensor([0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0], dtype=torch.float64)
    u_bound = torch.tensor([0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0], dtype=torch.float64)

    # Totals and Sobol draws
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    sobol = torch.quasirandom.SobolEngine(dimension=8, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Compute clean targets per contiguous source block
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    src_ids_all = torch.empty((total_n, 1), dtype=torch.float64)
    cursor = 0
    for idx, (src, n) in enumerate(zip(sources, total_per_source)):
        if n == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n]
        y_clean_all[cursor:cursor + n] = borehole_mixed_variables(x_block, source=src)
        src_ids_all[cursor:cursor + n, 0] = float(idx)
        cursor += n

    # Split into test then train per source; add scaled noise
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]
        s_block = src_ids_all[cursor:cursor + n_total]

        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 8), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        s_test_block = s_block[:n_test] if n_test > 0 else torch.empty((0, 1), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 8), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)
        s_train_block = s_block[n_test:] if n_train > 0 else torch.empty((0, 1), dtype=torch.float64)

        # Per-source test std
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_train_block) * (train_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_train_block) - 0.5) * 2 * (train_noise[idx] * test_std_value) * sqrt(3)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_test_block) * (test_noise[idx] * test_std_value)
            elif noise_type == 'uniform':
                noise = (torch.rand_like(y_test_block) - 0.5) * 2 * (test_noise[idx] * test_std_value) * sqrt(3)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_test_block = y_test_block + noise

        # Append numeric source id as 9th feature
        if n_train > 0:
            X_train_list.append(torch.cat([x_train_block, s_train_block], dim=1))
            y_train_list.append(y_train_block)
        if n_test > 0:
            X_test_list.append(torch.cat([x_test_block, s_test_block], dim=1))
            y_test_list.append(y_test_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 9), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 9), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


def ackley_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Ackley function for given input variables.
    
    The Ackley function is defined as:
    f(x) = -20 * exp(-0.2 * sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e
    
    where x ∈ [-32.768, 32.768]^d and d is the number of dimensions
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Ackley function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Constants
    a = 20.0
    b = 0.2
    c = 2 * torch.pi
    
    # Compute the two main terms
    sum_squares = torch.sum(X**2, dim=1)
    sum_cos = torch.sum(torch.cos(c * X), dim=1)
    
    # Ackley function
    term1 = -a * torch.exp(-b * torch.sqrt(sum_squares / dimensions))
    term2 = -torch.exp(sum_cos / dimensions)
    result = term1 + term2 + a + torch.e
    
    return result


def generate_ackley_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-5, 10], train_noise: float = 0.0, 
                        test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None, V2: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Ackley function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Ackley function
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Ackley bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Ackley function values
    y_all = ackley_function(X_all, dimensions)

    if V2:
        y_all = torch.log(y_all+1)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_train) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_train) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_test) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_test) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def rosenbrock_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Rosenbrock function for given input variables.
    
    The Rosenbrock function is defined as:
    f(x) = sum_{i=1}^{d-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    
    where x ∈ [-5, 10]^d and d is the number of dimensions
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Rosenbrock function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Rosenbrock function: sum over i=1 to d-1 of [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    # Vectorized computation: X[:, :-1] are x_i, X[:, 1:] are x_{i+1}
    x_i = X[:, :-1]  # All x_i for i=0 to d-2
    x_i_plus_1 = X[:, 1:]  # All x_{i+1} for i=0 to d-2
    
    term1 = 100 * (x_i_plus_1 - x_i**2)**2
    term2 = (1 - x_i)**2
    result = torch.sum(term1 + term2, dim=1)
    
    return result


def generate_rosenbrock_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-5, 10], train_noise: float = 0.0, 
                            test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Rosenbrock function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Rosenbrock function
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Rosenbrock bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Rosenbrock function values
    y_all = rosenbrock_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_train) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_train) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_test) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_test) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def zakharov_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Zakharov function for given input variables.
    
    The Zakharov function is defined as:
    f(x) = sum_{i=1}^{d} x_i^2 + (sum_{i=1}^{d} 0.5*i*x_i)^2 + (sum_{i=1}^{d} 0.5*i*x_i)^4
    
    where x ∈ [-5, 10]^d and d is the number of dimensions
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Zakharov function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # First term: sum of squares
    term1 = torch.sum(X**2, dim=1)
    
    # Second and third terms: sum of 0.5*i*x_i
    # Create indices [0.5, 1.0, 1.5, ..., 0.5*d] for each sample
    i_values = torch.arange(1, dimensions + 1, dtype=X.dtype, device=X.device) * 0.5
    weighted_sum = torch.sum(X * i_values.unsqueeze(0), dim=1)
    
    # Second term: (weighted_sum)^2
    term2 = weighted_sum**2
    
    # Third term: (weighted_sum)^4
    term3 = weighted_sum**4
    
    result = term1 + term2 + term3
    
    return result


def generate_zakharov_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-5, 10], train_noise: float = 0.0, 
                            test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Zakharov function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Zakharov function
        x_bounds (list[float]): Bounds for each dimension [lower, upper] (default: [-5, 10])
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Zakharov bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Zakharov function values
    y_all = zakharov_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_train) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_train) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_test) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_test) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def griewank_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Griewank function for given input variables.
    
    The Griewank function is defined as:
    f(x) = sum_{i=1}^{d} (x_i^2 / 4000) - product_{i=1}^{d} cos(x_i / sqrt(i)) + 1
    
    where x ∈ [-600, 600]^d and d is the number of dimensions
    
    Global minimum: f(x*) = 0, at x* = (0,...,0)
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Griewank function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # First term: sum_{i=1}^{d} (x_i^2 / 4000)
    sum_squares = torch.sum(X**2, dim=1) / 4000.0
    
    # Second term: product_{i=1}^{d} cos(x_i / sqrt(i))
    # Create indices [1, 2, 3, ..., d] for each sample
    i_values = torch.arange(1, dimensions + 1, dtype=X.dtype, device=X.device)
    sqrt_i = torch.sqrt(i_values)  # [sqrt(1), sqrt(2), ..., sqrt(d)]
    
    # Compute cos(x_i / sqrt(i)) for each dimension
    cos_terms = torch.cos(X / sqrt_i.unsqueeze(0))  # [n_samples, d]
    
    # Product over all dimensions
    product_cos = torch.prod(cos_terms, dim=1)  # [n_samples]
    
    # Griewank function: sum_squares - product_cos + 1
    result = sum_squares - product_cos + 1.0
    
    return result


def generate_griewank_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-600, 600], train_noise: float = 0.0, 
                            test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Griewank function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Griewank function
        x_bounds (list[float]): Bounds for each dimension [lower, upper] (default: [-600, 600])
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Griewank bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Griewank function values
    y_all = griewank_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_train) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_train) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_test) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_test) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def dixon_price_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Dixon-Price function for given input variables.
    
    The Dixon-Price function is defined as:
    f(x) = (x_1 - 1)^2 + sum_{i=2}^{d} i * (2x_i^2 - x_{i-1})^2
    
    where x ∈ [-10, 10]^d and d is the number of dimensions
    
    Global minimum: f(x*) = 0, at x_i = 2^((2^(i-1) - 1) / 2^(i-1)) for i = 1, ..., d
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Dixon-Price function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # First term: (x_1 - 1)^2
    first_term = (X[:, 0] - 1.0) ** 2
    
    # Sum term: sum_{i=2}^{d} i * (2x_i^2 - x_{i-1})^2
    # For i=2 to d: i * (2x_i^2 - x_{i-1})^2
    # x_i corresponds to X[:, i-1] (0-indexed)
    # x_{i-1} corresponds to X[:, i-2] (0-indexed)
    
    sum_term = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
    
    for i in range(2, dimensions + 1):  # i from 2 to d
        x_i = X[:, i - 1]  # x_i (0-indexed: column i-1)
        x_i_minus_1 = X[:, i - 2]  # x_{i-1} (0-indexed: column i-2)
        
        # Compute (2x_i^2 - x_{i-1})^2
        term = (2.0 * x_i**2 - x_i_minus_1) ** 2
        
        # Multiply by i and add to sum
        sum_term += float(i) * term
    
    # Dixon-Price function: first_term + sum_term
    result = first_term + sum_term
    
    return result


def generate_dixon_price_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-10, 10], train_noise: float = 0.0, 
                               test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Dixon-Price function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Dixon-Price function
        x_bounds (list[float]): Bounds for each dimension [lower, upper] (default: [-10, 10])
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Dixon-Price bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Dixon-Price function values
    y_all = dixon_price_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_train) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_train) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        if noise_type == 'gaussian':
            noise = torch.randn_like(y_test) * noise_scale
        elif noise_type == 'uniform':
            noise = (torch.rand_like(y_test) - 0.5) * 2 * noise_scale * sqrt(3)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'uniform'")
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


