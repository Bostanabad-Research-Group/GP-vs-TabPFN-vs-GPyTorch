import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math
import colorsys

import numpy as np
# import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
# import seaborn as sns
from scipy.stats import norm
import json
from typing import Dict, List, Optional, Tuple, Any

def plot_violin(df_results, save_path):
    """
    Creates two subplots:
      - Left subplot: NRMSE (log scale)
      - Right subplot: NIS   (log scale)
    Each subplot:
      - Violinplot per kernel
      - Individual scatter points
      - Lines indicating Q1, Median, Q3
      - A point (dot) indicating mean
    """
    # Extract kernel labels
    kernels = df_results["Kernel"].unique()

    # For each kernel, gather arrays of NRMSE and NIS over seeds
    nrmse_data = [df_results[df_results["Kernel"] == k]["NRMSE"].values for k in kernels]
    nis_data   = [df_results[df_results["Kernel"] == k]["NIS"].values   for k in kernels]

    # Create figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Violin Plots of NRMSE and NIS")

    # -------------------------------------------------------
    # 1) Left Subplot: NRMSE
    # -------------------------------------------------------
    positions = np.arange(len(kernels))
    parts1 = ax1.violinplot(nrmse_data, positions=positions, widths=0.5, showextrema=False)

    # Color the violins (optional)
    for pc in parts1['bodies']:
        pc.set_facecolor("red")
        pc.set_edgecolor("black")
        pc.set_alpha(0.5)

    # Overlay scatter plots for individual points
    for i, vals in enumerate(nrmse_data):
        ax1.scatter(
            np.full_like(vals, i, dtype=float),  # x-values (all = i)
            vals,                                # y-values
            color="red", alpha=0.6, s=10
        )

    # Overlay lines for Q1, Median, Q3, plus a dot for Mean
    for i, vals in enumerate(nrmse_data):
        q1     = np.percentile(vals, 25)
        median = np.median(vals)
        q3     = np.percentile(vals, 75)
        mean_  = np.mean(vals)

        # Q1 line
        ax1.plot([i - 0.15, i + 0.15], [q1, q1], color="black", lw=2)
        # Median line
        ax1.plot([i - 0.2, i + 0.2], [median, median], color="darkred", lw=2)
        # Q3 line
        ax1.plot([i - 0.15, i + 0.15], [q3, q3], color="black", lw=2)
        # Dot for mean
        ax1.scatter(i, mean_, color="blue", marker="o", s=40, zorder=3)

    # Set log scale if desired
    ax1.set_yscale("log")

    # Labeling, ticks, etc.
    ax1.set_xticks(range(len(kernels)))
    ax1.set_xticklabels(kernels, rotation=45, ha='right')
    ax1.set_ylabel("NRMSE (log scale)", color="red")
    ax1.set_title("NRMSE per Kernel")

    # -------------------------------------------------------
    # 2) Right Subplot: NIS
    # -------------------------------------------------------
    parts2 = ax2.violinplot(nis_data, positions=positions, widths=0.5, showextrema=False)

    # Color the violins
    for pc in parts2['bodies']:
        pc.set_facecolor("blue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.5)

    # Scatter points
    for i, vals in enumerate(nis_data):
        ax2.scatter(
            np.full_like(vals, i, dtype=float),
            vals,
            color="blue", alpha=0.6, s=10
        )

    # Q1, Median, Q3, Mean
    for i, vals in enumerate(nis_data):
        q1     = np.percentile(vals, 25)
        median = np.median(vals)
        q3     = np.percentile(vals, 75)
        mean_  = np.mean(vals)

        ax2.plot([i - 0.15, i + 0.15], [q1, q1], color="black", lw=2)
        ax2.plot([i - 0.2, i + 0.2], [median, median], color="darkblue", lw=2)
        ax2.plot([i - 0.15, i + 0.15], [q3, q3], color="black", lw=2)
        ax2.scatter(i, mean_, color="red", marker="o", s=40, zorder=3)

    ax2.set_yscale("log")
    ax2.set_xticks(range(len(kernels)))
    ax2.set_xticklabels(kernels, rotation=45, ha='right')
    ax2.set_ylabel("NIS (log scale)", color="blue")
    ax2.set_title("NIS per Kernel")

    # -------------------------------------------------------
    # Final layout, save, and close
    # -------------------------------------------------------
    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {save_path}")

def plot_predictions_vs_true(
    test_y, pred_mean, pred_var,
    source_indices, source_names,
    title="Predictions vs True Values", 
    save_dir="D:/gpplus_examples/examples/wing_MV_MF/main_wing_MV_MF",
    dpi=300,
    close_fig=True,
    ):
    """
    Plots true values (test_y) vs predictions (pred_mean) with prediction intervals.
    
    Args:
        test_y (array-like): True target values.
        pred_mean (array-like): Predicted mean values.
        pred_var (array-like): Predicted variances.
        source_indices (array-like): Array of source indices for each data point.
        source_names (list): List of source names corresponding to indices.
        title_base (str): Base string for plot titles.
        save_dir (str): Directory to save plots.
    """
    # Convert to numpy arrays if not already
    test_y = np.asarray(test_y)
    pred_mean = np.asarray(pred_mean)
    pred_var = np.asarray(pred_var)
    source_indices = np.asarray(source_indices)

    # Calculate standard deviation and prediction intervals
    pred_std = np.sqrt(pred_var)
    lower_bound = pred_mean - 1.96 * pred_std  # ~95% CI
    upper_bound = pred_mean + 1.96 * pred_std
    
    # print(source_names)
    # Create plots for each source
    for source_idx, source_name in enumerate(source_names):
        # Filter data for this source
        mask = source_indices == source_idx
        if not np.any(mask):
            continue  # Skip if no data for this source
            
        test_y_source = test_y[mask]
        pred_mean_source = pred_mean[mask]
        lower_bound_source = lower_bound[mask]
        upper_bound_source = upper_bound[mask]
        
        # Sort by PREDICTED values for smooth prediction bands
        sorted_indices = np.argsort(test_y_source)
        pred_mean_sorted = pred_mean_source[sorted_indices]
        test_y_sorted = test_y_source[sorted_indices]
        lower_bound_sorted = lower_bound_source[sorted_indices]
        upper_bound_sorted = upper_bound_source[sorted_indices]
        
        # Calculate RRMSE compared to source 's0'
        if source_name != 's0':
            mask_s0 = source_indices == 0
            test_y_s0 = test_y[mask_s0]
            rrmse = np.sqrt(np.mean((test_y_sorted - test_y_s0) ** 2)) / np.mean(test_y_s0)
        else:
            rrmse = 0

        # Create the plot
        plt.figure(figsize=(8, 8))

        # Plot the perfect prediction line (y=x)
        min_val = min(np.min(pred_mean_sorted), np.min(test_y_sorted))
        max_val = max(np.max(pred_mean_sorted), np.max(test_y_sorted))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

        # prediction interval (shaded)
        plt.fill_between(
            test_y_sorted,
            lower_bound_sorted,
            upper_bound_sorted,
            color='lightgreen',
            alpha=0.5,
            label='95% Prediction Interval'
        )
        
        # True vs Pred
        plt.scatter(
            test_y_sorted,
            pred_mean_sorted,
            color='blue',
            marker='.',
            alpha=0.7,
            label='Test Data'
        )

        # Compute MSE for this source
        mse = np.mean((pred_mean_sorted - test_y_sorted) ** 2)
        nrmse = np.sqrt(mse) / np.std(test_y_sorted)
        # Add MSE as text on the plot (top left corner)
        plt.text(
            0.05, 0.95,
            f"MSE: {mse:.4f}\nNRMSE: {nrmse:.4f}\nRRMSE (s0 vs {source_name}): {rrmse:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        # Customize plot
        plt.xlabel('Target Value')
        plt.ylabel('Prediction')
        plt.title(f"{title} - {source_name} Source")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axes limits to exactly match the data points (not prediction intervals)
        x_min, x_max = test_y_sorted.min(), test_y_sorted.max()
        y_min, y_max = pred_mean_sorted.min(), pred_mean_sorted.max()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Save plot
        save_path = os.path.join(save_dir, f"gp_predictions_{source_name.lower()}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        if close_fig:
            plt.close()

def plot_latent_space(latent_reps, labels, title, save_path, cat_dims=None, qual_dict=None, cat_combos=None, encoder=None, n_samples=100):
    """
    Plot 2D projection of latent space, supporting probabilistic encoders.
    If encoder is probabilistic, samples n_samples latent representations per label and plots the distribution.
    Otherwise, plots deterministic latent points as before.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    is_probabilistic = getattr(encoder, "is_probabilistic", False) if encoder is not None else False
    if is_probabilistic and hasattr(encoder, "forward"):
        unique_labels = sorted(set(labels))
        latent_samples = []
        sample_labels = []
        device = next(encoder.parameters()).device if hasattr(encoder, 'parameters') else 'cpu'
        z_dim = getattr(encoder, 'z_dim', 2)
        for label_idx, label in enumerate(unique_labels):
            one_hot = torch.zeros((1, len(unique_labels)), device=device)
            one_hot[0, label_idx] = 1
            one_hot = one_hot.repeat(n_samples, 1)
            with torch.no_grad():
                epsilon = torch.normal(mean=0, std=1, size=[n_samples, z_dim], device=device)
                z_samples = encoder(one_hot, epsilon=epsilon, visualize=True)
            latent_samples.append(z_samples.cpu().numpy())
            sample_labels.extend([label] * n_samples)
        latent_samples = np.vstack(latent_samples)
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
            idxs = [j for j, lbl in enumerate(sample_labels) if lbl == label]
            plt.scatter(latent_samples[idxs, 0], latent_samples[idxs, 1], label=str(label), alpha=0.5, s=20)
        plt.title(title + " (Probabilistic)")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Probabilistic latent space plot saved to: {save_path}")
        return
    # Deterministic case (original logic)
    # Reduce to 2D if needed
    if latent_reps.shape[1] > 2:
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2).fit_transform(latent_reps.cpu().numpy())
    else:
        proj = latent_reps.cpu().numpy()
    plt.figure(figsize=(15, 15))
    if isinstance(labels[0], str):
        # Source latent space plot
        plt.scatter(proj[:, 0], proj[:, 1], c=range(len(labels)), cmap='viridis', s=100)
        for i, txt in enumerate(labels):
            plt.annotate(txt, (proj[i, 0], proj[i, 1]), fontsize=24, xytext=(5, 5), textcoords='offset points')
        plt.title(f"{title}\nSource Fidelity Levels")
    else:
        # Categorical latent space plot
        if cat_dims is None or qual_dict is None or cat_combos is None:
            raise ValueError("cat_dims, qual_dict, and cat_combos must be provided for categorical plots")
        
        # Get all labels first
        all_labels = []
        for i in range(len(proj)):
            var_values = []
            current_pos = 0
            
            # For each categorical variable
            for var_idx, (col_idx, num_levels) in enumerate(sorted(qual_dict.items())):
                # Get the one-hot encoded values for this variable
                var_one_hot = cat_combos[i, current_pos:current_pos + num_levels]
                # Convert one-hot to actual level (0-based)
                level = torch.argmax(var_one_hot).item()
                var_values.append(f'Var{col_idx}={level}')
                current_pos += num_levels
            all_labels.append(', '.join(var_values))
        
        # Plot points with numbers
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=200, alpha=0.7)
        
        # Add numbered labels to points
        for i, (x, y) in enumerate(proj):
            plt.annotate(str(i), (x, y), fontsize=24, fontweight='bold', ha='center', va='center', color='black')
        legend_elements = []
        for i, label in enumerate(all_labels):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / len(all_labels)), markersize=24, label=f'{i}: {label}'))
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title='Categorical Combinations', fontsize=24, title_fontsize=24)
        plt.title(f"{title}")
    plt.xlabel("Latent Dimension 1", fontsize=24)
    plt.ylabel("Latent Dimension 2", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_distribution(data, save_dir):
    """
    Plot the distribution of data to verify data generation.
    Works for any example with arbitrary continuous/categorical/source columns.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    x_train = data['x_train_full'].cpu().numpy()
    y_train = data['y_train_full'].cpu().numpy()
    source_train = data['source_train_full'].cpu().numpy()
    cols = data['column_indices']
    source_cols = cols['source']
    continuous_cols = cols['continuous']
    categorical_cols = cols['categorical']
    source_names = data['metadata']['source_names']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Data Distribution Diagnostics", fontsize=16, fontweight='bold')

    # 1. Distribution of each continuous variable
    ax = axes[0, 0]
    if len(continuous_cols) > 0:
        for i, col in enumerate(continuous_cols):
            ax.hist(x_train[:, col], bins=30, alpha=0.6, label=f'Cont {i}')
        ax.set_title('Distribution of Continuous Variables')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No continuous variables', ha='center', va='center')
        ax.set_title('Distribution of Continuous Variables')

    # 2. Distribution of each categorical variable (if any)
    ax = axes[0, 1]
    if len(categorical_cols) > 0:
        for i, col in enumerate(categorical_cols):
            unique, counts = np.unique(x_train[:, col], return_counts=True)
            ax.bar(unique + i*0.1, counts, width=0.1, alpha=0.7, label=f'Cat {i}')
        ax.set_title('Distribution of Categorical Variables')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No categorical variables', ha='center', va='center')
        ax.set_title('Distribution of Categorical Variables')

    # 3. Output distribution by source
    ax = axes[1, 0]
    for source_idx, source_name in enumerate(source_names):
        mask = source_train == source_idx
        ax.hist(y_train[mask], bins=30, alpha=0.6, label=f'Source {source_name}')
    ax.set_title('Output Distribution by Source')
    ax.set_xlabel('Output Value')
    ax.set_ylabel('Count')
    ax.legend()

    # 4. Scatter plot of first continuous variable vs output by source
    ax = axes[1, 1]
    if len(continuous_cols) > 0:
        for source_idx, source_name in enumerate(source_names):
            mask = source_train == source_idx
            ax.scatter(x_train[mask, continuous_cols[0]], y_train[mask], alpha=0.6, label=f'Source {source_name}')
        ax.set_xlabel(f'Continuous Variable 0')
        ax.set_ylabel('Output')
        ax.set_title('First Continuous Variable vs Output by Source')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No continuous variables', ha='center', va='center')
        ax.set_title('First Continuous Variable vs Output by Source')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'data_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Data distribution plot saved to: {save_path}")

def plot_classification_results(y_true, y_pred_probs, y_pred_binary, title="Classification Results", save_dir=None):
    """
    Plot comprehensive classification results including ROC curve, confusion matrix, and probability distributions.
    
    Args:
        y_true (np.ndarray): True binary labels
        y_pred_probs (np.ndarray): Predicted probabilities for class 1
        y_pred_binary (np.ndarray): Binary predictions (thresholded at 0.5)
        title (str): Plot title
        save_dir (str): Directory to save plots
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall, precision)
    
    axes[0, 1].plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_title('Confusion Matrix')
    
    # Add text annotations for confusion matrix
    tn, fp, fn, tp = cm.ravel()
    axes[1, 0].text(0.5, -0.3, f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}', 
                    ha='center', va='center', transform=axes[1, 0].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # 4. Probability Distribution
    axes[1, 1].hist(y_pred_probs[y_true == 0], bins=30, alpha=0.7, label='Class 0', color='red', density=True)
    axes[1, 1].hist(y_pred_probs[y_true == 1], bins=30, alpha=0.7, label='Class 1', color='blue', density=True)
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Decision threshold')
    axes[1, 1].set_xlabel('Predicted Probability (Class 1)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Probability Distribution by True Class')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'classification_results.png'), dpi=300, bbox_inches='tight')
        print(f"Classification results plot saved to: {os.path.join(save_dir, 'classification_results.png')}")
    
    plt.show()
    
    # Create additional detailed plots
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle(f'{title} - Additional Analysis', fontsize=16, fontweight='bold')
    
    # 5. Calibration Plot
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_probs, n_bins=10)
    
    axes2[0].plot(mean_predicted_value, fraction_of_positives, "s-", label="GP Classifier")
    axes2[0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    axes2[0].set_xlabel('Mean Predicted Probability')
    axes2[0].set_ylabel('Fraction of Positives')
    axes2[0].set_title('Calibration Plot')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # 6. Decision Boundary Analysis
    # Create bins for probability ranges
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (y_pred_probs >= bins[i]) & (y_pred_probs < bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(y_true[mask] == y_pred_binary[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    axes2[1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='purple')
    axes2[1].set_xlabel('Predicted Probability Range')
    axes2[1].set_ylabel('Accuracy in Bin')
    axes2[1].set_title('Accuracy by Probability Range')
    axes2[1].grid(True, alpha=0.3)
    
    # Add count annotations
    for i, (center, acc, count) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
        if count > 0:
            axes2[1].text(center, acc + 0.02, f'n={count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'classification_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Classification analysis plot saved to: {os.path.join(save_dir, 'classification_analysis.png')}")
    
    plt.show()

def plot_softmax_probabilities(pred_probs_both, y_true, title="Softmax Class Probabilities", save_dir=None):
    """
    Plot the softmax class probabilities showing the distribution of predicted probabilities for each class.
    
    Args:
        pred_probs_both (torch.Tensor): Predicted probabilities for both classes [p(class=0), p(class=1)]
        y_true (np.ndarray): True binary labels
        title (str): Plot title
        save_dir (str): Directory to save plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Convert to numpy if needed
    if isinstance(pred_probs_both, torch.Tensor):
        pred_probs_both = pred_probs_both.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extract probabilities for each class
    probs_class0 = pred_probs_both[:, 0]  # p(class=0)
    probs_class1 = pred_probs_both[:, 1]  # p(class=1)
    
    # 1. Probability Distribution by True Class
    axes[0, 0].hist(probs_class1[y_true == 0], bins=30, alpha=0.7, label='True Class 0', 
                    color='red', density=True, edgecolor='black')
    axes[0, 0].hist(probs_class1[y_true == 1], bins=30, alpha=0.7, label='True Class 1', 
                    color='blue', density=True, edgecolor='black')
    axes[0, 0].axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Decision threshold')
    axes[0, 0].set_xlabel('Predicted Probability (Class 1)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Probability Distribution by True Class')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Probability Sum Verification
    prob_sums = probs_class0 + probs_class1
    axes[0, 1].hist(prob_sums, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.8, label='Expected sum = 1.0')
    axes[0, 1].set_xlabel('Sum of Class Probabilities')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Verification: Probabilities Sum to 1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter Plot: Class 0 vs Class 1 Probabilities
    axes[1, 0].scatter(probs_class0[y_true == 0], probs_class1[y_true == 0], 
                       alpha=0.6, label='True Class 0', color='red', s=20)
    axes[1, 0].scatter(probs_class0[y_true == 1], probs_class1[y_true == 1], 
                       alpha=0.6, label='True Class 1', color='blue', s=20)
    axes[1, 0].plot([0, 1], [1, 0], 'k--', alpha=0.5, label='Perfect separation line')
    axes[1, 0].set_xlabel('P(Class 0)')
    axes[1, 0].set_ylabel('P(Class 1)')
    axes[1, 0].set_title('Class 0 vs Class 1 Probabilities')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. prediction Distribution
    # Calculate prediction as max probability
    prediction = np.maximum(probs_class0, probs_class1)
    correct_predictions = (y_true == (probs_class1 > 0.5).astype(int))
    
    axes[1, 1].hist(prediction[correct_predictions], bins=20, alpha=0.7, 
                    label='Correct Predictions', color='green', density=True)
    axes[1, 1].hist(prediction[~correct_predictions], bins=20, alpha=0.7, 
                    label='Incorrect Predictions', color='red', density=True)
    axes[1, 1].set_xlabel('Prediction prediction (max probability)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('prediction Distribution by Prediction Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'softmax_probabilities.png'), dpi=300, bbox_inches='tight')
        print(f"Softmax probabilities plot saved to: {os.path.join(save_dir, 'softmax_probabilities.png')}")
    
    plt.show()
    
    # Create additional detailed analysis
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f'{title} - Detailed Analysis', fontsize=16, fontweight='bold')
    
    # 5. Probability Statistics by True Class
    class0_stats = probs_class1[y_true == 0]
    class1_stats = probs_class1[y_true == 1]
    
    stats_data = {
        'True Class 0': [class0_stats.mean(), class0_stats.std(), class0_stats.min(), class0_stats.max()],
        'True Class 1': [class1_stats.mean(), class1_stats.std(), class1_stats.min(), class1_stats.max()]
    }
    
    x = np.arange(4)
    width = 0.35
    
    axes2[0].bar(x - width/2, stats_data['True Class 0'], width, label='True Class 0', alpha=0.7)
    axes2[0].bar(x + width/2, stats_data['True Class 1'], width, label='True Class 1', alpha=0.7)
    axes2[0].set_xlabel('Statistics')
    axes2[0].set_ylabel('Probability Value')
    axes2[0].set_title('Probability Statistics by True Class')
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(['Mean', 'Std', 'Min', 'Max'])
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # 6. Probability Range Analysis
    prob_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    range_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    class0_counts = []
    class1_counts = []
    
    for low, high in prob_ranges:
        mask = (probs_class1 >= low) & (probs_class1 < high)
        class0_counts.append(np.sum((y_true == 0) & mask))
        class1_counts.append(np.sum((y_true == 1) & mask))
    
    x = np.arange(len(prob_ranges))
    axes2[1].bar(x - width/2, class0_counts, width, label='True Class 0', alpha=0.7)
    axes2[1].bar(x + width/2, class1_counts, width, label='True Class 1', alpha=0.7)
    axes2[1].set_xlabel('Probability Range (Class 1)')
    axes2[1].set_ylabel('Count')
    axes2[1].set_title('Sample Distribution by Probability Range')
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(range_labels, rotation=45)
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    # 7. Calibration Plot for Softmax Probabilities
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs_class1, n_bins=10)
    
    axes2[2].plot(mean_predicted_value, fraction_of_positives, "s-", label="Softmax Probabilities")
    axes2[2].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    axes2[2].set_xlabel('Mean Predicted Probability')
    axes2[2].set_ylabel('Fraction of Positives')
    axes2[2].set_title('Calibration Plot (Softmax)')
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'softmax_detailed_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Softmax detailed analysis plot saved to: {os.path.join(save_dir, 'softmax_detailed_analysis.png')}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nSoftmax Probability Summary:")
    print(f"Class 0 samples (n={np.sum(y_true == 0)}):")
    print(f"  Mean P(Class 1): {class0_stats.mean():.4f}")
    print(f"  Std P(Class 1): {class0_stats.std():.4f}")
    print(f"  Range: [{class0_stats.min():.4f}, {class0_stats.max():.4f}]")
    
    print(f"\nClass 1 samples (n={np.sum(y_true == 1)}):")
    print(f"  Mean P(Class 1): {class1_stats.mean():.4f}")
    print(f"  Std P(Class 1): {class1_stats.std():.4f}")
    print(f"  Range: [{class1_stats.min():.4f}, {class1_stats.max():.4f}]")
    
    print(f"\nProbability Sum Verification:")
    print(f"  Mean sum: {prob_sums.mean():.6f}")
    print(f"  Std sum: {prob_sums.std():.6f}")
    print(f"  All sum to 1: {np.allclose(prob_sums, 1.0, atol=1e-6)}")

def plot_calibration_analysis(
    y_test_full, pred_stddev, source_test_full, source_names,
    calibration_values, calibration_values_original, results_dict, seed,
    x_test_std, calib_cont_col_indices, dir_path,
    model, standardize_data
):
    """
    For each source, plots a scatter of true vs calibrated-predicted values,
    where the source indicator is set to the current source and calibration parameters are applied (using calib_cont_col_indices for the relevant columns),
    using only the high-fidelity (first source) test data as the base input.
    Adds a final subplot for error metrics (NRMSE, MAE, etc.) for all sources and a subplot for calibration parameter distributions.
    Layout adapts to the number of sources. Only shows calibrated predictions.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import math
    from scipy.stats import norm

    

    # Prepare calibration parameter values (skip std keys)
    cal_param_items = [
        (idx, calibration_values[param_name])
        for idx, param_name in zip(calib_cont_col_indices, calibration_values.keys())
        if 'std' not in param_name
    ]
    n_sources = len(source_names)
    n_plots = n_sources + 1 + len(calibration_values) # +1 for error metrics, +1 for calibration parameter distribution
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Use only high-fidelity (first source) test data as base
    high_fidelity_index = 0  # assumes s0 is first, or could search for 's0' in source_names
    if 's0' in source_names:
        high_fidelity_index = source_names.index('s0')
    mask_hf = source_test_full == high_fidelity_index
    x_test_hf = x_test_std[mask_hf].clone()
    y_true_hf = y_test_full[mask_hf].cpu().numpy()

    # For each source, plot true vs calibrated-predicted (using only high-fidelity test data)
    for i, source in enumerate(source_names):
        x_test_mod = x_test_hf.clone()
        # Set all test points' source indicator to current source (one-hot)
        x_test_mod[:, :n_sources] = 0
        x_test_mod[:, i] = 1
        # Apply all calibration parameters
        for idx, cal_param in cal_param_items:
            x_test_mod[:, idx] = cal_param
        # Predict
        with torch.no_grad():
            pred = model(x_test_mod)
            pred_mean_std = pred.mean.squeeze()
            pred_stddev_std = pred.stddev.squeeze() if hasattr(pred, 'stddev') else pred.variance.sqrt().squeeze()
        # Inverse transform
        pred_mean, _, _, _ = standardize_data.inverse_predictions(
            pred_mean_std, pred_mean_std, pred_mean_std, pred_stddev_std
        )
        y_pred = pred_mean.cpu().numpy()
        ax = axes[i]
        ax.scatter(y_true_hf, y_pred, alpha=0.7, color='C0', s=20)
        min_val = min(y_true_hf.min(), y_pred.min())
        max_val = max(y_true_hf.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        ax.set_xlabel('True Values (High-Fidelity)')
        ax.set_ylabel('Calibrated Prediction')
        ax.set_title(f'Calibrated: {source}')
        ax.grid(True, alpha=0.3)

    # Error metrics subplot
    ax = axes[n_sources]
    nrmse_values = [results_dict[seed]['NRMSE'][source] for source in source_names]
    mae_values = [results_dict[seed]['MAE'][source] for source in source_names]
    x_pos = np.arange(n_sources)
    width = 0.35
    ax.bar(x_pos - width/2, nrmse_values, width, label='NRMSE', alpha=0.7)
    ax.bar(x_pos + width/2, mae_values, width, label='MAE', alpha=0.7)
    ax.set_xlabel('Source')
    ax.set_ylabel('Error Metric')
    ax.set_title('Performance by Source')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(source_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calibration parameter distribution subplot
    ax = axes[n_sources + 1]
    
    # Extract mean and std for each calibration parameter (original space)
    orig_means = []
    orig_stds = []
    param_names = []
    for param_name, orig_val in calibration_values_original.items():
        if 'std' not in param_name:
            # Find corresponding std parameter in original space
            cal_id = param_name.split('_')[1]
            std_param_name = f'Theta_{cal_id}_std'
            orig_std = calibration_values_original.get(std_param_name, 0.1)  # Default std if not found
            orig_means.append(orig_val)
            orig_stds.append(orig_std)
            param_names.append(param_name)
    # Plot distribution for each calibration parameter (original space)
    colors = plt.cm.Set1(np.linspace(0, 1, len(orig_means)))
    for i, (mean_val, std_val, param_name, color) in enumerate(zip(orig_means, orig_stds, param_names, colors)):
        # Create x values for the distribution
        x_range = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 100)
        from scipy.stats import norm
        y_values = norm.pdf(x_range, mean_val, std_val)
        # Plot the distribution curve
        ax.plot(x_range, y_values, color=color, linewidth=2, label=f'{param_name}')
        # Fill under the curve
        ax.fill_between(x_range, y_values, alpha=0.3, color=color)
        # Mark the mean with a vertical line
        ax.axvline(mean_val, color=color, linestyle='--', alpha=0.8)
        # Add text annotation for mean and std
        ax.text(mean_val, max(y_values)*0.8, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    ax.set_xlabel('Calibration Parameter Value (Original Space)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Calibration Parameter Distributions (Original Space)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hide any unused axes
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    calibration_plot_path = os.path.join(dir_path, f"calibration_analysis_{seed}.png")
    plt.savefig(calibration_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration analysis plots saved to: {calibration_plot_path}")


def _extract_parameter_paths(params_dict: Dict, prefix: str = "", paths: Optional[List[Tuple[str, Any]]] = None) -> List[Tuple[str, Any]]:
    """
    Recursively extract all parameter paths and values from a nested parameter dictionary.
    Creates separate paths for each element in lists (e.g., lengthscales, neural network layers).
    
    Args:
        params_dict: Nested dictionary of parameters
        prefix: Current path prefix (for building full paths)
        paths: List to accumulate paths (internal use)
    
    Returns:
        List of tuples: (path_string, value)
    """
    if paths is None:
        paths = []
    
    for key, value in params_dict.items():
        current_path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively extract from nested dictionaries
            _extract_parameter_paths(value, current_path, paths)
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples
            if len(value) > 0:
                if isinstance(value[0], (int, float, np.number)):
                    # List of numbers (e.g., lengthscales) - create separate path for each element
                    for i, item in enumerate(value):
                        paths.append((f"{current_path}[{i}]", item))
                elif isinstance(value[0], dict):
                    # List of dicts (e.g., base_kernels, weight_kernels) - recurse
                    for i, item in enumerate(value):
                        _extract_parameter_paths(item, f"{current_path}[{i}]", paths)
                else:
                    # List of other types - create separate path for each element
                    for i, item in enumerate(value):
                        paths.append((f"{current_path}[{i}]", item))
        elif value is not None and isinstance(value, (int, float, np.number, bool)):
            # Leaf node - scalar value
            paths.append((current_path, value))
    
    return paths


def _get_best_run_index(data: Dict) -> Optional[str]:
    """
    Identify the best run based on minimum loss.
    
    Args:
        data: Dictionary with structure {"runs": {"0": {...}, "1": {...}, ...}}
              or {"initial_parameters": {...}, "records": [...]} for single run
    
    Returns:
        String run index of the best run, or None if single run
    """
    if "runs" not in data:
        # Single run - no best run to identify
        return None
    
    best_run_idx = None
    best_loss = float('inf')
    
    for run_idx, run_data in data["runs"].items():
        records = run_data.get("records", [])
        if not records:
            continue
        
        # Find minimum loss in this run
        losses = [r.get("loss") for r in records if r.get("loss") is not None]
        if losses:
            min_loss = min(losses)
            if min_loss < best_loss:
                best_loss = min_loss
                best_run_idx = run_idx
    
    return best_run_idx


def _get_green_gradient_colors(n_colors: int) -> List[str]:
    """
    Generate a list of green gradient colors from light to dark green.
    
    Args:
        n_colors: Number of colors to generate
    
    Returns:
        List of hex color strings in green gradient
    """
    if n_colors == 0:
        return []
    elif n_colors == 1:
        return ['#228B22']  # Forest green
    
    # Generate colors from light green to dark green
    # Using HSL: hue around 120 (green), saturation 0.5-1.0, lightness 0.3-0.6
    colors = []
    for i in range(n_colors):
        # Interpolate from light (i=0) to dark (i=n_colors-1)
        # Lightness: 0.6 -> 0.3 (darker as we go)
        # Saturation: 0.5 -> 1.0 (more saturated as we go)
        lightness = 0.6 - (i / (n_colors - 1)) * 0.3
        saturation = 0.5 + (i / (n_colors - 1)) * 0.5
        hue = 120 / 360  # Green hue
        
        # Convert HSL to RGB
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
        colors.append(hex_color)
    
    return colors


def _make_compact_title(param_path: str) -> str:
    """
    Create a compact title from a parameter path.
    
    Extracts:
    - "base_kernels[x]" where x is the index
    - "raw_parameter[y]" or "parameter[y]" where y is the index if it has multiple values
    
    Examples:
    - "covar_module.base_kernels[0].base_kernel.lengthscale[0]" -> "base_kernels[0].lengthscale[0]"
    - "covar_module.base_kernels[1].base_kernel.raw_lengthscale[0]" -> "base_kernels[1].raw_lengthscale[0]"
    - "covar_module.base_kernels[0].raw_outputscale" -> "base_kernels[0].raw_outputscale"
    - "covar_module.likelihood.raw_noise" -> "raw_noise"
    
    Args:
        param_path: Full parameter path (e.g., "covar_module.base_kernels[0].base_kernel.lengthscale[0]")
    
    Returns:
        Compact title (e.g., "base_kernels[0].lengthscale[0]")
    """
    import re
    
    # Remove "covar_module." prefix if present
    path = param_path
    if path.startswith("covar_module."):
        path = path[len("covar_module."):]
    
    # Check if this is a base_kernels parameter
    base_kernel_match = re.search(r'base_kernels\[(\d+)\]', path)
    if base_kernel_match:
        base_kernel_idx = base_kernel_match.group(1)
        # Extract the parameter name and index after base_kernels[x]
        # Pattern: base_kernels[x].base_kernel.parameter_name[index] or base_kernels[x].parameter_name[index]
        # Remove "base_kernels[x].base_kernel." or "base_kernels[x]."
        remaining = path.split(f"base_kernels[{base_kernel_idx}].")[-1]
        # Remove "base_kernel." if present
        if remaining.startswith("base_kernel."):
            remaining = remaining[len("base_kernel."):]
        # Return compact format: base_kernels[x].parameter_name[index]
        return f"base_kernels[{base_kernel_idx}].{remaining}"
    else:
        # Not a base_kernels parameter, just return the parameter name with index if present
        # Remove "likelihood." or "mean_module." prefixes
        if "." in path:
            parts = path.split(".")
            # Take the last part (parameter name with index)
            return parts[-1]
        return path


def plot_parameters_by_iteration(
    json_file: str,
    save_dir: Optional[str] = None,
    parameter_paths: Optional[List[str]] = None,
    max_parameters: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 300,
    alpha: float = 0.3,
    linewidth: float = 1.5,
    verbose: bool = True,
    rrmse_values: Optional[Dict] = None
):
    """
    Plot parameter evolution over iterations (for LBFGS optimizer).
    Highlights the best run of each fold in green.
    
    Args:
        json_file: Path to JSON file saved by IterationParameterCallback
        save_dir: Directory to save plots. If None, uses directory of json_file
        parameter_paths: List of specific parameter paths to plot (e.g., ["covar_module.base_kernel.lengthscale"]).
                        If None, plots all parameters (subject to max_parameters limit)
        max_parameters: Maximum number of parameters to plot if parameter_paths is None.
                       If None, plots all parameters without limit.
        figsize: Figure size (width, height)
        dpi: Resolution for saved plots
        alpha: Transparency for non-best runs
        linewidth: Line width for plots
        verbose: Whether to print progress messages
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle new fold-based format, old multi-run format, and single-run format
    if "folds" in data:
        # New format: organized by folds
        folds_data = data["folds"]
        is_fold_format = True
        # Flatten to get all runs across all folds for reference
        all_runs_data = {}
        for fold_key, fold_data in folds_data.items():
            runs = fold_data.get("runs", {})
            for run_key, run_data in runs.items():
                # Use fold_run as key to avoid collisions
                all_runs_data[f"{fold_key}_{run_key}"] = run_data
        runs_data = all_runs_data
    elif "runs" in data:
        # Old multi-run format (backward compatibility)
        runs_data = data["runs"]
        is_fold_format = False
        folds_data = None
    else:
        # Single run format
        runs_data = {"0": data}
        is_fold_format = False
        folds_data = None
    
    # Find median RRMSE run per fold (if rrmse_values provided) or best loss run per fold
    highlighted_runs_per_fold = {}  # {fold_key: highlighted_run_key}
    highlight_color = 'red'  # Default to red for median RRMSE
    highlight_label_suffix = " (Median RRMSE)"
    best_run_idx = None  # For old format
    
    if rrmse_values is not None:
        # Find median RRMSE run for each fold
        if is_fold_format:
            for fold_key, fold_data in folds_data.items():
                runs = fold_data.get("runs", {})
                if runs:
                    # Collect RRMSE values for all runs in this fold
                    rrmse_list = []
                    for run_key, run_data in runs.items():
                        rrmse_key = (fold_key, run_key) if isinstance(list(rrmse_values.keys())[0], tuple) else run_key
                        if rrmse_key in rrmse_values:
                            rrmse_value = rrmse_values[rrmse_key]
                            if rrmse_value is not None and not np.isnan(rrmse_value) and not np.isinf(rrmse_value):
                                rrmse_list.append((run_key, rrmse_value))
                    
                    if rrmse_list:
                        # Sort by RRMSE and find median
                        rrmse_list.sort(key=lambda x: x[1])
                        median_idx = len(rrmse_list) // 2
                        median_run_key = rrmse_list[median_idx][0]
                        highlighted_runs_per_fold[fold_key] = median_run_key
                        if verbose:
                            print(f"Fold {fold_key}: Median RRMSE run is {median_run_key} (RRMSE={rrmse_list[median_idx][1]:.6f})")
        else:
            # Old format: find overall median RRMSE run
            rrmse_list = []
            for run_key in runs_data.keys():
                if run_key in rrmse_values:
                    rrmse_value = rrmse_values[run_key]
                    if rrmse_value is not None and not np.isnan(rrmse_value) and not np.isinf(rrmse_value):
                        rrmse_list.append((run_key, rrmse_value))
            
            if rrmse_list:
                rrmse_list.sort(key=lambda x: x[1])
                median_idx = len(rrmse_list) // 2
                best_run_idx = rrmse_list[median_idx][0]
                if verbose:
                    print(f"Median RRMSE run is {best_run_idx} (RRMSE={rrmse_list[median_idx][1]:.6f})")
    else:
        # Fallback to best loss run (green)
        highlight_color = 'green'
        highlight_label_suffix = " (Best)"
        if is_fold_format:
            for fold_key, fold_data in folds_data.items():
                runs = fold_data.get("runs", {})
                if runs:
                    # Find best run within this fold by loss
                    best_loss = float('inf')
                    best_run_key = None
                    for run_key, run_data in runs.items():
                        records = run_data.get("records", [])
                        for record in records:
                            loss = record.get("loss")
                            if loss is not None and loss < best_loss:
                                best_loss = loss
                                best_run_key = run_key
                    if best_run_key is not None:
                        highlighted_runs_per_fold[fold_key] = best_run_key
            if verbose:
                print(f"Found {len(highlighted_runs_per_fold)} folds with best loss runs identified")
        else:
            # Old format: find overall best run
            best_run_idx = _get_best_run_index(data) if len(runs_data) > 1 else None
            if verbose:
                if best_run_idx is not None:
                    print(f"Best run identified: Run {best_run_idx}")
                else:
                    print("Single run detected (no best run to highlight)")
    
    # Extract all parameter paths from first run (or best run if available)
    if is_fold_format:
        # Use first fold's first run as reference
        first_fold = list(folds_data.keys())[0]
        first_run = list(folds_data[first_fold].get("runs", {}).keys())[0]
        reference_records = folds_data[first_fold]["runs"][first_run].get("records", [])
    else:
        reference_run = best_run_idx if best_run_idx is not None else list(runs_data.keys())[0]
        reference_records = runs_data[reference_run].get("records", [])
    
    if not reference_records:
        print(f"Warning: No records found")
        return
    
    # Extract parameter paths from first record
    all_param_paths = _extract_parameter_paths(reference_records[0].get("parameters", {}))
    
    # Filter out all neural network parameters - only keep GP parameters (lengthscales, outputscale, noise, etc.)
    # Remove all neural network layers: trunk_layers, head_layers, and layers (from InputTransformNet)
    filtered_param_paths = []
    for path, value in all_param_paths:
        path_str = path
        # Skip all neural network layers: trunk_layers, head_layers, and layers
        if ".trunk_layers[" in path_str or ".head_layers[" in path_str or ".layers[" in path_str:
            continue
        # Keep only GP parameters (lengthscales, outputscale, noise, etc.)
        filtered_param_paths.append((path, value))
    all_param_paths = filtered_param_paths
    
    # Filter parameter paths if specified
    if parameter_paths is not None:
        # Filter to only requested paths
        filtered_paths = [p for p in all_param_paths if any(req_path in p[0] for req_path in parameter_paths)]
        if not filtered_paths:
            print(f"Warning: No parameters found matching paths: {parameter_paths}")
            return
        param_paths_to_plot = filtered_paths
    else:
        # Use all parameters, optionally limited by max_parameters
        if max_parameters is not None:
            param_paths_to_plot = all_param_paths[:max_parameters]
            if verbose and len(all_param_paths) > max_parameters:
                print(f"Note: Found {len(all_param_paths)} parameters, plotting first {max_parameters}. "
                      f"Set max_parameters=None to plot all.")
        else:
            param_paths_to_plot = all_param_paths
    
    if verbose:
        print(f"Plotting {len(param_paths_to_plot)} parameters")
    
    # Determine number of subplots
    n_params = len(param_paths_to_plot)
    # Adjust ncols based on number of parameters for better layout
    if n_params <= 4:
        ncols = n_params
    elif n_params <= 12:
        ncols = 4
    elif n_params <= 24:
        ncols = 6
    else:
        ncols = 8  # For many parameters, use more columns
    nrows = math.ceil(n_params / ncols)
    
    # Adjust figure size based on number of parameters
    # Base size per subplot: approximately 3x2.5 inches
    base_width_per_subplot = 3.0
    base_height_per_subplot = 2.5
    # For many parameters, reduce size per subplot slightly to fit on screen
    if n_params > 50:
        width_per_subplot = 2.5
        height_per_subplot = 2.0
    elif n_params > 20:
        width_per_subplot = 2.8
        height_per_subplot = 2.2
    else:
        width_per_subplot = base_width_per_subplot
        height_per_subplot = base_height_per_subplot
    
    adjusted_figsize = (ncols * width_per_subplot, nrows * height_per_subplot)
    # Cap maximum figure size to prevent issues
    max_width = 40
    max_height = 60
    adjusted_figsize = (min(adjusted_figsize[0], max_width), min(adjusted_figsize[1], max_height))
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=adjusted_figsize)
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes
    
    # Generate green gradient colors for best runs (if using green highlighting)
    highlighted_runs_list = []
    if highlight_color == 'green' and is_fold_format:
        # Collect all highlighted runs to assign gradient colors
        for fold_key in sorted(folds_data.keys()):
            if fold_key in highlighted_runs_per_fold:
                highlighted_runs_list.append((fold_key, highlighted_runs_per_fold[fold_key]))
        green_colors = _get_green_gradient_colors(len(highlighted_runs_list))
        # Create mapping from (fold_key, run_key) to color
        highlighted_run_colors = {}
        for idx, (fold_key, run_key) in enumerate(highlighted_runs_list):
            highlighted_run_colors[(fold_key, run_key)] = green_colors[idx]
    else:
        highlighted_run_colors = {}
    
    # Plot each parameter
    for param_idx, (param_path, _) in enumerate(param_paths_to_plot):
        ax = axes[param_idx]
        
        # Collect data for all runs (grouped by fold if fold format)
        if is_fold_format:
            # Iterate through folds and runs
            for fold_key, fold_data in folds_data.items():
                runs = fold_data.get("runs", {})
                highlighted_run_key = highlighted_runs_per_fold.get(fold_key)
                
                for run_key, run_data in runs.items():
                    records = run_data.get("records", [])
                    if not records:
                        continue
                    
                    # Extract iterations and parameter values
                    iterations = []
                    param_values = []
                    
                    for record in records:
                        iter_num = record.get("iteration")
                        if iter_num is None:
                            continue
                        
                        # Navigate to parameter value
                        param_value = _get_nested_value(record.get("parameters", {}), param_path)
                        if param_value is not None:
                            iterations.append(iter_num)
                            # Handle list values
                            if isinstance(param_value, list):
                                if len(param_value) > 0:
                                    # If it's a single-element list (scalar wrapped in list), unwrap it
                                    if len(param_value) == 1:
                                        param_values.append(float(param_value[0]))
                                    else:
                                        # For multi-element lists, check if it's a neural network weight/bias
                                        # Neural network weights/biases: use L2 norm to show magnitude changes
                                        # Other lists (like lengthscales): use mean
                                        if "weight" in param_path.lower() or "bias" in param_path.lower():
                                            # For neural network weights/biases, use L2 norm to better show changes
                                            param_array = np.array(param_value)
                                            l2_norm = float(np.linalg.norm(param_array))
                                            param_values.append(l2_norm)
                                        else:
                                            # For other lists (e.g., lengthscales), use mean
                                            param_values.append(float(np.mean(param_value)))
                                else:
                                    continue
                            else:
                                param_values.append(float(param_value))
                    
                    if not iterations:
                        continue
                    
                    # Determine color and style: highlighted run (median RRMSE or best loss)
                    is_highlighted = (highlighted_run_key is not None and run_key == highlighted_run_key)
                    if is_highlighted:
                        # Use gradient green for best runs, or red for median RRMSE
                        if highlight_color == 'green' and (fold_key, run_key) in highlighted_run_colors:
                            color = highlighted_run_colors[(fold_key, run_key)]
                            annotation_color = color  # Use the same gradient color for annotation
                        else:
                            color = highlight_color
                            annotation_color = highlight_color
                    else:
                        color = 'gray'
                        annotation_color = 'gray'
                    line_alpha = 1.0 if is_highlighted else alpha
                    line_width = linewidth * 1.5 if is_highlighted else linewidth
                    fold_num = fold_key.replace("fold_", "") if fold_key.startswith("fold_") else fold_key
                    run_num = run_key.replace("run_", "") if run_key.startswith("run_") else run_key
                    # Create label with fold and run indices
                    if is_highlighted:
                        label = f"F{fold_num}R{run_num}{highlight_label_suffix}"
                    else:
                        label = f"F{fold_num}R{run_num}"
                    
                    # Plot (only add label to first subplot to avoid duplicate legends)
                    line = None
                    if param_idx == 0:
                        line, = ax.plot(iterations, param_values, color=color, alpha=line_alpha, linewidth=line_width, label=label)
                    else:
                        line, = ax.plot(iterations, param_values, color=color, alpha=line_alpha, linewidth=line_width)
                    
                    # Add annotation for highlighted runs (only on first parameter subplot to avoid clutter)
                    if is_highlighted and param_idx == 0 and len(iterations) > 0 and len(param_values) > 0:
                        # Annotate at the last point
                        last_idx = len(iterations) - 1
                        ax.annotate(f"F{fold_num}R{run_num}", 
                                   xy=(iterations[last_idx], param_values[last_idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=7, color=annotation_color, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=annotation_color, alpha=0.7))
        else:
            # Old format: iterate through runs
            for run_idx, run_data in runs_data.items():
                records = run_data.get("records", [])
                if not records:
                    continue
                
                # Extract iterations and parameter values
                iterations = []
                param_values = []
                
                for record in records:
                    iter_num = record.get("iteration")
                    if iter_num is None:
                        continue
                    
                    # Navigate to parameter value
                    param_value = _get_nested_value(record.get("parameters", {}), param_path)
                    if param_value is not None:
                        iterations.append(iter_num)
                        # Handle list values
                        if isinstance(param_value, list):
                            if len(param_value) > 0:
                                # If it's a single-element list (scalar wrapped in list), unwrap it
                                if len(param_value) == 1:
                                    param_values.append(float(param_value[0]))
                                else:
                                    # For multi-element lists, check if it's a neural network weight/bias
                                    # Neural network weights/biases: use L2 norm to show magnitude changes
                                    # Other lists (like lengthscales): use mean
                                    if "weight" in param_path.lower() or "bias" in param_path.lower():
                                        # For neural network weights/biases, use L2 norm to better show changes
                                        param_array = np.array(param_value)
                                        l2_norm = float(np.linalg.norm(param_array))
                                        param_values.append(l2_norm)
                                    else:
                                        # For other lists (e.g., lengthscales), use mean
                                        param_values.append(float(np.mean(param_value)))
                            else:
                                continue
                        else:
                            param_values.append(float(param_value))
                
                if not iterations:
                    continue
                
                # Determine color and style
                is_best = (best_run_idx is not None and run_idx == best_run_idx)
                color = 'green' if is_best else 'gray'
                line_alpha = 1.0 if is_best else alpha
                line_width = linewidth * 1.5 if is_best else linewidth
                label = f"Run {run_idx}" + (" (Best)" if is_best else "")
                
                # Plot (only add label to first subplot to avoid duplicate legends)
                if param_idx == 0:
                    ax.plot(iterations, param_values, color=color, alpha=line_alpha, linewidth=line_width, label=label)
                else:
                    ax.plot(iterations, param_values, color=color, alpha=line_alpha, linewidth=line_width)
        
        # Customize subplot
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Parameter Value")
        # Create compact title
        title = _make_compact_title(param_path)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])
    
    # Create legend showing fold and run indices
    # For fold format, show only best runs in legend to avoid clutter
    if is_fold_format:
        # Collect only best run labels for legend
        best_labels = []
        best_handles = []
        for ax in axes[:n_params]:
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            if ax_handles and ax_labels:
                for handle, label in zip(ax_handles, ax_labels):
                    if "(Best)" in label and label not in best_labels:
                        best_labels.append(label)
                        best_handles.append(handle)
                if best_handles:
                    break
        
        if best_handles and best_labels:
            # Show only best runs in legend
            fig.legend(best_handles, best_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=min(len(best_handles), 10), fontsize=7, frameon=True, 
                      title='Best Runs (Green)', title_fontsize=8)
        else:
            # Fallback: show all runs if not too many
            total_runs = sum(len(fold_data.get("runs", {})) for fold_data in folds_data.values())
            if total_runs <= 20:
                handles, labels = None, None
                for ax in axes[:n_params]:
                    ax_handles, ax_labels = ax.get_legend_handles_labels()
                    if ax_handles:
                        handles, labels = ax_handles, ax_labels
                        break
                if handles and labels:
                    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                              ncol=min(len(handles), 8), fontsize=7, frameon=True)
    else:
        # Old format: show all runs if not too many
        total_runs = len(runs_data)
        if total_runs <= 20:
            handles, labels = None, None
            for ax in axes[:n_params]:
                ax_handles, ax_labels = ax.get_legend_handles_labels()
                if ax_handles:
                    handles, labels = ax_handles, ax_labels
                    break
            
            if handles and labels:
                # Create a single legend for the entire figure
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                          ncol=min(len(handles), 8), fontsize=7, frameon=True)
    
    # Overall title
    fig.suptitle("Parameter Evolution Over Iterations", fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Adjust bottom margin for legend
    
    # Save plot
    if save_dir is None:
        save_dir = os.path.dirname(json_file)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "parameters_by_iteration.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Parameter iteration plot saved to: {save_path}")


def plot_parameters_by_epoch(
    json_file: str,
    save_dir: Optional[str] = None,
    parameter_paths: Optional[List[str]] = None,
    max_parameters: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = 300,
    alpha: float = 0.3,
    linewidth: float = 1.5,
    verbose: bool = True,
    rrmse_values: Optional[Dict] = None
):
    """
    Plot parameter evolution over epochs (for Adam and other epoch-based optimizers).
    Highlights the run with median RRMSE of each fold in red (if rrmse_values provided),
    otherwise highlights the best loss run in green.
    
    Args:
        json_file: Path to JSON file saved by EpochParameterCallback
        save_dir: Directory to save plots. If None, uses directory of json_file
        parameter_paths: List of specific parameter paths to plot (e.g., ["covar_module.base_kernel.lengthscale"]).
                        If None, plots all parameters (subject to max_parameters limit)
        max_parameters: Maximum number of parameters to plot if parameter_paths is None.
                       If None, plots all parameters without limit.
        figsize: Figure size (width, height)
        dpi: Resolution for saved plots
        alpha: Transparency for non-highlighted runs
        linewidth: Line width for plots
        verbose: Whether to print progress messages
        rrmse_values: Optional dict mapping (fold_key, run_key) or run_key to RRMSE values.
                     If provided, highlights median RRMSE run in red instead of best loss run in green.
                     Format: {(fold_key, run_key): rrmse} for fold format, or {run_key: rrmse} for old format.
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle new fold-based format, old multi-run format, and single-run format
    if "folds" in data:
        # New format: organized by folds
        folds_data = data["folds"]
        is_fold_format = True
        # Flatten to get all runs across all folds for reference
        all_runs_data = {}
        for fold_key, fold_data in folds_data.items():
            runs = fold_data.get("runs", {})
            for run_key, run_data in runs.items():
                # Use fold_run as key to avoid collisions
                all_runs_data[f"{fold_key}_{run_key}"] = run_data
        runs_data = all_runs_data
    elif "runs" in data:
        # Old multi-run format (backward compatibility)
        runs_data = data["runs"]
        is_fold_format = False
        folds_data = None
    else:
        # Single run format
        runs_data = {"0": data}
        is_fold_format = False
        folds_data = None
    
    # Find median RRMSE run per fold (if rrmse_values provided) or best loss run per fold
    highlighted_runs_per_fold = {}  # {fold_key: highlighted_run_key}
    highlight_color = 'red'  # Default to red for median RRMSE
    highlight_label_suffix = " (Median RRMSE)"
    best_run_idx = None  # For old format
    
    if rrmse_values is not None:
        # Find median RRMSE run for each fold
        if is_fold_format:
            for fold_key, fold_data in folds_data.items():
                runs = fold_data.get("runs", {})
                if runs:
                    # Collect RRMSE values for all runs in this fold
                    rrmse_list = []
                    for run_key, run_data in runs.items():
                        rrmse_key = (fold_key, run_key) if isinstance(list(rrmse_values.keys())[0], tuple) else run_key
                        if rrmse_key in rrmse_values:
                            rrmse_value = rrmse_values[rrmse_key]
                            if rrmse_value is not None and not np.isnan(rrmse_value) and not np.isinf(rrmse_value):
                                rrmse_list.append((run_key, rrmse_value))
                    
                    if rrmse_list:
                        # Sort by RRMSE and find median
                        rrmse_list.sort(key=lambda x: x[1])
                        median_idx = len(rrmse_list) // 2
                        median_run_key = rrmse_list[median_idx][0]
                        highlighted_runs_per_fold[fold_key] = median_run_key
                        if verbose:
                            print(f"Fold {fold_key}: Median RRMSE run is {median_run_key} (RRMSE={rrmse_list[median_idx][1]:.6f})")
        else:
            # Old format: find overall median RRMSE run
            rrmse_list = []
            for run_key in runs_data.keys():
                if run_key in rrmse_values:
                    rrmse_value = rrmse_values[run_key]
                    if rrmse_value is not None and not np.isnan(rrmse_value) and not np.isinf(rrmse_value):
                        rrmse_list.append((run_key, rrmse_value))
            
            if rrmse_list:
                rrmse_list.sort(key=lambda x: x[1])
                median_idx = len(rrmse_list) // 2
                best_run_idx = rrmse_list[median_idx][0]
                if verbose:
                    print(f"Median RRMSE run is {best_run_idx} (RRMSE={rrmse_list[median_idx][1]:.6f})")
    else:
        # Fallback to best loss run (green)
        highlight_color = 'green'
        highlight_label_suffix = " (Best)"
        if is_fold_format:
            for fold_key, fold_data in folds_data.items():
                runs = fold_data.get("runs", {})
                if runs:
                    # Find best run within this fold by loss
                    best_loss = float('inf')
                    best_run_key = None
                    for run_key, run_data in runs.items():
                        records = run_data.get("records", [])
                        for record in records:
                            loss = record.get("loss")
                            if loss is not None and loss < best_loss:
                                best_loss = loss
                                best_run_key = run_key
                    if best_run_key is not None:
                        highlighted_runs_per_fold[fold_key] = best_run_key
            if verbose:
                print(f"Found {len(highlighted_runs_per_fold)} folds with best loss runs identified")
        else:
            # Old format: find overall best run
            best_run_idx = _get_best_run_index(data) if len(runs_data) > 1 else None
            if verbose:
                if best_run_idx is not None:
                    print(f"Best run identified: Run {best_run_idx}")
                else:
                    print("Single run detected (no best run to highlight)")
    
    # Extract all parameter paths from first run (or best run if available)
    if is_fold_format:
        # Use first fold's first run as reference
        first_fold = list(folds_data.keys())[0]
        first_run = list(folds_data[first_fold].get("runs", {}).keys())[0]
        reference_records = folds_data[first_fold]["runs"][first_run].get("records", [])
    else:
        reference_run = best_run_idx if best_run_idx is not None else list(runs_data.keys())[0]
        reference_records = runs_data[reference_run].get("records", [])
    
    if not reference_records:
        print(f"Warning: No records found")
        return
    
    # Extract parameter paths from first record
    all_param_paths = _extract_parameter_paths(reference_records[0].get("parameters", {}))
    
    # Filter out all neural network parameters - only keep GP parameters (lengthscales, outputscale, noise, etc.)
    # Remove all neural network layers: trunk_layers, head_layers, and layers (from InputTransformNet)
    filtered_param_paths = []
    for path, value in all_param_paths:
        path_str = path
        # Skip all neural network layers: trunk_layers, head_layers, and layers
        if ".trunk_layers[" in path_str or ".head_layers[" in path_str or ".layers[" in path_str:
            continue
        # Keep only GP parameters (lengthscales, outputscale, noise, etc.)
        filtered_param_paths.append((path, value))
    all_param_paths = filtered_param_paths
    
    # Filter parameter paths if specified
    if parameter_paths is not None:
        # Filter to only requested paths
        filtered_paths = [p for p in all_param_paths if any(req_path in p[0] for req_path in parameter_paths)]
        if not filtered_paths:
            print(f"Warning: No parameters found matching paths: {parameter_paths}")
            return
        param_paths_to_plot = filtered_paths
    else:
        # Use all parameters, optionally limited by max_parameters
        if max_parameters is not None:
            param_paths_to_plot = all_param_paths[:max_parameters]
            if verbose and len(all_param_paths) > max_parameters:
                print(f"Note: Found {len(all_param_paths)} parameters, plotting first {max_parameters}. "
                      f"Set max_parameters=None to plot all.")
        else:
            param_paths_to_plot = all_param_paths
    
    if verbose:
        print(f"Plotting {len(param_paths_to_plot)} parameters")
    
    # Determine number of subplots
    n_params = len(param_paths_to_plot)
    # Adjust ncols based on number of parameters for better layout
    if n_params <= 4:
        ncols = n_params
    elif n_params <= 12:
        ncols = 4
    elif n_params <= 24:
        ncols = 6
    else:
        ncols = 8  # For many parameters, use more columns
    nrows = math.ceil(n_params / ncols)
    
    # Adjust figure size based on number of parameters
    # Base size per subplot: approximately 3x2.5 inches
    base_width_per_subplot = 3.0
    base_height_per_subplot = 2.5
    # For many parameters, reduce size per subplot slightly to fit on screen
    if n_params > 50:
        width_per_subplot = 2.5
        height_per_subplot = 2.0
    elif n_params > 20:
        width_per_subplot = 2.8
        height_per_subplot = 2.2
    else:
        width_per_subplot = base_width_per_subplot
        height_per_subplot = base_height_per_subplot
    
    adjusted_figsize = (ncols * width_per_subplot, nrows * height_per_subplot)
    # Cap maximum figure size to prevent issues
    max_width = 40
    max_height = 60
    adjusted_figsize = (min(adjusted_figsize[0], max_width), min(adjusted_figsize[1], max_height))
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=adjusted_figsize)
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes
    
    # Generate green gradient colors for best runs (if using green highlighting)
    highlighted_runs_list = []
    if highlight_color == 'green' and is_fold_format:
        # Collect all highlighted runs to assign gradient colors
        for fold_key in sorted(folds_data.keys()):
            if fold_key in highlighted_runs_per_fold:
                highlighted_runs_list.append((fold_key, highlighted_runs_per_fold[fold_key]))
        green_colors = _get_green_gradient_colors(len(highlighted_runs_list))
        # Create mapping from (fold_key, run_key) to color
        highlighted_run_colors = {}
        for idx, (fold_key, run_key) in enumerate(highlighted_runs_list):
            highlighted_run_colors[(fold_key, run_key)] = green_colors[idx]
    else:
        highlighted_run_colors = {}
    
    # Plot each parameter
    for param_idx, (param_path, _) in enumerate(param_paths_to_plot):
        ax = axes[param_idx]
        
        # Collect data for all runs (grouped by fold if fold format)
        if is_fold_format:
            # Iterate through folds and runs
            for fold_key, fold_data in folds_data.items():
                runs = fold_data.get("runs", {})
                highlighted_run_key = highlighted_runs_per_fold.get(fold_key)
                
                for run_key, run_data in runs.items():
                    records = run_data.get("records", [])
                    if not records:
                        continue
                    
                    # Extract epochs and parameter values
                    epochs = []
                    param_values = []
                    
                    for record in records:
                        epoch_num = record.get("epoch")
                        if epoch_num is None:
                            continue
                        
                        # Navigate to parameter value
                        param_value = _get_nested_value(record.get("parameters", {}), param_path)
                        if param_value is not None:
                            epochs.append(epoch_num)
                            # Handle list values
                            if isinstance(param_value, list):
                                if len(param_value) > 0:
                                    # If it's a single-element list (scalar wrapped in list), unwrap it
                                    if len(param_value) == 1:
                                        param_values.append(float(param_value[0]))
                                    else:
                                        # For multi-element lists, check if it's a neural network weight/bias
                                        # Neural network weights/biases: use L2 norm to show magnitude changes
                                        # Other lists (like lengthscales): use mean
                                        if "weight" in param_path.lower() or "bias" in param_path.lower():
                                            # For neural network weights/biases, use L2 norm to better show changes
                                            param_array = np.array(param_value)
                                            l2_norm = float(np.linalg.norm(param_array))
                                            param_values.append(l2_norm)
                                        else:
                                            # For other lists (e.g., lengthscales), use mean
                                            param_values.append(float(np.mean(param_value)))
                                else:
                                    continue
                            else:
                                param_values.append(float(param_value))
                    
                    if not epochs:
                        continue
                    
                    # Determine color and style: highlighted run (median RRMSE or best loss)
                    is_highlighted = (highlighted_run_key is not None and run_key == highlighted_run_key)
                    if is_highlighted:
                        # Use gradient green for best runs, or red for median RRMSE
                        if highlight_color == 'green' and (fold_key, run_key) in highlighted_run_colors:
                            color = highlighted_run_colors[(fold_key, run_key)]
                            annotation_color = color  # Use the same gradient color for annotation
                        else:
                            color = highlight_color
                            annotation_color = highlight_color
                    else:
                        color = 'gray'
                        annotation_color = 'gray'
                    line_alpha = 1.0 if is_highlighted else alpha
                    line_width = linewidth * 1.5 if is_highlighted else linewidth
                    fold_num = fold_key.replace("fold_", "") if fold_key.startswith("fold_") else fold_key
                    run_num = run_key.replace("run_", "") if run_key.startswith("run_") else run_key
                    # Create label with fold and run indices
                    if is_highlighted:
                        label = f"F{fold_num}R{run_num}{highlight_label_suffix}"
                    else:
                        label = f"F{fold_num}R{run_num}"
                    
                    # Plot (only add label to first subplot to avoid duplicate legends)
                    line = None
                    if param_idx == 0:
                        line, = ax.plot(epochs, param_values, color=color, alpha=line_alpha, linewidth=line_width, label=label)
                    else:
                        line, = ax.plot(epochs, param_values, color=color, alpha=line_alpha, linewidth=line_width)
                    
                    # Add annotation for highlighted runs (only on first parameter subplot to avoid clutter)
                    if is_highlighted and param_idx == 0 and len(epochs) > 0 and len(param_values) > 0:
                        # Annotate at the last point
                        last_idx = len(epochs) - 1
                        ax.annotate(f"F{fold_num}R{run_num}", 
                                   xy=(epochs[last_idx], param_values[last_idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=7, color=annotation_color, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=annotation_color, alpha=0.7))
        else:
            # Old format: iterate through runs
            for run_idx, run_data in runs_data.items():
                records = run_data.get("records", [])
                if not records:
                    continue
                
                # Extract epochs and parameter values
                epochs = []
                param_values = []
                
                for record in records:
                    epoch_num = record.get("epoch")
                    if epoch_num is None:
                        continue
                    
                    # Navigate to parameter value
                    param_value = _get_nested_value(record.get("parameters", {}), param_path)
                    if param_value is not None:
                        epochs.append(epoch_num)
                        # Handle list values
                        if isinstance(param_value, list):
                            if len(param_value) > 0:
                                # If it's a single-element list (scalar wrapped in list), unwrap it
                                if len(param_value) == 1:
                                    param_values.append(float(param_value[0]))
                                else:
                                    # For multi-element lists, check if it's a neural network weight/bias
                                    # Neural network weights/biases: use L2 norm to show magnitude changes
                                    # Other lists (like lengthscales): use mean
                                    if "weight" in param_path.lower() or "bias" in param_path.lower():
                                        # For neural network weights/biases, use L2 norm to better show changes
                                        param_array = np.array(param_value)
                                        l2_norm = float(np.linalg.norm(param_array))
                                        param_values.append(l2_norm)
                                    else:
                                        # For other lists (e.g., lengthscales), use mean
                                        param_values.append(float(np.mean(param_value)))
                            else:
                                continue
                        else:
                            param_values.append(float(param_value))
                
                if not epochs:
                    continue
                
                # Determine color and style
                is_best = (best_run_idx is not None and run_idx == best_run_idx)
                color = 'green' if is_best else 'gray'
                line_alpha = 1.0 if is_best else alpha
                line_width = linewidth * 1.5 if is_best else linewidth
                label = f"Run {run_idx}" + (" (Best)" if is_best else "")
                
                # Plot (only add label to first subplot to avoid duplicate legends)
                if param_idx == 0:
                    ax.plot(epochs, param_values, color=color, alpha=line_alpha, linewidth=line_width, label=label)
                else:
                    ax.plot(epochs, param_values, color=color, alpha=line_alpha, linewidth=line_width)
        
        # Customize subplot
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Parameter Value")
        # Create compact title
        title = _make_compact_title(param_path)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])
    
    # Create legend showing fold and run indices
    # For fold format, show only best runs in legend to avoid clutter
    if is_fold_format:
        # Collect only best run labels for legend
        best_labels = []
        best_handles = []
        for ax in axes[:n_params]:
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            if ax_handles and ax_labels:
                for handle, label in zip(ax_handles, ax_labels):
                    if "(Best)" in label and label not in best_labels:
                        best_labels.append(label)
                        best_handles.append(handle)
                if best_handles:
                    break
        
        if best_handles and best_labels:
            # Show only best runs in legend
            fig.legend(best_handles, best_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=min(len(best_handles), 10), fontsize=7, frameon=True, 
                      title='Best Runs (Green)', title_fontsize=8)
        else:
            # Fallback: show all runs if not too many
            total_runs = sum(len(fold_data.get("runs", {})) for fold_data in folds_data.values())
            if total_runs <= 20:
                handles, labels = None, None
                for ax in axes[:n_params]:
                    ax_handles, ax_labels = ax.get_legend_handles_labels()
                    if ax_handles:
                        handles, labels = ax_handles, ax_labels
                        break
                if handles and labels:
                    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                              ncol=min(len(handles), 8), fontsize=7, frameon=True)
    else:
        # Old format: show all runs if not too many
        total_runs = len(runs_data)
        if total_runs <= 20:
            handles, labels = None, None
            for ax in axes[:n_params]:
                ax_handles, ax_labels = ax.get_legend_handles_labels()
                if ax_handles:
                    handles, labels = ax_handles, ax_labels
                    break
            
            if handles and labels:
                # Create a single legend for the entire figure
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                          ncol=min(len(handles), 8), fontsize=7, frameon=True)
    
    # Overall title
    fig.suptitle("Parameter Evolution Over Epochs", fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Adjust bottom margin for legend
    
    # Save plot
    if save_dir is None:
        save_dir = os.path.dirname(json_file)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "parameters_by_epoch.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Parameter epoch plot saved to: {save_path}")


def _get_nested_value(data_dict: Dict, path: str) -> Any:
    """
    Get a value from a nested dictionary using a dot-separated path.
    Handles list indices in path (e.g., "base_kernel[0].lengthscale").
    
    Args:
        data_dict: Nested dictionary
        path: Dot-separated path (e.g., "covar_module.base_kernel.lengthscale")
    
    Returns:
        Value at path, or None if not found
    """
    parts = path.split('.')
    current = data_dict
    
    for part in parts:
        # Check for list index (e.g., "key[0]")
        if '[' in part and part.endswith(']'):
            key, index_str = part.split('[')
            index = int(index_str.rstrip(']'))
            if key:
                current = current.get(key)
            if current is None or not isinstance(current, (list, tuple)) or index >= len(current):
                return None
            current = current[index]
        else:
            current = current.get(part)
            if current is None:
                return None
    
    return current
