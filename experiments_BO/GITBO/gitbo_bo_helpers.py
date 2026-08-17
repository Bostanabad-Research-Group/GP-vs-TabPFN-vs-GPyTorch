"""
Helpers for gradient-informed (GITBO) PFN BO, used by run_BO_pfn when gi_pfn=True.
Extracted from _GITBO to avoid _utils dependency when called from run_BO.
"""
import numpy as np
import torch

from .tabpfn_wrapper import VanillaDirectTabPFNRegressor


def sample_dominant_subspace(
    x: torch.Tensor,
    y: torch.Tensor,
    DIM: int,
    grad_vals: torch.Tensor,
    SEED,
    rank_r: int = 15,
    n_samples: int = 5000,
    N_CANDIDATES: int = 1,
    scale: float = 0.2,
    GI_SUBSPACE: bool = True,
    new_origin=None,
    tkwargs=None,
):
    """
    Sample points in the gradient-informed dominant subspace.
    Returns X_pen [n_samples, N_CANDIDATES, DIM] in [0,1].
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, **tkwargs)

    n_points, batch_size, feat_dim = grad_vals.shape
    grad_vals_np = grad_vals.cpu().detach().numpy()
    x_np = x.cpu().detach().numpy()

    X_pen = torch.zeros((n_samples, batch_size, feat_dim), **tkwargs)

    # rank_r cannot exceed feat_dim (we only have feat_dim eigenvectors)
    r_actual = min(rank_r, feat_dim)

    for b in range(batch_size):
        H_est = (grad_vals_np[:, b, :].T @ grad_vals_np[:, b, :]) / n_points
        eigvals, eigvecs = np.linalg.eigh(H_est)
        idx_sorted = np.argsort(eigvals)[::-1]
        eigenvecs = eigvecs[:, idx_sorted]
        U_r = eigenvecs[:, :r_actual]

        origin = x_np.mean(axis=0) if new_origin is None else new_origin.cpu().numpy().mean(axis=0)
        alpha = np.random.uniform(-scale, scale, size=(n_samples, r_actual))
        samples = origin + (alpha @ U_r.T)
        samples = torch.tensor(samples, **tkwargs)
        samples = torch.clamp(samples, 0.0, 1.0).to(**tkwargs)
        X_pen[:, b, :] = samples.detach().to(**tkwargs)

    return X_pen


def compute_acquisition_values_gitbo(
    acquisition: str,
    DIM: int,
    N_PENDING: int,
    N_CANDIDATES: int,
    trained_X: torch.Tensor,
    trained_Y: torch.Tensor,
    X_pen: torch.Tensor,
    gpu_device: str,
    tkwargs: dict,
) -> tuple:
    """
    Compute acquisition values and gradients using VanillaDirectTabPFNRegressor.
    Returns: (acq [N_PENDING, N_CANDIDATES], None, grad_est [N_PENDING, N_CANDIDATES, DIM])
    trained_X, trained_Y, X_pen must be in [0,1]^D.
    """
    regressor = VanillaDirectTabPFNRegressor(device=gpu_device)
    single_eval_pos = trained_X.shape[0]

    X_train = trained_X.unsqueeze(1).expand(-1, N_CANDIDATES, -1)
    X_full = torch.cat([X_train, X_pen], dim=0)

    Y_pad = torch.zeros(N_PENDING, 1, **tkwargs)
    Y_full = torch.cat([trained_Y, Y_pad], dim=0).unsqueeze(1)
    Y_full = Y_full.expand(-1, N_CANDIDATES, -1)

    X_train_det = X_full[:single_eval_pos].detach()
    X_cand = X_full[single_eval_pos:].clone().requires_grad_(True)
    X_concat = torch.cat([X_train_det, X_cand], dim=0).to(gpu_device)

    amp_dtype = tkwargs["dtype"]
    amp_device = "cuda" if "cuda" in gpu_device else "cpu"

    acq_name = "ThompsonSampling" if acquisition.upper() == "TS" else acquisition.upper()

    if acq_name in ("EI", "TR_EI"):
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out = regressor.forward(X_concat, Y_full, single_eval_pos)
            logits = out["standard"]
            acq = regressor.predict_ei(logits, trained_Y.max())
        EI = acq[single_eval_pos:]
        grad_cand, = torch.autograd.grad(EI.sum(), X_cand, retain_graph=False, create_graph=False)
        grad_est = -grad_cand.view(N_PENDING, N_CANDIDATES, DIM).detach()
        return EI.to(**tkwargs), None, grad_est.to(**tkwargs)

    elif acq_name in ("ThompsonSampling", "TR_TS"):
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out = regressor.forward(X_concat, Y_full, single_eval_pos)
            logits = out["standard"]
            output_mean = regressor.predict_mean(logits)
            output_variance = regressor.predict_variance(logits)
        mu_cand = output_mean[single_eval_pos:]
        var_cand = output_variance[single_eval_pos:]
        std_cand = torch.clamp(var_cand, min=1e-8).sqrt()
        sample_count = 512
        mu_expanded = mu_cand.unsqueeze(-1).expand(-1, -1, sample_count)
        std_expanded = std_cand.unsqueeze(-1).expand(-1, -1, sample_count)
        sampled_y = torch.normal(mu_expanded, std_expanded).mean(dim=-1)
        loss = mu_cand.sum()
        grad_cand, = torch.autograd.grad(loss, X_cand, retain_graph=False, create_graph=False)
        grad_est = -grad_cand.view(N_PENDING, N_CANDIDATES, DIM).detach()
        return sampled_y.to(**tkwargs), None, grad_est.to(**tkwargs)

    else:
        raise ValueError(f"Unknown acquisition: {acquisition}")
