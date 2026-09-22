# 1D binary classification: the effect of alpha_epsilon
# Fits GP+ at several alpha_epsilon values and plots the posterior class
# probability for each, alongside TabPFN v2.5 and v2.0 on the same data
# Produces one figure

import os
import random

import gpytorch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpplus.models.gpc import GPC
from gpplus.training.trainer import GPTrainer
from gpplus.utils.factory import build_scaled_kernel
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Non-interactive backend: this script only ever writes a file.
matplotlib.use("Agg")

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    try:
        from tabpfn.constants import ModelVersion
        TABPFN_V2_AVAILABLE = True
    except ImportError:
        TABPFN_V2_AVAILABLE = False
except ImportError as _e:
    TABPFN_AVAILABLE = False
    TABPFN_V2_AVAILABLE = False
    print(f"TabPFN not available: {_e}")



ALPHA_VALUES = [1e-3, 1e-2, 1e-1]

# Training point positions
CLASS_1_POINTS = [-0.90, -0.75, 0.15, 0.30, 0.55]
CLASS_0_POINTS = [-0.65, -0.55, -0.45, -0.35, -0.25, 0.60, 0.72, 0.82, 0.92]

# Either class is complete on its own since probabilities add to 1 for both
PLOT_CLASS = 1

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "results",
    "onedim",
    "onedim_example.png",
)


# Experiment settings

KERNEL = "gaussian"
NUM_MC = 256          # MC samples for the posterior
BAND_Q = 95           # Prediction Interval
NUM_INITS = 16        # Restarts per fit

SEEDS = [0]
DTYPE = torch.float64
N_GRID = 300          # prediction grid resolution
GRID_MARGIN = 0.15    # how far the grid extends past the outermost point

ADAM = {"lr": 0.01, "num_epochs": 4000}

# Figure style

GP_COLOR = "#2ca02c"
TABPFN_STYLES = {
    "v25": {"label": "v2.5 mean", "color": "#9467bd"},
    "v2":  {"label": "v2.0 mean", "color": "#d62728"},
}
TABPFN_HEADER = "TabPFN"

PANEL_W, PANEL_H = 2.5, 1.8   # inches per panel
Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
GRID_ALPHA = 0.35
FS_PANEL_LABEL = 15
FS_TICKS = 12
FS_LEGEND = 12
DPI = 300


def make_1d_data():
    x1 = np.asarray(CLASS_1_POINTS, dtype=np.float64)
    x0 = np.asarray(CLASS_0_POINTS, dtype=np.float64)

    if len(x0) == 0 or len(x1) == 0:
        raise ValueError("Both classes need at least one training point.")

    X = np.concatenate([x0, x1]).reshape(-1, 1)
    y = np.concatenate([np.zeros(len(x0), dtype=np.int64),
                        np.ones(len(x1), dtype=np.int64)])
    return X, y


# Returns (model, best_loss) for the best restart of seed
def train_gpc(X_train, y_train, alpha_eps, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_x = torch.tensor(X_train, dtype=DTYPE)
    train_y = torch.tensor(y_train, dtype=torch.long)

    covar_module = build_scaled_kernel(
        kind=KERNEL, batch_shape=torch.Size([2]), ard_num_dims=1)

    model = GPC(train_x=train_x, train_y=train_y,
                covar_module=covar_module, alpha_epsilon=alpha_eps)

    trainer = GPTrainer(
        model=model,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": ADAM["lr"]},
        num_epochs=ADAM["num_epochs"],
        num_inits=NUM_INITS,
        seed=seed,
        dtype=DTYPE,
        stop_conditions=[],
    )
    results = trainer.train()

    valid = [r for r in results if r.get("loss") is not None and not r.get("error")]
    if not valid:
        raise RuntimeError(f"every initialization failed at alpha={alpha_eps}, seed={seed}")
    return model, float(min(r["loss"] for r in valid))


def predict_with_bands(model, x_grid):
    test_x = torch.tensor(x_grid.reshape(-1, 1), dtype=DTYPE)

    model.eval()
    model.likelihood.eval()

    with torch.no_grad():
        with gpytorch.settings.fast_pred_var():
            latent = model(test_x)
            logits = latent.mean                                   # (C, N)
            latent_std = torch.sqrt(latent.variance.clamp_min(1e-12))

        logits = logits + model.ymean.unsqueeze(1)

        eps = torch.randn(NUM_MC, *logits.shape,
                          device=logits.device, dtype=logits.dtype)
        samples = logits.unsqueeze(0) + eps * latent_std.unsqueeze(0)   # (S, C, N)
        probs = torch.softmax(samples.permute(0, 2, 1), dim=-1)         # (S, N, C)

    probs = probs.numpy()
    tail = (100 - BAND_Q) / 2
    return (
        probs.mean(axis=0),
        np.percentile(probs, tail, axis=0),
        np.percentile(probs, 100 - tail, axis=0),
    )


def describe_fit(model):
    cm = getattr(model, "covar_module", None)
    base = getattr(cm, "base_kernel", None) or getattr(cm, "data_covar_module", None)
    parts = []
    if base is not None and hasattr(base, "lengthscale"):
        ls = base.lengthscale.detach().flatten().tolist()
        parts.append("lengthscale=[" + ", ".join(f"{v:.4g}" for v in ls) + "]")
    if cm is not None and hasattr(cm, "outputscale"):
        os_ = cm.outputscale.detach().flatten().tolist()
        parts.append("outputscale=[" + ", ".join(f"{v:.4g}" for v in os_) + "]")
    return "  ".join(parts) if parts else "(no scalar hyperparameters found)"


def predict_tabpfn(X_train, y_train, x_grid, version_key, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if version_key == "v25":
        clf = TabPFNClassifier(device=device)
    else:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device)
    clf.fit(X_train, y_train)
    return clf.predict_proba(x_grid.reshape(-1, 1))


def _style_axis(ax, x_grid, show_y_labels):
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(Y_TICKS)
    ax.set_xticklabels([])
    if not show_y_labels:
        ax.set_yticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.5)
    ax.grid(axis="x", alpha=GRID_ALPHA * 0.7, linewidth=0.5)
    ax.axhline(0.5, color="#999999", lw=0.5, ls=":", zorder=1)
    ax.tick_params(labelsize=FS_TICKS, pad=2)


def _draw_training_points(ax, X_train, y_train):
    is_class = y_train == PLOT_CLASS
    ax.scatter(X_train[is_class, 0], np.ones(is_class.sum()),
               color="black", s=20, zorder=5, clip_on=False)
    ax.scatter(X_train[~is_class, 0], np.zeros((~is_class).sum()),
               s=20, zorder=5, clip_on=False,
               facecolors="white", edgecolors="black", linewidth=0.8)


def _legend_handles(tabpfn_results):
    handles = [
        Line2D([], [], marker="o", color="black", linestyle="none", markersize=4.5),
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=0.8, color="black", linestyle="none", markersize=4.5),
        Line2D([], [], color=GP_COLOR, lw=1.4),
        Patch(facecolor=GP_COLOR, alpha=0.25, linewidth=0),
    ]
    labels = [
        rf"Train ($y={PLOT_CLASS}$)",
        rf"Train ($y \neq {PLOT_CLASS}$)",
        "GP mean",
        f"GP {BAND_Q}% PI",
    ]
    for vkey, style in TABPFN_STYLES.items():
        if vkey in tabpfn_results:
            handles.append(Line2D([], [], color=style["color"], lw=1.4))
            labels.append(style["label"])
    return handles, labels


def _panel_label(index, text):
    return f"({chr(ord('a') + index)}) {text}"


def plot_figure(gp_results, tabpfn_results, X_train, y_train, x_grid, out_path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "savefig.pad_inches": 0.02,
    })

    n_panels = len(ALPHA_VALUES) + (1 if tabpfn_results else 0)
    fig, axes = plt.subplots(1, n_panels, squeeze=False,
                             figsize=(PANEL_W * n_panels, PANEL_H))
    axes = axes[0]

    for col, alpha_eps in enumerate(ALPHA_VALUES):
        ax = axes[col]
        res = gp_results[alpha_eps]
        ax.plot(x_grid, res["mu"][:, PLOT_CLASS], color=GP_COLOR, lw=1.2, zorder=3)
        ax.fill_between(x_grid, res["lb"][:, PLOT_CLASS], res["ub"][:, PLOT_CLASS],
                        color=GP_COLOR, alpha=0.25, zorder=2)
        _draw_training_points(ax, X_train, y_train)
        _style_axis(ax, x_grid, show_y_labels=col == 0)
        ax.set_xlabel(_panel_label(col, rf"$\mathbf{{\alpha_\epsilon = {alpha_eps}}}$"),
                      fontsize=FS_PANEL_LABEL, fontweight="bold", labelpad=6)

    if tabpfn_results:
        ax = axes[-1]
        for vkey, probs in tabpfn_results.items():
            ax.plot(x_grid, probs[:, PLOT_CLASS], color=TABPFN_STYLES[vkey]["color"],
                    lw=1.2, zorder=3)
        _draw_training_points(ax, X_train, y_train)
        _style_axis(ax, x_grid, show_y_labels=False)
        ax.set_xlabel(_panel_label(len(ALPHA_VALUES), TABPFN_HEADER),
                      fontsize=FS_PANEL_LABEL, fontweight="bold", labelpad=6)

    handles, labels = _legend_handles(tabpfn_results)
    fig.legend(handles, labels,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
               fontsize=FS_LEGEND, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), frameon=False, handlelength=1.8,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0)

    fig.tight_layout(pad=0.4, w_pad=0.6, rect=(0, 0, 1, 0.88))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    X_train, y_train = make_1d_data()
    print(f"N={len(y_train)}  class 0: {(y_train == 0).sum()}  "
          f"class 1: {(y_train == 1).sum()}")

    x_grid = np.linspace(X_train.min() - GRID_MARGIN,
                         X_train.max() + GRID_MARGIN, N_GRID)

    print(f"{len(SEEDS)} seed(s) x {NUM_INITS} restarts = "
          f"{len(SEEDS) * NUM_INITS} optimizations per alpha\n")

    gp_results = {}
    for alpha_eps in ALPHA_VALUES:
        print(f"alpha_epsilon={alpha_eps}")
        best_model, best_loss, best_seed = None, float("inf"), None
        losses = []
        for seed in SEEDS:
            model, loss = train_gpc(X_train, y_train, alpha_eps, seed)
            losses.append(loss)
            print(f"  seed {seed:2d}: loss={loss:.6f}")
            if loss < best_loss:
                best_model, best_loss, best_seed = model, loss, seed

        spread = max(losses) - min(losses)
        print(f"  best: seed {best_seed}, loss={best_loss:.6f}  "
              f"(spread across seeds: {spread:.6f})")
        print(f"  {describe_fit(best_model)}")

        mu, lb, ub = predict_with_bands(best_model, x_grid)
        gp_results[alpha_eps] = {"mu": mu, "lb": lb, "ub": ub}
        print()

    tabpfn_results = {}
    if TABPFN_AVAILABLE:
        versions = ["v25"] + (["v2"] if TABPFN_V2_AVAILABLE else [])
        for vkey in versions:
            print(f"TabPFN {vkey} over {len(SEEDS)} seed(s) ... ", end="", flush=True)
            probs = [predict_tabpfn(X_train, y_train, x_grid, vkey, s) for s in SEEDS]
            stacked = np.stack(probs, axis=0)
            tabpfn_results[vkey] = stacked.mean(axis=0)
            drift = float(np.abs(stacked - stacked[0]).max())
            print(f"done (max deviation across seeds: {drift:.2e})")

    plot_figure(gp_results, tabpfn_results, X_train, y_train, x_grid, OUTPUT_PATH)
    print(f"\nFigure saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
