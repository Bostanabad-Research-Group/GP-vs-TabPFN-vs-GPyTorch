import gpytorch
import torch

from ..config import logger


def compute_accuracy(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes classification accuracy from predicted class probabilities.

    Args:
        probs (torch.Tensor): Predicted class probabilities. Shape: (n_samples, num_classes).
        targets (torch.Tensor): True class labels (0 to num_classes-1). Shape: (n_samples,).

    Returns:
        float: Fraction of samples whose argmax-predicted class matches the target.
    """
    predicted = torch.argmax(probs, dim=-1)
    targets = targets.to(device=predicted.device).long().view(-1)
    return (predicted == targets).float().mean().item()


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (Milios et al., 2018, Eq. 1).

    Confidence is the max predicted probability per sample. Samples are grouped
    into ``n_bins`` equal-width bins over [0, 1] by confidence; each bin contributes
    the absolute gap between its mean confidence and its accuracy, weighted by the
    bin's share of the total sample count.

    Args:
        probs (torch.Tensor): Predicted class probabilities. Shape: (n_samples, num_classes).
        targets (torch.Tensor): True class labels. Shape: (n_samples,).
        n_bins (int, optional): Number of equal-width confidence bins. Default: 10.

    Returns:
        float: Expected Calibration Error in [0, 1].
    """
    confidences, predicted = probs.max(dim=-1)
    targets = targets.to(device=predicted.device).long().view(-1)
    accuracies = (predicted == targets).float()

    n = targets.shape[0]
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Upper-inclusive on the final bin so confidence == 1.0 is counted.
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_count = in_bin.sum()
        if bin_count > 0:
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += (bin_count.float() / n) * torch.abs(bin_confidence - bin_accuracy)

    return ece.item()


def evaluate_gpc_model(
    model,
    test_x: torch.Tensor,
    test_y: torch.Tensor = None,
    num_mc_samples: int = 256,
    n_bins: int = 10,
):
    """
    Evaluates a GPC on test data, returning class probabilities and,
    if labels are provided, accuracy and Expected Calibration Error.

    Similar to ``evaluate_gp_model`` (regression): puts the model in eval mode and
    predicts under the slower, stable ``fast_computations(False, ...)`` settings
    to reduce host-to-host variance. Prediction itself is delegated to
    ``GPC.predict``, which adds ``ymean`` back and computes probabilities
    via softmax(mean) (``num_mc_samples=0``) or MC integration of Eq. (8)
    (``num_mc_samples > 0``).

    Args:
        model (GPC):
            The trained classification model to evaluate.
        test_x (torch.Tensor):
            Test data features. Shape: (n_samples, n_features).
        test_y (torch.Tensor, optional):
            True class labels. Shape: (n_samples,). If None, metrics are skipped
            and only the prediction dictionary is returned.
        num_mc_samples (int, optional):
            Monte Carlo samples passed through to ``predict``. 0 uses the fast
            softmax(mean) path. Default: 256.
        n_bins (int, optional):
            Number of confidence bins for ECE. Default: 10.

    Returns:
        dict: The ``predict`` output (keys: ``probs``, ``classes``, ``logits``,
        and ``latent_std``/``entropy``), augmented with ``accuracy`` and ``ece``
        (floats) when ``test_y`` is provided.
    """
    model.eval()

    with (
        torch.no_grad(),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ),
    ):
        # Align test inputs to the training tensor's device/dtype, as in
        # evaluate_gp_model. predict() also guards dtype, but this keeps the
        # contract identical across the two evaluators.
        train_inputs = getattr(model, "train_inputs", None)
        if train_inputs and len(train_inputs) > 0:
            reference = train_inputs[0]
            test_x = test_x.to(device=reference.device, dtype=reference.dtype)

        results = model.predict(test_x, num_mc_samples=num_mc_samples)

    if test_y is not None:
        probs = results["probs"]
        test_y = test_y.to(device=probs.device)
        results["accuracy"] = compute_accuracy(probs, test_y)
        results["ece"] = compute_ece(probs, test_y, n_bins=n_bins)
        logger.info(
            "Classification evaluation completed. Accuracy: %.4f, ECE: %.4f",
            results["accuracy"],
            results["ece"],
        )
    else:
        logger.info("Classification prediction completed (no labels provided; metrics skipped).")

    return results
