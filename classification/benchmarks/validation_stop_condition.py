# Function for early stopping based on validation NLL

import copy

import numpy as np
import torch
from gpplus.training.stop_conditions import StopCondition, StopConditionContext
from gpplus.utils.dirichlet import prepare_dirichlet_targets


class ValidationNLLStopCondition(StopCondition):

    def __init__(
        self,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        patience: int = 20,
        check_every: int = 10,
        min_delta: float = 1e-2,
        verbose: bool = False,
    ):
        self.val_x       = val_x
        self.val_y       = val_y
        self.patience    = patience
        self.check_every = check_every
        self.min_delta   = min_delta
        self.verbose     = verbose

        self.best_val_nll    = float("inf")
        self.checks_no_imp   = 0
        self.best_state      = None
        self.stopped_epoch   = None
        self._epoch_count    = 0
        self._warned         = False
        self.val_nll_history = []

    def _compute_val_nll(self, model) -> float:
        # Compute negative log-likelihood on the validation set
        model.eval()
        with torch.no_grad():
            try:
                alpha_eps = model.likelihood.alpha_epsilon
                dtype     = model.likelihood.transformed_targets.dtype
                device    = next(model.parameters()).device

                sigma2_CxN, Y_tilde, _ = prepare_dirichlet_targets(
                    self.val_y.long(), alpha_epsilon=alpha_eps, dtype=dtype)

                ymean    = model.likelihood.ymean
                Y_tilde  = (Y_tilde - ymean.unsqueeze(0)).to(device)
                val_x    = self.val_x.to(device=device, dtype=dtype)

                output   = model(val_x)
                C        = Y_tilde.shape[1]
                nll      = 0.0
                for c in range(C):
                    dist = torch.distributions.Normal(
                        output.mean[c],
                        (output.variance[c] + sigma2_CxN[c].to(device)).sqrt(),
                    )
                    nll -= dist.log_prob(Y_tilde[:, c]).sum().item()

            # pylint: disable=broad-exception-caught
            except Exception as exc:  # noqa: BLE001
                if not self._warned:
                    self._warned = True
                    print(f"    [ValNLL] validation NLL failed ({exc}); "
                          f"early stopping is inactive for this fit.")
                nll = float("nan")

        model.train()
        return float(nll)

    def should_stop(self, context: StopConditionContext) -> tuple[bool, str]:
        self._epoch_count += 1

        # Check every check_every epochs
        if self._epoch_count % self.check_every != 0:
            return False, ""

        model     = context["model"]
        epoch     = context["epoch"]
        val_nll   = self._compute_val_nll(model)

        if np.isnan(val_nll):
            return False, ""

        self.val_nll_history.append(val_nll)

        if self.verbose:
            print(f"    [ValNLL] epoch {epoch+1}  val_nll={val_nll:.6f}  "
                  f"best={self.best_val_nll:.6f}  no_imp={self.checks_no_imp}")

        if val_nll < self.best_val_nll - self.min_delta:
            self.best_val_nll  = val_nll
            self.checks_no_imp = 0
            # Save best state
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.checks_no_imp += 1
            if self.checks_no_imp >= self.patience:
                # Restore best state before stopping
                if self.best_state is not None:
                    model.load_state_dict(self.best_state)
                self.stopped_epoch = epoch + 1
                return (
                    True,
                    (f"Val NLL no improvement for {self.patience} checks "
                     f"(check_every={self.check_every}, "
                     f"stopped at epoch {self.stopped_epoch})"),
                )

        return False, ""
