import copy
import time
from typing import List, Optional

import gpytorch
import linear_operator
import torch

try:
    from linear_operator.utils.errors import NotPSDError
except ImportError:
    NotPSDError = Exception  # type: ignore[misc, assignment]

from ..config import logger
from .callbacks import (
    Callback,
    CallbackOnEpochEndContext,
    CallbackOnEpochStartContext,
    CallbackOnTrainEndContext,
    CallbackOnTrainStartContext,
)
from .optimizers import LBFGSScipy
from .stop_conditions import (
    ConvergencePatienceStopCondition,
    MinLossChangeStopCondition,
    StopCondition,
    StopConditionContext,
)
from .training_metrics import compute_kf_metric_tensor


class GPTrainerSingleProcess:
    """
   

    
    - Optimization uses only -mll(output, y) (no optimization_loss modes).
    - Trainer does not compute any training metrics (NIS/LOO/KF/etc.);
      those are delegated entirely to callbacks.
    - LBFGS inner-iteration metrics are produced via a generic hook that
      calls `on_lbfgs_iteration(context)` on callbacks, allowing them
      to decide which metrics to compute and how often.
    """

    def __init__(
        self,
        model,
        optimizer_class,
        optimizer_kwargs,
        num_epochs: int,
        mll_class: gpytorch.mlls.MarginalLogLikelihood = None,
        cholesky_jitter: float = 1e-6,
        callbacks: Optional[List[Callback]] = None,
        device: str | torch.device | None = None,
        scheduler_class: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        scheduler_kwargs: Optional[dict] = None,
        stop_conditions: Optional[List[StopCondition]] = None,
        min_epochs: int = 0,
        run_index: Optional[int] = None,
        # Per–L-BFGS-iteration logging (inner-iteration callback).
        # None = auto: enabled when any callback implements `on_lbfgs_iteration`.
        log_lbfgs_inner: bool | None = None,
        # Loss configuration: either "nll" (default) or "kf"
        loss_type: str = "nll",
        kf_Nf: Optional[int] = None,
        kf_max_n: int = 2000,
        kf_seed: Optional[int] = None,
    ):
        self.model = model
        self.run_index = run_index  # 1-based init index for display / logging
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.mll_class = mll_class
        self.num_epochs = num_epochs
        self.cholesky_jitter = cholesky_jitter
        self.min_epochs = min_epochs
        self.callbacks = callbacks or []
        self.device = device
        self.loss_type = loss_type
        self.kf_Nf = kf_Nf
        self.kf_max_n = kf_max_n
        self.kf_seed = kf_seed
        if log_lbfgs_inner is None:
            # Auto-enable when any callback wants LBFGS inner-iteration contexts.
            self.log_lbfgs_inner = any(
                hasattr(cb, "on_lbfgs_iteration") for cb in self.callbacks
            )
        else:
            self.log_lbfgs_inner = log_lbfgs_inner

        # Set default stop conditions if none provided
        if stop_conditions is None:
            self.stop_conditions = [
                ConvergencePatienceStopCondition(patience=20),
                MinLossChangeStopCondition(min_loss_change=1e-7),
            ]
        else:
            self.stop_conditions = stop_conditions

        # Get dtype from the model (which should be set from input data)
        if hasattr(model, "dtype") and model.dtype is not None:
            self.dtype = model.dtype
        else:
            self.dtype = torch.float64
            logger.warning("Model has no dtype attribute. Using %s as fallback.", self.dtype)

        # Move the model to device and convert to specified dtype
        self.model = self.model.to(self.device, dtype=self.dtype)

        # Update the model's internal training data to be on the same device and dtype
        self.model.set_train_data(
            self.model.train_inputs[0].to(self.device, dtype=self.dtype),
            self.model.train_targets.to(self.device, dtype=self.dtype),
            strict=False,
        )

        # Store training data for easy access
        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}

    def train(self):
        """
        Train the GP model with optional gradual jitter increase on NotPSDError.
        """
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None

        # Create mll instance
        mll = self.mll_class(self.model.likelihood, self.model)

        # Determine which training function to use based on optimizer type
        if isinstance(optimizer, torch.optim.LBFGS):
            train_epoch = self._train_lbfgs_epoch
        elif isinstance(optimizer, LBFGSScipy):
            train_epoch = self._train_scipy_lbfgs_epoch
        else:
            train_epoch = self._train_standard_epoch

        # Local variables for early stopping
        best_loss = float("inf")
        best_state_dict = None
        no_improvement_epochs = 0
        previous_loss = None

        # Accumulator for time spent in callback logging (report as log_time); mutable list so inner/iteration callbacks can add to it
        self._log_time_accumulator_ref = [0.0]

        # ---------------------------
        # on_train_start
        # ---------------------------
        ctx_start: CallbackOnTrainStartContext = {
            "model": self.model,
            "trainer": self,
            "device": str(self.device),
            # Pass the 1-based init index so callbacks can label results per-init.
            # Without this, FinalParameterStorageCallback in each loky worker only
            # sees its own _run_count (always 1), making every parallel init record
            # collapse onto run=1 in the trainer-analysis JSON.
            "run_index": self.run_index,
        }
        t0 = time.perf_counter()
        for cb in self.callbacks:
            cb.on_train_start(ctx_start)
        self._log_time_accumulator_ref[0] += time.perf_counter() - t0

        # Register callbacks with LBFGS optimizer so iteration_callback is invoked each L-BFGS iter (e.g. IterationParameterCallback)
        if isinstance(optimizer, LBFGSScipy):
            import inspect
            iteration_callbacks = []
            for cb in self.callbacks:
                if hasattr(cb, "register_with_optimizer"):
                    if hasattr(cb, "_on_iteration"):
                        iteration_callbacks.append(cb)
                    cb.register_with_optimizer(optimizer, model=self.model, trainer=self)
            if iteration_callbacks:
                def chained_iteration_callback(iteration, loss=None, flat_params=None):
                    t0 = time.perf_counter()
                    for icb in iteration_callbacks:
                        if hasattr(icb, "_on_iteration"):
                            try:
                                sig = inspect.signature(icb._on_iteration)
                                if "flat_params" in sig.parameters:
                                    icb._on_iteration(iteration, loss, flat_params=flat_params)
                                else:
                                    icb._on_iteration(iteration, loss)
                            except Exception as e:
                                import warnings
                                warnings.warn(f"Iteration callback {icb.__class__.__name__}: {e}")
                    if hasattr(self, "_log_time_accumulator_ref"):
                        self._log_time_accumulator_ref[0] += time.perf_counter() - t0
                optimizer.iteration_callback = chained_iteration_callback

        # Jitter can increase on NotPSDError; track current value for this run
        run_jitter = float(self.cholesky_jitter)
        max_jitter = 1e-3
        # Expose current jitter as a stable attribute for callbacks/metrics.
        # (Callbacks historically look for trainer.current_jitter.)
        self.current_jitter = run_jitter

        # Set the model to training mode
        self.model.train()

        logger.info("Starting training for %d epochs.", self.num_epochs)

        # Optional: per–L-BFGS-iteration logging (only used when optimizer is LBFGSScipy)
        if self.log_lbfgs_inner:
            self._lbfgs_inner_metrics: list[dict] = []
        if isinstance(optimizer, LBFGSScipy) and self.log_lbfgs_inner:
            self._best_lbfgs_loss = float("inf")
            self._best_lbfgs_iter: int | None = None
            optimizer._inner_callback = self._make_lbfgs_inner_callback(mll)  # type: ignore[attr-defined]
        else:
            self._best_lbfgs_iter = None

        for epoch in range(self.num_epochs):
            # ---------------------------
            # on_epoch_start
            # ---------------------------
            ctx_epoch_start: CallbackOnEpochStartContext = {
                "epoch": epoch,
                "model": self.model,
                "trainer": self,
                "device": str(self.device),
            }
            t0 = time.perf_counter()
            for cb in self.callbacks:
                cb.on_epoch_start(ctx_epoch_start)
            self._log_time_accumulator_ref[0] += time.perf_counter() - t0

            self._current_epoch = epoch  # for callbacks that care

            # Train for a single epoch with retry on NotPSDError (increase jitter)
            epoch_successful = False
            while not epoch_successful:
                self._current_run_jitter = run_jitter  # for callbacks to read
                self.current_jitter = run_jitter
                with (
                    gpytorch.settings.cholesky_jitter(run_jitter),
                    linear_operator.settings.cholesky_jitter(
                        float_value=run_jitter, double_value=run_jitter
                    ),
                ):
                    try:
                        loss = train_epoch(optimizer, mll)
                        epoch_successful = True
                    except NotPSDError:
                        if run_jitter < max_jitter:
                            run_jitter = min(run_jitter * 10.0, max_jitter)
                            self._current_run_jitter = run_jitter
                            self.current_jitter = run_jitter
                            logger.warning(
                                "NotPSDError detected. Increasing jitter to %.1e.", run_jitter
                            )
                        else:
                            logger.warning(
                                "NotPSDError persists with jitter=%.1e. Re-raising.", run_jitter
                            )
                            raise
                    except (RuntimeError, ValueError) as exc:
                        err_str = str(exc).lower()
                        if (
                            "notpsd" in err_str
                            or "not p.d." in err_str
                            or "not positive definite" in err_str
                        ):
                            if run_jitter < max_jitter:
                                run_jitter = min(run_jitter * 10.0, max_jitter)
                                self._current_run_jitter = run_jitter
                                self.current_jitter = run_jitter
                                logger.warning(
                                    "NotPSD error detected. Increasing jitter to %.1e.", run_jitter
                                )
                            else:
                                raise
                        else:
                            raise

            # ---------------------------
            # on_epoch_end
            # ---------------------------
            ctx_epoch_end: CallbackOnEpochEndContext = {
                "epoch": epoch,
                "model": self.model,
                "trainer": self,
                "loss": loss,
                "device": str(self.device),
                "jitter": run_jitter,
                "run_index": self.run_index,
            }
            # v3: no epoch-level metrics; callbacks may compute them if desired.
            t0 = time.perf_counter()
            for cb in self.callbacks:
                cb.on_epoch_end(ctx_epoch_end)
            self._log_time_accumulator_ref[0] += time.perf_counter() - t0

            # Update best-loss and best-state tracking
            if loss < best_loss:
                best_loss = loss
                best_state_dict = copy.deepcopy(self.model.state_dict())
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            # Check for early stopping conditions
            stop_context: StopConditionContext = {
                "epoch": epoch,
                "model": self.model,
                "trainer": self,
                "loss": loss,
                "previous_loss": previous_loss,
                "best_loss": best_loss,
                "no_improvement_epochs": no_improvement_epochs,
                "device": str(self.device),
            }

            early_stop_triggered = False
            early_stop_reasons: list[str] = []

            # Check stop conditions only after min_epochs (epoch is 0-indexed; epoch+1 = completed count)
            if (epoch + 1) >= self.min_epochs:
                for stop_condition in self.stop_conditions:
                    should_stop, reason = stop_condition.should_stop(stop_context)
                    if should_stop:
                        early_stop_triggered = True
                        if reason:
                            early_stop_reasons.append(reason)

            if early_stop_triggered:
                early_stop_reason = " OR ".join(early_stop_reasons) if early_stop_reasons else "Stop condition met"
                logger.info(
                    "Early stopping triggered at epoch %d. Reason: %s. Best loss: %.6f",
                    epoch + 1,
                    early_stop_reason,
                    best_loss,
                )
                break  # Stop training

            # Update previous_loss for next iteration
            previous_loss = loss

        # Log training completion
        logger.info("Training completed. Best loss: %.6f", best_loss)
        logger.info("Total epochs trained: %d", epoch + 1)

        # Persist jitter on model (may have been increased on NotPSDError). Kernel and callbacks read from model.
        self.model.cholesky_jitter = run_jitter

        # ---------------------------
        # on_train_end
        # ---------------------------
        ctx_end: CallbackOnTrainEndContext = {
            "epoch": epoch,
            "model": self.model,
            "trainer": self,
            "best_loss": best_loss,
            "best_state_dict": best_state_dict,
            "device": str(self.device),
            "run_index": self.run_index,
        }
        best_lbfgs_iter = getattr(self, "_best_lbfgs_iter", None)
        if best_lbfgs_iter is not None:
            ctx_end["best_lbfgs_iter"] = best_lbfgs_iter
        lbfgs_stop_reason = getattr(self, "_lbfgs_stop_reason", None)
        if lbfgs_stop_reason is not None:
            ctx_end["lbfgs_stop_reason"] = lbfgs_stop_reason
        jitter_max = getattr(self, "jitter_max", None)
        if jitter_max is not None:
            ctx_end["jitter_max"] = jitter_max
        t0 = time.perf_counter()
        for cb in self.callbacks:
            cb.on_train_end(ctx_end)
        self._log_time_accumulator_ref[0] += time.perf_counter() - t0

        # Collect callback data
        t0 = time.perf_counter()
        callback_data: dict = {}
        for cb in self.callbacks:
            if hasattr(cb, "get_stored_parameters"):
                cb_name = cb.__class__.__name__
                stored_params = cb.get_stored_parameters()
                if stored_params:  # Only add if there's data
                    callback_data[cb_name] = stored_params
        self._log_time_accumulator_ref[0] += time.perf_counter() - t0

        log_time_accumulator = self._log_time_accumulator_ref[0]

        out = {
            "loss": best_loss,
            "state_dict": best_state_dict,
            "callback_data": callback_data,
            "cholesky_jitter": run_jitter,
            "log_time": log_time_accumulator,
        }
        if self.log_lbfgs_inner and hasattr(self, "_lbfgs_inner_metrics"):
            out["lbfgs_inner_metrics"] = self._lbfgs_inner_metrics
        best_lbfgs_iter = getattr(self, "_best_lbfgs_iter", None)
        if best_lbfgs_iter is not None:
            out["best_lbfgs_iter"] = best_lbfgs_iter
        return out

    def _train_standard_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch with standard optimizers.
        """
        optimizer.zero_grad()
        train_x = self.train_x.to(dtype=self.dtype)
        train_y = self.train_y.to(dtype=self.dtype)
        loss = self._compute_loss(mll, train_x, train_y)
        loss.backward()
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss.item())

    def _train_lbfgs_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch using torch.optim.LBFGS.
        """
        closure = self._lbfgs_closure(optimizer, mll)
        loss = optimizer.step(closure)
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss.item())

    def _train_scipy_lbfgs_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch using LBFGSScipy.
        """
        closure = self._lbfgs_closure(optimizer, mll)
        optimizer.step(closure)
        loss = optimizer._last_loss  # type: ignore[attr-defined]
        self._lbfgs_stop_reason = getattr(optimizer, "_lbfgs_stop_reason", None)
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss)

    def _make_lbfgs_inner_callback(self, mll):
        """
        Build a callback invoked each L-BFGS inner iteration for logging.

        v3: the trainer itself does not compute metrics like NLL/NIS/LOO/KF here.
        Instead it builds a generic context and lets callbacks implement
        `on_lbfgs_iteration(ctx)` to add whatever metrics they want, at whatever
        iteration frequency they choose.
        """

        def on_lbfgs_iter(lbfgs_iter: int, loss_tensor):
            run_jitter = getattr(self, "_current_run_jitter", self.cholesky_jitter)
            loss_val = loss_tensor.item() if hasattr(loss_tensor, "item") else float(loss_tensor)

            # LBFGS: no epoch (single effective epoch); record iter and loss only.
            record: dict = {"lbfgs_iter": lbfgs_iter, "loss": loss_val}

            # Build context for callbacks to add metrics / extra fields.
            ctx = {
                "lbfgs_iter": lbfgs_iter,
                "loss": loss_val,
                "model": self.model,
                "trainer": self,
                "device": str(self.device),
                "run_jitter": run_jitter,
                "mll": mll,
                "record": record,
            }
            t0 = time.perf_counter()
            for cb in self.callbacks:
                handler = getattr(cb, "on_lbfgs_iteration", None)
                if callable(handler):
                    try:
                        handler(ctx)
                    except Exception as exc:  # noqa: BLE001
                        if getattr(cb, "verbose", False):
                            logger.warning(
                                "%s: error in on_lbfgs_iteration at iter=%d: %s",
                                cb.__class__.__name__,
                                lbfgs_iter,
                                exc,
                            )
            if hasattr(self, "_log_time_accumulator_ref"):
                self._log_time_accumulator_ref[0] += time.perf_counter() - t0

            # Append record unless a callback set _skip_append (e.g. LBFGSInnerMetricsCallbackV3 log_record_every_n_iters)
            if not record.pop("_skip_append", False):
                self._lbfgs_inner_metrics.append(record)

            # Track which L-BFGS iteration had the best loss (for best_iter in output)
            if loss_val < getattr(self, "_best_lbfgs_loss", float("inf")):
                self._best_lbfgs_loss = loss_val
                self._best_lbfgs_iter = lbfgs_iter

            # Compact log line using whatever metrics callbacks attached.
            msg_parts = [
                f"iter={lbfgs_iter}",
                f"loss={loss_val:.6f}",
            ]
            known = ("NLL", "NIS", "LOO_NLL", "KF", "MSE", "R2")
            for key in known:
                if key in record:
                    msg_parts.append(f"{key}={record[key]:.6f}")
            for key, val in record.items():
                if key not in ("lbfgs_iter", "loss") and key not in known:
                    try:
                        v = float(val.item() if hasattr(val, "item") else val)
                        if not (v != v):  # skip nan
                            msg_parts.append(f"{key}={v:.6f}")
                    except (TypeError, ValueError):
                        pass
            logger.info("L-BFGS inner %s", " ".join(msg_parts))

        return on_lbfgs_iter

    def _lbfgs_closure(self, optimizer, mll):
        """
        Defines the closure for LBFGS-style optimizers.
        """

        def closure():
            optimizer.zero_grad()

            model_device = next(self.model.parameters()).device

            train_x = self.train_x.to(dtype=self.dtype, device=model_device)
            train_y = self.train_y.to(dtype=self.dtype, device=model_device)

            loss = self._compute_loss(mll, train_x, train_y)
            loss.backward()

            return loss

        return closure


    def _compute_loss(self, mll, train_x: torch.Tensor, train_y: torch.Tensor) -> torch.Tensor:
        """
        Compute the training loss based on the configured loss_type.

        - "nll": standard negative marginal log likelihood.
        - "loo": negative leave-one-out pseudo log likelihood (minimize = maximize LOO predictive accuracy).
        - "kf": kernel flows rho metric used directly as a loss (minimize rho).
        - "nll_kf": composite loss = NLL + lambda_kf * rho.
        """
        output = self.model(train_x)
        nll = -mll(output, train_y)  # for fallbacks and nll_kf

        if self.loss_type == "nll":
            return nll

        if self.loss_type == "loo":
            # LOO loss: minimize negative LOO pseudo log likelihood (differentiable).
            loo_mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(self.model.likelihood, self.model)
            pll = loo_mll(output, train_y)
            loss = -pll.sum() if pll.dim() > 0 else -pll
            return loss

        if self.loss_type == "kf":
            # Pure KF loss path: compute differentiable rho tensor.
            N = train_x.size(0)
            if self.kf_max_n is not None and N > self.kf_max_n:
                nll = -mll(output, train_y)
                return nll

            Nf = self.kf_Nf if self.kf_Nf is not None else min(64, N)
            rho = compute_kf_metric_tensor(
                self.model,
                train_x,
                train_y,
                Nf=Nf,
                cholesky_jitter=self.cholesky_jitter,
                seed=self.kf_seed,
                likelihood=self.model.likelihood,
            )
            return rho if torch.isfinite(rho) else nll

        if self.loss_type == "nll_kf":
            # Composite: NLL + lambda_kf * rho regularizer.
            N = train_x.size(0)
            lambda_kf = 5  # start small; tune per problem
            rho = torch.tensor(0.0, device=train_x.device, dtype=train_x.dtype)
            if self.kf_max_n is None or N <= self.kf_max_n:
                Nf = self.kf_Nf if self.kf_Nf is not None else min(64, N)
                rho_val = compute_kf_metric_tensor(
                    self.model,
                    train_x,
                    train_y,
                    Nf=Nf,
                    cholesky_jitter=self.cholesky_jitter,
                    seed=self.kf_seed,
                    likelihood=self.model.likelihood,
                )
                if torch.isfinite(rho_val):
                    rho = rho_val
            return -mll(output, train_y) + lambda_kf * rho
        


__all__ = ["GPTrainerSingleProcess"]