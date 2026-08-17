from functools import reduce

import numpy as np
import torch
from scipy.optimize import fmin_l_bfgs_b


class LBFGSScipy(torch.optim.Optimizer):
    """Wrap L-BFGS algorithm, using scipy routines.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now CPU only

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """

    def __init__(
        self, params, max_iter=2000, max_eval=5000, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10, iteration_callback=None
    ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
        )
        super(LBFGSScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self._n_iter = 0
        self._last_loss = None
        self.iteration_callback = iteration_callback

        # Numerical epsilon for scipy
        self.eps = np.finfo("double").eps

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset : offset + numel].view_as(p.data)
            offset += numel


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        group = self.param_groups[0]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        history_size = group["history_size"]

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params).to(self._params[0].device)
            self._distribute_flat_params(flat_params)
            loss = closure()
            self._last_loss = loss
            loss_value = loss.item()
            flat_grad = self._gather_flat_grad().cpu().numpy()
            return loss_value, flat_grad

        def callback(flat_params):
            self._n_iter += 1
            # Optional: print progress (can be disabled)
            # print('Iter %i Loss %.5f' % (self._n_iter, self._last_loss.item()))
            
            # Update model parameters to current candidate values from scipy
            # This allows iteration callbacks to extract the current parameter state
            # The parameters will be updated again at the end of the step with the final result
            flat_params_tensor = torch.from_numpy(flat_params).to(self._params[0].device)
            self._distribute_flat_params(flat_params_tensor)
            
            # Call external iteration callback if provided
            if self.iteration_callback is not None:
                try:
                    # Pass flat_params to the callback so it can track changes
                    self.iteration_callback(iteration=self._n_iter, loss=self._last_loss.item() if self._last_loss is not None else None, flat_params=flat_params_tensor)
                except Exception as e:
                    # Don't let callback errors break optimization
                    import warnings
                    warnings.warn(f"Iteration callback raised an error: {e}")
            # Optional: invoke trainer inner callback for per-iteration logging (e.g. NLL, NIS)
            inner_cb = getattr(self, "_inner_callback", None)
            if inner_cb is not None and self._last_loss is not None:
                inner_cb(self._n_iter, self._last_loss)

        initial_params = self._gather_flat_params().cpu().numpy()

        # Run scipy L-BFGS-B optimization
        result = fmin_l_bfgs_b(
            wrapped_closure,
            initial_params,
            maxiter=max_iter,
            maxfun=max_eval,
            factr=tolerance_change / self.eps,
            pgtol=tolerance_grad,
            epsilon=0,
            m=history_size,
            callback=callback,
        )

        # Store stop reason for on_train_end logging (result is (x, f, d) with d = info dict)
        info = result[2] if len(result) > 2 else {}
        self._lbfgs_info = info
        task = info.get("task", b"")
        if isinstance(task, bytes):
            task = task.decode("ascii", errors="replace")
        else:
            task = str(task)
        warnflag = info.get("warnflag", -1)
        if warnflag == 0:
            self._lbfgs_stop_reason = task or "CONVERGED"
        elif warnflag == 1:
            self._lbfgs_stop_reason = task or "TOO_MANY_FEVALS"
        elif warnflag == 2:
            self._lbfgs_stop_reason = task or "ABNORMAL_TERMINATION"
        else:
            self._lbfgs_stop_reason = task or "UNKNOWN"

        # Update parameters with final result.
        # IMPORTANT: scipy returns (x, f, d) where x is the recorded optimum and
        # f = func(x). Without resetting _last_loss here, downstream code may
        # report the loss from the LAST line-search probe (which can sit at a
        # slightly different point than result[0]) while the saved state_dict
        # comes from result[0]. That mismatch is what caused per-init
        # `final_parameters` to disagree across series and parallel trainers.
        target_device = self._params[0].device
        final_params = torch.from_numpy(result[0]).to(target_device)
        self._distribute_flat_params(final_params)
        try:
            final_loss_value = float(result[1])
            ref_dtype = self._params[0].dtype if self._params else torch.float64
            self._last_loss = torch.tensor(
                final_loss_value, device=target_device, dtype=ref_dtype
            )
        except (TypeError, ValueError):
            # Fall back to whatever closure recorded last; should never happen with scipy.
            pass

        return self._last_loss
