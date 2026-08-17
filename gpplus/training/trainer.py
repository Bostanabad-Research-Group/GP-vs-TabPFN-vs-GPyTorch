import copy
import os
import time
from contextlib import nullcontext
from typing import List, Optional

import gpytorch
import torch
from joblib import Parallel, delayed, parallel_config

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    threadpool_info = None
    threadpool_limits = None

from ..config import get_settings, logger
from .callbacks import Callback
from .optimizers import LBFGSScipy
from .parameter_initializer import DefaultParameterInitializer, ParameterInitializer
from .stop_conditions import StopCondition
from .training_single_run import GPTrainerSingleProcess


class GPTrainer:
    """
    GPTrainer (v3) combines the v2 trainer's richer training metrics and
    optimization-loss options with the original trainer's aggregation of
    callback outputs across runs (e.g., jitter/parameter tracking).

    Parameters mirror v2 with additional internal aggregation after training.
    """

    def __init__(
        self,
        model,
        optimizer_class: torch.optim.Optimizer = None,
        optimizer_kwargs: dict = None,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler = None,
        scheduler_kwargs: dict = None,
        num_epochs: int = 50,
        seed: int = None,
        num_inits: int = 64,
        mll_class: gpytorch.mlls.MarginalLogLikelihood = None,
        cholesky_jitter: float = 1e-6,
        callbacks: Optional[List[Callback]] = None,
        initializer_class: ParameterInitializer = None,
        initializer_kwargs: dict = None,
        device: str = "cpu",
        stop_conditions: Optional[List[StopCondition]] = None,
        min_epochs: int = 0,
        # Per–L-BFGS-iteration logging (inner-iteration callback).
        # None = auto: enabled when any callback implements `on_lbfgs_iteration`.
        log_lbfgs_inner: bool | None = None,
        # If set, build trainer analysis (inits, best_parameters, average_final_parameters) and save to this path after train()
        trainer_analysis_save_path: Optional[str] = None,
        # Number of joblib workers to use for multi-init training. None defaults to
        # max(1, cpu_count - 2) on CPU and num_gpus on CUDA. Pass 1 to force series.
        n_jobs: Optional[int] = None,
        # BLAS thread cap per worker (also applied to the main process when n_jobs == 1).
        # Default 1 keeps numerics consistent between series and parallel paths.
        inner_max_num_threads: Optional[int] = 1,
    ):
        self.trainer_analysis_save_path = trainer_analysis_save_path
        self.n_jobs = n_jobs
        self.inner_max_num_threads = inner_max_num_threads
        # -------------------------------------------------------
        # Set up the device (CPU or CUDA)
        # -------------------------------------------------------
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Keep the original model on CPU
        self.model = model  # no .to(self.device) here
        logger.info("Model stays on CPU in the constructor.")

        # --------------------------------------------------
        #  CORE CONFIG
        # --------------------------------------------------
        self.num_epochs = num_epochs
        self.num_inits = num_inits
        self.seed = seed
        self.callbacks = callbacks or []
        self.cholesky_jitter = cholesky_jitter
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs
        self.min_epochs = min_epochs
        self.loss_type = "nll"
        self.kf_Nf: Optional[int] = None
        self.kf_max_n: int = 2000
        self.kf_seed: Optional[int] = None

        # Set default stop conditions if none provided
        if stop_conditions is None:
            from .stop_conditions import ConvergencePatienceStopCondition, MinLossChangeStopCondition

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
            logger.warning(f"Model has no dtype attribute. Using {self.dtype} as fallback.")

        # Set up the initializer; use a default one if none is provided.
        if initializer_class is None:
            self.initializer = DefaultParameterInitializer(num_runs=self.num_inits, seed=self.seed)
        else:
            if initializer_kwargs is None:
                initializer_kwargs = {}
            self.initializer = initializer_class(num_runs=self.num_inits, seed=self.seed, **initializer_kwargs)

        # Precompute number of parameters and Sobol samples.
        self.initializer.setup(model)

        # --------------------------------------------------
        #  OPTIMIZER
        # --------------------------------------------------
        # Handle optimizer class, use LBFGS as default
        if optimizer_class is None:
            self.optimizer_class = LBFGSScipy
            logger.warning("No optimizer class passed. Defaulting to LBFGS Scipy optimizer.")
        else:
            self.optimizer_class = optimizer_class

        # Handle optimizer arguments: set defaults per optimizer type when not provided
        if optimizer_kwargs is not None:
            self.optimizer_kwargs = optimizer_kwargs
        else:
            opt_cls = self.optimizer_class
            if opt_cls is LBFGSScipy or (
                hasattr(opt_cls, "__name__") and opt_cls.__name__ == "LBFGSScipy"
            ):
                self.optimizer_kwargs = {
                    "max_iter": 2000,
                    "max_eval": 5000,
                    "tolerance_grad": 1e-5,
                    "tolerance_change": 1e-9,
                    "history_size": 10,
                }
            elif opt_cls is torch.optim.Adam or (
                isinstance(opt_cls, type) and issubclass(opt_cls, torch.optim.Adam)
            ):
                self.optimizer_kwargs = {"lr": 0.01}
            else:
                self.optimizer_kwargs = {"max_iter": 20}
                logger.warning(
                    "No optimizer arguments passed and no built-in defaults for "
                    f"{getattr(opt_cls, '__name__', opt_cls)}. Using max_iter=20."
                )

        # Handle MLL class
        if mll_class is None:
            self.mll_class = gpytorch.mlls.ExactMarginalLogLikelihood
            logger.warning("No MLL class passed. Defaulting to ExactMarginalLogLikelihood.")
        else:
            self.mll_class = mll_class

        self.log_lbfgs_inner = log_lbfgs_inner

    def train_single_process(self, run_index):
        """
        Train for a single initialization (run_index) using GPTrainerSingleProcess v3.
        """
        base_model = copy.deepcopy(self.model)
        # v3: always use the standard initializer; prescreening is disabled here.
        self.initializer.initialize(base_model, run_index)

        base_model = base_model.to(self.device)

        callbacks_copy = [copy.deepcopy(cb) for cb in self.callbacks] if self.callbacks else []
        stop_conditions_copy = [copy.deepcopy(sc) for sc in self.stop_conditions] if self.stop_conditions else None

        # For L-BFGS-style optimizers, we only run a single epoch and rely on
        # the optimizer's inner iterations (with per-iteration metrics).
        opt_cls = self.optimizer_class
        is_lbfgs_like = (
            opt_cls is LBFGSScipy
            or (isinstance(opt_cls, type) and issubclass(opt_cls, LBFGSScipy))
            or opt_cls is torch.optim.LBFGS
            or (isinstance(opt_cls, type) and issubclass(opt_cls, torch.optim.LBFGS))
        )
        num_epochs_for_run = 1 if is_lbfgs_like else self.num_epochs
        min_epochs_for_run = 1 if is_lbfgs_like else self.min_epochs

        run = GPTrainerSingleProcess(
            model=base_model,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            mll_class=self.mll_class,
            num_epochs=num_epochs_for_run,
            cholesky_jitter=self.cholesky_jitter,
            callbacks=callbacks_copy,
            device=self.device,
            scheduler_class=self.scheduler_class,
            scheduler_kwargs=self.scheduler_kwargs,
            stop_conditions=stop_conditions_copy,
            min_epochs=min_epochs_for_run,
            run_index=run_index + 1,
            log_lbfgs_inner=self.log_lbfgs_inner,
            loss_type=self.loss_type,
            kf_Nf=self.kf_Nf,
            kf_max_n=self.kf_max_n,
            kf_seed=self.kf_seed,
        )
        train_result = run.train()

        with torch.no_grad():
            for (_, param), (_, trained_param) in zip(
                self.model.named_parameters(), base_model.named_parameters()
            ):
                if param.requires_grad:
                    param.data.copy_(trained_param.data.to(dtype=param.dtype))

        callback_data = {}
        for cb in callbacks_copy:
            if hasattr(cb, "get_stored_parameters"):
                cb_name = cb.__class__.__name__
                stored_params = cb.get_stored_parameters()
                if stored_params:
                    callback_data[cb_name] = stored_params

        if "callback_data" in train_result:
            for key, value in train_result["callback_data"].items():
                if key not in callback_data:
                    callback_data[key] = value
        train_result["callback_data"] = callback_data

        return {"init_index": run_index, **train_result}

    def train_multiple_process_parallel(self, init_indices=None):
        """
        Train the model in parallel using different initializations.
        """
        if init_indices is None:
            init_indices = list(range(self.num_inits))

        num_inits_to_train = len(init_indices)



        def safe_single_process(init_index, device_override=None, tag="worker"):
            try:
                original_device = self.device
                if device_override is not None:
                    self.device = device_override
                _worker_init()
                result = self.train_single_process(init_index)
                self.device = original_device
                return result
            except Exception as e:
                logger.exception(f"Error in training init #{init_index}: {e}")
                return {
                    "init_index": init_index,
                    "state_dict": None,
                    "loss": None,
                    "error": str(e),
                }

        def _worker_init():
            get_settings().apply()
            # Reseed PyTorch's global RNG inside the worker. loky spawns fresh
            # processes whose default RNG state diverges from the main process,
            # which can affect any init paths that fall back to the global RNG.
            try:
                seed_val = self.seed if self.seed is not None else 0
                torch.manual_seed(int(seed_val))
            except Exception as exc:  # noqa: BLE001
                logger.warning("torch.manual_seed failed in worker: %s", exc)
            # Force deterministic CPU kernels where available; warn-only avoids
            # crashes on ops that lack a deterministic implementation.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("torch.use_deterministic_algorithms failed: %s", exc)

        # Apply deterministic settings to the MAIN process unconditionally so that
        # series mode (where _worker_init runs in main) and parallel mode (where
        # _worker_init runs only inside loky workers) leave the main process in
        # the same state. Without this, parallel left main with
        # torch.use_deterministic_algorithms=False while series left it True, which
        # selected different FP code paths in the post-train() eval phase and
        # produced very different per-init test RRMSE for ill-conditioned models.
        try:
            seed_val = self.seed if self.seed is not None else 0
            torch.manual_seed(int(seed_val))
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.manual_seed failed in main process: %s", exc)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.use_deterministic_algorithms failed in main: %s", exc)

        # Resolve effective n_jobs: explicit override wins, otherwise legacy default.
        if self.device.type == "cpu":
            requested_jobs = self.n_jobs if self.n_jobs is not None else max(1, (os.cpu_count() or 1) - 2)
            max_jobs = min(num_inits_to_train, max(1, requested_jobs))
            # max_jobs = 1
            inner = self.inner_max_num_threads

            # Pin BLAS in the main process when running series so its FP rounding
            # matches the per-worker pinning that joblib applies in the parallel path.
            blas_ctx = (
                threadpool_limits(limits=inner)
                if (max_jobs == 1 and threadpool_limits is not None and inner is not None)
                else nullcontext()
            )

            if max_jobs == 1:
                logger.info(
                    "Running %d inits in series (n_jobs=1) with BLAS threads pinned to %s.",
                    num_inits_to_train,
                    inner if inner is not None else "default",
                )
                with blas_ctx:
                    results = [
                        safe_single_process(init_index, tag="series_main")
                        for init_index in init_indices
                    ]
            else:
                logger.info(
                    "Running %d inits using %d parallel processes on CPU "
                    "(joblib 'loky', inner_max_num_threads=%s).",
                    num_inits_to_train,
                    max_jobs,
                    inner if inner is not None else "default",
                )
                # joblib exposes per-backend BLAS thread caps via parallel_config,
                # not directly on Parallel(...). Setting it here ensures each loky
                # worker gets the requested OMP/MKL/OPENBLAS thread count.
                pc_kwargs = {}
                if inner is not None:
                    pc_kwargs["inner_max_num_threads"] = inner
                with parallel_config(backend="loky", **pc_kwargs):
                    results = Parallel(n_jobs=max_jobs, verbose=0)(
                        delayed(safe_single_process)(init_index, tag="loky_worker")
                        for init_index in init_indices
                    )
        elif str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
            num_gpus = torch.cuda.device_count()
            requested_jobs = self.n_jobs if self.n_jobs is not None else (num_gpus if num_gpus > 0 else 1)
            max_jobs = min(num_inits_to_train, max(1, requested_jobs))
            logger.info(
                f"Running {num_inits_to_train} inits distributed across {num_gpus} GPUs "
                f"(using joblib 'threading' backend with verbose=0; n_jobs={max_jobs})."
            )
            results = Parallel(n_jobs=max_jobs, backend="threading", verbose=0)(
                delayed(safe_single_process)(
                    init_index,
                    device_override=torch.device(f"cuda:{i % max(1, num_gpus)}"),
                    tag="cuda_worker",
                )
                for i, init_index in enumerate(init_indices)
            )
        else:
            results = [safe_single_process(init_index, tag="local") for init_index in init_indices]

        return results

    def _aggregate_jitter_callbacks(self, results):
        """Aggregate jitter tracking data from all runs into a single file."""
        from .callbacks import JitterTrackingCallback

        jitter_callbacks = [cb for cb in (self.callbacks or []) if isinstance(cb, JitterTrackingCallback)]

        for jitter_cb in jitter_callbacks:
            if jitter_cb.save_file is not None:
                JitterTrackingCallback.aggregate_jitter_from_results(
                    results=results,
                    save_file=jitter_cb.save_file,
                    verbose=jitter_cb.verbose,
                )

    def _aggregate_parameter_callbacks(self, results):
        """Aggregate parameter tracking data from all runs into a single file."""
        from .callbacks import IterationParameterCallback, EpochParameterCallback

        iteration_callbacks = [cb for cb in (self.callbacks or []) if isinstance(cb, IterationParameterCallback)]
        for iter_cb in iteration_callbacks:
            if iter_cb.save_file is not None:
                IterationParameterCallback.aggregate_parameters_from_results(
                    results=results,
                    save_file=iter_cb.save_file,
                    verbose=iter_cb.verbose,
                )

        epoch_callbacks = [cb for cb in (self.callbacks or []) if isinstance(cb, EpochParameterCallback)]
        for epoch_cb in epoch_callbacks:
            if epoch_cb.save_file is not None:
                EpochParameterCallback.aggregate_parameters_from_results(
                    results=results,
                    save_file=epoch_cb.save_file,
                    verbose=epoch_cb.verbose,
                )

    def train(self):
        # ------------------------------------------------------
        #  TRAINING (no prescreening in v3)
        # ------------------------------------------------------
        t_full_start = time.perf_counter()
        results = self.train_multiple_process_parallel(init_indices=None)
        full_train_time = time.perf_counter() - t_full_start

        # Time breakdown: inits run in parallel so use max(log_time) across inits (wall-clock), not sum
        log_times = [float(r.get("log_time", 0)) for r in results if r.get("log_time") is not None]
        log_time = max(log_times) if log_times else 0.0
        train_time = max(0.0, full_train_time - log_time)
        self.full_train_time = full_train_time
        self.log_time = log_time
        self.train_time = train_time

        logger.info("Training completed.")
        logger.info(
            "Timing summary: full_train_time=%.3fs | train_time=%.3fs | log_time=%.3fs",
            full_train_time, train_time, log_time,
        )

        # Aggregate jitter and parameter tracking across runs (from original trainer)
        self._aggregate_jitter_callbacks(results)
        self._aggregate_parameter_callbacks(results)

        # ------------------------------------------------------
        #  Select the best run by comparing the 'loss' values
        # ------------------------------------------------------
        best_run = None
        best_loss = float("inf")

        for run_result in results:
            if (
                run_result.get("loss") is not None
                and run_result["loss"] < best_loss
                and run_result.get("state_dict") is not None
            ):
                best_loss = run_result["loss"]
                best_run = run_result

        # ------------------------------------------------------
        #  If a valid best run was found, load it into self.model
        # ------------------------------------------------------
        if best_run is not None and best_run.get("state_dict") is not None:
            state = best_run["state_dict"]
            # Load best weights onto the trainer's device so that model parameters,
            # training inputs, and likelihood live on a single device (CPU or CUDA).
            target_device = self.device
            state_on_device = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in state.items()}
            self.model.load_state_dict(state_on_device)
            # Ensure the whole model (including train_inputs / train_targets buffers) is on target_device.
            self.model = self.model.to(target_device)

            # Propagate best-run jitter onto the model for evaluation
            jitter = best_run.get("cholesky_jitter")
            if jitter is not None:
                self.model.cholesky_jitter = jitter

            logger.info(
                f"Best init found: #{best_run.get('init_index', 'N/A')} with loss={best_loss:.4f}. "
                "Original model state_dict updated with best weights."
            )

        else:
            logger.warning("No valid best run found. Model was not updated.")

        # Build and save trainer analysis (inits, best_parameters, average_final_parameters) when path set
        if getattr(self, "trainer_analysis_save_path", None):
            from .trainer_analysis import build_trainer_analysis_from_results, save_trainer_analysis
            payload = build_trainer_analysis_from_results(results, self.num_epochs)
            if payload is not None:
                payload["full_train_time"] = getattr(self, "full_train_time", None)
                payload["train_time"] = getattr(self, "train_time", None)
                payload["log_time"] = getattr(self, "log_time", None)
                try:
                    save_trainer_analysis(payload, self.trainer_analysis_save_path)
                    logger.info("Saved trainer analysis to %s", self.trainer_analysis_save_path)
                except Exception as e:
                    logger.warning("Could not save trainer analysis to %s: %s", self.trainer_analysis_save_path, e)

        return results

