"""
Initialization Pre-screener for GP Models

This module provides a class for pre-screening initializations by evaluating
the loss function for a large number of Sobol sequence samples and selecting
the best ones for optimization.
"""

import copy
import os
from typing import List, Optional

import gpytorch
import linear_operator
import numpy as np
import torch
from joblib import Parallel, delayed

from ..config import logger


class InitializationPrescreener:
    """
    Pre-screens initializations by evaluating loss for N_test Sobol samples
    and selecting the best N_initialization samples for optimization.
    
    This is useful when you want to generate a large number of initial guesses
    (N_test = N_initialization * D_x) and select the best ones before starting
    full optimization.
    
    The pre-screener uses a short "warmup" training step before evaluating loss.
    This makes the loss evaluation more meaningful, especially for models with
    neural network components where randomly initialized weights don't give
    meaningful loss values. The warmup allows the model to make a few optimization
    steps, making the loss a better predictor of final performance.
    
    Args:
        model: The GP model to pre-screen
        initializer: Parameter initializer instance (already set up with num_runs=N_test)
        num_test: Total number of initializations to evaluate
        num_runs: Number of best initializations to select
        mll_class: Marginal log likelihood class for loss evaluation
        cholesky_jitter: Jitter value for numerical stability
        device: Device to run on ("cpu" or "cuda")
        dtype: Data type for computations
        num_warmup_epochs: Number of training epochs before loss evaluation (default: 3)
        warmup_lr: Learning rate for warmup training (default: 0.01)
        optimizer_class: Optimizer class for warmup (default: torch.optim.Adam)
    """
    
    def __init__(
        self,
        model,
        initializer,
        num_test: int,
        num_runs: int,
        mll_class,
        cholesky_jitter: float = 1e-6,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        num_warmup_epochs: int = 3,
        warmup_lr: float = 0.01,
        optimizer_class=None,
        recorder=None,
    ):
        self.model = model
        self.initializer = initializer
        self.num_test = num_test
        self.num_runs = num_runs
        self.mll_class = mll_class
        self.cholesky_jitter = cholesky_jitter
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.num_warmup_epochs = num_warmup_epochs
        self.warmup_lr = warmup_lr
        # Use Adam for warmup by default (works well for neural networks)
        self.optimizer_class = optimizer_class if optimizer_class is not None else torch.optim.Adam
        self.recorder = recorder
    
    def evaluate_initial_loss(self, run_index: int) -> dict:
        """
        Evaluate the loss for a given initialization after a short warmup training.
        This makes the loss more meaningful, especially for models with neural networks.
        
        Args:
            run_index: Index of the initialization to evaluate
            
        Returns:
            dict: Dictionary with 'run_index', 'loss' (after warmup), and 'initial_loss' values
        """
        # Copy the model (which is on CPU)
        # Note: deepcopy is necessary to avoid modifying the original model
        base_model = copy.deepcopy(self.model)

        # Initialize parameters for the model copy on CPU using the initializer
        self.initializer.initialize(base_model, run_index)
        
        # Extract Sobol sample and initial parameters for recording
        # (Convert to CPU and detach for serialization in parallel processing)
        sobol_sample = None
        initial_parameters = None
        if self.recorder is not None:
            # Try to get Sobol sample from initializer
            if hasattr(self.initializer, 'sobol_samples') and self.initializer.sobol_samples is not None:
                if run_index < len(self.initializer.sobol_samples):
                    sobol_sample = self.initializer.sobol_samples[run_index].detach().cpu() if isinstance(self.initializer.sobol_samples[run_index], torch.Tensor) else self.initializer.sobol_samples[run_index]
            
            # Extract initial parameters (after initialization, before warmup)
            # Convert to CPU and detach for serialization
            initial_parameters = {}
            for name, param in base_model.named_parameters():
                if param.requires_grad:
                    initial_parameters[name] = param.data.detach().cpu().clone()

        # Move model_copy to device
        base_model = base_model.to(self.device, dtype=self.dtype)
        
        # Get training data (already stored in model)
        # Update stored training data to correct device/dtype
        # Check if train_inputs and train_targets exist
        if base_model.train_inputs is None or len(base_model.train_inputs) == 0:
            raise ValueError(f"Model has no training inputs after deepcopy for run_index {run_index}")
        if base_model.train_targets is None:
            raise ValueError(f"Model has no training targets after deepcopy for run_index {run_index}")
        
        train_x = base_model.train_inputs[0].to(self.device, dtype=self.dtype)
        train_y = base_model.train_targets.to(self.device, dtype=self.dtype)
        
        base_model.set_train_data(train_x, train_y, strict=False)
        
        # Create mll instance
        mll = self.mll_class(base_model.likelihood, base_model)
        
        # Evaluate initial loss (before warmup)
        # Use model() without arguments to avoid GPInputWarning when using stored training data
        initial_loss_value = float("inf")
        try:
            with torch.no_grad():
                with (
                    gpytorch.settings.cholesky_jitter(self.cholesky_jitter),
                    linear_operator.settings.cholesky_jitter(
                        float_value=self.cholesky_jitter, double_value=self.cholesky_jitter
                    ),
                ):
                    base_model.eval()
                    # Call model() without arguments to use stored training data (avoids warning)
                    # If that fails, try with explicit training data
                    try:
                        output = base_model()
                    except Exception as e:
                        if run_index == 0:
                            logger.warning(f"base_model() failed, trying with explicit train_x: {e}")
                        # Fallback: pass training data explicitly
                        output = base_model(train_x)
                    initial_loss = -mll(output, train_y)
                    initial_loss_value = initial_loss.item()
                    
                    # Debug: Log first few losses to verify they're being computed
                    if run_index < 3:
                        logger.info(f"DEBUG run_index {run_index}: initial_loss_value = {initial_loss_value}")
                    
                    if not torch.isfinite(torch.tensor(initial_loss_value)):
                        if run_index == 0:  # Only log for first run to avoid spam
                            logger.warning(f"Non-finite initial loss for run_index {run_index}: {initial_loss_value}")
                        initial_loss_value = float("inf")
        except Exception as e:
            if run_index == 0:  # Only log for first run to avoid spam
                logger.warning(f"Initial loss evaluation failed for run_index {run_index}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            initial_loss_value = float("inf")
        
        # Do a short warmup training to make loss evaluation more meaningful
        # This is especially important for models with neural network components
        loss_after_warmup = float("inf")
        if self.num_warmup_epochs > 0:
            try:
                base_model.train()
                optimizer = self.optimizer_class(base_model.parameters(), lr=self.warmup_lr)
                
                with (
                    gpytorch.settings.cholesky_jitter(self.cholesky_jitter),
                    linear_operator.settings.cholesky_jitter(
                        float_value=self.cholesky_jitter, double_value=self.cholesky_jitter
                    ),
                ):
                    for _ in range(self.num_warmup_epochs):
                        optimizer.zero_grad()
                        # In train mode, call model() without arguments to use stored training data
                        output = base_model()
                        loss = -mll(output, train_y)
                        
                        # Check for invalid loss
                        if not torch.isfinite(loss):
                            break
                        
                        loss.backward()
                        
                        # Clip gradients to prevent explosion
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=10.0)
                        
                        optimizer.step()
                    
                    # Evaluate loss after warmup
                    base_model.eval()
                    with torch.no_grad():
                        # Call model() without arguments to use stored training data (avoids warning)
                        try:
                            output = base_model()
                        except Exception:
                            # Fallback: pass training data explicitly
                            output = base_model(train_x)
                        loss = -mll(output, train_y)
                        loss_after_warmup = loss.item()
                        if not torch.isfinite(torch.tensor(loss_after_warmup)):
                            logger.debug(f"Non-finite loss after warmup for run_index {run_index}: {loss_after_warmup}")
                            loss_after_warmup = float("inf")
            except Exception as e:
                logger.debug(f"Warmup training failed for run_index {run_index}: {e}")
                loss_after_warmup = float("inf")
        else:
            # No warmup - use initial loss
            loss_after_warmup = initial_loss_value
        
        # Clean up model copy to free memory (helpful for large models)
        del base_model
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        
        # Debug: Verify losses are being returned
        if run_index < 3:
            logger.info(f"DEBUG run_index {run_index}: Returning loss={loss_after_warmup}, initial_loss={initial_loss_value}")
        
        # Return data including recording information (will be recorded after parallel execution)
        return {
            "run_index": run_index,
            "loss": loss_after_warmup,  # Use loss after warmup for selection
            "initial_loss": initial_loss_value,  # Keep initial loss for reference
            # Store data for recording (will be recorded after parallel execution)
            "_recording_data": {
                "sobol_sample": sobol_sample,
                "initial_parameters": initial_parameters,
            } if self.recorder is not None else None,
        }
    
    def prescreen(self) -> List[int]:
        """
        Pre-screen initializations by evaluating loss for N_test samples
        and selecting the best N_initialization samples.
        
        Returns:
            list: List of run indices (from N_test) to use for training,
                  sorted by loss (best first)
        """
        logger.info("="*60)
        logger.info("PRE-SCREENING INITIALIZATIONS")
        logger.info("="*60)
        logger.info(f"Total candidates to evaluate: {self.num_test}")
        logger.info(f"Will select best: {self.num_runs} initializations")
        logger.info(f"Selection ratio: {self.num_runs}/{self.num_test} = {self.num_runs/self.num_test:.2%}")
        if self.num_warmup_epochs > 0:
            logger.info(f"Using {self.num_warmup_epochs} warmup epochs (lr={self.warmup_lr}) to make loss evaluation more meaningful")
        logger.info("="*60)
        
        # Evaluate loss for all N_test initializations
        def safe_evaluate(run_index):
            try:
                return self.evaluate_initial_loss(run_index)
            except Exception as e:
                logger.exception(f"Error evaluating loss for run_index {run_index}: {e}")
                return {
                    "run_index": run_index, 
                    "loss": float("inf"),
                    "initial_loss": float("inf"),
                    "_recording_data": None,
                }
        
        # Use parallel evaluation for pre-screening
        if self.device.type == "cpu":
            max_jobs = min(self.num_test, max(1, (os.cpu_count() or 1) - 2))
            logger.info(f"Pre-screening using {max_jobs} parallel processes on CPU")
            prescreen_results = Parallel(n_jobs=max_jobs, backend="loky", verbose=0)(
                delayed(safe_evaluate)(run_index) for run_index in range(self.num_test)
            )
        elif str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
            num_gpus = torch.cuda.device_count()
            max_jobs = min(self.num_test, num_gpus)
            logger.info(f"Pre-screening using {max_jobs} parallel processes on GPU")
            prescreen_results = Parallel(n_jobs=max_jobs, backend="threading", verbose=0)(
                delayed(safe_evaluate)(run_index) for run_index in range(self.num_test)
            )
        else:
            # Fallback to sequential
            prescreen_results = [safe_evaluate(run_index) for run_index in range(self.num_test)]
        
        # Sort by loss and select best N_initialization
        # Handle NaN and Inf values properly
        def get_sort_key(x):
            loss = x["loss"]
            if loss == float("inf") or (isinstance(loss, float) and (loss != loss)):  # Check for NaN
                return float("inf")
            return loss
        
        prescreen_results.sort(key=get_sort_key)
        
        # Log sorting info for debugging
        if len(prescreen_results) > 0:
            best_loss = prescreen_results[0]["loss"]
            worst_loss = prescreen_results[-1]["loss"]
            logger.debug(f"Pre-screening losses: best={best_loss:.6f}, worst={worst_loss:.6f}")
            # Show first few run_indices to verify sorting
            first_few_indices = [r["run_index"] for r in prescreen_results[:min(5, len(prescreen_results))]]
            logger.debug(f"First {len(first_few_indices)} run_indices after sorting: {first_few_indices}")
        
        # Ensure we have enough results
        num_to_select = min(self.num_runs, len(prescreen_results))
        selected_indices = [result["run_index"] for result in prescreen_results[:num_to_select]]
        
        # Log selected indices for verification
        logger.info(f"Selected {num_to_select} best initializations from {len(prescreen_results)} candidates")
        if len(selected_indices) <= 10:
            logger.info(f"Selected run_indices: {selected_indices}")
        else:
            logger.info(f"Selected run_indices (first 5, last 5): {selected_indices[:5]} ... {selected_indices[-5:]}")
        
        # Record all prescreening data if recorder is available
        # (Do this after parallel execution to avoid serialization issues)
        if self.recorder is not None:
            num_inf_losses = 0
            num_valid_losses = 0
            for result in prescreen_results:
                recording_data = result.get("_recording_data")
                # Extract loss values from result (not from _recording_data)
                initial_loss = result.get("initial_loss", float("inf"))
                loss_after_warmup = result.get("loss", float("inf"))
                
                # Debug: Check first few results to verify losses are being extracted
                if result.get("run_index", -1) < 3:
                    logger.info(f"DEBUG Recording run_index {result.get('run_index')}: initial_loss={initial_loss}, loss_after_warmup={loss_after_warmup}")
                
                # Track loss statistics
                if loss_after_warmup == float("inf") or (isinstance(loss_after_warmup, float) and not np.isfinite(loss_after_warmup)):
                    num_inf_losses += 1
                else:
                    num_valid_losses += 1
                
                if recording_data is not None:
                    self.recorder.record_prescreening_candidate(
                        run_index=result["run_index"],
                        sobol_sample=recording_data.get("sobol_sample"),
                        initial_parameters=recording_data.get("initial_parameters"),
                        initial_loss=initial_loss,
                        loss_after_warmup=loss_after_warmup,
                    )
                else:
                    # Still record even if recording_data is None (e.g., from error case)
                    self.recorder.record_prescreening_candidate(
                        run_index=result["run_index"],
                        sobol_sample=None,
                        initial_parameters=None,
                        initial_loss=initial_loss,
                        loss_after_warmup=loss_after_warmup,
                    )
            
            # Log loss statistics
            if num_inf_losses > 0:
                logger.warning(f"Prescreening: {num_inf_losses} candidates had infinite/non-finite losses")
            if num_valid_losses > 0:
                logger.info(f"Prescreening: {num_valid_losses} candidates had valid losses")
            
            self.recorder.record_selected_indices(selected_indices)
        
        # Log results safely
        if len(prescreen_results) > 0:
            best_loss = prescreen_results[0]["loss"]
            worst_selected = prescreen_results[num_to_select - 1]["loss"] if num_to_select > 0 else float("inf")
            
            # Show improvement from warmup if available
            if "initial_loss" in prescreen_results[0]:
                best_initial = prescreen_results[0]["initial_loss"]
                improvement = best_initial - best_loss if best_initial != float("inf") else 0
                logger.info("="*60)
                logger.info("PRE-SCREENING RESULTS")
                logger.info("="*60)
                logger.info(f"Evaluated: {self.num_test} initializations")
                logger.info(f"Selected: {len(selected_indices)} best initializations")
                logger.info(f"Best loss after {self.num_warmup_epochs} warmup epochs: {best_loss:.6f}")
                logger.info(f"  (Initial loss: {best_initial:.6f}, Improvement: {improvement:.6f})")
                logger.info(f"Worst selected loss: {worst_selected:.6f}")
                logger.info("="*60)
            else:
                logger.info("="*60)
                logger.info("PRE-SCREENING RESULTS")
                logger.info("="*60)
                logger.info(f"Evaluated: {self.num_test} initializations")
                logger.info(f"Selected: {len(selected_indices)} best initializations")
                logger.info(f"Best loss: {best_loss:.6f}")
                logger.info(f"Worst selected: {worst_selected:.6f}")
                logger.info("="*60)
        else:
            logger.warning(f"Pre-screening: No valid results found! Using first {num_to_select} indices.")
            selected_indices = list(range(num_to_select))
        
        return selected_indices

