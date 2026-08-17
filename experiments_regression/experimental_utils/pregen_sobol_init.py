"""
Pre-generate Sobol samples in GP hyperparameter space for experiments_final.

Used by *_pregenParams scripts: one scrambled Sobol sequence of length
num_runs * num_inits, sliced per outer run for train_eval_gp.
"""

from __future__ import annotations

from typing import Any

import torch

from gpplus.config import logger
from gpplus.training.parameter_initializer import DefaultParameterInitializer


def build_master_hyper_sobol(
    initializer_class: type,
    template_model: torch.nn.Module,
    total_rows: int,
    sobol_seed: int,
    **initializer_extra_kwargs: Any,
) -> DefaultParameterInitializer:
    """
    Draw ``total_rows`` Sobol rows in hyperparameter space (same rules as DefaultParameterInitializer.setup).
    """
    init = initializer_class(num_runs=total_rows, seed=sobol_seed, **initializer_extra_kwargs)
    init.setup(template_model)
    return init


class SobolRowsSliceInitializer(DefaultParameterInitializer):
    """
    Use a fixed tensor of Sobol [0,1] rows instead of drawing in setup().
    ``seed`` is still used for Xavier / reproducible weight init (same as parent).
    """

    def __init__(
        self,
        num_runs: int,
        seed: int | None = None,
        sobol_rows: torch.Tensor | None = None,
        parameter_configs: dict | None = None,
    ):
        super().__init__(num_runs, seed, parameter_configs)
        self._preset_rows = sobol_rows

    def setup(self, model: torch.nn.Module) -> None:
        if self._preset_rows is None:
            raise ValueError("SobolRowsSliceInitializer requires sobol_rows=... in constructor")

        self.num_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad and ".weight" not in name and ".bias" not in name:
                self.num_params += param.numel()

        rows = self._preset_rows
        if rows.shape[0] != self.num_runs:
            raise ValueError(
                f"sobol_rows has {rows.shape[0]} rows but num_runs={self.num_runs}"
            )
        if rows.shape[1] != self.num_params:
            raise ValueError(
                f"sobol_rows second dim {rows.shape[1]} != num_params {self.num_params} "
                f"(Sobol hyperparameter count for this model)"
            )

        self.sobol_samples = rows.detach().clone()
        logger.info("Using SobolRowsSliceInitializer (pregen hyperparameter Sobol slice)")
        logger.debug(f"Pregen Sobol samples shape: {self.sobol_samples.shape}")
