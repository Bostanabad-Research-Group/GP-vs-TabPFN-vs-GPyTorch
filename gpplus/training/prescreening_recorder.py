"""
Pre-screening Recorder for GP Models

This module provides a class for recording and tracking pre-screening data,
including Sobol sequence parameters, losses, selected initializations, and final results.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..config import logger


class PrescreeningRecorder:
    """
    Records and tracks pre-screening data for GP model optimization.
    
    This class tracks:
    1. Sobol sequence parameters generated for each candidate initialization
    2. Loss values evaluated during pre-screening (initial and after warmup)
    3. Which initializations were selected for full optimization
    4. Final loss values after full optimization for each selected initialization
    
    Args:
        save_path: Optional path to save recorded data (JSON format)
        verbose: Whether to log recording activities
    """
    
    def __init__(self, save_path: Optional[str] = None, verbose: bool = True):
        self.save_path = save_path
        self.verbose = verbose
        
        # Storage for pre-screening data
        self.prescreening_data: List[Dict[str, Any]] = []
        self.selected_indices: List[int] = []
        
        # Storage for final optimization results
        self.final_results: Dict[int, Dict[str, Any]] = {}  # Maps selected_index -> final results
        
        # Track if recording has started
        self.recording_started = False
        self.num_test = None
        self.num_runs = None
        
    def start_recording(self, num_test: int, num_runs: int):
        """
        Initialize recording for a new pre-screening session.
        
        Args:
            num_test: Total number of candidates to evaluate
            num_runs: Number of candidates to select
        """
        self.prescreening_data = []
        self.selected_indices = []
        self.final_results = {}
        self.recording_started = True
        self.num_test = num_test
        self.num_runs = num_runs
        self.num_test = num_test
        self.num_runs = num_runs
        
        if self.verbose:
            logger.info(f"PrescreeningRecorder: Started recording for {num_test} candidates, will select {num_runs}")
    
    def record_prescreening_candidate(
        self,
        run_index: int,
        sobol_sample: Optional[torch.Tensor] = None,
        initial_parameters: Optional[Dict[str, torch.Tensor]] = None,
        initial_loss: float = float("inf"),
        loss_after_warmup: float = float("inf"),
    ):
        """
        Record data for a single pre-screening candidate.
        
        Args:
            run_index: Index of the candidate (0 to num_test-1)
            sobol_sample: The Sobol sequence sample used (if available)
            initial_parameters: Dictionary of parameter name -> tensor values after initialization
            initial_loss: Loss value before warmup training
            loss_after_warmup: Loss value after warmup training (used for selection)
        """
        if not self.recording_started:
            logger.warning("PrescreeningRecorder: Recording not started. Call start_recording() first.")
            return
        
        # Convert tensors to lists for JSON serialization
        sobol_sample_list = None
        if sobol_sample is not None:
            sobol_sample_list = sobol_sample.detach().cpu().numpy().tolist()
        
        initial_params_dict = None
        if initial_parameters is not None:
            initial_params_dict = {}
            for name, param in initial_parameters.items():
                if isinstance(param, torch.Tensor):
                    initial_params_dict[name] = param.detach().cpu().numpy().tolist()
                else:
                    initial_params_dict[name] = param
        
        record = {
            "run_index": run_index,
            "sobol_sample": sobol_sample_list,
            "initial_parameters": initial_params_dict,
            "initial_loss": float(initial_loss) if torch.isfinite(torch.tensor(initial_loss)) else float("inf"),
            "loss_after_warmup": float(loss_after_warmup) if torch.isfinite(torch.tensor(loss_after_warmup)) else float("inf"),
        }
        
        self.prescreening_data.append(record)
        
        if self.verbose and len(self.prescreening_data) % 10 == 0:
            logger.debug(f"PrescreeningRecorder: Recorded {len(self.prescreening_data)} candidates")
    
    def record_selected_indices(self, selected_indices: List[int]):
        """
        Record which candidate indices were selected for full optimization.
        
        Args:
            selected_indices: List of run_index values that were selected
        """
        self.selected_indices = selected_indices.copy()
        
        if self.verbose:
            logger.info(f"PrescreeningRecorder: Recorded {len(selected_indices)} selected indices")
            # Show first few and last few to avoid cluttering log
            if len(selected_indices) <= 10:
                logger.info(f"  Selected indices: {selected_indices}")
            else:
                logger.info(f"  Selected indices: {selected_indices[:5]} ... {selected_indices[-5:]}")
                logger.info(f"  (Full list: {selected_indices})")
    
    def record_final_result(
        self,
        selected_index: int,
        final_loss: float,
        final_parameters: Optional[Dict[str, torch.Tensor]] = None,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Record final optimization result for a selected initialization.
        
        Args:
            selected_index: The run_index that was selected (from pre-screening)
            final_loss: Final loss after full optimization
            final_parameters: Dictionary of parameter name -> tensor values after optimization
            additional_metrics: Optional dictionary of additional metrics to record
        """
        # Convert tensors to lists for JSON serialization
        final_params_dict = None
        if final_parameters is not None:
            final_params_dict = {}
            for name, param in final_parameters.items():
                if isinstance(param, torch.Tensor):
                    final_params_dict[name] = param.detach().cpu().numpy().tolist()
                else:
                    final_params_dict[name] = param
        
        result = {
            "selected_index": selected_index,
            "final_loss": float(final_loss) if torch.isfinite(torch.tensor(final_loss)) else float("inf"),
            "final_parameters": final_params_dict,
            "additional_metrics": additional_metrics or {},
        }
        
        self.final_results[selected_index] = result
        
        if self.verbose:
            logger.info(f"PrescreeningRecorder: Recorded final result for selected_index {selected_index}: loss={final_loss:.6f}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all recorded data.
        
        Returns:
            Dictionary containing summary statistics and data
        """
        if not self.prescreening_data:
            return {"message": "No pre-screening data recorded"}
        
        # Extract losses
        initial_losses = [d["initial_loss"] for d in self.prescreening_data if d["initial_loss"] != float("inf")]
        warmup_losses = [d["loss_after_warmup"] for d in self.prescreening_data if d["loss_after_warmup"] != float("inf")]
        final_losses = [r["final_loss"] for r in self.final_results.values() if r["final_loss"] != float("inf")]
        
        summary = {
            "num_candidates": len(self.prescreening_data),
            "num_selected": len(self.selected_indices),
            "selected_indices": self.selected_indices,
            "prescreening_stats": {
                "initial_loss": {
                    "min": float(np.min(initial_losses)) if initial_losses else None,
                    "max": float(np.max(initial_losses)) if initial_losses else None,
                    "mean": float(np.mean(initial_losses)) if initial_losses else None,
                    "std": float(np.std(initial_losses)) if initial_losses else None,
                },
                "warmup_loss": {
                    "min": float(np.min(warmup_losses)) if warmup_losses else None,
                    "max": float(np.max(warmup_losses)) if warmup_losses else None,
                    "mean": float(np.mean(warmup_losses)) if warmup_losses else None,
                    "std": float(np.std(warmup_losses)) if warmup_losses else None,
                },
            },
            "final_optimization_stats": {
                "num_completed": len(self.final_results),
                "final_loss": {
                    "min": float(np.min(final_losses)) if final_losses else None,
                    "max": float(np.max(final_losses)) if final_losses else None,
                    "mean": float(np.mean(final_losses)) if final_losses else None,
                    "std": float(np.std(final_losses)) if final_losses else None,
                },
            },
        }
        
        # Add improvement statistics
        if warmup_losses and final_losses:
            best_warmup = np.min(warmup_losses)
            best_final = np.min(final_losses)
            improvement = best_warmup - best_final
            improvement_pct = (improvement / abs(best_warmup)) * 100 if best_warmup != 0 else 0
            summary["improvement"] = {
                "best_warmup_loss": float(best_warmup),
                "best_final_loss": float(best_final),
                "absolute_improvement": float(improvement),
                "relative_improvement_percent": float(improvement_pct),
            }
        
        return summary
    
    def save(self, filepath: Optional[str] = None):
        """
        Save all recorded data to a JSON file.
        
        Args:
            filepath: Path to save file. If None, uses self.save_path
        """
        save_path = filepath or self.save_path
        if save_path is None:
            logger.warning("PrescreeningRecorder: No save path specified, skipping save")
            return
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        data_to_save = {
            "prescreening_data": self.prescreening_data,
            "selected_indices": self.selected_indices,
            "final_results": self.final_results,
            "summary": self.get_summary(),
        }
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        if self.verbose:
            logger.info(f"PrescreeningRecorder: Saved data to {os.path.abspath(save_path)}")
            logger.info(f"  - {len(self.prescreening_data)} candidates recorded")
            logger.info(f"  - {len(self.selected_indices)} selected indices")
            logger.info(f"  - {len(self.final_results)} final results")
    
    def load(self, filepath: str):
        """
        Load recorded data from a JSON file.
        
        Args:
            filepath: Path to load file from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.prescreening_data = data.get("prescreening_data", [])
        self.selected_indices = data.get("selected_indices", [])
        self.final_results = data.get("final_results", {})
        self.recording_started = True
        
        if self.verbose:
            logger.info(f"PrescreeningRecorder: Loaded data from {filepath}")
            logger.info(f"  - {len(self.prescreening_data)} candidates")
            logger.info(f"  - {len(self.selected_indices)} selected")
            logger.info(f"  - {len(self.final_results)} final results")
    
    def print_summary(self):
        """Print a human-readable summary of the recorded data."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PRE-SCREENING RECORDER SUMMARY")
        print("="*60)
        
        # Check if there's no data
        if "message" in summary:
            print(summary["message"])
            print("="*60 + "\n")
            return
        
        print(f"Candidates evaluated: {summary['num_candidates']}")
        print(f"Candidates selected: {summary['num_selected']}")
        selected_indices = summary['selected_indices']
        if len(selected_indices) <= 10:
            print(f"Selected run_indices (from original {self.num_test if self.num_test else 'N_test'} candidates): {selected_indices}")
        else:
            print(f"Selected run_indices (first 5, last 5): {selected_indices[:5]} ... {selected_indices[-5:]}")
            print(f"  (Full list of {len(selected_indices)} indices: {selected_indices})")
        
        if summary['prescreening_stats']['warmup_loss']['min'] is not None:
            warmup = summary['prescreening_stats']['warmup_loss']
            print(f"\nPre-screening loss (after warmup):")
            print(f"  Min: {warmup['min']:.6f}")
            print(f"  Max: {warmup['max']:.6f}")
            print(f"  Mean: {warmup['mean']:.6f}")
            print(f"  Std: {warmup['std']:.6f}")
        
        if summary['final_optimization_stats']['num_completed'] > 0:
            final = summary['final_optimization_stats']['final_loss']
            print(f"\nFinal optimization loss:")
            print(f"  Min: {final['min']:.6f}")
            print(f"  Max: {final['max']:.6f}")
            print(f"  Mean: {final['mean']:.6f}")
            print(f"  Std: {final['std']:.6f}")
        
        if 'improvement' in summary:
            imp = summary['improvement']
            print(f"\nImprovement from pre-screening to final:")
            print(f"  Best warmup loss: {imp['best_warmup_loss']:.6f}")
            print(f"  Best final loss: {imp['best_final_loss']:.6f}")
            print(f"  Absolute improvement: {imp['absolute_improvement']:.6f}")
            print(f"  Relative improvement: {imp['relative_improvement_percent']:.2f}%")
        
        print("="*60 + "\n")
    
    def to_dataframe(self) -> Optional['pd.DataFrame']:
        """
        Convert prescreening data to a pandas DataFrame.
        
        Returns:
            DataFrame with columns: rank, run_index, initial_loss, loss_after_warmup, 
            is_selected, and parameter values (if available)
        """
        if not PANDAS_AVAILABLE:
            logger.warning("pandas is not available. Install pandas to use to_dataframe().")
            return None
        
        if not self.prescreening_data:
            logger.warning("No prescreening data available.")
            return None
        
        # Prepare data for DataFrame
        rows = []
        for record in self.prescreening_data:
            row = {
                "run_index": record["run_index"],
                "initial_loss": record["initial_loss"],
                "loss_after_warmup": record["loss_after_warmup"],
                "is_selected": record["run_index"] in self.selected_indices,
            }
            
            # Add parameter values if available (flatten nested structure)
            if record.get("initial_parameters") is not None:
                for param_name, param_value in record["initial_parameters"].items():
                    # Handle different parameter shapes
                    if isinstance(param_value, list):
                        if len(param_value) == 1:
                            row[f"param_{param_name}"] = param_value[0]
                        else:
                            # For multi-element parameters, create multiple columns
                            for idx, val in enumerate(param_value):
                                row[f"param_{param_name}_{idx}"] = val
                    else:
                        row[f"param_{param_name}"] = param_value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by loss_after_warmup (best first)
        df = df.sort_values("loss_after_warmup", ascending=True).reset_index(drop=True)
        
        # Add rank column
        df.insert(0, "rank", range(1, len(df) + 1))
        
        return df
    
    def save_table(self, filepath: Optional[str] = None, format: str = "csv"):
        """
        Save prescreening data as a table (CSV or Excel).
        
        Args:
            filepath: Path to save file. If None, uses self.save_path with .csv/.xlsx extension
            format: Format to save ("csv" or "excel")
        """
        df = self.to_dataframe()
        if df is None:
            return
        
        if filepath is None:
            if self.save_path is None:
                logger.warning("No filepath specified and no save_path set. Cannot save table.")
                return
            # Change extension based on format
            base_path = Path(self.save_path).with_suffix('')
            if format == "excel":
                filepath = str(base_path) + "_table.xlsx"
            else:
                filepath = str(base_path) + "_table.csv"
        
        if format == "excel":
            if not PANDAS_AVAILABLE:
                logger.warning("pandas is required for Excel export. Saving as CSV instead.")
                format = "csv"
            else:
                try:
                    df.to_excel(filepath, index=False)
                    if self.verbose:
                        logger.info(f"PrescreeningRecorder: Saved table to {os.path.abspath(filepath)}")
                    return
                except Exception as e:
                    logger.warning(f"Could not save as Excel: {e}. Saving as CSV instead.")
                    format = "csv"
        
        if format == "csv":
            try:
                df.to_csv(filepath, index=False)
                if self.verbose:
                    logger.info(f"PrescreeningRecorder: Saved table to {os.path.abspath(filepath)}")
            except PermissionError as e:
                logger.error(f"Permission denied when saving table to {filepath}")
                logger.error("The file may be open in another program (e.g., Excel). Please close it and try again.")
                logger.error(f"Error details: {e}")
                raise
            except Exception as e:
                logger.error(f"Error saving table to {filepath}: {e}")
                raise
    
    def print_table(self, max_rows: int = 20, include_params: bool = False):
        """
        Print prescreening data as a formatted table.
        
        Args:
            max_rows: Maximum number of rows to display
            include_params: Whether to include parameter columns (can be very wide)
        """
        df = self.to_dataframe()
        if df is None:
            return
        
        # Select columns to display
        display_cols = ["rank", "run_index", "initial_loss", "loss_after_warmup", "is_selected"]
        
        if include_params:
            # Include all parameter columns
            param_cols = [col for col in df.columns if col.startswith("param_")]
            display_cols.extend(param_cols)
        
        df_display = df[display_cols].head(max_rows)
        
        print("\n" + "="*80)
        print("PRE-SCREENING DATA TABLE")
        print("="*80)
        print(df_display.to_string(index=False))
        if len(df) > max_rows:
            print(f"\n... ({len(df) - max_rows} more rows, use print_table(max_rows={len(df)}) to see all)")
        print("="*80 + "\n")

