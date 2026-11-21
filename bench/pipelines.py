from typing import List, Callable, Optional

from pathlib import Path
import json
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from .dataclasses import Problem, SearchResult
from .searchers.base import BaseSearcher
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_percentage_error

def compute_output_base_metrics(y_pred, y, tolerance=0.01):
    """
    Compute base metrics for equation discovery evaluation.
    
    Args:
        y_pred: Predicted values
        y: True values  
        tolerance: Tolerance threshold for accuracy calculation (default: 0.01)
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    nonnan_idx = np.argwhere(~np.isnan(y_pred))
    y_pred = y_pred[nonnan_idx]
    y = y[nonnan_idx]

    var = np.var(y)
    nmse = np.mean((y - y_pred)**2) / var 
    if np.sum((y - y.mean())**2) == 0:
        print(y)
    r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))
    kdt = kendalltau(y, y_pred)[0]
    mape = mean_absolute_percentage_error(y, y_pred)
    
    # Calculate accuracy to tolerance (Accτ) and Accτ_max
    # Accτ = 1 if max(1≤i≤Ntest |ŷi - yi|/|yi|) ≤ τ, else 0
    # Accτ_max = 1 if max(relative_errors after excluding worst 5%) ≤ τ, else 0
    # Both are binary metrics: 1 if predictions are within tolerance, 0 otherwise
    relative_errors = np.abs(y_pred - y) / np.abs(y)
    # Avoid division by zero for y=0 cases
    valid_idx = np.abs(y) > 1e-10
    if np.sum(valid_idx) > 0:
        relative_errors_valid = relative_errors[valid_idx]
        n_valid = len(relative_errors_valid)
        
        # Calculate acc_tau: max relative error across all valid points
        max_relative_error = np.max(relative_errors_valid)
        acc_tau = 1.0 if max_relative_error <= tolerance else 0.0
        
        # Calculate acc_tau_max: max relative error after excluding worst 5%
        if n_valid > 0:
            # Sort relative errors in ascending order
            sorted_errors = np.sort(relative_errors_valid)
            
            # Calculate number of worst points to exclude (5%, at least 1 if n_valid >= 20)
            if n_valid >= 20:
                n_exclude = max(1, int(n_valid * 0.05))
            else:
                # For small datasets, exclude at most 1 point if n_valid >= 2
                n_exclude = 1 if n_valid >= 2 else 0
            
            # Exclude worst n_exclude points, keep the remaining 95%
            if n_exclude > 0 and n_valid > n_exclude:
                remaining_errors = sorted_errors[:-n_exclude]
            else:
                # If we can't exclude or all points would be excluded, use all points
                remaining_errors = sorted_errors
            
            if len(remaining_errors) > 0:
                max_relative_error_95 = np.max(remaining_errors)
                acc_tau_max = 1.0 if max_relative_error_95 <= tolerance else 0.0
            else:
                acc_tau_max = 0.0
        else:
            acc_tau_max = 0.0
    else:

        acc_tau = 0.0  # If all y values are zero, set accuracy to 0
        acc_tau_max = 0.0

    return {
        "mse": np.mean((y - y_pred)**2),
        "nmse": nmse,
        "r2": r2,
        "kdt": kdt,
        "mape": mape,
        "acc_tau": acc_tau,
        "acc_tau_max": acc_tau_max,
        "num_valid_points": len(y_pred),
    }

class EvaluationPipeline:
    def __init__(self):
        pass

    def run_and_evaluate(self, searcher: BaseSearcher, problem: Problem, verbose_timing=False):
        # Time statistics
        timing_stats = {}
        
        # 1. Formula discovery/search time
        start_time = time.time()
        search_results: List[SearchResult] = searcher.discover(problem.create_task())
        search_time = time.time() - start_time
        timing_stats['discover_time'] = search_time

        # 2. Data preparation time
        start_time = time.time()
        # Assuming the first symbol is the output
        X_id = problem.test_samples[:, 1:]
        y_id = problem.test_samples[:, 0] 

        X_ood = y_ood = None
        if problem.ood_test_samples is not None:
            X_ood = problem.ood_test_samples[:, 1:]
            y_ood = problem.ood_test_samples[:, 0] 
        data_prep_time = time.time() - start_time
        timing_stats['data_prep_time'] = data_prep_time

        # 3. Evaluation time for each result
        outs = []
        total_eval_time = 0
        total_pred_time = 0
        total_metrics_time = 0
        
        for idx, result in enumerate(search_results):
            result_timing = {}
            
            # 3.1 Get lambda function
            start_time = time.time()
            equation = result.equation
            lambda_fn = equation.lambda_format
            lambda_prep_time = time.time() - start_time
            result_timing['lambda_prep_time'] = lambda_prep_time
            
            # 3.2 Compute predictions (ID)
            start_time = time.time()
            y_pred_id = lambda_fn(X_id)
            if np.sum(y_pred_id) == 0.0:
                print(f"Warning: All prediction values for problem.equation_idx {problem.equation_idx} are 0")
            pred_id_time = time.time() - start_time
            result_timing['pred_id_time'] = pred_id_time
            total_pred_time += pred_id_time
            
            # 3.3 Compute metrics (ID)
            start_time = time.time()
            id_output_base_metrics = compute_output_base_metrics(y_pred_id, y_id)
            metrics_id_time = time.time() - start_time
            result_timing['metrics_id_time'] = metrics_id_time
            total_metrics_time += metrics_id_time
            
            # 3.4 OOD evaluation (if applicable)
            ood_output_base_metrics = None
            if X_ood is not None:
                start_time = time.time()
                y_pred_ood = lambda_fn(X_ood)
                pred_ood_time = time.time() - start_time
                result_timing['pred_ood_time'] = pred_ood_time
                total_pred_time += pred_ood_time
                
                start_time = time.time()
                ood_output_base_metrics = compute_output_base_metrics(y_pred_ood, y_ood)
                metrics_ood_time = time.time() - start_time
                result_timing['metrics_ood_time'] = metrics_ood_time
                total_metrics_time += metrics_ood_time
            else:
                result_timing['pred_ood_time'] = 0
                result_timing['metrics_ood_time'] = 0
            
            total_eval_time += (lambda_prep_time + pred_id_time + metrics_id_time + 
                              result_timing.get('pred_ood_time', 0) + 
                              result_timing.get('metrics_ood_time', 0))
            
            if verbose_timing:
                print(f"\n  [Result {idx+1}/{len(search_results)}] Timing breakdown:")
                print(f"    - Lambda function preparation: {lambda_prep_time:.4f}s")
                print(f"    - ID prediction computation: {pred_id_time:.4f}s")
                print(f"    - ID metrics calculation: {metrics_id_time:.4f}s")
                if X_ood is not None:
                    print(f"    - OOD prediction computation: {result_timing['pred_ood_time']:.4f}s")
                    print(f"    - OOD metrics calculation: {result_timing['metrics_ood_time']:.4f}s")

            outs.append({
                "search_result": result,
                "search_time": search_time,
                "id_metrics": id_output_base_metrics,
                "ood_metrics": ood_output_base_metrics,
            })
        
        timing_stats['total_eval_time'] = total_eval_time
        timing_stats['total_pred_time'] = total_pred_time
        timing_stats['total_metrics_time'] = total_metrics_time
        timing_stats['num_results'] = len(search_results)
        
        if verbose_timing:
            total_time = timing_stats['discover_time'] + timing_stats['data_prep_time'] + timing_stats['total_eval_time']
            print(f"\n{'='*60}")
            print(f"Timing Summary for Problem: {problem.equation_idx}")
            print(f"{'='*60}")
            print(f"1. Formula Discovery:     {timing_stats['discover_time']:>10.4f}s ({timing_stats['discover_time']/total_time*100:>5.1f}%)")
            print(f"2. Data Preparation:      {timing_stats['data_prep_time']:>10.4f}s ({timing_stats['data_prep_time']/total_time*100:>5.1f}%)")
            print(f"3. Evaluation (Total):    {timing_stats['total_eval_time']:>10.4f}s ({timing_stats['total_eval_time']/total_time*100:>5.1f}%)")
            print(f"   ├─ Prediction:         {timing_stats['total_pred_time']:>10.4f}s")
            print(f"   └─ Metrics:            {timing_stats['total_metrics_time']:>10.4f}s")
            print(f"{'─'*60}")
            print(f"Total Time:               {total_time:>10.4f}s")
            print(f"Number of Results:        {timing_stats['num_results']:>10d}")
            print(f"{'='*60}\n")

        return outs


    @property
    def name(self):
        raise NotImplementedError

    def load_visited_problems(self, output_dir):
        result_files = list(Path(output_dir).glob("results*.jsonl"))
        visited = []
        if len(result_files) > 0:
            for result_file in result_files:
                with open(result_file, 'r') as f:
                    for line in f.readlines():
                        visited.append(json.loads(line)['equation_id'])
            visited = list(set(visited))
        return visited