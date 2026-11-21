"""Evaluation and execution utilities for SE method."""

import json
import bdb
import time
import logging
import numpy as np
import threading
from pathlib import Path
from typing import List, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import OmegaConf
from bench.dataclasses import Problem
from bench.pipelines import compute_output_base_metrics
from methods.se.method import SESearcher
from methods.se.config import SEEvolutionConfig, MemoryConfig
from dataclasses import fields, asdict


def _get_cfg_value(cfg, key: str, default=None):
    value = OmegaConf.select(cfg, key)
    return value if value is not None else default


def _apply_base_config(cfg, config_path: str = None):
    if not OmegaConf.is_config(cfg):
        return cfg
    
    base_ref = cfg.pop("base_config", None)
    if not base_ref:
        return cfg
    
    if config_path:
        base_path = (Path(config_path).parent / base_ref).resolve()
        base_cfg = OmegaConf.load(str(base_path))
    else:
        base_cfg = OmegaConf.load(base_ref)
    
    return OmegaConf.merge(base_cfg, cfg)


def _create_evolution_config_from_cfg(searcher_cfg) -> SEEvolutionConfig:
    config_dict = {}
    float_fields = {'improvement_epsilon', 'memory_min_improvement', 'memory_improvement_threshold',
                    'memory_stagnation_failed_parent_top_percent', 'w_mse', 'w_hist', 
                    'w_comp', 'w_sample', 'w_elite', 'selection_decay_factor', 'rank_decay_factor',
                    'initial_temperature', 'min_temperature', 'cooling_rate'}
    
    cfg = searcher_cfg
    memory_nested = OmegaConf.select(cfg, 'memory')
    use_nested = memory_nested is not None and (
        OmegaConf.is_config(memory_nested) and 
        ('normal' in memory_nested or 'stagnation' in memory_nested)
    )
    
    for field in fields(SEEvolutionConfig):
        value = None
        
        if field.name.startswith('memory_') and use_nested:
            memory_key = field.name.replace('memory_', '')
            
            if memory_key == 'enabled':
                value = OmegaConf.select(memory_nested, 'enabled')
            elif memory_key == 'epoch_interval':
                value = OmegaConf.select(memory_nested, 'normal.epoch_interval')
            elif memory_key == 'improvement_threshold':
                value = OmegaConf.select(memory_nested, 'normal.improvement_threshold')
            elif memory_key in ['stagnation_k', 'stagnation_cooldown', 'stagnation_top_k', 
                               'stagnation_residual_analysis_top_n', 'stagnation_failed_mutations',
                               'stagnation_failed_parent_top_percent']:
                key = memory_key.replace('stagnation_', '')
                value = OmegaConf.select(memory_nested, f'stagnation.{key}')
            elif memory_key in ['good_top_s', 'bad_top_f']:
                value = OmegaConf.select(memory_nested, f'normal.{memory_key}')
            elif memory_key in ['insights_n', 'max_entries', 'min_improvement', 'debug_save']:
                key = memory_key if memory_key != 'debug_save' else 'debug_save_memory'
                value = OmegaConf.select(memory_nested, f'general.{key}')
                if value is None:
                    value = OmegaConf.select(memory_nested, key)
        
        if value is None:
            value = OmegaConf.select(cfg, field.name)
        
        if value is not None:
            if field.name in float_fields and isinstance(value, (int, str)):
                try:
                    config_dict[field.name] = float(value)
                except (ValueError, TypeError):
                    pass
            else:
                config_dict[field.name] = value
    return SEEvolutionConfig(**config_dict)


def _convert_cfg_to_dict(cfg) -> dict:
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    elif hasattr(cfg, '__dict__'):
        return vars(cfg)
    elif isinstance(cfg, dict):
        return cfg
    else:
        return {}


def _extract_searcher_config(searcher: SESearcher) -> dict:
    return {
        "name": searcher.name,
        "api_type": searcher.api_type,
        "api_model": searcher.model,
        "n": searcher.n,
        "temperature": searcher.temperature,
        "rank_percent_threshold": searcher.strategy_table.rank_percent_threshold,
        "visit_threshold": searcher.strategy_table.base_visit_threshold,
        "visit_threshold_scale": searcher.strategy_table.visit_threshold_scale,
        "enable_tools": searcher.enable_tools,
        "max_tool_rounds": searcher.max_tool_rounds,
    }


def _extract_strategy_table_config(strategy_table) -> dict:
    return {
        "alpha": strategy_table.alpha,
        "rank_percent_threshold": strategy_table.rank_percent_threshold,
        "visit_threshold": strategy_table.base_visit_threshold,
        "visit_threshold_scale": strategy_table.visit_threshold_scale,
        "refine_suppression_factor": strategy_table.refine_suppression_factor,
        "regenerate_boost": strategy_table.regenerate_boost,
        "min_probability": strategy_table.min_probability,
        "max_reward_clip": strategy_table.max_reward_clip,
        "min_decay_factor": strategy_table.min_decay_factor,
    }


def _extract_evolution_config(evolution_config: Optional[SEEvolutionConfig]) -> Optional[dict]:
    if evolution_config is None:
        return None
    return asdict(evolution_config)


def _extract_memory_config(evolution_config: Optional[SEEvolutionConfig]) -> Optional[dict]:
    if evolution_config is None or not evolution_config.memory_enabled:
        return None
    
    mem_cfg = MemoryConfig(
        enabled=evolution_config.memory_enabled,
        normal_epoch_interval=evolution_config.memory_epoch_interval,
        improvement_threshold=evolution_config.memory_improvement_threshold,
        good_top_s=evolution_config.memory_good_top_s,
        bad_top_f=evolution_config.memory_bad_top_f,
        stagnation_k=evolution_config.memory_stagnation_k,
        stagnation_cooldown=evolution_config.memory_stagnation_cooldown,
        stagnation_top_k=evolution_config.memory_stagnation_top_k,
        stagnation_residual_analysis_top_n=evolution_config.memory_stagnation_residual_analysis_top_n,
        stagnation_failed_mutations=evolution_config.memory_stagnation_failed_mutations,
        stagnation_failed_parent_top_percent=evolution_config.memory_stagnation_failed_parent_top_percent,
        insights_n=evolution_config.memory_insights_n,
        max_entries=evolution_config.memory_max_entries,
        min_improvement=evolution_config.memory_min_improvement,
        debug_save_memory=evolution_config.memory_debug_save,
    )
    return asdict(mem_cfg)


def _save_experiment_config(
    searcher_cfg,
    example_searcher: SESearcher,
    evolution_config: Optional[SEEvolutionConfig],
    output_path: Path
):
    try:
        config_snapshot = {
            "experiment_metadata": {
                "experiment_name": example_searcher.name,
                "start_time": datetime.now().isoformat(),
                "use_evolution": evolution_config is not None,
            },
            "raw_config": _convert_cfg_to_dict(searcher_cfg),
            "actual_config": {
                "searcher": _extract_searcher_config(example_searcher),
                "strategy_table": _extract_strategy_table_config(example_searcher.strategy_table),
                "evolution": _extract_evolution_config(evolution_config),
                "memory": _extract_memory_config(evolution_config),
            },
            "default_values_source": {
                "note": "Defaults live in methods/se/config.py",
                "config_file": "methods/se/config.py"
            }
        }
        
        config_file_path = output_path / "experiment_config.json"
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment configuration saved to: {config_file_path}")
    except Exception as e:
        print(f"Warning: Failed to save experiment configuration: {e}")
        import traceback
        traceback.print_exc()


def load_visited_problems(output_dir: Path) -> List[str]:
    """Load list of already processed problems"""
    result_files = list(output_dir.glob("results*.jsonl"))
    visited = []
    if len(result_files) > 0:
        for result_file in result_files:
            with open(result_file, 'r') as f:
                for line in f.readlines():
                    try:
                        data = json.loads(line)
                        visited.append(data['equation_id'])
                    except:
                        continue
        visited = list(set(visited))
    return visited


def create_se_searcher(searcher_cfg) -> SESearcher:
    """Create a new SESearcher instance. Used for parallel processing.    """
    api_key = _get_cfg_value(searcher_cfg, 'api_key', None)
    base_url = _get_cfg_value(searcher_cfg, 'api_url', None) or _get_cfg_value(searcher_cfg, 'base_url', None)
    
    if api_key is None:
        raise ValueError("api_key must be provided in searcher_cfg")
    if base_url is None:
        raise ValueError("api_url or base_url must be provided in searcher_cfg")
    
    # Handle log_level: convert string to logging constant if needed
    log_level = _get_cfg_value(searcher_cfg, 'log_level')
    if log_level is None:
        log_level = logging.DEBUG
    elif isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.DEBUG)
    
    searcher_kwargs = {
        'name': _get_cfg_value(searcher_cfg, 'name'),
        'api_type': _get_cfg_value(searcher_cfg, 'api_type'),
        'api_model': _get_cfg_value(searcher_cfg, 'api_model'),
        'api_key': api_key,
        'base_url': base_url,
        'log_level': log_level,
    }
    
    temperature = _get_cfg_value(searcher_cfg, 'temperature')
    if temperature is not None:
        searcher_kwargs['temperature'] = float(temperature)
    
    rank_percent_threshold = _get_cfg_value(searcher_cfg, 'rank_percent_threshold')
    if rank_percent_threshold is not None:
        searcher_kwargs['rank_percent_threshold'] = float(rank_percent_threshold)
    
    visit_threshold = _get_cfg_value(searcher_cfg, 'visit_threshold')
    if visit_threshold is not None:
        searcher_kwargs['visit_threshold'] = int(visit_threshold)
    
    visit_threshold_scale = _get_cfg_value(searcher_cfg, 'visit_threshold_scale')
    if visit_threshold_scale is not None:
        searcher_kwargs['visit_threshold_scale'] = float(visit_threshold_scale)
    
    enable_tools = _get_cfg_value(searcher_cfg, 'enable_tools')
    if enable_tools is not None:
        searcher_kwargs['enable_tools'] = bool(enable_tools)
    
    max_tool_rounds = _get_cfg_value(searcher_cfg, 'max_tool_rounds')
    if max_tool_rounds is not None:
        searcher_kwargs['max_tool_rounds'] = int(max_tool_rounds)
    
    strategy_alpha = _get_cfg_value(searcher_cfg, 'strategy_alpha')
    if strategy_alpha is not None:
        searcher_kwargs['strategy_alpha'] = float(strategy_alpha)
    
    strategy_refine_suppression_factor = _get_cfg_value(searcher_cfg, 'strategy_refine_suppression_factor')
    if strategy_refine_suppression_factor is not None:
        searcher_kwargs['strategy_refine_suppression_factor'] = float(strategy_refine_suppression_factor)
    
    strategy_regenerate_boost = _get_cfg_value(searcher_cfg, 'strategy_regenerate_boost')
    if strategy_regenerate_boost is not None:
        searcher_kwargs['strategy_regenerate_boost'] = float(strategy_regenerate_boost)
    
    strategy_min_probability = _get_cfg_value(searcher_cfg, 'strategy_min_probability')
    if strategy_min_probability is not None:
        searcher_kwargs['strategy_min_probability'] = float(strategy_min_probability)
    
    strategy_max_reward_clip = _get_cfg_value(searcher_cfg, 'strategy_max_reward_clip')
    if strategy_max_reward_clip is not None:
        searcher_kwargs['strategy_max_reward_clip'] = float(strategy_max_reward_clip)
    
    strategy_min_decay_factor = _get_cfg_value(searcher_cfg, 'strategy_min_decay_factor')
    if strategy_min_decay_factor is not None:
        searcher_kwargs['strategy_min_decay_factor'] = float(strategy_min_decay_factor)
    
    return SESearcher(**searcher_kwargs)


def prepare_problem_with_split(problem: Problem, train_valid_split: float = 0.8, use_valid_split: bool = True) -> Problem:
    """Prepare a problem by splitting train_samples into train and valid sets.
    
    This function is called in the main thread before parallel processing to avoid
    race conditions when multiple threads access the same problem object.
    
    The returned Problem object will have samples with keys:
    - 'train': Training data (split from original train_samples if use_valid_split=True, 
              otherwise all original train_samples)
    - 'valid': Validation data (split from original train_samples, only if use_valid_split=True)
    - 'test': Test data (copied from original problem)
    - 'ood_test': OOD test data (if exists, copied from original problem)
    
    Args:
        problem: Original problem object
        train_valid_split: Ratio of training data to use for training (rest for validation).
                          Default 0.8 means 80% train, 20% valid.
                          Only used when use_valid_split=True.
        use_valid_split: Whether to split train_samples into train and valid sets.
                        If False, all train_samples go to train and no valid set is created.
                        Default True for backward compatibility.
    
    Returns:
        New Problem object with split train/valid data (if use_valid_split=True) 
        or all data in train (if use_valid_split=False), and preserved test data
    """
    train_samples_full = problem.train_samples.copy()
    
    if use_valid_split:
        n_samples = len(train_samples_full)
        n_train = int(n_samples * train_valid_split)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        valid_indices = indices[n_train:]
        
        train_samples_split = train_samples_full[train_indices].copy()
        valid_samples_split = train_samples_full[valid_indices].copy()
        

        samples = {
            'train': train_samples_split, 
            'valid': valid_samples_split,  
            'test': problem.test_samples.copy()  
        }
    else:
        samples = {
            'train': train_samples_full,  # All training data
            'test': problem.test_samples.copy()  # Preserve test data with explicit copy
        }
    
    if problem.ood_test_samples is not None:
        samples['ood_test'] = problem.ood_test_samples.copy()
    
    problem_prepared = Problem(
        dataset_identifier=problem.dataset_identifier,
        equation_idx=problem.equation_idx,
        gt_equation=problem.gt_equation,
        samples=samples,
        original_idx=problem.original_idx
    )
    
    return problem_prepared


def process_single_problem_se(
    problem: Problem,
    searcher: SESearcher,
    evolution_config: Optional[SEEvolutionConfig],
    output_path: Path,
    output_file_path: Path,
    verbose_timing: bool,
    file_lock: threading.Lock,
    visited_eqids_set: set,
    train_valid_split: float = 0.8
) -> Optional[dict]:
    """Process a single problem in a worker thread.
    
    Args:
        problem: Problem object with split train/valid data and test data in samples dict.
                samples should contain keys: 'train', 'valid', 'test', and optionally 'ood_test'
        train_valid_split: Ratio of training data to use for training (rest for validation).
                          Default 0.8 means 80% train, 20% valid.
    """
    try:
        with file_lock:
            if problem.equation_idx in visited_eqids_set:

                print(f"[Thread {threading.current_thread().name}] Skipping problem: {problem.equation_idx} (gt: {problem.gt_equation.expression})")
                return None
        
        print(f"[Thread {threading.current_thread().name}] Processing problem: {problem.equation_idx} (gt: {problem.gt_equation.expression})")
        
        # Time statistics
        timing_stats = {}
        use_evolution = evolution_config is not None
        
        # 1. Formula discovery/search time (uses train data for optimization)
        start_time = time.time()
        if use_evolution:
            search_results = searcher.evolve(problem, evolution_config, verbose_timing=verbose_timing, output_path=output_path)
        else:

            search_results = searcher.discover(problem)
        search_time = time.time() - start_time
        timing_stats['discover_time'] = search_time
        
        start_time = time.time()
        X_id = problem.test_samples[:, 1:]
        y_id = problem.test_samples[:, 0]

        X_id = np.asarray(X_id, dtype=float)
        y_id = np.asarray(y_id, dtype=float)
        
        X_ood = y_ood = None
        if problem.ood_test_samples is not None:
            X_ood = problem.ood_test_samples[:, 1:]
            y_ood = problem.ood_test_samples[:, 0]
            X_ood = np.asarray(X_ood, dtype=float)
            y_ood = np.asarray(y_ood, dtype=float)
        
        search_logs_dir = output_path / "search_logs"
        search_logs_dir.mkdir(exist_ok=True, parents=True)
        
        method_type = "evolve" if use_evolution else "discover"
        log_file_name = f"{problem.equation_idx}_{method_type}.json"
        log_file_path = search_logs_dir / log_file_name
        data_prep_time = time.time() - start_time
        timing_stats['data_prep_time'] = data_prep_time
        
        eval_results = []
        detailed_log_data = None
        total_eval_time = 0
        total_pred_time = 0
        total_metrics_time = 0
        
        if not search_results:
            print(f"  No search results available")
            return None
        
        if len(search_results) > 1:

            print(f"  Warning: Expected 1 result, got {len(search_results)}. Using first result.")
        
        best_result = search_results[0]
        
        result_timing = {}
        
        start_time = time.time()
        equation = best_result.equation
        lambda_fn = equation.lambda_format
        lambda_prep_time = time.time() - start_time
        result_timing['lambda_prep_time'] = lambda_prep_time
        
        if "_detailed_log" in best_result.aux:
            detailed_log_data = best_result.aux["_detailed_log"].copy()
            detailed_log_data["equation_id"] = problem.equation_idx
            detailed_log_data["method"] = best_result.aux.get("method", method_type)
            detailed_log_data["timestamp"] = datetime.now().isoformat()
            detailed_log_data["search_time"] = search_time
        
        start_time = time.time()
        try:
            y_pred_id = lambda_fn(X_id)
            if np.isscalar(y_pred_id):
                y_pred_id = np.full_like(y_id, y_pred_id, dtype=float)
            else:
                y_pred_id = np.asarray(y_pred_id, dtype=float).flatten()
            if y_pred_id.shape != y_id.shape:
                if y_pred_id.size == 1:
                    y_pred_id = np.full_like(y_id, y_pred_id.item(), dtype=float)
                else:
                    y_pred_id = y_pred_id[:len(y_id)]
            y_pred_id = np.asarray(y_pred_id, dtype=float)
        except Exception as e:
            print(f"Error computing predictions: {e}")
            y_pred_id = np.full_like(y_id, 0.0, dtype=float)
        pred_id_time = time.time() - start_time
        result_timing['pred_id_time'] = pred_id_time
        total_pred_time += pred_id_time
        
        start_time = time.time()
        try:
            id_metrics = compute_output_base_metrics(y_pred_id, y_id)
        except Exception as e:
            print(f"Error calculating test metrics: {e}")
            id_metrics = {"mse": float('inf'), "nmse": float('inf'), "r2": -float('inf'), 
                          "kdt": 0, "mape": float('inf'), "num_valid_points": 0}
        metrics_id_time = time.time() - start_time
        result_timing['metrics_id_time'] = metrics_id_time
        total_metrics_time += metrics_id_time
        
        ood_metrics = None
        if X_ood is not None:
            try:
                y_pred_ood = lambda_fn(X_ood)
                if np.isscalar(y_pred_ood):
                    y_pred_ood = np.full_like(y_ood, y_pred_ood, dtype=float)
                else:
                    y_pred_ood = np.asarray(y_pred_ood, dtype=float).flatten()
                if y_pred_ood.shape != y_ood.shape:
                    if y_pred_ood.size == 1:
                        y_pred_ood = np.full_like(y_ood, y_pred_ood.item(), dtype=float)
                    else:
                        y_pred_ood = y_pred_ood[:len(y_ood)]
                # Ensure y_pred_ood is numeric (float) type
                y_pred_ood = np.asarray(y_pred_ood, dtype=float)
                
                ood_metrics = compute_output_base_metrics(y_pred_ood, y_ood)
            except Exception as e:
                print(f"Error computing OOD predictions or metrics: {e}")
                # ood_metrics = {"mse": float('inf'), "nmse": float('inf'), "r2": -float('inf'), 
                #               "kdt": 0, "mape": float('inf'), "num_valid_points": 0}
        
        total_eval_time += (lambda_prep_time + pred_id_time + metrics_id_time)
        
        if verbose_timing:
            print(f"\n  [Best Result (selected from {len(search_results)} candidates)] Timing breakdown:")
            print(f"    - Lambda function preparation: {lambda_prep_time:.4f}s")
            print(f"    - ID prediction computation: {pred_id_time:.4f}s")
            print(f"    - ID metrics calculation: {metrics_id_time:.4f}s")
        
        # Add evaluation results to detailed log if present
        if detailed_log_data is not None:
            detailed_log_data["evaluation"] = {
                "discovered_equation": equation.expression,
                "id_metrics": id_metrics,
                "ood_metrics": ood_metrics
            }
        
        # Prepare simplified result for results.jsonl
        method = best_result.aux.get("method", method_type)
        best_generation = None
        train_mse = None
        eval_set_name = None
        log_file_ref = None
        
        if detailed_log_data is not None:
            log_file_ref = f"search_logs/{log_file_name}"
            if method == "se_evolution" and "final_results" in detailed_log_data:
                final_results = detailed_log_data["final_results"]
                best_solution_info = final_results.get("global_best") or final_results.get("unique_best")
                if best_solution_info:
                    best_generation = best_solution_info.get("generation")
                    train_mse = best_solution_info.get("mse")
                    eval_set_name = best_solution_info.get("eval_set_name", "validation")
        
        eval_result = {
            'search_time': search_time,
            'discovered_equation': equation.expression,
            'discovered_program': None,
            'id_metrics': id_metrics,
            'ood_metrics': ood_metrics,
            'method': method,
            'selected_from_candidates': len(search_results)  # Number of candidates (should be 1)
        }
        
        # Only add log_file, best_generation, train_mse, and eval_set_name if detailed log exists
        if log_file_ref:
            eval_result['log_file'] = log_file_ref
        if best_generation is not None:
            eval_result['best_generation'] = best_generation
        if train_mse is not None:
            eval_result['train_mse'] = train_mse
        if eval_set_name:
            eval_result['mse_eval_set'] = eval_set_name
        
        eval_results.append(eval_result)
        
        # Write detailed log file (no lock needed, each problem has unique file)
        if detailed_log_data is not None:
            try:
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    json.dump(detailed_log_data, f, indent=4, allow_nan=True, ensure_ascii=False)
                print(f"Detailed log saved to: {log_file_path}")
            except Exception as e:
                print(f"Error writing detailed log file: {e}")
        
        # Prepare simplified log data for results.jsonl
        log_data = {
            'original_idx': problem.original_idx,
            'equation_id': problem.equation_idx,
            'gt_equation': problem.gt_equation.expression,
            'num_train_datapoints': len(problem.train_samples),  # Actual train data used
            'num_valid_datapoints': len(problem.samples.get('valid', [])),  # Validation data used for fitness evaluation during evolution
            'num_eval_datapoints': len(problem.test_samples),  # Test data for final evaluation
            'train_valid_split_ratio': train_valid_split,
            'eval_results': eval_results
        }
        
        # Thread-safe file writing
        with file_lock:
            with open(output_file_path, mode='a') as f:
                f.write(json.dumps(log_data, allow_nan=True) + "\n")
            # Thread-safe update of visited_eqids_set
            visited_eqids_set.add(problem.equation_idx)
        
        timing_stats['total_eval_time'] = total_eval_time
        timing_stats['total_pred_time'] = total_pred_time
        timing_stats['total_metrics_time'] = total_metrics_time
        timing_stats['num_results'] = len(search_results)
        
        if verbose_timing:
            total_time = timing_stats['discover_time'] + timing_stats['data_prep_time'] + timing_stats['total_eval_time']
            print(f"\n{'='*60}")
            print(f"Timing Summary for Problem: {problem.equation_idx}")
            print(f"{'='*60}")
            if total_time > 0:
                print(f"1. Formula Discovery:     {timing_stats['discover_time']:>10.4f}s ({timing_stats['discover_time']/total_time*100:>5.1f}%)")
                print(f"2. Data Preparation:      {timing_stats['data_prep_time']:>10.4f}s ({timing_stats['data_prep_time']/total_time*100:>5.1f}%)")
                print(f"3. Evaluation (Total):    {timing_stats['total_eval_time']:>10.4f}s ({timing_stats['total_eval_time']/total_time*100:>5.1f}%)")
            else:
                print(f"1. Formula Discovery:     {timing_stats['discover_time']:>10.4f}s")
                print(f"2. Data Preparation:      {timing_stats['data_prep_time']:>10.4f}s")
                print(f"3. Evaluation (Total):    {timing_stats['total_eval_time']:>10.4f}s")
            print(f"   ├─ Prediction:         {timing_stats['total_pred_time']:>10.4f}s")
            print(f"   └─ Metrics:            {timing_stats['total_metrics_time']:>10.4f}s")
            print(f"{'─'*60}")
            print(f"Total Time:               {total_time:>10.4f}s")
            print(f"Number of Results:        {timing_stats['num_results']:>10d}")
            print(f"{'='*60}\n")
        
        # Return result for statistics
        if eval_results:
            result = {
                'equation_id': problem.equation_idx,
                'id_metrics': eval_results[0]['id_metrics']
            }
            # Include ood_metrics if available
            if eval_results[0].get('ood_metrics') is not None:
                result['ood_metrics'] = eval_results[0]['ood_metrics']
            return result
        return None
        
    except bdb.BdbQuit:
        raise
    except Exception as e:
        print(f"[Thread {threading.current_thread().name}] Error processing problem {problem.equation_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_se_evaluation(problems: List[Problem], searcher_cfg, output_path: Path, verbose_timing=False, parallel_workers: int = 1, searcher_factory: Optional[Callable[[], SESearcher]] = None, target_original_idxes: Optional[List[int]] = None, train_valid_split: float = 0.8, use_valid_split: Optional[bool] = None):
    
    # Check if evolution mode is enabled
    use_evolution = bool(_get_cfg_value(searcher_cfg, 'use_evolution', True))
    if use_evolution:
        evolution_config = _create_evolution_config_from_cfg(searcher_cfg)
    else:

        evolution_config = None
    print(f"evolution_config: {evolution_config}")
    example_searcher = create_se_searcher(searcher_cfg)
    
    _save_experiment_config(searcher_cfg, example_searcher, evolution_config, output_path)
    
    # Create output file
    output_file_path = output_path / "results.jsonl"
    
    # Track visited problems
    visited_eqids = []
    if output_path.exists():
        visited_eqids = load_visited_problems(output_path)
    
    visited_eqids_set = set(visited_eqids)
    
    # Collect all results for overall statistics
    all_problem_results = []
    
    if target_original_idxes is not None:
        # Filter by original_idx (user-specified)
        print(f"Filtering problems by original_idx: {target_original_idxes}")
        target_original_idxes_set = set(target_original_idxes)
        problems_to_process = []
        
        for _, problem in enumerate(problems):
            # Check if this problem's original_idx is in the target list
            if problem.original_idx is not None and problem.original_idx in target_original_idxes_set:
                if problem.equation_idx not in visited_eqids_set:
                    problems_to_process.append(problem)
                else:
                    print(f"Skipping already visited problem: {problem.equation_idx} (original_idx: {problem.original_idx})")
        
        if not problems_to_process:
            print(f"No unvisited problems found with original_idx in {target_original_idxes}")
            return
        print(f"Found {len(problems_to_process)} problems to process (out of {len(target_original_idxes)} requested)")
    else:
        # Filter out already visited problems
        problems_to_process = []
        for problem in problems:
            if problem.equation_idx not in visited_eqids_set:
                problems_to_process.append(problem)
    
    if not problems_to_process:
        print("All problems have already been processed.")
        return
    
    # Get use_valid_split from searcher_cfg if not provided as parameter
    if use_valid_split is None:
        use_valid_split = bool(_get_cfg_value(searcher_cfg, 'use_valid_split', True))
    
    # Prepare problem with train/valid split, preserving test data
    if use_valid_split:
        print(f"Pre-processing {len(problems_to_process)} problems (splitting train/valid data with ratio {train_valid_split})...")
    else:
        print(f"Pre-processing {len(problems_to_process)} problems (using all data for train, no valid split)...")
    problems_prepared = []
    for problem in problems_to_process:
        problem_prepared = prepare_problem_with_split(problem, train_valid_split, use_valid_split)
        problems_prepared.append(problem_prepared)
    
    print(f"Finished pre-processing all problems. Starting search/evaluation...")
    
    # Thread-safe lock for file writing and visited_eqids_set updates
    file_lock = threading.Lock()
    
    # Process problems sequentially or in parallel
    if parallel_workers > 1:
        # Parallel processing
        print(f"Using parallel processing with {parallel_workers} workers")
        
        # Create searcher factory if not provided
        if searcher_factory is None:

            def default_searcher_factory():
                return create_se_searcher(searcher_cfg)
            searcher_factory = default_searcher_factory
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # Submit all tasks with pre-processed problems
            future_to_problem = {
                executor.submit(
                    process_single_problem_se,
                    problem_prepared,  # Use pre-processed problem with all data in samples dict
                    searcher_factory(),  # Create a new searcher instance for each thread
                    evolution_config,
                    output_path,
                    output_file_path,
                    verbose_timing,
                    file_lock,
                    visited_eqids_set,
                    train_valid_split  # Pass train_valid_split parameter
                ): problem_prepared
                for problem_prepared in problems_prepared
            }
            
            # Collect results as they complete
            completed_eqids = []
            for future in as_completed(future_to_problem):
                problem = future_to_problem[future]
                try:
                    result = future.result()
                    if result:
                        all_problem_results.append(result)
                    completed_eqids.append(problem.equation_idx)
                except Exception as e:
                    print(f"Error processing problem {problem.equation_idx}: {e}")
                    import traceback
                    traceback.print_exc()

            
            # Update visited_eqids_list (already updated in set, but maintain list for consistency)
            visited_eqids.extend(completed_eqids)
    else:
        # Sequential processing (original behavior)
        # Create a single searcher instance for sequential processing

        searcher = create_se_searcher(searcher_cfg) if searcher_factory is None else searcher_factory()
        
        for problem_prepared in problems_prepared:
            result = process_single_problem_se(
                problem_prepared,  # Use pre-processed problem with all data in samples dict
                searcher,
                evolution_config,
                output_path,
                output_file_path,
                verbose_timing,
                file_lock,
                visited_eqids_set,
                train_valid_split  # Pass train_valid_split parameter
            )
            
            if result:
                all_problem_results.append(result)
                visited_eqids.append(problem_prepared.equation_idx)
    
    # Calculate overall statistics
    if all_problem_results:
        # Helper function to safely compute mean, filtering out inf and nan
        def safe_mean(values, metric_name=''):
            if not values:
                return None
            # Convert to numpy array for easier handling
            values_array = np.array(values, dtype=float)
            # Filter out NaN and Inf values
            valid_mask = ~(np.isnan(values_array) | np.isinf(values_array))
            valid_values = values_array[valid_mask]
            if len(valid_values) == 0:
                return None
            return float(np.mean(valid_values))
        
        # Collect all metrics for ID
        id_mse_values = []
        id_nmse_values = []
        id_r2_values = []
        id_kdt_values = []
        id_mape_values = []
        id_acc_tau_values = []
        id_acc_tau_max_values = []
        
        # Collect all metrics for OOD (if available)
        ood_mse_values = []
        ood_nmse_values = []
        ood_r2_values = []
        ood_kdt_values = []
        ood_mape_values = []
        ood_acc_tau_values = []
        ood_acc_tau_max_values = []
        
        has_ood_metrics = False
        
        for result in all_problem_results:
            # Process ID metrics
            id_metrics = result['id_metrics']
            if 'mse' in id_metrics:
                id_mse_values.append(id_metrics['mse'])
            if 'nmse' in id_metrics:
                id_nmse_values.append(id_metrics['nmse'])
            if 'r2' in id_metrics:
                id_r2_values.append(id_metrics['r2'])
            if 'kdt' in id_metrics:
                id_kdt_values.append(id_metrics['kdt'])
            if 'mape' in id_metrics:
                id_mape_values.append(id_metrics['mape'])
            if 'acc_tau' in id_metrics:
                id_acc_tau_values.append(id_metrics['acc_tau'])
            if 'acc_tau_max' in id_metrics:
                id_acc_tau_max_values.append(id_metrics['acc_tau_max'])
            
            # Process OOD metrics if available
            if 'ood_metrics' in result and result['ood_metrics'] is not None:
                has_ood_metrics = True
                ood_metrics = result['ood_metrics']
                if 'mse' in ood_metrics:
                    ood_mse_values.append(ood_metrics['mse'])
                if 'nmse' in ood_metrics:
                    ood_nmse_values.append(ood_metrics['nmse'])
                if 'r2' in ood_metrics:
                    ood_r2_values.append(ood_metrics['r2'])
                if 'kdt' in ood_metrics:
                    ood_kdt_values.append(ood_metrics['kdt'])
                if 'mape' in ood_metrics:
                    ood_mape_values.append(ood_metrics['mape'])
                if 'acc_tau' in ood_metrics:
                    ood_acc_tau_values.append(ood_metrics['acc_tau'])
                if 'acc_tau_max' in ood_metrics:
                    ood_acc_tau_max_values.append(ood_metrics['acc_tau_max'])
        
        # Calculate overall statistics
        overall_stats = {
            'total_problems': len(all_problem_results),
            'id_metrics': {
                'mse': safe_mean(id_mse_values, 'mse'),
                'nmse': safe_mean(id_nmse_values, 'nmse'),
                'r2': safe_mean(id_r2_values, 'r2'),
                'kdt': safe_mean(id_kdt_values, 'kdt'),
                'mape': safe_mean(id_mape_values, 'mape'),
                'acc_tau': safe_mean(id_acc_tau_values, 'acc_tau') if id_acc_tau_values else None,
                'acc_tau_max': safe_mean(id_acc_tau_max_values, 'acc_tau_max') if id_acc_tau_max_values else None
            },
            'id_success_count': {
                'r2_positive': sum(1 for r2 in id_r2_values if not (np.isnan(r2) or np.isinf(r2)) and r2 > 0),
                'acc_tau_1': sum(1 for acc in id_acc_tau_values if acc == 1.0) if id_acc_tau_values else 0,
                'acc_tau_max_1': sum(1 for acc in id_acc_tau_max_values if acc == 1.0) if id_acc_tau_max_values else 0
            }
        }
        
        # Add OOD metrics if available
        if has_ood_metrics:
            overall_stats['ood_metrics'] = {
                'mse': safe_mean(ood_mse_values, 'mse'),
                'nmse': safe_mean(ood_nmse_values, 'nmse'),
                'r2': safe_mean(ood_r2_values, 'r2'),
                'kdt': safe_mean(ood_kdt_values, 'kdt'),
                'mape': safe_mean(ood_mape_values, 'mape'),
                'acc_tau': safe_mean(ood_acc_tau_values, 'acc_tau') if ood_acc_tau_values else None,
                'acc_tau_max': safe_mean(ood_acc_tau_max_values, 'acc_tau_max') if ood_acc_tau_max_values else None
            }
            overall_stats['ood_success_count'] = {
                'r2_positive': sum(1 for r2 in ood_r2_values if not (np.isnan(r2) or np.isinf(r2)) and r2 > 0),
                'acc_tau_1': sum(1 for acc in ood_acc_tau_values if acc == 1.0) if ood_acc_tau_values else 0,
                'acc_tau_max_1': sum(1 for acc in ood_acc_tau_max_values if acc == 1.0) if ood_acc_tau_max_values else 0
            }
        
        # Save overall results
        all_results_path = output_path / "all_results.json"
        with open(all_results_path, mode='w') as f:
            json.dump(overall_stats, f, indent=2, allow_nan=True)
        
        print(f"\nOverall statistics saved to {all_results_path}")
        print(f"Total problems: {overall_stats['total_problems']}")
        print(f"\nID Metrics:")
        print(f"  Average R2: {overall_stats['id_metrics']['r2']}")
        print(f"  Average NMSE: {overall_stats['id_metrics']['nmse']}")
        if overall_stats['id_metrics']['acc_tau'] is not None:
            print(f"  Average Acc_tau: {overall_stats['id_metrics']['acc_tau']}")
        if overall_stats['id_metrics']['acc_tau_max'] is not None:
            print(f"  Average Acc_tau_max: {overall_stats['id_metrics']['acc_tau_max']}")
        
        if has_ood_metrics:
            print(f"\nOOD Metrics:")
            print(f"  Average R2: {overall_stats['ood_metrics']['r2']}")
            print(f"  Average NMSE: {overall_stats['ood_metrics']['nmse']}")
            if overall_stats['ood_metrics']['acc_tau'] is not None:
                print(f"  Average Acc_tau: {overall_stats['ood_metrics']['acc_tau']}")
            if overall_stats['ood_metrics']['acc_tau_max'] is not None:
                print(f"  Average Acc_tau_max: {overall_stats['ood_metrics']['acc_tau_max']}")

