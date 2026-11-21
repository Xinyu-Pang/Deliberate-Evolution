import re
import time
import copy
import json
import logging
import sys
import bdb
import numpy as np
import asyncio
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from bench.dataclasses import Problem, Equation, SearchResult
from methods.se.client import LLMClient
from methods.se.evolution import EvolvedSolution, SEExperienceBuffer, Island
from methods.se.optimization import optimize_expression_parameters
from methods.se.tool_group import SymbolicRegressionToolGroup
from methods.se.memory import MemoryManager
from methods.se.config import SEEvolutionConfig, MemoryConfig

# Create logger instance
log = logging.getLogger(__name__)

# Color log configuration (optional, requires colorlog package)
LOG_COLORS_CONFIG = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}

def configure_color_logging(log_level: int = logging.INFO):
    """
    Configure root logging with colorized output.
    
    Args:
        log_level: Logging level (default: logging.INFO)
    """
    try:
        import colorlog
        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
            log_colors=LOG_COLORS_CONFIG,
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)
        # Disable httpx and openai library's internal HTTP request logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
    except ImportError:
        # Fallback to standard logging if colorlog is not available
        logging.basicConfig(
            level=log_level,
            format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s"
        )
        # Disable httpx and openai library's internal HTTP request logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        # Use print here to avoid circular dependency during logging setup
        print("Warning: colorlog not available. Using standard logging. Install with: pip install colorlog")

# Try to import latex2sympy2 for LaTeX parsing
try:
    from latex2sympy2 import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    LATEX2SYMPY_AVAILABLE = False
    log.warning("latex2sympy2 not available. LaTeX parsing will be disabled.")
    log.warning("Install it with: pip install latex2sympy2")

# Import SymPy for expression validation
try:
    from sympy import symbols, parse_expr
    from sympy.parsing.sympy_parser import (
        standard_transformations,
        implicit_multiplication,
        convert_xor
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    log.warning("sympy not available. Variable validation will be limited.")


# ==================== Adaptive Strategy Table ====================

class AdaptiveStrategyTable:
    
    def __init__(self, alpha: float = 0.1, rank_percent_threshold: float = 20, visit_threshold: int = 3, 
                 visit_threshold_scale: float = 15.0, refine_suppression_factor: float = 0.05,
                 regenerate_boost: float = 0.25, min_probability: float = 0.05, 
                 max_reward_clip: float = 2.0, min_decay_factor: float = 0.1):
        self.alpha = alpha
        self.rank_percent_threshold = rank_percent_threshold
        self.base_visit_threshold = visit_threshold
        self.visit_threshold_scale = visit_threshold_scale
        self.refine_suppression_factor = refine_suppression_factor
        self.regenerate_boost = regenerate_boost
        self.min_probability = min_probability
        self.max_reward_clip = max_reward_clip
        self.min_decay_factor = min_decay_factor
        self._lock = threading.Lock()
        
        self.strategy_table = {
            "S1": {"refine": 0.6, "mutate": 0.2, "crossover": 0.1, "regenerate": 0.1},
            "S2": {"refine": 0.1, "mutate": 0.3, "crossover": 0.4, "regenerate": 0.2},
            "S3": {"refine": 0.3, "mutate": 0.5, "crossover": 0.1, "regenerate": 0.1},
            "S4": {"refine": 0.05, "mutate": 0.05, "crossover": 0.1, "regenerate": 0.8}
        }
        
        self._initial_strategy_table = copy.deepcopy(self.strategy_table)
    
    def _compute_dynamic_visit_threshold(self, population_size: int) -> float:
        import math
        dynamic_threshold = self.base_visit_threshold + math.log(population_size + 1) * self.visit_threshold_scale
        return max(self.base_visit_threshold, dynamic_threshold)
    
    def get_state(self, solution: EvolvedSolution, island: Island) -> str:
        rank = island.get_rank(solution)
        if rank is None:
            return "S3"
        
        population_size = len(island.population)
        if population_size == 0:
            return "S3"
        
        visit_threshold = self._compute_dynamic_visit_threshold(population_size)
        
        rank_percent = (rank / population_size) * 100
        visit = solution.selection_count
        
        if rank_percent <= self.rank_percent_threshold and visit < visit_threshold:
            return "S1"
        elif rank_percent <= self.rank_percent_threshold and visit >= visit_threshold:
            return "S2"
        elif rank_percent > self.rank_percent_threshold and visit < visit_threshold:
            return "S3"
        else:  # rank_percent > self.rank_percent_threshold and visit >= visit_threshold
            return "S4"
    
    def _apply_masking(self, probs: Dict[str, float], is_stagnated: bool) -> Dict[str, float]:
        if not is_stagnated:
            return probs
        
        modified_probs = probs.copy()
        
        modified_probs["refine"] *= self.refine_suppression_factor
        
        modified_probs["regenerate"] += self.regenerate_boost
        
        
        return modified_probs
    
    def select_operation(self, state: str, is_stagnated: bool = False) -> str:
        with self._lock:
            raw_probs = self.strategy_table.get(state, self._initial_strategy_table["S1"]).copy()
            
            final_probs = self._apply_masking(raw_probs, is_stagnated)
            
            ops = list(final_probs.keys())
            vals = np.array(list(final_probs.values()))
            
            vals = np.maximum(vals, 0.0)
            
            total = np.sum(vals)
            if total > 0:
                p_normalized = vals / total
            else:
                p_normalized = np.ones(len(ops)) / len(ops)
            
            return np.random.choice(ops, p=p_normalized)
    
    def update_strategy(self, state: str, op: str, reward: float):
        with self._lock:
            if state not in self.strategy_table or op not in self.strategy_table[state]:

                log.warning("Update skipped: Invalid state or operation: state=%s, op=%s", state, op)
                return
            
            current_prob = self.strategy_table[state][op]
            
            if reward > 0:
                safe_reward = min(reward, self.max_reward_clip)
                new_prob = current_prob * (1 + self.alpha * safe_reward)
            else:
                decay_factor = max(self.min_decay_factor, 1 - self.alpha * abs(reward))
                new_prob = current_prob * decay_factor
            
            self.strategy_table[state][op] = new_prob
            
            vals = np.array(list(self.strategy_table[state].values()))
            keys = list(self.strategy_table[state].keys())
            
            vals = np.maximum(vals, 0.0)
            total = np.sum(vals)
            
            if total > 0:
                probs = vals / total
                
                n_ops = len(keys)
                
                epsilon = min(1.0, n_ops * self.min_probability)
                
                for i, k in enumerate(keys):
                    self.strategy_table[state][k] = (1 - epsilon) * probs[i] + epsilon / n_ops
            else:
                initial_probs = self._initial_strategy_table.get(state, None)
                if initial_probs:
                    self.strategy_table[state] = copy.deepcopy(initial_probs)
                else:
                    n_ops = len(keys)
                    for k in keys:
                        self.strategy_table[state][k] = 1.0 / n_ops
    
    def get_strategy_table(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {state: probs.copy() for state, probs in self.strategy_table.items()}


# ==================== SESearcher Class ====================

class SESearcher:
    """Simple LLM-based Self-Evolving searcher"""
    
    def __init__(self, name: str, api_type: str, api_model: str,
                 n: int = 1,
                 temperature: float = 0.8,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 rank_percent_threshold: float = 20,
                 visit_threshold: int = 3,
                 visit_threshold_scale: float = 15.0,
                 enable_color_logging: bool = True,
                 log_level: int = logging.DEBUG,
                 enable_tools: bool = True,
                 max_tool_rounds: int = 4,
                 strategy_alpha: float = 0.1,
                 strategy_refine_suppression_factor: float = 0.05,
                 strategy_regenerate_boost: float = 0.25,
                 strategy_min_probability: float = 0.05,
                 strategy_max_reward_clip: float = 2.0,
                 strategy_min_decay_factor: float = 0.1):
        self.name = name
        self.api_type = api_type
        self.model = api_model
        
        client_api_type = "openai" if api_type == "openai" else "vllm"
        
        # Configure logging (color logging if requested)
        if enable_color_logging:
            configure_color_logging(log_level)
        else:
            # Ensure basic logging is configured
            if not logging.getLogger().handlers:
                logging.basicConfig(
                    level=log_level,
                    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s"
                )
        
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        
        if api_key is None:
            raise ValueError("api_key must be provided in searcher_cfg")
        if base_url is None:
            raise ValueError("base_url (or api_url) must be provided in searcher_cfg")
        
        self.client = LLMClient(
            api_type=client_api_type,
            api_key=api_key,
            base_url=base_url,
            model=api_model,
            track_tokens=True
        )
        
        self.n = n
        self.temperature = temperature
        self.stop_tokens = None
        
        # Built-in functions and constants for variable validation
        self.builtin_functions = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 
                                  'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'ln'}
        self.math_constants = {'pi', 'e', 'E', 'I', 'oo', 'zoo', 'nan', 'inf'}
        
        # Variable name mapping for prompt substitution (to avoid LaTeX parsing issues)
        # Maps original variable names to temporary names (e.g., "E_f" -> "Ef")
        self._var_mapping = {}  # original -> temporary
        self._reverse_mapping = {}  # temporary -> original
        
        self.strategy_table = AdaptiveStrategyTable(
            alpha=strategy_alpha,
            rank_percent_threshold=rank_percent_threshold,
            visit_threshold=visit_threshold,
            visit_threshold_scale=visit_threshold_scale,
            refine_suppression_factor=strategy_refine_suppression_factor,
            regenerate_boost=strategy_regenerate_boost,
            min_probability=strategy_min_probability,
            max_reward_clip=strategy_max_reward_clip,
            min_decay_factor=strategy_min_decay_factor
        )
        
        self.enable_tools = enable_tools
        self.max_tool_rounds = max_tool_rounds
        
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        
        self._tool_group: Optional[SymbolicRegressionToolGroup] = None
        self._memory_manager: Optional[MemoryManager] = None
        self._memory_section: str = ""
        
        if enable_tools:
            prompt_file = "methods/se/prompts/tool_system_prompt.txt"
        else:

            prompt_file = "methods/se/prompts/refine_system_prompt.txt"
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                self._system_prompt_template = f.read().strip()
        except Exception as e:
            log.error("Error reading system prompt file: %s", e, exc_info=True)

            self._system_prompt_template = ""
        
        self._tool_descriptions = {
            "analyze_data": "1. **`analyze_data`**\n   - **Purpose**: Analyzes the statistical properties and distribution of the training data. Call this if you need to understand variable ranges or data patterns.",
            "analyze_residuals": "2. **`analyze_residuals`**\n   - **Purpose**: Evaluates fitting performance and residual patterns of the parent solution.",
            "dimensional_analysis": "3. **`dimensional_analysis`**\n   - **Purpose**: Performs dimensional analysis on an expression to validate its dimensional consistency."
        }
        
        self.TAG_ANSWER = "answer"
        self.TAG_TOOL_CODE = "tool"
        # self.TAG_THINK = "think"
        self.TAG_THINK = "reasoning"
        self.TAG_REDACTED_REASONING = "redacted_reasoning"
        
        self._tools_section_header = """### Available Tools
You have access to specific tools to assist your reasoning. The system will automatically execute these tools based on your current reasoning context.
"""
        
        self._tool_call_format_section = f"""### Tool Call Format\n- To use a tool: <{self.TAG_TOOL_CODE}>function_name</{self.TAG_TOOL_CODE}>"""
        
        self._response_format_section = f"""### Output Format
You should first conduct reasoning process. This process MUST be enclosed within <{self.TAG_THINK}> </{self.TAG_THINK}> tags.
After you finish your reasoning, you MUST wrap your final solution inside <{self.TAG_ANSWER}>\\boxed{{expression}}</{self.TAG_ANSWER}>."""
        
        self._response_format_section_with_tool = f"""### Output Format
1. **Reasoning**: Always think inside <{self.TAG_THINK}>...</{self.TAG_THINK}> first to determine your next step.
After completing your reasoning, choose ONLY ONE of the following actions (do not perform both):
(1) <{self.TAG_TOOL_CODE}>function_name</{self.TAG_TOOL_CODE}> (Use this if you need more info. Do NOT generate the result yourself.)
(2) <{self.TAG_ANSWER}>\\boxed{{expression}}</{self.TAG_ANSWER}> (Use this only when you are confident in the solution.)"""
        
        if not enable_tools:
            self._system_prompt = None
        else:
            self._system_prompt = None

        if enable_tools:
            self._user_prompt_suffix_base = f"""Now it's your turn to respond for the current step.\nImportant: If tool interaction history is provided above, you should use those results for your reasoning."""
            self._user_prompt_suffix_with_tool = f"{self._user_prompt_suffix_base}\n{self._response_format_section_with_tool}"
            self._user_prompt_suffix_no_tool = f"{self._user_prompt_suffix_base}\n{self._response_format_section}"
        else:
            self._user_prompt_suffix_base = f"Now it's your turn to respond for the current step.\n{self._response_format_section}"
            self._user_prompt_suffix_with_tool = None
            self._user_prompt_suffix_no_tool = None
        
        self.PATTERN_ANSWER = rf"<{self.TAG_ANSWER}>(.*?)</{self.TAG_ANSWER}>"
        self.PATTERN_TOOL_CODE = rf"<{self.TAG_TOOL_CODE}>(.*?)</{self.TAG_TOOL_CODE}>"
        self.PATTERN_THINK = rf"<{self.TAG_THINK}>(.*?)</{self.TAG_THINK}>"
        self.PATTERN_REDACTED_REASONING = rf"<{self.TAG_REDACTED_REASONING}>(.*?)</{self.TAG_REDACTED_REASONING}>"
    
    def _build_local_dict(self, problem: Problem, additional_symbols: Optional[List[str]] = None) -> Dict:
        """
        Build a local_dict for parse_expr to treat specified names as symbols instead of functions.
        
        Args:
            problem: Problem object containing variable definitions
            additional_symbols: Additional symbol names that might conflict with SymPy functions
        
        Returns:
            Dictionary mapping symbol names to SymPy Symbol objects
        """
        from sympy import Symbol
        
        local_dict = {}
        
        # Add all variables (both input and output) as symbols
        all_vars = problem.gt_equation.symbols
        for var in all_vars:
            local_dict[var] = Symbol(var)
        
        # Add additional symbols that might conflict with SymPy built-ins
        # Common conflicts: gamma (Gamma function), lambda (Python keyword), I (imaginary unit), etc.
        conflict_symbols = additional_symbols or []
        common_conflicts = ['gamma', 'lambda', 'I', 'beta', 'alpha', 'omega', 'zeta', 'psi']
        
        for sym_name in conflict_symbols + common_conflicts:
            if sym_name not in local_dict:  # Don't override input vars
                local_dict[sym_name] = Symbol(sym_name)
        
        return local_dict
    
    def _create_variable_mapping(self, problem: Problem) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Create variable name mapping to replace underscores with temporary names.
        This helps avoid LaTeX parsing issues with subscript notation.
        
        Args:
            problem: Problem object containing variable definitions
            
        Returns:
            Tuple of (var_mapping, reverse_mapping):
            - var_mapping: Maps original names to temporary names (e.g., {"E_f": "Ef"})
            - reverse_mapping: Maps temporary names back to original (e.g., {"Ef": "E_f"})
        """
        var_mapping = {}
        reverse_mapping = {}
        
        # Process all variables (both input and output)
        for var_name in problem.gt_equation.symbols:
            if '_' in var_name:
                # Replace underscore with temporary name (e.g., "E_f" -> "Ef")
                temp_name = var_name.replace('_', '')
                
                # Check for conflicts (temporary name should not conflict with existing variables)
                all_vars = set(problem.gt_equation.symbols)
                if temp_name not in all_vars:
                    var_mapping[var_name] = temp_name
                    reverse_mapping[temp_name] = var_name
        
        return var_mapping, reverse_mapping
    
    def _restore_variable_names(self, text: str, reverse_mapping: Dict[str, str]) -> str:
        """
        Restore original variable names from temporary names.
        
        Args:
            text: Text containing temporary variable names
            reverse_mapping: Reverse mapping (temporary -> original)
            
        Returns:
            Text with original variable names restored
        """
        result = text
        for temporary, original in reverse_mapping.items():
            # Use word boundary to avoid partial matches
            pattern = rf'\b{re.escape(temporary)}\b'
            result = re.sub(pattern, original, result)
        return result

    def _build_dynamic_system_prompt(self, called_tools: set) -> Tuple[str, set]:
        all_tools = set(self._tool_descriptions.keys())
        
        available_tools = all_tools - called_tools
        
        if not available_tools:
            tools_section = ""
        else:
            tool_list = []
            tool_num = 1
            for tool_name in sorted(available_tools):
                tool_desc = self._tool_descriptions[tool_name]
                lines = tool_desc.split('\n')
                if lines:
                    lines[0] = re.sub(r'^\d+\.', f'{tool_num}.', lines[0])
                    tool_desc = '\n'.join(lines)
                tool_list.append(tool_desc)
                tool_num += 1
            
            tools_section = self._tools_section_header + "\n".join(tool_list) + "\n" + self._tool_call_format_section
        
        system_prompt = self._system_prompt_template.replace("{TOOLS_SECTION}", tools_section)
        
        memory_section = self._memory_section if self._memory_section else ""
        system_prompt = system_prompt.replace("{MEMORY_SECTION}", memory_section)
        
        if available_tools:
            system_prompt = system_prompt + "\n\n" + self._response_format_section_with_tool
        else:
            system_prompt = system_prompt + "\n\n" + self._response_format_section
        
        return system_prompt, available_tools
    
    def _get_user_prompt_suffix(self, available_tools: set = None) -> str:
        if not self.enable_tools:
            return self._user_prompt_suffix_base
        
        if available_tools and len(available_tools) > 0:
            return self._user_prompt_suffix_with_tool
        else:
            return self._user_prompt_suffix_no_tool

    def _collect_top_solutions(self, buffer: SEExperienceBuffer, top_k: int) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        solutions = []
        with buffer._lock:
            for island in buffer._islands:
                solutions.extend(island.population[:top_k])
        solutions = sorted(solutions, key=lambda s: s.mse)
        top = solutions[:top_k]
        return [
            {
                "expression": sol.expression,
                "mse": sol.mse
            }
            for sol in top
        ]
    
    def _collect_top_solution_objects(self, buffer: SEExperienceBuffer, top_k: int) -> List[EvolvedSolution]:
        if top_k <= 0:
            return []
        solutions = []
        with buffer._lock:
            for island in buffer._islands:
                solutions.extend(island.population[:top_k])
        solutions = sorted(solutions, key=lambda s: s.mse)
        return solutions[:top_k]
    
    def _collect_failed_mutations(
        self, 
        buffer: SEExperienceBuffer,
        stagnation_start_generation: Optional[int] = None,
        max_count: int = 3,
        parent_top_percent: float = 0.2
    ) -> List[Dict[str, Any]]:
        if max_count <= 0:
            return []
        
        failed = []
        with buffer._lock:
            for island in buffer._islands:
                if len(island.population) == 0:
                    continue
                
                top_percent_threshold = max(1, int(len(island.population) * parent_top_percent))
                
                for child_solution in island.population:
                    if stagnation_start_generation is not None:
                        if child_solution.generation < stagnation_start_generation:
                            continue
                    
                    if child_solution.parent_id is None:
                        continue
                    
                    parent_solution = island.find_solution_by_id(child_solution.parent_id)
                    if parent_solution is None:
                        continue
                    
                    if child_solution.mse < parent_solution.mse:
                        continue
                    
                    parent_rank = island.get_rank(parent_solution)
                    if parent_rank is None or parent_rank >= top_percent_threshold:
                        continue
                    
                    if parent_solution.mse <= 0:
                        continue
                    mse_increase_ratio = (child_solution.mse - parent_solution.mse) / parent_solution.mse
                    
                    failed.append({
                        "parent_expression": parent_solution.expression,
                        "parent_mse": parent_solution.mse,
                        "parent_rank": parent_rank,
                        "child_expression": child_solution.expression,
                        "child_mse": child_solution.mse,
                        "generation": child_solution.generation,
                        "mse_increase_ratio": mse_increase_ratio,
                    })
        
        failed.sort(key=lambda x: x.get("mse_increase_ratio", 0), reverse=True)
        return failed[:max_count]

    def _collect_bad_regressions(self, buffer: SEExperienceBuffer, top_f: int) -> List[Dict[str, Any]]:
        if top_f <= 0:
            return []
        
        bad = []
        with buffer._lock:
            for island in buffer._islands:
                if len(island.population) == 0:
                    continue
                
                top_10_percent_threshold = max(1, int(len(island.population) * 0.1))
                
                for child_solution in island.population:
                    if child_solution.parent_id is None:
                        continue
                    
                    parent_solution = island.find_solution_by_id(child_solution.parent_id)
                    if parent_solution is None:
                        continue
                    
                    parent_rank = island.get_rank(parent_solution)
                    if parent_rank is None:
                        continue
                    
                    if parent_rank >= top_10_percent_threshold:
                        continue
                    
                    if parent_solution.mse <= 0:
                        continue
                    
                    mse_ratio = child_solution.mse / parent_solution.mse
                    if mse_ratio > 100:
                        delta_mse = child_solution.mse - parent_solution.mse
                        bad.append(
                            {
                                "expression": child_solution.expression,
                                "mse": child_solution.mse,
                                "parent_mse": parent_solution.mse,
                                "delta_mse": delta_mse,
                                "mse_ratio": mse_ratio,
                                "parent_rank": parent_rank,
                                "island_id": island.island_id,
                            }
                        )
        
        if len(bad) <= top_f:
            return bad
        else:
            indices = np.random.choice(len(bad), size=top_f, replace=False)
            return [bad[i] for i in indices]

    @staticmethod
    def _parse_insights(raw_text: str, limit: int) -> List[str]:
        lines = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            stripped = stripped.lstrip("-•")
            stripped = stripped.lstrip("0123456789. ").strip()
            if stripped:
                lines.append(stripped)
        return lines[:limit]

    def _analyze_residuals_for_top2(
        self,
        top2_solutions: List[EvolvedSolution],
        problem: Problem
    ) -> str:
        if len(top2_solutions) == 0:

            return "No solutions available for residual analysis."
        
        try:
            if self._tool_group is None:
                self._tool_group = SymbolicRegressionToolGroup(problem, searcher=self)
            
            results = []
            for idx, solution in enumerate(top2_solutions):
                try:
                    analysis_result = self._tool_group.analyze_residuals(parent_solutions=[solution])
                    solution_header = f"=== Solution {idx + 1} (MSE: {solution.mse:.6e}, Expression: {solution.expression}) ===\n"
                    results.append(solution_header + analysis_result)
                except Exception as e:
                    log.warning("Residual analysis failed for solution %d: %s", idx + 1, e, exc_info=True)
                    results.append(f"=== Solution {idx + 1} ===\nResidual analysis failed: {str(e)}")
            
            return "\n\n".join(results)
        except Exception as e:

            log.warning("Residual analysis failed: %s", e, exc_info=True)
            return f"Residual analysis failed: {str(e)}"
    
    async def _async_summarize_normal_memory(
        self,
        memory_manager: MemoryManager,
        problem: Problem,
        good_samples: List[Dict[str, Any]],
        bad_samples: List[Dict[str, Any]],
        generation: int,
        trigger_type: str,
        improvement_mutation: Optional[Dict[str, Any]] = None,
        debug_path: Optional[Path] = None,
    ) -> List[str]:
        if not memory_manager.cfg.enabled:
            return []
        
        # Build problem description with variable descriptions
        symbols = problem.gt_equation.symbols
        symbol_descs = problem.gt_equation.symbol_descs
        
        output_var = symbols[0]
        output_desc = symbol_descs[0]
        input_vars = symbols[1:]
        input_descs = symbol_descs[1:]
        
        # Build input variables list with format: "var1 (desc1), var2 (desc2), ..."
        input_vars_list = []
        for var, desc in zip(input_vars, input_descs):
            input_vars_list.append(f"{var} ({desc})")
        
        # Join with commas, add "and" before the last item if there are multiple items
        if len(input_vars_list) > 1:
            input_vars_str = ", ".join(input_vars_list[:-1]) + ", and " + input_vars_list[-1]
        else:
            input_vars_str = input_vars_list[0] if input_vars_list else ""
        
        problem_desc = f"{output_var} ({output_desc}) as a function of {input_vars_str}"
        prompt_text = memory_manager.build_normal_summary_prompt(
            problem_desc, 
            good_samples, 
            bad_samples,
            improvement_mutation=improvement_mutation
        )
        messages = [
            {"role": "user", "content": prompt_text},
        ]
        try:
            response = await self.client.async_call_llm(
                messages,
                temperature=0.6,
                extra_body=self.extra_body,
            )
            
            if debug_path is not None:
                await asyncio.to_thread(
                    self._save_memory_debug,
                    debug_path,
                    generation,
                    trigger_type,
                    messages,
                    response,
                    prompt_text,
                    good_samples,
                    bad_samples,
                )
            
            return self._parse_insights(response, memory_manager.cfg.insights_n)
        except Exception as e:

            log.warning("Normal memory summarization failed: %s", e, exc_info=True)
            return []
    
    async def _async_summarize_stagnation_memory(
        self,
        memory_manager: MemoryManager,
        problem: Problem,
        top_k_samples: List[Dict[str, Any]],
        top_n_solutions: List[EvolvedSolution],
        failed_mutations: List[Dict[str, Any]],
        generation: int,
        debug_path: Optional[Path] = None,
    ) -> List[str]:
        if not memory_manager.cfg.enabled:
            return []
        
        # Build problem description
        symbols = problem.gt_equation.symbols
        symbol_descs = problem.gt_equation.symbol_descs
        
        output_var = symbols[0]
        output_desc = symbol_descs[0]
        input_vars = symbols[1:]
        input_descs = symbol_descs[1:]
        
        input_vars_list = []
        for var, desc in zip(input_vars, input_descs):
            input_vars_list.append(f"{var} ({desc})")
        
        if len(input_vars_list) > 1:
            input_vars_str = ", ".join(input_vars_list[:-1]) + ", and " + input_vars_list[-1]
        else:
            input_vars_str = input_vars_list[0] if input_vars_list else ""
        
        problem_desc = f"{output_var} ({output_desc}) as a function of {input_vars_str}"
        
        top_n_residual_analysis = self._analyze_residuals_for_top2(top_n_solutions, problem)
        
        prompt_text = memory_manager.build_stagnation_summary_prompt(
            problem_desc,
            top_k_samples,
            top_n_residual_analysis,
            failed_mutations,
            residual_analysis_count=len(top_n_solutions)
        )
        messages = [
            {"role": "user", "content": prompt_text},
        ]
        try:
            response = await self.client.async_call_llm(
                messages,
                temperature=0.6,
                extra_body=self.extra_body,
            )
            
            if debug_path is not None:
                await asyncio.to_thread(
                    self._save_memory_debug,
                    debug_path,
                    generation,
                    "stagnation",
                    messages,
                    response,
                    prompt_text,
                    top_k_samples,
                    failed_mutations,
                )
            
            return self._parse_insights(response, memory_manager.cfg.insights_n)
        except Exception as e:
            log.warning("Stagnation memory summarization failed: %s", e, exc_info=True)
            return []
    
    
    def _save_memory_debug(
        self,
        debug_path: Path,
        generation: int,
        trigger_type: str,
        messages: List[Dict[str, Any]],
        response: str,
        prompt_text: str,
        *args,
        **kwargs
    ) -> None:
        try:
            debug_data = {
                "generation": generation,
                "trigger_type": trigger_type,
                "prompt_text": prompt_text,
                "response": response,
            }
            
            if len(args) >= 2:
                if trigger_type == "stagnation":
                    debug_data["top_k_samples_count"] = len(args[0]) if args[0] else 0
                    debug_data["failed_mutations_count"] = len(args[1]) if args[1] else 0
                else:
                    debug_data["good_samples_count"] = len(args[0]) if args[0] else 0
                    debug_data["bad_samples_count"] = len(args[1]) if args[1] else 0
            
            filename = debug_path / f"generation_{generation:03d}_{trigger_type}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
        except Exception as e:

            log.warning("Failed to save memory debug info: %s", e, exc_info=True)

    def _clean_latex(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace('$$', '').replace('$', '').strip()
        return text

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """
        Extract content from \\boxed{} command, handling nested braces correctly.
        Uses brace counting to properly handle nested structures like \\frac{a}{b}.
        
        Args:
            text: Text containing \\boxed{} command
            
        Returns:
            Content inside \\boxed{} or None if not found
        """
        # Find the last occurrence of \boxed (most likely to be the final answer)
        idx = text.rfind("\\boxed")
        if idx < 0:
            return None
        
        # Find the opening brace after \boxed
        # Start searching from after "\\boxed" (6 characters)
        i = idx + 6
        left_brace_idx = None
        right_brace_idx = None
        
        # Skip whitespace and find the opening brace
        while i < len(text) and i < idx + 20:  # Allow some space for whitespace
            if text[i] == "{":
                left_brace_idx = i
                break
            elif not text[i].isspace():
                # If we hit a non-whitespace, non-brace character, no boxed found
                return None
            i += 1
        
        if left_brace_idx is None:
            return None
        
        # Count braces to find the matching closing brace
        num_left_braces_open = 1
        i = left_brace_idx + 1
        
        while i < len(text):
            if text[i] == "{":
                num_left_braces_open += 1
            elif text[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        
        if right_brace_idx is None:
            return None
        
        # Extract content between braces
        return text[left_brace_idx + 1 : right_brace_idx].strip()
    
    def _clean_reasoning_tags(self, text: str) -> str:
        text = text.strip()
        # Remove in order: redacted_reasoning, think, tool
        text = re.sub(self.PATTERN_REDACTED_REASONING, '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(self.PATTERN_THINK, '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(self.PATTERN_TOOL_CODE, '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()
    
    def _extract_expression_part(self, response: str, problem: Problem) -> Optional[str]:
        # Pre-processing: Remove all reasoning/tool tags (think, redacted_reasoning, tool)
        response = self._clean_reasoning_tags(response)
        
        # 1. Look for <answer> tag
        answer_blocks = re.findall(self.PATTERN_ANSWER, response, flags=re.DOTALL | re.IGNORECASE)
        if answer_blocks:
            candidate_text = answer_blocks[-1].strip()
            # Look for \boxed{} in candidate_text
            boxed_matches = self._extract_boxed_content(candidate_text)
            if boxed_matches:
                # Found -> Return Box content (this is the most perfect case)
                return self._clean_latex(boxed_matches)
            else:
                cleaned_text = self._clean_latex(candidate_text)
                if cleaned_text:
                    return cleaned_text
        
        # 2. Look for bare \boxed{} (fallback)
        boxed_matches = self._extract_boxed_content(response)
        if boxed_matches:
            return self._clean_latex(boxed_matches)
        
        # Failure: Return None
        return None
    
    def _normalize_latex_vars_for_parsing(self, latex_str: str, problem: Problem) -> str:
        if not self._var_mapping:
            return latex_str
        
        normalized_latex = latex_str
        
        # Process each variable mapping
        for original_name, temp_name in self._var_mapping.items():
            if '_' in original_name:
                # Split variable name: e.g., "E_f" -> ["E", "f"]
                parts = original_name.split('_')
                if len(parts) == 2:
                    base, subscript = parts[0], parts[1]
                    
                    # Pattern 1: E_f (simple subscript, no braces)
                    # Match in LaTeX context, considering word boundaries and LaTeX syntax
                    pattern1 = rf'(?<![a-zA-Z0-9\\]){re.escape(base)}_{re.escape(subscript)}(?![a-zA-Z0-9_])'
                    normalized_latex = re.sub(pattern1, temp_name, normalized_latex)
                    
                    # Pattern 2: E_{f} (explicit subscript with braces)
                    pattern2 = rf'(?<![a-zA-Z0-9\\]){re.escape(base)}_{{{re.escape(subscript)}}}'
                    normalized_latex = re.sub(pattern2, temp_name, normalized_latex)
        
        return normalized_latex
    
    def _is_latex_format(self, expression: str) -> bool:
        """
        Check if expression contains LaTeX formatting.
        
        Args:
            expression: Expression string to check
            
        Returns:
            True if LaTeX format detected
        """
        latex_patterns = [
            r'\\frac\{',  # \frac{
            r'\\sqrt\{',  # \sqrt{
            r'\\pi\b',    # \pi
            r'\\epsilon', # \epsilon
            r'\\alpha',   # \alpha
            r'\\beta',    # \beta
            r'\\gamma',   # \gamma
            r'\\theta',   # \theta
            r'\\cdot',    # \cdot
            r'\\times',   # \times
            r'\\sin',     # \sin
            r'\\cos',     # \cos
            r'\\log',     # \log
            r'\\exp',     # \exp
            r'\\left\(',  # \left(
            r'\\right\)', # \right)
            r'\^',        # ^ (escaped)
            r'_',         # _ (subscript)
        ]
        
        for pattern in latex_patterns:
            if re.search(pattern, expression):
                return True
        return False
    
    async def _convert_latex_to_infix_via_llm(self, latex_expr: str, problem: Problem) -> Optional[str]:
        try:
            # Get variable names for the prompt
            input_vars = problem.gt_equation.symbols[1:]
            output_var = problem.gt_equation.symbols[0]
            
            # Check if pi is in latex_expr and add it to input_vars if needed
            if ('\\pi' in latex_expr or 'pi' in latex_expr) and 'pi' not in input_vars:
                input_vars = list(input_vars) + ['pi']
            
            prompt = f"""You are a strict mathematical expression converter. Your task is to convert a LaTeX mathematical expression into a syntactically correct Python expression.

### Rules:
1. **Implicit Multiplication**: You MUST convert implicit multiplication to explicit `*`. 
   - Example: `2x` -> `2*x`, `x(y+z)` -> `x*(y+z)`, `3\pi` -> `3*pi`.
2. **Exponents**: Use `**` for exponents. NEVER use `^`.
   - Example: `x^2` -> `x**2`.
3. **Fractions**: Always enclose the numerator and denominator in parentheses if they contain operations.
   - Example: `\\frac{{a+b}}{{c}}` -> `(a+b)/c`.
4. **Variables**: Strictly use the variable names provided in the `Variables` list. Match LaTeX symbols to the provided variables accurately.
5. **Functions**: Use standard Python function names (e.g., `sin`, `cos`, `tan`, `exp`, `sqrt`, `log` for natural log).
6. **Format**: Output ONLY the raw Python string. NO Markdown, NO backticks (```), NO "Here is the answer".

### Examples:

Input LaTeX: \\frac{{4\\pi \\epsilon E_f r^5}}{{3z^2 - r^2}}
Variables: Ef, epsilon, r, pi, z
Output: 4*pi*epsilon*Ef*r**5/(3*z**2-r**2)

Input LaTeX: \\sqrt{{x^2 + y^2}} + \\sin(2\\theta)
Variables: x, y, theta
Output: sqrt(x**2 + y**2) + sin(2*theta)

Input LaTeX: e^{{-kt}} \\cdot \\left( A \\cos(\\omega t) + B \\sin(\\omega t) \\right)
Variables: k, t, A, B, omega
Output: exp(-k*t) * (A*cos(omega*t) + B*sin(omega*t))

### Task:

Input LaTeX: {latex_expr}
Variables: {', '.join(input_vars)}
Output: """

            # Call LLM
            messages = [
                {"role": "user", "content": prompt}
            ]

            llm_response = await self.client.async_call_llm(
                messages,
                temperature=0.0, 
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )

            if not llm_response:

                log.error("LLM returned empty response for LaTeX conversion")
                return None
            
            # Clean up the response
            converted_expr = self._clean_reasoning_tags(llm_response)
            
            # Extract from <answer> tag if present
            answer_blocks = re.findall(self.PATTERN_ANSWER, converted_expr, flags=re.DOTALL | re.IGNORECASE)
            if answer_blocks:
                converted_expr = answer_blocks[-1].strip()
            
            # Remove output variable and equals sign if present
            pattern = rf'{re.escape(output_var)}\s*='
            if re.search(pattern, converted_expr):
                parts = re.split(pattern, converted_expr, maxsplit=1)
                if len(parts) > 1:
                    converted_expr = parts[1].strip()
            
            # Remove trailing punctuation
            converted_expr = re.sub(r'[;,\s]+$', '', converted_expr)
            
            # Remove any remaining LaTeX math mode markers
            converted_expr = re.sub(r'\$\$', '', converted_expr)
            converted_expr = re.sub(r'\$', '', converted_expr)
            
            log.debug("LLM converted LaTeX to: %s", converted_expr)
            return converted_expr
            
        except Exception as e:
            log.error("Error converting LaTeX to infix via LLM: %s", e, exc_info=True)

            return None
    
    def _parse_latex(self, latex_str: str) -> Optional[str]:
        if not LATEX2SYMPY_AVAILABLE:
            return None
        
        try:
            # Use latex2sympy2 to convert LaTeX to SymPy expression
            sympy_expr = latex2sympy(latex_str)
            # Convert SymPy expression to Python string
            python_expr = str(sympy_expr)
            
            # Handle special SymPy functions that need conversion
            # Derivative, Integral, etc. are not needed for our use case
            # If they appear, it means the expression is too complex
            if 'Derivative' in python_expr or 'Integral' in python_expr:
                log.error("Expression contains derivatives/integrals, which are not supported: %s", python_expr)
                return None
            
            # Standardize operators
            python_expr = python_expr.replace('^', '**')
            
            return python_expr
            
        except Exception as e:

            log.error("Error parsing LaTeX with latex2sympy2: %s", e, exc_info=True)
            log.debug("LaTeX string: %s...", latex_str[:200])
            return None
    
    def _parse_plain_text(self, expression: str, problem: Problem) -> str:
        # Standardize operators
        expression = expression.replace('^', '**')
        
        # Remove extra whitespace
        expression = ' '.join(expression.split())
        
        return expression
    
    def _normalize_expression(self, expression: str) -> str:
        """
        Normalize mathematical expression by converting common variations to Python-compatible syntax.
        
        Handles:
        - Abs -> abs
        - ^ -> ** (power operator)
        - |x| -> abs(x) (absolute value notation)
        
        Args:
            expression: Raw expression string
            
        Returns:
            Normalized expression string
        """
        # Replace Abs with abs
        expression = expression.replace('Abs', 'abs')
        
        # Replace ^ with ** for power operator
        expression = expression.replace('^', '**')
        
        # Convert |variable| to abs(variable)
        # Match |variable_name| where variable_name is a valid identifier
        # This pattern matches |x|, |T|, |epsilon|, etc.
        def replace_abs_notation(match):
            var_name = match.group(1)
            return f'abs({var_name})'
        
        # Pattern: | followed by identifier, followed by |
        # Matches |x|, |T|, |epsilon|, etc.
        expression = re.sub(r'\|([a-zA-Z_][a-zA-Z0-9_]*)\|', replace_abs_notation, expression)
        
        # Also handle cases with spaces: | x | -> abs(x)
        expression = re.sub(r'\|\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+\|', r'abs(\1)', expression)
        
        return expression
    
    def _protect_multichar_variables(self, expression: str, problem: Problem) -> str:
        input_vars = problem.gt_equation.symbols[1:]
        
        multichar_vars = [var for var in input_vars if len(var) > 1]
        
        if not multichar_vars:
            return expression
        
        multichar_vars.sort(key=len, reverse=True)
        
        protected_expr = expression
        
        for var in multichar_vars:
            
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(var) + r'(?![a-zA-Z0-9_])'
            
            def replace_func(match):
                matched_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                before_char = protected_expr[start_pos - 1] if start_pos > 0 else ''
                after_char = protected_expr[end_pos] if end_pos < len(protected_expr) else ''
                
                if before_char == '(' and after_char == ')':
                    return matched_text
                
                if before_char == '*' or after_char == '*':
                    return matched_text
                
                return f'({matched_text})'
            
            protected_expr = re.sub(pattern, replace_func, protected_expr)
        
        return protected_expr
    
    def _validate_variable_names(
        self, 
        expression: str, 
        problem: Problem
    ) -> Tuple[str, List[str], List[str]]:
        if not SYMPY_AVAILABLE:

            # Fallback: simple validation without SymPy
            input_vars = problem.gt_equation.symbols[1:]
            matched_vars = [v for v in input_vars if v in expression]
            return expression, matched_vars, []
        
        try:
            input_vars = problem.gt_equation.symbols[1:]
            
            # Normalize expression before parsing (handles |x| -> abs(x), Abs -> abs, ^ -> **)
            expression = self._normalize_expression(expression)
            
            expression = self._protect_multichar_variables(expression, problem)
            
            # Build local_dict to treat all variables and common conflict symbols as symbols
            local_dict = self._build_local_dict(problem)
            
            # Parse expression using SymPy
            expr = parse_expr(
                expression,
                transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,),
                local_dict=local_dict
            )

            # Extract all symbols
            all_symbols = expr.free_symbols
            symbol_names = {str(s) for s in all_symbols}
            
            # Identify matched input variables
            matched_vars = [v for v in input_vars if v in symbol_names]
            
            # Identify unknown symbols (not input vars, not builtin functions, not math constants)
            unknown_symbols = []
            for sym_name in symbol_names:
                if (sym_name not in input_vars and 
                    sym_name.lower() not in self.builtin_functions and
                    sym_name.lower() not in self.math_constants):
                    unknown_symbols.append(sym_name)
            
            # Convert back to string (in case SymPy made any changes)
            validated_expr = str(expr)
            validated_expr = validated_expr.replace('^', '**')
            
            return validated_expr, matched_vars, unknown_symbols
            
        except Exception as e:
            log.error("Error validating variable names: %s", e, exc_info=True)
            # Return original expression if validation fails
            input_vars = problem.gt_equation.symbols[1:]
            matched_vars = [v for v in input_vars if v in expression]
            return expression, matched_vars, []
    
    def _sanitize_infix_expression(self, text: str) -> Optional[str]:
        if not text:
            return None
            
        text = text.strip()

        code_block_pattern = r"```(?:python)?\s*(.*?)\s*```"
        match = re.search(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

        text = text.replace('$$', '').replace('$', '')
        text = text.replace(r'\[', '').replace(r'\]', '')

        boxed_content = self._extract_boxed_content(text)
        if boxed_content:
            text = boxed_content
        else:
            boxed_pattern = r'\\+boxed\{'
            if re.search(boxed_pattern, text):
                match = re.search(boxed_pattern, text)
                if match:
                    start_idx = match.end()
                    brace_count = 1
                    i = start_idx
                    end_idx = None
                    while i < len(text):
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i
                                break
                        i += 1
                    if end_idx is not None:
                        text = text[start_idx:end_idx].strip()
                    else:
                        text = text[start_idx:].strip()
                        paren_count = text.count('(') - text.count(')')
                        brace_count = text.count('{') - text.count('}')
                        if text.endswith('}') and brace_count > 0:
                            text = text[:-1].rstrip()
                        elif text.endswith(')') and paren_count < 0:
                            text = text[:-1].rstrip()

        if "=" in text:
            parts = text.rsplit("=", 1)
            rhs = parts[1].strip()
            if rhs:
                text = rhs

        text = re.sub(r'^(?:Expression|Answer|Result|Output):\s*', '', text, flags=re.IGNORECASE)

        if "^" in text and "**" not in text:
            text = text.replace("^", "**")
            
        return text.strip()

    async def parse_expression(self, response: str, problem: Problem) -> Tuple[str, List[str], List[str]]:
        # Step 1: Extract expression part
        expression_part = self._extract_expression_part(response, problem)

        if not expression_part:
            # Check if the response itself looks like a plain expression (not an LLM response)
            response_clean = response.strip()
            has_llm_markers = (
                f'<{self.TAG_ANSWER}>' in response_clean.lower() or
                '\\boxed{' in response_clean or
                f'<{self.TAG_REDACTED_REASONING}>' in response_clean.lower() or
                f'<{self.TAG_TOOL_CODE}>' in response_clean.lower() or 
                f'<{self.TAG_THINK}>' in response_clean.lower()
            )
            if not has_llm_markers and response_clean:
                # Looks like a plain expression, use it directly
                log.debug("Using response as plain expression (no LLM markers found): %s", response_clean)
                expression_part = response_clean
            else:

                log.error("Could not extract expression from response")
                return "", [], []
        
        # Step 2: If LaTeX format detected, convert to infix via LLM first
        python_expr = None
        if self._is_latex_format(expression_part):
            # Step 2a: Convert LaTeX to infix using LLM
            llm_converted_expr = await self._convert_latex_to_infix_via_llm(expression_part, problem)
            
            # Step 2a.1: If the converted expression contains "=", extract only the right-hand side
            if llm_converted_expr:
                cleaned_expr = self._sanitize_infix_expression(llm_converted_expr)

                if cleaned_expr:
                    expression_part = cleaned_expr
                    python_expr = self._parse_plain_text(expression_part, problem)
                    if self._reverse_mapping:
                        python_expr = self._restore_variable_names(python_expr, self._reverse_mapping)
                else:
                    pass
            else:
                log.warning("LLM conversion failed, falling back to LaTeX parser...")

        # Step 3: Fallback to plain text parsing if LaTeX parsing failed
        if python_expr is None:
            # Clean expression_part (remove =, markdown blocks, etc.) before parsing
            cleaned_expression_part = self._sanitize_infix_expression(expression_part)
            if cleaned_expression_part:
                expression_part = cleaned_expression_part
            else:

                pass
            python_expr = self._parse_plain_text(expression_part, problem)
            if self._reverse_mapping:
                python_expr = self._restore_variable_names(python_expr, self._reverse_mapping)
        
        # Step 4: Validate variable names (using original variable names)
        try:
            validated_expr, matched_vars, unknown_symbols = \
                self._validate_variable_names(python_expr, problem)
        except Exception as e:

            log.error("Error validating variable names: %s", e, exc_info=True)
            return "", [], []
        
        # Step 5: Report validation results
        input_vars = problem.gt_equation.symbols[1:]
        missing_vars = []
        if len(matched_vars) < len(input_vars):
            missing_vars = [v for v in input_vars if v not in matched_vars]
        
        return validated_expr, missing_vars, unknown_symbols
    
    def expression_to_lambda(self, expression: str, problem: Problem):
        try:
            # Validate expression: check for non-mathematical content
            expression = expression.strip()
            if not expression:
                log.error("Empty expression provided for converting to lambda")
                return None
            
            # Check for XML/HTML tags (reasoning tags that might have been missed)
            if re.search(r'<[^>]+>', expression):
                log.error("Expression contains XML/HTML tags, likely contains reasoning text: %s", expression[:100])

                return None
            
            # Check for very long words (likely reasoning text, not math)
            words = re.findall(r'\b[a-zA-Z_]{20,}\b', expression)
            if words:
                log.error("Expression contains suspiciously long words (likely reasoning text): %s", words[:3])
                return None
            
            # Check for common reasoning keywords
            reasoning_keywords = ['given', 'previous', 'attempt', 'analysis', 'consider', 'observe', 
                                 'investigate', 'approach', 'insight', 'relationship', 'singularity']
            expr_lower = expression.lower()
            if any(keyword in expr_lower for keyword in reasoning_keywords):
                log.error("Expression contains reasoning keywords, likely not a mathematical expression")
                return None
            
            # Normalize expression: replace common variations
            # Handles Abs -> abs, ^ -> **, and |x| -> abs(x)
            expression = self._normalize_expression(expression)
            
            # Get variable names
            input_vars = problem.gt_equation.symbols[1:]
            
            # Create base lambda function that accepts individual arguments
            math_namespace = {
                'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'abs': np.abs,
                'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'pi': np.pi, 'e': np.e,
                'np': np  # Also include numpy for safety
            }
            # Create a combined namespace with builtins restricted
            eval_globals = {"__builtins__": {}}
            eval_globals.update(math_namespace)
            lambda_str = f"lambda {', '.join(input_vars)}: {expression}"
            lambda_base = eval(lambda_str, eval_globals, math_namespace)
            
            # Wrap it to accept array input (X: shape (n_samples, n_features))
            def lambda_fn(X: np.ndarray) -> np.ndarray:
                """
                X: shape (n_samples, len(input_vars))
                Returns: shape (n_samples,)
                """
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                
                n_samples = X.shape[0]
                
                # Extract columns and pass as separate arguments
                args = [X[:, i] for i in range(len(input_vars))]
                result = lambda_base(*args)
                
                result = np.asarray(result)
                if result.ndim == 0:
                    result = np.full(n_samples, result.item(), dtype=result.dtype)
                elif result.shape[0] != n_samples:
                    if result.size == 1:
                        result = np.full(n_samples, result.item(), dtype=result.dtype)
                    else:
                        try:
                            result = result.reshape(n_samples)
                        except ValueError:
                            log.warning(f"Shape mismatch in lambda_fn: result shape {result.shape} vs expected ({n_samples},). Returning zeros.")
                            result = np.zeros(n_samples, dtype=result.dtype)
                
                return result
            
            return lambda_fn
            
        except Exception as e:
            log.error("Error creating lambda function: %s", e, exc_info=True)

            return None
    
    def _calculate_r2(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)
    
    def _create_sum_initial_solution(self, problem: Problem) -> str:
        input_vars = problem.gt_equation.symbols[1:]
        if not input_vars:
            return "0"  # Fallback if no input variables
        
        # Protect multi-character variable names with parentheses to prevent
        # SymPy's implicit multiplication from splitting them (e.g., "kb" -> "k*b")
        protected_vars = []
        for var in input_vars:
            if len(var) > 1:
                # Multi-character variable: wrap in parentheses
                protected_vars.append(f"({var})")
            else:
                # Single-character variable: use as-is
                protected_vars.append(var)
        
        return " + ".join(protected_vars)
    
    def _evaluate_and_register_solution_sync(
        self,
        expression: str,
        problem: Problem,
        generation: int,
        parent_id: Optional[str] = None,
        island_id: Optional[int] = None,
        llm_response: Optional[str] = None
    ) -> Optional[EvolvedSolution]:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._evaluate_and_register_solution(
                        expression, problem, generation, parent_id, island_id, llm_response
                    )
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self._evaluate_and_register_solution(
                    expression, problem, generation, parent_id, island_id, llm_response
                )
            )
    
    async def _evaluate_and_register_solution(
        self,
        expression: str,
        problem: Problem,
        generation: int,
        parent_id: Optional[str] = None,
        island_id: Optional[int] = None,
        llm_response: Optional[str] = None
    ) -> Optional[EvolvedSolution]:
        if llm_response is None:
            llm_response = expression
        
        parsed_expr, missing_vars, unknown_symbols = await self.parse_expression(expression, problem)
        if not parsed_expr:

            log.error("Failed to parse expression from LLM response")
            return None
        
        log.debug("Parsed expression: %s", parsed_expr)
        if missing_vars:
            log.warning("Missing input variables: %s", missing_vars)
        
        # Rejection sampling: reject expressions with unknown symbols
        if unknown_symbols:
            log.warning("Rejecting expression with unknown symbols: %s", unknown_symbols)
            return None

        parsed_expr = re.sub(r'\be\*\*\(', 'exp(', parsed_expr)
        
        parsed_expr_before_opt = parsed_expr
        
        lambda_fn = self.expression_to_lambda(parsed_expr, problem)
        if lambda_fn is None:

            log.error("Failed to convert expression to lambda: %s", parsed_expr)
            return None

        optimized_lambda, optimized_expr, opt_info = \
            optimize_expression_parameters(lambda_fn, parsed_expr, problem)

        if opt_info.get('optimized', False):
            lambda_fn = optimized_lambda
            parsed_expr = optimized_expr
            log.debug("Optimized expression: %s", optimized_expr)
        else:
            log.debug("No constants found or optimization failed. Using original expression.")
        
        try:
            valid_samples = problem.samples.get('valid', None)
            if valid_samples is not None:
                X_eval = valid_samples[:, 1:]
                y_eval = valid_samples[:, 0]
                eval_set_name = "validation"
            else:

                X_eval = problem.train_samples[:, 1:]
                y_eval = problem.train_samples[:, 0]
                eval_set_name = "training"

            y_pred = lambda_fn(X_eval)
            
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                log.error("Invalid predictions (NaN or Inf detected)")
                return None
            
            mse = np.mean((y_pred - y_eval) ** 2)
            r2 = self._calculate_r2(y_pred, y_eval)
            
            if np.isnan(mse) or np.isinf(mse) or mse < 0:
                log.error("Invalid MSE: %s", mse)
                return None
            
        except Exception as e:
            log.error("Error evaluating solution: %s", e, exc_info=True)

            return None
        
        equation = Equation(
            symbols=problem.gt_equation.symbols,
            symbol_descs=problem.gt_equation.symbol_descs,
            symbol_properties=problem.gt_equation.symbol_properties,
            expression=parsed_expr,
            lambda_format=lambda_fn,
            missing_vars=missing_vars if missing_vars else None,
            unknown_symbols=unknown_symbols if unknown_symbols else None
        )
        
        aux_generation = {
            "llm_response": llm_response,
            "parsed_expression": parsed_expr,
            "parsed_expression_before_optimization": parsed_expr_before_opt,
            "generation": generation,
            "parent_id": parent_id,
            "optimization_info": opt_info,
            "method": "se_evolution"
        }
        
        search_result = SearchResult(
            equation=equation,
            aux=aux_generation
        )
        
        return EvolvedSolution(
            expression=parsed_expr,
            lambda_fn=lambda_fn,
            search_result=search_result,
            mse=mse,
            r2=r2,
            generation=generation,
            parent_id=parent_id,
            eval_set_name=eval_set_name,
            y_pred=y_pred
        )
    
    def _format_parent_errors(self, parent_solution: EvolvedSolution) -> str:
        errors = []
        equation = parent_solution.search_result.equation
        
        if equation.missing_vars:
            missing_vars_str = ", ".join(equation.missing_vars)
            errors.append(f"Missing input variables: {missing_vars_str}. The expression must include ALL input variables.\n")
        
        if equation.unknown_symbols:
            unknown_symbols_str = ", ".join(equation.unknown_symbols)
            errors.append(f"Unknown symbols detected: {unknown_symbols_str}. The expression contain unknown symbols. Try to include only the input variables.\n")
        
        if errors:
            return "".join(errors)
        else:
            return ""
    
    def build_mutation_prompt(
        self,
        problem: Problem,
        parent_solutions: List[EvolvedSolution],
        mutation_type: str = "refine"
    ) -> Tuple[str, str, str, str, str]:
        
        if not self._var_mapping:
            self._var_mapping, self._reverse_mapping = self._create_variable_mapping(problem)
        
        if self.enable_tools:
            system_prompt, _ = self._build_dynamic_system_prompt(set())
        else:
            memory_section = self._memory_section if self._memory_section else ""
            system_prompt = self._system_prompt_template.replace("{MEMORY_SECTION}", memory_section)
            system_prompt = system_prompt + "\n" + self._response_format_section
        
        symbols = problem.gt_equation.symbols
        symbol_descs = problem.gt_equation.symbol_descs
        
        output_var = symbols[0]
        output_desc = symbol_descs[0]
        input_vars = symbols[1:]
        input_descs = symbol_descs[1:]
        
        # Build input variables list with format: "var1 (desc1), var2 (desc2), ..."
        input_vars_list = []
        for var, desc in zip(input_vars, input_descs):
            input_vars_list.append(f"{var} ({desc})")
        
        # Join with commas, add "and" before the last item if there are multiple items
        if len(input_vars_list) > 1:
            input_vars_str = ", ".join(input_vars_list[:-1]) + ", and " + input_vars_list[-1]
        else:
            input_vars_str = input_vars_list[0] if input_vars_list else ""
        
        input_vars_symbols = ", ".join(input_vars)
        base_task_prompt = f"Your task: Find the mathematical function skeleton for {output_var} ({output_desc}), given data on {input_vars_str}. Express {output_var} as an explicit function of the other variables. The expression must ONLY use the following independent variables {input_vars_symbols} and must NOT contain the symbol {output_var} itself."
        
        data_prompt = ""

        if mutation_type == "regenerate":
            examples_text = ""
        elif mutation_type == "refine":
            if parent_solutions and len(parent_solutions) >= 1:
                examples_text = f"\nAttempt 1:\n"
                examples_text += f"Expression: {parent_solutions[0].expression}\n"
                examples_text += f"\nTask: Refine the parent expression to potentially improve it. Generate a /new mathematical expression that may achieve lower MSE.\n"
                error_hint = self._format_parent_errors(parent_solutions[0])
                if error_hint:
                    examples_text += error_hint
            else:

                raise ValueError(f"Mutation type {mutation_type} requires 1 parent solution")
        elif mutation_type == "mutate":
            if parent_solutions and len(parent_solutions) >= 1:
                examples_text = f"\nAttempt 1:\n"
                examples_text += f"Expression: {parent_solutions[0].expression}\n"
                examples_text += f"\nTask: Mutate the parent expression by making structural changes. Generate a new mathematical expression that may achieve lower MSE.\n"
                error_hint = self._format_parent_errors(parent_solutions[0])
                if error_hint:
                    examples_text += error_hint
            else:

                raise ValueError(f"Mutation type {mutation_type} requires 1 parent solution")
        elif mutation_type == "crossover":
            if parent_solutions and len(parent_solutions) >= 2:
                examples_text = ""
                examples_text += f"Attempt 1:"
                examples_text += f"Expression: {parent_solutions[0].expression}\n"
                error_hint_1 = self._format_parent_errors(parent_solutions[0])
                if error_hint_1:
                    examples_text += error_hint_1
                examples_text += f"Attempt 2:"
                examples_text += f"Expression: {parent_solutions[1].expression}\n"
                error_hint_2 = self._format_parent_errors(parent_solutions[1])
                if error_hint_2:
                    examples_text += error_hint_2
                examples_text += f"Task: Combine elements from the two parent expressions to create a new mathematical expression that may achieve lower MSE.\n"
            else:

                raise ValueError(f"Mutation type {mutation_type} requires 2 parent solutions")
        else:

            raise ValueError(f"Mutation type {mutation_type} is not supported")
        
        if examples_text != "":
            user_prompt = (
                            f"{base_task_prompt}\n"
                            f"=== Previous Attempts ===\n"
                            f"{examples_text}\n"
                            )
        else:
            user_prompt = base_task_prompt
        return system_prompt, user_prompt, examples_text, data_prompt, base_task_prompt
    
    def _parse_tool_calls(self, action: str) -> tuple[List[str], List[str]]:
        tool_calls = []
        tool_code_blocks = []
        
        tool_code_pattern = self.PATTERN_TOOL_CODE
        matches = re.finditer(tool_code_pattern, action, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            full_block = match.group(0)
            content = match.group(1).strip()
            if not content:
                continue
            
            tool_name = content.strip().lower()
            
            if tool_name == "analyze_data" or tool_name == "analyze_residuals" or tool_name == "dimensional_analysis":
                tool_calls.append(tool_name)
                tool_code_blocks.append(full_block)
            else:
                continue
        
        return tool_calls, tool_code_blocks
    
    def _has_tool_calls(self, action: str) -> bool:
        return bool(re.search(f"<{self.TAG_TOOL_CODE}>", action, re.IGNORECASE))
    
    async def _generate_single_sample_with_messages_async(
        self,
        problem: Problem,
        generation: int,
        buffer: SEExperienceBuffer,
        mutation_type: str,
        island_id: int,
        parent_solutions: List[EvolvedSolution],
        system_prompt: str,
        user_prompt: str,
        examples_text: str,
        data_prompt: str,
        verbose_timing: bool = False,
        p_main: Optional[EvolvedSolution] = None,
        state: Optional[str] = None,
        operation: Optional[str] = None,
        base_task_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        result = {
            "llm_interaction": None,
            "new_solutions_count": 0
        }
        
        sample_start_time = time.perf_counter()
        
        llm_time = 0.0
        process_time = 0.0
        register_time = 0.0
        
        try:
            token_stats_before = self.client.get_token_stats()
            
            if self.enable_tools:
                all_tools = set(self._tool_descriptions.keys())
                initial_suffix = self._get_user_prompt_suffix(all_tools)
            else:
                initial_suffix = self._get_user_prompt_suffix()
            full_user_prompt = f"{user_prompt}\n{initial_suffix}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_prompt}
            ]
            
            if not self.enable_tools:
                llm_start_time = time.perf_counter()
                final_response = await self.client.async_call_llm(
                    messages,
                    temperature=self.temperature,
                    extra_body=self.extra_body
                )
                llm_end_time = time.perf_counter()
                llm_time = llm_end_time - llm_start_time
                tool_rounds = 0
                tool_call_history = []
                dynamic_system_prompt = system_prompt
                current_user_prompt = full_user_prompt
            else:
                if self._tool_group is None:

                    self._tool_group = SymbolicRegressionToolGroup(problem, searcher=self)
                tool_group = self._tool_group
                
                tool_rounds = 0
                final_response = None
                tool_call_history = []
                tool_interaction_history = []
                called_tools = set()
                
                if base_task_prompt is None:
                    base_task_prompt = user_prompt
                
                while tool_rounds < self.max_tool_rounds:
                    is_last_round = (tool_rounds == self.max_tool_rounds - 1)
                    
                    if is_last_round:
                        all_tools = set(self._tool_descriptions.keys())
                        dynamic_system_prompt, available_tools = self._build_dynamic_system_prompt(all_tools)
                    else:
                        dynamic_system_prompt, available_tools = self._build_dynamic_system_prompt(called_tools)
                    
                    tool_interaction_section = ""
                    if tool_interaction_history:
                        tool_interaction_section = "\n=== Tool Interaction History ===\n" + "\n".join(tool_interaction_history)
                    dynamic_suffix = self._get_user_prompt_suffix(available_tools)
                    current_user_prompt = f"{user_prompt}{tool_interaction_section}\n{dynamic_suffix}"
                    
                    llm_start_time = time.perf_counter()
                    messages = [
                        {"role": "system", "content": dynamic_system_prompt},
                        {"role": "user", "content": current_user_prompt}
                    ]

                    response = await self.client.async_call_llm(
                        messages,
                        temperature=self.temperature,
                        extra_body=self.extra_body
                    )
                    llm_end_time = time.perf_counter()
                    llm_time += (llm_end_time - llm_start_time)
                    
                    if not self._has_tool_calls(response):
                        final_response = response
                        break
                    
                    tool_calls, tool_code_blocks = self._parse_tool_calls(response)
                    
                    if is_last_round:
                        final_response = response

                        break
                    
                    all_available_tools = set(self._tool_descriptions.keys())
                    valid_tool_calls = []
                    valid_tool_code_blocks = []
                    
                    for idx, tool_name in enumerate(tool_calls):
                        if tool_name not in all_available_tools:
                            continue
                        if tool_name in called_tools:
                            continue
                        valid_tool_calls.append(tool_name)
                        if idx < len(tool_code_blocks):
                            valid_tool_code_blocks.append(tool_code_blocks[idx])
                        else:
                            valid_tool_code_blocks.append(f"<{self.TAG_TOOL_CODE}>{tool_name}</{self.TAG_TOOL_CODE}>")
                    
                    for idx, tool_name in enumerate(valid_tool_calls):
                        tool_code_block = valid_tool_code_blocks[idx] if idx < len(valid_tool_code_blocks) else f"<{self.TAG_TOOL_CODE}>{tool_name}</{self.TAG_TOOL_CODE}>"
                        try:
                            if tool_name == "analyze_residuals":
                                tool_result = tool_group.execute_tool(tool_name, parent_solutions=parent_solutions)
                            else:
                                tool_result = tool_group.execute_tool(tool_name)
                            tool_result_text = f"<tool_response>\nTool: {tool_name}\nResult:\n{tool_result}\n</tool_response>"
                            
                            called_tools.add(tool_name)
                            tool_call_history.append({
                                "round": tool_rounds,
                                "tool_name": tool_name
                            })
                            
                            interaction_entry = f"[Round {tool_rounds}] {tool_code_block}\n{tool_result_text}"
                            tool_interaction_history.append(interaction_entry)
                        except Exception as e:

                            tool_result_text = f""
                            
                            # called_tools.add(tool_name)
                            tool_call_history.append({
                                "round": tool_rounds,
                                "tool_name": tool_name
                            })


                    tool_rounds += 1
                
                if final_response is None:
                    final_response = response
            
            token_stats_after_llm = self.client.get_token_stats()
            
            llm_interaction = {
                "island_id": island_id,
                "mutation_type": mutation_type,
                "tool_calls": tool_call_history,
                "tool_rounds": tool_rounds,
                "prompt": {
                    "system_prompt": dynamic_system_prompt,
                    "user_prompt": current_user_prompt,
                    "parent_solutions": [
                        {
                            "expression": sol.expression,
                            "mse": sol.mse,
                            "r2": sol.r2,
                            "solution_id": sol.solution_id
                        } for sol in parent_solutions
                    ] if parent_solutions else []
                },
                "llm_response": final_response,
                "token_usage": {
                    "prompt_tokens": token_stats_after_llm["prompt_tokens"] - token_stats_before["prompt_tokens"],
                    "completion_tokens": token_stats_after_llm["completion_tokens"] - token_stats_before["completion_tokens"],
                    "total_tokens": token_stats_after_llm["total_tokens"] - token_stats_before["total_tokens"]
                }
            }
            
            token_stats_before_parse = self.client.get_token_stats()
            process_start_time = time.perf_counter()
            solution = await self._evaluate_and_register_solution(
                final_response,
                problem,
                generation,
                parent_solutions[0].solution_id if parent_solutions else None,
                island_id,
                final_response
            )
            process_end_time = time.perf_counter()
            process_time = process_end_time - process_start_time
            
            if p_main is not None and state is not None and operation is not None:
                if solution is None:
                    reward = -0.5
                else:
                    if p_main.mse > 0:
                        improvement = (p_main.mse - solution.mse) / p_main.mse
                        reward = max(-0.5, min(1.0, improvement))
                    else:
                        if solution.mse < p_main.mse:
                            reward = 0.5
                        else:
                            reward = -0.5
                
                self.strategy_table.update_strategy(state, operation, reward)
                
                if verbose_timing:
                    log.debug("[Generation %d | Island %d | %s] reward: %.3f state: %s op: %s",
                             generation, island_id, mutation_type, reward, state, operation)
            else:

                log.error("p_main: %s, state: %s, operation: %s", p_main, state, operation)
                raise ValueError("p_main, state and operation must be provided")
            token_stats_after_parse = self.client.get_token_stats()
            parse_token_usage = {
                "prompt_tokens": token_stats_after_parse["prompt_tokens"] - token_stats_before_parse["prompt_tokens"],
                "completion_tokens": token_stats_after_parse["completion_tokens"] - token_stats_before_parse["completion_tokens"],
                "total_tokens": token_stats_after_parse["total_tokens"] - token_stats_before_parse["total_tokens"]
            }
            llm_interaction["token_usage"]["parse_prompt_tokens"] = parse_token_usage["prompt_tokens"]
            llm_interaction["token_usage"]["parse_completion_tokens"] = parse_token_usage["completion_tokens"]
            llm_interaction["token_usage"]["parse_total_tokens"] = parse_token_usage["total_tokens"]
            llm_interaction["token_usage"]["total_prompt_tokens"] = llm_interaction["token_usage"]["prompt_tokens"] + parse_token_usage["prompt_tokens"]
            llm_interaction["token_usage"]["total_completion_tokens"] = llm_interaction["token_usage"]["completion_tokens"] + parse_token_usage["completion_tokens"]
            llm_interaction["token_usage"]["total_all_tokens"] = llm_interaction["token_usage"]["total_tokens"] + parse_token_usage["total_tokens"]
            
            register_time = 0.0
            if solution:
                register_start_time = time.perf_counter()
                added = await asyncio.to_thread(
                    buffer.register_solution,
                    solution,
                    island_id
                )
                register_end_time = time.perf_counter()
                register_time = register_end_time - register_start_time
                if added:
                    result["new_solutions_count"] = 1
                    llm_interaction["parsed_expression"] = solution.expression
                    llm_interaction["solution"] = {
                        "expression": solution.expression,
                        "mse": solution.mse,
                        "r2": solution.r2,
                        "solution_id": solution.solution_id
                    }
                else:
                    llm_interaction["parsed_expression"] = solution.expression
                    llm_interaction["duplicate"] = True
            else:
                if verbose_timing:
                    log.debug("[Generation %d | Island %d | %s] parse failed, no solution",
                             generation, island_id, mutation_type)
            
            if verbose_timing:
                llm_interaction["timing"] = {
                    "llm_time": llm_time,
                    "process_time": process_time,
                    "register_time": register_time,
                    "total_time": (time.perf_counter() - sample_start_time)
                }
            
            result["llm_interaction"] = llm_interaction
            
        except Exception as e:
            log.error("[Generation %d | Island %d | %s] Error in async sample generation: %s", 
                     generation, island_id, mutation_type, e, exc_info=True)
            result["llm_interaction"] = {
                "island_id": island_id,
                "mutation_type": mutation_type,
                "error": str(e),
                "token_usage": {}
            }
        finally:
            if verbose_timing:
                sample_end_time = time.perf_counter()
                sample_total_time = sample_end_time - sample_start_time
                log.debug("[Generation %d | Island %d | %s] total time: %.3f s",
                         generation, island_id, mutation_type, sample_total_time)
        
        return result
    
    def evolve(self, problem: Problem, evolution_config: Optional[SEEvolutionConfig] = None, verbose_timing: bool = False, output_path: Optional[Path] = None) -> List[SearchResult]:
        try:
            if evolution_config is None:
                evolution_config = SEEvolutionConfig()
            
            # Record initial token stats for the entire evolution process
            initial_token_stats = self.client.get_token_stats()
            
            # Initialize detailed log structure
            detailed_log = {
            "evolution_config": {
                "num_islands": evolution_config.num_islands,
                "functions_per_prompt": evolution_config.functions_per_prompt,
                "solutions_per_generation": evolution_config.solutions_per_generation,
                "samples_per_prompt": evolution_config.samples_per_prompt,
                "max_generations": evolution_config.max_generations,
                "population_size_per_island": evolution_config.population_size_per_island,
                "selection_method": evolution_config.selection_method,
                "reset_period": evolution_config.reset_period
            },
            "initialization": {
                "num_islands": evolution_config.num_islands,
                "initial_solutions": []
            },
            "generations": []
            }
            
            buffer = SEExperienceBuffer(evolution_config)

            if evolution_config.memory_enabled:
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
                self._memory_manager = MemoryManager(mem_cfg)
                self._memory_section = self._memory_manager.format_memory_section()
                
                if mem_cfg.debug_save_memory and output_path is not None:
                    memory_debug_base = output_path / "memory_debug"
                    memory_debug_base.mkdir(parents=True, exist_ok=True)
                    self._memory_debug_path = memory_debug_base / str(problem.original_idx)
                    self._memory_debug_path.mkdir(parents=True, exist_ok=True)
                else:
                    self._memory_debug_path = None
            else:

                self._memory_manager = None
                self._memory_section = ""
                self._memory_debug_path = None
            
            if self.enable_tools:
                if self._tool_group is None:
                    self._tool_group = SymbolicRegressionToolGroup(problem, searcher=self)
                else:
                    self._tool_group.reset(problem, searcher=self)
            
            log.info("Initializing %d islands with initial solutions...", evolution_config.num_islands)
            for island_id in range(evolution_config.num_islands):
                initial_expr = self._create_sum_initial_solution(problem)
                log.debug("Island %d: Initial expression (sum): %s", island_id, initial_expr)
                solution = self._evaluate_and_register_solution_sync(
                    initial_expr,
                    problem,
                    generation=0,
                    island_id=island_id
                )
                if solution:
                    buffer.register_solution(solution, island_id)
                    log.info("Island %d: Initial solution MSE = %.6e", island_id, solution.mse)
                    detailed_log["initialization"]["initial_solutions"].append({
                        "island_id": island_id,
                        "expression": solution.expression,
                        "mse": solution.mse,
                        "r2": solution.r2,
                        "eval_set_name": solution.eval_set_name
                    })
                else:
                    log.warning("Island %d: Failed to create initial solution", island_id)
            
            log.info("Starting evolution for %d generations...", evolution_config.max_generations)            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                for generation in range(1, evolution_config.max_generations + 1):
                    generation_start_time = time.perf_counter()
                    if verbose_timing:
                        log.debug("\n%s", '='*60)
                        log.debug("Generation %d start", generation)
                        log.debug("%s", '='*60)
                    
                    new_solutions_count = 0
                    generation_log = {
                        "generation": generation,
                        "solutions_generated": 0,
                        "best_mse_so_far": None,
                        "island_best_solutions": [],
                        "llm_interactions": []
                    }
                    
                    async def run_generation():
                        prep_start_time = time.perf_counter() if verbose_timing else None
                        
                        best_mse_before_generation = None
                        if self._memory_manager and self._memory_manager.cfg.enabled:
                            with buffer._lock:
                                best_mse_before_generation = buffer._best_mse_global if buffer._best_solution_global else None
                        
                        is_stagnated = False
                        if self._memory_manager and self._memory_manager.cfg.enabled:
                            is_stagnated = self._memory_manager.is_stagnated()
                        
                        def get_main_parent_sync():
                            with buffer._lock:
                                island_id = np.random.randint(len(buffer._islands))
                                island = buffer._islands[island_id]
                                
                                parent_solutions = island.sample_parents(
                                    num_parents=1,
                                    method=buffer.config.selection_method,
                                    config=buffer.config,
                                    generation=generation,
                                    is_stagnated=is_stagnated
                                )
                                
                                return island_id, island, parent_solutions[0] if parent_solutions else None
                        
                        island_id, island, p_main = await asyncio.to_thread(get_main_parent_sync)
                        
                        if p_main is None:

                            log.warning("-"*30)
                            log.warning("[Generation %d | Island %d] Error: Failed to sample parent solution, skipping this generation", 
                                       generation, island_id)
                            log.warning("="*30)
                            prep_time = None
                            return [], prep_time
                        
                        state = self.strategy_table.get_state(p_main, island)
                        
                        mutation_type = self.strategy_table.select_operation(state, is_stagnated=is_stagnated)
                        
                        parent_solutions = [p_main]
                        if mutation_type == "crossover":
                            def get_second_parent_sync():
                                with buffer._lock:
                                    k = max(10, len(island.population) // 5)
                                    second_parent = island.sample_from_top_k(k, exclude_solutions=[p_main])
                                    return second_parent
                            
                            second_parent = await asyncio.to_thread(get_second_parent_sync)
                            if second_parent:
                                parent_solutions.append(second_parent)
                            else:
                                log.warning("[Generation %d | Island %d] Warning: Failed to sample second parent solution, downgrading crossover to mutate", 
                                           generation, island_id)
                                mutation_type = "mutate"
                        
                        system_prompt, user_prompt, examples_text, data_prompt, base_task_prompt = self.build_mutation_prompt(
                            problem,
                            parent_solutions,
                            mutation_type=mutation_type
                        )
                        
                        prep_end_time = time.perf_counter() if verbose_timing else None
                        prep_time = prep_end_time - prep_start_time if verbose_timing else None
                        
                        if verbose_timing:
                            log.debug("[Generation %d] prep (sample parents + build prompt): %.3f s", generation, prep_time)
                            log.debug("[Generation %d] generating %d child samples with shared prompt...",
                                     generation, evolution_config.samples_per_prompt)
                        
                        tasks = []
                        for _ in range(evolution_config.samples_per_prompt):
                            tasks.append(
                                self._generate_single_sample_with_messages_async(
                                    problem,
                                    generation,
                                    buffer,
                                    mutation_type=mutation_type,
                                    island_id=island_id,
                                    parent_solutions=parent_solutions,
                                    system_prompt=system_prompt,
                                    user_prompt=user_prompt,
                                    examples_text=examples_text,
                                    data_prompt=data_prompt,
                                    verbose_timing=verbose_timing,
                                    p_main=p_main,
                                    state=state,
                                    operation=mutation_type,
                                    base_task_prompt=base_task_prompt
                                )
                            )
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        if self._memory_manager and self._memory_manager.cfg.enabled:
                            current_best = buffer._best_mse_global if buffer._best_solution_global else None
                            improved = self._memory_manager.update_best(current_best, generation=generation)
                            
                            if improved:
                                if self._memory_manager.current_memory_type == "stagnation":
                                    self._memory_manager.current_memory_type = "normal"
                                    self._memory_section = self._memory_manager.format_memory_section()
                                self._memory_manager.last_stagnation_trigger_generation = None
                            
                            trigger_normal_epoch = self._memory_manager.should_trigger_normal_epoch(generation)
                            
                            is_fresh_improvement = False
                            if buffer._best_solution_global and buffer._best_solution_global.generation == generation:
                                if best_mse_before_generation is not None and current_best is not None:
                                    if self._memory_manager.cfg.improvement_threshold > 0:
                                        improvement_ratio = (best_mse_before_generation - current_best) / best_mse_before_generation
                                        if improvement_ratio >= self._memory_manager.cfg.improvement_threshold:
                                            is_fresh_improvement = True
                            
                            trigger_normal = trigger_normal_epoch or is_fresh_improvement
                            
                            trigger_stag = self._memory_manager.should_trigger_stagnation(generation)
                            
                            if trigger_normal and current_best is not None:
                                log.info("Normal memory summary triggered at generation %d (epoch=%s, fresh_improvement=%s)", 
                                        generation, trigger_normal_epoch, is_fresh_improvement)
                                
                                good_samples = self._collect_top_solutions(buffer, self._memory_manager.cfg.good_top_s)
                                bad_samples = self._collect_bad_regressions(buffer, self._memory_manager.cfg.bad_top_f)
                                
                                improvement_mutation = None    # for improvement trigger
                                
                                if is_fresh_improvement and buffer._best_solution_global:
                                    best_solution = buffer._best_solution_global
                                    if best_solution.parent_id is not None and parent_solutions:
                                        found_match = False
                                        for parent_sol in parent_solutions:
                                            if parent_sol.solution_id == best_solution.parent_id:
                                                improvement_mutation = {
                                                    "parent_expression": parent_sol.expression,
                                                    "parent_mse": parent_sol.mse,
                                                    "child_expression": best_solution.expression,
                                                    "child_mse": best_solution.mse,
                                                }
                                                found_match = True
                                                break
                                        
                                        if not found_match:

                                            log.warning("Best solution generated in Gen %d but parent not found in current buffer. Parent ID: %s", 
                                                      generation, best_solution.parent_id)
                                
                                trigger_type = "improvement" if is_fresh_improvement else "epoch"
                                insights = None
                                try:
                                    insights = await self._async_summarize_normal_memory(
                                        self._memory_manager,
                                        problem,
                                        good_samples,
                                        bad_samples,
                                        generation,
                                        trigger_type,
                                        improvement_mutation=improvement_mutation,
                                        debug_path=self._memory_debug_path,
                                    )
                                except Exception as e:

                                    log.warning("Normal memory summarization failed with exception: %s", e, exc_info=True)
                                    insights = None
                                
                                if self._memory_manager.current_memory_type != "stagnation" or improved:
                                    self._memory_manager.current_memory_type = "normal"
                                    self._memory_section = self._memory_manager.format_memory_section()
                                
                                if insights:
                                    self._memory_manager.add_insights(insights, memory_type="normal")
                            
                            elif trigger_stag and current_best is not None:
                                log.info("Stagnation memory summary triggered at generation %d", generation)
                                
                                top_k_samples = self._collect_top_solutions(buffer, self._memory_manager.cfg.stagnation_top_k)
                                
                                top_n_solutions = self._collect_top_solution_objects(buffer, self._memory_manager.cfg.stagnation_residual_analysis_top_n)
                                
                                stagnation_start_generation = self._memory_manager.last_improvement_generation
                                failed_mutations = self._collect_failed_mutations(
                                    buffer,
                                    stagnation_start_generation=stagnation_start_generation,
                                    max_count=self._memory_manager.cfg.stagnation_failed_mutations,
                                    parent_top_percent=self._memory_manager.cfg.stagnation_failed_parent_top_percent
                                )
                                
                                insights = None
                                try:
                                    insights = await self._async_summarize_stagnation_memory(
                                        self._memory_manager,
                                        problem,
                                        top_k_samples,
                                        top_n_solutions,
                                        failed_mutations,
                                        generation,
                                        debug_path=self._memory_debug_path,
                                    )
                                except Exception as e:

                                    log.warning("Stagnation memory summarization failed with exception: %s", e, exc_info=True)
                                    insights = None
                                
                                if insights:
                                    self._memory_manager.add_insights(insights, memory_type="stagnation")
                                    self._memory_manager.current_memory_type = "stagnation"
                                    self._memory_section = self._memory_manager.format_memory_section()
                                
                                self._memory_manager.last_stagnation_trigger_generation = generation
                        
                        return results, prep_time
                    
                    async_gen_start_time = time.perf_counter()
                    if verbose_timing:
                        log.debug("[Generation %d] async generation of %d samples...",
                                 generation, evolution_config.samples_per_prompt)
                    generation_result = loop.run_until_complete(run_generation())
                    async_gen_end_time = time.perf_counter()
                    async_gen_time = async_gen_end_time - async_gen_start_time
                    
                    if isinstance(generation_result, tuple) and len(generation_result) == 2:
                        results, prep_time = generation_result
                    else:
                        results = generation_result
                        prep_time = None
                    
                    if verbose_timing:
                        llm_times = []
                        process_times = []
                        register_times = []
                        
                        for result in results:
                            if isinstance(result, Exception):
                                continue
                            llm_interaction = result.get("llm_interaction")
                            if llm_interaction and "timing" in llm_interaction:
                                timing = llm_interaction["timing"]
                                if "llm_time" in timing:
                                    llm_times.append(timing["llm_time"])
                                if "process_time" in timing:
                                    process_times.append(timing["process_time"])
                                if "register_time" in timing:
                                    register_times.append(timing["register_time"])
                        
                        log.debug("\n[Generation %d] timing:", generation)
                        if prep_time is not None:
                            log.debug("  - prep (sample parents + build prompt): %.3f s", prep_time)
                        if llm_times:
                            avg_llm = np.mean(llm_times)
                            max_llm = np.max(llm_times)
                            log.debug("  - LLM (mean/max): %.3f s / %.3f s", avg_llm, max_llm)
                        if process_times:
                            avg_process = np.mean(process_times)
                            max_process = np.max(process_times)
                            log.debug("  - parse+optimize+MSE (mean/max): %.3f s / %.3f s", avg_process, max_process)
                        if register_times:
                            avg_register = np.mean(register_times)
                            log.debug("  - register (mean): %.3f s", avg_register)
                        log.debug("  - total (parallel wall): %.3f s (per sample: %.3f s)",
                                 async_gen_time, async_gen_time/evolution_config.samples_per_prompt)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            log.error("Error in async generation: %s", result, exc_info=True)
                            continue
                        
                        if result.get("llm_interaction"):
                            generation_log["llm_interactions"].append(result["llm_interaction"])
                        
                        new_solutions_count += result.get("new_solutions_count", 0)
                    
                    for island in buffer._islands:
                        if island.best_solution:
                            generation_log["island_best_solutions"].append({
                                "island_id": island.island_id,
                                "expression": island.best_solution.expression,
                                "mse": island.best_solution.mse,
                                "r2": island.best_solution.r2,
                                "generation": island.best_solution.generation,
                                "eval_set_name": island.best_solution.eval_set_name
                            })
                    
                    generation_log["solutions_generated"] = new_solutions_count
                    if buffer._best_solution_global:
                        generation_log["best_mse_so_far"] = buffer._best_mse_global
                    
                    detailed_log["generations"].append(generation_log)
                    
                    generation_end_time = time.perf_counter()
                    generation_total_time = generation_end_time - generation_start_time
                    
                    log.info("-"*30)
                    log.info("Pid %s Generation %d: Generated %d/%d valid solutions", 
                            problem.original_idx, generation, new_solutions_count, evolution_config.samples_per_prompt)
                    log.info("[Pid %s Generation %d] wall time: %.3f s",
                            problem.original_idx, generation, generation_total_time)
                    if buffer._best_solution_global:
                        log.info("  Best MSE so far: %.6e", buffer._best_mse_global)
                    log.info("="*30)
                    if verbose_timing:
                        log.debug("[Generation %d] wall time: %.3f s", generation, generation_total_time)
                        log.debug("%s\n", '='*60)
                    
                    if evolution_config.reset_period > 0 and generation % evolution_config.reset_period == 0:
                        log.info("  Pid %s Resetting weak islands at generation %d...", 
                                problem.original_idx, generation)
                        buffer.reset_weak_islands()
            
            finally:
                loop.close()
            
            best_solutions = []
            final_results = {
            "best_solutions": [],
            "global_best": None,
            "unique_best": None
            }
            
            for island in buffer._islands:
                if island.best_solution:
                    best_solutions.append(island.best_solution.search_result)
                    final_results["best_solutions"].append({
                        "island_id": island.island_id,
                        "expression": island.best_solution.expression,
                        "mse": island.best_solution.mse,
                        "r2": island.best_solution.r2,
                        "generation": island.best_solution.generation,
                        "eval_set_name": island.best_solution.eval_set_name
                    })
            
            unique_best_solution = None
            if buffer._best_solution_global:
                unique_best_solution = buffer._best_solution_global
                final_results["global_best"] = {
                    "expression": buffer._best_solution_global.expression,
                    "mse": buffer._best_solution_global.mse,
                    "r2": buffer._best_solution_global.r2,
                    "generation": buffer._best_solution_global.generation,
                    "eval_set_name": buffer._best_solution_global.eval_set_name
                }
                log.info("Using global best solution (MSE: %.6e)", buffer._best_solution_global.mse)
            else:
                candidates = [island.best_solution for island in buffer._islands if island.best_solution]
                if candidates:
                    unique_best_solution = min(candidates, key=lambda s: s.mse)
                    log.info("No global best solution found. Selected best from %d islands (MSE: %.6e)", 
                            len(candidates), unique_best_solution.mse)
                else:
                    log.warning("No valid solutions found in any island.")
            
            if unique_best_solution:
                final_results["unique_best"] = {
                    "expression": unique_best_solution.expression,
                    "mse": unique_best_solution.mse,
                    "r2": unique_best_solution.r2,
                    "generation": unique_best_solution.generation,
                    "source": "global_best" if buffer._best_solution_global else "islands_best",
                    "eval_set_name": unique_best_solution.eval_set_name
                }
            
            detailed_log["final_results"] = final_results
            
            final_token_stats = self.client.get_token_stats()
            detailed_log["total_token_usage"] = {
                "prompt_tokens": final_token_stats["prompt_tokens"] - initial_token_stats["prompt_tokens"],
                "completion_tokens": final_token_stats["completion_tokens"] - initial_token_stats["completion_tokens"],
                "total_tokens": final_token_stats["total_tokens"] - initial_token_stats["total_tokens"]
            }
            
            if unique_best_solution:
                result = [unique_best_solution.search_result]
                result[0].aux["_detailed_log"] = detailed_log
                result[0].aux["method"] = "se_evolution"
                result[0].aux["total_token_usage"] = detailed_log["total_token_usage"]
                
                total_token_usage = detailed_log["total_token_usage"]
                log.info("Evolution completed. Returning unique best solution.")
                log.info("Total token usage: %d tokens (prompt: %d, completion: %d)", 
                        total_token_usage['total_tokens'], total_token_usage['prompt_tokens'], total_token_usage['completion_tokens'])
                return result
            else:

                total_token_usage = detailed_log["total_token_usage"]
                log.warning("Evolution completed. No valid solutions found.")
                log.info("Total token usage: %d tokens (prompt: %d, completion: %d)", 
                        total_token_usage['total_tokens'], total_token_usage['prompt_tokens'], total_token_usage['completion_tokens'])
                return []
        except bdb.BdbQuit:
            log.info("Debugger quit in evolve(). Terminating program...")
            sys.exit(0)

