
import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class SEEvolutionConfig:
    num_islands: int = 1
    functions_per_prompt: int = 1
    solutions_per_generation: int = 4
    samples_per_prompt: int = 4
    max_generations: int = 10
    population_size_per_island: int = 100
    selection_method: str = "fitness_proportional"
    reset_period: int = 0
    
    w_mse: float = 0.6
    w_hist: float = 0.3
    w_comp: float = 0.0
    
    w_sample: float = 0.4
    w_elite: float = 0.6
    
    selection_decay_factor: float = 4.0
    
    improvement_epsilon: float = 1e-8
    
    fitness_mode: str = "rank"
    rank_decay_factor: float = 0.3
    
    use_boltzmann_sampling: bool = True
    initial_temperature: float = 0.5
    min_temperature: float = 0.01
    cooling_rate: float = 0.95
    temperature_schedule: str = "exponential"
    stagnation_escape_temperature: float = 2.0

    memory_enabled: bool = True
    memory_epoch_interval: int = 20
    memory_improvement_threshold: float = 0.1
    memory_stagnation_k: int = 10
    memory_stagnation_cooldown: int = 5
    memory_stagnation_top_k: int = 5
    memory_stagnation_residual_analysis_top_n: int = 2
    memory_stagnation_failed_mutations: int = 3
    memory_stagnation_failed_parent_top_percent: float = 0.2
    memory_good_top_s: int = 3
    memory_bad_top_f: int = 3
    memory_insights_n: int = 3
    memory_max_entries: int = 1
    memory_min_improvement: float = 0.0
    memory_debug_save: bool = False


@dataclass
class MemoryConfig:
    enabled: bool = False
    
    normal_epoch_interval: int = 20
    improvement_threshold: float = 0.1
    good_top_s: int = 3
    bad_top_f: int = 3
    
    stagnation_k: int = 10
    stagnation_cooldown: int = 5
    stagnation_top_k: int = 5
    stagnation_residual_analysis_top_n: int = 2
    stagnation_failed_mutations: int = 3
    stagnation_failed_parent_top_percent: float = 0.2
    
    insights_n: int = 3
    max_entries: int = 1
    min_improvement: float = 0.0
    debug_save_memory: bool = False
    
    @property
    def epoch_interval(self) -> int:
        return self.normal_epoch_interval


@dataclass
class AdaptiveStrategyConfig:
    alpha: float = 0.1
    refine_suppression_factor: float = 0.05
    regenerate_boost: float = 0.25
    min_probability: float = 0.05
    max_reward_clip: float = 2.0
    min_decay_factor: float = 0.1


@dataclass
class SESearcherConfig:
    name: str = ""
    api_type: str = ""
    api_model: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    n: int = 1
    temperature: float = 0.8
    rank_percent_threshold: float = 20
    visit_threshold: int = 3
    visit_threshold_scale: float = 15.0
    enable_color_logging: bool = True
    log_level: int = logging.INFO
    enable_tools: bool = True
    max_tool_rounds: int = 4
    
    strategy_alpha: float = 0.1
    strategy_refine_suppression_factor: float = 0.05
    strategy_regenerate_boost: float = 0.25
    strategy_min_probability: float = 0.05
    strategy_max_reward_clip: float = 2.0
    strategy_min_decay_factor: float = 0.1

