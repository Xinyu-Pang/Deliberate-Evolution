import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Import config from unified config module
from methods.se.config import MemoryConfig

log = logging.getLogger(__name__)

@dataclass
class MemoryManager:
    cfg: MemoryConfig
    normal_memory: List[str] = field(default_factory=list)
    stagnation_memory: List[str] = field(default_factory=list)
    
    current_memory_type: Optional[str] = None
    
    last_best_mse: Optional[float] = None
    last_improvement_generation: Optional[int] = None
    
    best_mse: Optional[float] = None
    stagnation_len: int = 0
    last_stagnation_trigger_generation: Optional[int] = None
    
    @property
    def memory(self) -> List[str]:
        if self.current_memory_type == "stagnation":
            return self.stagnation_memory
        return self.normal_memory

    def update_best(self, best_mse: Optional[float], generation: Optional[int] = None) -> bool:
        if not self.cfg.enabled or best_mse is None:
            return False
        if self.best_mse is None or (self.best_mse - best_mse) > self.cfg.min_improvement:
            self.last_best_mse = self.best_mse
            self.best_mse = best_mse
            if generation is not None:
                self.last_improvement_generation = generation
            else:

                print("No generation provided!!")
            self.stagnation_len = 0
            return True
        self.stagnation_len += 1
        return False

    def should_trigger_normal_epoch(self, generation: int) -> bool:
        return self.cfg.enabled and self.cfg.normal_epoch_interval > 0 and generation > 0 and generation % self.cfg.normal_epoch_interval == 0
    
    def should_trigger_normal_improvement(self) -> bool:
        if not self.cfg.enabled or self.best_mse is None or self.last_best_mse is None:
            return False
        if self.cfg.improvement_threshold <= 0:
            return False
        improvement_ratio = (self.last_best_mse - self.best_mse) / self.last_best_mse
        return improvement_ratio >= self.cfg.improvement_threshold
    
    def is_stagnated(self) -> bool:
        return self.cfg.enabled and self.cfg.stagnation_k > 0 and self.stagnation_len >= self.cfg.stagnation_k
    
    def should_trigger_stagnation(self, generation: int) -> bool:
        if not self.cfg.enabled or self.cfg.stagnation_k <= 0:
            return False
        
        if self.stagnation_len < self.cfg.stagnation_k:
            return False
        
        if self.last_stagnation_trigger_generation is not None:
            cooldown_period = self.cfg.stagnation_cooldown
            if cooldown_period > 0:
                generations_since_trigger = generation - self.last_stagnation_trigger_generation
                if generations_since_trigger < cooldown_period:
                    return False
        
        return True
    
    def should_trigger_epoch(self, generation: int) -> bool:
        return self.should_trigger_normal_epoch(generation)

    def add_insights(self, insights: List[str], memory_type: str = "normal") -> None:
        clean = [i.strip() for i in insights if i and i.strip()]
        if not clean:
            return
        
        if memory_type == "stagnation":
            self.stagnation_memory.extend(clean)
            if self.cfg.max_entries and self.cfg.max_entries > 0:
                self.stagnation_memory = self.stagnation_memory[-self.cfg.max_entries:]
        else:  # normal
            self.normal_memory.extend(clean)
            if self.cfg.max_entries and self.cfg.max_entries > 0:
                self.normal_memory = self.normal_memory[-self.cfg.max_entries:]

    def format_memory_section(self) -> str:
        if not self.cfg.enabled:
            return ""
        
        if self.current_memory_type == "stagnation":
            memory_list = self.stagnation_memory
        else:
            memory_list = self.normal_memory
        
        if not memory_list:
            return ""
        
        header = "### Memory"
        lines = [f"{item}" for item in memory_list]
        return f"{header}\n" + "\n".join(lines)

    def build_normal_summary_prompt(
        self, 
        problem_desc: str, 
        good_samples: List[Dict[str, Any]], 
        bad_samples: List[Dict[str, Any]],
        improvement_mutation: Optional[Dict[str, Any]] = None
    ) -> str:
        good_part = "\n".join([f"- {s.get('expression', '')} | mse={s.get('mse'):.3e}" for s in good_samples]) or "- (none)"
        bad_part = "\n".join([f"- {s.get('expression', '')} | mse={s.get('mse'):.3e} | Δmse=+{s.get('delta_mse'):.3e}" for s in bad_samples]) or "- (none)"
        old_memory = "\n".join([f"- {m}" for m in self.normal_memory]) if self.normal_memory else "- (none)"
        
        prompt = (
            f"You are an evolutionary algorithm strategist for Symbolic Regression. Your goal is to guide the search by maintaining a list of high-level insights (Memory).\n"
            f"###Problem: {problem_desc}\n"
            f"###Existing memory:\n{old_memory}\n"
            f"###Good samples (top):\n{good_part}\n"
            f"###Bad regressions (worst deltas):\n{bad_part}\n"
        )
        
        if improvement_mutation:
            parent_expr = improvement_mutation.get('parent_expression', '')
            child_expr = improvement_mutation.get('child_expression', '')
            parent_mse = improvement_mutation.get('parent_mse', 0)
            child_mse = improvement_mutation.get('child_mse', 0)
            improvement = parent_mse - child_mse
            prompt += (
                f"###Successful mutation (triggered this summary):\n"
                f"- Parent: {parent_expr} | mse={parent_mse:.3e}\n"
                f"- Child: {child_expr} | mse={child_mse:.3e} | Δmse={improvement:.3e}\n"
            )
        
        prompt += f"###Your Task: Output {self.cfg.insights_n} concise, concrete insights (one per line, no numbering)."
        return prompt
    
    def build_stagnation_summary_prompt(
        self,
        problem_desc: str,
        top_k_samples: List[Dict[str, Any]],
        top_n_residual_analysis: str,
        failed_mutations: List[Dict[str, Any]],
        residual_analysis_count: int = 2
    ) -> str:
        top_k_part = "\n".join([f"- {s.get('expression', '')} | mse={s.get('mse'):.3e}" for s in top_k_samples]) or "- (none)"
        old_memory = "\n".join([f"- {m}" for m in self.stagnation_memory]) if self.stagnation_memory else "- (none)"
        
        failed_part = "\n".join([
            f"- Parent: {m.get('parent_expression', '')} | mse={m.get('parent_mse', 0):.3e} -> "
            f"Child: {m.get('child_expression', '')} | mse={m.get('child_mse', 0):.3e} "
            f"(no improvement)"
            for m in failed_mutations
        ]) or "- (none)"
        
        return (
            f"You are an evolutionary algorithm strategist for Symbolic Regression. The search appears to be stuck in a local optimum. Your goal is to generate insights that help escape this local optimum.\n"
            f"###Problem: {problem_desc}\n"
            f"###Existing stagnation memory:\n{old_memory}\n"
            f"###Current top {len(top_k_samples)} formulas:\n{top_k_part}\n"
            f"###Residual analysis of top {residual_analysis_count} formulas:\n{top_n_residual_analysis}\n"
            f"###Recent failed mutations (avoid repeating these):\n{failed_part}\n"
            f"###Your Task: Output {self.cfg.insights_n} concise, concrete insights on how to escape this local optimum (one per line, no numbering)."
        )
    
    def build_summary_prompt(self, problem_desc: str, good_samples: List[Dict[str, Any]], bad_samples: List[Dict[str, Any]]) -> str:
        return self.build_normal_summary_prompt(problem_desc, good_samples, bad_samples)

