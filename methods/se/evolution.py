"""Evolution algorithm data structures and components for SE method."""

import numpy as np
import threading
import bisect
from typing import List, Optional
from dataclasses import dataclass
from bench.dataclasses import SearchResult

# Import config from unified config module


@dataclass
class EvolvedSolution:
    expression: str
    lambda_fn: callable
    search_result: SearchResult
    mse: float
    r2: float  # R2 score
    generation: int
    parent_id: Optional[str] = None
    solution_id: str = None
    eval_set_name: str = "validation"
    y_pred: Optional[np.ndarray] = None
    
    selection_count: int = 0
    best_child_mse: float = float('inf')
    children_count: int = 0
    
    def __post_init__(self):
        if self.solution_id is None:
            self.solution_id = f"sol_{id(self)}"
    
    def compute_fitness(self, config, rank: Optional[int] = None, population_size: Optional[int] = None) -> float:
        if rank is not None and population_size is not None and config.fitness_mode == "rank":
            # F_MSE = 1.0 - rank / (population_size + 1)
            normalized_rank = rank / max(population_size, 1)
            F_MSE = np.exp(-normalized_rank / config.rank_decay_factor)
        else:
            F_MSE = 1.0 / (1.0 + self.mse)
        
        # F_Hist = w_Sample * F_Sample + w_Elite * F_Elite
        
        if self.selection_count == 0:
            F_Sample = 1.0
        else:
            F_Sample = np.exp(-self.selection_count / config.selection_decay_factor)
        
        if self.children_count == 0:
            F_Elite = 0.0
            F_Hist = F_Sample
        else:
            if self.best_child_mse < float('inf'):
                if self.mse > 1e-9:
                    relative_improvement = (self.mse - self.best_child_mse) / (self.mse + config.improvement_epsilon)
                    F_Elite = max(0.0, relative_improvement)
                elif self.mse < 0:

                    print(f"Invalid MSE: {self.mse}")
                else:
                    # improvement = max(0.0, self.mse - self.best_child_mse)
                    # F_Elite = 1.0 / (1.0 + improvement) if improvement > 0 else 0.0
                    F_Elite = 0.0
            else:
                F_Elite = 0.0
        
            F_Hist = config.w_sample * F_Sample + config.w_elite * F_Elite
        
        F_Comp = 0.0
        
        F_total = (config.w_mse * F_MSE + 
                  config.w_hist * F_Hist + 
                  config.w_comp * F_Comp)
        
        return max(0.0, F_total)
    
    @property
    def fitness(self) -> float:
        return 1.0 / (1.0 + self.mse)


@dataclass
class Prompt:
    island_id: int
    parent_solutions: List[EvolvedSolution]


class Island:
    
    def __init__(self, island_id: int, max_population_size: int = 50):
        self.island_id = island_id
        self.max_population_size = max_population_size
        self.population: List[EvolvedSolution] = []
        self._best_solution: Optional[EvolvedSolution] = None
        self._best_mse: float = float('inf')
    
    def add_solution(self, solution: EvolvedSolution) -> bool:
        for i, existing_sol in enumerate(self.population):
            if existing_sol.expression == solution.expression:
                if solution.mse < existing_sol.mse:
                    self.population.pop(i)
                    mse_values = [s.mse for s in self.population]
                    insert_pos = bisect.bisect_left(mse_values, solution.mse)
                    self.population.insert(insert_pos, solution)
                    if solution.mse < self._best_mse:
                        self._best_solution = solution
                        self._best_mse = solution.mse
                    return True
                else:
                    return False
        
        mse_values = [s.mse for s in self.population]
        insert_pos = bisect.bisect_left(mse_values, solution.mse)
        self.population.insert(insert_pos, solution)
        
        if solution.mse < self._best_mse:
            self._best_solution = solution
            self._best_mse = solution.mse
        
        if len(self.population) > self.max_population_size:
            self.population.pop()
        
        return True
    
    def _trim_population(self):
        self.population = self.population[:self.max_population_size]
    
    def get_rank(self, solution: EvolvedSolution) -> Optional[int]:
        try:
            idx = bisect.bisect_left([s.mse for s in self.population], solution.mse)
            for i in range(idx, len(self.population)):
                if self.population[i].solution_id == solution.solution_id:
                    return i
            return None
        except:
            return None
    
    def find_solution_by_id(self, solution_id: str) -> Optional[EvolvedSolution]:
        for sol in self.population:
            if sol.solution_id == solution_id:
                return sol
        return None
    
    def sample_parents(self, num_parents: int = 1, method: str = "fitness_proportional", config=None, generation: Optional[int] = None, is_stagnated: bool = False) -> List[EvolvedSolution]:
        if len(self.population) == 0:
            return []
        
        num_parents = min(num_parents, len(self.population))
        
        if method == "fitness_proportional":
            if config is not None:
                if config.fitness_mode == "rank":
                    fitnesses = np.array([
                        s.compute_fitness(config, rank=i, population_size=len(self.population))
                        for i, s in enumerate(self.population)
                    ])
                else:

                    fitnesses = np.array([s.compute_fitness(config) for s in self.population])
            else:

                fitnesses = np.array([s.fitness for s in self.population])
            
            if config is not None and config.use_boltzmann_sampling:
                if is_stagnated:
                    temperature = config.stagnation_escape_temperature
                else:
                    if generation is not None and config.max_generations > 0:
                        if config.temperature_schedule == "linear":
                            temperature = config.initial_temperature * (1.0 - generation / max(config.max_generations, 1))
                            temperature = max(temperature, config.min_temperature)
                        elif config.temperature_schedule == "exponential":
                            temperature = config.initial_temperature * (config.cooling_rate ** generation)
                            temperature = max(temperature, config.min_temperature)
                        else:  # "fixed"
                            temperature = config.initial_temperature
                    else:
                        temperature = config.initial_temperature
                
                if temperature > 1e-6:
                    fitnesses_normalized = fitnesses - np.max(fitnesses)
                    boltzmann_weights = np.exp(fitnesses_normalized / temperature)
                    probabilities = boltzmann_weights / boltzmann_weights.sum()
                else:
                    max_idx = np.argmax(fitnesses)
                    probabilities = np.zeros(len(self.population))
                    probabilities[max_idx] = 1.0
            else:

                if fitnesses.sum() == 0:
                    probabilities = np.ones(len(self.population)) / len(self.population)
                else:
                    probabilities = fitnesses / fitnesses.sum()
            
            indices = np.random.choice(len(self.population), size=num_parents, p=probabilities, replace=False)
            selected = [self.population[i] for i in indices]
            
            for sol in selected:
                sol.selection_count += 1
            
            return selected
        
        elif method == "tournament":
            selected = []
            for _ in range(num_parents):
                k = min(3, len(self.population))
                tournament_indices = np.random.choice(len(self.population), size=k, replace=False)
                tournament = [self.population[i] for i in tournament_indices]
                winner = min(tournament, key=lambda s: s.mse)
                selected.append(winner)
            
            for sol in selected:
                sol.selection_count += 1
            
            return selected
        
        elif method == "top_k":
            selected = self.population[:num_parents]
            
            for sol in selected:
                sol.selection_count += 1
            
            return selected
        
        else:
            indices = np.random.choice(len(self.population), size=num_parents, replace=False)
            selected = [self.population[i] for i in indices]
            
            for sol in selected:
                sol.selection_count += 1
            
            return selected
    
    def sample_from_top_k(self, k: int, exclude_solutions: Optional[List[EvolvedSolution]] = None) -> Optional[EvolvedSolution]:
        if len(self.population) == 0:
            return None
        
        top_k_size = min(k, len(self.population)) if k > 0 else len(self.population)
        top_k_solutions = self.population[:top_k_size]
        
        if exclude_solutions:
            exclude_ids = {sol.solution_id for sol in exclude_solutions}
            available_solutions = [sol for sol in top_k_solutions if sol.solution_id not in exclude_ids]
        else:
            available_solutions = top_k_solutions
        
        if len(available_solutions) == 0:
            return None
        
        return np.random.choice(available_solutions)
    
    @property
    def best_solution(self) -> Optional[EvolvedSolution]:
        return self._best_solution


class SEExperienceBuffer:
    
    def __init__(self, config):
        self.config = config
        self._islands: List[Island] = []
        for i in range(config.num_islands):
            self._islands.append(Island(i, config.population_size_per_island))
        self._best_solution_global: Optional[EvolvedSolution] = None
        self._best_mse_global: float = float('inf')
        self._lock = threading.Lock()
    
    def get_prompt(self) -> Prompt:
        with self._lock:
            island_id = np.random.randint(len(self._islands))
            island = self._islands[island_id]
            
            parent_solutions = island.sample_parents(
                num_parents=self.config.functions_per_prompt,
                method=self.config.selection_method,
                config=self.config
            )
            
            return Prompt(
                island_id=island_id,
                parent_solutions=parent_solutions
            )
    
    def register_solution(self, solution: EvolvedSolution, island_id: int) -> bool:
        with self._lock:
            added = self._islands[island_id].add_solution(solution)
            
            if added and solution.mse < self._best_mse_global:
                self._best_solution_global = solution
                self._best_mse_global = solution.mse
            
            if added and solution.parent_id is not None:
                parent = self._islands[island_id].find_solution_by_id(solution.parent_id)
                if parent is not None:
                    parent.children_count += 1
                    if solution.mse < parent.best_child_mse:
                        parent.best_child_mse = solution.mse
        
        return added
    
    def reset_weak_islands(self):
        island_scores = []
        for i, island in enumerate(self._islands):
            if island.best_solution:
                island_scores.append((i, island.best_solution.mse))
            else:
                island_scores.append((i, float('inf')))
        
        island_scores.sort(key=lambda x: x[1])
        
        num_to_reset = max(1, len(self._islands) // 2)
        reset_islands = island_scores[:num_to_reset]
        keep_islands = island_scores[num_to_reset:]
        
        if not keep_islands:
            return
        
        for island_id, _ in reset_islands:
            keep_island_ids = [x[0] for x in keep_islands]
            founder_island_id = np.random.choice(keep_island_ids)
            founder = self._islands[founder_island_id].best_solution
            
            if founder:
                self._islands[island_id] = Island(island_id, self.config.population_size_per_island)
                self._islands[island_id].add_solution(founder)
