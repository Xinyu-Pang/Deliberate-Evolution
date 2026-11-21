"""
Symbolic Regression Tool Group

Provides tools for LLM to analyze data and expressions during symbolic regression.
Each problem gets one ToolGroup instance that holds the problem reference.
"""
from typing import Optional, List
from bench.dataclasses import Problem
from methods.se.evolution import EvolvedSolution
from methods.se.tool import analyze_train_samples, analyze_fitting_performance


class SymbolicRegressionToolGroup:
    
    def __init__(self, problem: Problem, searcher=None):
        self.problem = problem
        self.output_var = problem.gt_equation.symbols[0]
        self.input_vars = problem.gt_equation.symbols[1:]
        self.train_samples = problem.train_samples
        self.searcher = searcher
        valid_samples = problem.samples.get('valid', None) if hasattr(problem, 'samples') else None
        if valid_samples is not None:
            self.valid_samples = valid_samples
        else:
            self.valid_samples = problem.train_samples
    
    def reset(self, problem: Problem, searcher=None):
        self.problem = problem
        self.output_var = problem.gt_equation.symbols[0]
        self.input_vars = problem.gt_equation.symbols[1:]
        self.train_samples = problem.train_samples
        self.searcher = searcher
        valid_samples = problem.samples.get('valid', None)
        if valid_samples is not None:
            self.valid_samples = valid_samples
        else:

            self.valid_samples = problem.train_samples
    
    def analyze_data(self) -> str:
        try:
            return analyze_train_samples(
                self.train_samples,
                self.output_var,
                self.input_vars
            )
        except Exception as e:

            return f"Error analyzing data: {str(e)}"
    
    def analyze_residuals(
        self, 
        parent_solutions: Optional[List[EvolvedSolution]] = None
    ) -> str:
        try:
            results = []
            
            if parent_solutions is not None and len(parent_solutions) > 0:
                for idx, solution in enumerate(parent_solutions):
                    
                    y_pred = None
                    if (solution.y_pred is not None and 
                        solution.eval_set_name == "validation"):
                        y_pred = solution.y_pred
                    else:

                        if self.searcher is None:
                            results.append(f"Solution {idx + 1}: Searcher not available. Cannot evaluate expression.")
                            continue
                        
                        lambda_fn = self.searcher.expression_to_lambda(solution.expression, self.problem)
                        if lambda_fn is None:
                            results.append(f"Solution {idx + 1}: Failed to convert expression '{solution.expression}' to lambda function.")
                            continue
                        
                        X_eval = eval_samples[:, 1:]
                        try:
                            y_pred = lambda_fn(X_eval)
                        except Exception as e:
                            results.append(f"Solution {idx + 1}: Failed to evaluate expression '{solution.expression}': {str(e)}")
                            continue
                    
                    analysis_result = analyze_fitting_performance(
                        self.valid_samples,
                        self.input_vars,
                        self.output_var,
                        y_pred
                    )
                    results.append(analysis_result)
                
                return "\n".join(results)
            else:

                return "Error: Must provide either 'expression' or 'parent_solutions'."
        except Exception as e:

            return f"Error analyzing residuals: {str(e)}"


    def dimensional_analysis(self) -> str:
        """v1 placeholder: return a fixed response without checking dimensions."""
        return "Valid."

    def execute_tool(self, tool_name: str, parent_solutions: Optional[List[EvolvedSolution]] = None) -> str:
        if tool_name == "analyze_data":
            return self.analyze_data()
        elif tool_name == "analyze_residuals":
            return self.analyze_residuals(parent_solutions=parent_solutions)
        elif tool_name == "dimensional_analysis":
            return self.dimensional_analysis()
        else:
            raise ValueError(f"Unknown tool: {tool_name}. Available tools: analyze_data, analyze_residuals, dimensional_analysis")

