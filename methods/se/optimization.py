"""Parameter optimization utilities for SE method."""

import numpy as np
import time
import threading
from typing import List, Dict, Optional, Tuple, Callable
from scipy.optimize import minimize
from sympy import symbols, lambdify, parse_expr, Add, Mul, Function, Pow
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication,
    convert_xor
)
from bench.dataclasses import Problem


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def _substitute_undefined_symbols(expr):
    """
    Substitute undefined symbols in SymPy expression with their mathematical constant equivalents.
    
    Common substitutions:
    - 'e' -> E (natural base)
    - 'pi' -> pi (mathematical constant)
    
    Args:
        expr: SymPy expression that may contain undefined symbols
    
    Returns:
        SymPy expression with symbols substituted
    """
    from sympy import E, pi, Symbol
    
    # Check if expression is valid
    if expr is None:
        raise ValueError("Expression is None")
    
    # Check if expression has free symbols
    try:
        free_symbols = expr.free_symbols if hasattr(expr, 'free_symbols') else set()
    except Exception as e:
        # If we can't get free symbols, the expression is likely invalid
        raise ValueError(f"Cannot get free symbols from expression: {expr}, error: {e}")
    
    # Build substitution dictionary for common undefined symbols
    substitutions = {}
    for sym in free_symbols:
        sym_name = str(sym)
        # Replace 'e' with E (natural base)
        if sym_name == 'e':
            substitutions[sym] = E
        # Replace 'pi' with pi (already a constant, but handle if it's a symbol)
        elif sym_name == 'pi':
            substitutions[sym] = pi
    
    # Apply substitutions if any
    if substitutions:
        try:
            # Use a timeout mechanism to avoid infinite loops
            # Note: SymPy's subs can be slow for complex expressions, but shouldn't hang
            # If it does hang, it's likely due to an invalid expression structure
            result = expr.subs(substitutions)
            # Verify the result is still a valid expression
            if result is None:
                raise ValueError("Substitution returned None")
            return result
        except Exception as e:
            # If substitution fails, raise an error with context
            raise ValueError(f"Failed to substitute symbols in expression: {expr}, substitutions: {substitutions}, error: {e}")
    return expr


def _convert_result_to_array(result) -> np.ndarray:
    """
    Convert various result types (numpy array, SymPy objects, strings) to numpy float array.
    
    Args:
        result: Result from lambdify function, can be:
            - numpy.ndarray (with numeric, string, or SymPy elements)
            - SymPy expression
            - scalar value
            - iterable of any of the above
    
    Returns:
        numpy.ndarray of float dtype
    """
    from sympy import N as sympy_N
    
    # Case 1: result is already a numpy array
    if isinstance(result, np.ndarray):
        # Check if elements are already numeric
        try:
            return result.astype(float)
        except (TypeError, ValueError):
            # Elements are not numeric - could be strings or SymPy objects
            if result.size == 0:
                return result.astype(float)
            
            # Check first element to determine type
            first_elem = result.flat[0]
            
            if isinstance(first_elem, str):
                # Elements are strings - try to parse as SymPy expressions
                try:
                    # Use SymPy's E for natural base, pi for pi
                    # local_dict maps names to symbols/values
                    from sympy import E as sympy_E, pi as sympy_pi
                    local_dict = {'e': sympy_E, 'E': sympy_E, 'pi': sympy_pi, 'Pi': sympy_pi}
                    converted = []
                    for elem in result.flat:
                        try:
                            # Try to parse and evaluate the string expression
                            parsed = parse_expr(
                                elem.replace("^", "**"),
                                transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,),
                                local_dict=local_dict
                            )
                            # Evaluate the parsed expression to a float
                            converted.append(float(sympy_N(parsed)))
                        except Exception as parse_err:
                            # If parsing fails, try direct float conversion
                            try:
                                converted.append(float(elem))
                            except Exception:
                                raise ValueError(f"Cannot parse or convert string element: {elem}, parse error: {parse_err}")
                    return np.array(converted, dtype=float).reshape(result.shape)
                except Exception as e:
                    raise ValueError(f"Cannot convert string array to float: {result}, error: {e}")
            
            elif hasattr(first_elem, 'evalf'):
                # Elements are SymPy objects
                try:
                    # First substitute undefined symbols (like 'e' -> E), then evaluate
                    converted = []
                    for x in result.flat:
                        try:
                            # Substitute undefined symbols before evaluation
                            substituted = _substitute_undefined_symbols(x)
                            converted.append(float(sympy_N(substituted)))
                        except Exception as sub_err:
                            # If substitution fails for a single element, try to evaluate without substitution
                            try:
                                converted.append(float(sympy_N(x)))
                            except Exception as eval_err:
                                # If both substitution and direct evaluation fail, raise with context
                                raise ValueError(f"Cannot convert SymPy element to float: {x}, substitution error: {sub_err}, evaluation error: {eval_err}")
                    return np.array(converted, dtype=float).reshape(result.shape)
                except Exception as e:
                    raise ValueError(f"Cannot convert SymPy array to float: {result}, error: {e}")
            
            else:
                # Unknown type, try direct conversion
                try:
                    return result.astype(float)
                except Exception as e:
                    raise ValueError(f"Cannot convert array elements to float: {result}, type: {type(first_elem)}, error: {e}")
    
    # Case 2: result is a SymPy expression
    elif hasattr(result, 'evalf'):
        try:
            # Substitute undefined symbols before evaluation
            substituted = _substitute_undefined_symbols(result)
            return np.array([float(sympy_N(substituted))], dtype=float)
        except Exception as e:
            raise ValueError(f"Cannot convert SymPy expression to float: {result}, error: {e}")
    
    # Case 3: result is iterable (list, tuple, etc.)
    elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
        try:
            # Check first element
            first_elem = next(iter(result))
            
            if isinstance(first_elem, str):
                # Elements are strings
                from sympy import E as sympy_E, pi as sympy_pi
                local_dict = {'e': sympy_E, 'E': sympy_E, 'pi': sympy_pi, 'Pi': sympy_pi}
                converted = []
                for elem in result:
                    try:
                        parsed = parse_expr(
                            elem.replace("^", "**"),
                            transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,),
                            local_dict=local_dict
                        )
                        converted.append(float(sympy_N(parsed)))
                    except Exception as parse_err:
                        try:
                            converted.append(float(elem))
                        except Exception:
                            raise ValueError(f"Cannot parse or convert string element: {elem}, parse error: {parse_err}")
                return np.array(converted, dtype=float)
            
            elif hasattr(first_elem, 'evalf'):
                # Elements are SymPy objects
                # First substitute undefined symbols, then evaluate
                converted = []
                for x in result:
                    substituted = _substitute_undefined_symbols(x)
                    converted.append(float(sympy_N(substituted)))
                return np.array(converted, dtype=float)
            
            else:
                # Try direct conversion
                return np.array([float(x) for x in result], dtype=float)
        
        except StopIteration:
            # Empty iterable
            return np.array([], dtype=float)
        except Exception as e:
            raise ValueError(f"Cannot convert iterable to float array: {result}, error: {e}")
    
    # Case 4: scalar value
    else:
        try:
            return np.array([float(result)], dtype=float)
        except Exception as e:
            raise ValueError(f"Cannot convert scalar to float: {result}, type: {type(result)}, error: {e}")


def _build_local_dict(input_vars: List[str], additional_symbols: Optional[List[str]] = None) -> Dict:
    """
    Build a local_dict for parse_expr to treat specified names as symbols instead of functions.
    
    Args:
        input_vars: List of input variable names
        additional_symbols: Additional symbol names that might conflict with SymPy functions
                          (e.g., 'gamma', 'lambda', 'I', 'beta', 'alpha', 'omega')
    
    Returns:
        Dictionary mapping symbol names to SymPy Symbol objects
    """
    from sympy import Symbol
    
    local_dict = {}
    
    # Add all input variables as symbols
    for var in input_vars:
        local_dict[var] = Symbol(var)
    
    # Add additional symbols that might conflict with SymPy built-ins
    # Common conflicts: gamma (Gamma function), lambda (Python keyword), I (imaginary unit), etc.
    conflict_symbols = additional_symbols or []
    common_conflicts = ['gamma', 'lambda', 'I', 'beta', 'alpha', 'omega', 'zeta', 'psi']
    
    for sym_name in conflict_symbols + common_conflicts:
        if sym_name not in local_dict:  # Don't override input vars
            local_dict[sym_name] = Symbol(sym_name)
    
    return local_dict


def _get_math_namespace() -> Dict:
    """
    Get math function namespace for lambdify to ensure all functions are correctly mapped.
    
    Returns:
        Dictionary mapping function names to numpy functions
        
    Note:
        This namespace is used with lambdify's modules parameter. When lambdify is called with
        modules=[math_namespace, 'numpy'], it will first look up functions in math_namespace,
        then fall back to numpy module if not found. This ensures sqrt, sin, cos, etc. are
        correctly bound and available when the generated lambda function executes.
    """
    return {
        'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp, 'log': np.log, 'abs': np.abs, 'Abs': np.abs,  # Map both lowercase and uppercase Abs
        'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'pi': np.pi, 'e': np.e,
        'np': np  # Include numpy for safety (allows np.sqrt, np.sin, etc. in expressions)
    }


def extract_constants_from_expression(infix_expr: str, input_vars: List[str]) -> List[str]:
    """
    Extract constant symbols from a mathematical expression.
    
    Args:
        infix_expr: Infix expression string (e.g., "A*x + B*y + C")
        input_vars: List of input variable names (e.g., ['x', 'y', 'z'])
        
    Returns:
        List of constant symbols found in the expression (e.g., ['A', 'B', 'C'])
        
    Examples:
        >>> extract_constants_from_expression("A*x + B*y", ['x', 'y'])
        ['A', 'B']
        >>> extract_constants_from_expression("sin(x) + cos(y)", ['x', 'y'])
        []
    """
    try:
        # Build local_dict to treat all variables and common conflict symbols as symbols
        local_dict = _build_local_dict(input_vars)
        
        # Parse expression to SymPy expression
        expr = parse_expr(
            infix_expr.replace("^", "**"), 
            transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,),
            local_dict=local_dict
        )
        
        # Get all symbols in the expression
        all_symbols = expr.free_symbols
        
        # Filter out input variables and built-in functions
        builtin_functions = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 
                            'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh'}
        
        # Also exclude common mathematical constants
        math_constants = {'pi', 'e', 'E', 'I', 'oo', 'zoo', 'nan'}
        
        constants = []
        for sym in all_symbols:
            sym_name = str(sym)
            if (sym_name not in input_vars and 
                sym_name.lower() not in builtin_functions and
                sym_name.lower() not in math_constants):
                constants.append(sym_name)
        
        # Sort to ensure consistent order
        constants.sort()
        return constants
        
    except Exception as e:
        print(f"Error extracting constants: {e}")
        return []


def inject_parameters_into_expression(
    infix_expr: str,
    input_vars: List[str],
    max_params: int = 10
) -> Tuple[str, List[str]]:
    """
    Automatically inject parameter placeholders into an expression when no constants are found.
    This allows optimization even when the expression skeleton doesn't contain explicit constants.
    Recursively processes function arguments to inject parameters into nested expressions.
    
    Args:
        infix_expr: Infix expression string
        input_vars: List of input variable names
        max_params: Maximum number of parameters to inject (default: 10)
        
    Returns:
        Tuple of (modified_infix_expr, list_of_parameter_symbols)
        
    Examples:
        >>> inject_parameters_into_expression("x * y + z", ["x", "y", "z"])
        ("c0 * x * y + c1 * z", ["c0", "c1"])
        >>> inject_parameters_into_expression("M * n_rho * mom", ["M", "n_rho", "mom"])
        ("c0 * M * n_rho * mom", ["c0"])
        >>> inject_parameters_into_expression("x*exp(-alpha*omega*t)", ["x", "t"])
        ("c0*x*exp(-c1*alpha*omega*t) + c2", ["c0", "c1", "c2"])
    """
    try:
        # Build local_dict to treat all variables and common conflict symbols as symbols
        local_dict = _build_local_dict(input_vars)
        
        # Parse expression to SymPy expression
        expr = parse_expr(
            infix_expr.replace("^", "**"),
            transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,),
            local_dict=local_dict
        )
        
        constants = []
        param_counter = 0
        
        def create_param_symbol():
            """Create a new parameter symbol"""
            nonlocal param_counter
            if param_counter >= max_params:
                return None
            param_name = f"c{param_counter}"
            param_counter += 1
            constants.append(param_name)
            return symbols(param_name)
        
        def has_non_input_variables(sub_expr):
            """Check if sub-expression contains symbols that are not input variables"""
            if not hasattr(sub_expr, 'free_symbols'):
                return False
            free_syms = sub_expr.free_symbols
            for sym in free_syms:
                if str(sym) not in input_vars:
                    return True
            return False
        
        def recursively_inject_params(sub_expr, is_function_arg=False):
            """
            Recursively inject parameters into expression.
            
            Args:
                sub_expr: SymPy expression to process
                is_function_arg: Whether this expression is an argument of a function
            """
            # If we've reached max parameters, return as-is
            if param_counter >= max_params:
                return sub_expr
            
            # Handle Add: inject parameters for each term
            if isinstance(sub_expr, Add):
                paramized_terms = []
                for term in sub_expr.args:
                    paramized_term = recursively_inject_params(term, is_function_arg)
                    paramized_terms.append(paramized_term)
                return Add(*paramized_terms)
            
            # Handle Mul: check if it contains non-input variables or is a complex product
            elif isinstance(sub_expr, Mul):
                # If this is a function argument, check if we should inject a parameter
                if is_function_arg:
                    # Strategy: inject parameter if:
                    # 1. Contains non-input variables, OR
                    # 2. Is a product of multiple factors (not just a single variable)
                    should_inject = False
                    if has_non_input_variables(sub_expr):
                        should_inject = True
                    elif len(sub_expr.args) > 1:
                        # Check if it's not just a single input variable
                        # If it has multiple factors, it's likely a product that should be parameterized
                        should_inject = True
                    
                    if should_inject:
                        param_symbol = create_param_symbol()
                        if param_symbol is not None:
                            return param_symbol * sub_expr
                
                # Otherwise, recursively process each factor
                paramized_factors = []
                for factor in sub_expr.args:
                    paramized_factor = recursively_inject_params(factor, is_function_arg)
                    paramized_factors.append(paramized_factor)
                return Mul(*paramized_factors)
            
            # Handle function calls (exp, sin, cos, etc.): recursively process arguments
            elif isinstance(sub_expr, Function):
                # Get function name and arguments
                func_name = sub_expr.func.__name__ if hasattr(sub_expr.func, '__name__') else str(sub_expr.func)
                args = sub_expr.args
                
                # Recursively process each argument
                paramized_args = []
                for arg in args:
                    paramized_arg = recursively_inject_params(arg, is_function_arg=True)
                    paramized_args.append(paramized_arg)
                
                # Reconstruct function with parameterized arguments
                return sub_expr.func(*paramized_args)
            
            # Handle Pow: recursively process base and exponent
            elif isinstance(sub_expr, Pow):
                base = recursively_inject_params(sub_expr.base, is_function_arg)
                exp = recursively_inject_params(sub_expr.exp, is_function_arg)
                return Pow(base, exp)
            
            # For other structures (symbols, numbers, etc.), return as-is
            else:
                return sub_expr
        
        # Apply recursive parameter injection
        paramized_expr = recursively_inject_params(expr, is_function_arg=False)
        
        def is_parameter_symbol(sym):
            """Check if a symbol is a parameter (c0, c1, etc.)"""
            if sym is None:
                return False
            sym_str = str(sym)
            # Check if it matches pattern c0, c1, c2, etc.
            return sym_str.startswith('c') and len(sym_str) > 1 and sym_str[1:].isdigit()
        
        # Strategy 1: If top-level expression is an Add, ensure each term has a coefficient
        if isinstance(paramized_expr, Add):
            terms = paramized_expr.args
            final_terms = []
            
            for term in terms:
                # Check if term already has a parameter coefficient
                if isinstance(term, Mul) and term.args:
                    first_factor = term.args[0]
                    if is_parameter_symbol(first_factor):
                        # Already has parameter, keep as-is
                        final_terms.append(term)
                    else:
                        # Add parameter coefficient
                        param_symbol = create_param_symbol()
                        if param_symbol is not None:
                            final_terms.append(param_symbol * term)
                        else:
                            final_terms.append(term)
                else:
                    # Add parameter coefficient
                    param_symbol = create_param_symbol()
                    if param_symbol is not None:
                        final_terms.append(param_symbol * term)
                    else:
                        final_terms.append(term)
            
            # Add a constant term if possible
            if param_counter < max_params:
                param_symbol = create_param_symbol()
                if param_symbol is not None:
                    final_terms.append(param_symbol)
            
            paramized_expr = Add(*final_terms)
        
        # Strategy 2: If top-level expression is a Mul, add a coefficient at the front
        elif isinstance(paramized_expr, Mul):
            # Check if first factor is already a parameter
            if paramized_expr.args:
                first_factor = paramized_expr.args[0]
                if not is_parameter_symbol(first_factor):
                    param_symbol = create_param_symbol()
                    if param_symbol is not None:
                        paramized_expr = param_symbol * paramized_expr
            else:
                param_symbol = create_param_symbol()
                if param_symbol is not None:
                    paramized_expr = param_symbol * paramized_expr
            
            # Add a constant term if possible
            if param_counter < max_params:
                param_symbol = create_param_symbol()
                if param_symbol is not None:
                    paramized_expr = paramized_expr + param_symbol
                    
        # Strategy 3: For other structures (functions, powers, etc.), wrap with coefficient
        else:
            param_symbol = create_param_symbol()
            if param_symbol is not None:
                paramized_expr = param_symbol * paramized_expr
            
            # Add a constant term if possible
            if param_counter < max_params:
                param_symbol = create_param_symbol()
                if param_symbol is not None:
                    paramized_expr = paramized_expr + param_symbol
        
        # Convert back to string
        paramized_str = str(paramized_expr)
        
        return paramized_str, constants
        
    except Exception as e:
        print(f"Error injecting parameters: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: add a single coefficient parameter
        try:
            param_symbol = "c0"
            paramized_str = f"{param_symbol} * ({infix_expr})"
            return paramized_str, [param_symbol]
        except:
            return infix_expr, []


def create_parameterized_lambda(
    infix_expr: str, 
    input_vars: List[str], 
    constants: List[str]
) -> Optional[Callable]:
    """
    Create a parameterized lambda function that accepts input variables and parameters.
    
    Args:
        infix_expr: Infix expression string
        input_vars: List of input variable names
        constants: List of constant symbols to be optimized
        
    Returns:
        Lambda function: f(X, *params) where X is (N, len(input_vars)) array
        Returns None if creation fails
    """
    try:
        # Build local_dict to treat all variables and common conflict symbols as symbols
        local_dict = _build_local_dict(input_vars, constants)
        
        # Replace Abs with abs before parsing to avoid SymPy Abs objects
        # This ensures lambdify can properly convert to numpy abs function
        infix_expr_clean = infix_expr.replace("^", "**").replace("Abs", "abs")
        
        # Parse expression
        expr = parse_expr(
            infix_expr_clean,
            transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,),
            local_dict=local_dict
        )
        
        # Convert to numeric function using lambdify
        # Arguments: [input_vars..., constants...]
        # Use explicit math namespace to ensure sqrt, sin, cos, etc. are correctly bound
        math_namespace = _get_math_namespace()
        param_func = lambdify(
            [symbols(var) for var in input_vars + constants],
            expr,
            modules=[math_namespace, 'numpy']
        )
        
        # Wrap in a function that handles array inputs
        def parameterized_fn(X: np.ndarray, *params: float) -> np.ndarray:
            """
            X: shape (N, len(input_vars))
            params: values for constants in order
            Returns: shape (N,)
            """
            if len(params) != len(constants):
                raise ValueError(f"Expected {len(constants)} parameters, got {len(params)}")
            
            # Prepare arguments: X columns + params
            args = [X[:, i] for i in range(len(input_vars))] + list(params)
            result = param_func(*args)
            
            # Handle different result types: numpy array, SymPy objects, or string arrays
            result_array = _convert_result_to_array(result)
            
            return result_array
        
        return parameterized_fn
        
    except Exception as e:
        print(f"Error creating parameterized lambda: {e}")
        return None


def optimize_parameters(
    parameterized_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    initial_params: Optional[np.ndarray] = None,
    method: str = 'L-BFGS-B',
    bounds: Optional[List[Tuple[float, float]]] = None,
    options: Optional[Dict] = None,
    timeout: float = 30.0  # Maximum time in seconds for optimization
) -> Tuple[np.ndarray, float, Dict]:
    """
    Optimize parameters using gradient-based optimization.
    
    Args:
        parameterized_fn: Parameterized function f(X, *params) -> y_pred
        X: Input data array, shape (N, n_features)
        y: Target values, shape (N,)
        initial_params: Initial parameter values (default: random values)
        method: Optimization method (default: 'L-BFGS-B')
        bounds: List of (min, max) tuples for each parameter (optional)
        options: Additional options for optimizer (e.g., {'maxiter': 1000})
        timeout: Maximum time in seconds for optimization (default: 30.0)
        
    Returns:
        Tuple of (optimized_params, final_loss, optimization_info)
    """
    # Track optimization start time
    start_time = time.time()
    optimization_timed_out = threading.Event()
    
    # Default initial parameters: small random values
    if initial_params is None:
        # Try to infer number of parameters by testing the function
        # Start with a reasonable guess and adjust
        test_n_params = 10  # Maximum reasonable number of constants
        test_params = np.random.randn(test_n_params) * 0.1
        
        # Try to determine actual number by testing the function
        actual_n_params = test_n_params
        try:
            # Check timeout before testing
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout during parameter inference")
            test_result = parameterized_fn(X[:1], *test_params)
            # If successful, we can use test_n_params
        except TimeoutError:
            raise
        except:
            # Try with fewer parameters
            for n in range(1, test_n_params + 1):
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timeout during parameter inference")
                try:
                    test_result = parameterized_fn(X[:1], *test_params[:n])
                    actual_n_params = n
                    break
                except:
                    continue
            test_params = test_params[:actual_n_params]
        
        initial_params = test_params
    
    # Default options - reduce maxiter to avoid long runs
    if options is None:
        options = {'maxiter': 1000, 'disp': False}  # Reduced from 100 to 50
    
    def loss_function(params: np.ndarray) -> float:
        """Compute MSE loss with timeout check"""
        # Check if optimization has timed out
        if optimization_timed_out.is_set() or (time.time() - start_time) > timeout:
            optimization_timed_out.set()
            return float('inf')
        
        try:
            # Add timeout check before calling parameterized_fn
            if (time.time() - start_time) > timeout:
                optimization_timed_out.set()
                return float('inf')
            
            y_pred = parameterized_fn(X, *params)
            
            # Check timeout after function call
            if (time.time() - start_time) > timeout:
                optimization_timed_out.set()
                return float('inf')
            
            # Handle potential NaN or Inf values
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return float('inf')
            mse = np.mean((y_pred - y) ** 2)
            return mse
        except TimeoutError:
            optimization_timed_out.set()
            return float('inf')
        except Exception as e:
            # Don't print every error to avoid spam, only print first few
            if not hasattr(loss_function, '_error_count'):
                loss_function._error_count = 0
            loss_function._error_count += 1
            if loss_function._error_count <= 3:
                print(f"Error in loss_function: {e}")
            return float('inf')
    
    # Run optimization
    try:
        # Check timeout before starting optimization
        if (time.time() - start_time) > timeout:
            raise TimeoutError("Timeout before optimization started")
        
        result = minimize(
            loss_function,
            initial_params,
            method=method,
            bounds=bounds,
            options=options
        )
        
        # Check if we timed out during optimization
        if optimization_timed_out.is_set() or (time.time() - start_time) > timeout:
            raise TimeoutError(f"Optimization timed out after {timeout} seconds")
        
        optimized_params = result.x
        final_loss = result.fun
        
        optimization_info = {
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit,
            'n_evaluations': result.nfev
        }
        
        return optimized_params, final_loss, optimization_info
        
    except TimeoutError as e:
        print(f"Optimization timeout: {e}")
        return initial_params, float('inf'), {'success': False, 'error': str(e), 'timeout': True}
    except Exception as e:
        print(f"Error during optimization: {e}")
        return initial_params, float('inf'), {'success': False, 'error': str(e)}


def substitute_optimized_params(
    infix_expr: str,
    input_vars: List[str],
    constants: List[str],
    optimized_params: np.ndarray
) -> str:
    """
    Substitute optimized parameter values back into the expression string.
    
    Args:
        infix_expr: Original infix expression string
        input_vars: List of input variable names
        constants: List of constant symbols
        optimized_params: Optimized parameter values
    
    Returns:
        Modified expression string with constants replaced by their values
    """
    try:
        # Build local_dict to treat all variables and constants as symbols
        local_dict = _build_local_dict(input_vars, constants)
        
        # Replace Abs with abs before parsing to avoid SymPy Abs objects
        infix_expr_clean = infix_expr.replace("^", "**").replace("Abs", "abs")
        
        expr = parse_expr(
            infix_expr_clean,
            transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,),
            local_dict=local_dict
        )
        
        # Create substitution dictionary
        substitutions = {}
        for const_name, param_value in zip(constants, optimized_params):
            # Round to reasonable precision
            param_value_rounded = round(float(param_value), 6)
            substitutions[symbols(const_name)] = param_value_rounded
        
        # Substitute values
        expr_substituted = expr.subs(substitutions)
        
        # Convert back to string
        result_str = str(expr_substituted)
        
        # Replace Abs with abs in the string representation for lambdify compatibility
        # This ensures that when the expression is parsed again, it uses abs() instead of Abs()
        result_str = result_str.replace('Abs', 'abs')
        
        # Clean up formatting (remove unnecessary parentheses from numbers)
        return result_str
        
    except Exception as e:
        print(f"Error substituting parameters: {e}")
        return infix_expr


def optimize_expression_parameters(
    lambda_fn: Callable,
    expression: str,
    problem: Problem,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None
) -> Tuple[Optional[Callable], str, Dict]:
    """
    Complete workflow: extract constants, optimize them, and create optimized lambda function.
    
    Args:
        lambda_fn: Original lambda function
        expression: Original expression string (infix)
        problem: Problem object containing data
        X_train: Training input data (if None, uses problem.train_samples)
        y_train: Training target data (if None, uses problem.train_samples)
        
    Returns:
        Tuple of (optimized_lambda_fn, optimized_expression_str, optimization_info)
    """
    try:
        # Get input variables
        input_vars = problem.gt_equation.symbols[1:]
        
        infix_expr = expression
        
        # Extract constants
        constants = extract_constants_from_expression(infix_expr, input_vars)
        
        # If no constants found, automatically inject parameters into expression
        expression_modified = False
        if len(constants) == 0:
            print(f"No constants found in expression. Auto-injecting parameters...")
            # Inject parameter placeholders into the expression
            infix_expr, constants = inject_parameters_into_expression(infix_expr, input_vars)
            expression_modified = True
            print(f"Injected {len(constants)} parameters: {constants}")
            print(f"Modified expression: {infix_expr}")
        
        # Get training data
        if X_train is None or y_train is None:
            train_samples = problem.train_samples
            X_train = train_samples[:, 1:]  # Input variables
            y_train = train_samples[:, 0]   # Output variable
        
        # Create parameterized function
        parameterized_fn = create_parameterized_lambda(infix_expr, input_vars, constants)
        
        if parameterized_fn is None:
            return lambda_fn, expression, {'constants_found': len(constants), 'optimized': False, 'error': 'Failed to create parameterized function'}
        
        # Optimize parameters with timeout
        # initial_params = np.random.randn(len(constants)) * 0.1
        initial_params = np.ones(len(constants))
        try:
            optimized_params, final_loss, opt_info = optimize_parameters(
                parameterized_fn,
                X_train,
                y_train,
                initial_params=initial_params,
                timeout=30.0  # 30 second timeout
            )
            
            # Check if optimization failed or timed out
            if opt_info.get('timeout', False) or not opt_info.get('success', False):
                print(f"Optimization failed or timed out. Returning unoptimized expression.")
                return lambda_fn, expression, {
                    'constants_found': len(constants),
                    'optimized': False,
                    'auto_parameterized': expression_modified,
                    'error': opt_info.get('error', 'Optimization failed'),
                    'timeout': opt_info.get('timeout', False)
                }
        except TimeoutError as e:
            print(f"Optimization timeout: {e}. Returning unoptimized expression.")
            return lambda_fn, expression, {
                'constants_found': len(constants),
                'optimized': False,
                'auto_parameterized': expression_modified,
                'error': str(e),
                'timeout': True
            }
        except Exception as e:
            print(f"Error during optimization: {e}. Returning unoptimized expression.")
            return lambda_fn, expression, {
                'constants_found': len(constants),
                'optimized': False,
                'auto_parameterized': expression_modified,
                'error': str(e)
            }
        
        # Substitute optimized parameters back into expression
        try:
            optimized_expr = substitute_optimized_params(
                infix_expr,
                input_vars,
                constants,
                optimized_params
            )
        except Exception as e:
            print(f"Error substituting optimized parameters: {e}. Returning unoptimized expression.")
            return lambda_fn, expression, {
                'constants_found': len(constants),
                'optimized': False,
                'auto_parameterized': expression_modified,
                'error': f'Failed to substitute parameters: {str(e)}'
            }
        
        # Create optimized lambda function
        try:
            # Build local_dict to treat all variables as symbols
            local_dict = _build_local_dict(input_vars)
            
            # Use lambdify instead of eval to properly handle mathematical functions
            # This ensures sqrt, sin, cos, etc. are correctly bound
            # Replace Abs with abs before parsing to avoid SymPy Abs objects
            optimized_expr_clean = optimized_expr.replace("^", "**").replace("Abs", "abs")
            expr_parsed = parse_expr(
                optimized_expr_clean,
                transformations=standard_transformations + (implicit_multiplication,) + (convert_xor,),
                local_dict=local_dict
            )
            
            # Convert to numeric function using lambdify (same approach as create_parameterized_lambda)
            # Use explicit math namespace to ensure sqrt, sin, cos, etc. are correctly bound
            math_namespace = _get_math_namespace()
            optimized_lambda_base = lambdify(
                [symbols(var) for var in input_vars],
                expr_parsed,
                modules=[math_namespace, 'numpy']
            )
            
            # Wrap it to accept array input (X: shape (n_samples, n_features))
            def optimized_lambda_fn(X: np.ndarray) -> np.ndarray:
                """
                X: shape (n_samples, len(input_vars))
                Returns: shape (n_samples,)
                """
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                # Extract columns and pass as separate arguments
                args = [X[:, i] for i in range(len(input_vars))]
                result = optimized_lambda_base(*args)
                
                # Handle SymPy objects: convert to float/numpy array
                result_array = _convert_result_to_array(result)
                
                return result_array
                
        except Exception as e:
            print(f"Error creating optimized lambda: {e}. Returning unoptimized expression.")
            return lambda_fn, expression, {
                'constants_found': len(constants),
                'optimized': False,
                'auto_parameterized': expression_modified,
                'error': f'Failed to create optimized lambda: {str(e)}'
            }
        
        optimization_info = {
            'constants_found': len(constants),
            'constants': constants,
            'optimized_params': optimized_params.tolist(),
            'final_loss': final_loss,
            'optimized': True,
            'auto_parameterized': expression_modified,  # Flag indicating if parameters were auto-injected
            **opt_info
        }
        
        return optimized_lambda_fn, optimized_expr, optimization_info
        
    except TimeoutError as e:
        print(f"Timeout in optimize_expression_parameters: {e}. Returning unoptimized expression.")
        return lambda_fn, expression, {'error': str(e), 'optimized': False, 'timeout': True}
    except Exception as e:
        print(f"Error in optimize_expression_parameters: {e}. Returning unoptimized expression.")
        return lambda_fn, expression, {'error': str(e), 'optimized': False}

