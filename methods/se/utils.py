"""
Utility functions.
"""

from typing import Optional


def postfix_to_infix(postfix_expr: str, validate: bool = True) -> Optional[str]:
    if not postfix_expr or not postfix_expr.strip():
        return None
    
    try:
        # Split tokens
        tokens = postfix_expr.split()
        stack = []
        
        # Binary operators
        binary_ops = {'+', '-', '*', '/', '**', '^'}
        
        # Unary functions
        unary_funcs = {
            'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
            'log', 'ln', 'log10', 'log2',
            'exp',
            'sqrt',
            'abs',
            'atan', 'asin', 'acos', 'atan2',
        }
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            # Check if it's a number (integer or float, including negative)
            if _is_number(token):
                stack.append(token)
            
            # Check if it's a binary operator
            elif token in binary_ops:
                if len(stack) < 2:
                    if validate:
                        return None  # Invalid: not enough operands for binary op
                    else:
                        # Try to handle unary minus/plus
                        if token == '-' and len(stack) == 1:
                            operand = stack.pop()
                            stack.append(f"(-{operand})")
                        elif token == '+' and len(stack) == 1:
                            # Unary plus, keep the operand
                            pass
                        else:
                            return None
                else:
                    b = stack.pop()
                    a = stack.pop()
                    
                    # Handle different operators with proper formatting
                    if token in ('**', '^'):
                        # Exponentiation
                        stack.append(f"(({a})**({b}))")
                    elif token == '/':
                        # Division - need parentheses
                        stack.append(f"(({a}) / ({b}))")
                    else:
                        # Other binary operators: +, -, *
                        stack.append(f"(({a}) {token} ({b}))")
            
            # Check if it's a unary function
            elif token in unary_funcs:
                if len(stack) < 1:
                    return None  # Invalid: not enough operands for unary function
                
                operand = stack.pop()
                
                # Map function names (handle aliases)
                if token == 'ln':
                    func_name = 'log'
                else:
                    func_name = token
                
                stack.append(f"{func_name}({operand})")
            
            # Otherwise, treat as variable
            else:
                stack.append(token)
        
        # At the end, stack should contain exactly one element
        if len(stack) == 0:
            return None
        
        if validate and len(stack) != 1:
            return None
        
        result = stack[0]
        return result.strip() if result else None
        
    except Exception as e:
        # If any error occurs, return None
        return None


def _is_number(token: str) -> bool:
    """Check if a token represents a number."""
    try:
        float(token)
        return True
    except ValueError:
        return False


def validate_postfix(postfix_expr: str) -> bool:
    result = postfix_to_infix(postfix_expr, validate=True)
    return result is not None


def simplify_parens(infix_expr: str) -> str:
    # For now, return as-is (advanced simplification requires parsing tree)
    # This can be enhanced later with a proper expression parser
    return infix_expr

