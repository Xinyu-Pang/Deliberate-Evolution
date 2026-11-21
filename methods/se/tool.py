import numpy as np
from scipy.stats import pearsonr, spearmanr, kurtosis
import sympy


import numpy as np
from scipy.stats import pearsonr, spearmanr
import itertools

def analyze_train_samples(train_samples, output_var, input_vars) -> str:
    """
    Analyze train_samples and generate data_prompt.
    Enhanced to detect interactions (x*v) and specific shapes (sin, log).
    """
    try:
        # Extract data
        y_train = train_samples[:, 0]  # Output variable
        X_train = train_samples[:, 1:]  # Input variables
        
        n_samples = len(train_samples)
        n_inputs = X_train.shape[1]
        
        data_prompt = [] # Use list for better formatting
        
        for i, var in enumerate(input_vars):
            x_data = X_train[:, i]
            x_min, x_max = np.min(x_data), np.max(x_data)
            
            if x_min * x_max < 0:
                data_prompt.append(f"- Input {var} crosses zero (contains both positive and negative); avoid log({var}) or sqrt({var}).")
            
            if np.any(np.isclose(x_data, 0, atol=1e-6)):
                data_prompt.append(f"- Input {var} contains zero; avoid division 1/{var} or log({var}).")

        for i, var in enumerate(input_vars):
            x_data = X_train[:, i]
            
            pearson_corr = abs(pearsonr(x_data, y_train)[0])
            spearman_corr = abs(spearmanr(x_data, y_train)[0])
            
            transforms = {
                "linear": x_data,
                "squared": x_data**2,
                "sin": np.sin(x_data),
                "cos": np.cos(x_data),
                "exp_neg": np.exp(-np.abs(x_data)) if np.all(np.abs(x_data) < 10) else np.zeros_like(x_data)
            }
            if np.all(x_data > 0):
                transforms["log"] = np.log(x_data)
            
            best_trans_name = "none"
            best_trans_corr = 0
            
            for name, trans_x in transforms.items():
                try:
                    corr = abs(pearsonr(trans_x, y_train)[0])
                    if corr > best_trans_corr:
                        best_trans_corr = corr
                        best_trans_name = name
                except:
                    continue

            if best_trans_corr > 0.6:
                if best_trans_name == "linear":
                    data_prompt.append(f"- Input {var} has strong linear relationship with output (corr: {best_trans_corr:.2f}).")
                elif best_trans_name == "squared":
                    data_prompt.append(f"- Input {var} has parabolic/symmetric relationship (corr with {var}^2: {best_trans_corr:.2f}). Consider {var}^2 or abs({var}).")
                elif best_trans_name in ["sin", "cos"]:
                    data_prompt.append(f"- Input {var} shows PERIODIC relationship (corr with {best_trans_name}: {best_trans_corr:.2f}). Strongly consider sin({var}) or cos({var}).")
                elif best_trans_name == "log":
                    data_prompt.append(f"- Input {var} shows logarithmic relationship. Consider log({var}).")
            
            elif spearman_corr > 0.85 and pearson_corr < 0.6:
                 data_prompt.append(f"- Input {var} is strictly monotonic but highly non-linear. Consider exponentiation or powers.")

        if n_inputs >= 2:
            potential_interactions = []
            for (idx1, var1), (idx2, var2) in itertools.combinations(enumerate(input_vars), 2):
                interaction = X_train[:, idx1] * X_train[:, idx2]
                
                inter_corr = abs(pearsonr(interaction, y_train)[0])
                
                var1_corr = abs(pearsonr(X_train[:, idx1], y_train)[0])
                var2_corr = abs(pearsonr(X_train[:, idx2], y_train)[0])
                max_single_corr = max(var1_corr, var2_corr)
                
                if inter_corr > 0.5 and inter_corr > (max_single_corr + 0.15):
                    potential_interactions.append(f"{var1}*{var2} (corr: {inter_corr:.2f})")
            
            if potential_interactions:
                data_prompt.append(f"- STRONG INTERACTION DETECTED: The output is highly correlated with the product of variables: {', '.join(potential_interactions)}. You MUST include coupled terms (e.g., {potential_interactions[0]}).")
        
        # y_abs = np.abs(y_train)
        # magnitude_ratio = np.max(y_abs) / (np.min(y_abs) + 1e-9)
        # if magnitude_ratio > 1e4:
        #      data_prompt.append(f"- Output {output_var} spans multiple orders of magnitude (>10^4). This could indicate exponential growth or singularities (division by near-zero). Consider terms like exp(), BUT ALSO investigate Rational Functions (division) like 1/(...term...).")


        
        y_abs = np.abs(y_train)
        magnitude_ratio = np.max(y_abs) / (np.min(y_abs) + 1e-9)
        
        if magnitude_ratio > 1e3:
            structure_hints = []
            
            try:
                log_y = np.log(y_abs + 1e-9)
                from scipy.stats import kurtosis
                kurt_log = abs(kurtosis(log_y))
                
                max_log_corr = 0
                for i in range(X_train.shape[1]):
                    c = abs(pearsonr(X_train[:, i], log_y)[0])
                    if c > max_log_corr: max_log_corr = c
                    
                if max_log_corr > 0.8 or kurt_log < 2.0:
                    structure_hints.append(("Exponential", max_log_corr))
            except:
                pass

            try:
                recip_y = 1.0 / (y_train + 1e-9)
                
                max_recip_corr = 0
                is_periodic_denominator = False
                
                for i in range(X_train.shape[1]):
                    x_feat = X_train[:, i]
                    c_sin = abs(pearsonr(np.sin(x_feat), recip_y)[0])
                    c_cos = abs(pearsonr(np.cos(x_feat), recip_y)[0])
                    c_lin = abs(pearsonr(x_feat, recip_y)[0])
                    
                    current_max = max(c_sin, c_cos, c_lin)
                    if current_max > max_recip_corr:
                        max_recip_corr = current_max
                    if max(c_sin, c_cos) > 0.4:
                        is_periodic_denominator = True
                
                recip_range = np.max(np.abs(recip_y))
                if max_recip_corr > 0.5 or (recip_range < 100 and magnitude_ratio > 1e4):
                    structure_hints.append(("Rational", max_recip_corr, is_periodic_denominator))
            except:
                pass

            
            final_suggestion = "- Output spans multiple orders of magnitude."
            
            exp_score = next((x[1] for x in structure_hints if x[0] == "Exponential"), 0)
            rat_entry = next((x for x in structure_hints if x[0] == "Rational"), None)
            rat_score = rat_entry[1] if rat_entry else 0
            
            if rat_score > exp_score:
                final_suggestion += " [Strong Hint] The RECIPROCAL of the target (1/y) is simpler than y itself."
                final_suggestion += " -> You should construct a RATIONAL FUNCTION (Division)."
                if rat_entry and rat_entry[2]: # is_periodic_denominator
                    final_suggestion += " \n  -> CRITICAL: The denominator likely contains PERIODIC terms (e.g., 1/(A*cos(t) + B))."
                else:
                    final_suggestion += " (e.g., 1/(Linear terms))."
            elif exp_score > rat_score:
                final_suggestion += " [Strong Hint] The LOGARITHM of the target is simple/linear."
                final_suggestion += " -> You should construct an EXPONENTIAL function (e.g., exp(A*t))."
            else:
                final_suggestion += " Both Exponential and Rational forms are possible.."
            
            data_prompt.append(final_suggestion)

        from scipy.stats import kurtosis
        if len(y_train) > 20 and kurtosis(y_train) > 10.0:
            data_prompt.append(f"- Output distribution has extremely heavy tails (High Kurtosis). This suggests a RESONANCE or SINGULARITY. Consider dividing by terms that cross zero (e.g., 1/(cos(t)+c) or 1/(x+c)).")
        
        if not data_prompt:
            data_prompt.append("No simple patterns found. The relationship might be complex or involve hidden parameters.")

        return "\n".join(data_prompt)
        
    except Exception as e:
        print(f"Error analyzing train_samples: {e}")
        return f"Analysis failed: {str(e)}"

import numpy as np
from scipy.stats import pearsonr, spearmanr

def analyze_fitting_performance(
        train_samples,
        input_vars,
        output_var,
        y_pred
    ) -> str:
    """
    Advanced Residual Analysis Tool
    Now includes:
    1. Linear/Quadratic check (Existing)
    2. Transcendental Function Scanning (log, exp, inv, sin) -> Fixes missing log(v), 1/x
    3. Periodic Pattern Detection -> Fixes missing sin(t)
    4. Interaction Term Detection -> Fixes missing x*v
    5. Heteroscedasticity Check (Existing)
    """
    try:
        X_train = train_samples[:, 1:]
        y_train = train_samples[:, 0]
        
        # Basic Validation
        if np.any(np.isnan(y_pred)):
            return "Fitting analysis: prediction contains nan. Please check the expression."
        if np.any(np.isinf(y_pred)):
            return "Fitting analysis: prediction contains inf. Please check the expression."
        
        # Calculate Residuals
        residuals = y_pred - y_train
        abs_residuals = np.abs(residuals)
        
        fitting_analysis = []
        
        # --- Strategy 1: Standard Missing Variable (Linear/Poly) ---
        for i, var in enumerate(input_vars):
            x_data = X_train[:, i]
            
            # Pearson for Linear
            corr, _ = pearsonr(x_data, residuals)
            if abs(corr) > 0.5:
                fitting_analysis.append(f"- Residuals linearly correlate with {var}. May missing term: +c*{var}.")
            
            # Polynomial check (Quadratic)
            # Fit parabola
            if len(np.unique(x_data)) > 5: # Only if enough unique points
                try:
                    z = np.polyfit(x_data, residuals, 2)
                    p = np.poly1d(z)
                    r_quad = residuals - p(x_data)
                    var_orig = np.var(residuals)
                    var_quad = np.var(r_quad)
                    if var_quad < 0.5 * var_orig: 
                        fitting_analysis.append(f"- Residuals show parabolic shape with {var}. May missing term: +c*{var}**2.")
                except:
                    pass

        # --- Strategy 2: Transcendental & Periodic Scanning ---
        # We transform input x and check if correlation with residuals IMPROVES
        
        for i, var in enumerate(input_vars):
            x_data = X_train[:, i]
            
            # Avoid domain errors
            valid_mask = ~np.isnan(x_data) & ~np.isinf(x_data)
            x_safe = x_data[valid_mask]
            r_safe = residuals[valid_mask]
            
            if len(x_safe) < 10: continue

            # Define transformations to test
            transforms = {
                "sin": np.sin(x_safe),
                "cos": np.cos(x_safe),
                "log": np.log(np.abs(x_safe) + 1e-6), # Safe log
                "exp_neg": np.exp(-np.abs(x_safe)),   # Decay
                "inv": 1.0 / (np.abs(x_safe) + 1e-6)  # Inverse
            }
            
            best_trans_name = None
            best_trans_corr = 0
            
            for name, trans_x in transforms.items():
                try:
                    corr, _ = pearsonr(trans_x, r_safe) # Check linear relation with the transformed feature
                    
                    if abs(corr) > abs(best_trans_corr):
                        best_trans_corr = corr
                        best_trans_name = name
                except:
                    continue
            
            # Threshold for suggesting a specific function
            if abs(best_trans_corr) > 0.4:
                if best_trans_name in ["sin", "cos"]:
                    fitting_analysis.append(f"- Residuals correlate with {best_trans_name}({var}) (corr: {best_trans_corr:.2f}). Suggests missing periodic term involving {var}. Try adding {best_trans_name}({var}).")
                elif best_trans_name == "log":
                    fitting_analysis.append(f"- Residuals correlate with log({var}) (corr: {best_trans_corr:.2f}). Suggests missing logarithmic term.")
                elif best_trans_name == "exp_neg":
                    fitting_analysis.append(f"- Residuals correlate with exp(-|{var}|) (corr: {best_trans_corr:.2f}). Suggests missing decay term.")
                elif best_trans_name == "inv":
                    fitting_analysis.append(f"- Residuals correlate with 1/{var} (corr: {best_trans_corr:.2f}). Suggests trying RATIONAL term: try adding c/{var} or putting {var} in the DENOMINATOR.")

        # --- Strategy 3: Interaction Scanning (The Fix for PO17: x*v) ---
        # Check pairwise products of inputs
        import itertools
        if len(input_vars) > 1:
            for (idx1, var1), (idx2, var2) in itertools.combinations(enumerate(input_vars), 2):
                # x * v
                interaction_term = X_train[:, idx1] * X_train[:, idx2]
                corr, _ = pearsonr(interaction_term, residuals)
                
                if abs(corr) > 0.4:
                     fitting_analysis.append(f"- Residuals correlate with interaction {var1}*{var2} (corr: {corr:.2f}). Missing COUPLED term like {var1}*{var2}.")

        # --- Strategy 4: Heteroscedasticity (Existing but refined) ---
        # Check if error grows with predicted value
        hetero_corr, p_value = spearmanr(abs_residuals, y_pred)
        
        if p_value < 0.05 and abs(hetero_corr) > 0.3:
            if hetero_corr > 0:
                fitting_analysis.append(f"- [Critical] Error EXPLODES as value increases (Heteroscedasticity corr: {hetero_corr:.2f}). Suggestion: Your current function (likely Polynomial or simple Exp) grows too slowly or too smoothly. Try RATIONAL FUNCTIONS (Division) to model the singularity (e.g., 1/(input+c)).")
            else:
                fitting_analysis.append(f"- Error is large near zero(Heteroscedasticity corr: {hetero_corr:.2f}). Suggestion: Check for singularity or INVERSE terms or detailed periodic behavior near the origin.")

        # Final Assembly
        if not fitting_analysis:
            return "No obvious simple pattern found in residuals. The error might be random noise or high-frequency."
        
        return "Fitting Analysis for expression:\n" + "\n".join(fitting_analysis)

    except Exception as e:
        return f"Fitting analysis failed: {str(e)}"


