#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from arch import arch_model
from scipy.stats import kstest, norm
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import jarque_bera


# In[2]:


def calculate_p_values(arima_model):
    """Calculate p-values for ARIMA model coefficients."""
    coefs = arima_model.params
    std_errors = np.sqrt(np.diag(arima_model.cov_params()))
    t_values = coefs / std_errors
    p_values = 2 * (1 - norm.cdf(np.abs(t_values)))
    return p_values


# **Choose best ARIMA model**
# 
# This function chooses an ARIM

# In[189]:


# prop da unn poco igual, coger independientes y luego mejor AIC

def select_best_arima_model(ts, max_p, d, max_q, threshold):
    """
    Select the best ARIMA model based on independence, homoskedasticity, and normality.
    If no models satisfy all three, fallback to independence and homoskedasticity.
    """
    results = []
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            print(f"{p}, {q}")
            try:
                # Fit ARIMA model
                model = ARIMA(ts, order=(p, d, q)).fit()

                # Extract residuals and fitted values
                residuals = model.resid
                fitted_values = model.fittedvalues

                # Calculate proportion of significant p-values
                p_values = calculate_p_values(model)
                prop = np.mean(p_values <= 0.05)

                # Independence of residuals
                ljungbox_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
                independence = ljungbox_result["lb_pvalue"].iloc[0] >= 0.05

                # AIC
                aic_value = model.aic
                print(f"ARMA({p}, {q}) - Proportion p-values: {prop:.2f} - Independence: {independence} - AIC: {aic_value:.2f}")

                # Record results
                results.append((
                    f"ARIMA({p},{q})", 
                    prop, 
                    independence, 
                    aic_value
                ))
            except Exception as e:
                print(f"Failed to fit ARIMA({p},{q}): {e}")


    # Convert to DataFrame for filtering and sorting
    results_df = pd.DataFrame(results, columns=["Model", "Proportion", "Independence", "AIC"])
    
    # Filter models that satisfy independence and proportion
    all_criteria_models = results_df[
        (results_df["Independence"] == True) &
        (results_df["Proportion"] > threshold)
    ]
    
    # If no models meet all three, fallback to independence and homoskedasticity
    if all_criteria_models.empty:
        fallback_models = results_df[
            (results_df["Proportion"] > threshold)
        ]
        if not fallback_models.empty:
            print("No independent residuals ARIMA model. Choosing lowest AIc")
            return fallback_models, results_df
        else:
            print("No suitable independent ARIMA model found.")
            return None, results_df
    else:
        print("Models satisfying significability and independence criteria.")
        return all_criteria_models, results_df


# In[ ]:


def choose_best_garch(ts, best_p, max_r, max_s, threshold):
    """
    Choose the best GARCH(p, q) model based on the number of verified hypotheses and AIC.
    
    Returns:
        best_model: The best GARCH model object.
        results_df: A DataFrame with all models and their diagnostics.
        best_r: The ARCH order of the best model.
        best_s: The GARCH order of the best model.
    """
    results = []  # Initialize results list
    best_model = None  # Initialize the best model
    best_r, best_s = None, None  # To store the best (r, s)

    for r in range(max_r + 1):
        for s in range(max_s + 1):
            try:
                # Fit the GARCH model
                garch_model = arch_model(ts, vol="Garch", p=r, q=s, mean="ARX", lags=best_p, dist="normal")
                garch_model = garch_model.fit(disp="off", x=exog )
                
                # Calculate diagnostics
                p_values = garch_model.pvalues
                prop = np.mean(p_values <= 0.05)  # Proportion of significant coefficients
                aic_value = garch_model.aic
                
                
                # Prepare residuals and conditional volatility
                residuals = garch_model.std_resid
                fitted_values = garch_model.conditional_volatility
                
                # Normality
                lilliefors_stat, lilliefors_pvalue = lilliefors(residuals) 
                
                # Independence
                ljungbox_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
                ljungbox_pvalue = ljungbox_result['lb_pvalue'].iloc[0]
                
                # Homoskedasticity
                if fitted_values is not None:
                    # Add constant to fitted values for regression model
                    aux_data = pd.DataFrame({
                        "residuals_squared": residuals**2,
                        "fitted_values": fitted_values})

                    aux_data = aux_data.replace([np.inf, -np.inf], np.nan).dropna()
                    fitted_values_with_const = sm.add_constant(aux_data["fitted_values"])
                    bp_test_stat, bp_pvalue, _, _ = het_breuschpagan(aux_data["residuals_squared"], fitted_values_with_const)
                else:
                    print("Heteroskedasticity test skipped (fitted values not provided).")
                
                # Count the number of verified hypotheses
                verified_hypotheses = sum([
                    lilliefors_pvalue > 0.05,  # Normality
                    ljungbox_pvalue > 0.05,  # Independence
                    bp_pvalue > 0.05  # Homoscedasticity
                ])
                
                # Append results
                results.append({
                    "Model": f"GARCH({r},{s})",
                    "Proportion_Significant": prop,
                    "AIC": aic_value,
                    "Normality": lilliefors_pvalue,
                    "Independence_pvalue": ljungbox_pvalue,
                    "Homoscedasticity": bp_pvalue,
                    "Verified_Hypotheses": verified_hypotheses,
                    "GARCH_Object": garch_model,  # Store the model object
                    "r": r,  # Store r
                    "s": s   # Store s
                })
                
                print(f"GARCH({r}, {s}) - Proportion p-values: {prop:.2f} - AIC: {aic_value:.2f}")
                print(f"Independent: {ljungbox_pvalue > 0.05} - Homoscedastic: {bp_pvalue > 0.05} - Normal: {lilliefors_pvalue > 0.05}")
            
            except Exception as e:
                print(f"Error fitting GARCH({r}, {s}): {e}")
    
    # Convert results to a DataFrame
    if results:
        results_df = pd.DataFrame(results)
        # Sort by the number of hypotheses verified, then by AIC
        results_df = results_df.sort_values(
            by=["Verified_Hypotheses", "AIC"], ascending=[False, True]
        )
        # Select the best model
        if not results_df.empty:
            best_model_row = results_df.iloc[0]
            best_model = best_model_row["GARCH_Object"]
            independence = best_model_row["Independence_pvalue"]
            homoscedasticity = best_model_row["Homoscedasticity"]
            normality = best_model_row["Normality"]
            best_r = best_model_row["r"]
            best_s = best_model_row["s"]
            best_model_name = best_model_row["Model"]
            print(f"Selected Model: {best_model_name} with Verified Hypotheses: {best_model_row['Verified_Hypotheses']} and AIC: {best_model_row['AIC']:.2f}")
    else:
        results_df = pd.DataFrame(columns=["Model", "Proportion_Significant", "AIC", "Normality_pvalue", "Independence_pvalue", "Homoscedasticity", "Verified_Hypotheses", "r", "s"])
        print("No suitable GARCH model found.")
    
    return best_model, results_df, best_r, best_s, independence, homoscedasticity, normality


# In[174]:


def choose_best_garch(ts, best_p, max_r, max_s, threshold):
    """
    Choose the best GARCH(p, q) model based on the number of verified hypotheses and AIC.
    
    Returns:
        best_model: The best GARCH model object.
        results_df: A DataFrame with all models and their diagnostics.
        best_r: The ARCH order of the best model.
        best_s: The GARCH order of the best model.
    """
    results = []  # Initialize results list
    best_model = None  # Initialize the best model
    best_r, best_s = None, None  # To store the best (r, s)

    for r in range(max_r + 1):
        for s in range(max_s + 1):
            try:
                # Fit the GARCH model
                garch_model = arch_model(ts, vol="Garch", p=r, q=s, mean="ARX", lags=best_p, dist="normal").fit(disp="off")
                
                # Calculate diagnostics
                p_values = garch_model.pvalues
                prop = np.mean(p_values <= 0.05)  # Proportion of significant coefficients
                aic_value = garch_model.aic
                
                
                # Prepare residuals and conditional volatility
                residuals = garch_model.std_resid
                fitted_values = garch_model.conditional_volatility
                
                # Normality
                lilliefors_stat, lilliefors_pvalue = lilliefors(residuals) 
                
                # Independence
                ljungbox_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
                ljungbox_pvalue = ljungbox_result['lb_pvalue'].iloc[0]
                
                # Homoskedasticity
                if fitted_values is not None:
                    # Add constant to fitted values for regression model
                    aux_data = pd.DataFrame({
                        "residuals_squared": residuals**2,
                        "fitted_values": fitted_values})

                    aux_data = aux_data.replace([np.inf, -np.inf], np.nan).dropna()
                    fitted_values_with_const = sm.add_constant(aux_data["fitted_values"])
                    bp_test_stat, bp_pvalue, _, _ = het_breuschpagan(aux_data["residuals_squared"], fitted_values_with_const)
                else:
                    print("Heteroskedasticity test skipped (fitted values not provided).")
                
                # Count the number of verified hypotheses
                verified_hypotheses = sum([
                    lilliefors_pvalue > 0.05,  # Normality
                    ljungbox_pvalue > 0.05,  # Independence
                    bp_pvalue > 0.05  # Homoscedasticity
                ])
                
                # Append results
                results.append({
                    "Model": f"GARCH({r},{s})",
                    "Proportion_Significant": prop,
                    "AIC": aic_value,
                    "Normality": lilliefors_pvalue,
                    "Independence_pvalue": ljungbox_pvalue,
                    "Homoscedasticity": bp_pvalue,
                    "Verified_Hypotheses": verified_hypotheses,
                    "GARCH_Object": garch_model,  # Store the model object
                    "r": r,  # Store r
                    "s": s   # Store s
                })
                
                print(f"GARCH({r}, {s}) - Proportion p-values: {prop:.2f} - AIC: {aic_value:.2f}")
                print(f"Independent: {ljungbox_pvalue > 0.05} - Homoscedastic: {bp_pvalue > 0.05} - Normal: {lilliefors_pvalue > 0.05}")
            
            except Exception as e:
                print(f"Error fitting GARCH({r}, {s}): {e}")
    
    # Convert results to a DataFrame
    if results:
        results_df = pd.DataFrame(results)
        # Sort by the number of hypotheses verified, then by AIC
        results_df = results_df.sort_values(
            by=["Verified_Hypotheses", "AIC"], ascending=[False, True]
        )
        # Select the best model
        if not results_df.empty:
            best_model_row = results_df.iloc[0]
            best_model = best_model_row["GARCH_Object"]
            independence = best_model_row["Independence_pvalue"]
            homoscedasticity = best_model_row["Homoscedasticity"]
            normality = best_model_row["Normality"]
            best_r = best_model_row["r"]
            best_s = best_model_row["s"]
            best_model_name = best_model_row["Model"]
            print(f"Selected Model: {best_model_name} with Verified Hypotheses: {best_model_row['Verified_Hypotheses']} and AIC: {best_model_row['AIC']:.2f}")
    else:
        results_df = pd.DataFrame(columns=["Model", "Proportion_Significant", "AIC", "Normality_pvalue", "Independence_pvalue", "Homoscedasticity", "Verified_Hypotheses", "r", "s"])
        print("No suitable GARCH model found.")
    
    return best_model, results_df, best_r, best_s, independence, homoscedasticity, normality


# In[5]:


def analyze_time_series(d, ts, max_p, max_q, max_r, max_s, threshold):
    """Perform ARIMA and GARCH analysis on a time series."""
    
    # Step 2: Select best ARIMA model
    all_criteria_models, results_df = select_best_arima_model(ts, max_p, d, max_q, threshold)
    best_arima = all_criteria_models.iloc[0, 0]
    if best_arima is None:
        return None
    
    print(f"Best ARIMA model: {best_arima}")
    
    best_model_params = [int(i) for i in best_arima.replace("ARIMA(", "").replace(")", "").split(",")]
    best_p, best_q = best_model_params
    arima_model = ARIMA(ts, order=(best_p, d, best_q)).fit()
    
    # Step 3: Test for ARCH effects
    residuals = arima_model.resid
    arch_test = het_arch(residuals)
    print(f"ARCH Test p-value: {arch_test[1]:.4f}")
    
    if arch_test[1] < 0.05:
        print("ARCH effects detected.")
    else:
        print("No ARCH effects detected.")
    
    # Step 4: Select best GARCH model
    best_garch_model, results_df, best_r, best_s, independence, homoscedasticity, normality = choose_best_garch(ts, best_p, max_r, max_s, threshold)
    if best_garch_model is None:
        return None
    
    print(f"Best GARCH order: GARCH({best_r}, {best_s})")
    print(f"Independence: {independence}")
    print(f"Homoskedasticity: {homoscedasticity}")
    print(f"Normality: {normality}")
    
    return {
        "arima_model": arima_model,
        "best_garch_model": best_garch_model,
        "arima_order": (best_p, d, best_q),
        "garch_order": (best_r, best_s),
        "independent_residuals": independence,
        "normal_residuals": normality,
        "homoscedastic_residuals": homoscedasticity
    }


# In[ ]:




