#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
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


# In[4]:


def calculate_p_values(arima_model):
    """Calculate p-values for ARIMA model coefficients."""
    coefs = arima_model.params
    std_errors = np.sqrt(np.diag(arima_model.cov_params()))
    t_values = coefs / std_errors
    p_values = 2 * (1 - norm.cdf(np.abs(t_values)))
    return p_values


# In[5]:


def select_best_arima_model(ts, max_p, max_d, max_q, threshold=0.7):
    """Select the best ARIMA model based on AIC and proportion of significant p-values."""
    results = []
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q)).fit()
                    p_values = calculate_p_values(model)
                    prop = np.mean(p_values <= 0.05)
                    aic_value = model.aic
                    print(f"ARIMA({p}, {d}, {q}) - Proportion p-values: {prop:.2f} - AIC: {aic_value:.2f}")
                    results.append((f"ARIMA({p},{d},{q})", prop, aic_value))
                except Exception as e:
                    print(f"Error fitting ARIMA({p}, {d}, {q}): {e}")
    
    # Convert to DataFrame for filtering and sorting
    results_df = pd.DataFrame(results, columns=["Model", "Proportion", "AIC"])
    filtered_results = results_df[results_df["Proportion"] > threshold]
    if not filtered_results.empty:
        best_model = filtered_results.loc[filtered_results["AIC"].idxmin()]
        return best_model
    else:
        print("No suitable ARIMA model found.")
        return None


# In[6]:


def choose_best_garch(ts, p, q, max_r, max_s, garch_threshold=0.6):
    """Choose the best GARCH(p, q) model based on AIC and proportion of significant coefficients."""
    best_aic = np.inf
    best_model = None
    best_r, best_s = 0, 0
    
    for r in range(max_r + 1):
        for s in range(max_s + 1):
            try:
                model = arch_model(ts, vol="Garch", p=r, q=s, mean="ARX", lags=p, dist="normal").fit(disp="off")
                p_values = model.pvalues
                prop = np.mean(p_values <= 0.05)
                aic_value = model.aic
                print(f"GARCH({r}, {s}) - Proportion p-values: {prop:.2f} - AIC: {aic_value:.2f}")
                
                if prop > garch_threshold and aic_value < best_aic:
                    best_aic = aic_value
                    best_model = model
                    best_r, best_s = r, s
            except Exception as e:
                print(f"Error fitting GARCH({r}, {s}): {e}")
    
    if best_model:
        print(f"Best GARCH({best_r}, {best_s}) with AIC: {best_aic:.2f}")
    else:
        print("No suitable GARCH model found.")
    return best_model, best_aic, best_r, best_s

def analyze_time_series(ts, max_p, max_d, max_q, max_r, max_s, garch_threshold=0.6):
    """Perform ARIMA and GARCH analysis on a time series."""
    
    # Step 1: Plot ACF and PACF
    plt.figure(figsize=(12, 6))
    plot_acf(ts, lags=20, title="ACF of Returns")
    plot_pacf(ts, lags=20, title="PACF of Returns")
    plt.tight_layout()
    plt.show()
    
    # Step 2: Select best ARIMA model
    best_arima = select_best_arima_model(ts, max_p, max_d, max_q, threshold=0.7)
    if best_arima is None:
        return None
    
    print(f"Best ARIMA model: {best_arima['Model']}")
    best_model_params = [int(i) for i in best_arima["Model"].replace("ARIMA(", "").replace(")", "").split(",")]
    best_p, best_d, best_q = best_model_params
    arima_model = ARIMA(ts, order=(best_p, best_d, best_q)).fit()
    
    # Step 3: Test for ARCH effects
    residuals = arima_model.resid
    arch_test = het_arch(residuals)
    print(f"ARCH Test p-value: {arch_test[1]:.4f}")
    
    if arch_test[1] < 0.05:
        print("ARCH effects detected.")
        plt.figure(figsize=(12, 6))
        plot_acf(residuals**2, lags=20, title="ACF of Squared Residuals")
        plot_pacf(residuals**2, lags=20, title="PACF of Squared Residuals")
        plt.tight_layout()
        plt.show()
    else:
        print("No ARCH effects detected.")
    
    # Step 4: Select best GARCH model
    best_garch_model, best_aic, best_r, best_s = choose_best_garch(ts, best_p, best_q, max_r, max_s, garch_threshold=garch_threshold)
    if best_garch_model is None:
        return None
    
    print(f"Best GARCH order: GARCH({best_r}, {best_s})")
    
    # Step 5: Analyze residuals
    residuals = best_garch_model.std_resid
    fitted_values = best_garch_model.conditional_volatility
    
    # 1. Independence Test (Ljung-Box Test)
    ljungbox_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    ljungbox_pvalue = ljungbox_result['lb_pvalue'].iloc[0]
    if ljungbox_pvalue < 0.05:
        print("Residuals are not independent (Ljung-Box p-value < 0.05).")
        independent_residuals = False
    else:
        print("Residuals are independent (Ljung-Box p-value >= 0.05).")
        independent_residuals = True

    # 2. Normality Test (Lilliefors Test)
    lilliefors_stat, lilliefors_pvalue = lilliefors(residuals)
    if lilliefors_pvalue < 0.05:
        print("Residuals are not normal (Lilliefors p-value < 0.05).")
        normal_residuals = False
    else:
        print("Residuals are normal (Lilliefors p-value >= 0.05).")
        normal_residuals = True

    # 3. Heteroskedasticity Test (Breusch-Pagan Test)
    if fitted_values is not None:
        # Add constant to fitted values for regression model
        aux_data = pd.DataFrame({
            "residuals_squared": residuals**2,
            "fitted_values": fitted_values})

        aux_data = aux_data.replace([np.inf, -np.inf], np.nan).dropna()
        fitted_values_with_const = sm.add_constant(aux_data["fitted_values"])
        bp_test_stat, bp_pvalue, _, _ = het_breuschpagan(aux_data["residuals_squared"], fitted_values_with_const)
        if bp_pvalue < 0.05:
            print("Residuals are heteroscedastic (Breusch-Pagan p-value < 0.05).")
            homoscedastic_residuals = False
        else:
            print("Residuals are homoscedastic (Breusch-Pagan p-value >= 0.05).")
            homoscedastic_residuals = True
    else:
        print("Heteroskedasticity test skipped (fitted values not provided).")

    
    return {
        "arima_model": arima_model,
        "best_garch_model": best_garch_model,
        "arima_order": (best_p, best_d, best_q),
        "garch_order": (best_r, best_s),
        "independent_residuals": independent_residuals,
        "normal_residuals": normal_residuals,
        "homoscedastic_residuals": homoscedastic_residuals
    }

# Example usage:
# ts = your_time_series_data
# results = analyze_time_series(ts, max_p=5, max_d=2, max_q=5, max_r=2, max_s=2, garch_threshold=0.6)


# In[ ]:




