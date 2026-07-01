import pandas as pd
import numpy as np
import glob
import os
import statsmodels.api as sm
import patsy
from sklearn.preprocessing import StandardScaler  
# ----------------------------- Parameter Configuration -----------------------------
openalex_folder = "works_2023-2025" # Put data of 2023-2025 into this folder
topic_mapping_file = "OpenAlex_topic_mapping_table.xlsx" # OpenAlex discipline classification table
threshold = 0.1
outcome_vars = ['team_knowledge_variety', 'team_knowledge_distance', 'team_knowledge_balance']
# Original covariate names
raw_covariates_original = [
    'authors_count', 'institutions_distinct_count', 'countries_distinct_count', 'RS'
]
# Log-transformed covariate names
log_covariates = ['log_authors', 'log_institutions', 'log_countries']
# Final control variables used in regression
regression_controls = ['log_authors', 'log_institutions', 'log_countries', 'RS']
years = [2023, 2024, 2025]
# ----------------------------- 1. Load Data -----------------------------
print("Reading OpenAlex CSV files...")
all_dfs = []
for year in years:
    pattern = os.path.join(openalex_folder, f"merged_works_{year}*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: No files found for year {year}, skip")
        continue
    for f in files:
        df = pd.read_csv(f)
        df['year'] = year
        all_dfs.append(df)
        print(f"  Loaded: {f}, Rows: {len(df)}")
if not all_dfs:
    raise ValueError("No OpenAlex CSV files found, please check file path.")
df_all = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows after concatenation: {len(df_all)}")
# ----------------------------- 2. Topic Mapping -----------------------------
print("\nLoading topic mapping table...")
topic_map = pd.read_excel(topic_mapping_file)
topic_map['topic_id_clean'] = topic_map['topic_id'].astype(str).str.replace('T', '')
topic_to_field = dict(zip(topic_map['topic_id_clean'], topic_map['field_id']))
df_all['topic_id_str'] = df_all['topic_id'].astype(str).str.replace('T', '')
df_all['field_id'] = df_all['topic_id_str'].map(topic_to_field)
print(f"Successfully mapped topic_id count: {df_all['field_id'].notna().sum()} / {len(df_all)}")
# ----------------------------- Preliminary Data Filtering -----------------------------
print("\n【Preliminary Filtering】Drop rows where count variables ≤0...")
initial_count = len(df_all)
# Filter condition: all three count variables > 0
mask = (df_all['authors_count'] > 0) & \
       (df_all['institutions_distinct_count'] > 0) & \
       (df_all['countries_distinct_count'] > 0)
df_all = df_all[mask]
removed_count = initial_count - len(df_all)
print(f"  Removed {removed_count} rows ({removed_count/initial_count*100:.2f}%)")
print(f"  Remaining samples: {len(df_all)}")
# ----------------------------- Data Cleaning -----------------------------
keep_cols_initial = raw_covariates_original + outcome_vars + [f'alpha_gt_{threshold}'] + ['id', 'year', 'field_id', 'first_author_country']
df_clean = df_all[keep_cols_initial].copy().dropna(subset=raw_covariates_original + outcome_vars + [f'alpha_gt_{threshold}', 'field_id', 'first_author_country'])
df_clean = df_clean[df_clean[f'alpha_gt_{threshold}'].isin([0,1])]
# ----------------------------- Log Transformation for Covariates (RS unchanged) -----------------------------
print("\nApplying log transformation to control variables (RS remains original)...")
df_clean['log_authors'] = np.log(df_clean['authors_count']) 
df_clean['log_institutions'] = np.log(df_clean['institutions_distinct_count'])
df_clean['log_countries'] = np.log(df_clean['countries_distinct_count'])
df_clean['RS'] = df_clean['RS'] # RS remains raw value without log
# ----------------------------- Standardization of Continuous Variables -----------------------------
print("\n" + "="*60)
print("Step3: Standardize all continuous variables (mean=0, sd=1)")
print("="*60)
# Define variables to be standardized
continuous_vars = [
    'team_knowledge_variety', 'team_knowledge_distance', 'team_knowledge_balance',
    'log_authors', 'log_institutions', 'log_countries', 'RS'
]
df_clean_scaled = df_clean.copy()
scaler = StandardScaler()
# Standardize one by one
for var in continuous_vars:
    scaled_vals = scaler.fit_transform(df_clean[[var]]).flatten()
    df_clean_scaled[f"{var}_scaled"] = scaled_vals
# Define standardized new variable names
outcome_scaled = [f"{v}_scaled" for v in outcome_vars]
control_scaled = [f"{c}_scaled" for c in regression_controls]
print(" Standardization completed, variables used for regression:")
print(f"  Dependent variables: {outcome_scaled}")
print(f"  Control variables: {control_scaled}")
# Filter fields with small sample size
field_counts = df_clean_scaled['field_id'].value_counts()
min_field_size = 30
fields_to_keep = field_counts[field_counts >= min_field_size].index
df_clean_scaled = df_clean_scaled[df_clean_scaled['field_id'].isin(fields_to_keep)]
# Filter countries with small sample size
country_counts = df_clean_scaled['first_author_country'].value_counts()
min_country_size = 30  # Countries with at least 30 papers
countries_to_keep = country_counts[country_counts >= min_country_size].index
df_clean_scaled = df_clean_scaled[df_clean_scaled['first_author_country'].isin(countries_to_keep)]
print(f"Fields retained after filtering: {len(fields_to_keep)}, Countries retained: {len(countries_to_keep)}, Total samples: {len(df_clean_scaled)}")
# Convert to categorical variables
df_clean_scaled['year'] = df_clean_scaled['year'].astype('category')
df_clean_scaled['field_id'] = df_clean_scaled['field_id'].astype('category')
df_clean_scaled['first_author_country'] = df_clean_scaled['first_author_country'].astype('category')
# Rename treatment variable
treat_col_original = f"alpha_gt_{threshold}"
treat_col_clean = "alpha_gt_0_1"
df_clean_scaled[treat_col_clean] = df_clean_scaled[treat_col_original]
print(f"Final valid sample size: {len(df_clean_scaled)}")
print(f"Year distribution: {df_clean_scaled['year'].value_counts().sort_index().to_dict()}")
print(f"Number of fields: {df_clean_scaled['field_id'].nunique()}")
print(f"Number of countries: {df_clean_scaled['first_author_country'].nunique()}")
# ----------------------------- Significance Star Function -----------------------------
def sig_stars(pval):
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return ""
def format_coef_with_se(coef, se, pval, decimals=5):
    coef_str = f"{coef:.{decimals}f}"
    se_str = f"{se:.{decimals}f}"
    stars = sig_stars(pval)
    return f"{coef_str}{stars}\n({se_str})"
# ----------------------------- Regression Function (Preserve Intercept) -----------------------------
def run_all_models_with_fe(df, treatment, outcomes, controls):
    """Run regression models with intercept (include country fixed effects)"""
    models_results = {}
    
    for y in outcomes:
        print(f"\n>>> Processing dependent variable: {y}")
        
        # Model 1/3/5: No controls, no fixed effects
        X1 = sm.add_constant(df[[treatment]])
        model1 = sm.OLS(df[y], X1).fit(cov_type='HC1')
        models_results[f"{y}_NoControls_NoFE"] = model1
        print(f"  Model 1 finished (no controls), const coefficient={model1.params['const']:.4f}, R²={model1.rsquared:.4f}")
        
        # Model 2/4/6: With controls + fixed effects (year, field, country)
        formula = f"{y} ~ {treatment} + " + " + ".join(controls) + " + C(year) + C(field_id) + C(first_author_country)"
        
        try:
            # Build design matrix (patsy retains intercept by default)
            y_data, X_data = patsy.dmatrices(formula, df, return_type='dataframe')
            
            # Check if intercept exists
            if 'Intercept' not in X_data.columns:
                print("  Warning: No intercept found in design matrix, add manually")
                X_data = sm.add_constant(X_data)
            
            print(f"  Design matrix dimension: {X_data.shape}")
            print(f"  Intercept included: {'Intercept' in X_data.columns}")
            print(f"  Control variables: {controls}")
            
            # Clustered robust standard errors (cluster at field level)
            n_fields = df['field_id'].nunique()
            if n_fields >= 30:
                model2 = sm.OLS(y_data, X_data).fit(
                    cov_type='cluster', 
                    cov_kwds={'groups': df['field_id'].values}
                )
                print(f"  Model 2 finished (cluster SE), const coefficient={model2.params['Intercept']:.4f}, R²={model2.rsquared:.4f}")
            else:
                model2 = sm.OLS(y_data, X_data).fit(cov_type='HC1')
                print(f"  Model 2 finished (HC1 SE), const coefficient={model2.params['Intercept']:.4f}, R²={model2.rsquared:.4f}")
            
            # Rename intercept to 'const' for unified processing
            if 'Intercept' in model2.params.index:
                new_params = model2.params.copy()
                new_bse = model2.bse.copy()
                new_pvalues = model2.pvalues.copy()
                new_conf_int = model2.conf_int().copy()
                
                new_params.index = ['const' if idx == 'Intercept' else idx for idx in new_params.index]
                new_bse.index = ['const' if idx == 'Intercept' else idx for idx in new_bse.index]
                new_pvalues.index = ['const' if idx == 'Intercept' else idx for idx in new_pvalues.index]
                new_conf_int.index = ['const' if idx == 'Intercept' else idx for idx in new_conf_int.index]
                
                model2.params = new_params
                model2.bse = new_bse
                model2.pvalues = new_pvalues
                model2._results.conf_int = lambda alpha=0.05: new_conf_int
            
            models_results[f"{y}_WithControls_WithFE"] = model2
                
        except Exception as e:
            print(f"  Error: {e}")
            raise
    
    return models_results
# ----------------------------- Output Regression Table -----------------------------
def create_academic_table_with_fe(models_results, treatment, controls):
    """Generate academic-style table with intercept"""
    model_order = [
        "team_knowledge_variety_scaled_NoControls_NoFE",
        "team_knowledge_variety_scaled_WithControls_WithFE",
        "team_knowledge_distance_scaled_NoControls_NoFE",
        "team_knowledge_distance_scaled_WithControls_WithFE",
        "team_knowledge_balance_scaled_NoControls_NoFE",
        "team_knowledge_balance_scaled_WithControls_WithFE"
    ]
    
    display_names = [
        "(1)\nKnowledge Variety\nNo Controls (Scaled)",
        "(2)\nKnowledge Variety\nControls+FE (Scaled)",
        "(3)\nKnowledge Distance\nNo Controls (Scaled)",
        "(4)\nKnowledge Distance\nControls+FE (Scaled)",
        "(5)\nKnowledge Balance\nNo Controls (Scaled)",
        "(6)\nKnowledge Balance\nControls+FE (Scaled)"
    ]
    
    # Variable list
    all_vars = [treatment] + controls + ['const']
    
    table_data = {display_names[i]: [] for i in range(len(model_order))}
    
    for idx, model_name in enumerate(model_order):
        model = models_results[model_name]
        col_name = display_names[idx]
        
        # Core variables
        for var in all_vars:
            if var in model.params.index:
                coef = model.params[var]
                se = model.bse[var]
                pval = model.pvalues[var]
                formatted = format_coef_with_se(coef, se, pval, decimals=5)
                table_data[col_name].append(formatted)
            else:
                table_data[col_name].append("-\n(-)")
        
        # Fixed effect label
        if "WithFE" in model_name:
            fe_text = "Year + Field + Country"
            table_data[col_name].append(fe_text)
        else:
            table_data[col_name].append("Uncontrolled")
    
    # Create DataFrame
    df_table = pd.DataFrame(table_data, index=all_vars + ["Fixed_Effects"])
    
    # Append model statistics
    n_row = [f"{int(models_results[m].nobs):,}" for m in model_order]
    r2_row = [f"{models_results[m].rsquared:.5f}" for m in model_order]
    
    adj_r2_row = []
    for model_name in model_order:
        if hasattr(models_results[model_name], 'rsquared_adj'):
            adj_r2_row.append(f"{models_results[model_name].rsquared_adj:.5f}")
        else:
            adj_r2_row.append("-")
    
    df_table.loc["N_Observations"] = n_row
    df_table.loc["R_Squared"] = r2_row
    df_table.loc["Adj_R_Squared"] = adj_r2_row
    
    return df_table
# ----------------------------- Run Regressions -----------------------------
print("\n" + "="*80)
print("Start running 6 OLS regression models...")
print("="*80)
# Core: pass standardized variables
models_results = run_all_models_with_fe(
    df_clean_scaled, 
    treat_col_clean, 
    outcome_scaled,  # Standardized dependent variables
    control_scaled   # Standardized control variables
)
# Generate academic table
academic_table = create_academic_table_with_fe(models_results, treat_col_clean, control_scaled)
# ----------------------------- Export to Excel -----------------------------
save_path = os.path.join(openalex_folder, f"OLS_6models_Standardized_WithIntercept_alpha{threshold}_withCountryFE.xlsx")
with pd.ExcelWriter(save_path) as writer:
    academic_table.to_excel(writer, sheet_name="Academic_Regression_Table")
    
    # Full detailed results
    detailed_results = []
    for model_name, model in models_results.items():
        for var_name in model.params.index:
            var_display = var_name
            if 'C(year)' in var_name:
                var_display = f"Year_{var_name.split('[')[1].split(']')[0]}" if '[' in var_name else var_name
            elif 'C(field_id)' in var_name:
                var_display = f"Field_{var_name.split('[')[1].split(']')[0]}" if '[' in var_name else var_name
            elif 'C(first_author_country)' in var_name:
                var_display = f"Country_{var_name.split('[')[1].split(']')[0]}" if '[' in var_name else var_name
            
            detailed_results.append({
                "Model": model_name,
                "Variable": var_display,
                "Coefficient": model.params[var_name],
                "Std_Error": model.bse[var_name],
                "t_value": model.params[var_name] / model.bse[var_name] if model.bse[var_name] != 0 else np.nan,
                "p_value": model.pvalues[var_name],
                "Significance": sig_stars(model.pvalues[var_name])
            })
    
    pd.DataFrame(detailed_results).to_excel(writer, sheet_name="Full_Detailed_Results", index=False)
    
    # Model fit statistics
    fit_stats = []
    for model_name, model in models_results.items():
        fit_stats.append({
            "Model": model_name,
            "R-squared": model.rsquared,
            "Adj. R-squared": model.rsquared_adj if hasattr(model, 'rsquared_adj') else None,
            "N_Observations": int(model.nobs)
        })
    pd.DataFrame(fit_stats).to_excel(writer, sheet_name="Model_Fit_Statistics", index=False)
print(f"\nOLS regression completed!")
print(f"Results saved to: {save_path}")