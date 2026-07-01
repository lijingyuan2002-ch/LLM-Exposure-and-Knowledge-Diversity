import pandas as pd
import numpy as np
import glob
import os
import statsmodels.api as sm
# ----------------------------- Parameter Configuration -----------------------------
openalex_folder = "works_2023-2025" # Put data from 2023-2025 into this folder
topic_mapping_file = "OpenAlex_topic_mapping_table.xlsx" # OpenAlex discipline classification table
threshold = 0.1  
outcome_vars = ['team_knowledge_variety', 'team_knowledge_distance', 'team_knowledge_balance']
# Raw continuous / categorical covariates
raw_covariates = [
    'authors_count', 'institutions_distinct_count', 'countries_distinct_count','RS'
]
# Stratification variables for coarsening in CEM matching
coarsen_raw = [
    'authors_count', 'institutions_distinct_count', 'countries_distinct_count',
    'RS', 'topic_id', 'first_author_country'
]
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
# ----------------------------- 3. Covariate Coarsening -----------------------------
print("\nCoarsening covariates...")
bins_auth = [0, 1, 5, 10, np.inf]
labels_auth = ['1', '2-5', '5-10', '>10']
df_all['auth_group'] = pd.cut(df_all['authors_count'], bins=bins_auth, labels=labels_auth, right=False)
df_all['single_inst'] = (df_all['institutions_distinct_count'] == 1).astype(int)
df_all['single_country'] = (df_all['countries_distinct_count'] == 1).astype(int)
bins_rs = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, np.inf]
labels_rs = ['0.36-0.38', '0.38-0.40', '0.40-0.42', '0.42-0.44', '0.44-0.46', '0.46-0.48', '>0.48']
df_all['rs_group'] = pd.cut(df_all['RS'], bins=bins_rs, labels=labels_rs, right=False)
coarse_vars = ['auth_group', 'single_inst', 'single_country', 'rs_group', 'field_id', 'first_author_country', 'year']
keep_cols = coarse_vars + raw_covariates + outcome_vars + [f'alpha_gt_{threshold}'] + ['id']
df_clean = df_all[keep_cols].copy().dropna(subset=coarse_vars + raw_covariates + outcome_vars)
df_clean = df_clean[df_clean[f'alpha_gt_{threshold}'].isin([0,1])]
print(f"Valid sample size after cleaning & filtering binary treatment variable: {len(df_clean)}")
# ----------------------------- 4. Core Estimation Function -----------------------------
def cem_full_est(df, treat_col, out_col, strata_cols, ctrl_vars):
    sub = df.copy()
    # ========== 1. CEM Matching + Weighted ATT ==========
    group_stats = sub.groupby(strata_cols)[treat_col].agg(['sum','count'])
    group_stats.columns = ['n_t','n_all']
    group_stats['n_c'] = group_stats['n_all'] - group_stats['n_t']
    valid_idx = group_stats[(group_stats.n_t>0)&(group_stats.n_c>0)].index
    if len(valid_idx)==0:
        return None
    sub_valid = sub.set_index(strata_cols).loc[valid_idx].reset_index()
    sub_valid = sub_valid.merge(group_stats,on=strata_cols)
    sub_valid['weight'] = np.where(sub_valid[treat_col]==1,1,sub_valid.n_t/sub_valid.n_c)
    # CEM Weighted WLS
    X_cem = sm.add_constant(sub_valid[[treat_col]])
    fit_cem = sm.WLS(sub_valid[out_col],X_cem,weights=sub_valid['weight']).fit(cov_type="HC1")
    # ========== 2. Baseline OLS Regression (full controls, robust SE) ==========
    X_ols_data = sub[[treat_col] + ctrl_vars]
    X_ols = sm.add_constant(X_ols_data)
    fit_ols = sm.OLS(sub[out_col],X_ols).fit(cov_type="HC1")
    # ========== 3. Balance SMD Calculation ==========
    res_balance = {}
    t_pre = sub[sub[treat_col]==1]
    c_pre = sub[sub[treat_col]==0]
    t_post = sub_valid[sub_valid[treat_col]==1]
    c_post = sub_valid[sub_valid[treat_col]==0]
    for cv in ctrl_vars:
        std_pool_pre = np.sqrt((np.var(t_pre[cv],ddof=1)+np.var(c_pre[cv],ddof=1))/2)
        smd_pre = (t_pre[cv].mean()-c_pre[cv].mean())/std_pool_pre if std_pool_pre>1e-8 else np.nan
        std_pool_post = np.sqrt((np.var(t_post[cv],ddof=1)+np.var(c_post[cv],ddof=1))/2)
        smd_post = (t_post[cv].mean()-c_post[cv].mean())/std_pool_post if std_pool_post>1e-8 else np.nan
        res_balance[cv] = {"SMD_pre":smd_pre,"SMD_post":smd_post}
    # Sample statistics
    nt_total = len(t_pre)
    nc_total = len(c_pre)
    nt_match = len(t_post)
    nc_match = len(c_post)
    return {
        "outcome":out_col,
        # CEM-ATT
        "ATT":fit_cem.params[treat_col],
        "ATT_p":fit_cem.pvalues[treat_col],
        "CI95_L":fit_cem.conf_int().loc[treat_col,0],
        "CI95_U":fit_cem.conf_int().loc[treat_col,1],
        # OLS full controls robustness
        "OLS_full_coef":fit_ols.params[treat_col],
        "OLS_full_p":fit_ols.pvalues[treat_col],
        # Sample size
        "N_treat_all":nt_total,"N_ctrl_all":nc_total,
        "N_treat_match":nt_match,"N_ctrl_match":nc_match,
        "balance":res_balance
    }
# ----------------------------- 5. Batch Execution & Export -----------------------------
treat_col = f"alpha_gt_{threshold}"
res_list = []
for out in outcome_vars:
    est = cem_full_est(df_clean,treat_col,out,coarse_vars,raw_covariates)
    if est:
        res_list.append(est)
# Compile main result table
main_tab = pd.DataFrame([
    {
        "Outcome_Variable":r["outcome"],
        "CEM_ATT":round(r["ATT"],4),
        "ATT_pvalue":round(r["ATT_p"],4),
        "CI95_Lower":round(r["CI95_L"],4),
        "CI95_Upper":round(r["CI95_U"],4),
        "OLS_Full_Coeff":round(r["OLS_full_coef"],4),
        "OLS_Full_pvalue":round(r["OLS_full_p"],4),
        "Total_Treated":r["N_treat_all"],
        "Total_Control":r["N_ctrl_all"],
        "Matched_Treated":r["N_treat_match"],
        "Matched_Control":r["N_ctrl_match"]
    }
    for r in res_list
])
# Compile balance table
bal_rows = []
for r in res_list:
    for cv,b in r["balance"].items():
        bal_rows.append({
            "Outcome_Variable":r["outcome"],
            "Covariate":cv,
            "Pre_Match_SMD":round(b["SMD_pre"],4),
            "Post_Match_SMD":round(b["SMD_post"],4)
        })
bal_tab = pd.DataFrame(bal_rows)
# Save Excel
save_path = os.path.join(openalex_folder,f"CEM_alpha0.1_OLS_fullcontrols_summary_results.xlsx")
with pd.ExcelWriter(save_path) as w:
    main_tab.to_excel(w,sheet_name="Main_Effect_Full_Control_OLS_Robustness",index=False)
    bal_tab.to_excel(w,sheet_name="Covariate_Balance_SMD",index=False)
print("\n✅ Execution Completed!")
print(f"📁 Output File: {save_path}")