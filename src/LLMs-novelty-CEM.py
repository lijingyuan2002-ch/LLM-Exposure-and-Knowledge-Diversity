import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
import glob
from pathlib import Path
warnings.filterwarnings('ignore')
print("="*60)
print("Step 1: Load and Merge Raw Datasets")
print("="*60)
# 1. Load parquet file
df_parquet = pd.read_parquet("sciscinet_papers_filtered.parquet",
                              columns=['paperid', 'Atyp_10pct_Z', 'year'])
df_parquet = df_parquet[df_parquet['year'] >= 2023]
df_parquet['paperid_clean'] = df_parquet['paperid'].str.replace('https://openalex.org/', '')
# 2. Load topic mapping table
topic_mapping = pd.read_excel("OpenAlex_topic_mapping_table.xlsx")
topic_mapping['topic_id_with_prefix'] = 'T' + topic_mapping['topic_id'].astype(str)
# 3. Load all CSV files and perform matching
csv_folder = "works_2023-2025"
csv_files = sorted(glob.glob(f"{csv_folder}/*.csv"))
required_columns = ['id', 'team_knowledge_variety', 'team_knowledge_distance',
                    'team_knowledge_balance', 'alpha_gt_0.1', 'authors_count',
                    'institutions_distinct_count', 'countries_distinct_count',
                    'RS', 'topic_id', 'first_author_country']
all_matches = []
for csv_file in csv_files:
    print(f"Processing: {Path(csv_file).name}")
    try:
        df_csv = pd.read_csv(csv_file, usecols=required_columns)
        df_csv['id_clean'] = df_csv['id'].str.replace('https://openalex.org/', '')
        merged = df_parquet.merge(df_csv, left_on='paperid_clean', right_on='id_clean', how='inner')
        if not merged.empty:
            all_matches.append(merged)
    except Exception as e:
        print(f"Error: {e}")
df_final = pd.concat(all_matches, ignore_index=True)
df_final = df_final.drop_duplicates(subset=['paperid_clean'])
print(f"Sample size after matching: {len(df_final)}")
# 4. Map topic_id -> field_id
df_final['topic_id'] = df_final['topic_id'].astype(str)
df_final = df_final.merge(topic_mapping[['topic_id_with_prefix', 'field_id']],
                          left_on='topic_id', right_on='topic_id_with_prefix', how='left')
df_final['field_id'] = df_final['field_id'].fillna('Other').astype(str)
print("="*60)
print("Step 2: Data Cleaning & Variable Construction")
print("="*60)
# Dependent variable
df_final['novelty'] = (df_final['Atyp_10pct_Z'] < 0).astype(int)
# Ensure numeric columns are correctly typed
numeric_cols = ['alpha_gt_0.1', 'team_knowledge_variety', 'team_knowledge_distance',
                'team_knowledge_balance', 'authors_count', 'institutions_distinct_count',
                'countries_distinct_count', 'RS', 'year']
for col in numeric_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
# Drop missing values
df_clean = df_final.dropna(subset=['alpha_gt_0.1', 'novelty'] + numeric_cols)
df_clean = df_clean[df_clean['alpha_gt_0.1'].isin([0,1])]
print(f"Sample size after cleaning: {len(df_clean)}")
# Save raw cleaned dataset
df = df_clean.copy()
print("="*60)
print("Step 3: Construct Coarsened Variables (3 groups by specified bins)")
print("="*60)
# 3.1 Author count grouping
bins_auth = [0, 1, 5, 10, np.inf]
labels_auth = ['1', '2-5', '5-10', '>10']
df['auth_group'] = pd.cut(df['authors_count'], bins=bins_auth, labels=labels_auth, right=False)
# 3.2 Single institution indicator
df['single_inst'] = (df['institutions_distinct_count'] == 1).astype(int)
# 3.3 Single country indicator
df['single_country'] = (df['countries_distinct_count'] == 1).astype(int)
# 3.4 RS grouping
bins_rs = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, np.inf]
labels_rs = ['0.36-0.38', '0.38-0.40', '0.40-0.42', '0.42-0.44', '0.44-0.46', '0.46-0.48', '>0.48']
df['rs_group'] = pd.cut(df['RS'], bins=bins_rs, labels=labels_rs, right=False)
# 3.5 Knowledge indicators grouped into 3 tiers
bins_variety = [0, 0.4, 0.6, 1.0]
labels_variety = ['Low', 'Mid', 'High']
df['knowledge_variety_group'] = pd.cut(df['team_knowledge_variety'], bins=bins_variety, labels=labels_variety, include_lowest=True)
bins_distance = [0, 0.2, 0.4, 1.0]
labels_distance = ['Low', 'Mid', 'High']
df['knowledge_distance_group'] = pd.cut(df['team_knowledge_distance'], bins=bins_distance, labels=labels_distance, include_lowest=True)
bins_balance = [0, 0.3, 0.5, 1.0]
labels_balance = ['Low', 'Mid', 'High']
df['knowledge_balance_group'] = pd.cut(df['team_knowledge_balance'], bins=bins_balance, labels=labels_balance, include_lowest=True)
# 3.6 Categorical variables: use raw values directly
df['field_id'] = df['field_id'].astype(str)
df['country_id'] = df['first_author_country'].fillna('Unknown').astype(str)
df['year_id'] = df['year'].astype(str)
# List of coarsened variables for CEM
coarse_vars = [
    'auth_group',
    'single_inst',
    'single_country',
    'rs_group',
    'knowledge_variety_group',
    'knowledge_distance_group',
    'knowledge_balance_group',
    'field_id',
    'country_id',
    'year_id'
]
print("="*60)
print("Step 4: Perform CEM Stratified Matching")
print("="*60)
# Create stratum identifier
df['stratum'] = df[coarse_vars].astype(str).agg('_'.join, axis=1)
# Calculate treated and control counts per stratum
stratum_stats = df.groupby('stratum')['alpha_gt_0.1'].agg(
    n_treated='sum',
    n_control=lambda x: (x == 0).sum()
).reset_index()
# Keep strata containing both treated and control units
matched_strata = stratum_stats[(stratum_stats['n_treated'] > 0) & (stratum_stats['n_control'] > 0)]
print(f"Total strata count: {stratum_stats.shape[0]:,}")
print(f"Valid strata with both treated & control: {matched_strata.shape[0]:,}")
# Assign CEM weights
df['cem_weight'] = 1.0
for _, row in matched_strata.iterrows():
    stratum = row['stratum']
    n_t = row['n_treated']
    n_c = row['n_control']
    weight_control = n_t / n_c
    mask = df['stratum'] == stratum
    df.loc[mask & (df['alpha_gt_0.1'] == 1), 'cem_weight'] = 1.0
    df.loc[mask & (df['alpha_gt_0.1'] == 0), 'cem_weight'] = weight_control
# Subset matched sample
matched_df = df[df['stratum'].isin(matched_strata['stratum'])].copy()
print(f"\nTotal sample before matching: {len(df):,}")
print(f"Treated before matching: {(df['alpha_gt_0.1']==1).sum():,}")
print(f"Control before matching: {(df['alpha_gt_0.1']==0).sum():,}")
print(f"\nTotal sample after matching: {len(matched_df):,}")
print(f"Treated after matching: {(matched_df['alpha_gt_0.1']==1).sum():,}")
print(f"Control after matching: {(matched_df['alpha_gt_0.1']==0).sum():,}")
# Sample retention rate
retention_rate = len(matched_df) / len(df) * 100
print(f"\nSample retention rate: {retention_rate:.2f}%")
# ============================================================
# Step 5: Balance Test
# ============================================================
print("="*60)
print("Step 5: Balance Test")
print("="*60)
balance_vars = [
    'authors_count', 'institutions_distinct_count', 'countries_distinct_count',
    'RS', 'team_knowledge_variety', 'team_knowledge_distance', 'team_knowledge_balance'
]
var_names_en = {
    'authors_count': 'Author Count',
    'institutions_distinct_count': 'Institution Count',
    'countries_distinct_count': 'Country Count',
    'RS': 'Reference Diversity RS',
    'team_knowledge_variety': 'Knowledge Variety',
    'team_knowledge_distance': 'Knowledge Distance',
    'team_knowledge_balance': 'Knowledge Balance'
}
def balance_stats_unweighted(df, treatment_col, vars_list):
    results = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    for var in vars_list:
        mean_t = treated[var].mean()
        mean_c = control[var].mean()
        bias_pct = (mean_t - mean_c) / mean_c * 100 if mean_c != 0 else 0
        t_stat, p_val = stats.ttest_ind(treated[var], control[var], equal_var=False)
        var_t = treated[var].var()
        var_c = control[var].var()
        pooled_sd = np.sqrt((var_t + var_c) / 2)
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0
        results.append({'mean_t': mean_t, 'mean_c': mean_c, 'bias_pct': bias_pct,
                        'p_val': p_val, 'smd': smd})
    return pd.DataFrame(results)
def balance_stats_weighted(df, treatment_col, vars_list, weight_col):
    results = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    for var in vars_list:
        mean_t = np.average(treated[var], weights=treated[weight_col])
        mean_c = np.average(control[var], weights=control[weight_col])
        bias_pct = (mean_t - mean_c) / mean_c * 100 if mean_c != 0 else 0
        var_t = np.average((treated[var] - mean_t)**2, weights=treated[weight_col])
        var_c = np.average((control[var] - mean_c)**2, weights=control[weight_col])
        n_t_eff = (treated[weight_col].sum()**2) / (treated[weight_col]**2).sum()
        n_c_eff = (control[weight_col].sum()**2) / (control[weight_col]**2).sum()
        se_t = np.sqrt(var_t / n_t_eff) if n_t_eff > 0 else 0
        se_c = np.sqrt(var_c / n_c_eff) if n_c_eff > 0 else 0
        se_diff = np.sqrt(se_t**2 + se_c**2)
        t_stat = (mean_t - mean_c) / se_diff if se_diff > 0 else 0
        df_num = (se_t**2 + se_c**2)**2
        df_den = (se_t**4/(n_t_eff-1) + se_c**4/(n_c_eff-1)) if (n_t_eff>1 and n_c_eff>1) else 1
        dof = df_num / df_den if df_den > 0 else 1
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
        pooled_sd = np.sqrt((var_t + var_c) / 2)
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0
        results.append({'mean_t': mean_t, 'mean_c': mean_c, 'bias_pct': bias_pct,
                        'p_val': p_val, 'smd': smd})
    return pd.DataFrame(results)
before_df = balance_stats_unweighted(df, 'alpha_gt_0.1', balance_vars)
after_df = balance_stats_weighted(matched_df, 'alpha_gt_0.1', balance_vars, 'cem_weight')
balance_table = pd.DataFrame()
balance_table['Variable'] = [var_names_en[v] for v in balance_vars]
balance_table['Treated_Mean_Pre'] = before_df['mean_t'].round(4)
balance_table['Control_Mean_Pre'] = before_df['mean_c'].round(4)
balance_table['Bias_Pre(%)'] = before_df['bias_pct'].round(2)
balance_table['P_Value_Pre'] = before_df['p_val'].apply(lambda x: f'{x:.4e}' if x < 0.0001 else f'{x:.4f}')
balance_table['SMD_Pre'] = before_df['smd'].round(4)
balance_table['Treated_Mean_Post'] = after_df['mean_t'].round(4)
balance_table['Control_Mean_Post'] = after_df['mean_c'].round(4)
balance_table['Bias_Post(%)'] = after_df['bias_pct'].round(2)
balance_table['P_Value_Post'] = after_df['p_val'].apply(lambda x: f'{x:.4e}' if x < 0.0001 else f'{x:.4f}')
balance_table['SMD_Post'] = after_df['smd'].round(4)
bias_before_abs = before_df['bias_pct'].abs()
bias_after_abs = after_df['bias_pct'].abs()
balance_table['Bias_Reduction_Pct'] = ((bias_before_abs - bias_after_abs) / bias_before_abs * 100).round(2)
balance_table.loc[bias_before_abs == 0, 'Bias_Reduction_Pct'] = 100
print("\n" + "="*80)
print("Full Balance Test Results")
print("="*80)
print(balance_table.to_string(index=False))
# Export results
balance_table.to_csv("balance_test_full.csv", index=False, encoding='utf-8-sig')
print("\nBalance test results saved to balance_test_full.csv")
# ============================================================
# Step 6: ATT Estimation
# ============================================================
print("="*60)
print("Step 6: ATT Estimation")
print("="*60)
X = sm.add_constant(matched_df['alpha_gt_0.1'])
y = matched_df['novelty']
weights = matched_df['cem_weight']
model = sm.WLS(y, X, weights=weights).fit()
att = model.params['alpha_gt_0.1']
att_se = model.bse['alpha_gt_0.1']
att_pvalue = model.pvalues['alpha_gt_0.1']
ci_lower = att - 1.96 * att_se
ci_upper = att + 1.96 * att_se
# Export ATT results
att_results = pd.DataFrame([{
    'CEM_ATT': att,
    'ATT_pvalue': att_pvalue,
    'CI95_Lower': ci_lower,
    'CI95_Upper': ci_upper,
    'Total_Treated': (df['alpha_gt_0.1']==1).sum(),
    'Total_Control': (df['alpha_gt_0.1']==0).sum(),
    'Matched_Treated': (matched_df['alpha_gt_0.1']==1).sum(),
    'Matched_Control': (matched_df['alpha_gt_0.1']==0).sum(),
    'Avg_SMD_Pre_Matching': before_df['smd'].abs().mean(),
    'Avg_SMD_Post_Matching': after_df['smd'].abs().mean()
}])
att_results.to_csv("CEM_main_results.csv", index=False, encoding='utf-8-sig')
print("\nMain estimation results saved to CEM_main_results.csv")