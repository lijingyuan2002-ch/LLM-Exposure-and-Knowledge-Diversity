# LLMs and Research Team Diversity / Novelty Analysis
This repository contains the core Python scripts for analyzing the impact of large language models (LLMs) exposure on team knowledge diversity and output novelty in interdisciplinary research teams.

# Script Overview 
The four main scripts are located under the "src/" directory.

1. LLMs-diversity-CEM.py
Purpose: Estimate the causal effect of LLM usage on team knowledge diversity using Coarsened Exact Matching (CEM).

2. LLMs-diversity-OLS.py
Purpose: Run OLS regression models to examine the relationship between LLM usage and team knowledge diversity.

3. LLMs-novelty-CEM.py
Purpose: Estimate the causal effect of LLM usage on research novelty using Coarsened Exact Matching (CEM).

4. LLMs-novelty-logit.py
Purpose: Run logistic regression models to analyze the relationship between LLM usage and novelty.

# Data Description
Three datasets are required for analysis:
1. Main bibliometric CSV files
Ten CSV files containing all treatment, outcome and covariate indicators for regression & CEM analysis. All files are archived on Zenodo (https://doi.org/10.5281/zenodo.21090759).

2. OpenAlex_topic_mapping_table.xlsx
Topic-field classification lookup table for OpenAlex topics. Reference: https://developers.openalex.org/api-reference/topics/list-topics

3. sciscinet_papers_filtered.parquet
Novelty dataset from SciSciNet-v2. We use "Atyp_10pct_Z" to construct binary novelty indicators. Project page: https://huggingface.co/datasets/Northwestern-CSSI/sciscinet-v2/tree/main
