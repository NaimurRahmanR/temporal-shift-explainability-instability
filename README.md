Temporal Leakage, Distribution Shift, and Explainability Instability in High-Stakes Tabular Machine Learning

Naimur Rahman
Naazreen Tabassum

Abstract

Machine learning systems deployed in high-stakes tabular domains are commonly evaluated using random train–test splits that implicitly assume stationarity and exchangeability. In real-world deployment, however, temporal distribution shift, calibration drift, and model instability challenge these assumptions.

This repository presents a large-scale empirical investigation of temporal leakage, distribution shift, robustness, and explainability instability using a consumer credit dataset spanning 2007–2018 (≈1.35M observations). We compare random and strict forward temporal evaluation protocols, quantify numeric and categorical drift using population-level divergence measures, assess calibration degradation under deployment conditions, measure prediction sensitivity under controlled feature perturbations, and evaluate the temporal stability of SHAP-based feature attributions.

Our findings demonstrate that discrimination metrics (e.g., ROC AUC) can remain superficially stable under temporal deployment, while calibration error, probability stability, and feature importance rankings degrade as temporal distance increases. These results highlight a structural gap between conventional evaluation practice and real-world reliability requirements in high-stakes tabular ML systems.

1. Motivation

In applied machine learning for finance, healthcare, and public policy, model evaluation frequently relies on random splits of historical data. This practice assumes:

Temporal stationarity of feature distributions

Stability of conditional target relationships

Exchangeability of observations

Such assumptions rarely hold in non-stationary environments. When deployment occurs strictly forward in time, models may experience:

Distribution shift

Calibration drift

Sensitivity to small feature perturbations

Instability in explanation mechanisms

This work systematically quantifies these effects in a large-scale tabular credit risk setting.

2. Dataset

LendingClub-style consumer credit dataset

Time span: June 2007 – December 2018

Observations: ~1,347,681

Binary target: Default

Features include numeric credit indicators (FICO, loan amount, DTI, revenue, employment length) and high-cardinality categorical attributes (purpose, state, zip code, ownership class).

The raw dataset is not included due to size and licensing constraints.

3. Experimental Design
3.1 Evaluation Protocols

Three validation regimes are implemented:

Random Stratified Split (80/20)

Strict Temporal Split

Train: pre-2017

Test: 2017 onward

Rolling Forward Validation

Train cumulatively until year t

Test on year t+1

The rolling validation uses logistic regression to isolate temporal stability effects independent of gradient boosting capacity.

3.2 Models

Primary model:

XGBoost (native booster implementation)

300 boosting rounds

Depth 5

Learning rate 0.05

Subsampling and column subsampling

Histogram tree method

Preprocessing:

Median imputation for numeric features

Most-frequent imputation + one-hot encoding for categorical features

ColumnTransformer pipeline

To ensure compatibility with modern sklearn versions, the XGBoost model is trained using the native xgb.train() interface wrapped in a lightweight prediction wrapper.

3.3 Metrics

Discrimination:

ROC AUC

PR AUC

Calibration:

Brier Score

Expected Calibration Error (ECE)

Distribution Shift:

Kolmogorov–Smirnov statistic (numeric features)

Population Stability Index (PSI)

Jensen–Shannon divergence (categorical features)

Robustness:

Mean absolute probability shift

Classification flip rate (threshold = 0.5)

Explainability Stability:

SHAP global feature importance

Spearman rank correlation across temporal gaps

4. Empirical Findings
4.1 Random vs Temporal Evaluation
Setting	ROC AUC	PR AUC	Brier	ECE
Random Split	≈0.67	≈0.32	≈0.15	≈0.007
Temporal Split	≈0.68	≈0.35	≈0.16	≈0.025

Discrimination appears stable across splits. Calibration error increases materially under temporal deployment.

4.2 Rolling Forward Validation (2008–2018)

ROC AUC range: 0.59 – 0.66

ECE range: 0.0018 – 0.065

While discrimination fluctuates moderately, calibration exhibits substantial year-to-year instability.

4.3 Distribution Shift

Numeric features show statistically significant shift (KS test, p < 0.001).

Representative PSI values:

loan_amnt ≈ 0.0846

fico_n ≈ 0.0444

Categorical drift (JSD):

purpose ≈ 0.072

zip_code ≈ 0.057

These magnitudes indicate moderate but meaningful structural drift across time.

4.4 Robustness to Controlled Perturbations (2018 Deployment)
Feature	Perturbation	Mean Probability Shift	Flip Rate
FICO	-10	≈0.019	≈0.46%
FICO	+10	≈0.016	≈0.12%
DTI	±2	≈0.006	<0.1%
Revenue	±10%	≈0.006	<0.1%

Small financial shifts induce measurable probability changes, with FICO perturbations having the strongest effect.

4.5 Explainability Instability

SHAP global importance rankings degrade as temporal distance increases.

Spearman rank correlations:

Comparison	Gap (years)	Spearman
2012 vs 2015	3	≈0.50
2015 vs 2018	3	≈0.54
2012 vs 2018	6	≈0.42

Feature attribution stability decreases monotonically with temporal gap.

5. Interpretation

The central empirical result is not that discrimination collapses under temporal deployment. Rather, it is that:

Calibration degrades

Probability stability shifts

Feature importance rankings evolve

Small perturbations meaningfully alter predictions

These phenomena are largely invisible under conventional random evaluation protocols.

The implication is clear: evaluation strategies for high-stakes tabular ML must extend beyond ROC AUC and incorporate temporally aware validation, calibration analysis, robustness testing, and attribution stability measurement.

6. Reproducibility

Environment tested with:

numpy==2.0.2
pandas==2.2.2
scipy==1.14.1
scikit-learn==1.6.1
xgboost==2.1.3
shap==0.46.0


To reproduce:

pip install -r requirements.txt


Execute the notebook sequentially.
All generated tables and figures are exported to /results.

Random seed fixed at 42.

7. Repository Structure
notebook/
    temporal_shift_explainability_instability_tabular_ml.ipynb
results/
    rolling_performance.csv
    numeric_drift.csv
    categorical_drift.csv
    robustness_perturbation.csv
    shap_stability.csv
requirements.txt
README.md

8. Status

Working research manuscript.
Intended submission: arXiv (cs.LG).
