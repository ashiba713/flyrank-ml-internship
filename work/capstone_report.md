# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Ashiba Alben A
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/ashiba713/flyrank-ml-internship
- **Date:** 2026-08-30

## 1. Problem framing

This analysis supports the decision of which content items may deserve closer performance review based on their historical search signals. The unit of analysis is a content item within an anonymized client group. The output is a model-based assessment of whether an item will experience a future impression decline of at least 20%, intended as decision support rather than an automatic intervention rule.

A human editor could use the output to prioritize monitoring and further investigation of content performance. A wrong call could waste review time or cause an editor to overlook a content item that deserves attention. Machine learning is useful here because multiple historical signals can be considered together and evaluated on unseen client groups rather than relying only on a single manual rule.

## 2. Data safety

The analysis used the FlyRank ML Internship warehouse data. The warehouse end date used for the experiment was **2026-06-30**. The final modeling dataset contained **111,247 content items**.

The target was constructed as a future outcome:

- `future_declined = 1` when future impressions fell by at least 20% relative to the baseline period.
- `future_declined = 0` otherwise.

The final leakage-safe model used only three historical features:

- `impressions_feature_30d`
- `clicks_feature_30d`
- `avg_position_feature_30d`

`client_hash_id` and `content_hash_id` were retained only to identify groups and join data; they were never supplied as model features.

We deliberately excluded fields that were directly involved in constructing the target or could expose future information, including `impressions_last30`, `impressions_prev30`, `clicks_last30`, and other label-derived fields such as `trend_direction` and `trend_pct`. The earlier modeling attempt showed why this mattered: using variables directly involved in the target construction produced an artificially high result and was rejected as leakage.

The query table was inspected, but it did not provide a separate historical snapshot ending before the outcome window for this experiment, so its query features were not used as predictors. No client-identifying names, domains, URLs, private queries, or credentials are included in the analysis artifacts.

## 3. Baseline

The transparent baseline always predicts `declined` for every content item.

The future outcome distribution was:

- Declined: **71,592 (64.35%)**
- Not declined: **39,655 (35.65%)**

Therefore, the majority-class baseline achieved **64.35% accuracy**.

This is a fair baseline because it requires no learned model and represents the performance obtained simply by selecting the most common outcome. The Random Forest was evaluated on the same held-out test data and compared using the same accuracy metric.

## 4. Model / analysis

The model was a **Random Forest Classifier** with 200 trees, `random_state=42`, parallel processing, and balanced class weights.

Random Forest was selected because it can model nonlinear relationships between multiple numerical historical signals without requiring a linear relationship between the inputs and the outcome. The model was evaluated on client groups that were not present in training.

The exact features used were:

1. `impressions_feature_30d`
2. `clicks_feature_30d`
3. `avg_position_feature_30d`

The target was `future_declined`: an item is labeled 1 when its future impression change is at least a 20% decline.

The model intentionally excluded features that directly defined the target or could leak future information. Client identifiers were also excluded from the model itself.

## 5. Evaluation

A grouped train/test split was used so that complete client groups remained on only one side of the evaluation. The split used `random_state=42` and a 20% test size.

The final split contained:

- Training rows: **97,494**
- Testing rows: **13,753**
- Training clients: **40**
- Testing clients: **10**
- Client overlap: **0**

The Random Forest achieved:

- Accuracy: **60.52%**
- ROC-AUC: **0.5799**
- Precision: **68.47%**
- Recall: **73.50%**

The baseline accuracy was **64.35%**, so the Random Forest did not outperform the majority-class baseline on accuracy.

The confusion matrix on the held-out test set was:

- True negatives: **1,713**
- False positives: **3,045**
- False negatives: **2,384**
- True positives: **6,611**

The model identified many declining items, but it also produced a substantial number of false positives. Overall, the ROC-AUC of 0.5799 indicates limited discrimination between future declines and non-declines using these three historical features.

## 6. Interpretation

The Random Forest's feature importance values were:

| Feature | Importance |
|---|---:|
| `avg_position_feature_30d` | 55.29% |
| `impressions_feature_30d` | 37.95% |
| `clicks_feature_30d` | 6.75% |

Among the available historical signals, average position had the highest reported feature importance, followed by impressions. Clicks contributed less to the fitted model.

These importance values describe how the model used the available features; they do not show that any feature causes future impression decline.

The main negative result is important: the leakage-safe model did not beat the simple majority-class baseline. The earlier 92.62% accuracy result was not treated as valid because its features directly participated in the target construction. After removing those leakage sources and using a future outcome with grouped validation, performance was substantially lower.

## 7. Recommendation

The ranked action playbook is:

**1. Prioritize monitoring of historical average search position.**  
It had the highest model-reported feature importance (55.29%) among the three available signals. This makes it a useful signal for review, not proof of a causal effect.

**2. Use historical impressions as additional context.**  
`impressions_feature_30d` accounted for 37.95% of model feature importance and can provide context when reviewing content performance.

**3. Treat historical clicks as a supporting signal.**  
`clicks_feature_30d` had the lowest feature importance (6.75%) in this model.

**4. Do not automate intervention from this model.**  
Because the model's 60.52% accuracy was below the 64.35% majority-class baseline, the current model should be treated as an exploratory decision-support analysis rather than a production decision rule.

**5. Investigate additional leakage-safe historical features.**  
The weak predictive performance suggests that the three available historical signals are insufficient for reliable prediction in this validation design. Future work could investigate additional historical content and query-mix signals when a clean pre-outcome time window is available.

Confidence is **limited/moderate for the observed feature-importance pattern and low for production prediction**. These findings are directional and apply to this dataset, feature set, time window, and grouped validation design.

## 8. Reproducibility

The analysis was developed in the repository notebook:

`work/notebooks/capstone.ipynb`

The notebook uses the FlyRank warehouse release through the documented Hugging Face workflow and DuckDB for aggregation, with pandas and scikit-learn for modeling.

Key reproducibility settings include:

- Random seed: **42**
- Model: `RandomForestClassifier`
- Trees: **200**
- Test size: **20%**
- Grouping variable: `client_hash_id`
- Model features: `impressions_feature_30d`, `clicks_feature_30d`, `avg_position_feature_30d`

The notebook should be rerun from top to bottom in a fresh environment using its existing setup cells and dependencies. The final reported values in this document correspond to the completed notebook run.

---

## Claims checklist

The findings in this report use observed, measured, directional, and decision-support language. The **64.35% majority-class base rate** is reported alongside model accuracy so that the accuracy result is interpreted in context.

No causal claims are made. This analysis does not claim to predict Google's ranking algorithm, prove how Google ranks content, or establish that changing any feature will improve performance. No client-identifying details are included.
