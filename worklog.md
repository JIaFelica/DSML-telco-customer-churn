# Project Work Log - Telco Customer Churn Prediction

**Xinyu Jia (C00304524)** | Data Science & Machine Learning Portfolio

---

### Week 1 (Mon 05 Oct 2026 – Sun 11 Oct 2026)

**Completed Tasks**
- Read the DSML portfolio specification and checked the deliverables required for the draft and final submission.
- Chose the Telco Customer Churn dataset from Kaggle (simulated IBM data) for a binary-classification task.
- Set two research questions: which customer characteristics are associated with churn, and how well a supervised model can predict churn.

**Issues and Fixes**
- The dataset contained a mix of categorical and numeric columns, so I noted that encoding would be needed before modelling.

**Next Step**
- Load the CSV in Colab and inspect the structure, data types, and missing values.

---

### Week 2 (Mon 12 Oct 2026 – Sun 18 Oct 2026)

**Completed Tasks**
- Uploaded the CSV with `files.upload()` and read it with `pd.read_csv`.
- Ran `.shape` and `.info()`: the raw data has 7,043 rows and 21 columns.
- Checked the target `Churn` and identified the feature groups (demographics, tenure, services, contract, charges, payment method).

**Issues and Fixes**
- `TotalCharges` was stored as `object` (text) instead of numeric, even though it contains charge values. I planned to convert it with `pd.to_numeric(errors="coerce")` in the cleaning stage.

**Next Step**
- Clean the data: remove the ID column, fix `TotalCharges`, and handle the rows that do not convert.

---

### Week 3 (Mon 19 Oct 2026 – Sun 25 Oct 2026)

**Completed Tasks**
- Dropped `customerID` because it is only an identifier.
- Converted `TotalCharges` to numeric and dropped the 11 missing rows, which were new customers with `tenure = 0`. After cleaning, 7,032 rows remain: 5,163 stayed and 1,869 churned (about 26.6%).
- Binarized the target (`Yes = 1`, `No = 0`), one-hot encoded the categories with `drop_first=True`, and split the data 80/20 with `stratify=y`: train 5,625 rows, test 1,407 rows.
- Plotted the overall churn split and the first group-by-group charts (contract type, monthly charges, payment method).

**Issues and Fixes**
- The classes are mildly imbalanced, so I noted from the course that accuracy alone can be misleading; recall for the churn class also needs to be checked.

**Next Step**
- Finish the remaining EDA charts and start the Random Forest model.

---

### Week 4 (Mon 26 Oct 2026 – Sun 01 Nov 2026)

**Completed Tasks**
- Added the remaining EDA plots: feature correlation with churn, the numeric correlation heatmap, and the internet-service chart.
- Trained a `RandomForestClassifier(n_estimators=100, random_state=42)` on the training set.
- Evaluated on the held-out test set: Accuracy 0.7896, ROC-AUC 0.8165. Churn-class recall is 0.52, F1 0.57; confusion matrix TN 917, FP 116, FN 180, TP 194.
- Produced the ROC curve and feature-importance plot, and exported the figures to the `plots/` folder.

**Issues and Fixes**
- After reopening Colab, the kernel variables were cleared and an early `NameError: df is not defined` appeared. I fixed it by running all cells in order and re-uploading the CSV, rather than running a single cell on its own.

**Next Step**
- Write up the README overview and prepare the draft submission to the lecturer.

---

### Week 5 (Mon 02 Nov 2026 – Sun 08 Nov 2026)

**Completed Tasks**
- Wrote the README overview covering the aim, dataset, preprocessing, EDA findings, model, evaluation, and recommendations.
- Submitted the draft repository link to the course lecturer by email on 03 Nov 2026.
- Compared the README figures with the notebook outputs and confirmed the confusion matrix, ROC curve, and feature importance matched the saved plots.

**Issues and Fixes**
- The README referenced an internet-service chart that was not yet generated in the notebook. I kept this note for the final iteration and added the missing code in Week 7.

**Next Step**
- Re-check the notebook against the specification and list the items to improve before the final submission.

---

### Week 6 (Mon 09 Nov 2026 – Sun 15 Nov 2026)

**Completed Tasks**
- Re-checked the notebook against the specification: name and student ID were added as a comment at the top of the first code cell.
- Reviewed the classification report: the model identifies about half of churners, so it should be described as a risk flag rather than a final decision.
- Listed the final improvements needed: section explanations in the notebook, the missing internet-service code, clearer report wording, the work log, and the dependency list.

**Issues and Fixes**
- Some report sentences implied that features cause churn. I flagged these for rewriting to state association only, and planned a limitations note that the data are simulated and the findings are for academic demonstration.

**Next Step**
- Fill in the written section explanations inside the notebook and polish the report wording.

---

### Week 7 (Mon 16 Nov 2026 – Fri 20 Nov 2026)

**Completed Tasks**
- Added markdown explanations for each workflow stage: overview, data loading, initial inspection, cleaning, EDA, group comparisons, encoding and split, model choice, evaluation, and key findings.
- Added the internet-service churn chart that was missing from the earlier version, and ran all cells in order to confirm there were no errors.
- Revised the README wording: feature importance and EDA show association with churn, not causation. Terms such as "drivers" and "drive churn" were replaced with "associated with churn risk".
- Added the reason for dropping the missing `TotalCharges` rows, the `requirements.txt` dependency list, and the order for running the notebook.
- Prepared the final files: annotated notebook, README report, plots, work log, and reference list.

**Issues and Fixes**
- After the wording review, the README still used a few causal phrases. I replaced them with association language and added a limitation stating that the findings cannot prove a feature itself causes a customer to leave.

**Next Step**
- Submit the final repository link before the deadline, then prepare the short presentation for the assessment session.

---

### Week 8 (Mon 23 Nov 2026 – Sun 29 Nov 2026)

**Completed Tasks**
- Following the supervisor's feedback, added alternative models for comparison: a Logistic Regression scaled pipeline, a class-weighted Random Forest, and a Gradient Boosting classifier, all evaluated on the same test set with Accuracy, ROC-AUC, churn recall and churn F1.
- Applied SMOTE to balance the training set only; the test set was left unchanged to avoid data leakage, then a Random Forest was trained on the resampled data to test imbalance handling.
- Ran a decision-threshold analysis on the Logistic Regression probabilities across thresholds 0.30–0.60, and saved the precision/recall threshold plot to the `plots/` folder.
- Tuned the Random Forest with a small `GridSearchCV` (5-fold cross-validation, scored on churn F1) and recorded the best parameters and the held-out test-set results.
- Updated the README comparison table, added sections for SMOTE, threshold analysis and tuning, refreshed the conclusions and reference list, and added `imbalanced-learn` to `requirements.txt`.

**Issues and Fixes**
- The class-weighted Random Forest did not raise minority-class recall on this dataset, and grid tuning improved ROC-AUC but not churn recall. These mixed results were reported honestly in the README rather than presented as a clear improvement.

**Next Step**
- Upload the updated notebook, README, dependency list and new plot to the repository, then prepare the short presentation.

---

*All work above was completed independently for the continuous assessment. The dataset is public Kaggle data (simulated IBM data); tools and sources are listed in the README references.*
