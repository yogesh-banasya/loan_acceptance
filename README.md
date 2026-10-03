# Loan Approval Prediction (Classification)

A beginner-friendly classification project: given a loan application, predict whether it will be **approved or rejected**. The notebook walks through the full workflow in plain English: explore the data, handle missing values, train three simple models, compare them with a baseline, and explain *why* the model decides what it does.

## Dataset

`loan_data.csv` contains 614 loan applications from a public practice dataset often called the *Loan Prediction* dataset. About **68.7%** of applications are approved.

| Column | Description |
|---|---|
| `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed` | Applicant background |
| `ApplicantIncome`, `CoapplicantIncome` | Monthly income |
| `LoanAmount`, `Loan_Amount_Term` | Amount requested and repayment term |
| `Credit_History` | Good (1) or bad (0) past repayment record |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | **Target:** approved (Y) or rejected (N) |

> **Note:** The label is the lender's **historical decision**, not whether the borrower later repaid, so the model learns to imitate past approvals. How the data was collected is not documented, so treat it as a learning dataset.

## Approach

1. **Explore:** missing values, class balance, approval rate by feature
2. **Split:** 80/20 train/test, stratified (split *before* filling missing values to avoid leakage)
3. **Prepare:** one pipeline that fills blanks (median for numbers, a "Missing" category for text) and one-hot encodes categories
4. **Train:** Baseline ("always approve"), Logistic Regression, Decision Tree (depth 3), Random Forest
5. **Validate:** 5-fold cross-validation on the training set
6. **Evaluate:** accuracy, precision, recall, F1, confusion matrix, and "rejections caught"
7. **Explain:** decision-tree flowchart and Logistic Regression coefficients
8. **Fairness check:** does the model need Gender and Marital status?

## Results

### Cross-validation accuracy (training set, mean ± sd over 5 folds)

| Model | Accuracy |
|---|---|
| Baseline (always approve) | 0.69 ± 0.00 |
| Logistic Regression | 0.79 ± 0.04 |
| Decision Tree (depth 3) | 0.78 ± 0.04 |
| Random Forest | 0.80 ± 0.05 |

### Test set (123 applications)

| Model | Accuracy | Precision | Recall | F1 | Rejections caught |
|---|---|---|---|---|---|
| Baseline (always approve) | 0.69 | 0.69 | 1.00 | 0.82 | 0.00 |
| Rule: approve if credit history is Good | 0.80 | 0.82 | 0.91 | 0.86 | 0.55 |
| **Logistic Regression** | **0.86** | 0.84 | 0.99 | 0.91 | 0.58 |
| Decision Tree (depth 3) | 0.85 | 0.82 | 0.99 | 0.90 | 0.53 |
| Random Forest | 0.85 | 0.83 | 0.99 | 0.90 | 0.55 |

### Key takeaways

- **Credit history dominates:** about 80% of applicants with a good history are approved versus about 8% with a bad history. Income barely differs between approved and rejected applications.
- **A one-line credit-history rule already reaches 80% accuracy.** The models improve on it by only a few points, on a very small test set.
- **The models are good at spotting approvals but weak at catching rejections** (about 53–58%), which is the mistake that matters most to a lender.
- The three models perform about the same, so I recommend **Logistic Regression** because it is simple and explainable.
- **Fairness check:** dropping Gender and Marital status leaves cross-validated accuracy essentially unchanged (0.80 vs 0.79), so the model does not need them.

## How to run

```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
jupyter notebook loan_approval_classification.ipynb
```

Keep `loan_data.csv` in the same folder as the notebook.

## Project structure

```
├── loan_approval_classification.ipynb   # full walkthrough with explanations
├── loan_data.csv                        # dataset
└── README.md
```

## Limitations

- Small dataset (614 applications, 123 in the test set), so accuracies are only reliable to within a few points.
- The label is a past decision, not repayment, so any bias in past decisions is copied by the model.
- Little hyperparameter tuning; the goal is a clear, correct pipeline.

## Possible improvements

- Use repayment/default as the target
- Add features such as loan-to-income ratio
- Tune hyperparameters and choose a cost-based approval threshold

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter

## Author

Yogesh
