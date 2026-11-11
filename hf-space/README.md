---
title: Credit Risk Predictor
emoji: 📉
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Loan default risk with a breakdown of every prediction
tags:
  - tabular-classification
  - scikit-learn
  - credit-risk
  - explainability
---

# Credit Risk Predictor

Estimates how likely a loan applicant is to default, and shows which inputs pushed the estimate up or down.

By [Sangam Gupta](https://www.sangamgupta.in) · [Case study](https://www.sangamgupta.in/work/credit-risk) · [Code](https://github.com/TheSangamX/credit_risk_predictor)

## Model card

- **Model**: scikit-learn `HistGradientBoostingClassifier` (200 iterations, learning rate 0.05, L2 1.0) inside a pipeline with its preprocessing, saved with `joblib`. For this Space its trees and preprocessing are exported to `model.json` and evaluated in JavaScript in the browser (`credit-risk.js`); predictions match scikit-learn to within 1e-6, and nothing you enter leaves the page.
- **Data**: Home Credit Default Risk application data.
- **Decision threshold**: 15% instead of the default 50%, because missing a defaulter costs more than a second look at a safe applicant.

| Metric (test set) | Score |
|---|---:|
| ROC-AUC | 0.760 |
| Recall at 15% | 41.8% |
| Precision at 15% | 25.0% |
| F1 at 15% | 31.3% |
| Accuracy | 85.2% |

## How each prediction is explained

The page walks every tree in the model for the applicant you enter and credits each split to the feature it used (Saabas path attribution, with each node's value taken as the count-weighted average of the leaves below it). The starting point plus the contributions add up exactly to the model's raw output.

## Limits

A learning project, not a lending tool. Inputs not shown in the form are fixed at typical values from the training data. The data includes gender, which a real lender would need to review for fair-lending reasons.

The Space is deployed from the [GitHub repo](https://github.com/TheSangamX/credit_risk_predictor) by a GitHub Action.
