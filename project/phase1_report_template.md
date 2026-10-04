# Phase 1 Report — [Your name], [dataset]
*Target ~500 words total. Replace every bracket. Due: Session 5 (Oct 12).*

## 1. Problem & dataset (~80 words)
[What are you predicting, for whom, and why would a business care? Dataset, size, class balance.]


## 2. Data preparation decisions (~100 words)
[≥2 concrete decisions: what you found (evidence), what you did, why. Leakage check goes here.]
    -profile shows no missing values, therefore no rows were removed:
    -age is reasonable - 21-79, no Credit limit under 0, no Negative payments: 0, PAY_0 have expected values, Negative bills (overpayment)
    -35 pairs of rows show up as duplicates apart from ID, 30 of them have 0 bills and 5 have the same amount billed and repaid 
    each month which is possibly due to fixed recurring charges from the same bank, therefore all the rows are kept
    -dropped ID as it has no relationship with the credit risk and can confuse the model
    -EDUCATION 0, 5,6 merged(1.2%), as well as Marriage (0.2%)  into "others" category as they are not documented  
    -Education and Marriage values converted to labels as they are categories not quantities
    -sex is dropped as it is viewed as discriminatory 
    -all features describe past events, so there is no leakage


## 3. Model selection rationale (~100 words)
[What you tried, what you chose, why — reference a baseline comparison.]
-Exploratory analysis shows repayment status is strongly related to default:
    -the higher the months delayed for repayment, the higher the % of default next month
    -clients who paid on time defaulted in 13-17% of the cases, while those who left behind defaulted
    in 50-78% meaning payment history is very strong predictor


| Model | Accuracy | Precision | Recall | F1 | AUC (val) |
|---|---|---|---|---|---|
| Majority baseline | 0.779 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.809 | 0.697 | 
0.243 | 0.361 | 0.714 |
| XGBoost | 0.818 | 0.665 | 0.358 | 0.465 | 0.778 |

The majority baseline reaches 77.9%, however it does not catch any defaulters, which is the actual added value.
Both logistic regression with 80.9% accuracy and XGBoost with 81.8% score better. XGBoost catches about 47% more 
defaulters -35.8% compared to 24.3%. Therefore, XGBoost was chosen to be tuned. 

XGBoost Tuning:
| Hyperparameter | Value | Train AUC | Val AUC | Gap |
|---|---|---|---|---|
| max_depth | 2 | 0.800 | **0.778** | 0.023 |
| max_depth | 4 | 0.852 | 0.778 | 0.074 |
| max_depth | 8 | 0.982 | 0.763 | 0.220 |
| max_depth | 12 | 1.000 | 0.752 | 0.248 |
| learning_rate | 0.01 | 0.791 | 0.771 | 0.020 |
| learning_rate | **0.1** | 0.852 | **0.778** | 0.074 |
| learning_rate | 0.3 | 0.917 | 0.756 | 0.161 |

*max_depth tested with learning_rate = 0.1; learning_rate tested with max_depth = 4.*

Chosen hyperparameters are max_depth = 2 and learning_rate = 0.1.
Max_depth of 2 has the same validation accuracy as max_depth of 4, however it has a smaller gap,meaning
it generalizes better.
Learning_rate of 0.1 outperforms in terms of validation accuracy, as 0.01 underfits and 0.3 overfits substantially.

Additionally, simpler trees are especially useful in backing up credit decisions to clients or regulators. 


## 4. Evaluation (~120 words)
[Metrics appropriate to the problem and what they mean HERE — e.g., what does your recall
imply operationally? Include val vs. test numbers and your tuning evidence (≥2 hyperparameters).]

## 5. Trade-offs & limitations (~100 words)
[Precision/recall choice, cost of errors, what you'd do with more time/data. One honest weakness.]

## Appendix (not counted)
Endpoint name + invocation screenshot/cell, AI-assistance disclosure (per syllabus policy).
