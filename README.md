# Credit Scorecard: "Give Me Some Credit"

A beginner-friendly, end-to-end **credit scorecard** (PD model) built on the Kaggle
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (`cs-training.csv`, 150,000 borrowers).
Target: `SeriousDlqin2yrs` (1 = serious delinquency, 90+ days past due, within 2 years).

Pipeline: cleaning → stratified train/test split → binning + **WoE / IV** → **logistic regression** →
**points-based scorecard** (base score 600, odds 50:1, PDO 20) → evaluation (AUC, Gini, KS, calibration) → **Streamlit** app.

## Files

| File | What it is |
|---|---|
| `credit_scorecard.ipynb` | The tutorial notebook (one section per step, plain-words explanation before each one) |
| `app.py` | Streamlit app: sliders for the borrower's details, a Calculate button, a what-if comparison tab and a short explanation tab |
| `.streamlit/config.toml` | Colors and theme of the app |
| `scorecard.json` | The scorecard (bins, points, scale) written by the notebook and read by the app |
| `requirements.txt` | Python dependencies |
| `figures/` | Plots saved by the notebook (IV ranking, score distribution + ROC, calibration) |
| `data/` | Put `cs-training.csv` here (not committed, see `.gitignore`) |

## How to run

```bash
# 1. (optional) create an environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. put cs-training.csv in data/  (download it from Kaggle)

# 4. run the notebook top to bottom (this also creates scorecard.json)
jupyter notebook credit_scorecard.ipynb

# 5. launch the app
streamlit run app.py
```

`scorecard.json` is already included, so the app also works without re-running the notebook.

## Results (real numbers from the notebook run)

| Item | Value |
|---|---|
| Default rate | **6.68%** (10,026 defaults out of 150,000) |
| Split | 70% train (104,999) / 30% test (45,000), stratified; same 6.68% default rate in both |
| Top IVs | Credit utilization **1.140**, 90+ days late **0.892**, 30-59 days late **0.739**, 60-89 days late **0.601**, age **0.225** |
| Test **AUC** | **0.857** |
| Test **Gini** | **0.715** |
| Test **KS** | **0.562** |
| Train AUC / Gini / KS | 0.8575 / 0.715 / 0.554 (almost identical to test: no overfitting) |
| Calibration (test) | Average predicted PD 6.68% = observed default rate 6.68% |

Notes on the results:
- Calibration is good on average and in the riskiest decile (35.0% predicted vs 36.2% observed), but the model
  **over-predicts the safest deciles** (decile 1: 1.0% predicted vs 0.5% observed) and slightly under-predicts decile 9
  (10.2% vs 11.9%). The notebook shows this is not caused by rounding the points.
- Four variables have an IV above 0.5, which the usual rule of thumb flags as "check for leakage". They are past-due history and utilization,
  genuine behavioral signals, but the file alone cannot prove they were all measured before the 2-year window.

## Main modelling choices

- **Outliers / special values:** drop `age < 18` (1 row); cap past-due counts at 10 (the 96/98 values look like special codes; those rows default 54.6% of the time, so they are kept, not deleted); cap utilization at 2; `DebtRatio > 5` is set to *missing* (92% of those rows have no income: the column holds a raw amount, not a ratio).
- **Missing values:** `MonthlyIncome` and `DebtRatio` get their own "Missing" bin; `NumberOfDependents` is filled with the **train** median.
- **No leakage:** only fixed-threshold rules happen before the split; the median, bins, WoE, IV and the model are all learned on the train set only.
- **No class weighting** in the logistic regression, to keep the predicted PDs meaningful.
- **Scorecard scale:** `Factor = PDO / ln 2 = 28.85`, `Offset = 600 - Factor * ln 50 = 487.12`; points are rounded to integers (the scorecard PD differs from the model PD by about 0.001 on average).

## Limits (read before using this in an interview)

This is an educational application-style scorecard, **not** a production IFRS 9 or IRB model:
the target is a 2-year serious-delinquency flag (not a regulatory 12-month default), there is no lifetime PD,
no macro forward-looking adjustment, no LGD/EAD, no long-run-average calibration, no out-of-time validation (the file has no date),
no stability monitoring (PSI), and bins were chosen by hand from default-rate tables.
