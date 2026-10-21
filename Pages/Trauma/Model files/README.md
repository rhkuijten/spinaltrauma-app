# Trauma model files

Drop the fitted artefacts here:

- `trauma_final.pkl` — the fitted pipeline (required for predictions). It is expected
  to do all cleaning, imputation, spline expansion, one-hot encoding and scaling
  internally; the app passes raw values straight through.
- `explainer.pkl` — optional, a fitted SHAP explainer over the transformed feature space.
- `background.pkl` — optional, a transformed background sample. If present (and no
  `explainer.pkl`), `shap.Explainer` is built against it, which supports non-tree models.

If none of the optional files are present, a `shap.TreeExplainer` is attempted; if the
final estimator is not tree-based, the decision tool skips the explanation plot and
still shows the prediction.

## Expected input columns

`COLUMNS` in `Pages/Trauma/trauma_utils.py` defines the 18 raw columns handed to the
pipeline, in order:

| Column | Values |
| --- | --- |
| `LEEFTIJD_EXACT` | age in years, 18–120 |
| `Geslacht01` | 0 = male, 1 = female |
| `BMI` | 10–80 |
| `ISS08` | 0–75 |
| `NUMBER_OF_FRACTURES` | integer ≥ 1 |
| `MFI5` | 0, 1, 2, 3 (3 = "3 or more") |
| `COMORB` | ASA grade 1–5 |
| `ASIA` | 1 = A … 5 = E, 0 = exam not performable |
| `TBI_CT` | 0 / 1 |
| `TYPE_A_INJURY`, `TYPE_B_INJURY`, `TYPE_C_INJURY`, `TYPE_F_INJURY` | 0 / 1, not mutually exclusive |
| `C`, `T`, `L`, `S` | `'JA'` / `'NEE'` |
| `INTENTIE` | raw mechanism code, or `NaN` for unknown |

`INTENTIE` is sent as one representative raw code per group: 1 = MVA, 4 = OVA,
7 = violence, 10 = low-energy fall (reference), 11 = high-energy fall, 12 = other,
`NaN` = unknown.

`NEURO_NOT_EXAMINABLE` and `INTENTIE_cat` are derived by the pipeline and are
deliberately not collected by the app.

The outcome is `D30_Mortality`, so `predict_proba()[:, 1]` is the probability of death
within 30 days; the decision tool shows that alongside its complement.
