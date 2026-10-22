# Trauma evaluation figures

`Pages/Trauma/evaluation.py` looks for these file names. Any that are missing
render as a labelled placeholder rather than breaking the page.

Each file is a copy of an output of the analysis repository (`Spine-Trauma`):

| File | Source in `Spine-Trauma/outputs/` |
| --- | --- |
| `ROC Curve.png` | `stability/roc_instability.png` |
| `Calibration plot.png` | `stability/calibration_instability.png` |
| `DCA.png` | `stability/dca_instability.png` |
| `Coefficient forest plot.png` | `explainability/coefficient_forest.png` |
| `Age spline.png` | `explainability/spline_LEEFTIJD_EXACT.png` |
| `Waterfall low risk.png` | `explainability/shap_waterfall_p05_risk.png` |
| `Waterfall median risk.png` | `explainability/shap_waterfall_p50_risk.png` |
| `Waterfall high risk.png` | `explainability/shap_waterfall_p95_risk.png` |

The evaluation page quotes numbers from these figures (AUC, calibration, odds
ratios, waterfall contributions). When the figures are regenerated, update that
text as well.
