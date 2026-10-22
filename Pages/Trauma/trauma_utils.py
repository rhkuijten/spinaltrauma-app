"""Input handling and SHAP helpers for the spinal trauma mortality model.

The fitted pipeline does all cleaning, imputation, spline expansion, one-hot
encoding and scaling internally, so this module deliberately does NOT transform
anything. It only collects the raw widget values into a one-row DataFrame with
the column names the pipeline expects.

The model lives at Pages/Trauma/Model files/final_model.joblib and unpickles as a
`src.validation.bootstrap._FittedModel`, so the vendored `src/` package at the
repo root must stay importable.
"""

import os

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join("Pages", "Trauma", "Model files")
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.joblib")

# The model is fitted on D30_Mortality, so predict_proba()[:, 1] is the
# probability of death within 30 days.
OUTCOME_LABEL = "30-day mortality"

# Raw columns handed to the pipeline, in the order listed in the model spec.
COLUMNS = [
    "LEEFTIJD_EXACT",
    "Geslacht01",
    "BMI",
    "ISS08",
    "NUMBER_OF_FRACTURES",
    "MFI5",
    "COMORB",
    "ASIA",
    "TBI_CT",
    "TYPE_A_INJURY",
    "TYPE_B_INJURY",
    "TYPE_C_INJURY",
    "TYPE_F_INJURY",
    "C",
    "T",
    "L",
    "S",
    "INTENTIE",
]

AO_TYPE_COLUMNS = ["TYPE_A_INJURY", "TYPE_B_INJURY", "TYPE_C_INJURY", "TYPE_F_INJURY"]
AO_TYPE_LABELS = {
    "TYPE_A_INJURY": "Type A",
    "TYPE_B_INJURY": "Type B",
    "TYPE_C_INJURY": "Type C",
    "TYPE_F_INJURY": "Type F",
}

REGION_COLUMNS = ["C", "T", "L", "S"]
REGION_LABELS = {
    "C": "Cervical",
    "T": "Thoracic",
    "L": "Lumbar",
    "S": "Sacral",
}

# (display label, value sent to the pipeline)
SEX_OPTIONS = [("Male", 0), ("Female", 1)]

TBI_CT_OPTIONS = [
    ("No intracranial finding on CT", 0),
    ("Intracranial finding on CT", 1),
]

# Values >= 3 are collapsed to 3 by the pipeline, hence the "3 or more" label.
MFI5_OPTIONS = [("0", 0), ("1", 1), ("2", 2), ("3 or more", 3)]

ASA_OPTIONS = [
    ("I - healthy", 1),
    ("II - mild systemic disease", 2),
    ("III - severe systemic disease", 3),
    ("IV - severe disease, constant threat to life", 4),
    ("V - moribund", 5),
]

# ASIA 0 makes the pipeline set NEURO_NOT_EXAMINABLE itself; there is
# deliberately no separate widget for that flag.
ASIA_OPTIONS = [
    ("A - complete", 1),
    ("B - sensory incomplete", 2),
    ("C - motor incomplete", 3),
    ("D - motor incomplete, useful function", 4),
    ("E - normal", 5),
    ("Exam not performable (intubated / sedated / severe TBI)", 0),
]

# The pipeline groups raw INTENTIE codes into 7 levels; send one representative
# raw code per group. The pipeline's seventh level, "unknown" (a missing code), is
# deliberately not offered: the model has not been tested with missing values.
INTENTIE_OPTIONS = [
    ("Motor vehicle accident", 1),
    ("Other vehicle accident (bicycle / pedestrian / etc.)", 4),
    ("Violence", 7),
    ("Low-energy fall (standing height)", 10),
    ("High-energy fall (from height)", 11),
    ("Other", 12),
]

# Matches waterfall_group_threshold_pct in the training repo's config/pipeline.yaml,
# which the fitted model's embedded config predates. Predictors contributing this
# percentage or less of the patient's summed absolute contributions are pooled.
WATERFALL_GROUP_THRESHOLD_PCT = 5.0

# Labels the training repo's config/pipeline.yaml has but the fitted model's
# embedded config predates: INTENTIE_cat labels the row the injury-mechanism
# dummies collapse into. The model's own labels win wherever it has one.
FALLBACK_DISPLAY_NAMES = {"INTENTIE_cat": "Injury mechanism"}


def labels(options):
    """Display labels for a list of (label, value) option pairs."""
    return [label for label, _ in options]


def value_for(options, label):
    """Look the pipeline value up by display label."""
    return dict(options)[label]


def build_input_row(inputs):
    """Assemble the raw one-row input for the pipeline.

    `inputs` is keyed on the pipeline's own column names. Nothing is encoded,
    scaled or imputed here - the pipeline does all of that internally.
    """
    return {column: inputs[column] for column in COLUMNS}


def to_frame(row):
    """One-row DataFrame with the columns in the order the pipeline expects."""
    return pd.DataFrame([row], columns=COLUMNS)




def load_model():
    """Load the fitted model, or return None if it has not been added yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict(model, row):
    """Probability of 30-day mortality for one patient.

    `_FittedModel.predict_proba` returns a 1-D array of positive-class
    probabilities, not the sklearn-standard (n, 2), so index it accordingly.
    """
    return float(np.asarray(model.predict_proba(to_frame(row))).ravel()[0])


# --------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------
# The explanation reuses the training repo's own code, vendored under
# explainability/, so the figure shown here is the figure the manuscript reports.


def display_names(model):
    """Plot labels, read from the config carried inside the fitted model.

    Taking them from the model rather than from a copy in this repo means the
    labels cannot drift out of step with the model that is actually loaded.
    """
    config = model.pipeline.named_steps["cleaner"].config
    return {
        **FALLBACK_DISPLAY_NAMES,
        **config.get("explainability", {}).get("display_names", {}),
    }


def compute_shap_values(model, row):
    """Exact SHAP values for one patient, in the training repo's layout.

    The training repo computes these with `shap.LinearExplainer` over the whole
    cohort as background, giving phi = beta * (x - mean(x)). That needs the
    cohort, which this app cannot ship - it holds no patient data. It is not
    needed either: `Scaler` is the last preprocessing step and was fitted on that
    same cohort, so the standardized features have mean exactly 0 and the
    identity collapses to phi = beta * x with the intercept as base value. Same
    numbers, no cohort.

    Layout, grouping and labels all come from the vendored training-repo code, so
    the result is interchangeable with `explainability.shap_values.compute_shap`
    called with the groups the training repo's waterfalls use.

    Returns a shap.Explanation holding a single patient (index 0), on the
    log-odds scale, or None if it could not be built.
    """
    import shap

    from explainability.shap_values import (
        _collapse,
        _onehot_levels,
        display_frame,
        onehot_groups,
        spline_groups,
    )

    try:
        X = to_frame(row)
        X_model = model.preprocessor.transform(X)
        X_display = display_frame(model.preprocessor, X)

        if list(X_display.columns) != list(X_model.columns):
            raise ValueError(
                "Display frame columns differ from model input columns; the "
                "preprocessing pipeline no longer keeps the scaler layout-neutral."
            )

        coef = np.asarray(model.model.coef_, dtype=float).ravel()
        intercept = float(np.asarray(model.model.intercept_).ravel()[0])
        values = np.asarray(X_model, dtype=float) * coef

        expected = model.model.decision_function(X_model)
        if not np.allclose(values.sum(axis=1) + intercept, expected, atol=1e-6):
            raise ValueError("SHAP values are not additive w.r.t. decision_function.")

        # Grouped after the additivity check, which summing preserves. As in the
        # training repo's waterfalls, the age spline basis and the mutually
        # exclusive injury-mechanism dummies are each summed into one row, and the
        # mechanism row shows the patient's level rather than a dummy indicator.
        values, X_display = _collapse(
            values,
            X_display,
            {**spline_groups(model), **onehot_groups(model)},
            _onehot_levels(model, X_display),
        )

        names = [display_names(model).get(c, c) for c in X_display.columns]

        return shap.Explanation(
            values=values,
            base_values=np.full(len(X_model), intercept),
            data=X_display.to_numpy(),
            feature_names=names,
        )
    except Exception as e:  # noqa: BLE001 - the plot is optional, the prediction is not
        print(f"Could not compute SHAP values for the trauma model: {e}")
        return None


def plot_explanation(explanation):
    """Waterfall for this patient, drawn exactly as the training repo draws its examples.

    Contributions are shown as percentage points of absolute risk, chaining from
    the SHAP reference probability to this patient's prediction. That rescaling is
    an allocation of the total risk difference rather than each predictor's
    marginal effect; signs and ranking are unchanged. See
    `explainability.plots._probability_contributions`.
    """
    from explainability.plots import plot_waterfall

    return plot_waterfall(
        explanation, index=0, group_threshold_pct=WATERFALL_GROUP_THRESHOLD_PCT
    )
