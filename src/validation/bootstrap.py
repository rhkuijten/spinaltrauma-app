"""Inference-only stand-in for the training repo's src/validation/bootstrap.py.

The fitted trauma model is pickled as a `src.validation.bootstrap._FittedModel`,
so that class must exist at this exact import path for the model to unpickle.

The training repo's module pulls in statsmodels, mlflow, optuna and the rest of
the development stack via `src.metrics` and `src.preprocessing.pipeline`. None of
that is needed to score a patient, so this file defines only the container and
its `predict_proba`, kept identical to the original. The development-time
functions (develop_model, bootstrap_validate, BootstrapResult, ...) are
deliberately absent.

Keep `_FittedModel`'s field names and `predict_proba` in sync with the training
repo; if they change there, re-export the model and update this file.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class _FittedModel:
    pipeline: object             # full fitted sklearn Pipeline (preprocessor + logreg)
    preprocessor: object         # pipeline[:-1]: fitted preprocessing sub-pipeline
    model: object                # pipeline[-1]: fitted LogisticRegression
    best_params: dict
    non_zero_mask: np.ndarray    # bool array over post-encoding feature names
    feature_names: list          # post-encoding feature names
    feature_means: np.ndarray    # Scaler mean_ per feature, on the pre-scaling scale
    feature_scales: np.ndarray   # Scaler scale_ per feature
    spline_knots: dict           # {source col: knot array}; {} if none

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class, as a 1-D array of length len(X).

        Note this differs from the sklearn convention of an (n, 2) array; it
        matches the training repo so a model exported there behaves identically
        here.
        """
        X_pp = self.preprocessor.transform(X)
        return self.model.predict_proba(X_pp)[:, 1]
