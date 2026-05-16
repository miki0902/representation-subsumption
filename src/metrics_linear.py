"""
Linear containment metrics: bidirectional regression R² between Large and Small features.

Why: If Large linearly subsumes Small, we expect high R²_L→S and lower R²_S→L.
Bidirectionality lets us distinguish subsumption from isomorphism.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


@dataclass
class LinearMetrics:
    r2_l_to_s: float
    r2_s_to_l: float
    mse_l_to_s: float
    mse_s_to_l: float
    directional_gap: float  # r2_l_to_s - r2_s_to_l


def compute_linear_metrics(
    F_L: np.ndarray,
    F_S: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    use_ridge: bool = False,
    ridge_alpha: float = 1.0,
) -> LinearMetrics:
    """Compute bidirectional linear regression R² between F_L and F_S.

    Train/test split prevents overfitting artifacts in high-dimensional settings.
    Ridge regression is available to handle near-collinear features.

    Returns LinearMetrics with R², MSE, and directional_gap.
    """
    F_L_train, F_L_test, F_S_train, F_S_test = train_test_split(
        F_L, F_S, test_size=test_size, random_state=random_state
    )

    r2_l_to_s, mse_l_to_s = _fit_and_eval(F_L_train, F_S_train, F_L_test, F_S_test, use_ridge, ridge_alpha)
    r2_s_to_l, mse_s_to_l = _fit_and_eval(F_S_train, F_L_train, F_S_test, F_L_test, use_ridge, ridge_alpha)

    return LinearMetrics(
        r2_l_to_s=r2_l_to_s,
        r2_s_to_l=r2_s_to_l,
        mse_l_to_s=mse_l_to_s,
        mse_s_to_l=mse_s_to_l,
        directional_gap=r2_l_to_s - r2_s_to_l,
    )


def _fit_and_eval(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    use_ridge: bool,
    alpha: float,
) -> tuple[float, float]:
    """Fit regression X→Y and evaluate on held-out test set.

    Why: Evaluating on test set ensures the R² reflects generalization,
    not memorization of training features.
    """
    reg = Ridge(alpha=alpha) if use_ridge else LinearRegression()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    r2 = float(r2_score(Y_test, Y_pred, multioutput="uniform_average"))
    mse = float(mean_squared_error(Y_test, Y_pred))
    return r2, mse
