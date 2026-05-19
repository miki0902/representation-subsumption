"""
線形包含指標: Large と Small の特徴量間の双方向回帰 R²。

なぜ双方向で測るか:
  Large が Small を線形包含しているなら R²_L→S は高く R²_S→L は低いはず。
  双方向性を測ることで、包含と同型（両方向が高い）を区別できる。
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
    """F_L と F_S の双方向線形回帰 R² を計算する。

    なぜ train/test split を入れるか:
      高次元特徴量では訓練データへの過適合が起きやすく、
      R² が見かけ上高くなって包含の強さを過大評価してしまう。
      テストセットで評価することで汎化を確認する。

    Ridge 回帰は近似共線的な特徴量を扱う場合のオプション。

    戻り値: R², MSE, directional_gap を含む LinearMetrics。
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


def fit_linear_projector(
    F_X: np.ndarray,
    F_Y: np.ndarray,
    use_ridge: bool = False,
    ridge_alpha: float = 1.0,
) -> object:
    """F_X → F_Y の線形射影器を全データで学習して返す（可視化用）。
    戻り値: fitted sklearn regressor (has .predict(X) method)
    """
    reg = Ridge(alpha=ridge_alpha) if use_ridge else LinearRegression()
    reg.fit(F_X, F_Y)
    return reg


def _fit_and_eval(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    use_ridge: bool,
    alpha: float,
) -> tuple[float, float]:
    """X→Y の回帰をフィットし、ホールドアウトテストセットで評価する。

    なぜテストセットで評価するか:
      訓練データで R² を測ると過適合の影響で実際の汎化能力より高く見える。
      テストセットで評価することで「本当に予測できているか」を確認できる。
    """
    reg = Ridge(alpha=alpha) if use_ridge else LinearRegression()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    r2 = float(r2_score(Y_test, Y_pred, multioutput="uniform_average"))
    mse = float(mean_squared_error(Y_test, Y_pred))
    return r2, mse
