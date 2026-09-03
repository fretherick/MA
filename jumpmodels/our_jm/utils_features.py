from typing import Optional

import numpy as np
import pandas as pd

from jumpmodels.examples.nasdaq.feature import compute_ewm_DD


def compute_v0_features(
    ret: pd.Series,
    half_lives: list[int],
    feat_dict: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    for hl in half_lives:
        # Feature 1: EWM-ret
        feat_dict[f"ret_{hl}"] = ret.ewm(halflife=hl).mean()
        # Feature 2: log(EWM-DD)
        DD = compute_ewm_DD(ret, hl)
        feat_dict[f"DD-log_{hl}"] = np.log(DD)
        # Feature 3: EWM-Sortino-ratio = EWM-ret/EWM-DD
        if hl <= 5:
            # Compute for short half-lives, otherwise very similar to the EWM-ret
            feat_dict[f"sortino_{hl}"] = feat_dict[f"ret_{hl}"].div(DD)
        # Feature 4: EWM-DD-ret
        dd_ret = feat_dict[f"DD-log_{hl}"].diff().ewm(halflife=5).mean()
        feat_dict[f"DD-ret-ret_{hl}"] = dd_ret.diff().ewm(halflife=5).mean().diff()
    return feat_dict


def assign_weight_to_returns(rets: pd.Series, neg_weight: float) -> pd.Series:
    """Give different weights to positive and negative returns."""
    if neg_weight < 0 or neg_weight > 1:
        raise ValueError("The weight should be in [0, 1]")
    c_ret = rets.copy()
    c_ret[c_ret < 0] = c_ret[c_ret < 0] * neg_weight / 0.5
    c_ret[c_ret > 0] = c_ret[c_ret > 0] * (1 - neg_weight) / 0.5
    return c_ret


def feature_engineer(
    ret: pd.Series | pd.DataFrame,
    ver: str = "v0",
    cols_list: Optional[list[str]] = None,
    half_lives: Optional[list[int]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Engineer a set of features based on a return series.
    This function customizes the feature set according to the specified version string.

    Parameters
    ----------
    ret: pd.Series; input return series for feature engineering.
    ver: str; version of feature engineering to apply.
    cols_list: List[str]; list of column names to include in the output DataFrame.
    half_lives: tuple[int, int, int]; list of half-lives for the exponentially weighted moving averages.

    Returns
    -------
    pd.DataFrame; engineered feature set.
    """
    if half_lives is None:
        half_lives = [5, 20, 60]
    if isinstance(ret, pd.DataFrame):
        ret = ret[cols_list]
    feat_dict = {}
    if ver == "v0":
        feat_dict = compute_v0_features(ret, half_lives, feat_dict)
        return pd.DataFrame(feat_dict)
    if ver == "v1":
        asym_var_neg = kwargs.get("asym_var_neg", 0.5)
        if asym_var_neg < 0 or asym_var_neg > 1:
            raise ValueError(
                "The asymmetric factor to compute variance should be in [0, 1]"
            )
        feat_dict = compute_v0_features(ret, half_lives, feat_dict)
        feat_dict["cum_ret"] = ret.cumsum().copy()
        for hl in half_lives:
            # overwrite mean with sum (since log-returns)
            feat_dict[f"ret_{hl}"] = ret.ewm(halflife=hl).sum()
        c_ret = assign_weight_to_returns(ret, neg_weight=asym_var_neg)
        for hl in half_lives:
            feat_dict[f"var_{hl}"] = c_ret.ewm(halflife=hl).var()
        return pd.DataFrame(feat_dict)
    else:
        raise NotImplementedError()
