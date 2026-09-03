import logging
from typing import Callable, Tuple, Optional

import numpy as np
import pandas as pd

from jumpmodels.jumpmodels.jump import JumpModel

# TODO trans_cost_class needs to be implemented

def backtest_strategy_with_cash(
    df: pd.DataFrame,
    fee: float,
    logger: logging.Logger,
    cash_log_ret: float = 0.0,
    initial_value: float = 1.0,
    log_ret_col: str = "log_ret",
    alloc_col: str = "alloc",
    max_exposure_change: float = 1.0,
    max_capital: float = 1.0,
    min_capital: float = 0.0,
    jump_penalty: Optional[float] = None,
) -> pd.DataFrame:
    """
    Backtest a strategy where at the start of each period you hold a target allocation
    (a number between 0 and 1) in an asset that earns log returns, while the cash portion
    also earns a log return (e.g. a risk-free rate) specified by cash_log_ret. Changing
    your allocation incurs proportional transaction fees.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DateTime index and at least two columns:
          - log_return_col: the asset log-return for that period (will be exponentiated)
          - alloc_col: the target allocation (between 0 and 1) for the next period.
        (For the very last period no rebalancing is done.)
    fee : float
        Transaction fee rate (e.g. 0.001 for a 0.1% fee on the traded amount).
    logger : logging.Logger
    cash_log_ret : float, default is 0.0
        Log return for the cash portion each period (default is 0.0, i.e. cash is inert).
    initial_value : float, default is 1.0
        Initial portfolio value (cash we have to invest).
    log_ret_col : str, optional
        Name of the column in df with the asset’s log returns.
    alloc_col : str, optional
        Name of the column in df with the target allocation percentages.
    max_exposure_change : float, default is 1.0
        Maximum change in exposure allowed in a single period (e.g. one day).
    max_capital : float, default is 1.0
        Maximum capital that can be invested in the asset (considering potential leverage).
    min_capital : float, default is 0.0
        Minimum capital that can be invested in the asset (considering potential shorting).
    jump_penalty : float, optional
        Parameter of the jump model, only used for logging purposes.

    Returns
    -------
    result : pd.DataFrame
        DataFrame with the same index as df and the following columns:
          - 'portfolio_value': portfolio value at the end of the period (after rebalancing, when applicable)
          - 'daily_return': the net return during that period (as a fraction)
          - 'cumulative_return': cumulative return relative to the initial value.

    Simulation details
    ------------------
    For each period the simulation proceeds as follows:
      1. **Return Application:**
         - The asset portion grows by a factor of np.exp(asset_log_ret) where asset_log_ret is
           taken from the specified column.
         - The cash portion grows by a factor of np.exp(cash_log_ret).
      2. **Rebalancing (if not the last period):**
         At the end of the period, the portfolio is rebalanced to achieve the next period’s target
         allocation. Proportional transaction fees are incurred on the amount traded. The formulas
         solve for the trade amount that delivers the desired new fraction.
      3. **Performance Recording:**
         The function records the portfolio value, the net daily return, and the cumulative return.
    """
    df = df.sort_index().copy()
    n = len(df)
    portfolio_values = []  # Portfolio value at the end of each period (after rebalancing when applicable)
    daily_returns = []  # Net return during each period

    # import matplotlib.pyplot as plt
    # df["ret"].cumsum().plot(label="CumRet")
    # df["strat"].plot(label="Strategy")
    # plt.legend()
    # plt.show()

    # Initialize at time 0
    # Assume you start fully allocated as desired with no fee
    a0 = df.iloc[0][alloc_col]
    current_portfolio = initial_value
    asset = a0 * current_portfolio
    cash = (1 - a0) * current_portfolio

    for i in range(n):
        # 1. Apply the period's returns
        asset *= np.exp(df.iloc[i][log_ret_col])  # growth by its log return
        cash *= np.exp(cash_log_ret)  # similar for cash
        end_value = asset + cash  # ptf before rebalancing

        if i < n - 1:
            # 2. Rebalance for the next period
            desired_target = df.iloc[i + 1][alloc_col]
            current_weight = asset / end_value  # Current asset fraction

            # Limit the exposure change
            delta_desired = desired_target - current_weight
            if delta_desired > max_exposure_change:
                a_target = current_weight + max_exposure_change
            elif delta_desired < -max_exposure_change:
                a_target = current_weight - max_exposure_change
            else:
                a_target = desired_target
            # Ensure the effective target is within bounds
            a_target = min(max(a_target, min_capital), max_capital)

            if np.isclose(a_target, current_weight):
                new_asset = asset
                new_cash = cash
            elif a_target > current_weight:  # need to buy more asset
                # Let δ be the amount bought (which costs δ*(1+fee)). We solve:
                #   asset + δ = a_target*(end_value - fee*δ)
                # for δ:
                delta = (a_target * end_value - asset) / (1 + a_target * fee)
                new_asset = asset + delta
                new_cash = cash - delta * (1 + fee)
            else:  # a_target < current_weight: need to sell asset
                # When selling, you receive δ*(1-fee) dollars per δ sold.
                # We solve for δ:
                #   asset - δ = a_target*(end_value - fee*δ)
                delta = (asset - a_target * end_value) / (1 - a_target * fee)
                new_asset = asset - delta
                new_cash = cash + delta * (1 - fee)
            # Portfolio value after rebalancing (fees reduce this value)
            new_portfolio = new_asset + new_cash
            period_return = new_portfolio / current_portfolio - 1
            portfolio_values.append(new_portfolio)
            daily_returns.append(period_return)
            # Update portfolio state for the next period:
            current_portfolio = new_portfolio
            asset, cash = new_asset, new_cash
        else:
            # Final period: no rebalancing
            period_return = end_value / current_portfolio - 1
            portfolio_values.append(end_value)
            daily_returns.append(period_return)
    result = pd.DataFrame(
        {"pf_value": portfolio_values, "daily_returns": daily_returns}, index=df.index
    )
    result["cum_returns"] = result["pf_value"] / initial_value - 1
    if jump_penalty is not None:
        logger.info(f"Strategy output for lambda={jump_penalty}: \n{result}")
    return result


class JumpPenaltyFinder:
    """
    Class to find the best jump penalty for a given set of features and returns.
    The class uses a JumpModel to fit the data and evaluate the performance of a 0/1 strategy, i.e. a strategy that
    is either flat or fully invested, and, based on key indicators of such strategy on the validation set, decides
    the best jump penalty parameter.
    This approach is inspired by the paper "Dynamic asset allocation with asset-specific regime forecasts" by Shu, Yu
    and Mulvey.
    """
    def __init__(
        self,
        X_train_processed: pd.DataFrame,
        X_test_processed: pd.DataFrame,
        returns: pd.Series | pd.DataFrame,
        trans_cost_class: Callable,
        continuous_jm: bool = False,
        mode_loss: bool = True,
        number_jump_penalties: int = 11,
        jump_penalty_range: Tuple[float, float] = (0.0, 100.0),
        validation_only: bool = False,
        number_clusters: int = 2,
        logger: Optional[logging.Logger] = None,
        jp_logspace: bool = True,
        verbose: int = 0,
    ):
        if number_clusters != 2:
            raise ValueError("Only 2 clusters are supported for now.")
        self.number_clusters = number_clusters
        self.X_train_processed = X_train_processed
        self.X_test_processed = X_test_processed
        self.returns = returns
        self.trans_cost = trans_cost_class
        self.continuous_jm = continuous_jm
        self.mode_loss = mode_loss
        self.number_jump_penalties = number_jump_penalties
        self.jump_penalty_range = jump_penalty_range
        self.validation_only = validation_only
        self.verbose = verbose
        self.logger = logger
        self.best_model = {
            "jump_model": JumpModel(jump_penalty=0),
            "jump_penalty": 0.0,
            "sortino": -np.inf,
            "sharpe": -np.inf,
            "fin_cum_ret": -np.inf,
        }
        self.rets_dict = {
            "ret": returns,
        }
        jump_penalties = self._get_jump_penalties(jp_logspace)
        # Extend jump_penalties with extreme values
        self.jump_penalties = np.r_[
            jump_penalties[0] / 4, jump_penalties, jump_penalties[-1] * 3
        ]

    def _checks(self):
        if (start := self.jump_penalty_range[0]) < 0:
            raise ValueError("Jump penalties must be non-negative.")
        if (end := self.jump_penalty_range[1]) < start:
            raise ValueError("Invalid jump penalty range.")
        return start, end

    # @staticmethod
    def convert_labels_to_signal(self, clusters: pd.Series | pd.DataFrame) -> pd.Series:
        """
        Convert cluster assignments to signal: in `clusters`, 0 is bullish market (invest),
        1 is bearish market (do not invest). We need to invert those.
        """
        if isinstance(clusters, pd.DataFrame):
            # clusters[1] gives the probabilities of the bearish market (if I have a continuous Jump Model)
            clusters = clusters[1].copy()
        bull_index = clusters[clusters <= 0.5].index
        bear_index = clusters[clusters >= 0.5].index
        if not bull_index.empty:
            if clusters[bull_index].mean() < 0:
                raise ValueError("The mean of the bullish market should be positive.")
            if not bear_index.empty:
                if self.returns[bull_index].mean() < self.returns[bear_index].mean():
                    # import matplotlib.pyplot as plt
                    # self.returns[bull_index].plot.kde(label="Bull")
                    # self.returns[bear_index].plot.kde(label="Bear")
                    # plt.legend()
                    # plt.show()
                    self.logger.warning(
                        "The mean of the bullish market should be greater than the mean of the bearish market."
                    )
        # The following also works with the continuous version of JumpModels
        return clusters.apply(lambda x: 1 - x)

    def get_strategy_df(
        self,
        labels_test_online: pd.Series | pd.DataFrame,
        in_sample_labels: Optional[pd.Series],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        `joined_df` is the DataFrame containing returns and market participation, while `strat_df` is the DataFrame
        that has integers indicating the strategy. The strategy is 0 for no-investment and 1 for investment.
        `strat_df` needs to contain all samples as it is used also outside in `JumpModelEstimator.predict`.
        """
        labels = pd.concat([in_sample_labels, labels_test_online], axis=0)
        strat_df = self.convert_labels_to_signal(labels)
        if self.validation_only:
            len_validation = len(labels_test_online)
            strat_df = strat_df.iloc[-len_validation:]
            returns_df = self.returns[-len_validation:]
        else:
            returns_df = self.returns.copy()
        joined_df = pd.concat([returns_df, strat_df], axis=1, keys=["ret", "strat"])
        return joined_df, strat_df

    @staticmethod
    def compute_sharpe_sortino_ratios(
        cum_log_returns: pd.Series, risk_free_rate=0.0
    ) -> dict[str, float]:
        """
        Computes the Sharpe and Sortino ratios based on 0/1 strategy cumulative log-returns.
        :param cum_log_returns: pd.Series; cumulative log-returns.
        :param risk_free_rate: float; risk-free rate (annualized, as a fraction). Defaults to 0.
        :return: dictionary with daily Sharpe and Sortino ratios.
        """
        strategy_daily_returns = cum_log_returns.diff()
        excess_returns = strategy_daily_returns - (
            risk_free_rate / 252
        )  # todo: change risk-free to pd.Series
        # Sharpe ratio: mean of excess returns divided by standard deviation
        sharpe_ratio = excess_returns.mean() / excess_returns.std()
        # Sortino ratio: mean of excess returns divided by downside deviation
        downside_deviation = excess_returns[excess_returns < 0].std()
        sortino_ratio = (
            excess_returns.mean() / downside_deviation
            if downside_deviation > 0
            else np.nan
        )
        return {
            "Final Cum Return": cum_log_returns.iloc[-1],
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
        }

    def _update_dicts(
        self, jm: JumpModel, jp: float, ratio_dict: dict, strat_df: pd.Series
    ) -> None:
        self.best_model["jump_model"] = jm
        self.best_model["jump_penalty"] = jp
        self.best_model["sortino"] = ratio_dict["Sortino Ratio"]
        self.best_model["sharpe"] = ratio_dict["Sharpe Ratio"]
        self.best_model["fin_cum_ret"] = ratio_dict["Final Cum Return"]
        self.rets_dict["strat_df"] = strat_df

    def _get_jump_penalties(self, logspace: bool):
        """List of possible jump penalties obtained with linspace or logspace."""
        start, end = self._checks()
        if logspace:
            if end > 10:
                raise ValueError("The upper bound for the logspace should be <= 10.")
            return np.logspace(start, end, self.number_jump_penalties, base=np.exp(1))
        return np.linspace(start, end, self.number_jump_penalties)

    def find_best_jump_penalty(self) -> tuple[dict, dict]:
        """
        Find the best jump penalty for the given features and returns.
        The output is given by 2 dictionaries:
        - `self.best_model` contains information about the Jump Model and some key (scalar) statistics
        - `self.rets_dict` contains information in the form of dataframes
        """
        for jp in self.jump_penalties:
            jp = round(jp, 2)
            jm = JumpModel(
                n_components=self.number_clusters,
                jump_penalty=jp,
                cont=self.continuous_jm,
                mode_loss=self.mode_loss,
                verbose=self.verbose,
            )
            jm.fit(X=self.X_train_processed, ret_ser=self.returns, sort_by="cumret")

            if self.continuous_jm:
                labels_test_online = jm.predict_proba_online(self.X_test_processed)
                joined_df, strat_df = self.get_strategy_df(
                    labels_test_online, in_sample_labels=jm.proba_
                )
            else:
                labels_test_online = jm.predict_online(self.X_test_processed)
                joined_df, strat_df = self.get_strategy_df(
                    labels_test_online, in_sample_labels=jm.labels_
                )

            joined_df[joined_df.isna()] = 0  # Ok, since 0 is no-invest
            # NOTE that the regime at date `date` is the regime identified at the end of that date. In this case we
            # don't have any forecasting in place. We just have to classify the dates.

            perf_df = backtest_strategy_with_cash(
                df=joined_df,
                fee=self.trans_cost.prop_fees,
                logger=self.logger,
                cash_log_ret=0.0,
                initial_value=1.0,
                log_ret_col="ret",
                alloc_col="strat",
                jump_penalty=jp,
            )
            ratio_dict = self.compute_sharpe_sortino_ratios(perf_df["cum_returns"])
            self.rets_dict[f"jump_{jp}"] = perf_df["cum_returns"]
            if self.verbose > 0:
                self.logger.info(
                    f"Daily Sharpe and Sortino ratios for lambda={jp}: {ratio_dict}"
                )
            # sortino, sharpe = ratio_dict["Sortino Ratio"], ratio_dict["Sharpe Ratio"]
            # if (not np.isnan(sortino) and sortino > self.best_model["sortino"]) or (
            #     np.isnan(sortino) and sharpe > self.best_model["sharpe"]
            # ):
            # Use Sortino Ratio to decide when to update, unless it is NaN. In that case, use Sharpe Ratio.
            if perf_df["cum_returns"].iloc[-1] >= self.best_model["fin_cum_ret"]:
                self._update_dicts(jm, jp, ratio_dict, strat_df)
        return self.best_model, self.rets_dict



