
from pathlib import Path

import numpy as np
import pandas as pd

from jumpmodels.our_jm.penalty_finder import JumpPenaltyFinder
from jumpmodels.our_jm.utils_features import feature_engineer
from df_utils.df_utils import exclude_days_with_nan, remove_tz_from_index
from jumpmodels.jumpmodels.jump import JumpModel
from jumpmodels.jumpmodels.preprocess import StandardScalerPD, DataClipperStd
from jumpmodels.jumpmodels.utils import filter_date_range

pd.options.display.max_columns = None


symbol = "aapl"
# path_to_file = Path("data") / f"{symbol}_5min.pq"
path_to_file = Path("data") / f"{symbol}_daily.pq"
df = pd.read_parquet(path_to_file)


ret = np.log(df["close"]).diff().dropna()
features = feature_engineer(ret, ver="v0", half_lives=[10, 200])
features = exclude_days_with_nan(features)


perc_train = 0.7
train_start = features.index[0].date()
test_start_numerical = int(perc_train * len(features))
test_start = features.iloc[test_start_numerical].name.date()

features = remove_tz_from_index(features)
ret = remove_tz_from_index(ret)

X_train = filter_date_range(features, start_date=train_start, end_date=test_start)
X_test = filter_date_range(features, start_date=test_start)

y_train = filter_date_range(ret, start_date=train_start, end_date=test_start)
y_test = filter_date_range(ret, start_date=test_start)


# Preprocess train and test data
clipper = DataClipperStd(mul=4.)
scalar = StandardScalerPD()
X_train_proc = scalar.fit_transform(clipper.fit_transform(X_train))
X_test_proc = scalar.transform(clipper.transform(X_test))


# set the jump penalty
jump_penalty=50.
jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False)
jm.fit(X_train_proc, y_train, sort_by="cumret")

jm.centers_


nb_jump_penalties = 6

jp_finder = JumpPenaltyFinder(
    X_train_processed=X_train_proc,
    X_test_processed=X_test_proc,
    returns=y_train,
    trans_cost_class=self.transaction_fees,
    number_jump_penalties=nb_jump_penalties,
    jump_penalty_range=self.jp_range,
    validation_only=self.jp_validation_only,
    continuous_jm=False,
    mode_loss=self.mode_loss,
    logger=logger,
    verbose=0,  # set to 0 for no output
)
best_model_dict, rets_dict = jp_finder.find_best_jump_penalty()
logger.info(f"FIT_MODELS: col {col}, best_model_dict: {best_model_dict}")
logger.info(
    f"FIT_MODELS: Nb of clusters: {best_model_dict['strat_df'].nunique()}"
)


