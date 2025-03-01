import logging
import os

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb

from copy import copy
from functools import partial
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, MinMaxScaler

from src.training_utils import load_dataframe_from_npz
from src.utils import BaseLogger, get_callback_logger

from scripts.train import loss


def train_model(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        cat_features: list,
        model_params: dict,
        callbacks: dict = None
):
    if callbacks is None:
        callbacks = dict()

    logger = callbacks.get("logging_callback", BaseLogger())

    # Create LightGBM dataset
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_dataset = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train_dataset)


    def lgb_log_callback(env):
        logger.debug(f"Train loss - {env.evaluation_result_list[0][2]: ^12.6e}; Val loss - {env.evaluation_result_list[1][2]: ^12.6e}")

    model = lgb.train(
        model_params,
        train_dataset,
        valid_sets=[train_dataset, val_dataset],
        valid_names=["train", "val"],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(50),  # Stops training if no improvement for 50 rounds
            lgb_log_callback
        ]
    )
    
    return model

def objective_with_args(
        trial,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        model_params: dict,
        callbacks: dict = None
):
    
    if callbacks is None:
        callbacks = dict()
        
    logger = callbacks.get("logging_callback", BaseLogger())
    logger.set_new_formatter(
        logging.Formatter(
            fmt=f"%(asctime)s [TRIAL {trial.number}] [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    cat_features = ["transport_type", "origin_kma", "destination_kma", "pickup_year", "pickup_month", "pickup_day", "pickup_hour", "pickup_weekday"]
    log_features = ["valid_miles"]

    features_to_select = ["origin_kma", "destination_kma", "pickup_year", "pickup_month", "pickup_day", "pickup_hour", "pickup_weekday"]
    features_to_use = ["weight", "valid_miles", "transport_type"]

    # X_train[cat_features] = X_train[cat_features].astype("category")
    # X_val[cat_features] = X_val[cat_features].astype("category")

    # select features subset
    selected_features = []
    for feature in features_to_select:

        select_feature = trial.suggest_categorical(f"use_{feature}", [True, False])

        if select_feature:
            selected_features.append(feature)

    # update features lists
    selected_cat_features = list(
        (set(selected_features) & set(cat_features)) |(set(features_to_use) & set(cat_features))
    )
    selected_log_features = list(
        (set(selected_features) & set(log_features)) | (set(features_to_use) & set(log_features))
    )
    
    optuna_features = selected_features + features_to_use
    logger.info(f"optuna_features - {optuna_features}")

    # drop useless features
    X_train = X_train[optuna_features]
    X_val = X_val[optuna_features]

    # encode selected categorical features
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(X_train[selected_cat_features])
    X_train.loc[:, selected_cat_features] = ordinal_encoder.transform(X_train[selected_cat_features])
    X_val.loc[:, selected_cat_features] = ordinal_encoder.transform(X_val[selected_cat_features])

    X_train[selected_cat_features] = X_train[selected_cat_features].astype("category")
    X_val[selected_cat_features] = X_val[selected_cat_features].astype("category")

    # log features
    for feature in selected_log_features:

        log_feature = trial.suggest_categorical(f"log_{feature}", [True, False])
        
        if log_feature:
            X_train.loc[:, feature] = np.log(X_train[feature])
            X_val.loc[:, feature] = np.log(X_val[feature])

    # log target
    y_train_og = copy(y_train)
    y_val_og = copy(y_val)

    use_log_target = trial.suggest_categorical("use_log_target", [True, False])

    if use_log_target:
        y_train = np.log(y_train)
        y_val = np.log(y_val)
    else:
        use_log_target = False

    # train model
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cat_features=selected_cat_features,
        model_params=model_params,
        callbacks=callbacks
    )
    
    
    # get predictions
    y_val_est = model.predict(X_val)
    
    # get loss
    if use_log_target:
        y_val_est = np.exp(y_val_est)

    val_loss = loss(y_val_og, y_val_est)

    return val_loss



if __name__ == "__main__":

    # args
    study_name = "feature_selection"
    dest_dir = "../optuna/"

    train_features_src = "../dataset/processed/train_enhanced/features.npz"
    train_features_cols = [
          "valid_miles",
          "transport_type",
          "weight",
          "pickup_date",
          "origin_kma",
          "destination_kma",
          "pickup_year",
          "pickup_month",
          "pickup_day",
          "pickup_hour",
          "pickup_weekday"
    ]

    train_target_src = "../dataset/processed/train_enhanced/target.npz"
    train_target_cols = ['rate']

    val_features_src = "../dataset/processed/val_enhanced/features.npz"
    val_features_cols = [
        "valid_miles",
        "transport_type",
        "weight",
        "pickup_date",
        "origin_kma",
        "destination_kma",
        "pickup_year",
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "pickup_weekday"
    ]

    val_target_src = "../dataset/processed/val_enhanced/target.npz"
    val_target_cols = ['rate']

    train_params = dict(
        objective="mae",
        metric="mae",
        boosting_type="gbdt",
        learning_rate=0.2,
        num_leaves=512,
        n_estimators=300,
        max_depth=-1,
        verbosity=-1,
        seed=1234,
        num_threads=4
    )

    logger = get_callback_logger(
        log_path=os.path.join(dest_dir, study_name, "optuna.log"),
        level="debug",
        mode="a"
    )

    # load data
    train_features = load_dataframe_from_npz(
        src_path=train_features_src,
        use_cols=train_features_cols,
        callbacks={"logging_callback": logger}
    )
    train_target = load_dataframe_from_npz(
        src_path=train_target_src,
        use_cols=train_target_cols,
        callbacks={"logging_callback": logger}
    )

    val_features = load_dataframe_from_npz(
        src_path=val_features_src,
        use_cols=val_features_cols,
        callbacks={"logging_callback": logger}
    )
    val_target = load_dataframe_from_npz(
        src_path=val_target_src,
        use_cols=val_target_cols,
        callbacks={"logging_callback": logger}
    )

    # set optuna study
    sampler = optuna.samplers.RandomSampler(seed=1234)

    study = optuna.create_study(
        storage="sqlite:///" + str(os.path.join(dest_dir, study_name, f"trials.db")),
        sampler=sampler,
        study_name=study_name,
        direction="minimize",
        load_if_exists=True
    )
    study.set_user_attr("train_features_src", train_features_src)
    study.set_user_attr("train_features_cols", train_features_cols)
    study.set_user_attr("train_target_src", train_target_src)
    study.set_user_attr("train_target_cols", train_target_cols)
    study.set_user_attr("val_features_src", val_features_src)
    study.set_user_attr("val_features_cols", val_features_cols)
    study.set_user_attr("val_target_src", val_target_src)
    study.set_user_attr("val_target_cols", val_target_cols)
    study.set_user_attr("train_params", train_params)
    study.set_user_attr("sampler", "RandomSampler")

    objective = partial(
        objective_with_args,
        X_train=train_features.copy(),
        y_train=train_target.copy().values.ravel(),
        X_val=val_features.copy(),
        y_val=val_target.copy().values.ravel(),
        model_params=train_params,
        callbacks={"logging_callback": logger}
    )

    study.optimize(
        func=objective,
        n_trials=5000,
        n_jobs=1,
        gc_after_trial=True
    )
    logger.warning(f"Best value: {study.best_value} (params: {study.best_params})")