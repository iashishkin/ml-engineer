import os
import pickle
import warnings

import numpy as np
import lightgbm as lgb


from scripts.train import loss
from src.training_utils import load_dataframe_from_npz

if __name__ == "__main__":

    # args
    model_src_path = "../models/log_model/"
    data_src_path = "../dataset/processed/test_enhanced/"
    dest_path = "../dataset/inference/test"

    os.makedirs(dest_path)

    use_cols = [
        'valid_miles',
        'transport_type',
        'weight',
        'origin_kma',
        'destination_kma',
        'pickup_year',
        'pickup_month',
        'pickup_weekday'
    ]
    target_col = ["rate"]

    cat_features = ["transport_type", "origin_kma", "destination_kma", "pickup_year", "pickup_weekday"]

    features_oe = ["transport_type", "origin_kma", "destination_kma", "pickup_year", "pickup_weekday"]
    features_log = ["valid_miles"]

    model_params = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.2,
        "num_leaves": 512,
        "n_estimators": 300,
        "max_depth": -1,
        "verbosity": 1,
        "seed": 1234
    }

    # load data
    X_values = load_dataframe_from_npz(
        os.path.join(data_src_path, "features.npz"),
        use_cols=use_cols
    )
    try:
        y_values = load_dataframe_from_npz(
            os.path.join(data_src_path, "target.npz"),
            use_cols=target_col
        ).values.ravel()

    except:
        warnings.warn("no target data")
        y_values = None

    # load encoder
    with open(os.path.join(model_src_path, "ordinal_encoder.pkl"), "rb") as ef:
        ordinal_encoder = pickle.load(ef)

    X_values[features_oe] = ordinal_encoder.transform(X_values[features_oe])

    # log data
    X_values[features_log] = np.log(X_values[features_log])


    # load model
    with open(os.path.join(model_src_path, "model.pkl"), "rb") as mf:
        model = pickle.load(mf)


    y_est = model.predict(X_values, num_iteration=model.best_iteration)

    if y_values is not None:
        est_loss = loss(y_values, y_est)
        print(f"loss - {est_loss}")

    # save estimation
    np.savez_compressed(
        os.path.join(dest_path, "estimation.npz"),
        rate=y_est
    )