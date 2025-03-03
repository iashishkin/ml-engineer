import os
import pickle

import numpy as np
import lightgbm as lgb

from sklearn.preprocessing import OrdinalEncoder

from scripts.train import loss
from src.training_utils import load_dataframe_from_npz

if __name__ == "__main__":

    # args
    train_src_path = "../dataset/processed/train_enhanced/"
    val_src_path = "../dataset/processed/val_enhanced/"
    dest_path = "../models/log_model"

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
    X_train = load_dataframe_from_npz(
        os.path.join(train_src_path, "features.npz"),
        use_cols=use_cols
    )
    y_train = load_dataframe_from_npz(
        os.path.join(train_src_path, "target.npz"),
        use_cols=target_col
    ).values.ravel()

    X_val = load_dataframe_from_npz(
        os.path.join(val_src_path, "features.npz"),
        use_cols=use_cols
    )
    y_val = load_dataframe_from_npz(
        os.path.join(val_src_path, "target.npz"),
        use_cols=target_col
    ).values.ravel()
    
    # log data
    X_train[features_log] = np.log(X_train[features_log])
    X_val[features_log] = np.log(X_val[features_log])

    # encode categorical
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(X_train[features_oe])

    X_train[features_oe] = ordinal_encoder.transform(X_train[features_oe])
    X_val[features_oe] = ordinal_encoder.transform(X_val[features_oe])

    # create dataset
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_dataset = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train_dataset)

    # train model
    model = lgb.train(
        model_params,
        train_dataset,
        valid_sets=[ val_dataset, train_dataset],
        valid_names=["val", "train"],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(50),  # Stops training if no improvement for 50 rounds
            lgb.log_evaluation(1),  # Logs loss at each iteration
        ]
    )

    # get the best iteration loss
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    train_loss = loss(y_train, y_train_pred)
    val_loss = loss(y_val, y_val_pred)
    print(f"Train loss - {train_loss}")
    print(f"Val loss - {val_loss}")

    # save model and encoders
    with open(os.path.join(dest_path, "ordinal_encoder.pkl"), "wb") as ef:
        pickle.dump(ordinal_encoder, ef)

    with open(os.path.join(dest_path, "model.pkl"), "wb") as mf:
        pickle.dump(model, mf)