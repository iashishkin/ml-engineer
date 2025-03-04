import os
import pickle
import re


import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.preprocessing import OrdinalEncoder
from sklearn.exceptions import NotFittedError

from src.utils import BaseLogger, load_dataframe_from_npz, save_dataframe_to_npz
from scripts.train import loss

# Global registry for transforms
# The keys are transform names used in configuration, and the values are functions that create a new instance.
TRANSFORM_REGISTRY = {
    "OrdinalEncoder": OrdinalEncoder
}

def register_transform(transform_type):
    """
    Decorator to register a transformer in the global TRANSFORM_REGISTRY.

    This decorator adds the decorated transform class or function to the TRANSFORM_REGISTRY
    under the provided transform_type key, so that it can be referenced by name in configuration.

    Args:
        transform_type (str): The name under which the transform will be registered.

    Returns:
        function: A decorator function that registers the transformer.
    """
    def decorator(func):
        TRANSFORM_REGISTRY[transform_type] = func
        return func
    return decorator

@register_transform("DummyTransform")
class DummyTransform:
    """
    A dummy transformer that performs no changes on the data.

    This transformer is provided as an example and simply returns the input data unchanged.
    """
    def fit(self, X, y=None):
        """
        Fit method (does nothing).

        Args:
            X (pd.DataFrame): Input data.
            y (optional): Target values.

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transform method (returns input unchanged).

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: The unchanged input data.
        """
        return X


@register_transform("LogTransform")
class LogTransform(DummyTransform):
    """
    A transformer that applies a logarithmic transformation to the input data.

    Inherits from DummyTransformer and overrides the transform method to apply the natural logarithm.
    """
    def fit(self, X, y=None):
        """
        Fit method (does nothing).

        Args:
            X (pd.DataFrame): Input data.
            y (optional): Target values.

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Apply a logarithmic transformation to the input data.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: The data with a logarithmic transformation applied.
        """
        return np.log(X + 1e-8)


@register_transform("CategoryDTypeTransform")
class CategoryDTypeTransform(DummyTransform):
    """
    A transformer that converts the input data to a categorical data type.

    Inherits from DummyTransform and overrides the transform method to convert the input data's data type to "category".
    """

    def fit(self, X, y=None):
        """
        Fit method (does nothing).

        Args:
            X (pd.DataFrame): Input data.
            y (optional): Target values.

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Convert the input data to a categorical data type.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: The data with its dtype converted to "category".
        """
        return X.astype("category")


class ConfigurableColumnTransformer:
    """
    A configurable transformer that applies a sequence of transformations to DataFrame columns.

    This class finds columns that match a given regex pattern and applies a list of transformations (from a global registry)
    in sequence. The configuration is immutable after fitting.
    """

    def __init__(self, pattern: str, transform_names: list):
        """
        Initialize the ConfigurableColumnTransformer.

        Args:
            pattern (str): A regex pattern to match column names.
            transform_names (list): A list of transform names to apply (must exist in the global TRANSFORM_REGISTRY).

        Raises:
            ValueError: If any transform name is not registered.
        """
        self.__pattern = pattern
        self.__transform_names = transform_names
        self.__transformers = {}
        self.__matching_cols = []
        self.__is_fitted = False

        # Early validation: ensure each transform name is in the global registry
        for t_name in self.transform_names:
            if t_name not in TRANSFORM_REGISTRY:
                raise ValueError(f"Transform '{t_name}' is not registered in TRANSFORM_REGISTRY.")

    @property
    def pattern(self) -> str:
        """
        Get the regex pattern used to match column names.

        Returns:
            str: The regex pattern.
        """
        return self.__pattern

    @property
    def transform_names(self) -> list:
        """
        Get the list of transformation names to apply.

        Returns:
            list: The transformation names.
        """
        return self.__transform_names

    @property
    def matching_cols(self) -> list:
        """
        Get the list of columns that matched the regex pattern during fitting.

        Returns:
            list: The list of matching column names.
        """
        return self.__matching_cols

    @property
    def transformers(self):
        """
        Get the list of transformers instances.

        Returns:
            list: The list of transformers instances.
        """

        return self.__transformers

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer on the given DataFrame.

        This method identifies the columns matching the regex pattern and fits each configured transformation
        on those columns.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (optional): Target values.

        Returns:
            self

        Raises:
            ValueError: If no columns match the regex pattern.
        """
        # Identify columns matching the regex pattern
        matching_cols = [col for col in X.columns if re.fullmatch(self.pattern, col)]
        if not matching_cols:
            raise ValueError(
                f"No columns matched the pattern '{self.pattern}' in the provided DataFrame with columns {list(X.columns)}."
            )
        self.__matching_cols = matching_cols  # Store matching columns for later use

        for col in matching_cols:
            transforms_for_col = []
            for t_name in self.transform_names:
                # Use the global registry to get the transformer
                transformer = TRANSFORM_REGISTRY[t_name]()
                transformer.fit(X[[col]], y)
                transforms_for_col.append(transformer)
            self.__transformers[col] = transforms_for_col

        self.__is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by applying the fitted transformations.

        This method applies each fitted transformation sequentially to the columns that were matched during fitting.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame.

        Raises:
            NotFittedError: If the transformer has not been fitted.
        """
        # Pre-fit check: ensure fit() has been called
        if not self.__is_fitted:
            raise NotFittedError(
                "This ConfigurableColumnTransformer instance is not fitted yet. "
                "Call 'fit' with appropriate data before using 'transform'."
            )

        # Check that all columns matched during fit still exist in the input DataFrame.
        missing_cols = [col for col in self.__matching_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(
                f"The following columns, which were present during fitting, are missing in the input DataFrame: {missing_cols}"
            )

        X_transformed = X.copy()
        for col, transformers in self.__transformers.items():
            for transformer in transformers:
                # Sequentially apply each transformer
                X_transformed[col] = transformer.transform(X_transformed[[col]])
        return X_transformed


def build_transform_pipeline(
        transforms_config: list,
        X: pd.DataFrame,
        callbacks: dict = None
) -> list:
    """
    Build a transformation pipeline from the specified configuration.

    This function creates a list of fitted ConfigurableColumnTransformer instances by iterating
    over the provided transformation configuration. Each transformer is fitted on the given DataFrame X.

    Args:
        transforms_config (list): A list of dictionaries, each specifying a transformation rule.
            Each dictionary must contain the keys:
                - "pattern": A regex pattern to match column names.
                - "transforms": A list of transformation names to apply.
        X (pd.DataFrame): The DataFrame on which to fit the transformers.
        callbacks (dict, optional): Dictionary containing callback functions, including a 'logging_callback'.

    Returns:
        list: A list of fitted ConfigurableColumnTransformer instances.

    Raises:
        ValueError: If 'transforms_config' is not a list, if any specification is not a dictionary,
            or if a required key ("pattern" or "transforms") is missing.
    """

    if callbacks is None:
        callbacks = dict()

    logger = callbacks.get("logging_callback", BaseLogger())

    if not isinstance(transforms_config, list):
        logger.critical("transforms_config must be a list of transformation specifications.")

    if not isinstance(X, pd.DataFrame):
        logger.critical("X must be a pandas DataFrame.")

    pipeline = []
    for spec in transforms_config:
        if not isinstance(spec, dict):
            logger.critical("Each transform specification must be a dictionary.")
        if "pattern" not in spec or "transforms" not in spec:
            logger.critical("Each transform specification must contain 'pattern' and 'transforms' keys.")

        pattern = spec["pattern"]
        t_names = spec["transforms"]

        logger.debug(f"add {t_names} transforms with pattern {pattern} to the pipeline")

        # Create an instance of the transformer and fit it on the data.
        transformer = ConfigurableColumnTransformer(pattern, t_names)
        transformer.fit(X)
        pipeline.append(transformer)

    return pipeline


def apply_pipeline(
        pipeline: list,
        X: pd.DataFrame,
        callbacks: dict = None
) -> pd.DataFrame:
    """
    Apply a transformation pipeline to the given DataFrame.

    This function sequentially applies each fitted transformer in the pipeline to the input DataFrame X.

    Args:
        pipeline (list): A list of fitted ConfigurableColumnTransformer instances.
        X (pd.DataFrame): The input DataFrame to transform.
        callbacks (dict, optional): Dictionary containing callback functions, including a 'logging_callback'.

    Returns:
        pd.DataFrame: The transformed DataFrame.

    Raises:
        ValueError: If 'pipeline' is not a list or if X is not a pandas DataFrame.
    """

    if callbacks is None:
        callbacks = dict()

    logger = callbacks.get("logging_callback", BaseLogger())

    if not isinstance(pipeline, list):
        logger.critical("pipeline must be a list of ConfigurableColumnTransformer instances.")
    if not isinstance(X, pd.DataFrame):
        logger.critical("X must be a pandas DataFrame.")

    X_transformed = X.copy()
    for transformer in pipeline:
        if not hasattr(transformer, "transform"):
            logger.critical("All elements in pipeline must have a 'transform' method.")

        X_transformed = transformer.transform(X_transformed)
        logger.debug(f"Applied {transformer.transform_names} to columns {transformer.matching_cols}")

    return X_transformed


def train_lgb_model(
        train_src_path: os.PathLike | str,
        dest_dir: os.PathLike | str,

        target_col: str,
        use_features: list,
        categorical_features: list,

        model_params: dict,
        feature_transforms: list = None,

        val_src_path: os.PathLike | str = None,
        callbacks: dict = None
):
    """
        Train a LightGBM model using preprocessed training data and an optional feature transformation pipeline.

        This function performs the following steps:
          1. Validates input paths and parameters.
          2. Loads training features and target data from NPZ files.
          3. Builds and applies a feature transformation pipeline. If no specific transformations
             are provided, a default pipeline using a DummyTransform is used.
          4. Creates a LightGBM dataset from the transformed features.
          5. Trains a LightGBM model:
               - If validation data is not provided, training is performed without early stopping.
               - If validation data is provided, early stopping is enabled and validation metrics are logged.
          6. Computes and logs training (and optionally validation) loss.
          7. Saves training losses, the feature transformation pipeline, and the trained model to disk.

        Args:
            train_src_path (os.PathLike | str): Path to the directory containing training NPZ files
                ("features.npz" and "target.npz").
            dest_dir (os.PathLike | str): Destination directory where the trained model, transformation pipeline,
                and training losses will be saved.
            target_col (str): Name of the target column.
            use_features (list): List of feature column names to use for training.
            categorical_features (list): List of categorical feature column names.
            model_params (dict): Dictionary of parameters for LightGBM model training.
            feature_transforms (list, optional): List of transformation configurations for building a feature pipeline.
                If None, a default pipeline with DummyTransform is used.
            val_src_path (os.PathLike | str, optional): Path to the directory containing validation NPZ files.
                If not provided, validation is disabled.
            callbacks (dict, optional): Dictionary of callback functions (e.g., logging callbacks). Defaults to None.

        Returns:
            None

        Side Effects:
            - Trains a LightGBM model.
            - Saves training losses to "training_losses.npz" in dest_dir.
            - Saves the feature transformation pipeline to "feature_transforms_pipeline.pkl" in dest_dir.
            - Saves the trained model to "model.pkl" in dest_dir.
            - Logs training and validation losses using the provided logging callback.

        Raises:
            Critical errors are logged via the provided logging callback if:
                - The training source path does not exist.
                - `use_features` is not a non-empty list.
                - Required data files cannot be loaded.
        """


    if callbacks is None:
        callbacks = dict()

    logger = callbacks.get("logging_callback", BaseLogger())

    # Validate input paths and parameters
    if not os.path.exists(train_src_path):
        logger.critical(f"Data file not found: {train_src_path}")

    if not use_features or not isinstance(use_features, list):
        logger.critical("`use_features` must be a non-empty list.")

    # load data
    X_train = load_dataframe_from_npz(
        os.path.join(train_src_path, "features.npz"),
        use_cols=use_features,
        callbacks=callbacks
    )
    y_train = load_dataframe_from_npz(
        os.path.join(train_src_path, "target.npz"),
        use_cols=[target_col],
        callbacks=callbacks
    ).values.ravel()
    

    if feature_transforms is None:

        transforms_pipeline = build_transform_pipeline(
            transforms_config=[
                dict(
                    pattern="^(" + "|".join(list(X_train.columns)) + ")$",
                    transforms=["DummyTransform"]
                )
            ],
            X=X_train,
            callbacks=callbacks
        )
    else:
        transforms_pipeline = build_transform_pipeline(
            transforms_config=feature_transforms,
            X=X_train,
            callbacks=callbacks
        )

    # apply transforms
    X_train_transformed = apply_pipeline(
        pipeline=transforms_pipeline,
        X=X_train,
        callbacks=callbacks
    )

    # create dataset
    train_dataset = lgb.Dataset(
        X_train_transformed,
        label=y_train,
        categorical_feature=categorical_features
    )

    training_losses = {"train": [], "val": []}
    def log_callback(env):
        """
        Custom callback function to log training and validation loss.

        This callback is used during LightGBM training to capture loss metrics for both training
        and validation sets, and log them at each iteration.

        Args:
            env: A LightGBM callback environment containing training state information.
        """

        msg = f"Iteration {env.iteration}; "

        for item in env.evaluation_result_list:
            training_losses[item[0]].append(item[2])

            msg += f"{item[0]} {item[1]} loss - {item[2]: ^12.6e}; "

        logger.debug(msg)


    if val_src_path is None:

        logger.warning("valid data is not specified; early stop disabled")

        # train model
        model = lgb.train(
            model_params,
            train_dataset,
            valid_sets=[train_dataset],
            valid_names=["train"],
            callbacks=[
                log_callback
            ]
        )

        y_train_est = model.predict(X_train_transformed, num_iteration=model.best_iteration)

        train_loss = loss(y_train, y_train_est)
        logger.info(f"Train loss - {train_loss}")

    else:

        logger.warning("valid data is specified; early stop enabled")

        X_val = load_dataframe_from_npz(
            os.path.join(val_src_path, "features.npz"),
            use_cols=use_features,
            callbacks=callbacks
        )
        y_val = load_dataframe_from_npz(
            os.path.join(val_src_path, "target.npz"),
            use_cols=[target_col],
            callbacks=callbacks
        ).values.ravel()
        

        X_val_transformed = apply_pipeline(
            pipeline=transforms_pipeline,
            X=X_val,
            callbacks=callbacks
        )

        val_dataset = lgb.Dataset(
            X_val_transformed,
            label=y_val,
            categorical_feature=categorical_features,
            reference=train_dataset
        )

        # train model
        model = lgb.train(
            model_params,
            train_dataset,
            valid_sets=[val_dataset, train_dataset],
            valid_names=["val", "train"],
            num_boost_round=1000,
            callbacks=[
                log_callback,
                lgb.early_stopping(25, first_metric_only=True)
            ]
        )

        y_train_est = model.predict(X_train_transformed, num_iteration=model.best_iteration)
        y_val_est = model.predict(X_val_transformed, num_iteration=model.best_iteration)

        train_loss = loss(y_train, y_train_est)
        logger.info(f"Train loss - {train_loss}")

        val_loss = loss(y_val, y_val_est)
        logger.info(f"Val loss - {val_loss}")

    # save model and encoders
    np.savez_compressed(
        os.path.join(dest_dir, "training_losses.npz"),
        **training_losses
    )

    with open(os.path.join(dest_dir, "feature_transforms_pipeline.pkl"), "wb") as pf:
        pickle.dump(transforms_pipeline, pf)

    with open(os.path.join(dest_dir, "model.pkl"), "wb") as mf:
        pickle.dump(model, mf)


def inference_lgb_model(
        data_src_path: os.PathLike | str,
        model_src_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        use_features: list,
        target_col: str,
        load_target: bool = False,
        callbacks: dict = None
):
    """
    Perform inference using a trained LightGBM model and a pre-saved feature transformation pipeline.

    This function carries out the following steps:
      1. Loads input features (and optionally target values) from NPZ files.
      2. Loads the pre-saved feature transformation pipeline and applies it to the input features.
      3. Loads the trained LightGBM model.
      4. Uses the model to predict target values from the transformed features.
      5. If target values are provided, computes and logs the prediction loss.
      6. Saves the predicted target values to disk.

    Args:
        data_src_path (os.PathLike | str): Path to the directory containing input NPZ files
            ("features.npz" and optionally "target.npz").
        model_src_path (os.PathLike | str): Path to the directory containing the saved model and feature transformation pipeline.
        dest_dir (os.PathLike | str): Destination directory where the inference output will be saved.
        use_features (list): List of feature column names to use for inference.
        target_col (str): Name of the target column.
        load_target (bool, optional): If True, load target values from the data source to compute loss. Defaults to False.
        callbacks (dict, optional): Dictionary of callback functions (e.g., logging callbacks). Defaults to None.

    Returns:
        None

    Side Effects:
        - Generates predictions for the input data using the trained model.
        - Saves the predicted target values to "target_estimated.npz" in dest_dir.
        - Logs the computed loss if target data is provided.

    Raises:
        Critical errors are logged via the provided logging callback if:
            - The feature transformation pipeline or model cannot be loaded.
            - The target file is missing when load_target is True.
    """

    if callbacks is None:
        callbacks = dict()

    logger = callbacks.get("logging_callback", BaseLogger())

    # load data
    X_values = load_dataframe_from_npz(
        os.path.join(data_src_path, "features.npz"),
        use_cols=use_features,
        callbacks=callbacks
    )

    if load_target:
        try:
            y_values = load_dataframe_from_npz(
                os.path.join(data_src_path, "target.npz"),
                use_cols=[target_col]
            ).values.ravel()

        except FileNotFoundError as e:
            logger.critical(f"No target data. {e}")

    # load feature transforms pipeline
    try:
        with open(os.path.join(model_src_path, "feature_transforms_pipeline.pkl"), "rb") as ef:
            transforms_pipeline = pickle.load(ef)

        logger.info(f"Load feature_transforms_pipeline from {model_src_path}.")

    except FileNotFoundError as e:
        logger.critical(f"Failed to load feature_transforms_pipeline from {model_src_path}. {e}")


    # apply transforms
    X_values_transformed = apply_pipeline(
        pipeline=transforms_pipeline,
        X=X_values,
        callbacks=callbacks
    )

    # load model
    try:
        with open(os.path.join(model_src_path, "model.pkl"), "rb") as mf:
            model = pickle.load(mf)

        logger.info(f"Load model from {model_src_path}.")

    except FileNotFoundError as e:
        logger.critical(f"Failed to load fmodel from {model_src_path}. {e}")


    # get estimation
    y_est = model.predict(X_values_transformed, num_iteration=model.best_iteration)

    if load_target:
        est_loss = loss(y_values, y_est)
        logger.info(f"loss - {est_loss}")
    else:
        logger.info("target data is not provided")

    # save estimation
    save_dataframe_to_npz(
        data=pd.DataFrame({target_col: y_est}),
        dest_path=os.path.join(dest_dir, "target_estimated.npz")
    )
