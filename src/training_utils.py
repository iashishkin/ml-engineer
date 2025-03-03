import os
import pickle
import re


import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.preprocessing import OrdinalEncoder
from sklearn.exceptions import NotFittedError

from src.utils import BaseLogger
from scripts.train import loss


def load_dataframe_from_npz(
        src_path: str | os.PathLike,
        use_cols: list = None,
        callbacks: dict = None
) -> pd.DataFrame:
    """
    Load a DataFrame from a compressed .npz file.

    This function reads a .npz file containing arrays (one per column) and converts them into a pandas DataFrame.
    Optionally, only a subset of columns specified by `use_cols` is loaded.

    Args:
        src_path (str | os.PathLike): The path to the .npz file.
        use_cols (list, optional): List of column names to load. If None, all columns are loaded.
        callbacks (dict, optional): Dictionary containing callback functions, including a 'logging_callback'.

    Returns:
        pd.DataFrame: The DataFrame constructed from the .npz file.

    Raises:
        BrokenPipeError: If loading the .npz file fails.
    """
    if callbacks is None:
        callbacks = dict()

    logger = callbacks.get("logging_callback", BaseLogger())

    # Validity checks
    if not isinstance(src_path, (str, os.PathLike)) or not src_path.endswith(".npz"):
        logger.critical("`src_path` must be a valid string ending with '.npz'.")

    try:
        data = np.load(src_path, allow_pickle=True)
    except Exception as e:
        logger.critical(f"Failed to load data: {e}")
        raise BrokenPipeError

    if use_cols is None:
        logger.warning("use_cols is None; all data columns will be loaded")
        use_cols = list(data.keys())

    logger.debug(f"Loading data from {src_path}")

    data_fetched = {}
    for col in use_cols:
        data_fetched[col] = data[col]

    df_fetched = pd.DataFrame(data_fetched)
    return df_fetched

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
        """Custom callback to store training and validation loss"""

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
