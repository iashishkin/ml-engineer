import os
import re

import numpy as np
import pandas as pd

from numbers import Real

from src.utils import BaseLogger, CallbackLogger


def process_missing_data(
        data: pd.DataFrame,
        method: str = "drop",
        subset: list[str] = None,
        raise_error: bool = False,
        callbacks: dict = None,
) -> pd.DataFrame | tuple[pd.DataFrame, int]:
    """
    Process missing (NaN) values in a DataFrame based on the specified method.

    Args:
        data (pd.DataFrame): Input DataFrame.
        method (str): Method to handle missing values: 'drop', or 'report'. Defaults to 'drop'.
        subset (list[str], optional): List of columns to check for missing values. Defaults to all columns.
        raise_error (bool, optional): If True, raises a ValueError if missing data is found.
        callbacks (dict, optional): Callback functions, including 'logging_callback'. Defaults to None.

    Returns:
        pd.DataFrame: Filtered DataFrame after processing missing values.
    """
    if callbacks is None:
        callbacks = {}

    logger = callbacks.get("logging_callback", BaseLogger())

    # Identify missing values
    missing_mask = data.isna() if subset is None else data[subset].isna()
    total_missing = missing_mask.sum().sum()

    # Raise error if missing data is found and raise_error is True
    if total_missing > 0 and raise_error:
        logger.critical(f"Missing data detected. Missing value counts per column:\n{missing_mask.sum()}.")

    logger.debug(f"Total missing values found: {total_missing}")

    # Handle missing values based on the method
    if total_missing > 0:
        if method == "drop":
            data.dropna(subset=subset, inplace=True)
            logger.debug(f"Dropped rows with missing values. New data shape: {data.shape}")

        elif method == "report":
            logger.warning(f"Missing value counts per column:\n{missing_mask.sum()}")

        else:
            logger.critical(f"Unsupported method: '{method}'. Use 'drop' or 'report'.")

    else:
        logger.info("No missing values found.")

    return data


def process_duplicates(
        data: pd.DataFrame,
        params: dict,
        callbacks: dict = None
):
    """
    Process and remove duplicate rows from a DataFrame based on specified parameters.

    Args:
        data (pd.DataFrame): Input DataFrame.
        params (dict): Parameters for the `DataFrame.duplicated` method.
        callbacks (dict, optional): Callback functions, including 'logging_callback'. Defaults to None.

    Returns:
        pd.DataFrame | tuple: Filtered DataFrame, optionally with duplicate count.
    """

    if callbacks is None:
        callbacks = dict()

    logger = callbacks.get("logging_callback", BaseLogger())

    if not isinstance(params, dict):
        logger.critical("`params` must be a dictionary of arguments for `DataFrame.duplicated`.")

    logger.warning(f"Processing duplicates with params: {params}")

    duplicated_idx = data.duplicated(**params)
    total_duplicates = duplicated_idx.sum()
    logger.debug(f"Total duplicates found: {total_duplicates}")

    if total_duplicates > 0:
        data = data.loc[~duplicated_idx]
        logger.debug(f"Dropped duplicates. New data shape: {data.shape}")
    else:
        logger.info("No duplicates found.")

    return data


def __upper_constant_filter(
        values: np.ndarray,
        threshold: Real
) -> np.ndarray:
    """
    Filter values less than or equal to the given threshold.

    Args:
        values (np.ndarray): Array of values to filter.
        threshold (Real): Upper threshold for filtering.

    Returns:
        np.ndarray: Boolean array where True indicates values less than or equal to the threshold.

    Examples:
        >>> values = np.array([10, 20, 30, 40, 50])
        >>> __upper_constant_filter(values, 30)
        array([ True,  True,  True, False, False])
    """
    return values <= threshold


def __lower_constant_filter(
        values: np.ndarray,
        threshold: Real
) -> np.ndarray:
    """
    Filter values greater than or equal to the given threshold.

    Args:
        values (np.ndarray): Array of values to filter.
        threshold (Real): Lower threshold for filtering.

    Returns:
        np.ndarray: Boolean array where True indicates values greater than or equal to the threshold.

    Examples:
        >>> values = np.array([10, 20, 30, 40, 50])
        >>> __lower_constant_filter(values, 30)
        array([False, False,  True,  True,  True])
    """
    return threshold <= values


def __range_filter(
        values: np.ndarray,
        min_threshold: Real,
        max_threshold: Real
) -> np.ndarray:
    """
    Filter values within a specified range (inclusive).

    Args:
        values (np.ndarray): Array of values to filter.
        min_threshold (Real): Minimum threshold (inclusive).
        max_threshold (Real): Maximum threshold (inclusive).

    Returns:
        np.ndarray: Boolean array where True indicates values within the range.

    Examples:
        >>> values = np.array([10, 20, 30, 40, 50])
        >>> __range_filter(values, 20, 40)
        array([False,  True,  True,  True, False])
    """
    return (min_threshold <= values) & (values <= max_threshold)


def __upper_quantile_filter(
        values: np.ndarray,
        q: float
) -> np.ndarray:
    """
    Filter values below a specified upper quantile.

    Args:
        values (np.ndarray): Array of values to filter.
        q (float): Quantile threshold (0 < q < 1).

    Returns:
        np.ndarray: Boolean array where True indicates values below the quantile threshold.

    Raises:
        AssertionError: If q is not between 0 and 1.

    Examples:
        >>> values = np.array([10, 20, 30, 40, 50])
        >>> __upper_quantile_filter(values, 0.8)
        array([ True,  True,  True,  True, False])
    """
    assert 0 < q < 1, "Quantile must be between 0 and 1"
    threshold = np.quantile(values, q)
    return values <= threshold


def __lower_quantile_filter(
        values: np.ndarray,
        q: float
) -> np.ndarray:
    """
    Filter values above a specified lower quantile.

    Args:
        values (np.ndarray): Array of values to filter.
        q (float): Quantile threshold (0 < q < 1).

    Returns:
        np.ndarray: Boolean array where True indicates values above the quantile threshold.

    Raises:
        AssertionError: If q is not between 0 and 1.

    Examples:
        >>> values = np.array([10, 20, 30, 40, 50])
        >>> __lower_quantile_filter(values, 0.2)
        array([False,  True,  True,  True,  True])
    """
    assert 0 < q < 1, "Quantile must be between 0 and 1"
    threshold = np.quantile(values, q)
    return values >= threshold


def __categorical_filter(
        values: np.ndarray,
        categories: list | int | str
) -> np.ndarray:
    """
    Filter values that match specified categories.

    Args:
        values (np.ndarray): Array of values to filter.
        categories (list | int | str): Categories to match.

    Returns:
        np.ndarray: Boolean array where True indicates values matching the specified categories.

    Examples:
        >>> values = np.array(['A', 'B', 'C', 'A', 'D'])
        >>> __categorical_filter(values, ['A', 'C'])
        array([ True, False,  True,  True, False])
    """
    return np.isin(values, categories)


filters_map = {
    "upper_constant": __upper_constant_filter,
    "lower_constant": __lower_constant_filter,
    "range": __range_filter,
    "upper_quantile": __upper_quantile_filter,
    "lower_quantile": __lower_quantile_filter,
    "category": __categorical_filter
}


def filter_data(
        data: pd.DataFrame,
        filters: list,
        callbacks: dict = None
) -> pd.DataFrame:
    """
    Filter a DataFrame based on specified filters.

    Args:
        data (pd.DataFrame): Input DataFrame to filter.
        filters (list): List of filter dictionaries with keys: "pattern", "type", and "params".
        callbacks (dict, optional): Dictionary of callback functions. Defaults to None.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """

    if callbacks is None:
        callbacks = dict()

    logger = callbacks.get("logging_callback", BaseLogger())
    if not isinstance(filters, list):
        logger.critical("`filters` must be a list of filter definitions.")

    masks = []

    for filter_item in filters:

        logger.debug(f"Processing filter - {filter_item}")

        filter_name_pattern = filter_item["pattern"]
        filter_type = filter_item["type"]
        filter_params = filter_item.get("params", {})

        # Find matching columns using regex pattern
        matching_columns = [col for col in data.columns if re.fullmatch(filter_name_pattern, col)]

        if not matching_columns:
            logger.debug(f"No columns match pattern '{filter_name_pattern}'. Skipping filter.")
            continue

        for col in matching_columns:

            logger.debug(f"Filtering col - {col} ...")

            # Get corresponding filter function
            filter_func = filters_map.get(filter_type)
            if not filter_func:
                logger.critical(f"Unsupported filter type: {filter_type}")

            # Apply filter and update mask
            indices = filter_func(data[col].values, **filter_params)
            logger.debug(f"filter {indices.mean(): .6%} of values")

            masks.append(indices)

        if masks:
            final_mask = np.logical_and.reduce(masks)
        else:
            final_mask = np.ones(len(data), dtype=bool)

    logger.warning(f"All filters will filter {final_mask.mean(): .6%} of values")

    return data.loc[final_mask]


def save_dataframe_to_npz(
        data: pd.DataFrame | pd.Series,
        dest_path: str,
        callbacks: dict = None
):
    """
    Save a pandas DataFrame to an .npz file efficiently with validity checks.

    Args:
        data (pd.DataFrame | pd.Series): The DataFrame or Series to save.
        dest_path (str): Path to the output .npz file.
        callbacks (dict, optional): Dictionary of callback functions. Defaults to None.

    Raises:
        ValueError: If invalid inputs are provided or conversion fails.
        IOError: If the file cannot be written.
    """

    if callbacks is None:
        callbacks = dict()

    logger = callbacks.get("logging_callback", BaseLogger())

    # Validity checks
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        logger.critical("Input `df` must be a pandas DataFrame or Series.")
    if not isinstance(dest_path, str) or not dest_path.endswith(".npz"):
        logger.critical("`file_path` must be a valid string ending with '.npz'.")
    if data.empty:
        logger.critical("DataFrame is empty. Nothing to save.")

    arrays = {}

    # Handle Series
    if isinstance(data, pd.Series):
        try:
            arrays[data.name ] = data.to_numpy()
        except Exception as e:
            logger.critical(f"Failed to process data: {e}")

    # Handle DataFrame
    if isinstance(data, pd.DataFrame):
        for col in data.columns:

            try:
                arrays[col] = data[col].values
            except Exception as e:
                logger.critical(f"Failed to process column '{col}': {e}")


    # Save all arrays into an NPZ archive
    try:
        np.savez_compressed(dest_path, **arrays)
        logger.debug(f"data successfully saved to '{dest_path}'.")
    except IOError as e:
        logger.critical(f"Failed to write NPZ file: {e}")


def preprocess_data(
        src_path: str | os.PathLike,
        dest_dir: str | os.PathLike,
        use_features: list[str],
        process_missed_data: dict,
        drop_duplicates: bool,
        filters: list[dict],
        target_col: str = None,
        callbacks: dict = None
):
    """
    Preprocess data from the specified source path and save processed outputs.

    Args:
        src_path (str): Path to the source CSV file.
        dest_dir (str): Directory to save processed outputs.
        use_features (list[str]): List of feature columns to use.
        process_missed_data (dict): Configuration for handling missing data.
        drop_duplicates (bool): Whether to drop duplicate rows.
        filters (list[dict]): List of filters to apply.
        target_col (str, optional): Name of the target column. Defaults to None.
        callbacks (dict, optional): Dictionary of callback functions. Defaults to None.
    """

    # TODO: add config saving

    if callbacks is None:
        callbacks = dict()

    # get logger
    logger = callbacks.get("logging_callback", BaseLogger())

    # Validate input paths and parameters
    if not os.path.exists(src_path):
        logger.critical(f"Data file not found: {src_path}")

    if not use_features or not isinstance(use_features, list):
        logger.critical("`use_features` must be a non-empty list.")


    # Load data
    logger.debug(f"Loading data from: {src_path}...")
    try:
        data = pd.read_csv(src_path)
        logger.debug(f"Data shape: {data.shape}")
    except Exception as e:
        logger.critical(f"Failed to load data: {e}")
        raise BrokenPipeError

    # enhance features
    logger.info("Enhancing data ...")
    data["pickup_date"] = pd.to_datetime(data["pickup_date"], format="%Y-%m-%d %H:%M:%S")
    data["pickup_year"] = data["pickup_date"].dt.year
    data["pickup_month"] = data["pickup_date"].dt.month
    data["pickup_day"] = data["pickup_date"].dt.day
    data["pickup_hour"] = data["pickup_date"].dt.hour
    data["pickup_weekday"] = data["pickup_date"].dt.weekday

    data["pickup_ts"] = data["pickup_date"].apply(lambda x: int(x.timestamp() * 1000))
    data["pickup_year_month"] = data["pickup_date"].dt.to_period("M")

    # Prepare columns to use
    use_columns = use_features.copy()
    y_data = None

    if target_col:
        if target_col in data.columns:
            use_columns.append(target_col)
        else:
            logger.critical(f"Target column '{target_col}' not found in dataset")

    # Filter features
    use_columns = use_features + [target_col] if target_col is not None else use_features
    logger.info(f"Filtering target and features: target='{target_col}', features={use_features}")
    data = data.loc[:, use_columns]
    logger.debug(f"Filtered columns. New shape: {data.shape}")

    # Handle missing data
    if process_missed_data:
        logger.info("Processing missed data...")
        data = process_missing_data(
            data=data,
            **process_missed_data,
            callbacks={"logging_callback": logger}
        )

    # Handle duplicates
    if drop_duplicates:
        logger.info("Processing duplicates...")
        drop_duplicates_params = {"subset": None, "keep": "first"} if not isinstance(drop_duplicates, dict) else {
            "subset": drop_duplicates.get("subset", None),
            "keep": drop_duplicates.get("keep", "first")
        }
        data = process_duplicates(
            data=data,
            params=drop_duplicates_params,
            callbacks={"logging_callback": logger}
        )

    # Apply filters
    if filters:
        logger.info("Applying filters...")
        data = filter_data(
            data=data,
            filters=filters,
            callbacks={"logging_callback": logger}
        )

    # Split features and target
    X_data = data.loc[:, use_features].copy()
    if target_col:
        y_data = data[target_col].copy()
    else:
        y_data = None

    # Save processed data
    logger.info("Saving processed data...")
    save_dataframe_to_npz(X_data, os.path.join(dest_dir, "features.npz"), callbacks={"logging_callback": logger})

    if y_data is not None:
        save_dataframe_to_npz(y_data, os.path.join(dest_dir, "target.npz"), callbacks={"logging_callback": logger})
    else:
        logger.warning("No target column found. Skipping target save.")

    logger.info(f"Data preprocessing completed. Processed files saved in '{dest_dir}'.")
