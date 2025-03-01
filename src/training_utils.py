import os

import numpy as np
import pandas as pd

from src.utils import BaseLogger


def load_dataframe_from_npz(
        src_path: str | os.PathLike,
        use_cols: list = None,
        callbacks: dict = None
):
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

    df_fetched = pd.DataFrame(
        data_fetched
    )

    return df_fetched