import logging
import os
import sys
import yaml
import warnings


import numpy as np
import pandas as pd


class BaseLogger:
    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    time_format = "%Y-%m-%d %H:%M:%S"
    logformat = "%(asctime)s [%(levelname)s]: %(message)s"

    def __init__(self):
        self.formatter = logging.Formatter(fmt=self.logformat, datefmt=self.time_format)
        self.default_formatter = logging.Formatter(fmt=self.logformat, datefmt=self.time_format)

    def _format_message(self, level, msg):
        record = logging.LogRecord(name="EmptyLogger", level=self.logging_levels.get(level.lower(), logging.NOTSET), pathname="", lineno=0, msg=msg, args=(), exc_info=None)
        return self.formatter.format(record)

    def debug(self, msg, *args, **kwargs):
        print(self._format_message("debug", msg))

    def info(self, msg, *args, **kwargs):
        print(self._format_message("info", msg))

    def warning(self, msg, *args, **kwargs):
        warnings.warn(self._format_message("warning", msg))

    def error(self, msg, *args, **kwargs):
        warnings.warn(self._format_message("error", msg))

    def critical(self, msg, *args, **kwargs):
        raise RuntimeError(self._format_message("critical", msg))

    def exception(self, msg, *args, **kwargs):
        warnings.warn(self._format_message("exception", msg))

    def log(self, level, msg, **kwargs):
        if level.lower() in self.logging_levels:
            print(self._format_message(level, msg))
        else:
            print(self._format_message("unknown", msg))

    def set_new_formatter(self, formatter: logging.Formatter):
        self.formatter = formatter

    def set_default_formatter(self):
        self.formatter = self.default_formatter

    def add_handler(self, handler: logging.Handler, formatter: logging.Formatter = None):
        pass

    def remove_file_handler(self, filepath: os.PathLike | str):
        pass


class CallbackLogger(BaseLogger):

    def __init__(self, loggerName=None, level=logging.WARNING):
        super(CallbackLogger, self).__init__()
        self.logger = logging.getLogger(loggerName or f"CallbackLogger_{id(self)}")
        self.logger.setLevel(level)
        self._remove_existing_handlers()
        self._set_stream_handler()

        self.formatter = logging.Formatter(fmt=self.logformat, datefmt=self.time_format)

    def _remove_existing_handlers(self):
        while self.logger.hasHandlers():
            self.logger.removeHandler(self.logger.handlers[0])

    def _set_stream_handler(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(stream_handler)

    def add_file_handler(self, filepath: os.PathLike | str, mode: str = "a", encoding: str = "utf-8"):
        file_handler = logging.FileHandler(filepath, mode=mode, encoding=encoding)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def remove_file_handler(self, filepath: os.PathLike | str):
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.fspath(filepath):
                self.logger.removeHandler(handler)
                handler.close()
                break

    def remove_all_handlers(self):
        self._remove_existing_handlers()

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
        raise RuntimeError(self._format_message("critical", msg))

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)

    def log(self, level, msg, **kwargs):
        if level.lower() in self.logging_levels:
            self.logger.log(self.logging_levels[level.lower()], msg, **kwargs)
        else:
            self.logger.log(logging.NOTSET, msg, **kwargs)

    def set_new_formatter(self, formatter: logging.Formatter):
        """Apply a new formatter to all existing handlers."""
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    def set_default_formatter(self):
        """Reset all handlers to use the default formatter."""
        for handler in self.logger.handlers:
            handler.setFormatter(self.default_formatter)


def get_log_level(
        level: str | int
) -> int:
    """
    Convert log level from string or integer code.

    Args:
        level (str | int): Log level as string or integer code.

    Returns:
        int: Corresponding logging level.

    Raises:
        ValueError: If the log level is invalid.
    """
    if isinstance(level, int):
        if level in logging._levelToName:
            return level
        raise ValueError(f"Invalid integer log level: {level}")

    if isinstance(level, str):
        level = level.lower()
        if level not in BaseLogger.logging_levels:
            raise ValueError(f"Invalid string log level: '{level}'. Choose from: {list(BaseLogger.logging_levels.keys())}")
        return BaseLogger.logging_levels[level]


    raise TypeError(f"Unsupported log level type: {type(level)}")


def get_callback_logger(
        log_path: str | os.PathLike,
        level: str | int = "info",
        mode: str = "a",
        encoding: str = "utf-8"
) -> CallbackLogger:
    """
    Initialize a CallbackLogger and add a file handler.

    Args:
        log_path (str | os.PathLike): Path to the log file. Must end with '.log'.
        level (str | int, optional): Logging level as a string (e.g., 'info', 'debug')
            or an integer code (e.g., 10). Defaults to 'info'.
        mode (str, optional): File open mode for the log file. Defaults to 'a' (append mode).
        encoding (str, optional): Encoding used for the log file. Defaults to 'utf-8'.

    Returns:
        CallbackLogger: Configured logger instance with the specified log level and file handler.

    Raises:
        ValueError: If `log_path` is not a valid file path ending with '.log'.
        TypeError: If `log_level` is not a valid string or integer.
    """
    # Validate log_path
    if not isinstance(log_path, str) or not log_path.endswith(".log"):
        raise ValueError("`log_path` must be a valid string ending with '.log'.")

    # Ensure the directory exists
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Convert log level to integer
    log_level_int = get_log_level(level)

    # Initialize logger
    logger = CallbackLogger(level=log_level_int, )
    logger.add_file_handler(log_path, mode=mode, encoding=encoding)
    return logger


def load_config(config_path: str, required_keys: list[str] = None) -> dict:
    """
    Load YAML configuration file and validate required keys.

    Args:
        config_path (str): Path to the YAML config file.
        required_keys (list[str], optional): List of required keys to validate. Defaults to None.

    Returns:
        dict: Loaded configuration.

    Raises:
        ValueError: If required keys are missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if required_keys:
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

    return config


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
