import argparse
import os

from src.data_preprocessing_utils import preprocess_data
from src.utils import load_config, get_callback_logger



def main(
        config_path: str | os.PathLike
):
    # Load config
    cfg = load_config(
        config_path=config_path,
        required_keys=["src_path", "dest_dir", "target_col", "use_features"]
    )

    # Get logging level from config (default to "info")
    logger_params = cfg.get("logging", {})
    cfg.pop("logging", None)

    # Initialize logger (log level conversion happens inside get_logger)
    logger = get_callback_logger(
        log_path=os.path.join(cfg["dest_dir"], "data_preprocessing.log"),
        **logger_params
    )


    # Run preprocessing
    preprocess_data(
        **cfg,
        callbacks={"logging_callback": logger}
    )


if __name__ == "__main__":

    # args
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    main(args.config)


