import logging
import os
import sys
import warnings


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
