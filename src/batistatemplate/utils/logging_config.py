import logging
import logging.config
import os
from pathlib import Path

import tomli


def setup_logging() -> None:
    """Configures logging for the application.

    Reads the logging configuration from the 'pyproject.toml' file
    and applies it using `logging.config.dictConfig`.

    The log level can be overridden by setting the
    'BATISTATEMPLATE_LOG_LEVEL' environment variable. If the environment
    variable is not set or set to an invalid level, it defaults to 'INFO'.

    Raises:
        FileNotFoundError: If 'pyproject.toml' is not found.
        tomli.TOMLDecodeError: If 'pyproject.toml' is not a valid TOML file.
        KeyError: If the 'tool.logging' section is missing in 'pyproject.toml'.
    """
    # Construct the path to pyproject.toml, assuming it's 3 levels up from this file.
    pyproject_path = Path(__file__).parents[3] / "pyproject.toml"

    try:
        with open(pyproject_path, "rb") as f:
            config = tomli.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}: {e}") from e
    except tomli.TOMLDecodeError as e:
        raise tomli.TOMLDecodeError(f"Failed to decode pyproject.toml: {e}") from e

    # Determine the log level:
    # 1. Check for BATISTATEMPLATE_LOG_LEVEL environment variable.
    # 2. Default to 'INFO' if not set or invalid.
    log_level_env = os.getenv("BATISTATEMPLATE_LOG_LEVEL", "INFO").upper()

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level_env not in valid_levels:
        print(f"Invalid log level '{log_level_env}' from environment, defaulting to INFO")
        log_level = "INFO"
    else:
        log_level = log_level_env

    try:
        # Extract logging configuration from pyproject.toml
        logging_config = config["tool"]["logging"]
    except KeyError as e:
        raise KeyError(f"'tool.logging' section not found in pyproject.toml: {e}") from e

    # Override the log level specified in pyproject.toml with the determined log level.
    logging_config["loggers"]["batistatemplate"]["level"] = log_level

    # Configure logging using the dictionary configuration.
    logging.config.dictConfig(logging_config)


# Get the logger for the 'batistatemplate' application.
logger = logging.getLogger("batistatemplate")
