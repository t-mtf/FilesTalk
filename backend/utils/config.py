"""
Configure logging settings for the application.
"""

import logging
import os
from pathlib import Path


def configure_logging(
    logger_name: str,
    filename: str = "app.log",
    level: int = logging.ERROR,
    log_format: str = "%(asctime)s:%(levelname)s:%(message)s",
) -> None:
    """
    Configure logging settings for the application.

    This function sets up the logging system to log messages to a file.
    Args:
        logger_name (str): The name of the logger.
        filename (str): The name of the file to log messages to.
        level (int): The minimum severity level for messages to be logged.
        log_format (str): The format of log messages.

    Returns:
        None
    """
    log_file_path = Path(".") / "data" / filename
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def configure_azure_settings(
    chat_model: str = "gpt-4-128k", embedding_model: str = "text-embedding-ada-002"
) -> dict:
    """
    Configure and return Azure settings for various services like
    Azure OpenAI.

    Args:
        chat_model (str): The chat model to use.
        embedding_model (str): The embedding model to use.

    Returns:
        dict: A dictionary containing the configuration settings for Azure services.
    """
    batch_logger = logging.getLogger("batch_logger")

    azure_settings = {}

    # Fetch environment variables based on region

    azure_settings["azure_openai_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_key is None:
        batch_logger.error("Environment variable 'AZURE_OPENAI_API_KEY' is not set.")
        raise ValueError("Environment variable 'AZURE_OPENAI_API_KEY' is not set.")

    azure_settings["azure_openai_key"] = azure_key
    if embedding_model == "text-embedding-ada-002":

        azure_settings["azure_openai_embedding_deployment"] = os.getenv(
            "TEXT_EMBEDDING_ADA_MODEL_DEPLOYMENT"
        )
        azure_settings["azure_openai_embedding_api_version"] = os.getenv(
            "TEXT_EMBEDDING_ADA_API_VERSION"
        )
    elif embedding_model == "text-embedding-3-large":

        azure_settings["azure_openai_embedding_deployment"] = os.getenv(
            "TEXT_EMBEDDING_LARGE_MODEL_DEPLOYMENT"
        )
        azure_settings["azure_openai_embedding_api_version"] = os.getenv(
            "TEXT_EMBEDDING_LARGE_API_VERSION"
        )

    if chat_model == "gpt-4o-mini":
        azure_settings["azure_openai_chat_deployment"] = os.getenv(
            "GPT4O_MINI_MODEL_DEPLOYMENT"
        )
        azure_settings["azure_openai_chat_api_version"] = os.getenv(
            "GPT4O_MINI_AZURE_API_VERSION"
        )

    elif chat_model == "gpt-4o":

        azure_settings["azure_openai_chat_deployment"] = os.getenv(
            "GPT4O_MODEL_DEPLOYMENT"
        )
        azure_settings["azure_openai_chat_api_version"] = os.getenv(
            "GPT4O_AZURE_API_VERSION"
        )

    return azure_settings


def configure_lighton_settings(
    chat_model: str = "alfred-4", embedding_model: str = "multilingual-e5-large"
) -> dict:
    """
    Configure and return Lighton settings.

    Args:
        chat_model (str): The chat model to use.
        embedding_model (str): The embedding model to use.

    Returns:
        dict: A dictionary containing the configuration settings for Lighton services.
    """
    batch_logger = logging.getLogger("batch_logger")

    lighton_settings = {}

    lighton_settings["lighton_endpoint"] = os.getenv("lighton_ENDPOINT")
    lighton_key = os.getenv("lighton_API_KEY")
    if lighton_key is None:
        batch_logger.error("Environment variable 'lighton_API_KEY' is not set.")
        raise ValueError("Environment variable 'lighton_API_KEY' is not set.")

    lighton_settings["lighton_key"] = lighton_key.strip()

    return lighton_settings
