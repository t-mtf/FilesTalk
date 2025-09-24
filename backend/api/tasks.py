"""
This module contains the batch processing task for the API.
"""

import logging
from pathlib import Path
from shutil import rmtree
from typing import List

import pandas as pd
from rq import get_current_job

from batch_processing.main import run_batch
from utils.config import configure_logging


def batch_processing_task(cuid: str, scope: dict, prompts: List, fields: List):
    """
    This function performs batch processing task.

    Args:
        cuid (str): The unique identifier for the user.
        scope (dict): The scope of the batch processing task.
        prompts (List): The list of prompts for the batch processing task.
        fields (List): list of fields to extract from contracts api

    Returns:
        Tuple[DataFrame, Dict, DataFrame]: A tuple containing the results of the
        batch processing job as a DataFrame, a dictionary containing statistics about
        the batch processing job, and a DataFrame containing indexing statistics for
        contract amendments.
    """
    try:
        configure_logging(
            logger_name="batch_logger",
            filename=f"{get_current_job().id}.log",
            level=logging.WARNING,
        )

        batch_logger = logging.getLogger("batch_logger")
        batch_logger.info("This is an info message in batch_logger")

        # Remove data/tmp directory if it exists
        tmp_dir_path = Path(".") / "data" / "tmp"
        if tmp_dir_path.exists() and tmp_dir_path.is_dir():
            try:
                rmtree(tmp_dir_path)
            except Exception as e:
                batch_logger.error(
                    "An error occurred while deleting the tmp directory: %s", e
                )

        return run_batch(cuid, scope, prompts, fields, get_current_job())
    except Exception as e:
        batch_logger.error("An error occured in batch_processing_task function: %s", e)
        # return two empty dataframes and one empty dictionary
        return (pd.DataFrame([]), {}, pd.DataFrame([]))
