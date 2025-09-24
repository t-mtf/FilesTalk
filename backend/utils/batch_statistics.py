"""
This module contains the BatchStatistics class which is used to manage and update
the statistics related to a batch processing job. It includes methods to update
tokens and cost, and finalize statistics after processing.
"""

import time
from datetime import datetime


class BatchStatistics:
    """
    This class is used to manage and update the statistics related to a batch
    processing job.

    Attributes:
    start_time (str): The start time of the batch processing job.
    end_time (str): The end time of the batch processing job.
    total_execution_time (str): The total execution time of the batch processing job.
    total_cost (float): The total cost of the batch processing job.
    question_answer_cost (float): The cost associated with question answering in the
                                    batch processing job.
    document_embedding_cost (float): The cost associated with document embedding in
                                        the batch processing job.
    prompt_tokens (int): The number of prompt tokens used in the batch processing job.
    completion_tokens (int): The number of completion tokens used in the batch
                                processing job.
    total_document_tokens (int): The total number of document tokens processed in the
                                batch processing job.
    """

    def __init__(self):
        self.start_time = datetime.fromtimestamp(time.time()).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.end_time = ""
        self.total_execution_time = ""
        self.total_cost = 0
        self.question_answer_cost = 0
        self.document_embedding_cost = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_document_tokens = 0

    def update_tokens_and_cost(self, cb):
        """
        Update the number of tokens and the cost associated with question answering.

        Parameters:
        cb: The callback object containing the number of tokens and the cost.
        """
        self.prompt_tokens += cb.prompt_tokens
        self.completion_tokens += cb.completion_tokens
        self.question_answer_cost += cb.total_cost

    def finalize_statistics(self, df_index_stats, embedding_price):
        """
        Finalize the statistics by updating the total document tokens,
        document embedding cost, total cost, end time, and total execution time.

        Parameters:
        df_index_stats (DataFrame): The DataFrame containing the index statistics for
                                    contract amendments.
        embedding_price (float): The price of document embedding.
        """
        self.total_document_tokens = df_index_stats["tokens"].sum()
        self.document_embedding_cost += self.total_document_tokens * embedding_price
        self.total_cost = self.document_embedding_cost + self.question_answer_cost
        self.end_time = datetime.fromtimestamp(time.time()).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        total_execution_time = (
            time.time()
            - datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S").timestamp()
        )
        hours, remainder = divmod(total_execution_time, 3600)
        minutes, _ = divmod(remainder, 60)
        self.total_execution_time = f"{int(hours)} hours and {int(minutes)} minutes"
