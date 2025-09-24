"""
This module contains the main functionality for running batch processing jobs.
It includes the function `run_batch`, which takes a user ID, scope, and list of
prompts, and runs a batch processing job. The job involves processing a set of
contracts as specified by the scope, using the provided prompts. The results of
the job, statistics about the job, and index statistics for contract amendments
are returned. This module uses several external libraries including langchain,
and also relies on several utility modules. Some important parameters are set in
constants.py like large language, and embedding models.
"""

import logging
import sys
from typing import Any, Dict, Tuple
from pathlib import Path
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame

from utils.batch_statistics import BatchStatistics
from utils.constants import EMBEDDING_MODEL_AZURE, EMBEDDING_PRICE
from utils.utils import (
    apply_filters_on_contract_agreement_attachment_list,
    get_attachments_with_fields,
    get_contract_amendment_id_list_by_input,
    get_index_contracts_amendment_content,
    get_input_type,
    get_iterable,
    get_llm_and_embeddings_configs,
    get_vector_store,
    process_input_argument,
    add_zipfiles_dataframe,
)


def run_batch(
    cuid: str, scope: dict, prompts: list, fields: list, current_job: Any
) -> Tuple[DataFrame, Dict, DataFrame]:
    """
    This function runs a batch processing job for a given user ID (cuid), scope,
    and prompts.

    Parameters:
    cuid (str): The unique identifier for the user.
    scope (dict): The scope of the batch processing job.
    prompts (list): The list of prompts to be used in the batch processing job.
    fields (list): The list of fields to extract from contract api

    Returns:
    Tuple[DataFrame, Dict, DataFrame]: A tuple containing the results of the
    batch processing job as a DataFrame, a dictionary containing statistics about
    the batch processing job, and a DataFrame containing indexing statistics for
    contract amendments.
    """
    try:
        batch_logger = logging.getLogger("batch_logger")

        batch_statistics = BatchStatistics()

        batch_logger.info("=" * 80)
        batch_logger.info("JOB_ID: %s", current_job.id)
        batch_logger.info("=" * 80)
        load_dotenv(find_dotenv())

        zone = scope.get("zone", None)
        llm, embeddings = get_llm_and_embeddings_configs(zone)
        # Identify the input type
        input_type = get_input_type(scope)
        iterable = get_iterable(scope, input_type)

        # vectorstore
        collection_name = str(cuid)
        vector_store = get_vector_store(embeddings, collection_name)

        df_questions = pd.DataFrame(prompts)

        rows = []
        df_index_stats_contract_amendements_list = []
        contract_agreement_attachment_list = []

        for i in iterable:
            # get all contract-amendment id's and fields
            contract_amendment_id_list = get_contract_amendment_id_list_by_input(
                cuid, zone, input_type, i, scope.get("filters", None)
            )

            # get all contract-amendment-attachments id's and fields
            contract_agreement_attachment = []
            for amendment in contract_amendment_id_list:
                attachments = get_attachments_with_fields(zone, amendment, cuid)
                attachments_fields = [
                    {
                        "documentId": attachment.get("id"),
                        "documentType": attachment.get("type"),
                        "documentLabel": attachment.get("typeLabel"),
                        "originalFilename": attachment.get("originFileName"),
                        "documentCreationDate": attachment.get("creationDate"),
                    }
                    for attachment in attachments
                ]
                contract_agreement_attachment += [
                    {**amendment, **attachment} for attachment in attachments_fields
                ]

            contract_agreement_attachment_filtered = (
                apply_filters_on_contract_agreement_attachment_list(
                    contract_agreement_attachment, scope.get("filters")
                )
            )
            # Update the main list
            contract_agreement_attachment_list += contract_agreement_attachment_filtered

        # batch_logger.warning("%s", contract_agreement_attachment_list)

        ## FIN DU PREMIER MODULE
        if contract_agreement_attachment_list:

            input_argument_df = pd.DataFrame(contract_agreement_attachment_list)
            output_results_file_path = (
                Path(".") / "data" / f"{current_job.id}_document_list.xlsx"
            )
            writer = pd.ExcelWriter(output_results_file_path, engine="xlsxwriter")
            input_argument_df.to_excel(writer, sheet_name="Results", index=False)
            writer.close()

            # Drop rows with NaN values in "originalFilename"
            input_argument_df.dropna(subset=["originalFilename"], inplace=True)

            # Drop rows where "originalFilename" is an empty string or contains only spaces
            input_argument_df = input_argument_df[
                input_argument_df["originalFilename"].str.strip() != ""
            ]

            batch_logger.warning(
                "Dropped %s rows with 'originalFilename': None or empty",
                len(contract_agreement_attachment_list) - len(input_argument_df),
            )
            input_argument_df = input_argument_df.rename(
                columns={
                    "documentId": "document_id",
                    "originalFilename": "original_file_name",
                    "id": "contract_amendment_id",
                }
            )

            contract_amendment_documents, df_index_stats_contract_amendements_ic01 = (
                get_index_contracts_amendment_content(
                    cuid,
                    zone,
                    input_argument_df,
                    llm,
                    vector_store,
                )
            )

            df_index_stats_contract_amendements_list.append(
                df_index_stats_contract_amendements_ic01
            )

            # qa_system_prompt = (
            #     "You are an AI assistant specialized in contract analysis. Use only the "
            #     "following pieces of context to answer the question at the end. If you "
            #     "don't know the answer, answer me with 'NA' only. Else give detailed and"
            #     "structured answers. Please ensure your answer is in English.\n\n"
            #     "**Context:**\n"
            #     "{context}"
            # )

            if "clause_available" in df_questions.columns:

                # qa_system_prompt = (
                #     "You are a contract and revenue assurance manager tasked with analyzing contract documents. "
                #     "Use only the provided context to identify relevant clauses and answer the question at the end. "
                #     "If the answer is uncertain or not found within the context, say so. "
                #     "Provide clear and detailed answers when possible. \n\n"
                #     "**Context:**\n"
                #     "{context}"
                #     "Helpful Answer:"
                # )
                qa_system_prompt = (
                    "Use only the following pieces of context to answer the question at the end."
                    "If you don't know the answer or if it is not sure, give an answer in only one word: 'NA', don't try to make up an answer."
                    "Else give detailed answers."
                    "Act like a contract and revenue assurance manager."
                    "Context: {context}"
                    "Helpful Answer:"
                )

            else:
                qa_system_prompt = (
                    "You are an AI assistant specialized in contract analysis. Use only the "
                    "following pieces of context to answer the question at the end. If you "
                    "don't know the answer, answer me with 'NA' only. Else give detailed and"
                    "structured answers. Please ensure your answer is in English.\n\n"
                    "**Context:**\n"
                    "{context}"
                )

            input_argument_df = add_zipfiles_dataframe(input_argument_df, cuid, zone)

            rows = process_input_argument(
                cuid,
                zone,
                input_argument_df,
                fields,
                df_questions,
                qa_system_prompt,
                llm,
                vector_store,
                batch_statistics,
            )

            # Format output dataframe
            df_results = pd.DataFrame(rows)

            output_results_file_path = (
                Path(".") / "data" / f"{current_job.id}_results.xlsx"
            )
            writer = pd.ExcelWriter(output_results_file_path, engine="xlsxwriter")
            df_results.to_excel(writer, sheet_name="Results", index=False)
            writer.close()

            # Define a mapping of old column names to new column names
            column_mapping = {
                "contract_number": "contract_number",
                "amendment_number": "amendment_number",
                "contract_amendment_id": "id",
                "ic01": "ic01_id",
                "salesRegion": "sales_region",
                "salesCluster": "sales_cluster",
                "salesCountry": "sales_country",
                "customerName": "customer_name",
                "typeLabel": "contract_type_label",
                "effectiveDate": "effective_date",
                "isTermFixed": "contract_term_type_label",
                "expiryDate": "expiry_date",
                "statusLabel": "status_label",
                "initialDate": "contract_creation_date",
                "initialCreatedBy": "contract_team_member",
                "lastUpdateDate": "last_update_date",
                "original_file_name": "file_name",
                "documentCreationDate": "attachment_creation_date",
                "documentId": "document_id",
            }

            # The desired order of columns
            desired_order = [
                "contract_number",
                "amendment_number",
                "id",
                "ic01_id",
                "sales_region",
                "sales_cluster",
                "sales_country",
                "customer_name",
                "contract_type_label",
                "effective_date",
                "contract_term_type_label",
                "expiry_date",
                "status_label",
                "agreement_creation_date",
                "contract_creation_date",
                "contract_team_member",
                "last_update_date",
                "file_name",
                "document_id",
                "attachment_creation_date",
            ]

            # Rename columns based on the mapping
            df_results.rename(columns=column_mapping, inplace=True)

            # Add the agreement_creation_date column with values from contract_creation_date
            df_results["agreement_creation_date"] = df_results["contract_creation_date"]

            # Add columns from prompt answers
            additional_columns = df_questions["name"].unique().tolist()

            if any("inflation found YES-NO" in row for row in rows):
                additional_columns.append("inflation found YES-NO")
            if any("MRG found YES-NO-NOT SURE" in row for row in rows):
                additional_columns.append("MRG found YES-NO-NOT SURE")

            df_results = df_results.reindex(columns=desired_order + additional_columns)

            df_index_stats_contract_amendements = pd.concat(
                df_index_stats_contract_amendements_list, ignore_index=True
            )
            batch_statistics.finalize_statistics(
                df_index_stats_contract_amendements,
                EMBEDDING_PRICE[EMBEDDING_MODEL_AZURE],
            )

            return (
                df_results,
                batch_statistics.__dict__,
                df_index_stats_contract_amendements,
            )
        else:
            batch_logger.error("Aborting: contract_agreement_attachment_list is empty")
            sys.exit(0)

    except Exception as e:
        batch_logger.error("An error occured in run_batch function: %s", e)
