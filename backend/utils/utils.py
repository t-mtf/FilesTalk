"""
This module contains the helper functions of the project.
"""

import base64
import json
import logging
import os
import re
import secrets
import string
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal
from urllib.parse import quote_plus

import django
import jwt
import numpy as np
import pandas as pd
import pytesseract
import requests
import tiktoken
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    OutlookMessageLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel

from requests import RequestException
from requests.auth import HTTPBasicAuth
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from utils.config import configure_azure_settings, configure_lighton_settings
from utils.constants import (
    CHARACTERS_PER_TOKEN,
    CHAT_MODEL_AZURE,
    CHAT_MODEL_LIGHTON,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_AZURE,
    EMBEDDING_MODEL_LIGHTON,
    TEMPERATURE,
    K,
)
from utils.multilingual_e5_large_embeddings import MultilingualE5LargeEmbeddings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User

from api.models import UserContract

batch_logger = logging.getLogger("batch_logger")
ALL_SALES_COUNTRY: list = []
ALL_SALES_COUNTRY_REGION: list = []
ALL_SALES_COUNTRY_CLUSTER: list = []
# =================================================================
# Functions related to api


def is_dev_user(cuid: str) -> bool:
    """Check if a user (via its cuid) is a dev or not

    Args:
        cuid (str): cuid of the user

    Returns:
        bool: True if the user is a dev, False otherwise
    """
    user = User.objects.get(username=cuid.lower())
    user_groupes = user.groups.all()
    return "filestalk" in [groupe.name for groupe in user_groupes]


def generate_excel_file_on_success(
    job, connection, result, *args, output_file_name=None, **kwargs
):
    """
    Generates an Excel file on successful completion of batch processing.

    Args:
        job: The job object that was processed.
        connection:
        result: The result of batch processing.
        output_file_name (str, optional): The name of the output file (default=job_id).
    """
    try:
        (df_results, batch_statistics, df_index_stats_contract_amendements) = result
        if output_file_name is None:
            output_file_name = f"{job.id}"

        output_results_file_path = Path(".") / "data" / f"{output_file_name}.xlsx"
        writer = pd.ExcelWriter(output_results_file_path, engine="xlsxwriter")
        df_results.to_excel(writer, sheet_name="Results", index=False)
        df_batch_statistics = pd.DataFrame(batch_statistics, index=[0])
        df_batch_statistics = df_batch_statistics.round(
            {"total_cost": 3, "question_answer_cost": 3, "document_embedding_cost": 3}
        )
        df_batch_statistics = df_batch_statistics.rename(
            columns={
                "total_cost": "total_cost (USD)",
                "question_answer_cost": "question_answer_cost (USD)",
                "document_embedding_cost": "document_embedding_cost (USD)",
            }
        )

        df_batch_statistics.to_excel(writer, sheet_name="batch_statistics", index=False)

        df_index_stats_contract_amendements = (
            df_index_stats_contract_amendements.rename(
                columns={
                    "elapsed_time": "elapsed_time (second)",
                    "tokens": "number_of_tokens",
                }
            )
        )
        df_index_stats_contract_amendements = df_index_stats_contract_amendements[
            ["contract_amendment_id", "number_of_tokens", "elapsed_time (second)"]
        ]
        df_index_stats_contract_amendements["elapsed_time (second)"] = (
            df_index_stats_contract_amendements["elapsed_time (second)"]
            .fillna(0)  # replace NaN with 0
            .replace([np.inf, -np.inf], 0)  # replace inf with 0
            .round(0)
            .astype(int)
        )
        df_index_stats_contract_amendements.to_excel(
            writer, sheet_name="create_index_statistics", index=False
        )

        writer.close()
    except Exception as e:
        batch_logger.error("Error in generate_excel_file_on_success: %s", e)
        try:
            if output_file_name is None:
                output_file_name = f"{job.id}"
            df_results.to_csv(
                f"data/{output_file_name}_results.txt", sep="\t", index=False
            )
            df_batch_statistics.to_csv(
                f"data/{output_file_name}_batch_statistics.txt", sep="\t", index=False
            )
            df_index_stats_contract_amendements.to_csv(
                f"data/{output_file_name}_index_stats.txt", sep="\t", index=False
            )
        except Exception as csv_generation_error:
            batch_logger.error(
                "Error in generate_excel_file_on_success: %s",
                csv_generation_error,
            )


def job_queue_rank(queue: Any, job_id: str) -> int:
    """Get the job queue rank number

    Args:
        queue (Any): The queue from which to look up the job rank number
        job_id (str): the job id

    Returns:
        int: Job rank number (0-based rank number or -1 if job is not in the queue)
    """
    # Get the list of job IDs in the queue
    job_ids_in_queue = queue.job_ids
    try:
        return job_ids_in_queue.index(job_id)
    except ValueError:
        return -1


def generate_random_password(length=8) -> str:
    """Generate a random password for the given length"""
    characters = string.ascii_letters + string.digits + string.punctuation
    return "".join(secrets.choice(characters) for _ in range(length))


def is_valid_cuid(cuid: str) -> bool:
    """Check if a cuid is valid

    Args:
        cuid (str): cuid to check

    Returns:
        bool: True if cuid is valid, False otherwise
    """
    return len(cuid) == 8 and cuid[:4].isalpha() and cuid[4:].isdigit()


# ============================================================================
# Functions to connect to Contracts API
# TODO
@retry(
    stop=stop_after_attempt(10),
    wait=wait_fixed(5),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, RequestException)),
    before=before_log(logging.getLogger("batch_logger"), logging.INFO),
    after=after_log(logging.getLogger("batch_logger"), logging.INFO),
)
def generates_token(company_ref: bool = False) -> str | None:
    payload = "grant_type=client_credentials"
    if company_ref:
        url = os.getenv("BASIC_COMPANYREF_API_URL") + os.getenv("TOKEN_API_URL")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(
            url,
            headers=headers,
            data=payload,
            auth=HTTPBasicAuth(
                username=os.getenv("COMPANYREF_CLIENT_ID"),
                password=os.getenv("COMPANYREF_CLIENT_SECRET"),
            ),
        )
    else:
        url = os.getenv("BASIC_API_URL") + os.getenv("TOKEN_API_URL")
        headers = {
            "Authorization": os.getenv("TOKEN_AUTHORIZATION"),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        response_dict = json.loads(response.text)
        token = f"{response_dict.get('token_type')} {response_dict.get('access_token')}"
    else:
        token = None
    return token


def is_token_expired(raw_token: str) -> bool:
    """Check if the given token is expired

    Args:
        raw_token (str): The raw token

    Returns:
        bool: True if the token is expired, False otherwise
    """
    if raw_token:
        token = raw_token.split(" ")[1]  # remove 'Bearer' from token
        try:
            decoded_token = jwt.decode(token, options={"verify_signature": False})
            exp = decoded_token.get("exp")
            if exp:
                expiration_time = datetime.fromtimestamp(exp, tz=timezone.utc)
                return expiration_time < datetime.now(timezone.utc)

            return True
        except jwt.ExpiredSignatureError:
            return True
        except Exception as e:
            batch_logger.error("Error while decoding token: %s", e)
            return True
    else:
        batch_logger.error("Token must be given.")
        return True


def is_token_in_env(company_ref: bool = False) -> bool:
    """Check if the token is in the environment variables"""
    if company_ref:
        return os.getenv("TOKEN_ACCESS_COMPANY_REF") is not None
    return os.getenv("TOKEN_ACCESS_CONTRACT_API") is not None


def get_token(company_ref: bool = False):
    """Get the token from the environment variables or generate a new one"""
    if company_ref:
        token_company_ref = os.getenv("TOKEN_ACCESS_COMPANY_REF")
        if not is_token_in_env(company_ref) or (
            is_token_in_env and is_token_expired(token_company_ref)
        ):
            os.environ["TOKEN_ACCESS_COMPANY_REF"] = generates_token(company_ref)
        return os.getenv("TOKEN_ACCESS_COMPANY_REF")

    token_contract_api = os.getenv("TOKEN_ACCESS_CONTRACT_API")
    if not is_token_in_env(company_ref) or (
        is_token_in_env and is_token_expired(token_contract_api)
    ):
        os.environ["TOKEN_ACCESS_CONTRACT_API"] = generates_token(company_ref)
    return os.getenv("TOKEN_ACCESS_CONTRACT_API")


def make_get_request_to_contracts_api(url: str, cuid: str) -> requests.Response:
    """Make get request to contract api

    Args:
        url (str): url used for the request
        cuid (str): cuid of the user
    Returns:
        requests.Response: response object
    """
    if is_dev_user(cuid):
        headers = {
            "sm_universalid": "FILESTALK",
            "Authorization": get_token(company_ref=False),
        }
    else:
        headers = {
            "sm_universalid": cuid.upper(),
            "Authorization": get_token(company_ref=False),
        }
    payload = {}

    try:
        response = requests.get(url, headers=headers, data=payload)
        return response
    except requests.exceptions.RequestException as e:
        batch_logger.error(f"Request failed: {e}")
        raise


def make_get_request_to_companyref_api(url: str) -> requests.Response:
    """Make get request to company ref api

    Args:
        url (str): url used for the request
    Returns:
        requests.Response: response object
    """
    headers = {
        "Accept": "application/json;charset=utf-8",
        "Authorization": get_token(company_ref=True),
    }
    payload = {}
    return requests.get(url, headers=headers, data=payload)


def get_effective_date(milestones_field: list) -> str:
    """Get effective date field from milestones field

    Args:
        milestones_field (list): list containing effective date value
    Returns:
        str: effective date value
    """
    if milestones_field:
        for milestone in milestones_field:
            if milestone.get("name") == "EFFECTIVE_DATE":
                return milestone.get("effectiveDateTime")


def get_sales_country(characteristic_field: list) -> str:
    """Get sales country name field from characteristic field

    Args:
        characteristic_field (list): list containing sales country value
    Returns:
        str: value of the sales country field
    """
    if characteristic_field:
        for char in characteristic_field:
            if char.get("name") == "CONTRACT_ACCESS_ORIGIN":
                return char.get("valueLabel")


def get_ico1_id_for_all_contracts(
    engaged_party_role_field: list,
) -> str:
    """Get ico1 id field from engaged party role field

    Args:
        engaged_party_role_field (list): list containing engaged party value

    Returns:
        str: value of the ic01 field
    """
    if engaged_party_role_field:
        for eng in engaged_party_role_field:
            for party in eng.get("partyId"):
                if party.get("type") == "ICO1_ID":
                    return party.get("id")


def get_all_sales_country() -> list:
    """Get all sales country from company ref api

    Returns:
        list: list of all sales countries
    """
    if not ALL_SALES_COUNTRY:
        url = (
            os.getenv("BASIC_COMPANYREF_API_URL")
            + "/cyr_b2breferencedata_uat/v1/referenceData?refCode=sales_country"
        )
        batch_logger.info("Getting all sales country list")
        response = make_get_request_to_companyref_api(url)
        if response.status_code == 200:
            response_body = json.loads(response.text)
            if response_body:
                ALL_SALES_COUNTRY.extend(response_body)
            else:
                batch_logger.info("No sales country list found")
        else:
            batch_logger.error(
                "Error retrieving list of all sales country %s", response.status_code
            )
    return ALL_SALES_COUNTRY


def get_all_sales_country_region() -> list:
    """Get all sales country regions from company ref

    Returns:
        list: list of all sales country regions
    """
    if not ALL_SALES_COUNTRY_REGION:
        url = (
            os.getenv("BASIC_COMPANYREF_API_URL")
            + "/cyr_b2breferencedata_uat/v1/referenceData?refCode=sales_country_region"
        )
        batch_logger.info("Getting all sales country region list")
        response = make_get_request_to_companyref_api(url)
        if response.status_code == 200:
            response_body = json.loads(response.text)
            if response_body:
                ALL_SALES_COUNTRY_REGION.extend(response_body)
            else:
                batch_logger.info("No sales country region list found")
        else:
            batch_logger.error(
                "Error retrieving list of all sales country region %s",
                response.status_code,
            )
    return ALL_SALES_COUNTRY_REGION


def get_all_sales_country_cluster() -> list:
    """Get all sales country clusters from company ref

    Returns:
        list: list of all sales country clusters
    """
    if not ALL_SALES_COUNTRY_CLUSTER:
        url = (
            os.getenv("BASIC_COMPANYREF_API_URL")
            + "/cyr_b2breferencedata_uat/v1/referenceData?refCode=sales_country_cluster"
        )
        batch_logger.info("Getting all sales country cluster list")
        response = make_get_request_to_companyref_api(url)
        if response.status_code == 200:
            response_body = json.loads(response.text)
            if response_body:
                ALL_SALES_COUNTRY_CLUSTER.extend(response_body)
            else:
                batch_logger.info("No sales country cluster list found")
        else:
            batch_logger.error(
                "Error retrieving list of all sales country cluster %s",
                response.status_code,
            )
    return ALL_SALES_COUNTRY_CLUSTER


def get_sales_country_code_by_sales_country(contract_sales_country: str):
    """Get sales country code by contract sales country name

    Args:
        contract_sales_country (str): contract sales country name
    Returns:
        str: sales country code
    """
    for sales_country in get_all_sales_country():
        if sales_country.get("label") == contract_sales_country:
            return sales_country.get("code")


def get_sales_region_by_sales_country(raw_sales_country: str) -> str:
    """Get sales region by given sales country

    Args:
        raw_sales_country (str): raw sales country in "International.salescountry" format

    Returns:
        str: sales country region
    """
    contract_sales_country = raw_sales_country.split(".")[
        1
    ]  # Remove 'International' from sales country
    sales_country_code = get_sales_country_code_by_sales_country(contract_sales_country)
    for sales_country_region in get_all_sales_country_region():
        if sales_country_region.get("code") == sales_country_code:
            return sales_country_region.get("parentCode")


def get_sales_cluster_by_sales_country(raw_sales_country: str) -> str:
    """Get sales cluster by given sales country

    Args:
        raw_sales_country (str): raw sales country in "International.salescountry" format

    Returns:
        str: sales country cluster
    """
    contract_sales_country = raw_sales_country.split(".")[
        1
    ]  # Remove 'International' from sales country
    sales_country_code = get_sales_country_code_by_sales_country(contract_sales_country)
    for sales_country_cluster in get_all_sales_country_cluster():
        if sales_country_cluster.get("code") == sales_country_code:
            return sales_country_cluster.get("parentCode")


def get_customer_name(engaged_party_role_field: list) -> str:
    """Get customer name field from engaged party role field

    Args:
        engaged_party_role_field (list): list containing customer name value

    Returns:
        str: customer name value
    """
    if engaged_party_role_field:
        for party in engaged_party_role_field:
            if party.get("name") == "ENGAGED_COMPANY":
                return party.get("partyName")


@retry(
    stop=stop_after_delay(120),
    wait=wait_fixed(60),
    before=before_log(logging.getLogger("batch_logger"), logging.INFO),
    after=after_log(logging.getLogger("batch_logger"), logging.INFO),
)
def get_attachments(
    cuid: str, zone: str, contract_number: str, amendment_number: str
) -> list:
    """Get list of attachments for a contract and an amendment

    Args:
        cuid (str): cuid of the user
        zone (str): zone of contract extraction (ftsa or eqt)
        contract_number (str): contract number to get attachment for
        amendment_number (str): amendment number to get attachment for

    Returns:
        list: list of attachments for a contract and an amendment
    """
    zone_urls_mapping = {
        "eqt": os.getenv("ATTACHMENT_URL_PREFIX_ITL"),
        "ftsa": os.getenv("ATTACHMENT_URL_PREFIX_FR"),
    }
    zone_url = zone_urls_mapping.get(zone.lower(), "")

    url = (
        f"{os.getenv('BASIC_API_URL')}"
        f"{zone_url}"
        f"{contract_number}-{amendment_number}"
        f"{os.getenv('ATTACHMENT_URL_SUFFIX')}"
    )
    response = make_get_request_to_contracts_api(url, cuid)
    if response.status_code == 200:
        response_body = json.loads(response.text)
        if response_body:
            return response_body
        else:
            batch_logger.info(
                "No attachments found for contract %s and amendment %s.",
                contract_number,
                amendment_number,
            )
            return []
    else:
        batch_logger.error(
            "Error retrieving attachments for contract %s and amendment %s. "
            "Status code: %s",
            contract_number,
            amendment_number,
            response.status_code,
        )
    return []


@retry(
    stop=stop_after_delay(120),
    wait=wait_fixed(60),
    before=before_log(logging.getLogger("batch_logger"), logging.INFO),
    after=after_log(logging.getLogger("batch_logger"), logging.INFO),
)
def get_contract_amendment_id_list_by_input(
    cuid: str, zone: str, input_type: str, input_list: list, filters: dict = None
) -> list:
    """Get the list of contract-amendment IDs by ic01_list

    Args:
        cuid (str): cuid of the user
        zone (str): zone of contract extraction (FTSA or EQT)
        ic01_list (list): list of ic01
        filters (dict, optional): additional filters to apply to the request. Defaults to None.

    Returns:
        list: list of contract-amendment IDs
    """
    base_url = f"https://afi-obs.apibackbone.api.intraorange/agreementmanagement/beta/unv/{zone.upper()}/agreements/"
    params = {
        "fields": "id,status,statusLabel,type,typeLabel,effectiveDate,isTermFixed,initialDate,initialCreatedBy,expiryDate,engagedPartyRole,characteristic,milestones,associatedAgreement,lastUpdateDate",
        "limit": "200",
    }
    if input_type == "ic01_list":
        params["engagedPartyRole.partyId.type"] = "ICO1_ID"
        params["engagedPartyRole.partyId.id"] = input_list

    if input_type == "period":
        params["lastUpdatedDateInterval"] = input_list

    if input_type == "id_list":
        params["id"] = input_list

    if filters:
        if filters.get("status"):
            params["status"] = filters.get("status")
        if filters.get("contract_type"):
            params["type"] = filters.get("contract_type")

    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base_url}?{query_string}"
    response = make_get_request_to_contracts_api(url, cuid)

    if response.status_code == 200:
        response_body = json.loads(response.text)
        if response_body:
            contract_amendment_list = response_body
            for agreement in contract_amendment_list:
                # add missing fields
                sales_country = get_sales_country(agreement.get("characteristic"))
                field_mapping = {
                    "contract_number": agreement.get("id").split("-")[0],
                    "amendment_number": agreement.get("id").split("-")[1],
                    "ic01": get_ico1_id_for_all_contracts(
                        agreement.get("engagedPartyRole")
                    ),
                    "effectiveDate": get_effective_date(agreement.get("milestones")),
                    "salesCountry": sales_country,
                    "salesRegion": get_sales_region_by_sales_country(sales_country),
                    "salesCluster": get_sales_cluster_by_sales_country(sales_country),
                    "customerName": get_customer_name(
                        agreement.get("engagedPartyRole")
                    ),
                    "isTermFixed": (
                        "Fixed" if agreement.get("isTermFixed") else "Ragged"
                    ),
                }
                agreement.update(field_mapping)
            return contract_amendment_list
        else:
            batch_logger.info("No contract list id found for %s.", input_list)
            return []
    else:
        batch_logger.error(
            "Error retrieving contract list id for %s. Status code: %s",
            input_list,
            response.status_code,
        )
        return []


@retry(
    stop=stop_after_delay(90),
    wait=wait_fixed(30),
    before=before_log(logging.getLogger("batch_logger"), logging.INFO),
    after=after_log(logging.getLogger("batch_logger"), logging.INFO),
)
def get_attachment_content(
    ic01,
    cuid: str,
    zone: str,
    contract_number: str,
    amendment_number: str,
    document_id: str,
    llm,
    metadata: Dict,
) -> List[Document]:
    """Get attachment file and return its content

    Args:
        ic01 (str): Identifier of the client
        cuid (str): cuid of the user
        contract_number (str): contract number to get attachment for
        amendment_number (str): amendment number to get attachment for
        document_id (str): document id
        llm (AzureChatOpenAI): Large Language Model used
        metadata (Dict): metadata dictionary containing ic01, contract_number, ...

    Returns:

        contract_amendment_contents (List[Document]): Page content and metadata.
    """
    contract_amendment_contents = []
    try:
        directory_path = (
            Path(".") / "data" / "tmp" / f"{contract_number}_{amendment_number}"
        )
        directory_path.mkdir(parents=True, exist_ok=True)

        zone_urls_mapping = {
            "eqt": os.getenv("ATTACHMENT_URL_PREFIX_ITL"),
            "ftsa": os.getenv("ATTACHMENT_URL_PREFIX_FR"),
        }
        zone_url = zone_urls_mapping.get(zone.lower(), "")

        url = (
            f"{os.getenv('BASIC_API_URL')}"
            f"{zone_url}"
            f"{contract_number}-{amendment_number}"
            f"{os.getenv('ATTACHMENT_CONTENT_URL')}"
            f"{document_id}"
        )

        response = make_get_request_to_contracts_api(url, cuid)
        if response.status_code == 200:
            response_body = json.loads(response.text)
            if response_body:
                # check if id exists for user
                if check_user_contract(cuid, document_id):
                    batch_logger.info(
                        "document_id %s exists for user with cuid %s",
                        document_id,
                        cuid,
                    )

                else:
                    batch_logger.info(
                        "contract_number:%s, amendment_number: %s, document_id: %s, original_file_name: %s",
                        contract_number,
                        amendment_number,
                        response_body.get("id"),
                        response_body.get("originFileName"),
                    )

                    content_bytes = base64.b64decode(response_body.get("content"))
                    file_name = response_body.get("originFileName")

                    output_file_path = directory_path / file_name
                    with open(output_file_path, "wb") as f:
                        f.write(content_bytes)
                    metadata.update(
                        {
                            "creation_date": response_body.get("creationDate"),
                            "contract_type": response_body.get("type"),
                            "contract_type_label": response_body.get("typeLabel"),
                        }
                    )

                    contract_amendment_contents.extend(
                        read_attachment_content(
                            output_file_path,
                            llm,
                            contract_number,
                            amendment_number,
                            cuid,
                            ic01,
                            metadata,
                        )
                    )
        else:
            batch_logger.error(
                "Error retrieving attachment content for contract_number %s, amendment_number %s, document_id %s. Response: %s",
                contract_number,
                amendment_number,
                document_id,
                response,
            )

    except Exception as e:
        batch_logger.error(
            "Error in get_attachment_content, contract %s, amendment %s:, document_id: %s %s",
            contract_number,
            amendment_number,
            document_id,
            e,
        )
    return contract_amendment_contents


def get_attachments_with_fields(zone: str, agreement: Dict, cuid: str):
    try:
        attachments = get_attachments(
            cuid,
            zone,
            agreement.get("contract_number"),
            agreement.get("amendment_number"),
        )
        return attachments

    except Exception as e:
        batch_logger.error(
            "Error in get_attachment_content, contract %s, amendment %s: %s",
            agreement.get("contract_number"),
            agreement.get("amendment_number"),
            e,
        )


# ============================================================================
# Functions for batch processing
class InflationFoundBool(BaseModel):
    inflation_found_bool: Literal["YES", "NO"]


class MRGFoundBool(BaseModel):
    mrg_found_bool: Literal["YES", "NO", "NOT SURE"]


def get_index_contracts_amendment_content(
    cuid: str,
    zone: str,
    input_argument_df: pd.DataFrame,
    llm: AzureChatOpenAI,
    vector_store: PGVector,
) -> Tuple[List, pd.DataFrame]:
    """
    Retrieves and indexes the contract documents.

    Args:
        cuid (str): The unique identifier for the user.
        zone (str): zone of contract extraction (FTSA or EQT).
        input_argument_df (pd.DataFrame): DataFrame containing contract details.
        llm (AzureChatOpenAI): Large Language Model used.
        vector_store (PGVector): Vector database for storing embeddings.

    Returns:
        Tuple[List, pd.DataFrame]:
            contract_amendment_documents (List): Documents content and metadata.
            df_index_stats_contract_amendements (DataFrame): Statistics of the
                                                                indexing process.
    """
    contract_amendment_documents = []
    index_stats_contract_amendements = []
    for input_index, input_row in input_argument_df.iterrows():
        try:
            metadata = {
                "ic01": input_row["ic01"],
                "contract_number": input_row["contract_number"],
                "amendment_number": input_row["amendment_number"],
                "document_id": input_row["document_id"],
                "file_name": input_row["original_file_name"],
            }

            contract_amendment_document = get_attachment_content(
                input_row["ic01"],
                cuid,
                zone,
                input_row["contract_number"],
                input_row["amendment_number"],
                input_row["document_id"],
                llm,
                metadata,
            )

            index_stats_contract_amendement = pd.DataFrame(
                {
                    "contract_amendment_id": input_row["contract_amendment_id"],
                    "tokens": [0],
                    "elapsed_time": [0],
                }
            )

            if contract_amendment_document:
                index_stats_contract_amendement = index_documents(
                    vector_store,
                    contract_amendment_document,
                    input_row["contract_amendment_id"],
                    zone,
                )
                contract_amendment_documents.extend(contract_amendment_document)
            index_stats_contract_amendements.append(index_stats_contract_amendement)
        except Exception as e:
            batch_logger.error(
                "Error in get_index_contracts_amendment_content: %s",
                e,
            )

    if index_stats_contract_amendements:
        df_index_stats_contract_amendements = pd.concat(
            index_stats_contract_amendements, ignore_index=True
        )
    else:
        df_index_stats_contract_amendements = pd.DataFrame(
            {
                "contract_amendment_id": [None],
                "tokens": [0],
                "elapsed_time": [0],
            }
        )

    return contract_amendment_documents, df_index_stats_contract_amendements


def clean_text(text):
    """
    Removes NUL characters and normalizes whitespace in the input text.

    Args:
        text (str): The text to be cleaned.

    Returns:
        text (str): The cleaned text with NUL characters removed and whitespace trimmed.
    """
    # Remove NUL characters
    text = text.replace("\x00", "")
    # Clean excess whitespace around newlines and trim
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return text


def convert_pdf_to_image(file_path: Path, page_number: int = 1) -> str:
    """
    Convert a given page of a PDF file to a base64 encoded image.

    Args:
        file_path (Path): The path to the PDF file.

    Returns:
        encoded_image (str): The base64 encoded string of the image.
    """
    tmp_directory_path = Path(".") / "data" / "tmp"
    tmp_directory_path.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(
        str(file_path), grayscale=True, first_page=page_number, last_page=page_number
    )
    tmp_image_path = tmp_directory_path / "tmp_page.jpg"
    pages[0].save(str(tmp_image_path), "JPEG")

    with open(tmp_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    tmp_image_path.unlink()
    return encoded_image


def detect_document_language(
    file_path: Path, llm: AzureChatOpenAI, page_number: int = 1
) -> str:
    """
    Detect the dominant language of the document based on the content of a given page.

    Args:
        file_path (Path): The path to the document.

    Returns:
        document_language (str): The dominant language of the document.
    """
    try:
        if llm.model_name == "gpt-4o":
            page_image = convert_pdf_to_image(file_path, page_number)
            batch_logger.info(
                "Page %s of file %s read successfully.",
                page_number,
                file_path.parent.name + "/" + file_path.stem,
            )

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "What is the dominant language in this document?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{page_image}"},
                    },
                ],
            )
            response = llm.invoke([message])
            document_language = response.content

            batch_logger.info(
                "Dominant language(s) for file %s: %s",
                file_path.parent.name + "/" + file_path.stem,
                document_language,
            )

            return document_language
        batch_logger.warning("%s is not multimodal.", llm.model_name)
        return ""
    except Exception as e:
        batch_logger.error(
            "Error in detect_document_language, file %s: %s",
            file_path.parent.name + "/" + file_path.stem,
            e,
        )
        return ""


def load_scanned_pdf(
    file_path: Path, document_language: str, pages_to_read: List[int]
) -> List[Document]:
    """
    Load a scanned PDF file and return the content of each page.

    Args:
        file_path (Path): The path to the PDF file.
        pages_to_read (List[int]): List of page numbers to read.

    Returns:
        pages_content (list): Page content and metadata.
    """
    pages_content = []
    if "french" in document_language.lower():
        language_code = "fra"
    elif "portuguese" in document_language.lower():
        language_code = "por"
    elif "chinese" in document_language.lower():
        language_code = "eng+chi_sim+chi_tra"
    else:
        language_code = "eng"

    for i in pages_to_read:
        try:
            pages = convert_from_path(
                file_path,
                dpi=200,
                grayscale=True,
                thread_count=1,
                first_page=i + 1,
                last_page=i + 1,
            )
        except Image.DecompressionBombError:
            batch_logger.warning(
                "Image size exceeds limit, retrying with lower DPI for file %s",
                file_path.parent.name + "/" + file_path.stem,
            )
            pages = convert_from_path(
                file_path,
                dpi=100,
                grayscale=True,
                thread_count=1,
                first_page=i + 1,
                last_page=i + 1,
            )

        try:
            text = pytesseract.image_to_string(pages[0], lang=language_code)
            if text == "":
                batch_logger.warning(
                    "Tesseract output is empty for page %s of file %s",
                    i + 1,
                    file_path.parent.name + "/" + file_path.stem,
                )
            cleaned_text = clean_text(text)
            doc = Document(
                metadata={"page": i + 1},
                page_content=cleaned_text,
            )

            pages_content.append(doc)
        except Exception as e:
            batch_logger.error(
                "Error in load_scanned_pdf, page %s of %s: %s",
                i + 1,
                file_path.parent.name + "/" + file_path.stem,
                e,
            )
    return pages_content


def load_pdf_file(file_path: Path) -> List[Document]:
    """
    Load a PDF file and return the content of each page.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        pages (list): A list of Document objects.
    """
    cleaned_pages = []
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # clean the page_content for each Document object
        for page in pages:
            cleaned_content = clean_text(
                page.page_content
            )  # re.sub(r"\s*\n\s*", "\n", page.page_content).strip()
            cleaned_pages.append(
                Document(metadata=page.metadata, page_content=cleaned_content)
            )

    except Exception as e:
        batch_logger.error(
            "Error in load_pdf_file, file %s: %s",
            file_path.parent.name + "/" + file_path.stem,
            e,
        )
    return cleaned_pages


def load_pdf(file_path: Path, document_language: str = "english") -> List[Document]:
    """
    Load a PDF file and return the content of each page.

    Args:
        file_path (Path): The path to the PDF file.

    Returns:
        pages_content (List[Document]): Page content and metadata.
    """
    pages = []
    try:
        pages = load_pdf_file(file_path)
        scanned_pages = [
            i for i, page in enumerate(pages) if len(page.page_content) < 200
        ]

        if scanned_pages:
            scanned_pages_content = load_scanned_pdf(
                file_path, document_language, scanned_pages
            )

            for i, scanned_page in zip(scanned_pages, scanned_pages_content):
                pages[i] = scanned_page
    except Exception as e:
        batch_logger.error(
            "Error in load_pdf for file %s: %s",
            file_path.parent.name + "/" + file_path.stem,
            e,
        )
    return pages


def load_word_file(file_path: Path) -> List[Document]:
    """
    Load a DOCX file and return the content of the file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        data (list): A list of Document object.
    """
    file_content = []
    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

        cleaned_content = clean_text(documents[0].page_content)
        file_content.append(
            Document(metadata=documents[0].metadata, page_content=cleaned_content)
        )

    except Exception as e:
        batch_logger.error(
            "Error in load_word_file for file %s: %s",
            file_path.parent.name + "/" + file_path.stem,
            e,
        )
    return file_content


def load_msg_file(file_path: Path) -> List[Document]:
    """
    Load a Microsoft Outlook (.msg) file and return the content of the file.

    Args:
        file_path (Path): The path to the file.

    Returns:
        data (List): A list of Document objects.
    """
    file_content = []
    try:
        loader = OutlookMessageLoader(file_path)
        documents = loader.load()

        cleaned_content = clean_text(documents[0].page_content)
        file_content.append(
            Document(metadata=documents[0].metadata, page_content=cleaned_content)
        )

    except Exception as e:
        batch_logger.error(
            "Error in load_msg_file for file %s: %s",
            file_path.parent.name + "/" + file_path.stem,
            e,
        )
    return file_content


@retry(
    stop=stop_after_delay(180),
    wait=wait_fixed(10),
    before=before_log(logging.getLogger("batch_logger"), logging.INFO),
    after=after_log(logging.getLogger("batch_logger"), logging.INFO),
)
def get_vector_store(
    embeddings: AzureOpenAIEmbeddings, collection_name: str
) -> PGVector:
    """
    Get a vector store from a Postgres database.

    Args:
        embeddings: The embeddings model to be used.
        collection_name (str): The name of the collection in the database.

    Returns:
        vector_store (PGVector): A PGVector object that represents the vector store.
    """
    encoded_password = quote_plus(os.getenv("POSTGRES_PASSWORD"))
    connection = (
        f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{encoded_password}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    return vector_store


@retry(
    stop=stop_after_attempt(6),
    wait=wait_fixed(20),
    before=before_log(logging.getLogger("batch_logger"), logging.INFO),
    after=after_log(logging.getLogger("batch_logger"), logging.INFO),
)
def add_documents_to_vector_store(vector_store, split):
    vector_store.add_documents(split)


def index_documents(
    vector_store: PGVector,
    documents: List[Document],
    contract_amendment_id: str,
    zone: str,
) -> pd.DataFrame:
    """
    Indexes the content of documents into a vector store and returns a DataFrame with
    the total number of tokens and elapsed time for each document.

    Args:
        vector_store (PGVector): The vector store where the documents will be indexed.
        documents (List[Document]): A list of documents to be indexed.

    Returns:
        df_index (pd.DataFrame): A DataFrame with the total number of tokens and elapsed
        time for each document.
    """

    try:
        tokenizer = None

        if zone == "eqt":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=[
                    "\n\n",
                    "\n",
                    ".",
                    ",",
                    "\uff0c",  # Fullwidth comma
                    "\u3001",  # Ideographic comma
                    "\uff0e",  # Fullwidth full stop
                    "\u3002",  # Ideographic full stop
                    "",
                ],
            )
        elif zone == "ftsa":
            tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=500,
                chunk_overlap=100,
                separators=[
                    "\n\n",
                    "\n",
                    ".",
                    ",",
                    "\uff0c",  # Fullwidth comma
                    "\u3001",  # Ideographic comma
                    "\uff0e",  # Fullwidth full stop
                    "\u3002",  # Ideographic full stop
                    "",
                ],
            )
        splits = text_splitter.split_documents(documents=documents)

        chunk_ids = {}
        for i, split in enumerate(splits):
            chunk_id = f"{contract_amendment_id}-{split.metadata['document_id']}-{i}"
            split.metadata.update({"chunk_id": chunk_id})

            if (contract_amendment_id, split.metadata["document_id"]) not in chunk_ids:
                chunk_ids[(contract_amendment_id, split.metadata["document_id"])] = {
                    "chunk_id_first": chunk_id,
                    "chunk_id_last": chunk_id,
                }
            else:
                chunk_ids[(contract_amendment_id, split.metadata["document_id"])][
                    "chunk_id_last"
                ] = chunk_id
        # pylint: disable=no-member
        for (ca_id, document_id), chunk_id in chunk_ids.items():
            UserContract.objects.filter(
                contract_amendment_id=ca_id, document_id=document_id
            ).update(
                chunk_id_first=chunk_id["chunk_id_first"],
                chunk_id_last=chunk_id["chunk_id_last"],
            )

    except Exception as e:
        batch_logger.error(
            "Error in index_documents while splitting documents: %s",
            e,
        )
        splits = []
    index_stats = []

    for split in splits:
        try:
            if zone == "ftsa" and EMBEDDING_MODEL_LIGHTON == "multilingual-e5-large":
                split.page_content = "passage: " + split.page_content

            num_tokens = num_tokens_from_string(
                split.page_content, CHARACTERS_PER_TOKEN, tokenizer
            )

            start_time = time.time()
            add_documents_to_vector_store(vector_store, [split])
            end_time = time.time()

            elapsed_time = end_time - start_time

            index_stats.append(
                {
                    "tokens": num_tokens,
                    "elapsed_time": elapsed_time,
                }
            )

        except Exception as e:
            batch_logger.error("Error in index_documents: %s", e)
            batch_logger.error("split: %s", split)
            batch_logger.error("documents: %s", documents)
            index_stats.append(
                {
                    "tokens": 0,
                    "elapsed_time": 0,
                }
            )

    df_index_stats = pd.DataFrame(index_stats)

    df_sum = df_index_stats.sum().to_frame().T
    df_sum["contract_amendment_id"] = contract_amendment_id

    return df_sum


def num_tokens_from_string(
    text: str,
    characters_per_token: int,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
) -> int:
    """
    Returns the number of tokens in a text string.

    Args:
        text (str): The text string to tokenize.

        characters_per_token (int): The average number of characters per token.

        tokenizer: the tokenizer for multilingual-e5-large

    Returns:
        num_tokens (int): The number of tokens in the text string.
    """
    try:
        if tokenizer is not None:
            tokens = tokenizer(text, truncation=False, padding=False)
            num_tokens = len(tokens["input_ids"])
        else:
            """
            encoding_name: The name of the encoding to use for tokenization.
            The available encodings are:
            - 'o200k_base': gpt-4o, gpt-4o-mini
            - 'cl100k_base': gpt-4-turbo, text-embedding-ada-002,
                            text-embedding-3-small, text-embedding-3-large
            """
            encoding_name = "cl100k_base"
            encoding = tiktoken.get_encoding(encoding_name)
            num_tokens = len(encoding.encode(text))
    except Exception as e:
        batch_logger.error("Error in num_tokens_from_string: %s", e)
        num_tokens = len(text) // characters_per_token
    return num_tokens


def read_attachment_content(
    file_path: Path,
    llm: AzureChatOpenAI,
    contract_number: str,
    amendment_number: str,
    cuid: str,
    ic01: str,
    metadata: Dict,
) -> List[Document]:
    """
    Read the content of a file, and return a list of the documents.

    Args:
        file_path (Path): The path to the file.
        llm (AzureChatOpenAI):  Large language model for language detection
                                in scanned pdf files.
        contract_number (str)
        amendment_number (str)
        cuid (str): User's cuid.
        ic01 (str): Identifier of the client.
        metadata (Dict): metadata dictionary containing ic01, contract_number, ...


    Returns:
        file_documents (List[Document]): List of documents read from the file.
    """

    file_documents = []
    file_content = []
    try:
        if file_path.suffix.lower() in [".docx"]:
            file_content = load_word_file(file_path)
        elif file_path.suffix.lower() in [".pdf"]:
            document_language = detect_document_language(file_path, llm)
            file_content = load_pdf(file_path, document_language)
        elif file_path.suffix.lower() in [".zip"]:
            with zipfile.ZipFile(file_path, "r") as opened_zip_file:
                try:
                    unzipped_directory = file_path.parent / file_path.stem
                    unzipped_directory.mkdir(exist_ok=True)
                    opened_zip_file.extractall(unzipped_directory)
                    file_path.unlink()
                except Exception as e:
                    batch_logger.error(
                        "Error in read_attachment_content unzip process: %s",
                        e,
                    )
                for unzipped_file_path in unzipped_directory.rglob("*"):
                    if unzipped_file_path.is_file():
                        file_documents.extend(
                            read_attachment_content(
                                unzipped_file_path,
                                llm,
                                contract_number,
                                amendment_number,
                                cuid,
                                ic01,
                                metadata,
                            )
                        )

                return file_documents
        else:
            batch_logger.warning(
                "Unsupported file type for file %s: %s",
                file_path.parent.name + "/" + file_path.stem,
                file_path.suffix.lower(),
            )
        if file_content:
            file_source = f"{contract_number}-{amendment_number}/{file_path.name}"
            metadata.update(
                {
                    "source": file_source,
                }
            )
            for document in file_content:
                document.metadata.update(metadata)
            file_documents.extend(file_content)

            # store the information in the database
            user = User.objects.get(username=cuid)

            user_contract = UserContract(
                user=user,
                ic01=ic01,
                file_name=metadata["file_name"],
                source=metadata["source"],
                contract_amendment_id=f"{contract_number}-{amendment_number}",
                contract_type=metadata["contract_type"],
                contract_type_label=metadata["contract_type_label"],
                document_id=metadata["document_id"],
                creation_date=metadata["creation_date"],
                indexation_date=datetime.now(timezone.utc),
            )

            user_contract.save()

            batch_logger.info(
                "File %s with document_id %s saved for user_name %s",
                metadata["file_name"],
                metadata["document_id"],
                cuid,
            )

    except Exception as e:
        batch_logger.error("Error in read_attachment_content: %s", e)
    return file_documents


def check_user_contract(cuid, document_id):
    user = User.objects.get(username=cuid)
    return UserContract.objects.filter(user=user, document_id=document_id).exists()


def get_token_from_gatape():
    SCOPES = ["api-mail2fed-v1-prd:emailsend_access"]

    try:
        batch_logger.info("Getting token from gatape")
        response = requests.post(
            "https://okapi-v2.api.hbx.geo.infra.ftgroup/v2/token",
            auth=HTTPBasicAuth(
                os.getenv("OKAPI_CLIENT_ID"), os.getenv("OKAPI_CLIENT_SECRET")
            ),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"grant_type": "client_credentials", "scope": " ".join(SCOPES)},
        )
        if response.status_code == 200:
            return response.json().get("access_token")
    except Exception as e:
        batch_logger.error(f"Error while getting token from gatape, {str(e)}")


def send_email(job, connection, result, *args, email_address):
    """Send an email to the specified email address with batch result file as an attachment"""

    try:
        file_name = f"{job.id}.xlsx"

        with open(f"data/{file_name}", "rb") as file:
            encoded_content = base64.b64encode(file.read()).decode("utf-8")

        mail_template = {
            "messages": [
                {
                    "to": [{"email": email_address}],
                    "from": {"email": "filestalk.notification@orange.com"},
                    "replyTo": {"email": "filestalk.notification@orange.com"},
                    "subject": "Filestalk notification",
                    "html": f"<!DOCTYPE html><html><body><p>Hello,</p><p>Please find attached the result of your batch {job.id}.</p><p>Kind regards.</p><p>FilesTalk Team</p></body></html>",
                    "attachments": [
                        {
                            "fileName": file_name,
                            "contentType": "text/plain",
                            "content": encoded_content,
                        }
                    ],
                }
            ]
        }
        token = get_token_from_gatape()

        batch_logger.info("Sending mail")
        response = requests.post(
            "https://mail2fed.prod.api.hbx.geo.infra.ftgroup/v1/email/send",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=mail_template,
        )
        if response.status_code == 200:
            batch_logger.info("Mail sent successfully")
        else:
            batch_logger.error(
                "Error in response, status code: %s, response text:%s",
                response.status_code,
                response.text,
            )
    except Exception as e:
        batch_logger.error("Error while sending mail: %s", e)


def on_success(job, connection, result, *args, **kwargs):
    """Behavior when batch is successful"""
    output_file_name = f"{job.id}"
    email_address = job.meta.get("email_address")
    generate_excel_file_on_success(job, connection, result, *args, output_file_name)
    send_email(job, connection, result, *args, email_address=email_address)


def get_input_type(scope: Dict):
    """Return the input type for the given scope: ic01_list, period, id_list"""
    if scope.get("ic01_list", None):
        return "ic01_list"
    elif scope.get("period_start", None) and scope.get("period_end", None):
        return "period"
    elif scope.get("id_list", None):
        return "id_list"
    return None


def get_date_intervals(start_date: datetime, end_date: datetime) -> list:
    """
    Génère une liste d'intervalles de dates d'un jour entre deux dates.

    Args:
        start_date (str): Date de début au format 'YYYY-MM-DD'
        end_date (str): Date de fin au format 'YYYY-MM-DD'

    Returns:
        list: Liste de strings représentant les intervalles de dates
              Format de chaque intervalle: 'YYYY-MM-DD,YYYY-MM-DD'

    Example:
        >>> get_date_intervals('2024-01-01', '2024-01-03')
        ['2024-01-01,2024-01-02', '2024-01-02,2024-01-03', '2024-01-03,2024-01-04']

    Raises:
        ValueError: Si le format des dates n'est pas valide
    """
    intervals = []
    current_date = start_date

    # Tant qu'on n'a pas atteint la date de fin
    while current_date <= end_date:
        # Calculer la date du lendemain
        next_date = current_date + timedelta(days=1)

        # Ajouter l'intervalle au format string
        interval = (
            f"{current_date.strftime('%Y-%m-%d')},{next_date.strftime('%Y-%m-%d')}"
        )
        intervals.append(interval)

        # Passer au jour suivant
        current_date = next_date

    return intervals


def get_iterable(scope: Dict, input_type: str):
    """Return an iterable for the given input type: ic01_list, period, id_list"""
    if input_type == "ic01_list":
        return scope.get("ic01_list")
    elif input_type == "period":
        return get_date_intervals(scope.get("period_start"), scope.get("period_end"))
    elif input_type == "id_list":
        return scope.get("id_list")
    return None


def process_approval_workflow_file(
    file_path: Path,
    df_questions: pd.DataFrame,
    llm: AzureChatOpenAI,
    batch_statistics: Dict,
    new_row: Dict,
) -> Dict:
    """
    Process a .docx file with 'approval workflow' in its name using the entire document as context.

    Args:
        file_path (Path): The path to the DOCX file.
        df_questions (pd.DataFrame): DataFrame containing all questions.
        llm (AzureChatOpenAI): The language model to use.
        batch_statistics (Dict): Dictionary to update batch statistics.
        new_row (Dict): The row to update with answers.

    Returns:
        Dinew_row (Dict): Updated row with answers.
    """
    file_content = load_word_file(file_path)
    if not file_content:
        return new_row

    qa_system_prompt_workflow = (
        "You are a contract and revenue assurance manager tasked with analyzing contract documents. "
        "Use only the provided context to identify relevant clauses and answer the question at the end. "
        "If the answer is uncertain or not found within the context, say so. "
        "Provide clear and detailed answers when possible. \n\n"
        "**Important Note:** The context may include sections formatted as forms or tables, "
        "where questions might have YES/NO answers. Pay close attention to whether some of these sections are completed or whether they are all empty, "
        "as a simple 'Yes' or 'No' may not indicate the presence or absence of a clause without additional information. "
        "Consider a clause as present if a clause number is provided, even if other details like commencement date are missing. "
        "If no clause number is provided, treat it as not found. \n\n"
        "**Context:**\n"
        "{context}"
        "Helpful Answer:"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt_workflow),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    context = file_content[0].page_content
    context_document = Document(page_content=context)

    for j, question in df_questions.iterrows():
        try:
            with get_openai_callback() as cb:
                answer = question_answer_chain.invoke(
                    {"input": question["value"], "context": [context_document]}
                )
                new_row[question["name"]] = answer
                batch_statistics.update_tokens_and_cost(cb)

                if question["name"] == "inflation found":
                    response_llm_structured = llm.with_structured_output(
                        InflationFoundBool
                    ).invoke(
                        qa_system_prompt_workflow.format(context=answer)
                        + "Is any inflation clause found based on this context?"
                    )

                    clause_found_answer_bool = getattr(
                        response_llm_structured, "inflation_found_bool"
                    )

                    new_row["inflation found YES-NO"] = clause_found_answer_bool

                elif question["name"] == "clause_available":
                    response_llm_structured = llm.with_structured_output(
                        MRGFoundBool
                    ).invoke(
                        qa_system_prompt_workflow.format(context=answer)
                        + "Is any mrg clause available based on this context?"
                    )

                    clause_found_answer_bool = getattr(
                        response_llm_structured, "mrg_found_bool"
                    )

                    new_row["MRG found YES-NO-NOT SURE"] = clause_found_answer_bool

        except Exception as e:
            batch_logger.error("An error occurred in questions chain invoke %s", e)
            answer = str(e)
            new_row[question["name"]] = answer

    return new_row


def process_input_argument(
    cuid: str,
    zone: str,
    input_argument_df: pd.DataFrame,
    fields: List[str],
    df_questions: pd.DataFrame,
    qa_system_prompt: str,
    llm: AzureChatOpenAI,
    vector_store: PGVector,
    batch_statistics: Dict,
) -> List:
    rows = []
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    df_questions_independent = df_questions[df_questions["dependencies"].isnull()]
    df_questions_dependent = df_questions[df_questions["dependencies"].notnull()]

    for input_index, input_row in input_argument_df.iterrows():
        new_row = input_row.to_dict()

        for question in df_questions.itertuples():
            new_row[question.name] = ""
            if question.name == "inflation found":
                new_row["inflation found YES-NO"] = ""
            if question.name == "clause_available":
                new_row["MRG found YES-NO-NOT SURE"] = ""

        contract_amendment_id_file = (
            f"{new_row['contract_number']}-"
            f"{new_row['amendment_number']}/{new_row['original_file_name']}"
        )

        # Check if "approval workflow" is in the file name
        if "approval workflow" in new_row["original_file_name"].lower() and new_row[
            "original_file_name"
        ].lower().endswith(".docx"):

            directory_path = (
                Path(".")
                / "data"
                / "tmp"
                / f"{new_row['contract_number']}_{new_row['amendment_number']}"
            )
            directory_path.mkdir(parents=True, exist_ok=True)

            zone_urls_mapping = {
                "eqt": os.getenv("ATTACHMENT_URL_PREFIX_ITL"),
                "ftsa": os.getenv("ATTACHMENT_URL_PREFIX_FR"),
            }
            zone_url = zone_urls_mapping.get(zone.lower(), "")

            url = (
                f"{os.getenv('BASIC_API_URL')}"
                f"{zone_url}"
                f"{new_row['contract_number']}-{new_row['amendment_number']}"
                f"{os.getenv('ATTACHMENT_CONTENT_URL')}"
                f"{new_row['document_id']}"
            )
            response = make_get_request_to_contracts_api(url, cuid)
            if response.status_code == 200:
                response_body = json.loads(response.text)
                content_bytes = base64.b64decode(response_body.get("content"))
                file_name = response_body.get("originFileName")

                file_path = directory_path / file_name
                with open(file_path, "wb") as f:
                    f.write(content_bytes)

                new_row = process_approval_workflow_file(
                    file_path,
                    df_questions,
                    llm,
                    batch_statistics,
                    new_row,
                )
                rows.append(new_row)
                continue

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": K,
                "filter": {"source": {"$ilike": f"%{contract_amendment_id_file}%"}},
            },
        )
        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

        independent_answers = {}
        clause_found_answer_bool = None
        for j, question in df_questions_independent.iterrows():
            try:
                with get_openai_callback() as cb:
                    response = retrieval_chain.invoke({"input": question["value"]})
                    batch_statistics.update_tokens_and_cost(cb)
                    answer = response.get("answer", "ERROR")
                    independent_answers[question["name"]] = answer
                    new_row[question["name"]] = answer

                    if question["name"] == "inflation found":
                        response_llm_structured = llm.with_structured_output(
                            InflationFoundBool
                        ).invoke(
                            f"Is any inflation clause found based on this context?\ncontext:\n{answer}"
                        )

                        clause_found_answer_bool = getattr(
                            response_llm_structured, "inflation_found_bool"
                        )

                        new_row["inflation found YES-NO"] = clause_found_answer_bool

                        if clause_found_answer_bool == "NO":
                            rows.append(new_row)
                            break

                    elif question["name"] == "clause_available":
                        response_llm_structured = llm.with_structured_output(
                            MRGFoundBool
                        ).invoke(
                            f"Is any clause available based on this context?\ncontext:\n{answer}"
                        )

                        clause_found_answer_bool = getattr(
                            response_llm_structured, "mrg_found_bool"
                        )

                        new_row["MRG found YES-NO-NOT SURE"] = clause_found_answer_bool

                        if clause_found_answer_bool == "NO":
                            rows.append(new_row)
                            break

            except Exception as e:
                batch_logger.error(
                    "An error occurred in independent questions chain invoke %s", e
                )
                answer = "ERROR"
                new_row[question["name"]] = answer

        if clause_found_answer_bool == "NO":
            continue

        for j, question in df_questions_dependent.iterrows():
            prompt_name = question["name"]
            dependencies = question["dependencies"]

            context_dependencies = "\n".join(
                f"{dep}: {independent_answers.get(dep, 'NA')}" for dep in dependencies
            )

            try:
                with get_openai_callback() as cb:
                    context_document = Document(page_content=context_dependencies)
                    answer = question_answer_chain.invoke(
                        {"input": question["value"], "context": [context_document]}
                    )
                    batch_statistics.update_tokens_and_cost(cb)
            except Exception as e:
                batch_logger.error(
                    "An error occurred in dependent questions invoke %s", e
                )
                answer = str(e)  # "ERROR"
            new_row[prompt_name] = answer

        rows.append(new_row)

    return rows


def add_zipfiles_dataframe(
    input_argument_df: pd.DataFrame, cuid: str, zone: str
) -> pd.DataFrame:
    new_rows = []

    zip_rows = input_argument_df[
        input_argument_df["original_file_name"].str.endswith(".zip", na=False)
    ]

    for _, input_row in zip_rows.iterrows():
        original_file_name = input_row["original_file_name"]
        file_path = Path(original_file_name)

        directory_path = (
            Path(".")
            / "data"
            / "tmp"
            / f"{input_row['contract_number']}_{input_row['amendment_number']}"
        )
        directory_path.mkdir(parents=True, exist_ok=True)

        zone_urls_mapping = {
            "eqt": os.getenv("ATTACHMENT_URL_PREFIX_ITL"),
            "ftsa": os.getenv("ATTACHMENT_URL_PREFIX_FR"),
        }
        zone_url = zone_urls_mapping.get(zone.lower(), "")

        url = (
            f"{os.getenv('BASIC_API_URL')}"
            f"{zone_url}"
            f"{input_row['contract_number']}-{input_row['amendment_number']}"
            f"{os.getenv('ATTACHMENT_CONTENT_URL')}"
            f"{input_row['document_id']}"
        )
        response = make_get_request_to_contracts_api(url, cuid)
        if response.status_code == 200:
            response_body = json.loads(response.text)
            content_bytes = base64.b64decode(response_body.get("content"))
            file_name = response_body.get("originFileName")

            output_file_path = directory_path / file_name
            with open(output_file_path, "wb") as f:
                f.write(content_bytes)

            with zipfile.ZipFile(output_file_path, "r") as opened_zip_file:
                unzipped_directory = directory_path / file_path.stem
                unzipped_directory.mkdir(exist_ok=True)
                opened_zip_file.extractall(unzipped_directory)

            for unzipped_file_path in unzipped_directory.rglob("*"):
                if unzipped_file_path.is_file():
                    new_row = input_row.copy()
                    new_row["original_file_name"] = unzipped_file_path.name
                    new_rows.append(new_row)

    new_rows_df = pd.DataFrame(new_rows, columns=input_argument_df.columns)
    updated_df = pd.concat([input_argument_df, new_rows_df], ignore_index=True)

    return updated_df


def apply_original_file_name_filter(filtered_list: list, values: list) -> list:
    """Apply filter on originalFilename field

    Args:
        filtered_list (list): list to be filtered by originalFilename
        values (list): values of originalFilename to be removed

    Returns:
        list: list of item filtered by originalFilename
    """
    return [
        item
        for item in filtered_list
        if item.get("originalFilename")
        and not any(
            value
            in re.sub(
                r"[^a-zA-Z0-9\s]",
                "",
                item["originalFilename"].split(".pdf")[0].lower().strip(),
            )
            for value in values
        )
    ]


def apply_document_creation_date_filter(
    filtered_list: list, start_period: datetime.date, end_period: datetime.date
) -> list:
    """Apply filter on documentCreationDate field

    Args:
        filtered_list (list): list to be filtered by documentCreationDate
        start_period (datetime.date): start datetime filter (include)
        end_period (datetime.date): end datetime filter (exclude)

    Returns:
        list: list of item filtered by documentCreationDate
    """
    return [
        item
        for item in filtered_list
        if item.get("documentCreationDate")
        and str(start_period)
        <= item.get("documentCreationDate").split("T")[0]
        < str(end_period)
    ]


def apply_filters_on_contract_agreement_attachment_list(
    unfiltered_list: list, filters: dict
) -> list:
    """Apply filters on contract attachment

    Args:
        unfiltered_list (list): unfiltered list to be filtered
        filters (dict): filters to be applied on contract attachment

    Returns:
        list: filtered list
    """
    if not filters:
        return unfiltered_list

    else:
        filters_dict_copy = filters.copy()

        if filters_dict_copy.get("status"):
            del filters_dict_copy["status"]

        if filters_dict_copy.get("contract_type"):
            del filters_dict_copy["contract_type"]

        filtered_list = unfiltered_list.copy()

        actual_keys = {
            "document_label": "documentLabel",
            "sales_region": "salesRegion",
            "sales_country": "salesCountry",
        }

        for filter_key, filter_params in filters_dict_copy.items():
            values = filter_params.get("value")
            filter_type = filter_params.get("filter_type")
            if filter_key == "original_file_name":
                filtered_list = apply_original_file_name_filter(filtered_list, values)

            elif filter_key == "document_creation_date":
                start_period = filter_params.get("start_period")
                end_period = filter_params.get("end_period")
                filtered_list = apply_document_creation_date_filter(
                    filtered_list, start_period, end_period
                )
            else:
                if filter_type == "keep":
                    filtered_list = [
                        item
                        for item in filtered_list
                        if not (current_value := item.get(actual_keys.get(filter_key)))
                        or current_value.lower()
                        in {filter_value.lower() for filter_value in values}
                    ]
                elif filter_type == "remove":
                    filtered_list = [
                        item
                        for item in filtered_list
                        if not (current_value := item.get(actual_keys.get(filter_key)))
                        or current_value.lower()
                        not in {filter_value.lower() for filter_value in values}
                    ]
        return filtered_list


def get_llm_and_embeddings_configs(zone: str) -> tuple:
    """Configure and return LLM and embeddings models based on the specified zone.

    Args:
        zone (str): String identifier for the zone ("eqt" or "ftsa")

    Returns:
        tuple: (llm, embeddings) configured models
    """
    zone = zone.lower()

    if zone == "eqt":
        return _configure_azure_models()
    elif zone == "ftsa":
        return _configure_lighton_models()


def _configure_azure_models():
    azure_settings = configure_azure_settings(
        chat_model=CHAT_MODEL_AZURE, embedding_model=EMBEDDING_MODEL_AZURE
    )

    llm = AzureChatOpenAI(
        openai_api_key=azure_settings["azure_openai_key"],
        azure_endpoint=azure_settings["azure_openai_endpoint"],
        openai_api_version=azure_settings["azure_openai_chat_api_version"],
        azure_deployment=azure_settings["azure_openai_chat_deployment"],
        model=CHAT_MODEL_AZURE,
        temperature=TEMPERATURE,
        timeout=120,
        max_retries=3,
    )

    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=azure_settings["azure_openai_key"],
        azure_endpoint=azure_settings["azure_openai_endpoint"],
        openai_api_version=azure_settings["azure_openai_embedding_api_version"],
        azure_deployment=azure_settings["azure_openai_embedding_deployment"],
        model=EMBEDDING_MODEL_AZURE,
        max_retries=3,
        retry_min_seconds=20,
    )

    return llm, embeddings


def _configure_lighton_models():
    lighton_settings = configure_lighton_settings(
        chat_model=CHAT_MODEL_LIGHTON, embedding_model=EMBEDDING_MODEL_LIGHTON
    )

    llm = ChatOpenAI(
        openai_api_key=lighton_settings["lighton_key"],
        openai_api_base=lighton_settings["lighton_endpoint"],
        model_name=CHAT_MODEL_LIGHTON,
    )

    embeddings = MultilingualE5LargeEmbeddings(
        api_url=lighton_settings["lighton_endpoint"],
        api_key=lighton_settings["lighton_key"],
    )

    return llm, embeddings
