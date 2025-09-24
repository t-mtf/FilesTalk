import json

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI

from utils.config import configure_azure_settings
from utils.constants import CHAT_MODEL_AZURE, EMBEDDING_MODEL_AZURE, TEMPERATURE


def compare_function(llm: AzureChatOpenAI, result1: str, result2: str) -> str:
    """
    This function compares two text results and determines their similarity level
    using a large language model as "identical", "nearly identical" or "distinct".

    Parameters:
        llm (AzureChatOpenAI): The model used for the comparison.
        result1 (str): The first text to be compared.
        result2 (str): The second text to be compared.

    Returns:
        str: The similarity level of the two texts.
    """
    if pd.isna(result1) and pd.isna(result2):
        return np.nan
    if pd.isna(result1) and "Assistant: NA" in result2:
        return np.nan

    context = f"Text1:\n{result1} \n Text2:\n{result2}"

    qa_system_prompt = f"""
You are an AI assistant specialized in contract analysis. You will be \
provided with the necessary context.

Please generate an answer based solely on the provided context. Ensure that your response is in English.

**Context:**

{context}
"""
    human_prompt = """
Please compare the provided texts in the context and determine whether the meaning is identical, nearly identical
or distinct. If the exact texts vary but still convey the same message, it should be considered as identical.
Return a JSON object with key "similarity" and a value of "identical", "nearly identical" or "distinct"

Here are some examples:

Text1:
This contract shall be governed by and interpreted in accordance with the laws of France. Any disputes arising from this contract will be resolved in the courts of France.

Text 2:
This contract will be governed by and interpreted following the laws of France. Any disputes that arise from this contract shall be settled in the courts of France.

Expected output:
{"similarity":"identical"}

Text1:
World Bank

**Clause 1.4**: "The inflation rate will be based on the most recent annual headline consumer price index (CPI) inflation information (series name “Headline Consumer Price Inflation”) for the United States, as published by the **World Bank** at https://www.worldbank.org/en/research/brief/inflation-database (the “Index”)."

Text 2:
World Bank, United States

Expected output:
{"similarity":"identical"}

Text1:
"annually

**Clause 5.6 Charges Adjustment**: upon each anniversary of the Effective Date (the “Revision Date”), Orange reserves the right to adjust its Charges by reference to the most recent annual headline consumer price index (CPI) inflation information...

Text 2:
Annually

Expected output:
{"similarity":"identical"}
"""

    messages = [
        ("system", qa_system_prompt),
        ("human", human_prompt),
    ]

    response = llm.invoke(
        messages,
    )

    try:
        result = json.loads(response.content)
        return result["similarity"]  # Return just the similarity value
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return response.content


def main():

    load_dotenv(find_dotenv())
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
        seed=123,
    )

    json_llm = llm.bind(response_format={"type": "json_object"})

    df1 = pd.read_excel("document_IZ_infl_sept_output_v10.xlsx")
    df2 = pd.read_excel("f6840e33-645a-4541-8785-e6ba12ac68e2_lighton.xlsx")
    # df2 = pd.read_excel("34e5e708-7afb-45d1-8910-91ab4d7bada7_azure.xlsx")
    llm_provider = "lighton"  # lighton

    df_merged = pd.merge(
        df1,
        df2,
        on=["ic01", "contract_number", "amendment_number", "filename"],
        suffixes=("_reference", f"_{llm_provider}"),
    )

    columns_to_compare = [
        "Inflation",
        "Comments",
        "Periodicity",
        "World Bank",
        "Language",
        "Signature date",
        "contract effective date",
        "amendment effective date",
    ]

    for column in columns_to_compare:
        df_merged[f"{column}_comparison"] = df_merged.apply(
            lambda row: compare_function(
                json_llm, row[f"{column}_reference"], row[f"{column}_{llm_provider}"]
            ),
            axis=1,
        )
    columns_to_keep = ["ic01", "contract_number", "amendment_number", "filename"]
    for column in columns_to_compare:
        columns_to_keep.extend(
            [f"{column}_reference", f"{column}_{llm_provider}", f"{column}_comparison"]
        )

    df_merged = df_merged[columns_to_keep]

    output_excel_file_path = f"comparison_{llm_provider}_reference_json.xlsx"
    df_merged.to_excel(output_excel_file_path, sheet_name="Sheet1", index=False)


def filter_results_azure():

    df = pd.read_excel("comparison_azure_reference_json.xlsx")

    df_filtered = df[
        (df["Inflation_reference"].notna()) | (df["Inflation_azure"].notna())
    ]

    df_filtered.to_excel("filtered_comparison_azure_reference_json.xlsx", index=False)


def filter_results_lighton():

    df = pd.read_excel("comparison_lighton_reference_json.xlsx")

    df.insert(
        df.columns.get_loc("Inflation_reference") + 1,
        "Inflation_lighton_original",
        df["Inflation_lighton"],
    )

    df.loc[
        df["Inflation_lighton"].astype(str).str.contains("Assistant: NA"),
        "Inflation_lighton",
    ] = np.nan

    df_filtered = df[
        (df["Inflation_reference"].notna()) | (df["Inflation_lighton"].notna())
    ]

    df_filtered.to_excel("filtered_comparison_lighton_reference_json.xlsx", index=False)


def select_rows():
    df_azure = pd.read_excel("filtered_comparison_azure_reference_json.xlsx")
    df_lighton = pd.read_excel("filtered_comparison_lighton_reference_json.xlsx")

    # Drop the specific row
    df_azure = df_azure[
        df_azure.filename != "MSA ICICI Pru - Orange Main Agreement.pdf"
    ]
    df_lighton = df_lighton[
        df_lighton.filename != "MSA ICICI Pru - Orange Main Agreement.pdf"
    ]

    identical_df_azure = df_azure[
        df_azure["Inflation_comparison"] == "identical"
    ].sample(n=9)
    nearly_identical_df_azure = df_azure[
        df_azure["Inflation_comparison"] == "nearly identical"
    ].sample(n=5)
    distinct_df_azure = df_azure[df_azure["Inflation_comparison"] == "distinct"].sample(
        n=12
    )

    filtered_azure = pd.concat(
        [identical_df_azure, nearly_identical_df_azure, distinct_df_azure]
    )

    filtered_lighton = pd.merge(
        df_lighton,
        filtered_azure[["ic01", "contract_number", "amendment_number", "filename"]],
        on=["ic01", "contract_number", "amendment_number", "filename"],
        how="inner",
    )

    filtered_azure.to_excel("filtered_azure.xlsx", index=False)
    filtered_lighton.to_excel("filtered_lighton.xlsx", index=False)


if __name__ == "__main__":
    select_rows()
