"""
Custom embeddings class for multilingual-e5-large model through Paradigm API.
"""

from typing import List

import requests
from langchain_core.embeddings import Embeddings


class MultilingualE5LargeEmbeddings(Embeddings):
    """
    Custom embeddings class for multilingual-e5-large model through Paradigm API.
    """

    def __init__(
        self, api_url: str, api_key: str, model_name: str = "multilingual-e5-large"
    ):
        """
        Initialize the embeddings class with API configuration.

        Args:
            api_url: The base URL of the API endpoint
            api_key: The API key for authentication
            model_name: The name of the model to use
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.embeddings_endpoint = f"{self.api_url}/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents using the Paradigm API.

        Args:
            texts: List of text strings to generate embeddings for

        Returns:
            List of embedding vectors, each represented as a list of floats
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        # Paradigm API expects a list of individual inputs, not a list of texts
        embeddings = []
        for text in texts:
            payload = {"model": self.model_name, "input": text}

            response = requests.post(
                self.embeddings_endpoint, headers=headers, json=payload, timeout=180
            )

            if response.status_code != 200:
                raise Exception(
                    f"API request failed: {response.status_code}: {response.text}"
                )

            response_json = response.json()

            embedding = response_json["data"][0]["embedding"]
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query text.

        Args:
            text: The query text to generate an embedding for

        Returns:
            The embedding vector as a list of floats
        """
        embeddings = self.embed_documents([text])
        return embeddings[0]
