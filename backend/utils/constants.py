"""
This module defines constants for the batch processing.
"""

# The name of the chat model to use: gpt-4o
CHAT_MODEL_AZURE = "gpt-4o"

# The name of the chat model to use:
CHAT_MODEL_LIGHTON = "alfred-4"

# The name of the embedding model to use: text-embedding-ada-002, text-embedding-3-large
EMBEDDING_MODEL_AZURE = "text-embedding-3-large"

# The name of the embedding model to use:
EMBEDDING_MODEL_LIGHTON = "multilingual-e5-large"

# multilingual-e5-large price can be calculated once we have it.
EMBEDDING_PRICE = {
    "text-embedding-ada-002": 0.0001 / 1000,
    "text-embedding-3-large": 0.00013 / 1000,
    "multilingual-e5-large": 1 / 1000000,
}

# Controls the randomness in the generation of responses by the chat model.
TEMPERATURE = 0

# number of retrieved documents from the vector database
K = 10

# Number of characters to include in each chunk of text.
CHUNK_SIZE = 4000
# CHUNK_SIZE = 500

# Number of characters that overlap between consecutive chunks.
CHUNK_OVERLAP = 500
# CHUNK_OVERLAP = 100

# TODO
# azure api limit for embedding model (tokens per minute)
EMBEDDING_MODEL_API_LIMIT = 5000
CHARACTERS_PER_TOKEN = 3
