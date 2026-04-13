
import os
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class AgentConfig:
    """
    Configuration management for Document Insight & Summarization Agent.
    Handles environment variable loading, API key management, LLM config,
    domain-specific settings, validation, error handling, and fallbacks.
    """

    # LLM and domain settings
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4.1"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 2000
    LLM_SYSTEM_PROMPT = (
        "You are a professional Document Insight & Summarization Agent specializing in rapid comprehension of uploaded files in various formats (PDF, DOCX, TXT, CSV, YAML, JSON, code files). Your primary role is to read, parse, and summarize documents, extracting key insights, main topics, action items, and structural summaries. \n\n"
        "Instructions:\n\n"
        "- For each uploaded document, generate a hierarchical summary (one-liner, paragraph, and full summary).\n"
        "- Extract entities such as dates, names, numbers, decisions, and action items, providing a confidence score for each.\n"
        "- For long documents, generate a table of contents.\n"
        "- When users ask questions about the document, answer using only the content from the uploaded file or the retrieved knowledge base, citing the section or page.\n"
        "- In comparison mode, highlight differences between two documents in a structured format.\n"
        "- For each section or page, extract key-point bullets and cite the source location.\n"
        "- Never alter or modify the original document content.\n"
        "- Always handle unsupported formats gracefully, providing a clear and professional error message.\n"
        "- Ensure all outputs comply with data privacy, do not persist document content, and require user consent for processing.\n"
        "- If information is not found in the knowledge base or document, respond with a clear fallback message.\n\n"
        "Output Format:\n"
        "- Use structured text with clear headings for each output type (Summary, Entities, Table of Contents, Q&A, Comparison, Key Points).\n"
        "- Always include section/page citations and confidence levels where applicable.\n"
        "- For errors or unsupported formats, provide a concise and user-friendly message.\n\n"
        "Fallback Response:\n"
        "- If the requested information is not found in the document or knowledge base, respond: \"The requested information could not be found in the provided document or available knowledge base.\""
    )
    LLM_USER_PROMPT_TEMPLATE = (
        "Please upload your document(s) for analysis. Specify if you need a summary, entity extraction, table of contents, Q&A, or document comparison. "
        "For Q&A, ask your question clearly. For comparison, upload two files. All outputs will include citations and confidence levels."
    )
    LLM_FEW_SHOT_EXAMPLES = [
        "Summarize the attached PDF and extract all deadlines.",
        "Compare these two contract files and highlight the main differences."
    ]

    # Supported file types
    SUPPORTED_FILE_TYPES = ["PDF", "DOCX", "TXT", "CSV", "YAML", "JSON", "code"]

    # RAG / Azure AI Search settings
    RAG_ENABLED = True
    RAG_SERVICE = "azure_ai_search"
    RAG_EMBEDDING_MODEL = "text-embedding-ada-002"
    RAG_TOP_K = 5
    RAG_SEARCH_TYPE = "vector_semantic"

    # Required environment variables
    REQUIRED_ENV_VARS = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]

    # Fallbacks
    FALLBACK_RESPONSE = "The requested information could not be found in the provided document or available knowledge base."

    @classmethod
    def get_env(cls, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        value = os.getenv(key, default)
        if required and not value:
            logger.error(f"Missing required environment variable: {key}")
            raise ConfigError(f"Missing required environment variable: {key}")
        return value

    @classmethod
    def validate_env(cls):
        missing = [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            raise ConfigError(f"Missing required environment variables: {missing}")

    @classmethod
    def get_llm_config(cls):
        return {
            "provider": cls.LLM_PROVIDER,
            "model": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "system_prompt": cls.LLM_SYSTEM_PROMPT,
            "user_prompt_template": cls.LLM_USER_PROMPT_TEMPLATE,
            "few_shot_examples": cls.LLM_FEW_SHOT_EXAMPLES
        }

    @classmethod
    def get_rag_config(cls):
        return {
            "enabled": cls.RAG_ENABLED,
            "service": cls.RAG_SERVICE,
            "embedding_model": cls.RAG_EMBEDDING_MODEL,
            "top_k": cls.RAG_TOP_K,
            "search_type": cls.RAG_SEARCH_TYPE,
            "endpoint": cls.get_env("AZURE_SEARCH_ENDPOINT", required=True),
            "api_key": cls.get_env("AZURE_SEARCH_API_KEY", required=True),
            "index_name": cls.get_env("AZURE_SEARCH_INDEX_NAME", required=True),
            "embedding_deployment": cls.get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", required=True)
        }

    @classmethod
    def get_openai_config(cls):
        return {
            "endpoint": cls.get_env("AZURE_OPENAI_ENDPOINT", required=True),
            "api_key": cls.get_env("AZURE_OPENAI_API_KEY", required=True),
            "embedding_deployment": cls.get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", required=True)
        }

    @classmethod
    def get_supported_file_types(cls):
        return cls.SUPPORTED_FILE_TYPES

    @classmethod
    def get_fallback_response(cls):
        return cls.FALLBACK_RESPONSE

    @classmethod
    def validate(cls):
        """
        Validate all required environment variables and settings.
        """
        try:
            cls.validate_env()
        except Exception as e:
            logger.error(f"Agent configuration validation failed: {e}")
            raise

# Example usage:
# try:
#     AgentConfig.validate()
#     llm_cfg = AgentConfig.get_llm_config()
#     rag_cfg = AgentConfig.get_rag_config()
# except ConfigError as ce:
#     print(f"Configuration error: {ce}")
#     exit(1)
