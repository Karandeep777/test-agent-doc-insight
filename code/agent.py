try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import io
import sys
import time as _time
import asyncio
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field, model_validator
from loguru import logger

import openai
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from PyPDF2 import PdfReader
import docx
import pandas as pd
import yaml
import json

# Observability wrappers are injected by the runtime
# from observability import trace_step, trace_step_sync

SUPPORTED_FILE_TYPES = {
    "pdf": "PDF",
    "docx": "DOCX",
    "txt": "TXT",
    "csv": "CSV",
    "yaml": "YAML",
    "yml": "YAML",
    "json": "JSON",
    "py": "code",
    "js": "code",
    "java": "code",
    "c": "code",
    "cpp": "code",
    "cs": "code",
    "go": "code",
    "rb": "code",
    "rs": "code",
    "ts": "code",
    "sh": "code",
    "md": "code",
    "html": "code",
    "xml": "code",
    "css": "code",
    "ipynb": "code"
}

ENHANCED_SYSTEM_PROMPT = (
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
FALLBACK_RESPONSE = "The requested information could not be found in the provided document or available knowledge base."

# =========================
# Configuration Management
# =========================

class Config:
    @staticmethod
    def get_openai_api_key() -> str:
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not key:
            raise ValueError("AZURE_OPENAI_API_KEY not set in environment.")
        return key

    @staticmethod
    def get_openai_endpoint() -> str:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not set in environment.")
        return endpoint

    @staticmethod
    def get_openai_embedding_deployment() -> str:
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        if not deployment:
            raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT not set in environment.")
        return deployment

    @staticmethod
    def get_search_endpoint() -> str:
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_SEARCH_ENDPOINT not set in environment.")
        return endpoint

    @staticmethod
    def get_search_api_key() -> str:
        key = os.getenv("AZURE_SEARCH_API_KEY")
        if not key:
            raise ValueError("AZURE_SEARCH_API_KEY not set in environment.")
        return key

    @staticmethod
    def get_search_index_name() -> str:
        index = os.getenv("AZURE_SEARCH_INDEX_NAME")
        if not index:
            raise ValueError("AZURE_SEARCH_INDEX_NAME not set in environment.")
        return index

    @staticmethod
    def validate():
        # Optional: Validate all required keys
        try:
            Config.get_openai_api_key()
            Config.get_openai_endpoint()
            Config.get_openai_embedding_deployment()
            Config.get_search_endpoint()
            Config.get_search_api_key()
            Config.get_search_index_name()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

# =========================
# Utility Functions
# =========================

def get_file_extension(filename: str) -> str:
    ext = filename.split(".")[-1].lower()
    return ext

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_text(text: str) -> str:
    # Remove control characters, excessive whitespace, etc.
    return text.replace('\x00', '').strip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_pii(text: str) -> str:
    # Dummy PII masking for demo; replace with real implementation as needed
    # For demo, mask emails and phone numbers
    import re
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', '[EMAIL REDACTED]', text)
    text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', '[PHONE REDACTED]', text)
    return text

def get_file_type(filename: str) -> Optional[str]:
    ext = get_file_extension(filename)
    return SUPPORTED_FILE_TYPES.get(ext, None)

def is_supported_file_type(filename: str) -> bool:
    return get_file_type(filename) is not None

def get_temp_file_path(file: UploadFile) -> str:
    # Save file to a temp file and return path
    suffix = "." + get_file_extension(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        return tmp.name

# =========================
# Pydantic Models
# =========================

class UploadRequestModel(BaseModel):
    user_consent: bool = Field(..., description="User consent for processing")
    task_type: str = Field(..., description="Task type (summary, entities, toc, qa, compare, keypoints)")
    additional_params: Optional[dict] = Field(default_factory=dict)
    # file(s) handled via UploadFile

    @field_validator("task_type")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_task_type(cls, v):
        allowed = {"summary", "entities", "toc", "qa", "compare", "keypoints"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid task_type: {v}. Allowed: {allowed}")
        return v.lower()

    @model_validator(mode="after")
    def check_consent(self):
        if not self.user_consent:
            raise ValueError("User consent is required for processing.")
        return self

# =========================
# Base Service
# =========================

class BaseService:
    def __init__(self):
        self.logger = logger

    def log_event(self, event_type: str, details: Any):
        self.logger.info(f"[{event_type}] {details}")

    def log_error(self, error_type: str, details: Any):
        self.logger.error(f"[{error_type}] {details}")

# =========================
# UserConsentValidator
# =========================

class UserConsentValidator(BaseService):
    def validate_consent(self, user_consent: bool) -> bool:
        if not user_consent:
            self.log_error("CONSENT_REQUIRED", "User consent not provided.")
            raise ValueError("User consent is required for processing.")
        return True

# =========================
# FileTypeValidator
# =========================

class FileTypeValidator(BaseService):
    def validate_file_type(self, filename: str) -> str:
        file_type = get_file_type(filename)
        if not file_type:
            self.log_error("UNSUPPORTED_FORMAT", f"Unsupported file type: {filename}")
            raise ValueError(f"Unsupported file type: {filename}. Supported: {list(SUPPORTED_FILE_TYPES.values())}")
        return file_type

# =========================
# FileParser
# =========================

class FileParser(BaseService):
    def parse(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Returns: dict with keys: content (str), sections (list), pages (list), etc.
        """
        try:
            if file_type == "PDF":
                return self._parse_pdf(file_path)
            elif file_type == "DOCX":
                return self._parse_docx(file_path)
            elif file_type == "TXT":
                return self._parse_txt(file_path)
            elif file_type == "CSV":
                return self._parse_csv(file_path)
            elif file_type == "YAML":
                return self._parse_yaml(file_path)
            elif file_type == "JSON":
                return self._parse_json(file_path)
            elif file_type == "code":
                return self._parse_code(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            self.log_error("PARSING_ERROR", f"{file_type}: {e}")
            raise

    def _parse_pdf(self, file_path: str) -> Dict[str, Any]:
        content = ""
        pages = []
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    pages.append({"page": i + 1, "content": sanitize_text(text)})
                    content += text + "\n"
            return {"content": sanitize_text(content), "pages": pages}
        except Exception as e:
            self.log_error("PDF_PARSE_ERROR", str(e))
            raise

    def _parse_docx(self, file_path: str) -> Dict[str, Any]:
        content = ""
        sections = []
        try:
            doc = docx.Document(file_path)
            for i, para in enumerate(doc.paragraphs):
                text = para.text.strip()
                if text:
                    sections.append({"section": i + 1, "content": sanitize_text(text)})
                    content += text + "\n"
            return {"content": sanitize_text(content), "sections": sections}
        except Exception as e:
            self.log_error("DOCX_PARSE_ERROR", str(e))
            raise

    def _parse_txt(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                _obs_t0 = _time.time()
                content = f.read()
                try:
                    trace_tool_call(
                        tool_name='f.read',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(content)[:200] if content is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
            return {"content": sanitize_text(content)}
        except Exception as e:
            self.log_error("TXT_PARSE_ERROR", str(e))
            raise

    def _parse_csv(self, file_path: str) -> Dict[str, Any]:
        try:
            df = pd.read_csv(file_path)
            content = df.to_csv(index=False)
            return {"content": sanitize_text(content), "table": df.head(10).to_dict(orient="records")}
        except Exception as e:
            self.log_error("CSV_PARSE_ERROR", str(e))
            raise

    def _parse_yaml(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            content = yaml.dump(data)
            return {"content": sanitize_text(content), "yaml": data}
        except Exception as e:
            self.log_error("YAML_PARSE_ERROR", str(e))
            raise

    def _parse_json(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            content = json.dumps(data, indent=2)
            return {"content": sanitize_text(content), "json": data}
        except Exception as e:
            self.log_error("JSON_PARSE_ERROR", str(e))
            raise

    def _parse_code(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                _obs_t0 = _time.time()
                content = f.read()
                try:
                    trace_tool_call(
                        tool_name='f.read',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(content)[:200] if content is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
            return {"content": sanitize_text(content)}
        except Exception as e:
            self.log_error("CODE_PARSE_ERROR", str(e))
            raise

# =========================
# LLMClient
# =========================

class LLMClient(BaseService):
    def __init__(self, model: str = "gpt-4.1", temperature: float = 0.7, max_tokens: int = 2000):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def get_client(self):
        if self._client is None:
            api_key = Config.get_openai_api_key()
            endpoint = Config.get_openai_endpoint()
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def call_llm(self, prompt: str, parameters: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        """
        Calls the LLM with the given prompt and parameters.
        """
        client = self.get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": ENHANCED_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if parameters:
            params.update(parameters)

        _t0 = _time.time()
        try:
            async with trace_step(
                "llm_call", step_type="llm_call",
                decision_summary="Call LLM to produce a reply",
                output_fn=lambda r: f"length={len(r) if r else 0}",
            ) as step:
                response = await client.chat.completions.create(**params)
                content = response.choices[0].message.content
                step.capture(content)
                try:
                    trace_model_call(
                        provider="openai",
                        model_name=self.model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else ""
                    )
                except Exception:
                    pass
                return content
        except Exception as e:
            self.log_error("LLM_CALL_ERROR", str(e))
            raise

# =========================
# RAGRetriever
# =========================

class RAGRetriever(BaseService):
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        self._search_client = None
        self._embedding_client = None

    def get_search_client(self):
        if self._search_client is None:
            endpoint = Config.get_search_endpoint()
            index_name = Config.get_search_index_name()
            api_key = Config.get_search_api_key()
            self._search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key)
            )
        return self._search_client

    def get_embedding_client(self):
        if self._embedding_client is None:
            api_key = Config.get_openai_api_key()
            endpoint = Config.get_openai_endpoint()
            self._embedding_client = openai.AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )
        return self._embedding_client

    @trace_agent(agent_name='Document Insight & Summarization Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve(self, query: str, document_context: Optional[List[str]] = None) -> List[str]:
        """
        Retrieves top-K relevant chunks from Azure AI Search for the given query.
        """
        async with trace_step(
            "rag_retrieve", step_type="tool_call",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r)}",
        ) as step:
            embedding_client = self.get_embedding_client()
            search_client = self.get_search_client()
            embedding_resp = embedding_client.embeddings.create(
                input=query,
                model=Config.get_openai_embedding_deployment()
            )
            vector_query = VectorizedQuery(
                vector=embedding_resp.data[0].embedding,
                k_nearest_neighbors=self.top_k,
                fields="vector"
            )
            results = search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=self.top_k,
                select=["chunk", "title"]
            )
            context_chunks = [r["chunk"] for r in results if r.get("chunk")]
            step.capture(context_chunks)
            return context_chunks

# =========================
# NLPSummarizer
# =========================

class NLPSummarizer(BaseService):
    def __init__(self, llm_client: LLMClient):
        super().__init__()
        self.llm_client = llm_client

    async def summarize(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        content = parsed_content.get("content", "")
        prompt = (
            "Summarize the following document in three levels: "
            "1. One-liner summary\n"
            "2. Paragraph summary\n"
            "3. Full summary\n"
            "Include citations to section/page where possible."
            f"\n\nDocument Content:\n{content[:5000]}"
        )
        retries = 0
        while retries < 3:
            try:
                async with trace_step(
                    "summarize_document", step_type="llm_call",
                    decision_summary="Generate hierarchical summaries",
                    output_fn=lambda r: f"summary_length={len(str(r))}",
                ) as step:
                    result = await self.llm_client.call_llm(prompt)
                    step.capture(result)
                    return {"summary": result}
            except Exception as e:
                self.log_error("SUMMARY_ERROR", f"Attempt {retries+1}: {e}")
                retries += 1
                await asyncio.sleep(2 ** retries)
        return {"summary": FALLBACK_RESPONSE}

# =========================
# EntityExtractor
# =========================

class EntityExtractor(BaseService):
    def __init__(self, llm_client: LLMClient):
        super().__init__()
        self.llm_client = llm_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def extract_entities(self, parsed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = parsed_content.get("content", "")
        prompt = (
            "Extract all entities (dates, names, numbers, decisions, action items) from the following document. "
            "For each entity, provide a confidence score (0-1) and cite the section/page."
            f"\n\nDocument Content:\n{content[:5000]}"
        )
        retries = 0
        while retries < 3:
            try:
                async with trace_step(
                    "extract_entities", step_type="llm_call",
                    decision_summary="Extract entities with confidence",
                    output_fn=lambda r: f"entities={len(str(r))}",
                ) as step:
                    result = await self.llm_client.call_llm(prompt)
                    step.capture(result)
                    # Try to parse as JSON, fallback to text
                    try:
                        entities = json.loads(result)
                        return entities
                    except Exception:
                        return [{"entities": result}]
            except Exception as e:
                self.log_error("EXTRACTION_FAILURE", f"Attempt {retries+1}: {e}")
                retries += 1
                await asyncio.sleep(2 ** retries)
        return []

# =========================
# TOCGenerator
# =========================

class TOCGenerator(BaseService):
    def __init__(self, llm_client: LLMClient):
        super().__init__()
        self.llm_client = llm_client

    async def generate_toc(self, parsed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = parsed_content.get("content", "")
        prompt = (
            "Generate a structured table of contents for the following document. "
            "Include section/page references."
            f"\n\nDocument Content:\n{content[:5000]}"
        )
        try:
            async with trace_step(
                "generate_toc", step_type="llm_call",
                decision_summary="Generate table of contents",
                output_fn=lambda r: f"toc_length={len(str(r))}",
            ) as step:
                result = await self.llm_client.call_llm(prompt)
                step.capture(result)
                try:
                    toc = json.loads(result)
                    return toc
                except Exception:
                    return [{"toc": result}]
        except Exception as e:
            self.log_error("TOC_ERROR", str(e))
            return []

# =========================
# DocumentComparator
# =========================

class DocumentComparator(BaseService):
    def __init__(self, llm_client: LLMClient):
        super().__init__()
        self.llm_client = llm_client

    async def compare(self, parsed_content_1: Dict[str, Any], parsed_content_2: Dict[str, Any]) -> Dict[str, Any]:
        content1 = parsed_content_1.get("content", "")
        content2 = parsed_content_2.get("content", "")
        prompt = (
            "Compare the following two documents and highlight the main differences in a structured format. "
            "Include citations to section/page where possible.\n\n"
            f"Document 1:\n{content1[:2500]}\n\nDocument 2:\n{content2[:2500]}"
        )
        try:
            async with trace_step(
                "compare_documents", step_type="llm_call",
                decision_summary="Compare two documents",
                output_fn=lambda r: f"comparison_length={len(str(r))}",
            ) as step:
                result = await self.llm_client.call_llm(prompt)
                step.capture(result)
                return {"comparison": result}
        except Exception as e:
            self.log_error("COMPARISON_ERROR", str(e))
            return {"comparison": FALLBACK_RESPONSE}

# =========================
# KeyPointExtractor
# =========================

class KeyPointExtractor(BaseService):
    def __init__(self, llm_client: LLMClient):
        super().__init__()
        self.llm_client = llm_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def extract_key_points(self, parsed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = parsed_content.get("content", "")
        prompt = (
            "Extract key-point bullets for each section or page of the following document. "
            "Cite the source location for each key point."
            f"\n\nDocument Content:\n{content[:5000]}"
        )
        try:
            async with trace_step(
                "extract_key_points", step_type="llm_call",
                decision_summary="Extract key points per section/page",
                output_fn=lambda r: f"keypoints_length={len(str(r))}",
            ) as step:
                result = await self.llm_client.call_llm(prompt)
                step.capture(result)
                try:
                    keypoints = json.loads(result)
                    return keypoints
                except Exception:
                    return [{"keypoints": result}]
        except Exception as e:
            self.log_error("KEYPOINTS_ERROR", str(e))
            return []

# =========================
# SecurityComplianceManager
# =========================

class SecurityComplianceManager(BaseService):
    def enforce_policies(self, output: Any) -> Any:
        # Mask PII, ensure no persistent storage, etc.
        if isinstance(output, str):
            return mask_pii(output)
        elif isinstance(output, dict):
            return {k: self.enforce_policies(v) for k, v in output.items()}
        elif isinstance(output, list):
            return [self.enforce_policies(item) for item in output]
        else:
            return output

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def mask_pii(self, text: str) -> str:
        return mask_pii(text)

# =========================
# ErrorHandler
# =========================

class ErrorHandler(BaseService):
    def handle_error(self, error_code: str, context: Optional[Any] = None) -> Dict[str, Any]:
        error_map = {
            "UNSUPPORTED_FORMAT": {
                "message": "The uploaded file format is not supported. Please upload PDF, DOCX, TXT, CSV, YAML, JSON, or code files.",
                "tips": "Check the file extension and try again."
            },
            "CONSENT_REQUIRED": {
                "message": "User consent is required for processing files.",
                "tips": "Please provide explicit consent before uploading."
            },
            "EXTRACTION_FAILURE": {
                "message": "Failed to extract entities from the document.",
                "tips": "Try again or upload a clearer document."
            },
            "SUMMARY_ERROR": {
                "message": "Failed to generate summary.",
                "tips": "Try again or upload a different document."
            },
            "COMPARISON_ERROR": {
                "message": "Failed to compare documents.",
                "tips": "Ensure both files are supported and not empty."
            },
            "PARSING_ERROR": {
                "message": "Failed to parse the uploaded file.",
                "tips": "Ensure the file is not corrupted and is in a supported format."
            },
            "TOC_ERROR": {
                "message": "Failed to generate table of contents.",
                "tips": "Try again or upload a different document."
            },
            "KEYPOINTS_ERROR": {
                "message": "Failed to extract key points.",
                "tips": "Try again or upload a different document."
            },
            "GENERIC_ERROR": {
                "message": "An unexpected error occurred.",
                "tips": "Try again later or contact support."
            }
        }
        entry = error_map.get(error_code, error_map["GENERIC_ERROR"])
        self.log_error(error_code, context)
        return {
            "success": False,
            "error_code": error_code,
            "error_message": entry["message"],
            "tips": entry["tips"]
        }

# =========================
# AuditLogger
# =========================

class AuditLogger(BaseService):
    def log_event(self, event_type: str, details: Any):
        self.logger.info(f"[AUDIT][{event_type}] {details}")

# =========================
# Main Agent
# =========================

class DocumentInsightSummarizationAgent(BaseService):
    def __init__(self):
        super().__init__()
        self.user_consent_validator = UserConsentValidator()
        self.file_type_validator = FileTypeValidator()
        self.file_parser = FileParser()
        self.llm_client = LLMClient()
        self.rag_retriever = RAGRetriever(top_k=5)
        self.summarizer = NLPSummarizer(self.llm_client)
        self.entity_extractor = EntityExtractor(self.llm_client)
        self.toc_generator = TOCGenerator(self.llm_client)
        self.comparator = DocumentComparator(self.llm_client)
        self.keypoint_extractor = KeyPointExtractor(self.llm_client)
        self.security_manager = SecurityComplianceManager()
        self.error_handler = ErrorHandler()
        self.audit_logger = AuditLogger()

    @trace_agent(agent_name='Document Insight & Summarization Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_upload(
        self,
        file: UploadFile,
        user_consent: bool,
        task_type: str,
        additional_params: Optional[dict] = None,
        file2: Optional[UploadFile] = None
    ) -> Dict[str, Any]:
        """
        Entry point for handling uploaded files, validating consent and file type, and orchestrating downstream processing.
        """
        async with trace_step(
            "process_upload", step_type="process",
            decision_summary="Validate input and orchestrate processing",
            output_fn=lambda r: f"success={r.get('success', False)}",
        ) as step:
            try:
                self.user_consent_validator.validate_consent(user_consent)
                file_type = self.file_type_validator.validate_file_type(file.filename)
                file_path = get_temp_file_path(file)
                parsed_content = self.file_parser.parse(file_path, file_type)
                os.remove(file_path)
                parsed_content = self.security_manager.enforce_policies(parsed_content)
                self.audit_logger.log_event("FILE_PARSED", {"filename": file.filename, "file_type": file_type})

                if task_type == "summary":
                    result = await self.summarize_document(parsed_content)
                elif task_type == "entities":
                    result = await self.extract_entities(parsed_content)
                elif task_type == "toc":
                    result = await self.generate_toc(parsed_content)
                elif task_type == "qa":
                    question = (additional_params or {}).get("question", "")
                    if not question:
                        return self.error_handler.handle_error("GENERIC_ERROR", "No question provided for Q&A.")
                    result = await self.answer_question(question, parsed_content, document_context=None)
                elif task_type == "compare":
                    if not file2:
                        return self.error_handler.handle_error("GENERIC_ERROR", "Second file required for comparison.")
                    file_type2 = self.file_type_validator.validate_file_type(file2.filename)
                    file_path2 = get_temp_file_path(file2)
                    parsed_content2 = self.file_parser.parse(file_path2, file_type2)
                    os.remove(file_path2)
                    parsed_content2 = self.security_manager.enforce_policies(parsed_content2)
                    result = await self.compare_documents(parsed_content, parsed_content2)
                elif task_type == "keypoints":
                    result = await self.extract_key_points(parsed_content)
                else:
                    return self.error_handler.handle_error("GENERIC_ERROR", f"Unknown task_type: {task_type}")

                step.capture({"success": True, "result": result})
                return {"success": True, "result": result}
            except Exception as e:
                tb = traceback.format_exc()
                self.log_error("PROCESS_UPLOAD_ERROR", f"{e}\n{tb}")
                return self.error_handler.handle_error("GENERIC_ERROR", str(e))

    async def summarize_document(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        return await self.summarizer.summarize(parsed_content)

    @trace_agent(agent_name='Document Insight & Summarization Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def extract_entities(self, parsed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        return await self.entity_extractor.extract_entities(parsed_content)

    async def generate_toc(self, parsed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        return await self.toc_generator.generate_toc(parsed_content)

    @trace_agent(agent_name='Document Insight & Summarization Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def answer_question(
        self,
        question: str,
        parsed_content: Dict[str, Any],
        document_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Answers user Q&A using document and/or retrieved knowledge base content.
        """
        # Retrieve context from Azure AI Search (RAG)
        async with trace_step(
            "answer_question", step_type="llm_call",
            decision_summary="Answer user Q&A using RAG",
            output_fn=lambda r: f"answer_length={len(str(r))}",
        ) as step:
            try:
                context_chunks = await self.rag_retriever.retrieve(question, document_context)
                context_text = "\n\n".join(context_chunks)
                doc_text = parsed_content.get("content", "")
                prompt = (
                    f"User Question: {question}\n\n"
                    f"Relevant Knowledge Base Chunks:\n{context_text[:4000]}\n\n"
                    f"Document Content:\n{doc_text[:4000]}\n\n"
                    "Answer the question using only the provided content. Cite section/page. "
                    f"If the answer is not found, reply: \"{FALLBACK_RESPONSE}\""
                )
                answer = await self.llm_client.call_llm(prompt)
                if FALLBACK_RESPONSE.lower() in answer.lower():
                    return {
                        "answer": FALLBACK_RESPONSE,
                        "citations": [],
                        "confidence": 0.0
                    }
                step.capture({"answer": answer})
                return {
                    "answer": answer,
                    "citations": [],  # Could extract citations from answer if needed
                    "confidence": 1.0
                }
            except Exception as e:
                self.log_error("QA_ERROR", str(e))
                return {
                    "answer": FALLBACK_RESPONSE,
                    "citations": [],
                    "confidence": 0.0
                }

    async def compare_documents(
        self,
        parsed_content_1: Dict[str, Any],
        parsed_content_2: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self.comparator.compare(parsed_content_1, parsed_content_2)

    @trace_agent(agent_name='Document Insight & Summarization Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def extract_key_points(self, parsed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        return await self.keypoint_extractor.extract_key_points(parsed_content)

# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Document Insight & Summarization Agent",
    description="API for rapid document understanding, summarization, entity extraction, TOC, Q&A, and comparison.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = DocumentInsightSummarizationAgent()

# ========== Exception Handlers ==========

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "error_message": str(exc),
            "tips": "Check your input fields and JSON formatting."
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": "HTTP_ERROR",
            "error_message": exc.detail,
            "tips": "Check your request and try again."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "GENERIC_ERROR",
            "error_message": str(exc),
            "tips": "Try again later or contact support."
        }
    )

# ========== API Endpoints ==========

@app.post("/upload", summary="Upload a document for processing")
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    user_consent: bool = Form(..., description="User consent for processing"),
    task_type: str = Form(..., description="Task type (summary, entities, toc, qa, compare, keypoints)"),
    additional_params: Optional[str] = Form(None, description="Additional parameters as JSON string"),
    file2: Optional[UploadFile] = File(None, description="Second file for comparison (if needed)")
):
    """
    Receives file uploads, validates user consent, checks file type, and initiates processing.
    """
    try:
        # Parse additional_params JSON if provided
        params = {}
        if additional_params:
            try:
                params = json.loads(additional_params)
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "error_code": "MALFORMED_JSON",
                        "error_message": f"Malformed JSON in additional_params: {e}",
                        "tips": "Ensure additional_params is a valid JSON string."
                    }
                )
        # Validate input
        try:
            UploadRequestModel(
                user_consent=user_consent,
                task_type=task_type,
                additional_params=params
            )
        except ValidationError as ve:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "error_code": "VALIDATION_ERROR",
                    "error_message": str(ve),
                    "tips": "Check your input fields and JSON formatting."
                }
            )
        result = await agent.process_upload(
            file=file,
            user_consent=user_consent,
            task_type=task_type,
            additional_params=params,
            file2=file2
        )
        return JSONResponse(status_code=200 if result.get("success", False) else 400, content=result)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_code": "GENERIC_ERROR",
                "error_message": str(e),
                "tips": "Try again later or contact support."
            }
        )

@app.get("/health", summary="Health check")
async def health_check():
    return {"success": True, "status": "ok"}

# =========================
# Main Entry Point
# =========================



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting Document Insight & Summarization Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())