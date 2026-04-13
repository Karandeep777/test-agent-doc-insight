
import pytest
from unittest.mock import patch, MagicMock
import io

@pytest.fixture
def sample_docx_file():
    """
    Fixture that returns a BytesIO object simulating a valid DOCX file.
    """
    # Simulate a DOCX file as bytes (not a real DOCX, but sufficient for mocking)
    return io.BytesIO(b"PK\x03\x04FakeDOCXContent")

@pytest.fixture
def mock_entity_extractor():
    """
    Fixture that returns a mock entity extraction function.
    """
    def _extract_entities(file_obj, user_consent, task_type, additional_params=None):
        # Simulate successful extraction
        return {
            "success": True,
            "result": [
                {
                    "entity": "Acme Corp",
                    "confidence": 0.97,
                    "citation": "Page 1"
                },
                {
                    "entity": "John Doe",
                    "confidence": 0.91,
                    "citation": "Page 2"
                }
            ]
        }
    return MagicMock(side_effect=_extract_entities)

@pytest.fixture
def mock_llm_failure():
    """
    Fixture that simulates an LLM call failure.
    """
    def _extract_entities(*args, **kwargs):
        return {
            "success": False,
            "error": "LLM service unavailable"
        }
    return MagicMock(side_effect=_extract_entities)

@pytest.fixture
def mock_malformed_entities():
    """
    Fixture that simulates a malformed entities output.
    """
    def _extract_entities(*args, **kwargs):
        return {
            "success": True,
            "result": [
                {
                    "entity": "Acme Corp",
                    # Missing confidence and citation
                }
            ]
        }
    return MagicMock(side_effect=_extract_entities)

@pytest.fixture
def mock_file_parsing_error():
    """
    Fixture that simulates a file parsing error.
    """
    def _extract_entities(*args, **kwargs):
        return {
            "success": False,
            "error": "File parsing error"
        }
    return MagicMock(side_effect=_extract_entities)

def get_entity_extraction_response(file_obj, user_consent, task_type, additional_params=None, extractor=None):
    """
    Helper function to simulate the entity extraction API.
    """
    # In real code, this would call the actual entity extraction logic.
    # Here, we call the provided extractor mock.
    return extractor(file_obj, user_consent, task_type, additional_params)

class TestEntityExtractionFunctional:
    def test_functional_entity_extraction_from_docx(self, sample_docx_file, mock_entity_extractor):
        """
        Functional test: Checks that entity extraction works for a DOCX file,
        returning entities with confidence scores and citations.
        """
        response = get_entity_extraction_response(
            file_obj=sample_docx_file,
            user_consent=True,
            task_type='entities',
            additional_params=None,
            extractor=mock_entity_extractor
        )
        assert response['success'] is True, "Expected success=True in response"
        assert isinstance(response['result'], list), "Expected result to be a list"
        for entity in response['result']:
            assert 'confidence' in entity, "Entity missing confidence score"
            assert 'citation' in entity, "Entity missing citation"

    def test_entity_extraction_file_parsing_error(self, sample_docx_file, mock_file_parsing_error):
        """
        Functional test: Simulates a file parsing error during entity extraction.
        """
        response = get_entity_extraction_response(
            file_obj=sample_docx_file,
            user_consent=True,
            task_type='entities',
            additional_params=None,
            extractor=mock_file_parsing_error
        )
        assert response['success'] is False, "Expected success=False on file parsing error"
        assert "parsing" in response.get("error", ""), "Expected file parsing error message"

    def test_entity_extraction_llm_call_failure(self, sample_docx_file, mock_llm_failure):
        """
        Functional test: Simulates an LLM call failure during entity extraction.
        """
        response = get_entity_extraction_response(
            file_obj=sample_docx_file,
            user_consent=True,
            task_type='entities',
            additional_params=None,
            extractor=mock_llm_failure
        )
        assert response['success'] is False, "Expected success=False on LLM failure"
        assert "LLM" in response.get("error", ""), "Expected LLM error message"

    def test_entity_extraction_malformed_entities_output(self, sample_docx_file, mock_malformed_entities):
        """
        Functional test: Simulates a malformed entities output (missing confidence/citation).
        """
        response = get_entity_extraction_response(
            file_obj=sample_docx_file,
            user_consent=True,
            task_type='entities',
            additional_params=None,
            extractor=mock_malformed_entities
        )
        assert response['success'] is True, "Expected success=True even if output is malformed"
        assert isinstance(response['result'], list), "Expected result to be a list"
        for entity in response['result']:
            assert 'confidence' in entity and 'citation' in entity, "Malformed entity: missing confidence or citation"

