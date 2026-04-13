
import pytest
from unittest.mock import patch, MagicMock
import io

@pytest.fixture
def valid_pdf_file():
    """
    Fixture that returns a file-like object representing a valid PDF.
    """
    # Simulate a PDF file using BytesIO
    return io.BytesIO(b'%PDF-1.4\n%Mock PDF content\n%%EOF')

@pytest.fixture
def expected_summary_structure():
    """
    Fixture that returns the expected summary structure.
    """
    return {
        "one_liner": "This is a one-liner summary.",
        "paragraph": "This is a paragraph summary of the PDF content.",
        "full": "This is a full, detailed summary of the PDF content."
    }

@pytest.fixture
def mock_parse_pdf():
    """
    Fixture that mocks the PDF parsing function.
    """
    with patch("your_module.parse_pdf") as mock_parse:
        mock_parse.return_value = "Extracted PDF text content."
        yield mock_parse

@pytest.fixture
def mock_llm_summarize(expected_summary_structure):
    """
    Fixture that mocks the LLM summarization function.
    """
    with patch("your_module.llm_summarize") as mock_llm:
        mock_llm.return_value = {
            "summary": expected_summary_structure
        }
        yield mock_llm

@pytest.fixture
def agent():
    """
    Fixture that returns an instance of the agent under test.
    Replace 'YourAgent' with the actual class name.
    """
    from your_module import YourAgent
    return YourAgent()

def build_expected_response(expected_summary_structure):
    return {
        "success": True,
        "result": {
            "summary": expected_summary_structure
        }
    }

def build_error_response(message):
    return {
        "success": False,
        "error": message
    }

def simulate_upload_and_summarize(agent, pdf_file, user_consent, task_type, additional_params=None):
    """
    Helper to simulate the upload and summarize workflow.
    """
    # Replace with the actual method signature of your agent
    return agent.upload_and_summarize(
        file=pdf_file,
        user_consent=user_consent,
        task_type=task_type,
        additional_params=additional_params or {}
    )

def test_functional_upload_pdf_and_get_summary(
    agent,
    valid_pdf_file,
    expected_summary_structure,
    mock_parse_pdf,
    mock_llm_summarize
):
    """
    Functional test: Validates that uploading a supported PDF file with user consent and task_type 'summary'
    returns a hierarchical summary as specified.
    """
    response = simulate_upload_and_summarize(
        agent=agent,
        pdf_file=valid_pdf_file,
        user_consent=True,
        task_type='summary'
    )
    assert response['success'] is True, "Response should indicate success"
    assert 'summary' in response['result'], "Result should contain 'summary'"
    summary = response['result']['summary']
    assert isinstance(summary, dict), "Summary should be a dictionary"
    for key in ['one_liner', 'paragraph', 'full']:
        assert key in summary, f"Summary should contain '{key}'"
        assert summary[key] == expected_summary_structure[key], f"Summary '{key}' does not match expected"

def test_functional_upload_pdf_and_get_summary_file_parsing_error(
    agent,
    valid_pdf_file
):
    """
    Functional test: Simulates a file parsing error and verifies error handling.
    """
    with patch("your_module.parse_pdf", side_effect=Exception("File parsing error")):
        response = simulate_upload_and_summarize(
            agent=agent,
            pdf_file=valid_pdf_file,
            user_consent=True,
            task_type='summary'
        )
        assert response['success'] is False, "Response should indicate failure"
        assert "parsing" in response['error'].lower(), "Error message should mention parsing"

def test_functional_upload_pdf_and_get_summary_llm_call_failure(
    agent,
    valid_pdf_file,
    mock_parse_pdf
):
    """
    Functional test: Simulates an LLM call failure and verifies error handling.
    """
    with patch("your_module.llm_summarize", side_effect=Exception("LLM call failure")):
        response = simulate_upload_and_summarize(
            agent=agent,
            pdf_file=valid_pdf_file,
            user_consent=True,
            task_type='summary'
        )
        assert response['success'] is False, "Response should indicate failure"
        assert "llm" in response['error'].lower() or "summarize" in response['error'].lower(), \
            "Error message should mention LLM or summarize"

def test_functional_upload_pdf_and_get_summary_missing_summary_in_result(
    agent,
    valid_pdf_file,
    mock_parse_pdf
):
    """
    Functional test: Simulates missing summary in LLM result and verifies error handling.
    """
    with patch("your_module.llm_summarize", return_value={}):
        response = simulate_upload_and_summarize(
            agent=agent,
            pdf_file=valid_pdf_file,
            user_consent=True,
            task_type='summary'
        )
        assert response['success'] is False, "Response should indicate failure"
        assert "summary" in response['error'].lower(), "Error message should mention missing summary"
