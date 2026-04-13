
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_file_parser():
    """
    Fixture to mock the file parsing logic for PDF and DOCX files.
    """
    with patch('your_module.file_parser') as mock_parser:
        # Simulate successful parsing for both files
        mock_parser.parse_file.side_effect = lambda file_obj: f"parsed_content_of_{file_obj.name}"
        yield mock_parser

@pytest.fixture
def mock_llm_compare():
    """
    Fixture to mock the LLM comparison endpoint.
    """
    with patch('your_module.llm_compare') as mock_llm:
        # Simulate a successful LLM comparison response
        mock_llm.return_value = {
            "comparison": "Difference between parsed_content_of_file1.pdf and parsed_content_of_file2.docx"
        }
        yield mock_llm

@pytest.fixture
def agent():
    """
    Fixture to provide the agent or function under test.
    Replace 'your_module.Agent' with the actual import.
    """
    from your_module import Agent  # Replace with actual import
    return Agent()

def make_mock_file(name: str):
    """
    Helper to create a mock file-like object with a name attribute.
    """
    file_obj = MagicMock()
    file_obj.name = name
    return file_obj

def test_integration_document_comparison(agent, mock_file_parser, mock_llm_compare):
    """
    Integration test:
    Validates that uploading two supported files with task_type 'compare' returns a structured comparison highlighting differences.
    """
    # Arrange
    file1 = make_mock_file("file1.pdf")
    file2 = make_mock_file("file2.docx")
    user_consent = True
    task_type = "compare"
    additional_params = None

    # Act
    # Replace 'compare_documents' with the actual method to invoke
    response = agent.compare_documents(
        files=[file1, file2],
        user_consent=user_consent,
        task_type=task_type,
        additional_params=additional_params
    )

    # Assert
    assert response['success'] is True, "Expected success=True in response"
    assert 'comparison' in response['result'], "Expected 'comparison' key in result"

    # Error scenario: File parsing error for either file
    mock_file_parser.parse_file.side_effect = Exception("Parsing failed")
    with pytest.raises(Exception) as excinfo:
        agent.compare_documents(
            files=[file1, file2],
            user_consent=user_consent,
            task_type=task_type,
            additional_params=additional_params
        )
    assert "Parsing failed" in str(excinfo.value)

    # Error scenario: LLM call failure
    mock_file_parser.parse_file.side_effect = lambda file_obj: f"parsed_content_of_{file_obj.name}"  # Reset
    mock_llm_compare.side_effect = Exception("LLM service unavailable")
    with pytest.raises(Exception) as excinfo:
        agent.compare_documents(
            files=[file1, file2],
            user_consent=user_consent,
            task_type=task_type,
            additional_params=additional_params
        )
    assert "LLM service unavailable" in str(excinfo.value)

    # Error scenario: Missing comparison in result
    mock_llm_compare.side_effect = None
    mock_llm_compare.return_value = {}  # No 'comparison' key
    response = agent.compare_documents(
        files=[file1, file2],
        user_consent=user_consent,
        task_type=task_type,
        additional_params=additional_params
    )
    assert response['success'] is True
    assert 'comparison' not in response['result'], "Expected missing 'comparison' key in result"

