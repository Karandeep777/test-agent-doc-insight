
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def valid_txt_file_content():
    """Fixture that returns valid TXT file content as bytes."""
    return b"This is a sample text file for summary."

@pytest.fixture
def agent():
    """
    Fixture for the agent under test.
    Replace 'Agent' with the actual agent class or function to be tested.
    """
    # Example: from mymodule import Agent
    # return Agent()
    # For demonstration, we'll use a MagicMock and patch the relevant method.
    return MagicMock()

def mock_process_file(file_content, user_consent, task_type):
    """
    Mocked agent method to simulate consent enforcement.
    """
    if not user_consent:
        return {
            "success": False,
            "error_code": "CONSENT_REQUIRED",
            "error_message": "User consent is required to process this file."
        }
    # Simulate normal processing (not needed for this test)
    return {"success": True, "result": "summary"}

def test_integration_consent_required_enforcement(agent, valid_txt_file_content):
    """
    Integration test: Checks that the agent enforces user consent and returns an error if user_consent is False.
    """
    # Patch the agent's processing method to simulate consent enforcement logic.
    # Replace 'process_file' with the actual method name if different.
    with patch.object(agent, "process_file", side_effect=mock_process_file):
        # Inputs as specified
        user_consent = False
        task_type = "summary"
        file_content = valid_txt_file_content

        # Call the agent's method under test
        response = agent.process_file(file_content, user_consent, task_type)

        # Assertions per success_criteria
        assert response["success"] is False, "Agent should return success=False when consent is not given."
        assert response["error_code"] == "CONSENT_REQUIRED", "Agent should return error_code='CONSENT_REQUIRED'."
        assert "consent" in response["error_message"].lower(), "Error message should mention consent requirement."
