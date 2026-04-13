
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def valid_txt_file(tmp_path):
    """
    Fixture to create a valid TXT file for testing.
    """
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test file.")
    return file_path

@pytest.fixture
def agent():
    """
    Fixture to provide a mock agent with a process_file method.
    Replace this with the actual agent import as needed.
    """
    class MockAgent:
        def process_file(self, file_path, user_consent, task_type, additional_params):
            # Simulate the logic that parses additional_params as JSON
            import json
            try:
                params = json.loads(additional_params)
            except Exception:
                return {
                    "success": False,
                    "error_code": "MALFORMED_JSON",
                    "error_message": "Malformed JSON in additional_params"
                }
            # If JSON is valid, simulate normal processing (not needed for this test)
            return {"success": True}
    return MockAgent()

def test_functional_malformed_additional_params_json(agent, valid_txt_file):
    """
    Functional test: Validates that malformed JSON in additional_params returns a clear error
    and does not process the file.
    """
    # Inputs
    file_path = valid_txt_file
    user_consent = True
    task_type = 'qa'
    additional_params = 'not_a_json_string'  # Malformed JSON

    # No external dependencies to mock for this functional test, as agent is local

    response = agent.process_file(
        file_path=file_path,
        user_consent=user_consent,
        task_type=task_type,
        additional_params=additional_params
    )

    # Success criteria
    assert response['success'] is False, "Expected success to be False for malformed JSON"
    assert response['error_code'] == 'MALFORMED_JSON', "Expected error_code to be 'MALFORMED_JSON'"
    assert 'malformed json' in response['error_message'].lower(), "Error message should mention 'malformed json'"
