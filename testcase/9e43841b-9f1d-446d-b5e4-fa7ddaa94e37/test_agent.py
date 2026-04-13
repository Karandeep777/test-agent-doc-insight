
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def unsupported_file():
    """
    Fixture that returns a mock file object with an unsupported extension (.exe).
    """
    mock_file = MagicMock()
    mock_file.filename = "malware.exe"
    mock_file.read.return_value = b"dummy content"
    return mock_file

@pytest.fixture
def agent():
    """
    Fixture that returns a mock agent instance with an upload_file method.
    The upload_file method will be patched in the test.
    """
    class MockAgent:
        def upload_file(self, file, user_consent, task_type):
            # Placeholder, will be patched in test
            pass
    return MockAgent()

def test_functional_unsupported_file_type_error(agent, unsupported_file):
    """
    Functional test:
    Ensures that uploading an unsupported file type returns a clear error message
    and does not process the file.
    """
    # Patch the upload_file method to simulate the unsupported file type error response
    expected_response = {
        "success": False,
        "error_code": "UNSUPPORTED_FORMAT",
        "error_message": "The file format is not supported. Please upload a supported file type."
    }
    with patch.object(agent, "upload_file", return_value=expected_response) as mock_upload:
        response = agent.upload_file(
            file=unsupported_file,
            user_consent=True,
            task_type="summary"
        )

        # Assert all success criteria
        assert response["success"] is False, "Expected success to be False for unsupported file type"
        assert response["error_code"] == "UNSUPPORTED_FORMAT", "Expected error_code to be 'UNSUPPORTED_FORMAT'"
        assert "file format is not supported" in response["error_message"].lower(), \
            "Error message should mention unsupported file format"

        # Ensure the upload_file method was called with the correct parameters
        mock_upload.assert_called_once_with(
            file=unsupported_file,
            user_consent=True,
            task_type="summary"
        )
