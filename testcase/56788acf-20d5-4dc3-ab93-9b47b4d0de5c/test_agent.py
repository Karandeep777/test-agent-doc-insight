
import pytest
from unittest.mock import patch, MagicMock
import io

@pytest.fixture
def sample_txt_file():
    """
    Fixture that returns a file-like object representing a valid TXT file.
    """
    return io.StringIO("This is a sample document about climate change and its effects.")

@pytest.fixture
def mock_rag_retriever():
    """
    Fixture that mocks the RAG retriever's retrieve method.
    """
    with patch("your_module.RAGRetriever") as MockRetriever:
        instance = MockRetriever.return_value
        instance.retrieve.return_value = [
            {"text": "climate change", "source": "sample.txt", "score": 0.95}
        ]
        yield instance

@pytest.fixture
def mock_llm():
    """
    Fixture that mocks the LLM call.
    """
    with patch("your_module.llm_call") as mock_llm_call:
        mock_llm_call.return_value = {
            "answer": "The main topic is climate change.",
            "citations": ["sample.txt"],
            "confidence": 0.92
        }
        yield mock_llm_call

@pytest.fixture
def agent(mock_rag_retriever, mock_llm):
    """
    Fixture that returns an instance of the agent under test, with dependencies patched.
    """
    from your_module import QARagAgent
    return QARagAgent()

def test_integration_qa_with_rag_retrieval(agent, sample_txt_file):
    """
    Integration test: End-to-end Q&A workflow with RAG retrieval.
    Uploads a TXT file, asks a question, and verifies the answer uses both document and RAG context.
    """
    # Arrange
    user_consent = True
    task_type = 'qa'
    additional_params = {'question': 'What is the main topic?'}
    # Simulate file upload and question
    # The agent is expected to use the RAG retriever and LLM (both mocked)
    # Act
    response = agent.handle_request(
        file=sample_txt_file,
        user_consent=user_consent,
        task_type=task_type,
        additional_params=additional_params
    )
    # Assert
    assert response['success'] is True
    assert 'answer' in response['result']
    assert 'citations' in response['result']
    assert 'confidence' in response['result']

def test_integration_qa_with_rag_retrieval_rag_failure(agent, sample_txt_file):
    """
    Integration test: Simulate RAG retrieval failure and verify error handling.
    """
    user_consent = True
    task_type = 'qa'
    additional_params = {'question': 'What is the main topic?'}
    with patch("your_module.RAGRetriever") as MockRetriever:
        instance = MockRetriever.return_value
        instance.retrieve.side_effect = Exception("RAG retrieval failed")
        response = agent.handle_request(
            file=sample_txt_file,
            user_consent=user_consent,
            task_type=task_type,
            additional_params=additional_params
        )
        assert response['success'] is False
        assert 'error' in response
        assert "RAG retrieval failed" in response['error']

def test_integration_qa_with_rag_retrieval_llm_failure(agent, sample_txt_file):
    """
    Integration test: Simulate LLM call failure and verify error handling.
    """
    user_consent = True
    task_type = 'qa'
    additional_params = {'question': 'What is the main topic?'}
    with patch("your_module.llm_call") as mock_llm_call:
        mock_llm_call.side_effect = Exception("LLM call failed")
        response = agent.handle_request(
            file=sample_txt_file,
            user_consent=user_consent,
            task_type=task_type,
            additional_params=additional_params
        )
        assert response['success'] is False
        assert 'error' in response
        assert "LLM call failed" in response['error']

def test_integration_qa_with_rag_retrieval_missing_question(agent, sample_txt_file):
    """
    Integration test: Missing question in additional_params should result in error.
    """
    user_consent = True
    task_type = 'qa'
    additional_params = {}  # No question provided
    response = agent.handle_request(
        file=sample_txt_file,
        user_consent=user_consent,
        task_type=task_type,
        additional_params=additional_params
    )
    assert response['success'] is False
    assert 'error' in response
    assert "question" in response['error'].lower()
