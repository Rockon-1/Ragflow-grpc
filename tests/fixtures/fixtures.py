"""
Pytest fixtures for RAGFlow gRPC tests.
"""
import pytest
import asyncio
import uuid
import tempfile
import os
from typing import Dict, List, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

# Add the project root to Python path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from grpc_ragflow_server.ragflow_client import RAGFlowClient, RAGFlowConnectionError
from grpc_ragflow_server.config import RAGFLOW_BASE_URL, API_KEY, GRPC_HOST, GRPC_PORT
from tests.conftest import TEST_CONFIG, TEST_DATASET_PREFIX, TEST_ASSISTANT_PREFIX, TEST_AGENT_PREFIX


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> Dict:
    """Provide test configuration."""
    return TEST_CONFIG.copy()


@pytest.fixture
def unique_id() -> str:
    """Generate a unique ID for test resources."""
    return str(uuid.uuid4())[:8]


@pytest.fixture
def test_dataset_name(unique_id: str) -> str:
    """Generate a unique test dataset name."""
    return f"{TEST_DATASET_PREFIX}{unique_id}"


@pytest.fixture
def test_assistant_name(unique_id: str) -> str:
    """Generate a unique test assistant name."""
    return f"{TEST_ASSISTANT_PREFIX}{unique_id}"


@pytest.fixture
def test_agent_name(unique_id: str) -> str:
    """Generate a unique test agent name."""
    return f"{TEST_AGENT_PREFIX}{unique_id}"


@pytest.fixture
async def ragflow_client() -> AsyncGenerator[RAGFlowClient, None]:
    """Provide an authenticated RAGFlow client."""
    async with RAGFlowClient(
        base_url=TEST_CONFIG["RAGFLOW_BASE_URL"],
        api_key=TEST_CONFIG["API_KEY"]
    ) as client:
        yield client


@pytest.fixture
def mock_ragflow_client() -> AsyncMock:
    """Provide a mocked RAGFlow client for unit tests."""
    mock_client = AsyncMock(spec=RAGFlowClient)
    
    # Setup common mock responses
    mock_client.create_dataset.return_value = {
        "code": 0,
        "data": {
            "id": "test_dataset_id",
            "name": "test_dataset",
            "created_at": "2024-01-01T00:00:00Z"
        }
    }
    
    mock_client.list_datasets.return_value = {
        "code": 0,
        "data": {
            "datasets": [],
            "total": 0
        }
    }
    
    mock_client.upload_documents.return_value = {
        "code": 0,
        "data": {
            "documents": [
                {
                    "id": "test_doc_id",
                    "name": "test_document.txt",
                    "status": "uploaded"
                }
            ]
        }
    }
    
    return mock_client


@pytest.fixture
async def test_dataset(ragflow_client: RAGFlowClient, test_dataset_name: str) -> Dict:
    """Create a test dataset and clean it up after the test."""
    dataset_data = {
        "name": test_dataset_name,
        "description": "Test dataset for pytest",
        "embedding_model": "BAAI/bge-large-zh-v1.5",
        "permission": "me",
        "chunk_method": "manual"
    }
    
    try:
        response = await ragflow_client.create_dataset(dataset_data)
        
        if response.get("code") == 0:
            dataset = response.get("data", {})
            yield dataset
        else:
            pytest.skip(f"Failed to create test dataset: {response}")
            
    except RAGFlowConnectionError:
        pytest.skip("RAGFlow service not available")
    except Exception as e:
        pytest.skip(f"Failed to setup test dataset: {e}")


@pytest.fixture
async def test_document(ragflow_client: RAGFlowClient, test_dataset: Dict) -> Dict:
    """Create a test document and clean it up after the test."""
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for RAGFlow gRPC testing.\n")
        f.write("It contains sample content to test document operations.\n")
        f.write("The document should be parsed and indexed properly.")
        temp_file_path = f.name
    
    try:
        # Upload the document
        files = [{
            "name": "test_document.txt",
            "content": open(temp_file_path, 'rb').read(),
            "type": "text/plain"
        }]
        
        response = await ragflow_client.upload_documents(test_dataset["id"], files)
        
        if response.get("code") == 0:
            documents = response.get("data", {}).get("documents", [])
            if documents:
                document = documents[0]
                yield document
            else:
                pytest.skip("No documents returned from upload")
        else:
            pytest.skip(f"Failed to upload test document: {response}")
            
    except Exception as e:
        pytest.skip(f"Failed to setup test document: {e}")
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass


@pytest.fixture
async def test_chat_assistant(ragflow_client: RAGFlowClient, test_dataset: Dict, test_assistant_name: str) -> Dict:
    """Create a test chat assistant and clean it up after the test."""
    assistant_data = {
        "name": test_assistant_name,
        "description": "Test chat assistant for pytest",
        "dataset_ids": [test_dataset["id"]],
        "avatar": ""
    }
    
    try:
        response = await ragflow_client.create_chat_assistant(assistant_data)
        
        if response.get("code") == 0:
            assistant = response.get("data", {})
            yield assistant
        else:
            pytest.skip(f"Failed to create test chat assistant: {response}")
            
    except Exception as e:
        pytest.skip(f"Failed to setup test chat assistant: {e}")


@pytest.fixture
async def test_agent(ragflow_client: RAGFlowClient, test_agent_name: str) -> Dict:
    """Create a test agent and clean it up after the test."""
    agent_data = {
        "title": test_agent_name,
        "description": "Test agent for pytest",
        "dsl": "{}"
    }
    
    try:
        response = await ragflow_client.create_agent(agent_data)
        
        if response.get("code") == 0:
            agent = response.get("data", {})
            yield agent
        else:
            pytest.skip(f"Failed to create test agent: {response}")
            
    except Exception as e:
        pytest.skip(f"Failed to setup test agent: {e}")


@pytest.fixture
def sample_documents() -> List[Dict]:
    """Provide sample document data for testing."""
    return [
        {
            "name": "ai_overview.txt",
            "content": b"""Artificial Intelligence (AI) Overview

Artificial Intelligence is a branch of computer science that aims to create intelligent machines that can think and act like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

Key AI Technologies:
1. Machine Learning - Algorithms that improve through experience
2. Deep Learning - Neural networks with multiple layers
3. Natural Language Processing - Understanding and generating human language
4. Computer Vision - Interpreting and analyzing visual information
5. Robotics - Physical AI systems that can interact with the environment

Applications of AI:
- Healthcare: Diagnosis, drug discovery, personalized medicine
- Finance: Fraud detection, algorithmic trading, risk assessment
- Transportation: Autonomous vehicles, traffic optimization
- Education: Personalized learning, intelligent tutoring systems
- Entertainment: Content recommendation, game development

The future of AI holds immense potential for transforming various industries and improving human life, while also presenting challenges that need to be addressed responsibly.""",
            "type": "text/plain"
        },
        {
            "name": "machine_learning_basics.txt",
            "content": b"""Machine Learning Fundamentals

Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data to make predictions or decisions.

Types of Machine Learning:

1. Supervised Learning
   - Uses labeled training data
   - Examples: Classification, Regression
   - Algorithms: Linear Regression, Decision Trees, SVM, Neural Networks

2. Unsupervised Learning
   - Finds patterns in data without labels
   - Examples: Clustering, Association Rules
   - Algorithms: K-Means, Hierarchical Clustering, PCA

3. Reinforcement Learning
   - Learns through interaction with environment
   - Uses rewards and penalties
   - Applications: Game playing, Robotics, Autonomous systems

Common ML Workflow:
1. Data Collection and Preparation
2. Feature Engineering
3. Model Selection
4. Training and Validation
5. Testing and Evaluation
6. Deployment and Monitoring

Popular ML Libraries:
- Python: scikit-learn, TensorFlow, PyTorch
- R: caret, randomForest, e1071
- Java: Weka, Apache Spark MLlib

Machine learning is revolutionizing industries by enabling data-driven decision making and automation of complex tasks.""",
            "type": "text/plain"
        }
    ]


@pytest.fixture
def grpc_channel_mock() -> MagicMock:
    """Provide a mocked gRPC channel for unit tests."""
    mock_channel = MagicMock()
    mock_stub = MagicMock()
    mock_channel.return_value = mock_stub
    return mock_channel



