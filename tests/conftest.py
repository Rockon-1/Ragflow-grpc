"""
Configuration for tests.
"""
import os
import pytest
from typing import Dict, Any

# Test configuration
TEST_CONFIG = {
    "RAGFLOW_BASE_URL": os.environ.get("TEST_RAGFLOW_BASE_URL", "http://localhost:80"),
    "API_KEY": os.environ.get("TEST_API_KEY", "ragflow-QzMzk4NzE0NzQ2ZDExZjBhMDgzYTZiOW"),
    "GRPC_HOST": os.environ.get("TEST_GRPC_HOST", "localhost"),
    "GRPC_PORT": int(os.environ.get("TEST_GRPC_PORT", "50051")),
    "TIMEOUT": 30,
    "MAX_RETRIES": 3,
}

# Test data configuration
TEST_DATASET_PREFIX = "test_dataset_"
TEST_ASSISTANT_PREFIX = "test_assistant_"
TEST_AGENT_PREFIX = "test_agent_"

# Test file paths
EXAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example_data")
DOCUMENTS_DIR = os.path.join(EXAMPLE_DATA_DIR, "documents")

# Test data configuration
TEST_DATASET_PREFIX = "test_dataset_"
TEST_ASSISTANT_PREFIX = "test_assistant_"
TEST_AGENT_PREFIX = "test_agent_"

# Test file paths
EXAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example_data")
DOCUMENTS_DIR = os.path.join(EXAMPLE_DATA_DIR, "documents")

def get_test_config() -> Dict[str, Any]:
    """Get test configuration."""
    return TEST_CONFIG.copy()

def is_grpc_server_available() -> bool:
    """Check if gRPC server is available for testing."""
    import grpc
    
    try:
        channel = grpc.insecure_channel(f"{TEST_CONFIG['GRPC_HOST']}:{TEST_CONFIG['GRPC_PORT']}")
        grpc.channel_ready_future(channel).result(timeout=5)
        channel.close()
        return True
    except:
        return False

# Pytest markers
pytest_markers = {
    "unit": "Unit tests that don't require external services",
    "grpc": "Tests that require gRPC server",
}
