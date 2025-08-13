"""
RAGFlow gRPC Server Package

This package provides a gRPC server implementation for RAGFlow API,
enabling high-performance, language-agnostic access to RAGFlow's
document management, chat, and retrieval capabilities.
"""

__version__ = "0.1.0"
__author__ = "Sachin Goyal"
__email__ = "goyal01sachin@gmail.com"

# Import main components for easy access
from .server import RagServicesServicer, serve
from .ragflow_client import RAGFlowClient, RAGFlowConnectionError
from .config import RAGFLOW_BASE_URL, API_KEY, GRPC_HOST, GRPC_PORT

__all__ = [
    "RagServicesServicer",
    "serve", 
    "RAGFlowClient",
    "RAGFlowConnectionError",
    "RAGFLOW_BASE_URL",
    "API_KEY", 
    "GRPC_HOST",
    "GRPC_PORT"
]
