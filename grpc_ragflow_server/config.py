import os

# RAGFlow API Configuration
RAGFLOW_BASE_URL = os.environ.get("RAGFLOW_BASE_URL", "http://localhost:80")
API_KEY = os.environ.get("API_KEY", "ragflow-QzMzk4NzE0NzQ2ZDExZjBhMDgzYTZiOW")

# gRPC Server Configuration  
GRPC_HOST = os.environ.get("GRPC_HOST", "localhost")
GRPC_PORT = int(os.environ.get("GRPC_PORT", "50051"))
