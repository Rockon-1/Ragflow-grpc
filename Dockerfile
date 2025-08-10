# Use Python 3.13 slim image as base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GRPC_HOST=0.0.0.0
ENV GRPC_PORT=50051

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml uv.lock ./
COPY grpc_ragflow_server/ ./grpc_ragflow_server/
COPY main.py ./
COPY grpc_client_example.py ./
COPY grpc_test_connection.py ./

# Create virtual environment and install dependencies
RUN uv venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv sync

# Generate protobuf files
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc_ragflow_server/ragflow_service.proto

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); channel.close()" || exit 1

# Run the gRPC server
CMD ["python", "main.py", "server"]
