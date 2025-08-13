```
Created by:-
Sachin Goyal
goyal01sachin@gmail.com
+91-6397851632
```


## Quick Setup

Use the Makefile to setup the project easily with platform-independent commands:

### Full Setup (New Environment)
For first-time setup or fresh environment:
```bash
make setup
```
This command will:
* Clone the official RAGFlow repository from https://github.com/infiniflow/ragflow.git
* Start the necessary Docker containers (prerequisite to run RAGFlow)
* Install required Python dependencies
* Generate protobuf files

### Setup Without RAGFlow (Existing Docker Environment)
If you already have RAGFlow Docker containers running:
```bash
make setup-without-ragflow
```
This command will:
* Install the uv package manager
* Check for git and docker dependencies
* Install required Python dependencies
* Generate protobuf files
* **Skip** cloning RAGFlow repository and starting Docker containers

### Available Make Commands

#### Project Setup
- `make setup` - Full project setup (clone + docker + dependencies)
- `make setup-without-ragflow` - Setup without RAGFlow (useful when docker is already running)
- `make install-uv` - Install uv package manager
- `make check-deps` - Check for git and docker
- `make clone-ragflow` - Clone RAGFlow repository only
- `make run-docker` - Start RAGFlow Docker containers only
- `make install-deps` - Install Python dependencies only
- `make protobuf` - Generate protobuf files only

#### Docker Commands
- `make docker-build` - Build the gRPC server Docker image
- `make docker-run` - Run the gRPC server in Docker container
- `make docker-stop` - Stop the gRPC server Docker container
- `make docker-logs` - View logs from the Docker container
- `make docker-build-and-run` - Build and run the Docker container

#### Testing Commands
- `make test` - Run all tests
- `make test-unit` - Run unit tests only
- `make test-grpc` - Run gRPC tests (requires server running)
- `make test-check` - Check test environment setup

#### Other Commands
- `make help` - Show all available commands with descriptions

**Note:** Most commands are **platform independent** and successfully tested on Windows. Minor issues might occur on non-Windows systems.

## Manual Setup (Alternative)

If you prefer to set up manually instead of using the Makefile:

```bash
# 1. Install uv package manager
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Create virtual environment
uv venv .venv 

# 3. Activate virtual environment
.venv\Scripts\activate.bat    # Windows

# 4. Install dependencies
uv sync

# 5. Generate protobuf files 
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ragflow_service.proto
```

## Configuration

Edit `config.py` with your RAGFlow server details:

```python
# RAGFlow API Configuration
RAGFLOW_BASE_URL = "http://localhost:80"
API_KEY = "your-api-key-here"

# gRPC Server Configuration
GRPC_HOST = "0.0.0.0"  # Listen on all interfaces
GRPC_PORT = 50051      # gRPC server port
```

## Running the Application

The application can be run in multiple ways: directly with Python, or using Docker containers.

### Option 1: Docker Deployment (Recommended)

#### Quick Start with Docker
```bash
# Build and run the gRPC server in Docker
make docker-build-and-run
```

#### Manual Docker Commands
```bash
# Build the Docker image
make docker-build
# or
docker build -t ragflow-grpc-server .

# Run with docker-compose (recommended)
make docker-run
# or  
docker compose up -d

# View logs
make docker-logs
# or
docker compose logs -f ragflow-grpc-server

# Stop the container
make docker-stop
# or
docker compose down
```

#### Environment Configuration for Docker
Create a `.env` file in the project root to override default settings:
```env
RAGFLOW_BASE_URL=http://host.docker.internal:80
API_KEY=your-api-key-here
GRPC_HOST=0.0.0.0
GRPC_PORT=50051
```

**Note:** Use `host.docker.internal` instead of `localhost` when running in Docker to access services on the host machine.

### Option 2: Direct Python Execution

The application requires two separate processes to run successfully - one for the server and another for the client.

#### Interactive Mode
Shows a menu where you can start any application depending upon your choice:
```bash
python main.py
```
Run the above command in 2 different processes with different menu options.

#### Direct Mode
**Start Server** (in first process):
```bash
python main.py server
```

**Start Client** (in second process):
```bash
python main.py client
```

#### Advanced Options
Run client with debug logging:
```bash
python main.py client --log-level DEBUG
```

Run both with error-only logging:
```bash
python main.py both --log-level ERROR
```

## Testing

The project includes comprehensive test suites to ensure reliability and functionality. Tests are organized into different categories and can be run using multiple methods.

### Test Categories

- **Unit Tests**: Test individual components without external dependencies (fast, no server required)
- **gRPC Tests**: Test gRPC server functionality (requires running RAGFlow server)

### Quick Test Commands

#### Using Make (Recommended)
```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run only gRPC tests (requires server running)
make test-grpc

# Check test environment setup
make test-check
```

#### Using Test Runner Script
```bash
# Run all tests
python test_runner.py all

# Run only unit tests
python test_runner.py unit

# Run only gRPC tests
python test_runner.py grpc

# Check test environment
python test_runner.py check
```

#### Using Pytest Directly
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run only unit tests
uv run pytest -m unit

# Run only gRPC tests
uv run pytest -m grpc

# Run specific test file
uv run pytest tests/unit/test_grpc_server.py

# Run specific test method
uv run pytest tests/unit/test_grpc_server.py::TestRagServicesServicerDatasets::test_create_dataset_with_all_fields
```

#### Debug Mode
```bash
# Run with detailed output and no capture
uv run pytest -v -s --tb=long

# Run single test with debugging
uv run pytest tests/unit/test_grpc_server.py::test_specific_function -v -s
```

### Test Environment Setup

#### Prerequisites
Before running tests, ensure:

1. **Python Environment**: Virtual environment is activated
2. **Dependencies**: All test dependencies are installed
3. **Protobuf Files**: Generated protobuf files exist
4. **Example Data**: Test data files are present

```bash
# Check test environment
python test_runner.py check
# or
make test-check
```

#### For gRPC Tests
gRPC tests require a running RAGFlow server:

```bash
# Start RAGFlow server (if not already running)
make run-docker

# Verify gRPC server connection
uv run python grpc_test_connection.py

# Run gRPC tests
make test-grpc
```

### Test Configuration

Tests use configuration from `tests/conftest.py`. You can override settings using environment variables:

```bash
# Override test server URLs
export TEST_RAGFLOW_BASE_URL="http://localhost:80"
export TEST_API_KEY="your-test-api-key"
export TEST_GRPC_HOST="localhost"
export TEST_GRPC_PORT="50051"

# Run tests with custom config
uv run pytest
```

### Coverage Reports

Generate detailed coverage reports:

```bash
# HTML coverage report using pytest directly
uv run pytest --cov=grpc_ragflow_server --cov-report=html
# Report saved to htmlcov/index.html

# Terminal coverage report
uv run pytest --cov=grpc_ragflow_server --cov-report=term-missing

# XML coverage report (for CI/CD)
uv run pytest --cov=grpc_ragflow_server --cov-report=xml
```

### Test Data Management

The project includes example test data in `example_data/`:

- **Documents**: Sample text files for testing document operations
- **Datasets**: Predefined dataset configurations
- **Fixtures**: Reusable test fixtures and mock data

Test resources are automatically cleaned up after test runs to prevent interference between tests.

### Continuous Integration

For CI/CD pipelines, use:

```bash
# Fast test run (unit tests only)
python test_runner.py unit

# Full test suite
python test_runner.py all

# Environment check
python test_runner.py check
```

### Troubleshooting Tests

#### Common Issues

1. **Server Connection Failed**
   ```bash
   # Check if RAGFlow server is running
   docker ps | grep ragflow
   
   # Start server if needed
   make run-docker
   ```

2. **Missing Protobuf Files**
   ```bash
   # Regenerate protobuf files
   make protobuf
   ```

3. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate.bat  # Windows
   
   # Reinstall dependencies
   uv sync
   ```

4. **Test Data Missing**
   ```bash
   # Verify example data exists
   ls example_data/documents/
   
   # Check test environment
   python test_runner.py check
   ```

### Writing New Tests

When adding new tests:

1. **Unit Tests**: Place in `tests/unit/`
2. **Mark Tests**: Use `@pytest.mark.unit` or `@pytest.mark.grpc`
3. **Use Fixtures**: Leverage existing fixtures from `tests/fixtures/`
4. **Follow Naming**: Use `test_*.py` file naming convention

Example test structure:
```python
import pytest
from tests.fixtures.fixtures import mock_ragflow_client

@pytest.mark.unit
def test_new_functionality(mock_ragflow_client):
    # Test implementation
    pass

@pytest.mark.grpc
@pytest.mark.asyncio
async def test_grpc_functionality():
    # Async gRPC test implementation
    pass
```



