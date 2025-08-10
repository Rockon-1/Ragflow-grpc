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

### Test gRPC Server Connection
Verify that the gRPC server is running and accessible:
```bash
uv run python test_grpc_connection.py
```

### Additional Tests



