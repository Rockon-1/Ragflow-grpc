Note:- 
Most of the commands are platform independent and all commands and make file are successfully tested in windows as I have developed inside windows system(Issue might come in non-windows system)


Kindly use Make file to setup the project:-
Run below command to complete setup in 1 line
```
make setup
```
* This will clone official RAGFLOW repo from :- https://github.com/infiniflow/ragflow.git
* start the neccessary docker engine which is prerequistite to run ragflow 
* install required python dependencies

OR do it manually

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
RAGFLOW_BASE_URL = "http://localhost:9380"
API_KEY = "your-api-key-here"

# gRPC Server Configuration
GRPC_HOST = "0.0.0.0"  # Listen on all interfaces
GRPC_PORT = 50051      # gRPC server port
```

## Running the Server

*Interactive mode - shows menu
python main.py

*Direct server start
python main.py server

*Run client examples with debug logging
python main.py client --log-level DEBUG

*Run both with error-only logging
python main.py both --log-level ERROR


--------------------------------------------
## Test

# Test grpc server is running
    uv run python test_grpc_connection.py

# 


