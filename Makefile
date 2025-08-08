
.PHONY: setup install-uv check-deps clone-ragflow run-docker protobuf install-deps

setup: install-uv check-deps clone-ragflow run-docker install-deps protobuf ## Full project setup
	@echo "Project setup complete!"

install-uv: ## Install uv package manager if not present
ifeq ($(OS),Windows_NT)
	@where uv >nul 2>&1 || powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
else
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
endif

check-deps: ## Check for git and docker
ifeq ($(OS),Windows_NT)
	@where git >nul 2>&1 || (echo Error: git is not installed. && exit 1)
	@where docker >nul 2>&1 || (echo Error: docker is not installed. && exit 1)
else
	@command -v git >/dev/null 2>&1 || (echo "Error: git is not installed." && exit 1)
	@command -v docker >/dev/null 2>&1 || (echo "Error: docker is not installed." && exit 1)
endif

clone-ragflow: ## Clone ragflow repo and checkout version
ifeq ($(OS),Windows_NT)
	@if not exist ragflow git clone https://github.com/infiniflow/ragflow.git
	@cd ragflow\docker && git checkout -f v0.20.1
else
	@test -d ragflow || git clone https://github.com/infiniflow/ragflow.git
	@cd ragflow/docker && git checkout -f v0.20.1
endif

run-docker: ## Run ragflow docker engine with smart container management
ifeq ($(OS),Windows_NT)
	@echo Checking ragflow-server container status...
	@docker ps -q -f name=ragflow-server | findstr . >nul 2>&1 && ( \
		echo ragflow-server is already running \
	) || ( \
		docker ps -aq -f name=ragflow-server | findstr . >nul 2>&1 && ( \
			echo ragflow-server exists but stopped, starting existing container... && \
			docker compose -f ragflow\docker\docker-compose.yml start \
		) || ( \
			echo ragflow-server not found, creating and starting new container... && \
			docker compose -f ragflow\docker\docker-compose.yml up -d \
		) \
	)
else
	@echo "Checking ragflow-server container status..."
	@if [ $$(docker ps -q -f name=ragflow-server) ]; then \
		echo "ragflow-server is already running"; \
	elif [ $$(docker ps -aq -f name=ragflow-server) ]; then \
		echo "ragflow-server exists but stopped, starting existing container..."; \
		docker compose -f ragflow/docker/docker-compose.yml start; \
	else \
		echo "ragflow-server not found, creating and starting new container..."; \
		docker compose -f ragflow/docker/docker-compose.yml up -d; \
	fi
endif


protobuf: ## Generate protobuf files
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grpc_ragflow_server\ragflow_service.proto || echo "protoc not found, using pre-generated files"

install-deps: ## Install required Python dependencies
	@uv venv .venv
ifeq ($(OS),Windows_NT)
	@if exist pyproject.toml (uv sync) else (if exist requirements.txt (uv pip install -r requirements.txt && uv pip install -e .) else echo No pyproject.toml or requirements.txt found)
else
	@test -f pyproject.toml && uv sync || (test -f requirements.txt && (uv pip install -r requirements.txt && uv pip install -e .) || echo "No pyproject.toml or requirements.txt found")
endif
