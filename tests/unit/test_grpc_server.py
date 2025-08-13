"""
Unit tests for gRPC server implementation.
"""
import pytest
import asyncio
import grpc
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from grpc_ragflow_server.server import RagServicesServicer, serve
from grpc_ragflow_server import ragflow_service_pb2 as pb2
from grpc_ragflow_server import ragflow_service_pb2_grpc as pb2_grpc
from grpc_ragflow_server.ragflow_client import RAGFlowConnectionError


class TestRagServicesServicerDatasets:
    """Unit tests for dataset management operations."""

    @pytest.fixture
    def servicer(self):
        """Create servicer instance for testing."""
        return RagServicesServicer()

    @pytest.fixture
    def mock_context(self):
        """Create mock gRPC context."""
        return MagicMock()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_dataset_with_all_fields(self, servicer, mock_context):
        """Test dataset creation with all fields provided."""
        mock_client = AsyncMock()
        mock_client.create_dataset.return_value = {
            "code": 0,
            "data": {
                "id": "dataset_123",
                "name": "Test Dataset",
                "description": "Test description",
                "chunk_count": 0,
                "document_count": 0,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.CreateDatasetRequest(
                name="Test Dataset",
                description="Test description",
                embedding_model="BAAI/bge-large-zh-v1.5",
                permission="me",
                chunk_method="intelligent"
            )

            response = await servicer.CreateDataset(request, mock_context)

            # Verify client was called with correct data
            mock_client.create_dataset.assert_called_once()
            call_args = mock_client.create_dataset.call_args[0][0]
            assert call_args["name"] == "Test Dataset"
            assert call_args["description"] == "Test description"
            assert call_args["embedding_model"] == "BAAI/bge-large-zh-v1.5"

            # Verify response
            assert hasattr(response, 'data')
            assert response.data.id == "dataset_123"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_dataset_success(self, servicer, mock_context):
        """Test successful dataset update."""
        mock_client = AsyncMock()
        mock_client.update_dataset.return_value = {
            "code": 0,
            "data": {
                "id": "dataset_123",
                "name": "Updated Dataset",
                "description": "Updated description"
            }
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.UpdateDatasetRequest(
                dataset_id="dataset_123",
                name="Updated Dataset",
                description="Updated description"
            )

            response = await servicer.UpdateDataset(request, mock_context)

            mock_client.update_dataset.assert_called_once_with(
                "dataset_123",
                {"name": "Updated Dataset", "description": "Updated description"}
            )
            assert response.code == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_datasets_success(self, servicer, mock_context):
        """Test successful dataset deletion."""
        mock_client = AsyncMock()
        mock_client.delete_datasets.return_value = {"code": 0}

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.DeleteDatasetsRequest(ids=["id1", "id2", "id3"])

            response = await servicer.DeleteDatasets(request, mock_context)

            mock_client.delete_datasets.assert_called_once_with(
                {"ids": ["id1", "id2", "id3"]}
            )
            assert response.code == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_datasets_with_pagination(self, servicer, mock_context):
        """Test dataset listing with pagination."""
        mock_client = AsyncMock()
        mock_client.list_datasets.return_value = {
            "code": 0,
            "data": {
                "datasets": [
                    {"id": "1", "name": "Dataset 1"},
                    {"id": "2", "name": "Dataset 2"}
                ],
                "total": 10
            }
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.ListDatasetsRequest(page=2, page_size=5)

            response = await servicer.ListDatasets(request, mock_context)

            mock_client.list_datasets.assert_called_once_with({
                "page": 2,
                "page_size": 5
            })
            assert response.code == 0
            assert len(response.data) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_knowledge_graph_success(self, servicer, mock_context):
        """Test knowledge graph retrieval."""
        mock_client = AsyncMock()
        mock_client.get_dataset_knowledge_graph.return_value = {
            "code": 0,
            "data": {
                "nodes": [{"id": "node1", "label": "Entity 1"}],
                "edges": [{"from": "node1", "to": "node2", "relation": "related_to"}]
            }
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.GetDatasetKnowledgeGraphRequest(dataset_id="dataset_123")

            response = await servicer.GetDatasetKnowledgeGraph(request, mock_context)

            mock_client.get_dataset_knowledge_graph.assert_called_once_with("dataset_123")
            assert response.code == 0
            assert hasattr(response, 'data')


class TestRagServicesServicerDocuments:
    """Unit tests for document management operations."""

    @pytest.fixture
    def servicer(self):
        return RagServicesServicer()

    @pytest.fixture
    def mock_context(self):
        return MagicMock()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_upload_documents_success(self, servicer, mock_context):
        """Test successful document upload."""
        mock_client = AsyncMock()
        mock_client.upload_documents.return_value = {
            "code": 0,
            "data": {
                "documents": [
                    {
                        "id": "doc_123",
                        "name": "test.txt",
                        "status": "uploaded",
                        "size": 1024
                    }
                ]
            }
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            document_file = pb2.DocumentFile(
                filename="test.txt",
                content=b"Test document content"
            )
            request = pb2.UploadDocumentsRequest(
                dataset_id="dataset_123",
                files=[document_file]
            )

            response = await servicer.UploadDocuments(request, mock_context)

            mock_client.upload_documents.assert_called_once()
            assert response.code == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_documents_with_filters(self, servicer, mock_context):
        """Test document listing with filters."""
        mock_client = AsyncMock()
        mock_client.list_documents.return_value = {
            "code": 0,
            "data": {
                "documents": [
                    {"id": "1", "name": "doc1.txt", "status": "parsed"},
                    {"id": "2", "name": "doc2.pdf", "status": "parsing"}
                ],
                "total": 2
            }
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.ListDocumentsRequest(
                dataset_id="dataset_123",
                page=1,
                page_size=10,
                keywords="test"
            )

            response = await servicer.ListDocuments(request, mock_context)

            mock_client.list_documents.assert_called_once()
            call_args = mock_client.list_documents.call_args[0]
            assert call_args[0] == "dataset_123"
            assert "keywords" in call_args[1]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_documents_success(self, servicer, mock_context):
        """Test document parsing initiation."""
        mock_client = AsyncMock()
        mock_client.parse_documents.return_value = {"code": 0}

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.ParseDocumentsRequest(
                dataset_id="dataset_123",
                document_ids=["doc1", "doc2"]
            )

            response = await servicer.ParseDocuments(request, mock_context)

            mock_client.parse_documents.assert_called_once_with(
                "dataset_123",
                {"document_ids": ["doc1", "doc2"]}
            )
            assert response.code == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_documents_success(self, servicer, mock_context):
        """Test document deletion."""
        mock_client = AsyncMock()
        mock_client.delete_documents.return_value = {"code": 0}

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.DeleteDocumentsRequest(
                dataset_id="dataset_123",
                ids=["doc1", "doc2"]
            )

            response = await servicer.DeleteDocuments(request, mock_context)

            mock_client.delete_documents.assert_called_once_with(
                "dataset_123",
                {"ids": ["doc1", "doc2"]}
            )
            assert response.code == 0


class TestRagServicesServicerChunks:
    """Unit tests for chunk management operations."""

    @pytest.fixture
    def servicer(self):
        return RagServicesServicer()

    @pytest.fixture
    def mock_context(self):
        return MagicMock()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_chunk_success(self, servicer, mock_context):
        """Test successful chunk addition."""
        mock_client = AsyncMock()
        mock_client.add_chunk.return_value = {
            "code": 0,
            "data": {
                "id": "chunk_123",
                "content": "Test chunk content",
                "important_keywords": ["test", "chunk"]
            }
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.AddChunkRequest(
                dataset_id="dataset_123",
                document_id="doc_123",
                content="Test chunk content",
                important_keywords=["test", "chunk"],
                questions=["What is this chunk about?"]
            )

            response = await servicer.AddChunk(request, mock_context)

            mock_client.add_chunk.assert_called_once()
            assert response.code == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_chunks_success(self, servicer, mock_context):
        """Test chunk retrieval."""
        mock_client = AsyncMock()
        mock_client.retrieve_chunks.return_value = {
            "code": 0,
            "data": {
                "chunks": [
                    {
                        "id": "chunk1",
                        "content": "Relevant content 1",
                        "similarity": 0.85,
                        "document_id": "doc1"
                    },
                    {
                        "id": "chunk2", 
                        "content": "Relevant content 2",
                        "similarity": 0.78,
                        "document_id": "doc2"
                    }
                ],
                "total": 2
            }
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.RetrieveChunksRequest(
                question="What is machine learning?",
                dataset_ids=["dataset1", "dataset2"],
                top_k=5,
                similarity_threshold=0.7
            )

            response = await servicer.RetrieveChunks(request, mock_context)

            mock_client.retrieve_chunks.assert_called_once()
            call_args = mock_client.retrieve_chunks.call_args[0][0]
            assert call_args["question"] == "What is machine learning?"
            assert call_args["dataset_ids"] == ["dataset1", "dataset2"]
            assert call_args["top_k"] == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_update_chunk_success(self, servicer, mock_context):
        """Test chunk update."""
        mock_client = AsyncMock()
        mock_client.update_chunk.return_value = {
            "code": 0,
            "data": {"id": "chunk_123", "content": "Updated content"}
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.UpdateChunkRequest(
                dataset_id="dataset_123",
                document_id="doc_123",
                chunk_id="chunk_123",
                content="Updated content",
                important_keywords=["updated", "content"]
            )

            response = await servicer.UpdateChunk(request, mock_context)

            mock_client.update_chunk.assert_called_once()
            call_args = mock_client.update_chunk.call_args
            assert call_args[0][0] == "dataset_123"
            assert call_args[0][1] == "doc_123"
            assert call_args[0][2] == "chunk_123"


class TestRagServicesServicerChatCompletion:
    """Unit tests for chat completion operations."""

    @pytest.fixture
    def servicer(self):
        return RagServicesServicer()

    @pytest.fixture
    def mock_context(self):
        return MagicMock()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_chat_completion_streaming(self, servicer, mock_context):
        """Test streaming chat completion."""
        from unittest.mock import MagicMock
        
        # Create a proper async generator function
        async def mock_completion_generator(chat_id, data):
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " world"}}]}
            yield {"choices": [{"delta": {"content": "!"}}]}
        
        # Create a mock client and assign the async generator directly
        mock_client = MagicMock()
        mock_client.create_chat_completion = mock_completion_generator

        with patch.object(servicer, '_get_client', return_value=mock_client):
            messages = [pb2.ChatMessage(role="user", content="Hello")]
            request = pb2.CreateChatCompletionRequest(
                chat_id="chat_123",
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True
            )

            # Collect streaming responses
            responses = []
            async for response in servicer.CreateChatCompletion(request, mock_context):
                responses.append(response)

            print("Chat completion responses:", responses)
            assert len(responses) == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_agent_completion_success(self, servicer, mock_context):
        """Test agent completion."""
        from unittest.mock import MagicMock
        
        # Create a proper async generator function
        async def mock_agent_completion_generator(agent_id, data):
            yield {
                "id": "completion_123",
                "choices": [{"delta": {"content": "Agent response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            }
        
        # Create a mock client and assign the async generator directly
        mock_client = MagicMock()
        mock_client.create_agent_completion = mock_agent_completion_generator

        with patch.object(servicer, '_get_client', return_value=mock_client):
            messages = [pb2.ChatMessage(role="user", content="Execute task")]
            request = pb2.CreateAgentCompletionRequest(
                agent_id="agent_123",
                model="gpt-4",
                messages=messages,
                stream=True
            )

            responses = []
            async for response in servicer.CreateAgentCompletion(request, mock_context):
                responses.append(response)

            assert len(responses) == 1


class TestRagServicesServicerAgents:
    """Unit tests for agent management operations."""

    @pytest.fixture
    def servicer(self):
        return RagServicesServicer()

    @pytest.fixture
    def mock_context(self):
        return MagicMock()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_agent_success(self, servicer, mock_context):
        """Test agent creation."""
        import json
        
        mock_client = AsyncMock()
        mock_client.create_agent.return_value = {
            "code": 0,
            "data": True,
            "message":"sucess"
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            dsl_dict = {
                "components": {
                    "begin": {"obj": {"component_name": "Begin"}},
                    "answer": {"obj": {"component_name": "Answer"}}
                }
            }
            dsl_json = json.dumps(dsl_dict)
            
            request = pb2.CreateAgentRequest(
                title="Test Agent",
                description="Test description",
                dsl=dsl_json
            )

            response = await servicer.CreateAgent(request, mock_context)

            mock_client.create_agent.assert_called_once_with({
                "title": "Test Agent",
                "description": "Test description", 
                "dsl": dsl_json
            })
            assert response.code == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_agents_success(self, servicer, mock_context):
        """Test agent listing."""
        mock_client = AsyncMock()
        mock_client.list_agents.return_value = {
            "code": 0,
            "data": {
                "agents": [
                    {"id": "1", "title": "Agent 1"},
                    {"id": "2", "title": "Agent 2"}
                ],
                "total": 2
            }
        }

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.ListAgentsRequest(page=1, page_size=10)

            response = await servicer.ListAgents(request, mock_context)

            mock_client.list_agents.assert_called_once()
            assert response.code == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_agent_success(self, servicer, mock_context):
        """Test agent deletion."""
        mock_client = AsyncMock()
        mock_client.delete_agent.return_value = {"code": 0}

        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.DeleteAgentRequest(agent_id="agent_123")

            response = await servicer.DeleteAgent(request, mock_context)

            mock_client.delete_agent.assert_called_once_with("agent_123")
            assert response.code == 0


class TestGRPCServerStartup:
    """Unit tests for gRPC server startup."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_serve_function_setup(self):
        """Test gRPC server setup."""
        with patch('grpc.aio.server') as mock_server_func:
            with patch('grpc_ragflow_server.server.pb2_grpc.add_RagServicesServicer_to_server') as mock_add_servicer:
                mock_server = AsyncMock()
                mock_server_func.return_value = mock_server
                mock_server.add_insecure_port.return_value = 50051
                mock_server.start.return_value = None
                mock_server.wait_for_termination.return_value = None

                # Mock the serve function to avoid actual server startup
                with patch('grpc_ragflow_server.server.serve') as mock_serve:
                    mock_serve.return_value = None
                    
                    # Test that serve function can be called without errors
                    result = await serve()
                    
                    # The function should complete without raising exceptions
                    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
