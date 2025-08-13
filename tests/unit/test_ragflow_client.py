"""
Unit tests for RAGFlow gRPC server.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from grpc_ragflow_server.ragflow_client import RAGFlowClient, RAGFlowConnectionError
from grpc_ragflow_server.server import RagServicesServicer
from grpc_ragflow_server import ragflow_service_pb2 as pb2


class TestRAGFlowClient:
    """Unit tests for RAGFlowClient."""

    @pytest.mark.unit
    def test_init(self):
        """Test RAGFlowClient initialization."""
        client = RAGFlowClient("http://test.com", "test_key")
        assert client.base_url == "http://test.com"
        assert client.api_key == "test_key"
        assert client.session is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test RAGFlowClient as async context manager."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Create a proper mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            # Set up the session.get to return the mock response
            mock_session.get.return_value = mock_response
            
            async with RAGFlowClient("http://test.com", "test_key") as client:
                assert client.session is not None
                # Verify session was created with correct headers
                mock_session_class.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_connection_success(self):
        """Test successful connection check."""
        client = RAGFlowClient("http://test.com", "test_key")
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            # Create a proper mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            # Set up the session.get to return the mock response
            mock_session.get.return_value = mock_response
            
            result = await client.check_connection()
            assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_connection_failure(self):
        """Test connection check failure."""
        client = RAGFlowClient("http://invalid.com", "test_key")
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            mock_session.get.side_effect = Exception("Connection failed")
            
            result = await client.check_connection()
            assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful API request."""
        client = RAGFlowClient("http://test.com", "test_key")
        client.session = AsyncMock()
        
        # Create a proper mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"code": 0, "data": {"test": "value"}})
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        # Set up the session.request to return the mock response
        client.session.request.return_value = mock_response
        
        result = await client._make_request("GET", "/test")
        assert result == {"code": 0, "data": {"test": "value"}}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_make_request_connection_error(self):
        """Test API request with connection error."""
        client = RAGFlowClient("http://test.com", "test_key")
        client.session = AsyncMock()
        
        import aiohttp
        import os
        # Create a proper OSError
        os_error = OSError("Connection failed")
        os_error.errno = 2
        os_error.strerror = "Connection failed"
        
        client.session.request.side_effect = aiohttp.ClientConnectorError(
            connection_key=None, os_error=os_error
        )
        
        with pytest.raises(RAGFlowConnectionError):
            await client._make_request("GET", "/test")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_dataset(self):
        """Test dataset creation."""
        client = RAGFlowClient("http://test.com", "test_key")
        client._make_request = AsyncMock(return_value={"code": 0, "data": {"id": "test_id"}})
        
        data = {"name": "test_dataset", "description": "test"}
        result = await client.create_dataset(data)
        
        assert result == {"code": 0, "data": {"id": "test_id"}}
        client._make_request.assert_called_once_with(
            'POST', '/api/v1/datasets', 
            json=data, 
            headers={'Authorization': 'Bearer test_key'}
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_datasets(self):
        """Test dataset listing."""
        client = RAGFlowClient("http://test.com", "test_key")
        client._make_request = AsyncMock(return_value={
            "code": 0, 
            "data": {"datasets": [], "total": 0}
        })
        
        params = {"page": 1, "page_size": 10}
        result = await client.list_datasets(params)
        
        assert result["code"] == 0
        client._make_request.assert_called_once_with(
            'GET', '/api/v1/datasets',
            params=params,
            headers={'Authorization': 'Bearer test_key'}
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_delete_datasets(self):
        """Test dataset deletion."""
        client = RAGFlowClient("http://test.com", "test_key")
        client._make_request = AsyncMock(return_value={"code": 0})
        
        data = {"ids": ["id1", "id2"]}
        result = await client.delete_datasets(data)
        
        assert result == {"code": 0}
        client._make_request.assert_called_once_with(
            'DELETE', '/api/v1/datasets',
            json=data,
            headers={'Authorization': 'Bearer test_key'}
        )


class TestRagServicesServicer:
    """Unit tests for RagServicesServicer."""

    @pytest.mark.unit
    def test_init(self):
        """Test servicer initialization."""
        servicer = RagServicesServicer()
        assert servicer.client is None

    @pytest.mark.unit
    def test_handle_error(self):
        """Test error handling."""
        servicer = RagServicesServicer()
        error_dict = {"code": 400, "message": "Bad request"}
        
        error_response = servicer._handle_error(error_dict)
        
        assert error_response.code == 400
        assert error_response.message == "Bad request"

    @pytest.mark.unit
    def test_handle_connection_error(self):
        """Test connection error handling."""
        servicer = RagServicesServicer()
        connection_error = RAGFlowConnectionError("Service unavailable")
        
        error_response = servicer._handle_connection_error(connection_error)
        
        assert error_response.code == 503
        assert "RAGFlow service unavailable" in error_response.message

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_client_success(self):
        """Test successful client creation."""
        servicer = RagServicesServicer()
        
        with patch('grpc_ragflow_server.server.RAGFlowClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            client = await servicer._get_client()
            assert client == mock_client

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_client_connection_error(self):
        """Test client creation with connection error."""
        servicer = RagServicesServicer()
        
        with patch('grpc_ragflow_server.server.RAGFlowClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.side_effect = RAGFlowConnectionError("Connection failed")
            mock_client_class.return_value = mock_client
            
            with pytest.raises(RAGFlowConnectionError):
                await servicer._get_client()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_dataset_success(self):
        """Test successful dataset creation via gRPC."""
        servicer = RagServicesServicer()
        
        # Mock the client
        mock_client = AsyncMock()
        mock_client.create_dataset.return_value = {
            "code": 0,
            "data": {"id": "test_id", "name": "test_dataset"}
        }
        
        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.CreateDatasetRequest(
                name="test_dataset",
                description="test description"
            )
            context = MagicMock()
            
            response = await servicer.CreateDataset(request, context)
            
            # Verify the response structure
            assert response.code == 0
            assert hasattr(response, 'data')
            mock_client.create_dataset.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_dataset_connection_error(self):
        """Test dataset creation with connection error."""
        servicer = RagServicesServicer()
        
        with patch.object(servicer, '_get_client', side_effect=RAGFlowConnectionError("Connection failed")):
            request = pb2.CreateDatasetRequest(name="test_dataset")
            context = MagicMock()
            
            response = await servicer.CreateDataset(request, context)
            
            # Should return error response
            assert response.code == 500

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_datasets_success(self):
        """Test successful dataset listing via gRPC."""
        servicer = RagServicesServicer()
        
        mock_client = AsyncMock()
        mock_client.list_datasets.return_value = {
            "code": 0,
            "data": {
                "datasets": [
                    {"id": "1", "name": "dataset1"},
                    {"id": "2", "name": "dataset2"}
                ],
                "total": 2
            }
        }
        
        with patch.object(servicer, '_get_client', return_value=mock_client):
            request = pb2.ListDatasetsRequest(page=1, page_size=10)
            context = MagicMock()
            
            response = await servicer.ListDatasets(request, context)
            
            assert response.code == 0
            assert len(response.data) >= 0
            mock_client.list_datasets.assert_called_once()


class TestProtobufMessages:
    """Unit tests for protobuf message creation."""

    @pytest.mark.unit
    def test_create_dataset_request(self):
        """Test CreateDatasetRequest message creation."""
        request = pb2.CreateDatasetRequest(
            name="test_dataset",
            description="test description",
            embedding_model="test_model",
            permission="me",
            chunk_method="manual"
        )
        
        assert request.name == "test_dataset"
        assert request.description == "test description"
        assert request.embedding_model == "test_model"
        assert request.permission == "me"
        assert request.chunk_method == "manual"

    @pytest.mark.unit
    def test_error_response(self):
        """Test ErrorResponse message creation."""
        error = pb2.ErrorResponse(
            code=400,
            message="Bad request"
        )
        
        assert error.code == 400
        assert error.message == "Bad request"

    @pytest.mark.unit
    def test_chat_message(self):
        """Test ChatMessage message creation."""
        message = pb2.ChatMessage(
            role="user",
            content="Hello, world!"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, world!"

    @pytest.mark.unit
    def test_chat_completion_request(self):
        """Test CreateChatCompletionRequest message creation."""
        messages = [
            pb2.ChatMessage(role="user", content="Hello"),
            pb2.ChatMessage(role="assistant", content="Hi there!")
        ]
        
        request = pb2.CreateChatCompletionRequest(
            chat_id="test_chat_id",
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )
        
        assert request.chat_id == "test_chat_id"
        assert request.model == "gpt-3.5-turbo"
        assert len(request.messages) == 2
        assert request.stream is True


class TestErrorHandling:
    """Unit tests for error handling scenarios."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test handling of timeout errors."""
        client = RAGFlowClient("http://test.com", "test_key")
        client.session = AsyncMock()
        
        import asyncio
        import aiohttp
        client.session.request.side_effect = asyncio.TimeoutError()
        
        result = await client._make_request("GET", "/test")
        assert "error" in result
        assert result["code"] == 500

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_http_error(self):
        """Test handling of HTTP errors."""
        client = RAGFlowClient("http://test.com", "test_key")
        client.session = AsyncMock()
        
        # Create a proper mock response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        # Set up the session.request to return the mock response
        client.session.request.return_value = mock_response
        
        result = await client._make_request("GET", "/test")
        assert "error" in result

    @pytest.mark.unit
    def test_servicer_generic_exception(self):
        """Test generic exception handling in servicer."""
        servicer = RagServicesServicer()
        
        # Test with a generic exception
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_response = servicer._handle_error({"error": e})
            assert error_response.code == 500
            assert "Test error" in error_response.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
