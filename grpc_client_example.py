"""
RAGFlow gRPC Client Example

This module provides comprehensive examples of how to use the RAGFlow gRPC service
with async client calls for all major RPC methods.
"""
import asyncio
import grpc
import json
import logging
from typing import Dict, List, Optional, AsyncIterator

# Import the generated gRPC modules
from grpc_ragflow_server import ragflow_service_pb2 as pb2
from grpc_ragflow_server import ragflow_service_pb2_grpc as pb2_grpc
from grpc_ragflow_server.config import GRPC_HOST, GRPC_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGFlowGRPCClient:
    """Async gRPC client for RAGFlow service"""
    
    def __init__(self, host: str = GRPC_HOST, port: int = GRPC_PORT):
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Create async gRPC channel
        self.channel = grpc.aio.insecure_channel(f'{self.host}:{self.port}')

        # Wait for channel to be ready
        try:
            await self.channel.channel_ready()
            self.stub = pb2_grpc.RagServicesStub(self.channel)
            logger.info(f"Connected to gRPC server at {self.host}:{self.port}")
        except grpc.aio.AioRpcError as e:
            logger.error(f"Channel connection failed: {e}")
            print(f"Channel connection failed: {e}")
            raise
        
        except grpc.RpcError as e:
            logger.error(f"Failed to connect to gRPC server: {e}")
            raise
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from gRPC server")
    
    # Dataset Management Methods
    async def create_dataset(self, name: str, description: str = "", 
                           embedding_model: str = "BAAI/bge-large-zh-v1.5",
                           permission: str = "me", chunk_method: str = "manual") -> Dict:
        """Create a new dataset"""
        request = pb2.CreateDatasetRequest(
            name=name,
            description=description,
            embedding_model=embedding_model,
            permission=permission,
            chunk_method=chunk_method
        )
        
        try:
            response = await self.stub.CreateDataset(request)
            logger.info(f"Created dataset: {response.data.name} (ID: {response.data.id})")
            return {
                "code": response.code,
                "message": response.message,
                "data": {
                    "id": response.data.id,
                    "name": response.data.name,
                    "description": response.data.description,
                    "embedding_model": response.data.embedding_model,
                    "chunk_count": response.data.chunk_count,
                    "document_count": response.data.document_count
                }
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to create dataset: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def list_datasets(self, page: int = 1, page_size: int = 10) -> Dict:
        """List all datasets"""
        request = pb2.ListDatasetsRequest(
            page=page,
            page_size=page_size
        )
        
        try:
            if not self.stub:
                self.stub = pb2_grpc.RagServicesStub(self.channel)
            response = await self.stub.ListDatasets(request)
            datasets = []

            if response.code >= 400 or (response.code < 200 and response.code != 0):
                print(f"Error in list_datasets: {response.message} (code: {response.code})")

            else:
                
                for dataset in response.data:
                    datasets.append({
                        "id": dataset.id,
                        "name": dataset.name,
                        "description": dataset.description,
                        "embedding_model": dataset.embedding_model,
                        "chunk_count": dataset.chunk_count,
                        "document_count": dataset.document_count,
                        "create_date": dataset.create_date
                    })

                logger.info(f"Retrieved {len(datasets)} datasets")
            return {
                "code": response.code,
                "message": response.message,
                "data": datasets
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to list datasets: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def update_dataset(self, dataset_id: str, name: str = None, 
                           description: str = None) -> Dict:
        """Update an existing dataset"""
        request = pb2.UpdateDatasetRequest(
            dataset_id=dataset_id
        )
        
        if name:
            request.name = name
        if description:
            request.description = description
        
        try:
            response = await self.stub.UpdateDataset(request)
            logger.info(f"Updated dataset {dataset_id}")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to update dataset: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def delete_datasets(self, dataset_ids: List[str]) -> Dict:
        """Delete datasets by IDs"""
        request = pb2.DeleteDatasetsRequest(ids=dataset_ids)
        
        try:
            response = await self.stub.DeleteDatasets(request)
            logger.info(f"Deleted {len(dataset_ids)} datasets")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to delete datasets: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def get_dataset_knowledge_graph(self, dataset_id: str) -> Dict:
        """Get knowledge graph for a dataset"""
        request = pb2.GetDatasetKnowledgeGraphRequest(dataset_id=dataset_id)
        
        try:
            response = await self.stub.GetDatasetKnowledgeGraph(request)
            logger.info(f"Retrieved knowledge graph for dataset {dataset_id}")
            return {
                "code": response.code,
                "message": response.message,
                "data": {
                    "graph": json.loads(response.data.graph) if response.data.graph else {},
                    "mind_map": json.loads(response.data.mind_map) if response.data.mind_map else {}
                }
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to get knowledge graph: {e}")
            return {"error": str(e), "code": e.code()}
    
    # Document Management Methods
    async def upload_documents(self, dataset_id: str, file_paths: List[str]) -> Dict:
        """Upload documents to a dataset"""
        document_files = []
        for file_path in file_paths:
            # Read file content (for demo, using text files)
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                doc_file = pb2.DocumentFile(
                    name=file_path.split('\\')[-1],  # Extract filename
                    content=content
                )
                document_files.append(doc_file)
            except FileNotFoundError:
                logger.warning(f"File not found: {file_path}")
                continue
        
        request = pb2.UploadDocumentsRequest(
            dataset_id=dataset_id,
            files=document_files
        )
        
        try:
            response = await self.stub.UploadDocuments(request)
            logger.info(f"Uploaded {len(document_files)} documents to dataset {dataset_id}")
            return {
                "code": response.code,
                "message": response.message,
                "data": response.data
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to upload documents: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def list_documents(self, dataset_id: str, page: int = 1, page_size: int = 10) -> Dict:
        """List documents in a dataset"""
        request = pb2.ListDocumentsRequest(
            dataset_id=dataset_id,
            page=page,
            page_size=page_size
        )
        
        try:
            response = await self.stub.ListDocuments(request)
            documents = []
            if response.code >= 400 or (response.code < 200 and response.code != 0):
                logger.error(f"Failed to list documents: {response.message}")
                return {"error": response.message, "code": response.code}

            for doc in response.data.docs:
                documents.append({
                    "id": doc.id,
                    "name": doc.name,
                    "type": doc.type,
                    "size": doc.size,
                    "status": doc.status,
                    "create_date": doc.create_date
                })
            
            logger.info(f"Retrieved {len(documents)} documents from dataset {dataset_id}")
            return {
                "code": response.code,
                "message": response.message,
                "data": documents
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to list documents: {e}")
            return {"error": str(e), "code": e.code()}
    
    # Chunk Management Methods
    async def retrieve_chunks(self, question: str, dataset_ids: List[str], 
                            top_k: int = 5, similarity_threshold: float = 0.7) -> Dict:
        """Retrieve relevant chunks for a question"""
        request = pb2.RetrieveChunksRequest(
            question=question,
            dataset_ids=dataset_ids,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        try:
            response = await self.stub.RetrieveChunks(request)
            chunks = []
            for chunk in response.data.chunks:
                chunks.append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "similarity": chunk.similarity,
                    "kb_id": chunk.kb_id,
                    "document_id": chunk.document_id
                })
            
            logger.info(f"Retrieved {len(chunks)} chunks for question: '{question}'")
            return {
                "code": response.code,
                "message": response.message,
                "data": {
                    "chunks": chunks,
                    "total": response.data.total
                }
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            return {"error": str(e), "code": e.code()}
    
    # Chat Completion Methods
    async def create_chat_completion(self, chat_id: str, model: str, 
                                   messages: List[Dict[str, str]], stream: bool = True):
        """Create chat completion with streaming response"""
        # Convert messages to protobuf format
        pb_messages = []
        for msg in messages:
            pb_messages.append(pb2.ChatMessage(
                role=msg["role"],
                content=msg["content"]
            ))
        
        request = pb2.CreateChatCompletionRequest(
            chat_id=chat_id,
            model=model,
            messages=pb_messages,
            stream=stream
        )
        
        try:
            logger.info(f"Starting chat completion for chat_id: {chat_id}")
            
            # Handle streaming response
            async for response in self.stub.CreateChatCompletion(request):
                if response.error.code != 0:
                    logger.error(f"Chat completion error: {response.error.message}")
                    yield {"error": response.error.message, "code": response.error.code}
                else:
                    # Extract the response data
                    result = {
                        "id": response.id,
                        "object": response.object,
                        "created": response.created,
                        "model": response.model,
                        "choices": []
                    }
                    
                    for choice in response.choices:
                        choice_data = {
                            "index": choice.index,
                            "finish_reason": choice.finish_reason
                        }
                        
                        if choice.message.content:
                            choice_data["message"] = {
                                "role": choice.message.role,
                                "content": choice.message.content
                            }
                        
                        if choice.delta.content:
                            choice_data["delta"] = {
                                "role": choice.delta.role,
                                "content": choice.delta.content
                            }
                        
                        result["choices"].append(choice_data)
                    
                    yield result
        
        except grpc.RpcError as e:
            logger.error(f"Failed to create chat completion: {e}")
            yield {"error": str(e), "code": e.code()}
    
    # Agent Management Methods
    async def list_agents(self, page: int = 1, page_size: int = 10) -> Dict:
        """List available agents"""
        request = pb2.ListAgentsRequest(
            page=page,
            page_size=page_size
        )
        
        try:
            response = await self.stub.ListAgents(request)
            agents = []
            for agent in response.data:
                agents.append({
                    "id": agent.id,
                    "name": agent.name,
                    "description": agent.description,
                    "status": agent.status,
                    "create_date": agent.create_date
                })
            
            logger.info(f"Retrieved {len(agents)} agents")
            return {
                "code": response.code,
                "message": response.message,
                "data": agents
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to list agents: {e}")
            return {"error": str(e), "code": e.code()}


class RAGFlowGRPCExamples:
    """Example usage of RAGFlow gRPC client"""
    
    def __init__(self):
        self.client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = RAGFlowGRPCClient()
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def test_dataset_operations(self):
        """Test complete dataset lifecycle"""
        logger.info("=== Testing Dataset Operations ===")
        
        try:
            # List existing datasets
            datasets = await self.client.list_datasets()
            logger.info(f"Current datasets: {json.dumps(datasets, indent=2)}")
            
            # Create a new dataset
            dataset_result = await self.client.create_dataset(
                name="gRPC Test Dataset for Chunk",
                description="A test dataset created via gRPC client",
                embedding_model="BAAI/bge-large-zh-v1.5@BAAI",
                permission="me",
                chunk_method="manual"
            )
            logger.info(f"Create dataset result: {json.dumps(dataset_result, indent=2)}")
            
            if dataset_result.get('code') == 0:
                dataset_id = dataset_result.get('data', {}).get('id')
                if dataset_id:
                    # Update the dataset
                    update_result = await self.client.update_dataset(
                        dataset_id, 
                        description="Updated description via gRPC"
                    )
                    logger.info(f"Update result: {json.dumps(update_result, indent=2)}")
                    if update_result.get('code') == 0:
                        logger.info("Dataset updated successfully")

                    # Get knowledge graph

                    #test - will removed
                    dataset_id = datasets["data"][0].get('id')

                    kg_result = await self.client.get_dataset_knowledge_graph(dataset_id)
                    logger.info(f"Knowledge graph: {json.dumps(kg_result, indent=2)}")
                    
                    return dataset_id
            
            return None
        except Exception as e:
            logger.error(f"Dataset operations failed: {e}")
            return None
    
    async def test_document_operations(self, dataset_id: str):
        """Test document management operations"""
        if not dataset_id:
            logger.warning("No dataset ID provided, skipping document tests")
            return
        
        logger.info("=== Testing Document Operations ===")
        
        try:
            # List documents in the dataset
            docs = await self.client.list_documents(dataset_id)
            logger.info(f"Below response received for list documents: \n",docs)
            logger.info(f"Documents in dataset: {json.dumps(docs, indent=2)}")
            
            # Note: Upload documents would require actual files
            # You can uncomment and modify this if you have test files
            # upload_result = await self.client.upload_documents(
            #     dataset_id, 
            #     ["path/to/test1.txt", "path/to/test2.pdf"]
            # )
            # logger.info(f"Upload result: {json.dumps(upload_result, indent=2)}")
        except Exception as e:
            logger.error(f"Document operations failed: {e}", repr(e))

    async def test_chunk_retrieval(self, dataset_ids: List[str]):
        """Test chunk retrieval"""
        if not dataset_ids:
            logger.warning("No dataset IDs provided, skipping chunk retrieval tests")
            return
        
        logger.info("=== Testing Chunk Retrieval ===")
        
        try:
            # Retrieve chunks for a sample question
            chunks_result = await self.client.retrieve_chunks(
                question="What is machine learning and artificial intelligence?",
                dataset_ids=dataset_ids[:2],  # Use first 2 datasets
                top_k=5,
                similarity_threshold=0.7
            )
            logger.info(f"Retrieved chunks: {json.dumps(chunks_result, indent=2)}")
        except Exception as e:
            logger.error(f"Chunk retrieval failed: {e}")
    
    async def test_chat_completion(self):
        """Test streaming chat completion"""
        logger.info("=== Testing Chat Completion ===")
        
        try:
            messages = [
                {"role": "user", "content": "Hello! Can you help me understand RAGFlow?"}
            ]
            
            logger.info("Starting streaming chat completion...")
            chunk_count = 0
            async for chunk in self.client.create_chat_completion(
                chat_id="test-grpc-chat",
                model="deepseek-chat",
                messages=messages,
                stream=True
            ):
                chunk_count += 1
                logger.info(f"Chunk {chunk_count}: {json.dumps(chunk, indent=2)}")
                
                # Limit output for demo
                if chunk_count >= 3:
                    logger.info("Stopping after 3 chunks for demo purposes...")
                    break
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
    
    async def test_agent_operations(self):
        """Test agent management operations"""
        logger.info("=== Testing Agent Operations ===")
        
        try:
            # List available agents
            agents = await self.client.list_agents()
            logger.info(f"Available agents: {json.dumps(agents, indent=2)}")
        except Exception as e:
            logger.error(f"Agent operations failed: {e}")


async def main():
    """Main function to run all examples"""
    logger.info("Starting RAGFlow gRPC Client Examples")
    
    try:
        async with RAGFlowGRPCExamples() as examples:
            # Test dataset operations
            dataset_id = await examples.test_dataset_operations()
            
            # Test document operations
            await examples.test_document_operations(dataset_id)
            
            # Test chunk retrieval (using any available datasets)
            datasets = await examples.client.list_datasets()
            if datasets.get('code') == 0 and datasets.get('data'):
                dataset_ids = [ds['id'] for ds in datasets['data']]
                await examples.test_chunk_retrieval(dataset_ids)
            
            # Test agent operations
            await examples.test_agent_operations()
            
            # Test chat completion
            await examples.test_chat_completion()
        
        logger.info("All examples completed successfully!")
    
    except Exception as e:
        logger.error(f"Examples failed with error: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
