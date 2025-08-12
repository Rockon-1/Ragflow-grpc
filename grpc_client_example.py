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
    level=logging.DEBUG,
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
            logger.debug(f"Updated dataset {dataset_id}")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to update dataset: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def delete_datasets(self, dataset_ids: List[str]) -> Dict:
        """Delete datasets by IDs"""
        print("Deleting datasets with IDs:", dataset_ids)
        request = pb2.DeleteDatasetsRequest(ids=[dataset_ids])
        
        try:
            response = await self.stub.DeleteDatasets(request)
            logger.debug(f"Got server response for deleting datasets: {dataset_ids}")
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
    
    async def parse_documents(self, dataset_id: str, document_ids: List[str]) -> Dict:
        """Parse documents in a dataset"""
        request = pb2.ParseDocumentsRequest(
            dataset_id=dataset_id,
            document_ids=document_ids
        )
        
        try:
            response = await self.stub.ParseDocuments(request)
            logger.info(f"Started parsing {len(document_ids)} documents in dataset {dataset_id}")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to parse documents: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def stop_parsing_documents(self, dataset_id: str, document_ids: List[str]) -> Dict:
        """Stop parsing documents in a dataset"""
        request = pb2.StopParsingDocumentsRequest(
            dataset_id=dataset_id,
            document_ids=document_ids
        )
        
        try:
            response = await self.stub.StopParsingDocuments(request)
            logger.info(f"Stopped parsing {len(document_ids)} documents in dataset {dataset_id}")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to stop parsing documents: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def delete_documents(self, dataset_id: str, document_ids: List[str]) -> Dict:
        """Delete documents from a dataset"""
        request = pb2.DeleteDocumentsRequest(
            dataset_id=dataset_id,
            ids=document_ids
        )
        
        try:
            response = await self.stub.DeleteDocuments(request)
            logger.info(f"Deleted {len(document_ids)} documents from dataset {dataset_id}")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to delete documents: {e}")
            return {"error": str(e), "code": e.code()}
    
    # Chunk Management Methods
    async def add_chunk(self, dataset_id: str, document_id: str, content: str, 
                       important_keywords: List[str] = None, questions: List[str] = None) -> Dict:
        """Add a chunk to a document"""
        request = pb2.AddChunkRequest(
            dataset_id=dataset_id,
            document_id=document_id,
            content=content
        )
        
        if important_keywords:
            request.important_keywords.extend(important_keywords)
        if questions:
            request.questions.extend(questions)
        
        try:
            response = await self.stub.AddChunk(request)
            logger.info(f"Added chunk to document {document_id} in dataset {dataset_id}")
            return {
                "code": response.code,
                "message": response.message,
                "data": {
                    "chunk_id": response.data.chunk.id,
                    "content": response.data.chunk.content,
                    "create_time": response.data.chunk.create_time
                }
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to add chunk: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def list_chunks(self, dataset_id: str, document_id: str, page: int = 1, 
                         page_size: int = 10, keywords: str = "") -> Dict:
        """List chunks in a document"""
        request = pb2.ListChunksRequest(
            dataset_id=dataset_id,
            document_id=document_id,
            page=page,
            page_size=page_size,
            keywords=keywords
        )
        
        try:
            response = await self.stub.ListChunks(request)
            chunks = []
            for chunk in response.data.chunks:
                chunks.append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "available": chunk.available,
                    "create_time": chunk.create_time,
                    "important_keywords": chunk.important_keywords
                })
            
            logger.info(f"Got {len(chunks)} chunks from list_chunks {document_id}")
            return {
                "code": response.code,
                "message": response.message,
                "data": {
                    "chunks": chunks,
                    "total": response.data.total,
                    "document": {
                        "id": response.data.doc.id,
                        "name": response.data.doc.name,
                        "chunk_count": response.data.doc.chunk_count
                    }
                }
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to list chunks: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def update_chunk(self, dataset_id: str, document_id: str, chunk_id: str,
                          content: str = None, important_keywords: List[str] = None,
                          available: bool = None) -> Dict:
        """Update a chunk"""
        request = pb2.UpdateChunkRequest(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_id=chunk_id
        )
        
        if content:
            request.content = content
        if important_keywords:
            request.important_keywords.extend(important_keywords)
        if available is not None:
            request.available = available
        
        try:
            response = await self.stub.UpdateChunk(request)
            logger.info(f"Updated chunk {chunk_id} in document {document_id}")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to update chunk: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def delete_chunks(self, dataset_id: str, document_id: str, chunk_ids: List[str]) -> Dict:
        """Delete chunks from a document"""
        request = pb2.DeleteChunksRequest(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_ids=chunk_ids
        )
        
        try:
            response = await self.stub.DeleteChunks(request)
            logger.info(f"Deleted {len(chunk_ids)} chunks from document {document_id}")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to delete chunks: {e}")
            return {"error": str(e), "code": e.code()}
    
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
    
    # Chat Assistant Management Methods
    async def create_chat_assistant(self, name: str, dataset_ids: List[str], 
                                   avatar: str = "", description: str = "") -> Dict:
        """Create a chat assistant"""
        request = pb2.CreateChatAssistantRequest(
            name=name,
            avatar=avatar,
            dataset_ids=dataset_ids
        )
        
        # Set default prompt configuration
        request.prompt.similarity_threshold = 0.2
        request.prompt.keywords_similarity_weight = 0.3
        request.prompt.top_n = 6
        request.prompt.top_k = 1024
        request.prompt.empty_response = "Sorry! No relevant content was found in the knowledge base!"
        request.prompt.opener = "Hi! I'm your assistant, what can I do for you?"
        request.prompt.show_quote = True
        request.prompt.prompt = "You are an intelligent assistant. Please summarize the content of the knowledge base to answer the question."
        
        try:
            response = await self.stub.CreateChatAssistant(request)
            logger.info(f"Created chat assistant: {response.data.name} (ID: {response.data.id})")
            return {
                "code": response.code,
                "message": response.message,
                "data": {
                    "id": response.data.id,
                    "name": response.data.name,
                    "description": response.data.description,
                    "create_date": response.data.create_date
                }
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to create chat assistant: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def list_chat_assistants(self, page: int = 1, page_size: int = 10) -> Dict:
        """List chat assistants"""
        request = pb2.ListChatAssistantsRequest(
            page=page,
            page_size=page_size
        )
        
        try:
            response = await self.stub.ListChatAssistants(request)
            assistants = []
            for assistant in response.data:
                assistants.append({
                    "id": assistant.id,
                    "name": assistant.name,
                    "description": assistant.description,
                    "status": assistant.status,
                    "create_date": assistant.create_date,
                    "dataset_ids": list(assistant.dataset_ids)
                })
            
            logger.info(f"Retrieved {len(assistants)} chat assistants")
            return {
                "code": response.code,
                "message": response.message,
                "data": assistants
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to list chat assistants: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def update_chat_assistant(self, chat_id: str, name: str = None, 
                                   dataset_ids: List[str] = None) -> Dict:
        """Update a chat assistant"""
        request = pb2.UpdateChatAssistantRequest(chat_id=chat_id)
        
        if name:
            request.name = name
        if dataset_ids:
            request.dataset_ids.extend(dataset_ids)
        
        try:
            response = await self.stub.UpdateChatAssistant(request)
            logger.info(f"Updated chat assistant: {chat_id}")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to update chat assistant: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def delete_chat_assistants(self, assistant_ids: List[str]) -> Dict:
        """Delete chat assistants"""
        request = pb2.DeleteChatAssistantsRequest(ids=assistant_ids)
        
        try:
            response = await self.stub.DeleteChatAssistants(request)
            logger.info(f"Deleted {len(assistant_ids)} chat assistants")
            return {
                "code": response.code,
                "message": response.message
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to delete chat assistants: {e}")
            return {"error": str(e), "code": e.code()}
    
    # Session Management Methods
    async def create_session_with_chat_assistant(self, chat_id: str, name: str, user_id: str = "") -> Dict:
        """Create a session with a chat assistant"""
        request = pb2.CreateSessionWithChatAssistantRequest(
            chat_id=chat_id,
            name=name,
            user_id=user_id
        )
        
        try:
            response = await self.stub.CreateSessionWithChatAssistant(request)
            logger.info(f"Created session: {response.data.name} (ID: {response.data.id})")
            return {
                "code": response.code,
                "message": response.message,
                "data": {
                    "id": response.data.id,
                    "name": response.data.name,
                    "chat_id": response.data.chat_id,
                    "create_date": response.data.create_date
                }
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to create session: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def list_chat_assistant_sessions(self, chat_id: str, page: int = 1, page_size: int = 10) -> Dict:
        """List sessions for a chat assistant"""
        request = pb2.ListChatAssistantSessionsRequest(
            chat_id=chat_id,
            page=page,
            page_size=page_size
        )
        
        try:
            response = await self.stub.ListChatAssistantSessions(request)
            sessions = []
            for session in response.data:
                sessions.append({
                    "id": session.id,
                    "name": session.name,
                    "chat_id": session.chat_id,
                    "create_date": session.create_date,
                    "message_count": len(session.messages)
                })
            
            logger.info(f"Retrieved {len(sessions)} sessions for chat assistant {chat_id}")
            return {
                "code": response.code,
                "message": response.message,
                "data": sessions
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to list sessions: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def converse_with_chat_assistant(self, chat_id: str, question: str, 
                                          session_id: str = "", stream: bool = True):
        """Converse with a chat assistant"""
        request = pb2.ConverseWithChatAssistantRequest(
            chat_id=chat_id,
            question=question,
            stream=stream,
            session_id=session_id
        )
        
        try:
            logger.info(f"Starting conversation with chat assistant: {chat_id}")
            
            async for response in self.stub.ConverseWithChatAssistant(request):
                if response.code != 0:
                    logger.error(f"Conversation error: {response.message}")
                    yield {"error": response.message, "code": response.code}
                else:
                    result = {
                        "answer": response.data.answer,
                        "reference": response.data.reference,
                        "session_id": response.data.session_id
                    }
                    yield result
        
        except grpc.RpcError as e:
            logger.error(f"Failed to converse with chat assistant: {e}")
            yield {"error": str(e), "code": e.code()}
    
    async def create_agent(self, title: str, description: str = "", dsl: str = "{}") -> Dict:
        """Create a new agent"""
        request = pb2.CreateAgentRequest(
            title=title,
            description=description,
            dsl=dsl
        )
        
        try:
            response = await self.stub.CreateAgent(request)
            logger.info(f"Created agent: {title}")
            return {
                "code": response.code,
                "message": response.message,
                "data": response.data
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to create agent: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def update_agent(self, agent_id: str, title: str = None, 
                          description: str = None, dsl: str = None) -> Dict:
        """Update an agent"""
        request = pb2.UpdateAgentRequest(agent_id=agent_id)
        
        if title:
            request.title = title
        if description:
            request.description = description
        if dsl:
            request.dsl = dsl
        
        try:
            response = await self.stub.UpdateAgent(request)
            logger.info(f"Updated agent: {agent_id}")
            return {
                "code": response.code,
                "message": response.message,
                "data": response.data
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to update agent: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def delete_agent(self, agent_id: str) -> Dict:
        """Delete an agent"""
        request = pb2.DeleteAgentRequest(agent_id=agent_id)
        
        try:
            response = await self.stub.DeleteAgent(request)
            logger.info(f"Deleted agent: {agent_id}")
            return {
                "code": response.code,
                "message": response.message,
                "data": response.data
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to delete agent: {e}")
            return {"error": str(e), "code": e.code()}
    
    async def converse_with_agent(self, agent_id: str, question: str, 
                                 session_id: str = "", stream: bool = True, 
                                 parameters: Dict[str, str] = None):
        """Converse with an agent"""
        request = pb2.ConverseWithAgentRequest(
            agent_id=agent_id,
            question=question,
            stream=stream,
            session_id=session_id,
            sync_dsl=False
        )
        
        if parameters:
            for key, value in parameters.items():
                request.parameters[key] = value
        
        try:
            logger.info(f"Starting conversation with agent: {agent_id}")
            
            async for response in self.stub.ConverseWithAgent(request):
                if response.code != 0:
                    logger.error(f"Agent conversation error: {response.message}")
                    yield {"error": response.message, "code": response.code}
                else:
                    result = {
                        "answer": response.data.answer,
                        "reference": response.data.reference,
                        "session_id": response.data.session_id
                    }
                    yield result
        
        except grpc.RpcError as e:
            logger.error(f"Failed to converse with agent: {e}")
            yield {"error": str(e), "code": e.code()}
    
    async def generate_related_questions(self, question: str, industry: str = "general") -> Dict:
        """Generate related questions"""
        request = pb2.GenerateRelatedQuestionsRequest(
            question=question,
            industry=industry
        )
        
        try:
            response = await self.stub.GenerateRelatedQuestions(request)
            logger.info(f"Generated {len(response.data)} related questions")
            return {
                "code": response.code,
                "message": response.message,
                "data": list(response.data)
            }
        except grpc.RpcError as e:
            logger.error(f"Failed to generate related questions: {e}")
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
            if response.code >= 400 or (response.code < 200 and response.code != 0):
                logger.error(f"Error listing agents: {response.message}")
                return {"error": response.message, "code": response.code}
            for agent in response.data:
                agents.append({
                    "id": agent.id,
                    "title": agent.title,
                    "description": agent.description,
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


                    #Now Deleting new Dataset
                    logger.info(f"Deleting dataset with ID: {dataset_id}")
                    delete_result = await self.client.delete_datasets(dataset_id)
                    logger.debug(f"Delete result: {json.dumps(delete_result, indent=2)}")
                    if delete_result.get('code') == 0:
                        logger.info("Dataset deleted successfully")

            # Get knowledge graph
            dataset_id = datasets["data"][0].get('id')
            kg_result = await self.client.get_dataset_knowledge_graph(dataset_id)
            logger.info(f"Knowledge graph: {json.dumps(kg_result, indent=2)}")
            logger.info(f"Dataset operations completed successfully {dataset_id}")
            logger.info(f" {"==="*5}")
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
                question="What is task?",
                dataset_ids=dataset_ids[:2],  # Use first 2 datasets
                top_k=5,
                similarity_threshold=0.2
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
                chat_id="2e5803da761011f0a638ea9746a4f026",
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
            print("Available agents:", agents)
            logger.info(f"Available agents: {json.dumps(agents, indent=2)}")
        except Exception as e:
            logger.error(f"Agent operations failed: {e}",repr(e))
    
    async def test_document_parsing_operations(self, dataset_id: str):
        """Test document parsing operations"""
        if not dataset_id:
            logger.warning("No dataset ID provided, skipping document parsing tests")
            return
            
        logger.info("=== Testing Document Parsing Operations ===")
        
        try:
            # List documents to get document IDs for parsing
            docs_result = await self.client.list_documents(dataset_id)
            if docs_result.get('code') == 0 and docs_result.get('data'):
                document_ids = [doc['id'] for doc in docs_result['data'][:2]]  # Take first 2 documents
                
                if document_ids:
                    # Start parsing documents
                    parse_result = await self.client.parse_documents(dataset_id, document_ids)
                    logger.info(f"Parse documents result: {json.dumps(parse_result, indent=2)}")
                    
                    # Wait a moment then stop parsing (for demo)
                    await asyncio.sleep(2)
                    
                    stop_result = await self.client.stop_parsing_documents(dataset_id, document_ids)
                    logger.info(f"Stop parsing result: {json.dumps(stop_result, indent=2)}")
                else:
                    logger.info("No documents found for parsing test")
            else:
                logger.warning("Could not retrieve documents for parsing test")
                
        except Exception as e:
            logger.error(f"Document parsing operations failed: {e}")
    
    async def test_chunk_management_operations(self, dataset_id: str):
        """Test chunk management operations"""
        if not dataset_id:
            logger.warning("No dataset ID provided, skipping chunk management tests")
            return
            
        logger.info("=== Testing Chunk Management Operations ===")
        
        try:
            # Get documents to work with chunks
            docs_result = await self.client.list_documents(dataset_id)
            if docs_result.get('code') == 0 and docs_result.get('data'):
                document_id = docs_result['data'][0]['id']
                logger.info(f"Using document ID: {document_id}")
                
                # Add a new chunk
                add_result = await self.client.add_chunk(
                    dataset_id=dataset_id,
                    document_id=document_id,
                    content="This is a test chunk created via gRPC client.",
                    important_keywords=["test", "grpc", "chunk"],
                    questions=["What is this chunk about?"]
                )
                logger.info(f"Add chunk result: {json.dumps(add_result, indent=2)}")
                
                # If we successfully added a chunk, try to update and delete it
                if add_result.get('code') == 0 and add_result.get('data', {}).get('chunk_id'):
                    chunk_id = add_result['data']['chunk_id']
                    
                    # Update the chunk
                    update_result = await self.client.update_chunk(
                        dataset_id=dataset_id,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        content="Updated test chunk content via gRPC client(made by Sachin).",
                        important_keywords=["updated", "test", "grpc"]
                    )
                    logger.debug(f"Update chunk result: {json.dumps(update_result, indent=2)}")
                    if update_result.get('code') == 0:
                        logger.info("Chunk updated successfully")
                    else:
                        logger.warning(f"Failed to update chunk: {update_result.get('message', 'Unknown error')}")

                    # List chunks in the document
                    chunks_result = await self.client.list_chunks(
                        dataset_id=dataset_id,
                        document_id=document_id,
                        page=1,
                        page_size=5
                    )
                    logger.info(f"List chunks result: {json.dumps(chunks_result, indent=2)}")
                

                    # Delete the chunk
                    delete_result = await self.client.delete_chunks(
                        dataset_id=dataset_id,
                        document_id=document_id,
                        chunk_ids=[chunk_id]
                    )
                    logger.info(f"Delete chunk result: {json.dumps(delete_result, indent=2)}")
                
            else:
                logger.warning("No documents found for chunk management test")
                
        except Exception as e:
            logger.error(f"Chunk management operations failed: {e}")
    
    async def test_chat_assistant_operations(self):
        """Test chat assistant management operations"""
        logger.info("=== Testing Chat Assistant Operations ===")
        
        try:
            # Get datasets to use with chat assistant
            datasets_result = await self.client.list_datasets()
            if datasets_result.get('code') == 0 and datasets_result.get('data'):
                dataset_ids = [ds['id'] for ds in datasets_result['data'][:2]]  # Use first 2 datasets
                
                # Create a chat assistant
                create_result = await self.client.create_chat_assistant(
                    name="gRPC Test Assistant",
                    dataset_ids=dataset_ids,
                    description="A test chat assistant created via gRPC"
                )
                logger.info(f"Create chat assistant result: {json.dumps(create_result, indent=2)}")
                
                # List chat assistants
                list_result = await self.client.list_chat_assistants()
                logger.info(f"List chat assistants result: {json.dumps(list_result, indent=2)}")
                
                # If we successfully created an assistant, test other operations
                if create_result.get('code') == 0 and create_result.get('data', {}).get('id'):
                    chat_id = create_result['data']['id']
                    
                    # Update the chat assistant
                    update_result = await self.client.update_chat_assistant(
                        chat_id=chat_id,
                        name="Updated gRPC Test Assistant"
                    )
                    logger.info(f"Update chat assistant result: {json.dumps(update_result, indent=2)}")
                    
                    # Test session operations
                    await self.test_session_operations(chat_id)
                    
                    # Clean up - delete the chat assistant
                    delete_result = await self.client.delete_chat_assistants([chat_id])
                    logger.info(f"Delete chat assistant result: {json.dumps(delete_result, indent=2)}")
                
            else:
                logger.warning("No datasets found for chat assistant test")
                
        except Exception as e:
            logger.error(f"Chat assistant operations failed: {e}")
    
    async def test_session_operations(self, chat_id: str):
        """Test session management operations"""
        if not chat_id:
            logger.warning("No chat ID provided, skipping session tests")
            return
            
        logger.info("=== Testing Session Operations ===")
        
        try:
            # Create a session with the chat assistant
            session_result = await self.client.create_session_with_chat_assistant(
                chat_id=chat_id,
                name="Test Session via gRPC",
                user_id="grpc_test_user"
            )
            logger.info(f"Create session result: {json.dumps(session_result, indent=2)}")
            
            # List sessions for the chat assistant
            sessions_list = await self.client.list_chat_assistant_sessions(chat_id)
            logger.info(f"List sessions result: {json.dumps(sessions_list, indent=2)}")
            
            # Test conversation with chat assistant
            if session_result.get('code') == 0 and session_result.get('data', {}).get('id'):
                session_id = session_result['data']['id']
                
                logger.info("Testing conversation with chat assistant...")
                conversation_count = 0
                async for response in self.client.converse_with_chat_assistant(
                    chat_id=chat_id,
                    question="Hello! Can you tell me about the documents in the knowledge base?",
                    session_id=session_id
                ):
                    conversation_count += 1
                    logger.info(f"Conversation response {conversation_count}: {json.dumps(response, indent=2)}")
                    
                    # Limit responses for demo
                    if conversation_count >= 2:
                        break
                
        except Exception as e:
            logger.error(f"Session operations failed: {e}")
    
    async def test_agent_lifecycle_operations(self):
        """Test complete agent lifecycle operations"""
        logger.info("=== Testing Agent Lifecycle Operations ===")
        
        try:
            # Create a new agent
            agent_dsl = {
                "components": {
                    "begin": {"obj": {"component_name": "Begin"}},
                    "answer": {"obj": {"component_name": "Answer"}}
                }
            }
            
            create_result = await self.client.create_agent(
                title="gRPC Test Agent",
                description="A test agent created via gRPC client",
                dsl=json.dumps(agent_dsl)
            )
            logger.info(f"Create agent result: {json.dumps(create_result, indent=2)}")
            
            # List agents to verify creation
            agents_list = await self.client.list_agents()
            logger.info(f"List agents result: {json.dumps(agents_list, indent=2)}")
            
            # If creation was successful, test other operations
            if create_result.get('code') == 0:
                # Find our created agent (this is a simplified approach)
                agent_id = None
                if agents_list.get('code') == 0 and agents_list.get('data'):
                    for agent in agents_list['data']:
                        if agent['title'] == "gRPC Test Agent":
                            agent_id = agent['id']
                            break
                
                if agent_id:
                    # Update the agent
                    update_result = await self.client.update_agent(
                        agent_id=agent_id,
                        title="Updated gRPC Test Agent",
                        description="Updated description for test agent"
                    )
                    logger.info(f"Update agent result: {json.dumps(update_result, indent=2)}")
                    
                    # Test conversation with agent
                    logger.info("Testing conversation with agent...")
                    conversation_count = 0
                    async for response in self.client.converse_with_agent(
                        agent_id=agent_id,
                        question="Hello! What can you do?",
                        parameters={"test_param": "test_value"}
                    ):
                        conversation_count += 1
                        logger.info(f"Agent conversation response {conversation_count}: {json.dumps(response, indent=2)}")
                        
                        # Limit responses for demo
                        if conversation_count >= 2:
                            break
                    
                    # Clean up - delete the agent
                    delete_result = await self.client.delete_agent(agent_id)
                    logger.info(f"Delete agent result: {json.dumps(delete_result, indent=2)}")
                
        except Exception as e:
            logger.error(f"Agent lifecycle operations failed: {e}")
    
    async def test_advanced_features(self):
        """Test advanced features like related question generation"""
        logger.info("=== Testing Advanced Features ===")
        
        try:
            # Test related question generation
            questions_result = await self.client.generate_related_questions(
                question="What are the benefits of using RAGFlow?",
                industry="software_development"
            )
            logger.info(f"Related questions result: {json.dumps(questions_result, indent=2)}")
            
        except Exception as e:
            logger.error(f"Advanced features test failed: {e}")


async def main():
    """Main function to run all examples"""
    logger.info("Starting RAGFlow gRPC Client Examples")
    
    try:
        async with RAGFlowGRPCExamples() as examples:
            # Test dataset operations
            dataset_id = await examples.test_dataset_operations()
            
            # Test document operations
            await examples.test_document_operations(dataset_id)
            
            # Test chunk management operations
            await examples.test_chunk_management_operations(dataset_id)
            
            # Test chunk retrieval (using any available datasets)
            datasets = await examples.client.list_datasets()
            if datasets.get('code') == 0 and datasets.get('data'):
                dataset_ids = [ds['id'] for ds in datasets['data'] if ds['chunk_count'] != 0]
                await examples.test_chunk_retrieval(dataset_ids)
            
            # Test chat assistant operations
            await examples.test_chat_assistant_operations()
            
            # Test agent lifecycle operations
            await examples.test_agent_lifecycle_operations()
            
            # Test traditional agent operations
            await examples.test_agent_operations()
            
            # Test advanced features
            await examples.test_advanced_features()
            
            # Test chat completion
            await examples.test_chat_completion()
        
        logger.info("All examples completed successfully!")
    
    except Exception as e:
        logger.error(f"Examples failed with error: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
