import asyncio
import grpc
import json
from concurrent import futures
from typing import Dict, List, Optional, AsyncIterator
from . import ragflow_service_pb2 as pb2
from . import ragflow_service_pb2_grpc as pb2_grpc
from .ragflow_client import RAGFlowClient, RAGFlowConnectionError
from .config import GRPC_HOST, GRPC_PORT



class RagServicesServicer(pb2_grpc.RagServicesServicer):
    """Async gRPC servicer for RAGFlow API"""
    
    def __init__(self):
        self.client = None
    
    async def _get_client(self) -> RAGFlowClient:
        """Get or create RAGFlow client"""
        try:
            if self.client is None:
                self.client = RAGFlowClient()
                await self.client.__aenter__()
            return self.client
        except RAGFlowConnectionError as e:
            print(f"RAGFlow connection error: {e}")
            # Reset client so next call will retry
            self.client = None
            raise e
    
    def _handle_error(self, error_dict: Dict) -> pb2.ErrorResponse:
        """Convert error dict to protobuf ErrorResponse"""
        return pb2.ErrorResponse(
            code=error_dict.get('code', 500),
            message=error_dict.get('message', str(error_dict.get('error', 'Unknown error')))
        )
    
    def _handle_connection_error(self, error: RAGFlowConnectionError) -> pb2.ErrorResponse:
        """Handle RAGFlow connection errors"""
        return pb2.ErrorResponse(
            code=503,  # Service Unavailable
            message=f"RAGFlow service unavailable: {str(error)}"
        )
    
    # OpenAI-Compatible API
    async def CreateChatCompletion(self, request, context):
        """Create chat completion"""
        try:
            client = await self._get_client()
            
            # Convert protobuf request to dict
            data = {
                'model': request.model,
                'messages': [{'role': msg.role, 'content': msg.content} for msg in request.messages],
                'stream': request.stream
            }
            
            async for chunk in client.create_chat_completion(request.chat_id, data):
                if 'error' in chunk:
                    yield pb2.ChatCompletionResponse(error=self._handle_error(chunk))
                else:
                    # Convert response to protobuf
                    response = pb2.ChatCompletionResponse()
                    if 'id' in chunk:
                        response.id = chunk['id']
                    if 'object' in chunk:
                        response.object = chunk['object']
                    if 'created' in chunk:
                        response.created = chunk['created']
                    if 'model' in chunk:
                        response.model = chunk['model']
               
                    if 'choices' in chunk:
                        for choice in chunk['choices']:
                            choice_pb = pb2.ChatCompletionChoice()
                            choice_pb.index = choice.get('index', 0)
                            if 'message' in choice:
                                choice_pb.message.role = choice['message'].get('role', '')
                                choice_pb.message.content = choice['message'].get('content', '') if choice['message'].get('content', '') is not None else ''
                            if 'delta' in choice:
                                choice_pb.delta.role = choice['delta'].get('role', '')
                                choice_pb.delta.content = choice['delta'].get('content', '') if choice['delta'].get('content', '') is not None else ''

                            finish_reason = choice.get('finish_reason', '')
                            if finish_reason is None:
                                finish_reason = ''
                            choice_pb.finish_reason = str(finish_reason)
                            response.choices.append(choice_pb)
                    if 'usage' in chunk and chunk['usage'] is not None:
                        response.usage.prompt_tokens = chunk['usage'].get('prompt_tokens', 0)
                        response.usage.completion_tokens = chunk['usage'].get('completion_tokens', 0)
                        response.usage.total_tokens = chunk['usage'].get('total_tokens', 0)

                    # Check for error codes in the chunk and set error fields directly
                    if chunk.get('code', 0) >= 400 or (chunk.get('code', 0) < 200 and chunk.get('code', 0) != 0):
                        response.error.code = chunk.get('code', 0)
                        response.error.message = chunk.get('message', '')

                    print("\n**Yielding response for CreateChatCompletion", response)
                    yield response
                    
        except RAGFlowConnectionError as e:
            print(f"RAGFlow connection error in CreateChatCompletion: {e}")
            yield pb2.ChatCompletionResponse(error=self._handle_connection_error(e))
        except Exception as e:
            print(f"Exception in CreateChatCompletion: {e}",repr(e))
            yield pb2.ChatCompletionResponse(error=pb2.ErrorResponse(code=500, message=str(e)))
    
    async def CreateAgentCompletion(self, request, context):
        """Create agent completion"""
        try:
            client = await self._get_client()
            
            data = {
                'model': request.model,
                'messages': [{'role': msg.role, 'content': msg.content} for msg in request.messages],
                'stream': request.stream
            }
            
            async for chunk in client.create_agent_completion(request.agent_id, data):
                if 'error' in chunk:
                    yield pb2.AgentCompletionResponse(error=self._handle_error(chunk))
                else:
                    response = pb2.AgentCompletionResponse()
                    if 'id' in chunk:
                        response.id = chunk['id']
                    if 'object' in chunk:
                        response.object = chunk['object']
                    if 'created' in chunk:
                        response.created = chunk['created']
                    if 'model' in chunk:
                        response.model = chunk['model']
                    # Add choices and usage similar to chat completion
                    yield response
                    
        except Exception as e:
            yield pb2.AgentCompletionResponse(error=pb2.ErrorResponse(code=500, message=str(e)))
    
    # Dataset Management
    async def CreateDataset(self, request, context):
        """Create dataset"""
        try:
            client = await self._get_client()
            
            data = {
                'name': request.name,
                'description': request.description,
                'embedding_model': request.embedding_model,
                'permission': request.permission,
                'chunk_method': request.chunk_method
            }
            
            if request.parser_config:
                data['parser_config'] = json.loads(request.parser_config)
            
            result = await client.create_dataset(data)
            
            response = pb2.CreateDatasetResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')

            if result.get('code', 0) >= 400 or (result.get('code', 0) < 200 and result.get('code', 0) != 0):
                print(f"*{"----------"*2}*\nError in CreateDataset:", result)
                return response

            print(f"*{"----------"*2}*\nCreateDataset response: {result}")
            if 'data' in result and result['data'] :
                dataset = result['data']
                response.data.id = str(dataset.get('id', ''))
                response.data.name = str(dataset.get('name', ''))
                response.data.avatar = str(dataset.get('avatar', ''))
                response.data.description = str(dataset.get('description', ''))
                response.data.embedding_model = str(dataset.get('embedding_model', ''))
                response.data.permission = str(dataset.get('permission', ''))
                response.data.chunk_method = str(dataset.get('chunk_method', ''))
                response.data.chunk_count = int(dataset.get('chunk_count', 0))
                response.data.document_count = int(dataset.get('document_count', 0))
                response.data.parser_config = json.dumps(dataset.get('parser_config', {}))
                response.data.create_date = str(dataset.get('create_date', ''))
                response.data.create_time = int(dataset.get('create_time', 0))
                response.data.update_date = str(dataset.get('update_date', ''))
                response.data.update_time = int(dataset.get('update_time', 0))
                response.data.created_by = str(dataset.get('created_by', ''))
                response.data.tenant_id = str(dataset.get('tenant_id', ''))
                response.data.language = str(dataset.get('language', ''))
                response.data.pagerank = int(dataset.get('pagerank', 0))
                response.data.similarity_threshold = float(dataset.get('similarity_threshold', 0.0))
                response.data.vector_similarity_weight = float(dataset.get('vector_similarity_weight', 0.0))
                response.data.status = str(dataset.get('status', ''))
                response.data.token_num = int(dataset.get('token_num', 0))
            
            return response
            
        except Exception as e:
            print(f"Exception in CreateDataset: {e}",repr(e))
            return pb2.CreateDatasetResponse(
                code=500,
                message=str(e)
            )
    
    async def DeleteDatasets(self, request, context):
        """Delete datasets"""
        try:
            client = await self._get_client()
            data = {}
            if request.ids:
                # request.ids is already a repeated field (list), no need to convert
                data['ids'] = list(request.ids)
            else:
                data['ids'] = None  # Delete all

            print(f"DeleteDatasets request: ",data)

            result = await client.delete_datasets(data)
            print(f"DeleteDatasets response: ",result)

            return pb2.DeleteDatasetsResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.DeleteDatasetsResponse(code=500, message=str(e))
    
    async def UpdateDataset(self, request, context):
        """Update dataset"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.name:
                data['name'] = request.name
            if request.avatar:
                data['avatar'] = request.avatar
            if request.description:
                data['description'] = request.description
            if request.embedding_model:
                data['embedding_model'] = request.embedding_model
            if request.permission:
                data['permission'] = request.permission
            if request.chunk_method:
                data['chunk_method'] = request.chunk_method
            if request.pagerank:
                data['pagerank'] = request.pagerank
            if request.parser_config:
                data['parser_config'] = json.loads(request.parser_config)
            
            result = await client.update_dataset(request.dataset_id, data)
            
            return pb2.UpdateDatasetResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.UpdateDatasetResponse(code=500, message=str(e))
    
    async def ListDatasets(self, request, context):
        """List datasets"""
        try:
            # print("$$ListDatasets request by sachin-$$", request)
            client = await self._get_client()
            
            params = {}
            if request.page:
                params['page'] = request.page
            if request.page_size:
                params['page_size'] = request.page_size
            if request.orderby:
                params['orderby'] = request.orderby
            if request.desc:
                params['desc'] = request.desc
            if request.name:
                params['name'] = request.name
            if request.id:
                params['id'] = request.id
            
            result = await client.list_datasets(params)

            
            # Check for HTTP error status code
            if result.get('code', 0) >= 400 or (result.get('code', 0) < 200 and result.get('code', 0) != 0):
                print(f"Error in list_datasets: {result.get('message', '')} (code: {result.get('code', 0)})",result)
                response = pb2.ListDatasetsResponse()
                response.code = result.get('code', 0)
                response.message = result.get('message', '')
                response.data.clear()
                print("Returning empty response due to error in list_datasets")
                return response

            print("$$result by sachin-$$", result)
            response = pb2.ListDatasetsResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            print("**response by sachin-**", response,response.code)
            if 'data' in result:
                data = result['data']
                # Handle both formats: data as list or data as dict with datasets key
                datasets_list = data if isinstance(data, list) else data.get('datasets', [])
                for dataset_data in datasets_list:
                    dataset = pb2.Dataset()
                    dataset.id = str(dataset_data.get('id', ''))
                    dataset.name = str(dataset_data.get('name', ''))
                    dataset.avatar = str(dataset_data.get('avatar', ''))
                    dataset.description = str(dataset_data.get('description', ''))
                    dataset.embedding_model = str(dataset_data.get('embedding_model', ''))
                    dataset.permission = str(dataset_data.get('permission', ''))
                    dataset.chunk_method = str(dataset_data.get('chunk_method', ''))
                    dataset.chunk_count = int(dataset_data.get('chunk_count', 0))
                    dataset.document_count = int(dataset_data.get('document_count', 0))
                    dataset.parser_config = json.dumps(dataset_data.get('parser_config', {}))
                    dataset.create_date = str(dataset_data.get('create_date', ''))
                    dataset.create_time = int(dataset_data.get('create_time', 0))
                    dataset.update_date = str(dataset_data.get('update_date', ''))
                    dataset.update_time = int(dataset_data.get('update_time', 0))
                    dataset.created_by = str(dataset_data.get('created_by', ''))
                    dataset.tenant_id = str(dataset_data.get('tenant_id', ''))
                    dataset.language = str(dataset_data.get('language', ''))
                    dataset.pagerank = int(dataset_data.get('pagerank', 0))
                    dataset.similarity_threshold = float(dataset_data.get('similarity_threshold', 0.0))
                    dataset.vector_similarity_weight = float(dataset_data.get('vector_similarity_weight', 0.0))
                    dataset.status = str(dataset_data.get('status', ''))
                    dataset.token_num = int(dataset_data.get('token_num', 0))
                    response.data.append(dataset)
            
            return response
            
        except Exception as e:
            print("Exception occured in ListDatasets", e)
            return pb2.ListDatasetsResponse(code=500, message=str(e))
    
    async def GetDatasetKnowledgeGraph(self, request, context):
        """Get dataset knowledge graph"""
        try:
            client = await self._get_client()
            result = await client.get_dataset_knowledge_graph(request.dataset_id)
            
            response = pb2.GetDatasetKnowledgeGraphResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result:
                data = result['data']
                response.data.graph = json.dumps(data.get('graph', {}))
                response.data.mind_map = json.dumps(data.get('mind_map', {}))
            
            return response
            
        except Exception as e:
            return pb2.GetDatasetKnowledgeGraphResponse(code=500, message=str(e))
    
    async def DeleteDatasetKnowledgeGraph(self, request, context):
        """Delete dataset knowledge graph"""
        try:
            client = await self._get_client()
            result = await client.delete_dataset_knowledge_graph(request.dataset_id)
            
            return pb2.DeleteDatasetKnowledgeGraphResponse(
                code=result.get('code', 0),
                message=result.get('message', ''),
                data=result.get('data', False)
            )
            
        except Exception as e:
            return pb2.DeleteDatasetKnowledgeGraphResponse(code=500, message=str(e), data=False)
    
    # Document Management (Add similar implementations for other methods)
    async def UploadDocuments(self, request, context):
        """Upload documents"""
        try:
            client = await self._get_client()
            
            files = []
            for file_pb in request.files:
                files.append({
                    'filename': file_pb.filename,
                    'content': file_pb.content
                })
            
            result = await client.upload_documents(request.dataset_id, files)
            
            response = pb2.UploadDocumentsResponse()
            # Ensure result is a dict before calling .get()
            if isinstance(result, dict):
                response.code = result.get('code', 0)
                response.message = result.get('message', '')
            else:
                response.code = 500
                response.message = f"Unexpected result type: {type(result)}"
                return response
            
            if 'data' in result and isinstance(result, dict):
                # Handle different possible data structures
                data = result['data']
                if isinstance(data, dict) and 'documents' in data:
                    # If data is a dict with 'documents' key, use that
                    documents = data['documents']
                elif isinstance(data, list):
                    # If data is directly a list of documents
                    documents = data
                else:
                    # If data is a single document, wrap in list
                    documents = [data] if data else []
                
                for doc_data in documents:
                    doc = pb2.Document()
                    doc.id = doc_data.get('id', '')
                    doc.name = doc_data.get('name', '')
                    doc.location = doc_data.get('location', '')
                    doc.type = doc_data.get('type', '')
                    doc.size = doc_data.get('size', 0)
                    doc.thumbnail = doc_data.get('thumbnail', '')
                    doc.chunk_method = doc_data.get('chunk_method', '')
                    doc.parser_config = json.dumps(doc_data.get('parser_config', {}))
                    doc.run = doc_data.get('run', '')
                    doc.status = doc_data.get('status', '')
                    doc.progress_msg = doc_data.get('progress_msg', '')
                    doc.progress = doc_data.get('progress', 0.0)
                    doc.process_duration = doc_data.get('process_duration', 0.0)
                    doc.process_begin_at = doc_data.get('process_begin_at', '')
                    doc.chunk_count = doc_data.get('chunk_count', 0)
                    doc.token_count = doc_data.get('token_count', 0)
                    doc.dataset_id = doc_data.get('dataset_id', '')
                    doc.created_by = doc_data.get('created_by', '')
                    doc.create_date = doc_data.get('create_date', '')
                    doc.create_time = doc_data.get('create_time', 0)
                    doc.update_date = doc_data.get('update_date', '')
                    doc.update_time = doc_data.get('update_time', 0)
                    doc.source_type = doc_data.get('source_type', '')
                    response.data.append(doc)
            
            return response
            
        except Exception as e:
            return pb2.UploadDocumentsResponse(code=500, message=str(e))
    
    async def UpdateDocument(self, request, context):
        """Update document"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.name:
                data['name'] = request.name
            if request.meta_fields:
                data['meta_fields'] = json.loads(request.meta_fields)
            if request.chunk_method:
                data['chunk_method'] = request.chunk_method
            if request.parser_config:
                data['parser_config'] = json.loads(request.parser_config)
            
            result = await client.update_document(request.dataset_id, request.document_id, data)
            
            return pb2.UpdateDocumentResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.UpdateDocumentResponse(code=500, message=str(e))
    
    async def DownloadDocument(self, request, context):
        """Download document"""
        try:
            client = await self._get_client()
            
            result = await client.download_document(request.dataset_id, request.document_id)
            
            response = pb2.DownloadDocumentResponse()
            response.code = result.get('code', result.get('status', 0))
            response.message = result.get('message', '')
            
            if 'content' in result:
                response.content = result['content']
                response.filename = result.get('headers', {}).get('content-disposition', '').split('filename=')[-1].strip('"') if result.get('headers') else 'document'
            
            return response
            
        except Exception as e:
            return pb2.DownloadDocumentResponse(code=500, message=str(e))
    
    async def ListDocuments(self, request, context):
        """List documents"""
        try:
            client = await self._get_client()
            
            params = {}
            if request.keywords:
                params['keywords'] = request.keywords
            if request.page:
                params['page'] = request.page
            if request.page_size:
                params['page_size'] = request.page_size
            if request.orderby:
                params['orderby'] = request.orderby
            if request.desc:
                params['desc'] = request.desc
            if request.id:
                params['id'] = request.id
            if request.name:
                params['name'] = request.name
            if request.create_time_from:
                params['create_time_from'] = request.create_time_from
            if request.create_time_to:
                params['create_time_to'] = request.create_time_to
            
            result = await client.list_documents(request.dataset_id, params)
            print("\nListDocuments result:\n", result)

            response = pb2.ListDocumentsResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            # Check for HTTP error status code
            if result.get('code', 0) >= 400 or (result.get('code', 0) < 200 and result.get('code', 0) != 0):
                print(f"Error in ListDocuments: {result}")
                return response
            
            if 'data' in result:
                response.data.total = result['data'].get('total', 0)
                for doc_data in result['data'].get('docs', []):
                    doc = pb2.Document()
                    doc.id = str(doc_data.get('id', ''))
                    doc.name = str(doc_data.get('name', ''))
                    doc.location = str(doc_data.get('location', ''))
                    doc.type = str(doc_data.get('type', ''))
                    doc.size = int(doc_data.get('size', 0))
                    doc.thumbnail = str(doc_data.get('thumbnail', ''))
                    doc.chunk_method = str(doc_data.get('chunk_method', ''))
                    doc.parser_config = json.dumps(doc_data.get('parser_config', {}))
                    doc.run = str(doc_data.get('run', ''))
                    doc.status = str(doc_data.get('status', ''))
                    doc.progress_msg = str(doc_data.get('progress_msg', ''))
                    doc.progress = float(doc_data.get('progress', 0.0))
                    doc.process_duration = float(doc_data.get('process_duration', 0.0))
                    doc.process_begin_at = str(doc_data.get('process_begin_at', ''))
                    doc.chunk_count = int(doc_data.get('chunk_count', 0))
                    doc.token_count = int(doc_data.get('token_count', 0))
                    doc.dataset_id = str(doc_data.get('dataset_id', ''))
                    doc.created_by = str(doc_data.get('created_by', ''))
                    doc.create_date = str(doc_data.get('create_date', ''))
                    doc.create_time = int(doc_data.get('create_time', 0))
                    doc.update_date = str(doc_data.get('update_date', ''))
                    doc.update_time = int(doc_data.get('update_time', 0))
                    doc.source_type = str(doc_data.get('source_type', ''))
                    response.data.docs.append(doc)
            
            return response
            
        except Exception as e:
            print(f"Exception in ListDocuments: {e}")
            return pb2.ListDocumentsResponse(code=500, message=str(e))
    
    async def DeleteDocuments(self, request, context):
        """Delete documents"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.ids:
                data['ids'] = list(request.ids)
            else:
                data['ids'] = None  # Delete all documents
            
            result = await client.delete_documents(request.dataset_id, data)
            
            return pb2.DeleteDocumentsResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.DeleteDocumentsResponse(code=500, message=str(e))
    
    async def ParseDocuments(self, request, context):
        """Parse documents"""
        try:
            client = await self._get_client()
            
            data = {
                'document_ids': list(request.document_ids)
            }
            
            result = await client.parse_documents(request.dataset_id, data)
            
            return pb2.ParseDocumentsResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.ParseDocumentsResponse(code=500, message=str(e))
    
    async def StopParsingDocuments(self, request, context):
        """Stop parsing documents"""
        try:
            client = await self._get_client()
            
            data = {
                'document_ids': list(request.document_ids)
            }
            
            result = await client.stop_parsing_documents(request.dataset_id, data)
            
            return pb2.StopParsingDocumentsResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.StopParsingDocumentsResponse(code=500, message=str(e))
    
    # Chunk Management implementations
    async def AddChunk(self, request, context):
        """Add chunk"""
        try:
            client = await self._get_client()
            
            data = {
                'content': request.content
            }
            
            if request.important_keywords:
                data['important_keywords'] = list(request.important_keywords)
            
            if request.questions:
                data['questions'] = list(request.questions)
            
            result = await client.add_chunk(request.dataset_id, request.document_id, data)
            print("\n* Add Chunk operation done, Ragflow server response",result)
            response = pb2.AddChunkResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result and result['data']:
                chunk_data = result['data']
                if 'chunk' in chunk_data:
                    chunk = chunk_data['chunk']
                    response.data.chunk.id = chunk.get('id', '')
                    response.data.chunk.content = chunk.get('content', '')
                    response.data.chunk.document_id = chunk.get('document_id', '')
                    response.data.chunk.important_keywords = ",".join(chunk.get('important_keywords', []))
                    response.data.chunk.available = chunk.get('available', True)
                    response.data.chunk.create_time = chunk.get('create_time', '')
                    response.data.chunk.create_timestamp = chunk.get('create_timestamp', 0.0)
                    response.data.chunk.dataset_id = chunk.get('dataset_id', '')
                    if chunk.get('questions'):
                        response.data.chunk.questions.extend(chunk['questions'])
            print("\n\n* Returning Add Chunk operation response: ",response)
            return response
            
        except Exception as e:
            print("\n\n* Got Exception Add Chunk operation error response: ", e)
            return pb2.AddChunkResponse(code=500, message=str(e))
    
    async def ListChunks(self, request, context):
        """List chunks"""
        try:
            client = await self._get_client()
            
            params = {}
            if request.keywords:
                params['keywords'] = request.keywords
            if request.page:
                params['page'] = request.page
            if request.page_size:
                params['page_size'] = request.page_size
            if request.id:
                params['id'] = request.id
            
            result = await client.list_chunks(request.dataset_id, request.document_id, params)
            
            response = pb2.ListChunksResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result and result['data']:
                data = result['data']
                
                # Set total count
                response.data.total = data.get('total', 0)
                
                # Set document info
                if 'doc' in data and data['doc']:
                    doc = data['doc']
                    response.data.doc.id = doc.get('id', '')
                    response.data.doc.name = doc.get('name', '')
                    response.data.doc.location = doc.get('location', '')
                    response.data.doc.type = doc.get('type', '')
                    response.data.doc.size = doc.get('size', 0)
                    response.data.doc.thumbnail = doc.get('thumbnail', '')
                    response.data.doc.chunk_method = doc.get('chunk_method', '')
                    response.data.doc.parser_config = str(doc.get('parser_config', {}))
                    response.data.doc.run = doc.get('run', '')
                    response.data.doc.status = doc.get('status', '')
                    response.data.doc.progress_msg = doc.get('progress_msg', '')
                    response.data.doc.progress = doc.get('progress', 0.0)
                    response.data.doc.process_duration = float(doc.get('process_duration', 0.0))
                    response.data.doc.process_begin_at = str(doc.get('process_begin_at', ''))
                    response.data.doc.chunk_count = doc.get('chunk_count', 0)
                    response.data.doc.token_count = doc.get('token_count', 0)
                    response.data.doc.dataset_id = doc.get('dataset_id', '')
                    response.data.doc.created_by = doc.get('created_by', '')
                    response.data.doc.create_date = doc.get('create_date', '')
                    response.data.doc.create_time = doc.get('create_time', 0)
                    response.data.doc.update_date = doc.get('update_date', '')
                    response.data.doc.update_time = doc.get('update_time', 0)
                    response.data.doc.source_type = doc.get('source_type', '')
                # Set chunks
                if 'chunks' in data and data['chunks']:
                    for chunk_data in data['chunks']:
                        chunk = response.data.chunks.add()
                        chunk.id = chunk_data.get('id', '')
                        chunk.content = chunk_data.get('content', '')
                        chunk.docnm_kwd = chunk_data.get('docnm_kwd', '')
                        chunk.document_id = chunk_data.get('document_id', '')
                        chunk.image_id = chunk_data.get('image_id', '')
                        chunk.important_keywords = str(chunk_data.get('important_keywords', ''))
                        chunk.available = chunk_data.get('available', True)
                        if chunk_data.get('positions'):
                            chunk.positions.extend(chunk_data['positions'])
                        if chunk_data.get('questions'):
                            chunk.questions.extend(chunk_data['questions'])
            
            return response
            
        except Exception as e:
            print(f"Error in ListChunks: {e}", repr(e))
            return pb2.ListChunksResponse(code=500, message=str(e))
    
    async def DeleteChunks(self, request, context):
        """Delete chunks"""
        try:
            client = await self._get_client()
            
            data = {}
           
            # Convert protobuf repeated field to Python list
            chunk_ids_list = list(request.chunk_ids)
            data["chunk_ids"] = chunk_ids_list
            print("DeleteChunks request data:", data)
            result = await client.delete_chunks(request.dataset_id, request.document_id, data)
            print("DeleteChunks API result:", result)
            return pb2.DeleteChunksResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.DeleteChunksResponse(code=500, message=str(e))
    
    async def UpdateChunk(self, request, context):
        """Update chunk"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.content:
                data['content'] = request.content
            if request.important_keywords:
                data['important_keywords'] = list(request.important_keywords)
            
            data['available'] = request.available if hasattr(request, 'available') else True
            print("UpdateChunk request data:", data)

            result = await client.update_chunk(request.dataset_id, request.document_id, request.chunk_id, data)
            
            return pb2.UpdateChunkResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.UpdateChunkResponse(code=500, message=str(e))
    
    
    async def RetrieveChunks(self, request, context):
        """Retrieve chunks"""
        try:
            client = await self._get_client()
            
            data = {
                'question': request.question,
                'page': request.page if request.page else 1,
                'page_size': request.page_size if request.page_size else 30,
                'similarity_threshold': request.similarity_threshold if request.similarity_threshold else 0.2,
                'vector_similarity_weight': request.vector_similarity_weight if request.vector_similarity_weight else 0.3,
                'top_k': request.top_k if request.top_k else 1024,
                'keyword': request.keyword,
                'highlight': request.highlight,
                'cross_languages': list(request.cross_languages) if request.cross_languages else []
            }
            if request.dataset_ids:
                data['dataset_ids'] = list(request.dataset_ids)

            if request.document_ids:
                data['document_ids'] = list(request.document_ids)

            if request.rerank_id:
                data['rerank_id'] = request.rerank_id
            
            print("RetrieveChunks request data:", data)
            result = await client.retrieve_chunks(data)
            print("Retrieved chunks API result:", result)

            response = pb2.RetrieveChunksResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result:
                retrieval_data = result['data']
                response.data.total = retrieval_data.get('total', 0)
                
                # Add chunks
                for chunk_data in retrieval_data.get('chunks', []):
                    chunk = pb2.RetrievedChunk()
                    
                    chunk.id = str(chunk_data.get('id', ''))
                    chunk.content = str(chunk_data.get('content', ''))
                    chunk.content_ltks = str(chunk_data.get('content_ltks', ''))
                    chunk.document_id = str(chunk_data.get('document_id', ''))
                    chunk.document_keyword = str(chunk_data.get('document_keyword', ''))
                    chunk.highlight = str(chunk_data.get('highlight', ''))
                    chunk.image_id = str(chunk_data.get('image_id', ''))
                    # Ensure important_keywords is a list of strings
                    keywords = chunk_data.get('important_keywords', [])
                    if keywords:
                        chunk.important_keywords.extend([str(kw) for kw in keywords])
                    chunk.kb_id = str(chunk_data.get('kb_id', ''))
                    # Ensure positions is a list of strings
                    positions = chunk_data.get('positions', [])
                    if positions:
                        chunk.positions.extend([str(pos) for pos in positions])
                    chunk.similarity = float(chunk_data.get('similarity', 0.0))
                    chunk.term_similarity = float(chunk_data.get('term_similarity', 0.0))
                    chunk.vector_similarity = float(chunk_data.get('vector_similarity', 0.0))
                    response.data.chunks.append(chunk)

                print(f"\nRetrieved {len(response.data.chunks)} chunks")
                # Add document aggregations
                for doc_agg_data in retrieval_data.get('doc_aggs', []):
                    doc_agg = pb2.DocumentAgg()
                    doc_agg.doc_id = str(doc_agg_data.get('doc_id', ''))
                    doc_agg.doc_name = str(doc_agg_data.get('doc_name', ''))
                    doc_agg.count = int(doc_agg_data.get('count', 0))
                    response.data.doc_aggs.append(doc_agg)
            
            return response
            
        except Exception as e:
            print(f"Exception in RetrieveChunks: {e}", repr(e))
            return pb2.RetrieveChunksResponse(code=500, message=str(e))
    
    async def ConverseWithChatAssistant(self, request, context):
        """Converse with chat assistant"""
        try:
            client = await self._get_client()
            
            data = {
                'question': request.question,
                'stream': request.stream
            }
            
            if request.session_id:
                data['session_id'] = request.session_id
            if request.user_id:
                data['user_id'] = request.user_id
            
            async for chunk in client.converse_with_chat_assistant(request.chat_id, data):
                print(f"\n{'*'*5} Received chunk from ConverseWithChatAssistant: {chunk}")
                if 'error' in chunk:
                    yield pb2.ConverseWithChatAssistantResponse(
                        code=500,
                        message=str(chunk['error'])
                    )
                else:
                    response = pb2.ConverseWithChatAssistantResponse()
                    response.code = chunk.get('code', 0)
                    response.message = chunk.get('message', '')
                    
                    if 'data' in chunk and isinstance(chunk['data'],bool):
                        response.message += str(chunk['data'])


                    elif 'data' in chunk:
                        conv_data = chunk['data']
                        response.data.answer = conv_data.get('answer', '')
                        response.data.reference = json.dumps(conv_data.get('reference', {}))
                        if 'audio_binary' in conv_data and conv_data['audio_binary']:
                            response.data.audio_binary = conv_data['audio_binary']
                        response.data.id = conv_data.get('id', '')
                        response.data.session_id = conv_data.get('session_id', '')
                        response.data.prompt = conv_data.get('prompt', '')
                    
                    yield response
                    
        except Exception as e:
            yield pb2.ConverseWithChatAssistantResponse(code=500, message=str(e))
    
    async def ConverseWithAgent(self, request, context):
        """Converse with agent"""
        try:
            client = await self._get_client()
            
            data = {
                'question': request.question,
                'stream': request.stream
            }
            if request.sync_dsl:
                data['sync_dsl'] = request.sync_dsl

            if request.session_id:
                data['session_id'] = request.session_id
            if request.user_id:
                data['user_id'] = request.user_id
            
            # Add parameters
            for key, value in request.parameters.items():
                data[key] = value

            print(f"\n{"*"*5} Sending request to agent in ConverseWithAgent: {data}")
            async for chunk in client.converse_with_agent(request.agent_id, data):
                print(f"\nAgent response chunk in ConverseWithAgent: {chunk}")
                if 'error' in chunk:
                    yield pb2.ConverseWithAgentResponse(
                        code=500,
                        message=str(chunk['error'])
                    )
                else:
                    response = pb2.ConverseWithAgentResponse()
                    response.code = chunk.get('code', 0)
                    response.message = chunk.get('message', '')
                    
                    if 'data' in chunk:
                        conv_data = chunk['data']
                        response.data.answer = conv_data.get('answer', '')
                        response.data.reference = json.dumps(conv_data.get('reference', {}))
                        if 'audio_binary' in conv_data and conv_data['audio_binary']:
                            response.data.audio_binary = conv_data['audio_binary']
                        response.data.id = conv_data.get('id', '')
                        response.data.session_id = conv_data.get('session_id', '')
                        response.data.prompt = conv_data.get('prompt', '')
                        
                        # Add parameters
                        for param_data in conv_data.get('param', []):
                            param = pb2.ConversationParam()
                            param.key = param_data.get('key', '')
                            param.name = param_data.get('name', '')
                            param.optional = param_data.get('optional', False)
                            param.type = param_data.get('type', '')
                            param.value = param_data.get('value', '')
                            response.data.param.append(param)
                    
                    yield response
                    
        except Exception as e:
            yield pb2.ConverseWithAgentResponse(code=500, message=str(e))

    async def ListAgents(self, request, context):
        """List agents"""
        try:
            client = await self._get_client()
            
            params = {}
            if request.page:
                params['page'] = request.page
            if request.page_size:
                params['page_size'] = request.page_size
            if request.orderby:
                params['orderby'] = request.orderby
            if request.desc:
                params['desc'] = request.desc
            if request.name:
                params['name'] = request.name
            if request.id:
                params['id'] = request.id
            
            result = await client.list_agents(params)
            
            # Ensure result is a dict before processing
            if not isinstance(result, dict):
                print(f"Error: Expected dict but got {type(result)}: {result}")
                return pb2.ListAgentsResponse(
                    code=500, 
                    message=f"Unexpected result type: {type(result)}"
                )
            
            # Check for HTTP error status code
            if result.get('code', 0) >= 400 or (result.get('code', 0) < 200 and result.get('code', 0) != 0):
                print(f"Error in list_agents: {result.get('message', '')} (code: {result.get('code', 0)})", result)
                response = pb2.ListAgentsResponse()
                response.code = result.get('code', 0)
                response.message = result.get('message', '')
                response.data.clear()
                print("Returning empty response due to error in list_agents")
                return response

            response = pb2.ListAgentsResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            print("\n\nProcessing agents:")
            if 'data' in result and isinstance(result.get('data'), (list, dict)):
                data = result['data']
                # Handle both dict with 'agents' key and direct list
                agents_list = data.get('agents', []) if isinstance(data, dict) else data
                for agent_data in agents_list:
                    if isinstance(agent_data, dict):
                        agent = pb2.Agent()
                        agent.id = str(agent_data.get('id', '')) 
                        agent.title = str(agent_data.get('title', ''))
                        agent.avatar = str(agent_data.get('avatar', ''))
                        agent.canvas_type = str(agent_data.get('canvas_type', ''))
                        agent.description = str(agent_data.get('description', ''))
                        agent.dsl = str(agent_data.get('dsl', ''))
                        agent.create_date = str(agent_data.get('create_date', ''))
                        agent.create_time = int(agent_data.get('create_time', 0))
                        agent.update_date = str(agent_data.get('update_date', ''))
                        agent.update_time = int(agent_data.get('update_time', 0))
                        agent.user_id = str(agent_data.get('user_id', ''))
                        response.data.append(agent)
            print("Finished processing agents",response)
            return response
            
        except Exception as e:
            print(f"Exception in ListAgents: {e}",repr(e))
            return pb2.ListAgentsResponse(code=500, message=str(e))

    async def CreateAgent(self, request, context):
        """Create agent"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.title:
                data['title'] = request.title
            if request.description:
                data['description'] = request.description
            if request.dsl:
                data['dsl'] = request.dsl
            
            result = await client.create_agent(data)
            
            print(f"CreateAgent result: {result}")
            # Ensure result is a dict before processing
            if isinstance(result, dict):
                return pb2.CreateAgentResponse(
                    code=result.get('code', 0),
                    message=result.get('message', ''),
                    data=result.get('data', False)
                )
            else:
                return pb2.CreateAgentResponse(
                    code=500,
                    message=f"Unexpected result type: {type(result)}",
                    data=False
                )
            
        except Exception as e:
            print(f"Exception in CreateAgent: {e}")
            return pb2.CreateAgentResponse(code=500, message=str(e), data=False)

    async def UpdateAgent(self, request, context):
        """Update agent"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.title:
                data['title'] = request.title
            if request.description:
                data['description'] = request.description
            if request.dsl:
                data['dsl'] = request.dsl
            
            result = await client.update_agent(request.agent_id, data)
            
            print(f"UpdateAgent result: {result}")
            return pb2.UpdateAgentResponse(
                code=result.get('code', 0),
                message=result.get('message', ''),
                data=result.get('data', False)
            )
            
        except Exception as e:
            print(f"Exception in UpdateAgent: {e}")
            return pb2.UpdateAgentResponse(code=500, message=str(e), data=False)

    async def DeleteAgent(self, request, context):
        """Delete agent"""
        try:
            client = await self._get_client()
            
            result = await client.delete_agent(request.agent_id)
            
            print(f"DeleteAgent result: {result}")
            return pb2.DeleteAgentResponse(
                code=result.get('code', 0),
                message=result.get('message', ''),
                data=result.get('data', False)
            )
            
        except Exception as e:
            print(f"Exception in DeleteAgent: {e}")
            return pb2.DeleteAgentResponse(code=500, message=str(e), data=False)

    # Chat Assistant Management
    async def CreateChatAssistant(self, request, context):
        """Create chat assistant"""
        try:

            print(f"\n{"***"*5} Creating chat assistant...")
            client = await self._get_client()
            
            data = {
                'name': request.name
            }
            
            if request.avatar:
                data['avatar'] = request.avatar
            if request.dataset_ids:
                data['dataset_ids'] = list(request.dataset_ids)
            
            # Handle LLM config
            if request.llm:
                llm_config = {}
                if request.llm.model_name:
                    llm_config['model_name'] = request.llm.model_name
                if request.llm.temperature:
                    llm_config['temperature'] = request.llm.temperature
                if request.llm.top_p:
                    llm_config['top_p'] = request.llm.top_p
                if request.llm.presence_penalty:
                    llm_config['presence_penalty'] = request.llm.presence_penalty
                if request.llm.frequency_penalty:
                    llm_config['frequency_penalty'] = request.llm.frequency_penalty
                data['llm'] = llm_config
            
            # Handle prompt config
            if request.prompt:
                prompt_config = {}
                if request.prompt.similarity_threshold:
                    prompt_config['similarity_threshold'] = request.prompt.similarity_threshold
                if request.prompt.keywords_similarity_weight:
                    prompt_config['keywords_similarity_weight'] = request.prompt.keywords_similarity_weight
                if request.prompt.top_n:
                    prompt_config['top_n'] = request.prompt.top_n
                if request.prompt.rerank_model:
                    prompt_config['rerank_model'] = request.prompt.rerank_model
                if request.prompt.top_k:
                    prompt_config['top_k'] = request.prompt.top_k
                if request.prompt.empty_response:
                    prompt_config['empty_response'] = request.prompt.empty_response
                if request.prompt.opener:
                    prompt_config['opener'] = request.prompt.opener
                if hasattr(request.prompt, 'show_quote'):
                    prompt_config['show_quote'] = request.prompt.show_quote
                if request.prompt.prompt:
                    prompt_config['prompt'] = request.prompt.prompt
                
                # Handle variables
                if request.prompt.variables:
                    variables = []
                    for var in request.prompt.variables:
                        variables.append({
                            'key': var.key,
                            'optional': var.optional
                        })
                    prompt_config['variables'] = variables
                
                data['prompt'] = prompt_config
            print("Creating chat assistant request data:", json.dumps(data, indent=2))
            result = await client.create_chat_assistant(data)
            
            response = pb2.CreateChatAssistantResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result and result['data']:
                assistant_data = result['data']
                response.data.id = assistant_data.get('id', '')
                response.data.name = assistant_data.get('name', '')
                response.data.avatar = assistant_data.get('avatar', '')
                response.data.description = assistant_data.get('description', '')
                response.data.do_refer = assistant_data.get('do_refer', '')
                response.data.language = assistant_data.get('language', '')
                response.data.prompt_type = assistant_data.get('prompt_type', '')
                response.data.status = assistant_data.get('status', '')
                response.data.tenant_id = assistant_data.get('tenant_id', '')
                response.data.top_k = assistant_data.get('top_k', 0)
                response.data.create_date = assistant_data.get('create_date', '')
                response.data.create_time = assistant_data.get('create_time', 0)
                response.data.update_date = assistant_data.get('update_date', '')
                response.data.update_time = assistant_data.get('update_time', 0)
                
                if assistant_data.get('dataset_ids'):
                    response.data.dataset_ids.extend(assistant_data['dataset_ids'])
                
                # Handle LLM config in response
                if assistant_data.get('llm'):
                    llm = assistant_data['llm']
                    response.data.llm.model_name = llm.get('model_name', '')
                    response.data.llm.temperature = llm.get('temperature', 0.0)
                    response.data.llm.top_p = llm.get('top_p', 0.0)
                    response.data.llm.presence_penalty = llm.get('presence_penalty', 0.0)
                    response.data.llm.frequency_penalty = llm.get('frequency_penalty', 0.0)
                
                # Handle prompt config in response
                if assistant_data.get('prompt'):
                    prompt = assistant_data['prompt']
                    response.data.prompt.similarity_threshold = prompt.get('similarity_threshold', 0.0)
                    response.data.prompt.keywords_similarity_weight = prompt.get('keywords_similarity_weight', 0.0)
                    response.data.prompt.top_n = prompt.get('top_n', 0)
                    response.data.prompt.rerank_model = prompt.get('rerank_model', '')
                    response.data.prompt.empty_response = prompt.get('empty_response', '')
                    response.data.prompt.opener = prompt.get('opener', '')
                    response.data.prompt.prompt = prompt.get('prompt', '')
                    
                    if prompt.get('variables'):
                        for var in prompt['variables']:
                            var_obj = response.data.prompt.variables.add()
                            var_obj.key = var.get('key', '')
                            var_obj.optional = var.get('optional', False)
            
            return response
            
        except Exception as e:
            return pb2.CreateChatAssistantResponse(code=500, message=str(e))
    
    async def UpdateChatAssistant(self, request, context):
        """Update chat assistant"""
        try:
            client = await self._get_client()
            
            data = {}
            data['name'] = request.name
            if request.avatar:
                data['avatar'] = request.avatar
            if request.dataset_ids:
                data['dataset_ids'] = list(request.dataset_ids)
            
            # Handle LLM config
            if request.llm:
                llm_config = {}
                if request.llm.model_name:
                    llm_config['model_name'] = request.llm.model_name
                if request.llm.temperature:
                    llm_config['temperature'] = request.llm.temperature
                if request.llm.top_p:
                    llm_config['top_p'] = request.llm.top_p
                if request.llm.presence_penalty:
                    llm_config['presence_penalty'] = request.llm.presence_penalty
                if request.llm.frequency_penalty:
                    llm_config['frequency_penalty'] = request.llm.frequency_penalty
                if llm_config:
                    data['llm'] = llm_config

            # Handle prompt config
            if request.prompt:
                prompt_config = {}
                if request.prompt.similarity_threshold:
                    prompt_config['similarity_threshold'] = request.prompt.similarity_threshold
                if request.prompt.keywords_similarity_weight:
                    prompt_config['keywords_similarity_weight'] = request.prompt.keywords_similarity_weight
                if request.prompt.top_n:
                    prompt_config['top_n'] = request.prompt.top_n
                if request.prompt.rerank_model:
                    prompt_config['rerank_model'] = request.prompt.rerank_model
                if request.prompt.top_k:
                    prompt_config['top_k'] = request.prompt.top_k
                if request.prompt.empty_response:
                    prompt_config['empty_response'] = request.prompt.empty_response
                if request.prompt.opener:
                    prompt_config['opener'] = request.prompt.opener
                if hasattr(request.prompt, 'show_quote'):
                    prompt_config['show_quote'] = request.prompt.show_quote
                if request.prompt.prompt:
                    prompt_config['prompt'] = request.prompt.prompt
                
                # Handle variables
                if request.prompt.variables:
                    variables = []
                    for var in request.prompt.variables:
                        variables.append({
                            'key': var.key,
                            'optional': var.optional
                        })
                    prompt_config['variables'] = variables
                
                data['prompt'] = prompt_config
            print(f"\n{'*'*5} Sending request to UpdateChatAssistant: {data}")
            result = await client.update_chat_assistant(request.chat_id, data)
            
            return pb2.UpdateChatAssistantResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.UpdateChatAssistantResponse(code=500, message=str(e))
    
    async def DeleteChatAssistants(self, request, context):
        """Delete chat assistants"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.ids:
                data['ids'] = list(request.ids)
            
            result = await client.delete_chat_assistants(data)
            
            return pb2.DeleteChatAssistantsResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.DeleteChatAssistantsResponse(code=500, message=str(e))
    
    async def ListChatAssistants(self, request, context):
        """List chat assistants"""
        try:
            client = await self._get_client()
            
            params = {}
            if request.page:
                params['page'] = request.page
            if request.page_size:
                params['page_size'] = request.page_size
            if request.orderby:
                params['orderby'] = request.orderby
            if request.desc:
                params['desc'] = request.desc
            if request.id:
                params['id'] = request.id
            if request.name:
                params['name'] = request.name
            
            result = await client.list_chat_assistants(params)
            
            response = pb2.ListChatAssistantsResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result and result['data']:
                for assistant_data in result['data']:
                    assistant = response.data.add()
                    assistant.id = assistant_data.get('id', '')
                    assistant.name = assistant_data.get('name', '')
                    assistant.avatar = assistant_data.get('avatar', '')
                    assistant.description = assistant_data.get('description', '')
                    assistant.do_refer = assistant_data.get('do_refer', '')
                    assistant.language = assistant_data.get('language', '')
                    assistant.prompt_type = assistant_data.get('prompt_type', '')
                    assistant.status = assistant_data.get('status', '')
                    assistant.tenant_id = assistant_data.get('tenant_id', '')
                    assistant.top_k = assistant_data.get('top_k', 0)
                    assistant.create_date = assistant_data.get('create_date', '')
                    assistant.create_time = assistant_data.get('create_time', 0)
                    assistant.update_date = assistant_data.get('update_date', '')
                    assistant.update_time = assistant_data.get('update_time', 0)
                    
                    if assistant_data.get('dataset_ids'):
                        assistant.dataset_ids.extend(assistant_data['dataset_ids'])
                    
                    # Handle LLM config
                    if assistant_data.get('llm'):
                        llm = assistant_data['llm']
                        assistant.llm.model_name = llm.get('model_name', '')
                        assistant.llm.temperature = llm.get('temperature', 0.0)
                        assistant.llm.top_p = llm.get('top_p', 0.0)
                        assistant.llm.presence_penalty = llm.get('presence_penalty', 0.0)
                        assistant.llm.frequency_penalty = llm.get('frequency_penalty', 0.0)
                    
                    # Handle prompt config
                    if assistant_data.get('prompt'):
                        prompt = assistant_data['prompt']
                        assistant.prompt.similarity_threshold = prompt.get('similarity_threshold', 0.0)
                        assistant.prompt.keywords_similarity_weight = prompt.get('keywords_similarity_weight', 0.0)
                        assistant.prompt.top_n = prompt.get('top_n', 0)
                        assistant.prompt.rerank_model = prompt.get('rerank_model', '')
                        assistant.prompt.empty_response = prompt.get('empty_response', '')
                        assistant.prompt.opener = prompt.get('opener', '')
                        assistant.prompt.show_quote = prompt.get('show_quote', False)
                        assistant.prompt.prompt = prompt.get('prompt', '')
                        
                        if prompt.get('variables'):
                            for var in prompt['variables']:
                                var_obj = assistant.prompt.variables.add()
                                var_obj.key = var.get('key', '')
                                var_obj.optional = var.get('optional', False)
            
            return response
            
        except Exception as e:
            return pb2.ListChatAssistantsResponse(code=500, message=str(e))

    # Session Management
    async def CreateSessionWithChatAssistant(self, request, context):
        """Create session with chat assistant"""
        try:
            client = await self._get_client()
            
            data = {
                'name': request.name
            }
            
            if request.user_id:
                data['user_id'] = request.user_id
            
            result = await client.create_session_with_chat_assistant(request.chat_id, data)
            
            response = pb2.CreateSessionWithChatAssistantResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result and result['data']:
                session_data = result['data']
                response.data.id = session_data.get('id', '')
                response.data.chat_id = session_data.get('chat_id', '')
                response.data.name = session_data.get('name', '')
                response.data.create_date = session_data.get('create_date', '')
                response.data.create_time = session_data.get('create_time', 0)
                response.data.update_date = session_data.get('update_date', '')
                response.data.update_time = session_data.get('update_time', 0)
                
                if session_data.get('messages'):
                    for msg in session_data['messages']:
                        message = response.data.messages.add()
                        message.role = msg.get('role', '')
                        message.content = msg.get('content', '')
            
            return response
            
        except Exception as e:
            return pb2.CreateSessionWithChatAssistantResponse(code=500, message=str(e))
    
    async def UpdateChatAssistantSession(self, request, context):
        """Update chat assistant session"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.name:
                data['name'] = request.name
            if request.user_id:
                data['user_id'] = request.user_id
            
            result = await client.update_chat_assistant_session(request.chat_id, request.session_id, data)
            
            return pb2.UpdateChatAssistantSessionResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.UpdateChatAssistantSessionResponse(code=500, message=str(e))
    
    async def ListChatAssistantSessions(self, request, context):
        """List chat assistant sessions"""
        try:
            client = await self._get_client()
            
            params = {}
            if request.page:
                params['page'] = request.page
            if request.page_size:
                params['page_size'] = request.page_size
            if request.orderby:
                params['orderby'] = request.orderby
            if request.desc:
                params['desc'] = request.desc
            if request.name:
                params['name'] = request.name
            if request.id:
                params['id'] = request.id
            if request.user_id:
                params['user_id'] = request.user_id
            
            result = await client.list_chat_assistant_sessions(request.chat_id, params)
            
            response = pb2.ListChatAssistantSessionsResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result and result['data']:
                for session_data in result['data']:
                    session = response.data.add()
                    session.id = session_data.get('id', '')
                    session.chat_id = session_data.get('chat', '')
                    session.name = session_data.get('name', '')
                    session.create_date = session_data.get('create_date', '')
                    session.create_time = session_data.get('create_time', 0)
                    session.update_date = session_data.get('update_date', '')
                    session.update_time = session_data.get('update_time', 0)
                    
                    if session_data.get('messages'):
                        for msg in session_data['messages']:
                            message = session.messages.add()
                            message.role = msg.get('role', '')
                            message.content = msg.get('content', '')
            
            return response
            
        except Exception as e:
            return pb2.ListChatAssistantSessionsResponse(code=500, message=str(e))
    
    async def DeleteChatAssistantSessions(self, request, context):
        """Delete chat assistant sessions"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.ids:
                data['ids'] = list(request.ids)
            
            result = await client.delete_chat_assistant_sessions(request.chat_id, data)
            
            return pb2.DeleteChatAssistantSessionsResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.DeleteChatAssistantSessionsResponse(code=500, message=str(e))
    
    async def CreateSessionWithAgent(self, request, context):
        """Create session with agent"""
        try:
            client = await self._get_client()
            
            data = {}
            
            # Add parameters from request
            for key, value in request.parameters.items():
                data[key] = value
            
            params = {}
            if request.user_id:
                params['user_id'] = request.user_id
            
            result = await client.create_session_with_agent(request.agent_id, data, params)
            
            response = pb2.CreateSessionWithAgentResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result and result['data']:
                session_data = result['data']
                response.data.id = session_data.get('id', '')
                response.data.agent_id = session_data.get('agent_id', '')
                response.data.dsl = str(session_data.get('dsl', ''))
                response.data.source = session_data.get('source', '')
                response.data.user_id = session_data.get('user_id', '')
                
                if session_data.get('message'):
                    for msg in session_data['message']:
                        message = response.data.message.add()
                        message.role = msg.get('role', '')
                        message.content = msg.get('content', '')
            
            return response
            
        except Exception as e:
            return pb2.CreateSessionWithAgentResponse(code=500, message=str(e))
    
    async def ListAgentSessions(self, request, context):
        """List agent sessions"""
        try:
            client = await self._get_client()
            
            params = {}
            if request.page:
                params['page'] = request.page
            if request.page_size:
                params['page_size'] = request.page_size
            if request.orderby:
                params['orderby'] = request.orderby
            if request.desc:
                params['desc'] = request.desc
            if request.id:
                params['id'] = request.id
            if request.user_id:
                params['user_id'] = request.user_id
            if hasattr(request, 'dsl'):
                params['dsl'] = request.dsl
            
            result = await client.list_agent_sessions(request.agent_id, params)
            
            response = pb2.ListAgentSessionsResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result and result['data']:
                for session_data in result['data']:
                    session = response.data.add()
                    session.id = session_data.get('id', '')
                    session.agent_id = session_data.get('agent_id', '')
                    session.dsl = str(session_data.get('dsl', ''))
                    session.source = session_data.get('source', '')
                    session.user_id = session_data.get('user_id', '')
                    
                    if session_data.get('message'):
                        for msg in session_data['message']:
                            message = session.message.add()
                            message.role = msg.get('role', '')
                            message.content = msg.get('content', '')
            
            return response
            
        except Exception as e:
            return pb2.ListAgentSessionsResponse(code=500, message=str(e))
    
    async def DeleteAgentSessions(self, request, context):
        """Delete agent sessions"""
        try:
            client = await self._get_client()
            
            data = {}
            if request.ids:
                data['ids'] = list(request.ids)
            
            result = await client.delete_agent_sessions(request.agent_id, data)
            
            return pb2.DeleteAgentSessionsResponse(
                code=result.get('code', 0),
                message=result.get('message', '')
            )
            
        except Exception as e:
            return pb2.DeleteAgentSessionsResponse(code=500, message=str(e))

    # Need special Bearer token for this method which expires in 24 hr
    async def GenerateRelatedQuestions(self, request, context):
        """Generate related questions"""
        try:
            client = await self._get_client()
            
            data = {
                'question': request.question,
                'industry': request.industry
            }
            
            result = await client.generate_related_questions(data)
            
            response = pb2.GenerateRelatedQuestionsResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result and result['data']:
                response.data.extend(result['data'])
            
            return response
            
        except Exception as e:
            return pb2.GenerateRelatedQuestionsResponse(code=500, message=str(e))


async def serve():
    """Start the gRPC server"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_RagServicesServicer_to_server(RagServicesServicer(), server)
    
    listen_addr = f'{GRPC_HOST}:{GRPC_PORT}'
    
    try:
        # Try to add the port
        port = server.add_insecure_port(listen_addr)
        if port == 0:
            print(f"Failed to bind to {listen_addr}. Port might be in use.")
            return
        
        print(f"Starting RAGFlow gRPC server on {listen_addr}...")
        await server.start()
        print(f"Server successfully started on {listen_addr}")
        
        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            print("Shutting down RAGFlow gRPC server...")
            await server.stop(5)
            
    except Exception as e:
        print(f"Failed to start server: {e}")
        print(f"Port {GRPC_PORT} might be in use. Try using a different port or stop existing processes.")
        
        # Suggest alternative ports
        alternative_ports = [50052, 50053, 50054, 9090, 9091]
        print(f"Alternative ports to try: {alternative_ports}")
        
        # Try to find an available port
        import socket
        for alt_port in alternative_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((GRPC_HOST, alt_port))
                    print(f"Port {alt_port} is available. You can update GRPC_PORT in config.py")
                    break
            except OSError:
                continue


if __name__ == '__main__':
    asyncio.run(serve())
