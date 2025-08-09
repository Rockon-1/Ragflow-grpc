import asyncio
import grpc
import json
from concurrent import futures
from typing import Dict, List, Optional, AsyncIterator
import ragflow_service_pb2 as pb2
import ragflow_service_pb2_grpc as pb2_grpc
from ragflow_client import RAGFlowClient
from config import GRPC_HOST, GRPC_PORT


class RagServicesServicer(pb2_grpc.RagServicesServicer):
    """Async gRPC servicer for RAGFlow API"""
    
    def __init__(self):
        self.client = None
    
    async def _get_client(self) -> RAGFlowClient:
        """Get or create RAGFlow client"""
        if self.client is None:
            self.client = RAGFlowClient()
            await self.client.__aenter__()
        return self.client
    
    def _handle_error(self, error_dict: Dict) -> pb2.ErrorResponse:
        """Convert error dict to protobuf ErrorResponse"""
        return pb2.ErrorResponse(
            code=error_dict.get('code', 500),
            message=error_dict.get('message', str(error_dict.get('error', 'Unknown error')))
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
                                choice_pb.message.content = choice['message'].get('content', '')
                            if 'delta' in choice:
                                choice_pb.delta.role = choice['delta'].get('role', '')
                                choice_pb.delta.content = choice['delta'].get('content', '')
                            choice_pb.finish_reason = choice.get('finish_reason', '')
                            response.choices.append(choice_pb)
                    if 'usage' in chunk:
                        response.usage.prompt_tokens = chunk['usage'].get('prompt_tokens', 0)
                        response.usage.completion_tokens = chunk['usage'].get('completion_tokens', 0)
                        response.usage.total_tokens = chunk['usage'].get('total_tokens', 0)
                    
                    yield response
                    
        except Exception as e:
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
                data['ids'] = list(request.ids)
            else:
                data['ids'] = None  # Delete all
            
            result = await client.delete_datasets(data)
            
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
                for dataset_data in result['data']:
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
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result:
                for doc_data in result['data']:
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
    
    # Add implementations for other methods following the same pattern...
    # For brevity, I'll show the structure for a few more key methods
    
    async def RetrieveChunks(self, request, context):
        """Retrieve chunks"""
        try:
            client = await self._get_client()
            
            data = {
                'question': request.question,
                'dataset_ids': list(request.dataset_ids) if request.dataset_ids else None,
                'document_ids': list(request.document_ids) if request.document_ids else None,
                'page': request.page if request.page else 1,
                'page_size': request.page_size if request.page_size else 30,
                'similarity_threshold': request.similarity_threshold if request.similarity_threshold else 0.2,
                'vector_similarity_weight': request.vector_similarity_weight if request.vector_similarity_weight else 0.3,
                'top_k': request.top_k if request.top_k else 1024,
                'keyword': request.keyword,
                'highlight': request.highlight,
                'cross_languages': list(request.cross_languages) if request.cross_languages else []
            }
            
            if request.rerank_id:
                data['rerank_id'] = request.rerank_id
            
            result = await client.retrieve_chunks(data)
            
            response = pb2.RetrieveChunksResponse()
            response.code = result.get('code', 0)
            response.message = result.get('message', '')
            
            if 'data' in result:
                retrieval_data = result['data']
                response.data.total = retrieval_data.get('total', 0)
                
                # Add chunks
                for chunk_data in retrieval_data.get('chunks', []):
                    chunk = pb2.RetrievedChunk()
                    chunk.id = chunk_data.get('id', '')
                    chunk.content = chunk_data.get('content', '')
                    chunk.content_ltks = chunk_data.get('content_ltks', '')
                    chunk.document_id = chunk_data.get('document_id', '')
                    chunk.document_keyword = chunk_data.get('document_keyword', '')
                    chunk.highlight = chunk_data.get('highlight', '')
                    chunk.image_id = chunk_data.get('image_id', '')
                    chunk.important_keywords.extend(chunk_data.get('important_keywords', []))
                    chunk.kb_id = chunk_data.get('kb_id', '')
                    chunk.positions.extend(chunk_data.get('positions', []))
                    chunk.similarity = chunk_data.get('similarity', 0.0)
                    chunk.term_similarity = chunk_data.get('term_similarity', 0.0)
                    chunk.vector_similarity = chunk_data.get('vector_similarity', 0.0)
                    response.data.chunks.append(chunk)
                
                # Add document aggregations
                for doc_agg_data in retrieval_data.get('doc_aggs', []):
                    doc_agg = pb2.DocumentAgg()
                    doc_agg.doc_id = doc_agg_data.get('doc_id', '')
                    doc_agg.doc_name = doc_agg_data.get('doc_name', '')
                    doc_agg.count = doc_agg_data.get('count', 0)
                    response.data.doc_aggs.append(doc_agg)
            
            return response
            
        except Exception as e:
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
                if 'error' in chunk:
                    yield pb2.ConverseWithChatAssistantResponse(
                        code=500,
                        message=str(chunk['error'])
                    )
                else:
                    response = pb2.ConverseWithChatAssistantResponse()
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
                    
                    yield response
                    
        except Exception as e:
            yield pb2.ConverseWithChatAssistantResponse(code=500, message=str(e))
    
    async def ConverseWithAgent(self, request, context):
        """Converse with agent"""
        try:
            client = await self._get_client()
            
            data = {
                'question': request.question,
                'stream': request.stream,
                'sync_dsl': request.sync_dsl
            }
            
            if request.session_id:
                data['session_id'] = request.session_id
            if request.user_id:
                data['user_id'] = request.user_id
            
            # Add parameters
            for key, value in request.parameters.items():
                data[key] = value
            
            async for chunk in client.converse_with_agent(request.agent_id, data):
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
