import aiohttp
import asyncio
import json
from typing import Dict, List, Optional, Union, AsyncIterator
from .config import RAGFLOW_BASE_URL, API_KEY



class RAGFlowConnectionError(Exception):
    """Raised when RAGFlow service is not accessible"""
    pass


class RAGFlowClient:
    """Async HTTP client for RAGFlow REST API"""
    
    def __init__(self, base_url: str = RAGFLOW_BASE_URL, api_key: str = API_KEY):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        self.sp_api_key = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            },
            timeout=aiohttp.ClientTimeout(total=300)
        )
        
        # Check if RAGFlow service is running
        await self._check_ragflow_connection()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _check_ragflow_connection(self) -> bool:
        """Check if RAGFlow service is accessible"""
        try:
            # Try to reach the base URL with a simple health check
            # Using a short timeout for the connection check
            async with self.session.get(
                f"{self.base_url}/",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                # If we get any response (even 404), the service is running
                print(f"RAGFlow connection check: Status {response.status}")
                return True
                
        except aiohttp.ClientConnectorError as e:
            error_msg = f"RAGFlow service is not running at {self.base_url}. Please ensure RAGFlow is started and accessible."
            print(f"Connection error: {error_msg}")
            raise RAGFlowConnectionError(error_msg) from e
            
        except asyncio.TimeoutError as e:
            error_msg = f"RAGFlow service at {self.base_url} is not responding (timeout). Please check if the service is running properly."
            print(f"Timeout error: {error_msg}")
            raise RAGFlowConnectionError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to connect to RAGFlow service at {self.base_url}: {str(e)}"
            print(f"Unexpected error: {error_msg}")
            raise RAGFlowConnectionError(error_msg) from e
    
    async def check_connection(self) -> bool:
        """Public method to check RAGFlow connection without context manager"""
        session = None
        try:
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
            async with session.get(f"{self.base_url}/") as response:
                print(f"RAGFlow connection check: Status {response.status}")
                return True
        except aiohttp.ClientConnectorError:
            return False
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False
        finally:
            if session:
                await session.close()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request to RAGFlow API"""
        url = f"{self.base_url}{endpoint}"

        if 'headers' not in kwargs:
            kwargs['headers'] = {}
        
        # Only add default headers if they're not already present
        if 'Authorization' not in kwargs['headers']:
            kwargs['headers']['Authorization'] = f'Bearer {self.api_key}'
        if 'Content-Type' not in kwargs['headers']:
            kwargs['headers']['Content-Type'] = 'application/json'

        try:
            async with self.session.request(method, url, **kwargs) as response:
                content_type = response.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    return await response.json()
                else:
                    # Handle binary/text responses
                    content = await response.read()
                    return {
                        'content': content,
                        'status': response.status,
                        'headers': dict(response.headers)
                    }
        except aiohttp.ClientConnectorError as e:
            error_msg = f"Cannot connect to RAGFlow service at {self.base_url}. Please ensure RAGFlow is running."
            print(f"Connection error in _make_request: {error_msg}")
            return {'error': error_msg, 'code': 503}
        except asyncio.TimeoutError as e:
            error_msg = f"RAGFlow service at {self.base_url} is not responding (timeout)."
            print(f"Timeout error in _make_request: {error_msg}")
            return {'error': error_msg, 'code': 504}
        except Exception as e:
            return {'error': str(e), 'code': 500}
    
    async def _stream_request(self, method: str, endpoint: str, **kwargs) -> AsyncIterator[Dict]:
        """Make streaming request to RAGFlow API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                print(f"Stream response status: {response.status}")
                print(f"Stream response headers: {dict(response.headers)}")
                
                # Check if response is successful
                if response.status >= 400:
                    error_text = await response.text()
                    print(f"HTTP Error {response.status}: {error_text}")
                    yield {'error': f'HTTP {response.status}: {error_text}', 'code': response.status}
                    return
                
                # Read the stream line by line
                buffer = ""
                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        # Decode the chunk and add to buffer
                        buffer += chunk.decode('utf-8', errors='ignore')
                        
                        # Process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            # Skip empty lines or lines that only contain whitespace
                            if not line:
                                continue

                            print(f"Received line: {line}")
                            
                            # Handle Server-Sent Events format
                            if line.startswith('data:'):
                                # Extract JSON data after 'data:' prefix
                                json_str = line[5:].strip()  # Remove 'data:' prefix
                                
                                # Skip if it's just the stream end marker
                                if json_str == '[DONE]':
                                    print("Stream ended with [DONE] marker")
                                    continue
                                
                                try:
                                    # Parse JSON data
                                    json_data = json.loads(json_str)
                                    print(f"Yielding JSON: {json_data}")
                                    yield json_data
                                except json.JSONDecodeError as e:
                                    print(f"JSON decode error: {e}, data: {json_str}")
                                    continue
                            elif line.startswith('event:') or line.startswith('id:'):
                                # Skip SSE metadata lines
                                print(f"Skipping SSE metadata: {line}")
                                continue
                            else:
                                # Try to parse as direct JSON (fallback for non-SSE streams)
                                try:
                                    json_data = json.loads(line)
                                    print(f"Yielding direct JSON: {json_data}")
                                    yield json_data
                                except json.JSONDecodeError as e:
                                    print(f"JSON decode error for direct line: {e}, data: {line}")
                                    continue
                                
        except aiohttp.ClientConnectorError as e:
            error_msg = f"Cannot connect to RAGFlow service at {self.base_url}. Please ensure RAGFlow is running."
            print(f"Connection error in _stream_request: {error_msg}")
            yield {'error': error_msg, 'code': 503}
        except asyncio.TimeoutError as e:
            error_msg = f"RAGFlow service at {self.base_url} is not responding (timeout)."
            print(f"Timeout error in _stream_request: {error_msg}")
            yield {'error': error_msg, 'code': 504}
        except Exception as e:
            print(f"Exception in _stream_request: {e}")
            yield {'error': str(e), 'code': 500}
    
    # OpenAI-Compatible API
    async def create_chat_completion(self, chat_id: str, data: Dict) -> AsyncIterator[Dict]:
        """Create chat completion"""
        endpoint = f"/api/v1/chats_openai/{chat_id}/chat/completions"
        
        try:
            print("Creating chat completion...in ragflow_client.py")
            if data.get('stream', True):
                print("Streaming enabled for chat completion for chat_id:", chat_id)
                
                # Set up headers for streaming
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                    'Cache-Control': 'no-cache'
                }
                
                async for chunk in self._stream_request('POST', endpoint, json=data, headers=headers):
                    print("yielding chunk for chat completion", chunk)
                    yield chunk
            else:
                headers = {'Authorization': f'Bearer {self.api_key}'}
                result = await self._make_request('POST', endpoint, json=data, headers=headers)
                yield result
        except Exception as e:
            print(f"Exception in CreateChatCompletion in ragflow_client.py: {e}", repr(e))
            yield {'error': str(e), 'code': 500}

    async def create_agent_completion(self, agent_id: str, data: Dict) -> AsyncIterator[Dict]:
        """Create agent completion"""
        endpoint = f"/api/v1/agents_openai/{agent_id}/chat/completions"
        headers = {'Authorization': f'Bearer {self.api_key}'}

        if data.get('stream', True):
            async for chunk in self._stream_request('POST', endpoint, json=data, headers=headers):
                yield chunk
        else:
            result = await self._make_request('POST', endpoint, json=data, headers=headers)
            yield result
    
    # Dataset Management
    async def create_dataset(self, data: Dict) -> Dict:
        """Create dataset"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('POST', '/api/v1/datasets', json=data, headers=headers)

    async def delete_datasets(self, data: Dict) -> Dict:
        """Delete datasets"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('DELETE', '/api/v1/datasets', json=data, headers=headers)

    async def update_dataset(self, dataset_id: str, data: Dict) -> Dict:
        """Update dataset"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('PUT', f'/api/v1/datasets/{dataset_id}', json=data, headers=headers)

    async def list_datasets(self, params: Dict) -> Dict:
        """List datasets"""
        headers = {'Authorization': f'Bearer {self.api_key}'}

        return await self._make_request('GET', '/api/v1/datasets', params=params, headers=headers)

    async def get_dataset_knowledge_graph(self, dataset_id: str) -> Dict:
        """Get dataset knowledge graph"""

        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('GET', f'/api/v1/datasets/{dataset_id}/knowledge_graph', headers=headers)

    async def delete_dataset_knowledge_graph(self, dataset_id: str) -> Dict:
        """Delete dataset knowledge graph"""

        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('DELETE', f'/api/v1/datasets/{dataset_id}/knowledge_graph', headers=headers)

    # Document Management
    async def upload_documents(self, dataset_id: str, files: List[Dict]) -> Dict:
        """Upload documents"""
        # Convert to multipart form data
        data = aiohttp.FormData()
        for file_info in files:
            data.add_field('file', file_info['content'], filename=file_info['filename'])
        
        # Temporarily update headers for multipart
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        url = f"{self.base_url}/api/v1/datasets/{dataset_id}/documents"
        async with self.session.post(url, data=data, headers=headers) as response:
            return await response.json()
    
    async def update_document(self, dataset_id: str, document_id: str, data: Dict) -> Dict:
        """Update document"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('PUT', f'/api/v1/datasets/{dataset_id}/documents/{document_id}', json=data, headers=headers)

    async def download_document(self, dataset_id: str, document_id: str) -> Dict:
        """Download document"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('GET', f'/api/v1/datasets/{dataset_id}/documents/{document_id}', headers=headers)

    async def list_documents(self, dataset_id: str, params: Dict) -> Dict:
        """List documents"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('GET', f'/api/v1/datasets/{dataset_id}/documents', params=params, headers=headers)

    async def delete_documents(self, dataset_id: str, data: Dict) -> Dict:
        """Delete documents"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('DELETE', f'/api/v1/datasets/{dataset_id}/documents', json=data, headers=headers)

    async def parse_documents(self, dataset_id: str, data: Dict) -> Dict:
        """Parse documents"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('POST', f'/api/v1/datasets/{dataset_id}/chunks', json=data, headers=headers)

    async def stop_parsing_documents(self, dataset_id: str, data: Dict) -> Dict:
        """Stop parsing documents"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('DELETE', f'/api/v1/datasets/{dataset_id}/chunks', json=data, headers=headers)

    # Chunk Management
    async def add_chunk(self, dataset_id: str, document_id: str, data: Dict) -> Dict:
        """Add chunk"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('POST', f'/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks', json=data, headers=headers)

    async def list_chunks(self, dataset_id: str, document_id: str, params: Dict) -> Dict:
        """List chunks"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('GET', f'/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks', params=params, headers=headers)

    async def delete_chunks(self, dataset_id: str, document_id: str, data: Dict) -> Dict:
        """Delete chunks"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('DELETE', f'/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks', json=data, headers=headers)

    async def update_chunk(self, dataset_id: str, document_id: str, chunk_id: str, data: Dict) -> Dict:
        """Update chunk"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('PUT', f'/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}', json=data, headers=headers)

    async def retrieve_chunks(self, data: Dict) -> Dict:
        """Retrieve chunks"""
        headers = {'Authorization': f'Bearer {self.api_key}'}

        return await self._make_request('POST', '/api/v1/retrieval', json=data, headers=headers)

    # Chat Assistant Management
    async def create_chat_assistant(self, data: Dict) -> Dict:
        """Create chat assistant"""
        headers = {'Authorization': f'Bearer {self.api_key}'}

        return await self._make_request('POST', '/api/v1/chats', json=data, headers=headers)

    async def update_chat_assistant(self, chat_id: str, data: Dict) -> Dict:
        """Update chat assistant"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('PUT', f'/api/v1/chats/{chat_id}', json=data, headers=headers)

    async def delete_chat_assistants(self, data: Dict) -> Dict:
        """Delete chat assistants"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('DELETE', '/api/v1/chats', json=data, headers=headers)

    async def list_chat_assistants(self, params: Dict) -> Dict:
        """List chat assistants"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('GET', '/api/v1/chats', params=params, headers=headers)

    # Session Management
    async def create_session_with_chat_assistant(self, chat_id: str, data: Dict) -> Dict:
        """Create session with chat assistant"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('POST', f'/api/v1/chats/{chat_id}/sessions', json=data, headers=headers)

    async def update_chat_assistant_session(self, chat_id: str, session_id: str, data: Dict) -> Dict:
        """Update chat assistant session"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('PUT', f'/api/v1/chats/{chat_id}/sessions/{session_id}', json=data, headers=headers)

    async def list_chat_assistant_sessions(self, chat_id: str, params: Dict) -> Dict:
        """List chat assistant sessions"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('GET', f'/api/v1/chats/{chat_id}/sessions', params=params, headers=headers)

    async def delete_chat_assistant_sessions(self, chat_id: str, data: Dict) -> Dict:
        """Delete chat assistant sessions"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('DELETE', f'/api/v1/chats/{chat_id}/sessions', json=data, headers=headers)

    async def converse_with_chat_assistant(self, chat_id: str, data: Dict) -> AsyncIterator[Dict]:
        """Converse with chat assistant"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        endpoint = f"/api/v1/chats/{chat_id}/completions"
        if data.get('stream', True):
            async for chunk in self._stream_request('POST', endpoint, json=data, headers=headers):
                yield chunk
        else:
            result = await self._make_request('POST', endpoint, json=data, headers=headers)
            yield result
    
    async def create_session_with_agent(self, agent_id: str, data: Dict, params: Dict = None) -> Dict:
        """Create session with agent"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('POST', f'/api/v1/agents/{agent_id}/sessions', json=data, params=params, headers=headers)
    
    async def converse_with_agent(self, agent_id: str, data: Dict) -> AsyncIterator[Dict]:
        """Converse with agent"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        endpoint = f"/api/v1/agents/{agent_id}/completions"
        
        # Convert data to binary JSON
        json_data = json.dumps(data).encode('utf-8')
        print(f"\n{'*'*5} Sending request to converse_with_agent in ragflow_client : {json_data}")
        if data.get('stream', True):
            async for chunk in self._stream_request('POST', endpoint, data=json_data, headers=headers):
                yield chunk
        else:
            result = await self._make_request('POST', endpoint, data=json_data, headers=headers)
            yield result
    
    async def list_agent_sessions(self, agent_id: str, params: Dict) -> Dict:
        """List agent sessions"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('GET', f'/api/v1/agents/{agent_id}/sessions', params=params, headers=headers)
    
    async def delete_agent_sessions(self, agent_id: str, data: Dict) -> Dict:
        """Delete agent sessions"""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return await self._make_request('DELETE', f'/api/v1/agents/{agent_id}/sessions', json=data, headers=headers)

    async def generate_related_questions(self, data: Dict) -> Dict:
        """Generate related questions"""
        # Note: This endpoint uses login token, may need different auth  #need to check
        headers = {'Authorization': f'Bearer {self.sp_api_key}'}
        return await self._make_request('POST', '/v1/sessions/related_questions', json=data, headers=headers)

    # Agent Management
    async def list_agents(self, params: Dict) -> Dict:
        """List agents"""
        headers = {'Authorization': f"Bearer {self.api_key}"}
        return await self._make_request('GET', '/api/v1/agents', params=params, headers=headers)

    async def create_agent(self, data: Dict) -> Dict:
        """Create agent"""
        headers = {'Authorization': f"Bearer {self.api_key}"}
        return await self._make_request('POST', '/api/v1/agents', json=data, headers=headers)

    async def update_agent(self, agent_id: str, data: Dict) -> Dict:
        """Update agent"""
        headers = {'Authorization': f"Bearer {self.api_key}"}
        return await self._make_request('PUT', f'/api/v1/agents/{agent_id}', json=data, headers=headers)

    async def delete_agent(self, agent_id: str) -> Dict:
        """Delete agent"""
        headers = {'Authorization': f"Bearer {self.api_key}"}
        return await self._make_request('DELETE', f'/api/v1/agents/{agent_id}', json={}, headers=headers)
