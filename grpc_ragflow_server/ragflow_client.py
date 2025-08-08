import aiohttp
import asyncio
import json
from typing import Dict, List, Optional, Union, AsyncIterator
from config import RAGFLOW_BASE_URL, API_KEY


class RAGFlowClient:
    """Async HTTP client for RAGFlow REST API"""
    
    def __init__(self, base_url: str = RAGFLOW_BASE_URL, api_key: str = API_KEY):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            },
            timeout=aiohttp.ClientTimeout(total=300)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request to RAGFlow API"""
        url = f"{self.base_url}{endpoint}"

        if 'headers' not in kwargs:
            kwargs['headers'] = {}
        kwargs['headers'].update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })

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
        except Exception as e:
            return {'error': str(e), 'code': 500}
    
    async def _stream_request(self, method: str, endpoint: str, **kwargs) -> AsyncIterator[Dict]:
        """Make streaming request to RAGFlow API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data:'):
                            try:
                                data_str = line_str[5:].strip()
                                if data_str and data_str != '[DONE]':
                                    yield json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield {'error': str(e), 'code': 500}
    
    # OpenAI-Compatible API
    async def create_chat_completion(self, chat_id: str, data: Dict) -> AsyncIterator[Dict]:
        """Create chat completion"""
        endpoint = f"/api/v1/chats_openai/{chat_id}/chat/completions"
        if data.get('stream', True):
            async for chunk in self._stream_request('POST', endpoint, json=data):
                yield chunk
        else:
            result = await self._make_request('POST', endpoint, json=data)
            yield result
    
    async def create_agent_completion(self, agent_id: str, data: Dict) -> AsyncIterator[Dict]:
        """Create agent completion"""
        endpoint = f"/api/v1/agents_openai/{agent_id}/chat/completions"
        if data.get('stream', True):
            async for chunk in self._stream_request('POST', endpoint, json=data):
                yield chunk
        else:
            result = await self._make_request('POST', endpoint, json=data)
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
        return await self._make_request('PUT', f'/api/v1/datasets/{dataset_id}/documents/{document_id}', json=data)
    
    async def download_document(self, dataset_id: str, document_id: str) -> Dict:
        """Download document"""
        return await self._make_request('GET', f'/api/v1/datasets/{dataset_id}/documents/{document_id}')
    
    async def list_documents(self, dataset_id: str, params: Dict) -> Dict:
        """List documents"""
        return await self._make_request('GET', f'/api/v1/datasets/{dataset_id}/documents', params=params)
    
    async def delete_documents(self, dataset_id: str, data: Dict) -> Dict:
        """Delete documents"""
        return await self._make_request('DELETE', f'/api/v1/datasets/{dataset_id}/documents', json=data)
    
    async def parse_documents(self, dataset_id: str, data: Dict) -> Dict:
        """Parse documents"""
        return await self._make_request('POST', f'/api/v1/datasets/{dataset_id}/chunks', json=data)
    
    async def stop_parsing_documents(self, dataset_id: str, data: Dict) -> Dict:
        """Stop parsing documents"""
        return await self._make_request('DELETE', f'/api/v1/datasets/{dataset_id}/chunks', json=data)
    
    # Chunk Management
    async def add_chunk(self, dataset_id: str, document_id: str, data: Dict) -> Dict:
        """Add chunk"""
        return await self._make_request('POST', f'/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks', json=data)
    
    async def list_chunks(self, dataset_id: str, document_id: str, params: Dict) -> Dict:
        """List chunks"""
        return await self._make_request('GET', f'/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks', params=params)
    
    async def delete_chunks(self, dataset_id: str, document_id: str, data: Dict) -> Dict:
        """Delete chunks"""
        return await self._make_request('DELETE', f'/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks', json=data)
    
    async def update_chunk(self, dataset_id: str, document_id: str, chunk_id: str, data: Dict) -> Dict:
        """Update chunk"""
        return await self._make_request('PUT', f'/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}', json=data)
    
    async def retrieve_chunks(self, data: Dict) -> Dict:
        """Retrieve chunks"""
        return await self._make_request('POST', '/api/v1/retrieval', json=data)
    
    # Chat Assistant Management
    async def create_chat_assistant(self, data: Dict) -> Dict:
        """Create chat assistant"""
        return await self._make_request('POST', '/api/v1/chats', json=data)
    
    async def update_chat_assistant(self, chat_id: str, data: Dict) -> Dict:
        """Update chat assistant"""
        return await self._make_request('PUT', f'/api/v1/chats/{chat_id}', json=data)
    
    async def delete_chat_assistants(self, data: Dict) -> Dict:
        """Delete chat assistants"""
        return await self._make_request('DELETE', '/api/v1/chats', json=data)
    
    async def list_chat_assistants(self, params: Dict) -> Dict:
        """List chat assistants"""
        return await self._make_request('GET', '/api/v1/chats', params=params)
    
    # Session Management
    async def create_session_with_chat_assistant(self, chat_id: str, data: Dict) -> Dict:
        """Create session with chat assistant"""
        return await self._make_request('POST', f'/api/v1/chats/{chat_id}/sessions', json=data)
    
    async def update_chat_assistant_session(self, chat_id: str, session_id: str, data: Dict) -> Dict:
        """Update chat assistant session"""
        return await self._make_request('PUT', f'/api/v1/chats/{chat_id}/sessions/{session_id}', json=data)
    
    async def list_chat_assistant_sessions(self, chat_id: str, params: Dict) -> Dict:
        """List chat assistant sessions"""
        return await self._make_request('GET', f'/api/v1/chats/{chat_id}/sessions', params=params)
    
    async def delete_chat_assistant_sessions(self, chat_id: str, data: Dict) -> Dict:
        """Delete chat assistant sessions"""
        return await self._make_request('DELETE', f'/api/v1/chats/{chat_id}/sessions', json=data)
    
    async def converse_with_chat_assistant(self, chat_id: str, data: Dict) -> AsyncIterator[Dict]:
        """Converse with chat assistant"""
        endpoint = f"/api/v1/chats/{chat_id}/completions"
        if data.get('stream', True):
            async for chunk in self._stream_request('POST', endpoint, json=data):
                yield chunk
        else:
            result = await self._make_request('POST', endpoint, json=data)
            yield result
    
    async def create_session_with_agent(self, agent_id: str, data: Dict, params: Dict = None) -> Dict:
        """Create session with agent"""
        return await self._make_request('POST', f'/api/v1/agents/{agent_id}/sessions', json=data, params=params)
    
    async def converse_with_agent(self, agent_id: str, data: Dict) -> AsyncIterator[Dict]:
        """Converse with agent"""
        endpoint = f"/api/v1/agents/{agent_id}/completions"
        if data.get('stream', True):
            async for chunk in self._stream_request('POST', endpoint, json=data):
                yield chunk
        else:
            result = await self._make_request('POST', endpoint, json=data)
            yield result
    
    async def list_agent_sessions(self, agent_id: str, params: Dict) -> Dict:
        """List agent sessions"""
        return await self._make_request('GET', f'/api/v1/agents/{agent_id}/sessions', params=params)
    
    async def delete_agent_sessions(self, agent_id: str, data: Dict) -> Dict:
        """Delete agent sessions"""
        return await self._make_request('DELETE', f'/api/v1/agents/{agent_id}/sessions', json=data)
    
    async def generate_related_questions(self, data: Dict) -> Dict:
        """Generate related questions"""
        # Note: This endpoint uses login token, may need different auth
        return await self._make_request('POST', '/v1/sessions/related_questions', json=data)
    
    # Agent Management
    async def list_agents(self, params: Dict) -> Dict:
        """List agents"""
        return await self._make_request('GET', '/api/v1/agents', params=params)
    
    async def create_agent(self, data: Dict) -> Dict:
        """Create agent"""
        return await self._make_request('POST', '/api/v1/agents', json=data)
    
    async def update_agent(self, agent_id: str, data: Dict) -> Dict:
        """Update agent"""
        return await self._make_request('PUT', f'/api/v1/agents/{agent_id}', json=data)
    
    async def delete_agent(self, agent_id: str) -> Dict:
        """Delete agent"""
        return await self._make_request('DELETE', f'/api/v1/agents/{agent_id}', json={})
