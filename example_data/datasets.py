"""
Example datasets configuration for RAGFlow gRPC testing.
"""
import os
from typing import List, Dict, Any

# Path to example documents
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")

def get_example_datasets() -> List[Dict[str, Any]]:
    """Get example dataset configurations."""
    return [
        {
            "name": "AI_Knowledge_Base",
            "description": "Comprehensive knowledge base about Artificial Intelligence, Machine Learning, and related technologies",
            "embedding_model": "BAAI/bge-large-zh-v1.5",
            "permission": "me",
            "chunk_method": "intelligent",
            "documents": [
                "ai_overview.txt",
                "machine_learning_basics.txt",
                "nlp_essentials.txt",
            ]
        },
        {
            "name": "ML_Fundamentals",
            "description": "Essential machine learning concepts and techniques",
            "embedding_model": "text-embedding-ada-002",
            "permission": "me", 
            "chunk_method": "manual",
            "documents": [
                "machine_learning_basics.txt",
            ]
        }
    ]

def get_document_path(document_name: str) -> str:
    """Get full path to a document file."""
    return os.path.join(DOCUMENTS_DIR, document_name)

def read_document_content(document_name: str) -> bytes:
    """Read document content as bytes."""
    document_path = get_document_path(document_name)
    with open(document_path, 'rb') as f:
        return f.read()

def get_example_documents() -> List[Dict[str, Any]]:
    """Get example document configurations."""
    documents = []
    
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith('.txt'):
            documents.append({
                "name": filename,
                "path": get_document_path(filename),
                "type": "text/plain",
                "description": f"Example document: {filename}"
            })
    
    return documents

def get_sample_chunks() -> List[Dict[str, Any]]:
    """Get sample chunk data for testing."""
    return [
        {
            "content": "Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "important_keywords": ["Machine Learning", "artificial intelligence", "learn", "experience"],
            "questions": [
                "What is Machine Learning?",
                "How does Machine Learning relate to AI?",
                "Can machines learn without programming?"
            ]
        },
        {
            "content": "Deep Learning uses artificial neural networks with multiple layers to model complex patterns in data, inspired by the human brain structure.",
            "important_keywords": ["Deep Learning", "neural networks", "multiple layers", "human brain"],
            "questions": [
                "What is Deep Learning?",
                "How are neural networks structured?",
                "What inspires Deep Learning architecture?"
            ]
        },
        {
            "content": "Natural Language Processing enables computers to understand, interpret, and generate human language in meaningful ways.",
            "important_keywords": ["Natural Language Processing", "NLP", "understand", "interpret", "generate", "human language"],
            "questions": [
                "What is Natural Language Processing?",
                "How do computers understand human language?",
                "What can NLP systems do?"
            ]
        }
    ]

def get_sample_chat_messages() -> List[List[Dict[str, str]]]:
    """Get sample chat message sequences for testing."""
    return [
        [
            {"role": "user", "content": "What is artificial intelligence?"},
            {"role": "assistant", "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can think and act like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation."}
        ],
        [
            {"role": "user", "content": "Explain machine learning in simple terms"},
            {"role": "assistant", "content": "Machine Learning is like teaching a computer to recognize patterns and make decisions based on examples. Instead of programming specific rules, we show the computer lots of data examples, and it learns to make predictions or decisions on new, unseen data."}
        ],
        [
            {"role": "user", "content": "What are the main types of machine learning?"},
            {"role": "assistant", "content": "There are three main types of machine learning: 1) Supervised Learning - learns from labeled examples, 2) Unsupervised Learning - finds patterns in data without labels, and 3) Reinforcement Learning - learns through trial and error with rewards and penalties."}
        ]
    ]

def get_sample_agent_dsl() -> Dict[str, Any]:
    """Get sample agent DSL configuration."""
    return {
        "graph": {
            "nodes": [
                {
                    "id": "start",
                    "type": "start",
                    "name": "Start"
                },
                {
                    "id": "analyze",
                    "type": "categorize",
                    "name": "Analyze Query",
                    "parameters": {
                        "categories": ["technical", "general", "specific"]
                    }
                },
                {
                    "id": "respond",
                    "type": "generate", 
                    "name": "Generate Response",
                    "parameters": {
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.7
                    }
                }
            ],
            "edges": [
                {"from": "start", "to": "analyze"},
                {"from": "analyze", "to": "respond"}
            ]
        }
    }
