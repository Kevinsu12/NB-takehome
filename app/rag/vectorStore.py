import numpy as np
import faiss
import logging
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import os
from app.rag.pdfLoader import ChunkMetadata
from app.clients.api_clients import LLMClient

logger = logging.getLogger(__name__)

# Deterministic constants
EMBEDDING_MODEL = "text-embedding-3-small"  # Fixed model name for consistency
EMBEDDING_DIM = 1536  # Dimension for text-embedding-3-small
INDEX_DIR = "rag/index"


class VectorStore:
    """FAISS-based vector store with persistence and deterministic behavior."""
    
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index = None
        self.documents = []
        self.metadata = []
        self.embedding_model = EMBEDDING_MODEL
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM client for embeddings
        self.llm_client = LLMClient()
        
        # File paths for persistence
        self.index_path = self.index_dir / "faiss.index"
        self.documents_path = self.index_dir / "documents.pkl"
        self.metadata_path = self.index_dir / "metadata.json"
        self.config_path = self.index_dir / "config.json"
        
    async def build_index(self, documents: List[str], metadata: List[ChunkMetadata]) -> None:
        """Build FAISS index from documents and metadata."""
        logger.info(f"Building vector index for {len(documents)} documents")
        
        if len(documents) != len(metadata):
            raise ValueError("Documents and metadata lists must have the same length")
        
        # Store documents and metadata
        self.documents = documents
        self.metadata = [self._metadata_to_dict(m) for m in metadata]
        
        # Generate real embeddings using OpenAI API
        embeddings = await self._generate_embeddings(documents)
        
        # Build FAISS index with inner product similarity
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        
        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        logger.info(f"Vector index built with {self.index.ntotal} vectors")
        
        # Persist the index
        await self.save_index()
    
    async def load_index(self) -> bool:
        """Load persisted index from disk."""
        try:
            if not all(p.exists() for p in [self.index_path, self.documents_path, self.metadata_path]):
                logger.info("Index files not found, index not loaded")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load documents
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Verify consistency
            if len(self.documents) != len(self.metadata) != self.index.ntotal:
                logger.error("Inconsistent index state, rebuild required")
                return False
            
            # Load and verify config
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    if config.get('embedding_model') != self.embedding_model:
                        logger.warning(f"Model mismatch: expected {self.embedding_model}, got {config.get('embedding_model')}")
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    async def save_index(self) -> None:
        """Persist index to disk."""
        try:
            if self.index is None:
                raise ValueError("No index to save")
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save documents
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save config
            config = {
                'embedding_model': self.embedding_model,
                'embedding_dim': EMBEDDING_DIM,
                'document_count': len(self.documents),
                'index_type': 'IndexFlatIP'
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Index saved to {self.index_dir}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def _metadata_to_dict(self, metadata: ChunkMetadata) -> Dict[str, Any]:
        """Convert ChunkMetadata to dictionary for JSON serialization."""
        return {
            'chunk_id': metadata.chunk_id,
            'source_file': metadata.source_file,
            'page_number': metadata.page_number,
            'is_market_context': metadata.is_market_context,
            'confidence_score': metadata.confidence_score
        }
    
    def _generate_chunk_id(self, document: str) -> str:
        """Generate deterministic document ID."""
        return hashlib.md5(document.encode()).hexdigest()[:16]
    
    async def _generate_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API with optimized batch processing."""
        logger.info(f"Generating embeddings for {len(documents)} documents using {self.embedding_model}")
        
        # Optimize batch size based on document count and API limits
        # OpenAI allows up to 2048 embedding inputs per request for text-embedding-3-small
        optimal_batch_size = min(100, len(documents)) if len(documents) <= 1000 else 50
        
        # Use concurrent batch processing for efficiency
        embeddings_list = await self.llm_client.get_embeddings(
            documents, 
            self.embedding_model, 
            batch_size=optimal_batch_size
        )
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        logger.info(f"Successfully generated {len(embeddings)} real embeddings")
        return embeddings
    
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter_market_context: bool = True,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents with optional filtering."""
        if not self.index or not self.documents:
            logger.warning("Vector store not initialized")
            return []
        
        # Generate query embedding using real embeddings
        query_embedding = await self._generate_embeddings([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search with larger k to allow for filtering
        search_k = min(k * 3, len(self.documents))
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                metadata = self.metadata[idx]
                
                # Apply filters
                if filter_market_context and not metadata['is_market_context']:
                    continue
                
                if metadata['confidence_score'] < min_confidence:
                    continue
                
                result = {
                    'document': self.documents[idx],
                    'metadata': metadata,
                    'score': float(score),
                    'chunk_id': metadata['chunk_id']
                }
                results.append(result)
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
        
        logger.info(f"Retrieved {len(results)} similar documents (filtered from {search_k} candidates)")
        return results
    
    async def get_documents_by_source(self, source_file: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source file."""
        if not self.documents:
            return []
        
        results = []
        for i, metadata in enumerate(self.metadata):
            if metadata['source_file'] == source_file:
                results.append({
                    'document': self.documents[i],
                    'metadata': metadata,
                    'index': i
                })
        
        return results
    
    async def get_market_context_summary(self) -> Dict[str, Any]:
        """Get summary statistics about market context documents."""
        if not self.metadata:
            return {}
        
        total_docs = len(self.metadata)
        market_docs = sum(1 for m in self.metadata if m['is_market_context'])
        
        # Group by source file
        by_source = {}
        confidence_scores = []
        
        for metadata in self.metadata:
            source = metadata['source_file']
            if source not in by_source:
                by_source[source] = {'total': 0, 'market_context': 0}
            
            by_source[source]['total'] += 1
            if metadata['is_market_context']:
                by_source[source]['market_context'] += 1
                confidence_scores.append(metadata['confidence_score'])
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            'total_documents': total_docs,
            'market_context_documents': market_docs,
            'market_context_ratio': market_docs / total_docs if total_docs > 0 else 0,
            'average_confidence': float(avg_confidence),
            'by_source': by_source,
            'embedding_model': self.embedding_model
        }
    
    def is_indexed(self) -> bool:
        """Check if the vector store has been built."""
        return self.index is not None and len(self.documents) > 0
    
    async def rebuild_index(self, documents: List[str], metadata: List[ChunkMetadata]) -> None:
        """Rebuild the index from scratch."""
        logger.info("Rebuilding vector store index")
        
        # Clear existing index
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Build new index
        await self.build_index(documents, metadata)