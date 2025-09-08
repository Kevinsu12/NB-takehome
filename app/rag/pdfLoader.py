import os
import logging
from pathlib import Path
import pypdf
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
import asyncio
from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    chunk_id: str
    source_file: str
    page_number: int
    is_market_context: bool = True

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

# Deterministic constants
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100


class PDFLoader:
    """Load and process PDF documents with deterministic chunking."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def load_documents(self, pdf_directory: str) -> List[str]:
        """Load all PDF documents from the specified directory concurrently."""
        pdf_dir = Path(pdf_directory)
        
        if not pdf_dir.exists():
            logger.warning(f"PDF directory {pdf_directory} does not exist")
            return []
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found in directory")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process concurrently")
        
        # Process all PDFs concurrently using asyncio.gather
        tasks = [self._extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]
        
        try:
            # Execute all tasks concurrently, but handle exceptions individually
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Unexpected error during concurrent processing: {str(e)}")
            return []
        
        # Process results and create chunks
        documents = []
        successful_files = 0
        
        for pdf_file, result in zip(pdf_files, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {pdf_file.name}: {str(result)}")
                continue
            
            text = result
            if text and text.strip():
                chunks = self._chunk_text(text)
                documents.extend(chunks)
                logger.info(f"Processed {pdf_file.name}: {len(chunks)} chunks")
                successful_files += 1
            else:
                logger.warning(f"No extractable text found in {pdf_file.name}")
        
        logger.info(f"Successfully processed {successful_files}/{len(pdf_files)} files, "
                   f"total chunks: {len(documents)}")
        return documents

    async def load_documents_with_metadata(self, pdf_directory: str) -> tuple[List[str], List[ChunkMetadata]]:
        """Load all PDF documents and return both chunks and their metadata."""
        pdf_dir = Path(pdf_directory)
        
        if not pdf_dir.exists():
            logger.warning(f"PDF directory {pdf_directory} does not exist")
            return [], []
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found in directory")
            return [], []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process concurrently")
        
        # Process all PDFs concurrently using asyncio.gather
        tasks = [self._extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]
        
        try:
            # Execute all tasks concurrently, but handle exceptions individually
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Unexpected error during concurrent processing: {str(e)}")
            return [], []
        
        # Process results and create chunks with metadata
        documents = []
        metadata = []
        successful_files = 0
        chunk_counter = 0
        
        for pdf_file, result in zip(pdf_files, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {pdf_file.name}: {str(result)}")
                continue
            
            text = result
            if text and text.strip():
                chunks = self._chunk_text(text)
                
                # Create metadata for each chunk from this PDF
                for i, chunk in enumerate(chunks):
                    meta = ChunkMetadata(
                        chunk_id=f"chunk_{chunk_counter:04d}",
                        source_file=pdf_file.name,  # Use actual PDF filename
                        page_number=i + 1,  # Estimate page number within this PDF
                        is_market_context=True
                    )
                    metadata.append(meta)
                    chunk_counter += 1
                
                documents.extend(chunks)
                logger.info(f"Processed {pdf_file.name}: {len(chunks)} chunks")
                successful_files += 1
            else:
                logger.warning(f"No text extracted from {pdf_file.name}")
        
        if successful_files == 0:
            logger.error("No PDF files were successfully processed")
            return [], []
        
        logger.info(f"Successfully processed {successful_files}/{len(pdf_files)} files, total chunks: {len(documents)}")
        return documents, metadata
    
    async def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file using thread pool for true async behavior."""
        try:
            # Run the blocking PDF extraction in a thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self._extract_pdf_sync, pdf_path)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise
    
    def _extract_pdf_sync(self, pdf_path: Path) -> str:
        """Synchronous PDF extraction for thread pool execution."""
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into overlapping segments at natural sentence boundaries."""
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        # Split text into sentences using NLTK
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length + 1 > self.chunk_size and current_chunk:
                # We have a complete chunk, save it
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_chunk = self._create_overlap_chunk(current_chunk, sentences, i)
                current_chunk = overlap_chunk
                current_length = len(overlap_chunk)
                
                # Don't increment i, try to add the same sentence to new chunk
                continue
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                    current_length += sentence_length + 1
                else:
                    current_chunk = sentence
                    current_length = sentence_length
                
                i += 1
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very small chunks
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
        
        return chunks
    
    
    def _create_overlap_chunk(self, previous_chunk: str, sentences: List[str], current_index: int) -> str:
        """Create overlap by including last few sentences from previous chunk."""
        if not previous_chunk or self.chunk_overlap <= 0:
            return ""
        
        # Split previous chunk into sentences to find overlap
        prev_sentences = sent_tokenize(previous_chunk)
        
        # Start with empty overlap and add sentences from the end
        overlap = ""
        overlap_length = 0
        
        # Add sentences from the end of previous chunk until we reach overlap size
        for sentence in reversed(prev_sentences):
            sentence_length = len(sentence)
            if overlap_length + sentence_length + 1 <= self.chunk_overlap:
                if overlap:
                    overlap = sentence + " " + overlap
                else:
                    overlap = sentence
                overlap_length += sentence_length + 1
            else:
                break
        
        return overlap