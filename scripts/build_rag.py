#!/usr/bin/env python3
"""
CLI script to build RAG index from PDF documents.

Usage:
    python scripts/build_rag.py [--pdf-dir data/pdf] [--index-dir rag/index] [--rebuild]
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.pdfLoader import PDFLoader, ChunkMetadata
from app.rag.vectorStore import VectorStore


def setup_logging(verbose: bool = False):
    """Configure logging for the CLI script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


async def build_index(pdf_dir: str, index_dir: str, rebuild: bool = False):
    """Build or rebuild the RAG index."""
    logger = logging.getLogger(__name__)
    
    # Initialize components
    pdf_loader = PDFLoader()
    vectorstore = VectorStore(index_dir=index_dir)
    
    # Check if index already exists
    if not rebuild and vectorstore.is_indexed():
        logger.info(f"Index already exists at {index_dir}. Use --rebuild to force rebuild.")
        return
    
    logger.info(f"Loading PDFs from {pdf_dir}")
    documents, metadata = await pdf_loader.load_documents_with_metadata(pdf_dir)
    
    if not documents:
        logger.error(f"No documents found in {pdf_dir}")
        return
    
    logger.info(f"Loaded {len(documents)} documents with proper source file tracking")
    
    logger.info("Building vector index...")
    await vectorstore.build_index(documents, metadata)
    
    logger.info(f"Successfully built index with {len(documents)} documents")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Build RAG index from PDF documents")
    parser.add_argument("--pdf-dir", default="data/pdf", help="Directory containing PDF files")
    parser.add_argument("--index-dir", default="rag/index", help="Directory to store the index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if index exists")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Ensure directories exist
    Path(args.pdf_dir).mkdir(parents=True, exist_ok=True)
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        await build_index(args.pdf_dir, args.index_dir, args.rebuild)
    except Exception as e:
        logging.error(f"Failed to build index: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())