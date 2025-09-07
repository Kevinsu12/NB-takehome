from app.rag.vectorStore import VectorStore
import logging

logger = logging.getLogger(__name__)

# RAG constants for determinism
K = 2


async def retrieve_node(state: dict) -> dict:
    """Retrieve relevant documents based on the period."""
    period = state["period"]
    vectorstore = state["vectorstore"]
    logger.info(f"Retrieving documents for period: {period}")
    
    try:
        query = f"market trends analysis {period}"
        
        # Use the actual vectorstore to retrieve relevant documents
        if vectorstore and vectorstore.is_indexed():
            search_results = await vectorstore.similarity_search(
                query=query,
                k=K,
                filter_market_context=True,
                min_confidence=0.5
            )
            documents = [result["document"] for result in search_results]
            logger.info(f"Retrieved {len(documents)} documents from vectorstore")
        else:
            # Fallback to mock data if vectorstore not available
            logger.warning("VectorStore not available, using mock data")
            documents = [
                f"Economic indicators for {period} show mixed signals...",
                f"Market volatility in {period} driven by geopolitical factors..."
            ]
        
        logger.info(f"Retrieved {len(documents)} documents")
        return {**state, "documents": documents}
        
    except Exception as e:
        logger.error(f"Error in retrieve_node: {str(e)}")
        return {**state, "error": str(e)}