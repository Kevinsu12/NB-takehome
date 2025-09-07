from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from app.nodes.retrieve import retrieve_node
from app.nodes.ingest import ingest_node
from app.nodes.draft import draft_node
from app.nodes.validate import validate_node
from app.nodes.revise import revise_node
from app.nodes.output import output_node
from app.schemas.market_context import MarketContext
from app.rag.vectorStore import VectorStore
from app.rag.pdfLoader import ChunkMetadata
from app.rag.pdfLoader import PDFLoader
import logging

logger = logging.getLogger(__name__)


class MarketContextState(TypedDict):
    period: str
    documents: list[str]
    processed_data: dict
    draft_context: dict
    validated_context: MarketContext
    final_context: MarketContext
    formatted_context: str
    vectorstore: VectorStore
    error: str | None


class MarketContextPipeline:
    """LangGraph pipeline for market context generation."""
    
    def __init__(self):
        self.graph = None
        self.vectorstore = None
        
    async def initialize(self):
        """Initialize the pipeline components."""
        # Initialize vector store
        self.vectorstore = VectorStore()
        pdf_loader = PDFLoader()
        
        # Load and index documents with proper metadata
        documents, metadata = await pdf_loader.load_documents_with_metadata("data/pdf")
        if documents:
            await self.vectorstore.build_index(documents, metadata)
            logger.info(f"Indexed {len(documents)} documents with proper source file tracking")
        else:
            logger.warning("No documents found to index")
        
        # Build the graph
        self._build_graph()
        
    def _build_graph(self):
        """Build the LangGraph DAG."""
        workflow = StateGraph(MarketContextState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("ingest", ingest_node)
        workflow.add_node("draft", draft_node)
        workflow.add_node("validate", validate_node)
        workflow.add_node("revise", revise_node)
        workflow.add_node("output", output_node)
        
        # Define edges
        workflow.add_edge("retrieve", "ingest")
        workflow.add_edge("ingest", "draft")
        workflow.add_edge("draft", "validate")
        
        # Conditional edges from validate
        workflow.add_conditional_edges(
            "validate",
            self._should_revise,
            {
                True: "revise",   # Revise if validation issues found
                False: "output"   # Go to output if validation passed
            }
        )
        workflow.add_edge("revise", "output")
        workflow.add_edge("output", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        self.graph = workflow.compile()
    
    def _should_revise(self, state: dict) -> bool:
        """Determine if revision is needed based on validation results."""
        # If there's an error, we should revise
        if state.get("error"):
            logger.info("Revision needed due to validation error")
            return True
        
        # If validation passed cleanly (no warnings), skip revision
        validated_context = state.get("validated_context")
        if validated_context:
            logger.info("Validation passed - skipping revision for efficiency")
            return False
        
        # Default to revision if unclear
        logger.info("Validation status unclear - proceeding with revision")
        return True
        
    async def run(self, period: str) -> str:
        """Run the pipeline for a given period."""
        if not self.graph:
            raise RuntimeError("Pipeline not initialized")
            
        initial_state = MarketContextState(
            period=period,
            documents=[],
            processed_data={},
            draft_context={},
            validated_context=None,
            final_context=None,
            formatted_context="",
            vectorstore=self.vectorstore,
            error=None
        )
        
        # VectorStore is now passed to nodes via the state object
        
        result = await self.graph.ainvoke(initial_state)
        
        if result.get("error"):
            raise RuntimeError(result["error"])
            
        return result["formatted_context"]