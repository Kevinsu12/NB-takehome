from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from app.app import MarketContextPipeline
from app.schemas.market_context import MarketContext
from pydantic import BaseModel
import logging
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Market Context Generator", 
    version="1.0.0",
    description="Generate market context reports using LangGraph DAG processing and RAG capabilities"
)

pipeline = MarketContextPipeline()


class MarketContextResponse(BaseModel):
    """Response model for formatted market context."""
    formatted_context: str
    period: str


@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup."""
    try:
        await pipeline.initialize()
        logger.info("Market Context Generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise


@app.post("/market-context", response_model=MarketContextResponse)
async def generate_market_context(
    period: str = Query(..., description="Time period (e.g., '2025-Q2')", example="2025-Q2")
):
    """
    Generate market context for the specified period.
    
    Args:
        period: Time period in format YYYY-QX (e.g., "2025-Q2")
        
    Returns:
        MarketContext: Generated market context report with validation
        
    Raises:
        HTTPException: 400 for validation errors, 500 for processing errors
    """
    try:
        logger.info(f"Generating market context for period: {period}")
        
        # Validate period format
        if not _is_valid_period_format(period):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid period format. Expected YYYY-QX format, got: {period}"
            )
        
        # Run the LangGraph pipeline
        formatted_context = await pipeline.run(period)
        
        logger.info(f"Successfully generated market context for {period}")
        return MarketContextResponse(
            formatted_context=formatted_context,
            period=period
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except ValueError as e:
        # Business rule validation failures
        error_msg = str(e)
        logger.error(f"Validation error for {period}: {error_msg}")
        
        # Extract specific validation failure reason
        if "Schema validation failed" in error_msg:
            detail = f"Schema validation failed: {error_msg.split('Schema validation failed: ', 1)[1]}"
        elif "Business rule validation failed" in error_msg:
            detail = f"Business rule validation failed: {error_msg.split('Business rule validation failed: ', 1)[1]}"
        else:
            detail = f"Validation failed: {error_msg}"
            
        raise HTTPException(status_code=400, detail=detail)
    except Exception as e:
        # System/processing errors
        error_msg = str(e)
        logger.error(f"Error generating market context for {period}: {error_msg}")
        
        # Include traceback in debug mode
        if os.getenv("DEBUG", "false").lower() == "true":
            error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
            
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {error_msg}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if pipeline is initialized
        if not pipeline.graph:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "Pipeline not initialized"}
            )
        
        # Check if vector store is available
        vector_status = "available" if pipeline.vectorstore and pipeline.vectorstore.is_indexed() else "not_indexed"
        
        return {
            "status": "healthy", 
            "service": "market-context-generator",
            "vector_store": vector_status,
            "version": "1.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "reason": str(e),
                "service": "market-context-generator"
            }
        )


def _is_valid_period_format(period: str) -> bool:
    """Validate period format (YYYY-QX)."""
    import re
    pattern = r'^\d{4}-Q[1-4]$'
    return bool(re.match(pattern, period))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")