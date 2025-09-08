from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from app.app import MarketContextPipeline
from pydantic import BaseModel
import logging
import os
import traceback
import re

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
    draft_json: dict  # Add the draft JSON output
    retrieved_chunks: list  # Add the retrieved chunk information


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
        result = await pipeline.run(period)
        
        logger.info(f"Successfully generated market context for {period}")
        return MarketContextResponse(
            formatted_context=result["formatted_context"],
            period=period,
            draft_json=result["draft_json"],
            retrieved_chunks=result.get("retrieved_chunks", [])
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
        
        # Get rate limiter status
        rate_limiter_status = "active" if pipeline.rate_limiter else "inactive"
        
        return {
            "status": "healthy", 
            "service": "market-context-generator",
            "vector_store": vector_status,
            "rate_limiter": rate_limiter_status,
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


@app.get("/rate-limit-status")
async def rate_limit_status():
    """Get current rate limiting status."""
    try:
        if not pipeline.rate_limiter:
            return {"error": "Rate limiter not initialized"}
        
        # Get current usage
        current_requests = len(pipeline.rate_limiter.request_times)
        current_tokens = await pipeline.rate_limiter.token_tracker.get_usage_in_window()
        
        return {
            "rate_limiter": {
                "requests_per_minute": pipeline.rate_limiter.config.requests_per_minute,
                "tokens_per_minute": pipeline.rate_limiter.config.tokens_per_minute,
                "max_concurrent_requests": pipeline.rate_limiter.config.max_concurrent_requests,
                "current_requests": current_requests,
                "current_tokens": current_tokens,
                "available_requests": pipeline.rate_limiter.config.requests_per_minute - current_requests,
                "available_tokens": pipeline.rate_limiter.config.tokens_per_minute - current_tokens
            }
        }
    except Exception as e:
        return {"error": str(e)}


def _is_valid_period_format(period: str) -> bool:
    """Validate period format (YYYY-QX)."""
    pattern = r'^\d{4}-Q[1-4]$'
    return bool(re.match(pattern, period))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")