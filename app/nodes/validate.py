from app.schemas.market_context import MarketContext
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)


async def validate_node(state: dict) -> dict:
    """Simple validation - just check if draft_context can be parsed as MarketContext."""
    draft_context = state.get("draft_context", {})
    
    logger.info("Validating market context")
    
    try:
        # Simply try to parse as MarketContext - if it works, it's valid
        validated_context = MarketContext(**draft_context)
        
        logger.info("Validation passed - draft context is valid")
        return {**state, "validated_context": validated_context, "final_context": validated_context}
        
    except ValidationError as e:
        logger.error(f"Schema validation error: {str(e)}")
        return {**state, "error": f"Schema validation failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {**state, "error": str(e)}