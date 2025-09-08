from app.utils.llm_utils import create_llm_client
from app.schemas.market_context import MarketContext
import logging
import json

logger = logging.getLogger(__name__)


async def revise_node(state: dict) -> dict:
    """Revise and finalize the market context."""
    validated_context = state["validated_context"]
    
    logger.info("Revising market context")
    
    try:
        # Create LLM client with optional rate limiting and config
        rate_limiter = state.get("rate_limiter")
        config = state.get("config")
        llm_client = create_llm_client(rate_limiter, config)
        
        # Load revision prompts
        with open("app/prompts/system.md", "r") as f:
            system_prompt = f.read()
        
        revision_prompt = f"""
        Please review and refine the following market context for final publication.
        Ensure clarity, consistency, and professional tone.
        
        Current context:
        {validated_context.model_dump_json(indent=2)}
        
        Return the refined context in the same JSON format.
        """
        
        # Generate revision with deterministic settings
        revised_response = await llm_client.generate(
            system_prompt=system_prompt + "\n\nFocus on clarity and consistency.",
            user_prompt=revision_prompt
            # temperature will use config value
        )
        
        # Parse and validate the revised context
        revised_data = json.loads(revised_response)
        final_context = MarketContext(**revised_data)
        
        logger.info("Revision completed")
        return {**state, "final_context": final_context}
        
    except Exception as e:
        logger.error(f"Error in revise_node: {str(e)}")
        # Fall back to validated context if revision fails
        return {**state, "final_context": validated_context}