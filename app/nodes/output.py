"""
Output node for formatting market context into readable paragraphs.
"""

import json
import logging
from typing import Dict, Any

from app.clients.api_clients import LLMClient

logger = logging.getLogger(__name__)


async def output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the validated market context JSON into a readable paragraph.
    
    Args:
        state: Pipeline state containing validated_context
        
    Returns:
        Updated state with formatted_context
    """
    logger.info("Formatting market context into paragraph")
    
    try:
        validated_context = state.get("validated_context")
        if not validated_context:
            logger.error("No validated context found in state")
            return {**state, "formatted_context": "No market context available to format"}
        
        # Initialize LLM client
        llm_client = LLMClient()
        
        # Create formatting prompt
        system_prompt = """You are a financial report formatter. Convert structured market context data into a well-formatted paragraph report.

Format the output as a professional market context report with:
1. A clear headline
2. Key market drivers as bullet points
3. A comprehensive narrative paragraph
4. Key statistics highlighted
5. Data sources listed

Make it readable and professional for financial professionals."""

        user_prompt = f"""Convert this market context JSON into a formatted paragraph report:

{json.dumps(validated_context.model_dump(), indent=2)}

Format it as a professional market context report that could be sent to clients or included in a quarterly report."""

        # Generate formatted output
        logger.info("Generating formatted paragraph using OpenAI")
        formatted_context = await llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1  # Slightly higher for more natural formatting
        )
        
        logger.info("Successfully generated formatted market context")
        return {**state, "formatted_context": formatted_context}
        
    except Exception as e:
        logger.error(f"Error in output_node: {str(e)}")
        return {**state, "formatted_context": f"Error formatting context: {str(e)}"}
