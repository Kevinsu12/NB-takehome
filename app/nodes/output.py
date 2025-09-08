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
        
        # Initialize LLM client with config
        config = state.get("config")
        llm_client = LLMClient(config)
        
        # Get retrieved chunks information
        retrieved_chunks = state.get("retrieved_chunks", [])
        
        # Create formatting prompt
        system_prompt = """You are a financial report formatter. Convert structured market context data into a well-formatted paragraph report.

Format the output as a professional market context report with:
1. A clear headline
2. Key market drivers as bullet points
3. A comprehensive narrative paragraph
4. Key statistics highlighted
5. Data sources listed
6. Source document references (chunk IDs and source files)

Make it readable and professional for financial professionals."""

        # Include chunk information in the prompt
        chunk_info = ""
        if retrieved_chunks:
            chunk_info = "\n\nSource Document References:\n"
            for i, chunk in enumerate(retrieved_chunks, 1):
                chunk_info += f"{i}. Chunk ID: {chunk['chunk_id']} | Source: {chunk['source_file']} | Page: {chunk['page_number']} | Similarity: {chunk['similarity_score']:.3f}\n"

        user_prompt = f"""Convert this market context JSON into a formatted paragraph report:

{json.dumps(validated_context.model_dump(), indent=2)}{chunk_info}

Format it as a professional market context report that could be sent to clients or included in a quarterly report. Include the source document references at the end."""

        # Generate formatted output using config temperature
        logger.info("Generating formatted paragraph using OpenAI")
        formatted_context = await llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
            # temperature will use config value
        )
        
        logger.info("Successfully generated formatted market context")
        return {**state, "formatted_context": formatted_context}
        
    except Exception as e:
        logger.error(f"Error in output_node: {str(e)}")
        return {**state, "formatted_context": f"Error formatting context: {str(e)}"}
