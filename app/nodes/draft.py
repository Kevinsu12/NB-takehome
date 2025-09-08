from app.utils.llm_utils import create_llm_client
import logging
import json

logger = logging.getLogger(__name__)


async def draft_node(state: dict) -> dict:
    """Generate initial market context draft using OpenAI."""
    processed_data = state["processed_data"]
    period = state["period"]
    
    logger.info(f"Drafting market context for {period}")
    
    try:
        # Create LLM client with optional rate limiting and config
        rate_limiter = state.get("rate_limiter")
        config = state.get("config")
        llm_client = create_llm_client(rate_limiter, config)
        
        # Load prompts
        with open("app/prompts/system.md", "r") as f:
            system_prompt = f.read()
        with open("app/prompts/user.md", "r") as f:
            user_template = f.read()
        with open("app/prompts/style.md", "r") as f:
            style_guide = f.read()
        with open("app/prompts/fewshot.md", "r") as f:
            fewshot_examples = f.read()
        
        # Prepare context from retrieved documents
        retrieved_context = "\n\n".join(processed_data.get("documents", [])[:3])
        
        # Format key statistics for the prompt
        market_data = processed_data.get("market_data", {})
        key_stats_json = json.dumps(market_data, indent=2)
        
        # Format user prompt with actual data
        user_prompt = user_template.format(
            period=period,
            retrieved_context=retrieved_context,
            key_stats_json=key_stats_json
        )
        
        # Combine system prompt with style guide and few-shot examples
        full_system_prompt = system_prompt + "\n\n" + style_guide + "\n\n" + fewshot_examples
        
        logger.info(f"Generating draft using OpenAI with config temperature")
        
        # Generate draft using config temperature
        draft_response = await llm_client.generate(
            system_prompt=full_system_prompt,
            user_prompt=user_prompt
            # temperature will use config value
        )
        
        # Parse the response as JSON
        try:
            draft_context = json.loads(draft_response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.error(f"Response was: {draft_response[:500]}...")
            # Try to extract JSON from response if wrapped in markdown
            if "```json" in draft_response:
                json_start = draft_response.find("```json") + 7
                json_end = draft_response.find("```", json_start)
                if json_end > json_start:
                    json_content = draft_response[json_start:json_end].strip()
                    draft_context = json.loads(json_content)
                else:
                    raise ValueError("Could not extract JSON from LLM response")
            else:
                raise ValueError("LLM response is not valid JSON")
        
        logger.info("Draft generation completed successfully")
        return {**state, "draft_context": draft_context}
        
    except Exception as e:
        logger.error(f"Error in draft_node: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {**state, "error": str(e), "draft_context": {}}