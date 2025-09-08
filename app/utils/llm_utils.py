"""Utility functions for LLM client creation and management."""
from app.clients.api_clients import LLMClient
from app.clients.rate_limiter import RateLimitedLLMClient, RateLimiter
from app.config import RateLimitConfig


def create_llm_client(rate_limiter: RateLimiter = None, config=None) -> RateLimitedLLMClient:
    """
    Create a rate-limited LLM client.
    
    Args:
        rate_limiter: Optional rate limiter instance. If None, creates a default one.
        config: Optional API config instance. If None, uses environment variables.
        
    Returns:
        RateLimitedLLMClient with rate limiting enabled
    """
    base_llm_client = LLMClient(config)
    
    # Always use rate limiting - create default if none provided
    if not rate_limiter:
        rate_limiter = RateLimiter(RateLimitConfig())
    
    return RateLimitedLLMClient(base_llm_client, rate_limiter)
