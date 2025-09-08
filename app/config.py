import os
from dataclasses import dataclass
from typing import Optional

# Constants
DEFAULT_WINDOW_SIZE = 60  # seconds
DEFAULT_TOKEN_ESTIMATION_RATIO = 4  # characters per token
DEFAULT_SAFETY_MARGIN = 1  # second


@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting."""
    
    # OpenAI GPT-4 Rate Limits (per minute)
    requests_per_minute: int = 50  # OpenAI's rate limit
    tokens_per_minute: int = 40000  # OpenAI's token limit
    
    # Concurrent request limits
    max_concurrent_requests: int = 5  # Max simultaneous requests
    burst_limit: int = 3  # Allow burst of requests
    
    # Token estimation
    avg_tokens_per_request: int = 2000  # For estimation purposes
    
    # Retry configuration
    max_retries: int = 3  # Max retry attempts
    base_backoff: float = 1.0  # Base wait time between retries
    
    @classmethod
    def from_env(cls) -> 'RateLimitConfig':
        """Create config from environment variables."""
        return cls(
            requests_per_minute=int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "50")),
            tokens_per_minute=int(os.getenv("RATE_LIMIT_TOKENS_PER_MINUTE", "40000")),
            max_concurrent_requests=int(os.getenv("RATE_LIMIT_MAX_CONCURRENT", "5")),
            burst_limit=int(os.getenv("RATE_LIMIT_BURST", "3")),
            avg_tokens_per_request=int(os.getenv("RATE_LIMIT_AVG_TOKENS", "2000")),
            max_retries=int(os.getenv("RATE_LIMIT_MAX_RETRIES", "3")),
            base_backoff=float(os.getenv("RATE_LIMIT_BASE_BACKOFF", "1.0"))
        )


@dataclass
class APIConfig:
    """Configuration for API clients."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_max_tokens: int = 2000
    openai_temperature: float = 0.0
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Rate Limiting
    rate_limit_config: RateLimitConfig = None
    
    # Mock Data
    use_mock_data: bool = True
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create config from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            rate_limit_config=RateLimitConfig.from_env(),
            use_mock_data=os.getenv("USE_MOCK_DATA", "true").lower() == "true"
        )
