import asyncio
import time
import logging
from typing import List
from collections import deque
from app.config import RateLimitConfig, DEFAULT_WINDOW_SIZE, DEFAULT_TOKEN_ESTIMATION_RATIO, DEFAULT_SAFETY_MARGIN
import random
logger = logging.getLogger(__name__)


class TokenTracker:
    """Track token usage over time windows."""
    
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE):
        self.window_size = window_size  # seconds
        self.usage_history = deque()  # (timestamp, tokens_used)
        self.lock = asyncio.Lock()
    
    async def add_usage(self, tokens: int):
        """Add token usage and clean old entries."""
        async with self.lock:
            now = time.monotonic()
            self.usage_history.append((now, tokens))
            
            # Remove entries older than window
            cutoff = now - self.window_size
            while self.usage_history and self.usage_history[0][0] < cutoff:
                self.usage_history.popleft()
    
    async def get_usage_in_window(self) -> int:
        """Get total tokens used in the current window."""
        async with self.lock:
            now = time.monotonic()
            cutoff = now - self.window_size
            
            total = 0
            for timestamp, tokens in self.usage_history:
                if timestamp >= cutoff:
                    total += tokens
            
            return total


class RateLimiter:
    """Rate limiter for API calls with token tracking."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.token_tracker = TokenTracker()
        self.request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.request_times = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self, estimated_tokens: int = 1000) -> bool:
        """Acquire permission to make a request."""
        # Wait for semaphore (concurrent request limit)
        await self.request_semaphore.acquire()
        
        try:
            # Check rate limits
            await self._wait_for_rate_limit()
            await self._wait_for_token_limit(estimated_tokens)
            
            # Record this request
            async with self.lock:
                self.request_times.append(time.monotonic())
            
            return True
            
        except Exception as e:
            self.request_semaphore.release()
            raise e
    
    def release(self, actual_tokens: int = 0):
        """Release the semaphore and record actual token usage."""
        self.request_semaphore.release()
        
        if actual_tokens > 0:
            # Record actual token usage
            asyncio.create_task(self.token_tracker.add_usage(actual_tokens))
    

    async def _wait_for_rate_limit(self):
        """Block until we're under the requests-per-minute cap, without sleeping under lock."""
        while True:
            wait_time = 0.0
            async with self.lock:
                now = time.monotonic()
                cutoff = now - DEFAULT_WINDOW_SIZE

                # prune out-of-window start times
                while self.request_times and self.request_times[0] < cutoff:
                    self.request_times.popleft()

                # if we're under the cap, we're done
                if len(self.request_times) < self.config.requests_per_minute:
                    return

                # otherwise compute how long until the oldest falls out of window
                oldest = self.request_times[0]
                wait_time = max(0.0, DEFAULT_WINDOW_SIZE - (now - oldest) + DEFAULT_SAFETY_MARGIN)

            # sleep OUTSIDE the lock; if wait_time == 0, yield once to let others progress
            if wait_time <= 0:
                await asyncio.sleep(0)  # cooperative yield, then recheck
            else:
                # tiny jitter prevents a thundering herd when many tasks wake at the same time
                jitter = 0.002 + 0.006 * random.random()
                to_sleep = wait_time + jitter
                logger.warning(f"RPM limit hit: sleeping {to_sleep:.3f}s")
                await asyncio.sleep(to_sleep)
    
    async def _wait_for_token_limit(self, estimated_tokens: int):
        """Wait if we've hit the tokens per minute limit."""
        while True:
            current_usage = await self.token_tracker.get_usage_in_window()
            
            if current_usage + estimated_tokens > self.config.tokens_per_minute:
                # Calculate wait time (simplified - wait for window to reset)
                wait_time = DEFAULT_WINDOW_SIZE  # Wait for full minute
                logger.warning(f"Token limit reached. Current: {current_usage}, "
                              f"Requested: {estimated_tokens}, Waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                # Continue loop to re-check after waking
            else:
                break


class RateLimitedLLMClient:
    """Wrapper around LLMClient that adds rate limiting."""
    
    def __init__(self, llm_client, rate_limiter: RateLimiter):
        self.llm_client = llm_client
        self.rate_limiter = rate_limiter
    
    async def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0) -> str:
        """Generate text with rate limiting."""
        # Estimate tokens (rough approximation)
        estimated_tokens = len(system_prompt + user_prompt) // DEFAULT_TOKEN_ESTIMATION_RATIO
        
        # Acquire rate limit permission
        await self.rate_limiter.acquire(estimated_tokens)
        
        try:
            # Make the actual API call
            result = await self.llm_client.generate(system_prompt, user_prompt, temperature)
            
            # Estimate actual tokens used (rough approximation)
            actual_tokens = len(result) // DEFAULT_TOKEN_ESTIMATION_RATIO + estimated_tokens
            
            # Release and record usage
            self.rate_limiter.release(actual_tokens)
            
            return result
            
        except Exception as e:
            # Release on error
            self.rate_limiter.release(0)
            raise e
    
    async def get_embeddings(self, texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100):
        """Get embeddings with rate limiting."""
        # Estimate tokens for embeddings
        estimated_tokens = sum(len(text) for text in texts) // DEFAULT_TOKEN_ESTIMATION_RATIO
        
        await self.rate_limiter.acquire(estimated_tokens)
        
        try:
            result = await self.llm_client.get_embeddings(texts, model, batch_size)
            
            # Record usage
            actual_tokens = estimated_tokens  # Embeddings use similar token count
            self.rate_limiter.release(actual_tokens)
            
            return result
            
        except Exception as e:
            self.rate_limiter.release(0)
            raise e
