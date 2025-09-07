import httpx
import asyncio
import logging
import json
import time
import random
from typing import Any, Dict, List, Optional
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class BaseAPIClient:
    """Base class for API clients with retry and backoff logic."""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers={"Cache-Control": "max-age=300"}  # 5 minute cache hint
        )
    
    async def _request_with_retry(
        self, 
        method: str, 
        url: str, 
        max_retries: int = 3,
        base_backoff: float = 1.0,
        per_call_timeout: float = 10.0,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with jittered exponential backoff retry logic."""
        for attempt in range(max_retries + 1):
            try:
                # Construct full URL
                full_url = f"{self.base_url}{url}" if url.startswith('/') else url
                # Per-call timeout override
                timeout = httpx.Timeout(per_call_timeout)
                response = await self.client.request(method, full_url, timeout=timeout, **kwargs)
                
                # Check for retry-worthy status codes
                if response.status_code in [429, 500, 502, 503, 504]:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", 
                        request=response.request, 
                        response=response
                    )
                
                response.raise_for_status()
                return response
                
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                if attempt == max_retries:
                    logger.error(f"Request failed after {max_retries} retries: {str(e)}")
                    raise
                
                # Jittered exponential backoff
                wait_time = base_backoff * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time:.2f}s: {str(e)}")
                await asyncio.sleep(wait_time)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class MarketDataClient(BaseAPIClient):
    """Client for fetching market data from external APIs with concurrent support."""
    
    def __init__(self):
        super().__init__("https://api.marketdata.example.com")
        self.use_mock_data = os.getenv("USE_MOCK_DATA", "true").lower() == "true"
    
    async def get_market_data(self, period: str) -> Dict[str, Any]:
        """Fetch comprehensive market data for the specified period."""
        logger.info(f"Fetching market data for {period}")
        
        if self.use_mock_data:
            return await self._get_mock_market_data(period)
        
        try:
            # Use TaskGroup for concurrent fetching
            async with asyncio.TaskGroup() as tg:
                sp500_task = tg.create_task(self.get_sp500_tr(period))
                ust10y_task = tg.create_task(self.get_ust10y(period))
                dxy_task = tg.create_task(self.get_dxy(period))
                vix_task = tg.create_task(self.get_vix_peak(period))
            
            # Combine results
            return {
                "period": period,
                "sp500_tr": sp500_task.result(),
                "ust10y_yield": ust10y_task.result(),
                "dxy_chg": dxy_task.result(),
                "vix_peak": vix_task.result(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            # Fallback to mock data on error
            return await self._get_mock_market_data(period)
    
    async def get_sp500_tr(self, period: str) -> float:
        """Fetch S&P 500 Total Return for the period."""
        if self.use_mock_data:
            return 12.3
            
        try:
            response = await self._request_with_retry(
                "GET", 
                f"/sp500/total-return/{period}",
                per_call_timeout=8.0
            )
            data = response.json()
            return float(data.get("total_return", 0.0))
        except Exception as e:
            logger.error(f"Error fetching S&P 500 TR: {str(e)}")
            return 12.3  # Mock fallback
    
    async def get_ust10y(self, period: str) -> float:
        """Fetch 10-Year US Treasury yield for the period."""
        if self.use_mock_data:
            return 4.25
            
        try:
            response = await self._request_with_retry(
                "GET", 
                f"/treasury/10y/{period}",
                per_call_timeout=8.0
            )
            data = response.json()
            return float(data.get("yield", 0.0))
        except Exception as e:
            logger.error(f"Error fetching UST 10Y: {str(e)}")
            return 4.25  # Mock fallback
    
    async def get_dxy(self, period: str) -> float:
        """Fetch DXY (Dollar Index) change for the period."""
        if self.use_mock_data:
            return -2.1
            
        try:
            response = await self._request_with_retry(
                "GET", 
                f"/currency/dxy/change/{period}",
                per_call_timeout=8.0
            )
            data = response.json()
            return float(data.get("change_percent", 0.0))
        except Exception as e:
            logger.error(f"Error fetching DXY: {str(e)}")
            return -2.1  # Mock fallback
    
    async def get_vix_peak(self, period: str) -> float:
        """Fetch VIX peak value for the period."""
        if self.use_mock_data:
            return 28.7
            
        try:
            response = await self._request_with_retry(
                "GET", 
                f"/volatility/vix/peak/{period}",
                per_call_timeout=8.0
            )
            data = response.json()
            return float(data.get("peak_value", 0.0))
        except Exception as e:
            logger.error(f"Error fetching VIX peak: {str(e)}")
            return 28.7  # Mock fallback
    
    async def _get_mock_market_data(self, period: str) -> Dict[str, Any]:
        """Generate consistent mock data for testing."""
        # Add small delay to simulate network call
        await asyncio.sleep(0.1)
        
        return {
            "period": period,
            "sp500_tr": 12.3,
            "ust10y_yield": 4.25,
            "dxy_chg": -2.1,
            "vix_peak": 28.7,
            "market_cap": 45000000000.0,
            "trading_volume": 2500000.0,
            "volatility_index": 18.5,
            "sector_performance": {
                "technology": 12.3,
                "healthcare": 8.7,
                "financials": -2.1
            },
            "timestamp": time.time()
        }
    
    async def get_economic_indicators(self, period: str) -> Dict[str, Any]:
        """Fetch economic indicators for the specified period."""
        logger.info(f"Fetching economic indicators for {period}")
        
        try:
            # Simulate concurrent economic data fetching
            async with asyncio.TaskGroup() as tg:
                gdp_task = tg.create_task(self._fetch_gdp(period))
                inflation_task = tg.create_task(self._fetch_inflation(period))
                unemployment_task = tg.create_task(self._fetch_unemployment(period))
                rates_task = tg.create_task(self._fetch_interest_rates(period))
            
            return {
                "period": period,
                "gdp_growth": gdp_task.result(),
                "inflation_rate": inflation_task.result(),
                "unemployment_rate": unemployment_task.result(),
                "interest_rate": rates_task.result(),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {str(e)}")
            # Mock fallback
            return {
                "period": period,
                "gdp_growth": 2.4,
                "inflation_rate": 3.2,
                "unemployment_rate": 4.1,
                "interest_rate": 5.25,
                "timestamp": time.time()
            }
    
    async def _fetch_gdp(self, period: str) -> float:
        """Fetch GDP growth rate."""
        if self.use_mock_data:
            await asyncio.sleep(0.05)  # Simulate network delay
            return 2.4
            
        try:
            response = await self._request_with_retry(
                "GET", 
                f"/economic/gdp/{period}",
                per_call_timeout=6.0
            )
            data = response.json()
            return float(data.get("growth_rate", 0.0))
        except Exception:
            return 2.4
    
    async def _fetch_inflation(self, period: str) -> float:
        """Fetch inflation rate."""
        if self.use_mock_data:
            await asyncio.sleep(0.05)
            return 3.2
            
        try:
            response = await self._request_with_retry(
                "GET", 
                f"/economic/inflation/{period}",
                per_call_timeout=6.0
            )
            data = response.json()
            return float(data.get("rate", 0.0))
        except Exception:
            return 3.2
    
    async def _fetch_unemployment(self, period: str) -> float:
        """Fetch unemployment rate."""
        if self.use_mock_data:
            await asyncio.sleep(0.05)
            return 4.1
            
        try:
            response = await self._request_with_retry(
                "GET", 
                f"/economic/unemployment/{period}",
                per_call_timeout=6.0
            )
            data = response.json()
            return float(data.get("rate", 0.0))
        except Exception:
            return 4.1
    
    async def _fetch_interest_rates(self, period: str) -> float:
        """Fetch federal funds rate."""
        if self.use_mock_data:
            await asyncio.sleep(0.05)
            return 5.25
            
        try:
            response = await self._request_with_retry(
                "GET", 
                f"/economic/fed-funds/{period}",
                per_call_timeout=6.0
            )
            data = response.json()
            return float(data.get("rate", 0.0))
        except Exception:
            return 5.25


class LLMClient(BaseAPIClient):
    """Client for LLM API calls with deterministic settings."""
    
    def __init__(self):
        super().__init__("https://api.openai.com/v1")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    async def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0
    ) -> str:
        """Generate text using OpenAI API."""
        logger.info("Generating text using OpenAI GPT-4")
        
        return await self._generate_openai(system_prompt, user_prompt, temperature)
    
    async def get_embeddings(self, texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI's embedding API with smart batching."""
        logger.info(f"Getting embeddings for {len(texts)} texts using {model} with batch size {batch_size}")
        
        return await self._get_openai_embeddings_batched(texts, model, batch_size)
    
    async def _generate_openai(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Generate using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        response = await self._request_with_retry(
            "POST", 
            "/chat/completions", 
            headers=headers, 
            json=payload,
            per_call_timeout=60.0  # Increase timeout for complex prompts
        )
        
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        
        logger.info(f"Successfully generated {len(content)} characters of text")
        return content
    
    
    async def _get_openai_embeddings_batched(self, texts: List[str], model: str, batch_size: int) -> List[List[float]]:
        """Get embeddings from OpenAI API with smart batching and concurrent processing."""
        import asyncio
        
        if len(texts) <= batch_size:
            # Single batch - process directly
            return await self._get_openai_embeddings_single_batch(texts, model)
        
        # Multiple batches - process concurrently
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        logger.info(f"Processing {len(texts)} texts in {len(batches)} concurrent batches")
        
        # Process all batches concurrently
        batch_tasks = [
            self._get_openai_embeddings_single_batch(batch, model) 
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results from all batches
        all_embeddings = []
        for batch_embeddings in batch_results:
            all_embeddings.extend(batch_embeddings)
        
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings across {len(batches)} batches")
        return all_embeddings
    
    async def _get_openai_embeddings_single_batch(self, texts: List[str], model: str) -> List[List[float]]:
        """Get embeddings for a single batch from OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # OpenAI embeddings API batch processing
        payload = {
            "model": model,
            "input": texts,
            "encoding_format": "float"
        }
        
        response = await self._request_with_retry(
            "POST", 
            "/embeddings", 
            headers=headers, 
            json=payload
        )
        
        response_data = response.json()
        embeddings = [item["embedding"] for item in response_data["data"]]
        
        logger.debug(f"Generated {len(embeddings)} embeddings for batch of {len(texts)} texts")
        return embeddings
    