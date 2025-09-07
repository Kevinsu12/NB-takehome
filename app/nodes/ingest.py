from app.clients.api_clients import MarketDataClient
import logging
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def ingest_node(state: dict) -> dict:
    """Process and structure the retrieved documents with snapshot management."""
    documents = state["documents"]
    period = state["period"]
    
    logger.info(f"Ingesting {len(documents)} documents for {period}")
    
    try:
        client = MarketDataClient()
        
        # Check for existing snapshot
        snapshot_data = await load_snapshot(period)
        
        if snapshot_data:
            logger.info(f"Using existing snapshot for {period}")
            market_data = snapshot_data.get("market_data", {})
            economic_data = snapshot_data.get("economic_data", {})
        else:
            # Fetch fresh data concurrently
            logger.info(f"Fetching fresh data for {period}")
            market_data = await client.get_market_data(period)
            economic_data = await client.get_economic_indicators(period)
            
            # Save snapshot
            await save_snapshot(period, market_data, economic_data)
        
        # Normalize data format
        normalized_data = normalize_market_data(period, market_data, economic_data)
        
        # Process documents into structured data
        processed_data = {
            "period": period,
            "document_count": len(documents),
            "documents": documents,
            "market_data": normalized_data,
            "raw_market_data": market_data,
            "raw_economic_data": economic_data,
            "key_themes": extract_key_themes(documents),
            "processing_timestamp": time.time()
        }
        
        logger.info("Document ingestion completed")
        return {**state, "processed_data": processed_data}
        
    except Exception as e:
        logger.error(f"Error in ingest_node: {str(e)}")
        return {**state, "error": str(e)}


def normalize_market_data(period: str, market_data: Dict[str, Any], economic_data: Dict[str, Any]) -> Dict[str, float]:
    """Normalize market and economic data to standard snapshot format."""
    normalized = {
        "period": period,
        "sp500_tr": float(market_data.get("sp500_tr", 0.0)),
        "ust10y_yield": float(market_data.get("ust10y_yield", 0.0)),
        "dxy_chg": float(market_data.get("dxy_chg", 0.0)),
        "vix_peak": float(market_data.get("vix_peak", 0.0)),
        "gdp_growth": float(economic_data.get("gdp_growth", 0.0)),
        "inflation_rate": float(economic_data.get("inflation_rate", 0.0)),
        "unemployment_rate": float(economic_data.get("unemployment_rate", 0.0)),
        "interest_rate": float(economic_data.get("interest_rate", 0.0)),
        "market_cap": float(market_data.get("market_cap", 0.0)),
        "trading_volume": float(market_data.get("trading_volume", 0.0)),
        "volatility_index": float(market_data.get("volatility_index", 0.0)),
        "timestamp": time.time()
    }
    
    # Add sector performance if available
    sector_perf = market_data.get("sector_performance", {})
    for sector, performance in sector_perf.items():
        normalized[f"{sector}_performance"] = float(performance)
    
    return normalized


async def load_snapshot(period: str) -> Dict[str, Any] | None:
    """Load existing snapshot data for the period."""
    snapshot_path = get_snapshot_path(period)
    
    if not snapshot_path.exists():
        return None
    
    try:
        with open(snapshot_path, 'r') as f:
            snapshot_data = json.load(f)
        
        # Check if snapshot is pinned or recent enough
        if is_snapshot_valid(snapshot_data):
            logger.info(f"Loaded valid snapshot from {snapshot_path}")
            return snapshot_data
        else:
            logger.info(f"Snapshot {snapshot_path} is stale, will refresh")
            return None
            
    except Exception as e:
        logger.error(f"Error loading snapshot {snapshot_path}: {str(e)}")
        return None


async def save_snapshot(period: str, market_data: Dict[str, Any], economic_data: Dict[str, Any]) -> None:
    """Save market data snapshot to disk."""
    snapshot_path = get_snapshot_path(period)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if snapshots are pinned (for testing)
    if os.getenv("PIN_SNAPSHOTS", "false").lower() == "true":
        logger.info("Snapshots are pinned, skipping save")
        return
    
    snapshot_data = {
        "period": period,
        "market_data": market_data,
        "economic_data": economic_data,
        "normalized": normalize_market_data(period, market_data, economic_data),
        "timestamp": time.time(),
        "pinned": False
    }
    
    try:
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        logger.info(f"Saved snapshot to {snapshot_path}")
        
    except Exception as e:
        logger.error(f"Error saving snapshot {snapshot_path}: {str(e)}")
        raise


def get_snapshot_path(period: str) -> Path:
    """Get the file path for a period snapshot."""
    return Path("data/snapshot") / f"{period}.json"


def is_snapshot_valid(snapshot_data: Dict[str, Any]) -> bool:
    """Check if snapshot is still valid based on age and pinning."""
    # If pinned, always valid
    if snapshot_data.get("pinned", False):
        return True
    
    # Check age (valid for 1 hour by default)
    max_age = int(os.getenv("SNAPSHOT_MAX_AGE", "3600"))  # 1 hour default
    snapshot_age = time.time() - snapshot_data.get("timestamp", 0)
    
    return snapshot_age < max_age


def extract_key_themes(documents: list[str]) -> list[str]:
    """Extract key themes from documents using simple keyword analysis."""
    theme_keywords = {
        "market_volatility": ["volatility", "volatile", "uncertainty", "fluctuation"],
        "economic_resilience": ["resilience", "resilient", "stable", "recovery"],
        "sector_rotation": ["rotation", "sector", "outperform", "underperform"],
        "geopolitical_factors": ["geopolitical", "trade", "tariff", "sanctions"],
        "monetary_policy": ["fed", "federal reserve", "interest rates", "monetary"],
        "inflation_concerns": ["inflation", "cpi", "price", "deflation"],
        "technology_growth": ["technology", "tech", "ai", "innovation"],
        "consumer_spending": ["consumer", "spending", "retail", "consumption"]
    }
    
    detected_themes = []
    combined_text = " ".join(documents).lower()
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in combined_text for keyword in keywords):
            detected_themes.append(theme.replace("_", " "))
    
    return detected_themes[:5]  # Return top 5 themes