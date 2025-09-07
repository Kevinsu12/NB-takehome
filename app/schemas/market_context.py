from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime


class MarketContext(BaseModel):
    """Schema for market context data with validation."""
    
    period: str = Field(..., description="Time period (e.g., '2025-Q2')")
    headline: str = Field(..., description="Main headline summarizing the market context")
    macro_drivers: List[str] = Field(..., description="List of key macroeconomic drivers")
    key_stats: Dict[str, float] = Field(..., description="Dictionary of key statistics")
    narrative: str = Field(..., description="Detailed narrative explaining the market context")
    sources: List[str] = Field(..., description="List of data sources used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "period": "2025-Q2",
                "headline": "Market resilience amid economic uncertainty",
                "macro_drivers": [
                    "Federal Reserve policy changes",
                    "Geopolitical tensions",
                    "Technology sector growth"
                ],
                "key_stats": {
                    "gdp_growth": 2.4,
                    "inflation_rate": 3.2,
                    "market_cap": 45000000000.0
                },
                "narrative": "The market shows resilience with GDP growth of 2.4 percent and inflation at 3.2 percent...",
                "sources": [
                    "Federal Reserve Economic Data",
                    "S&P Market Intelligence"
                ]
            }
        }

