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
    

