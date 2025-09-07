# Few-Shot Examples for Market Context Generation

## Example 1: Q4 2024 Technology Sector Strength

**Input Data:**
- Period: 2024-Q4
- Key Stats: {"sp500_tr": 11.8, "ust10y_yield": 4.35, "vix_peak": 31.2, "dxy_chg": 1.7}
- Context: Technology companies drove market gains with strong cloud computing revenue. Interest rate environment remained elevated. Dollar strength affected international markets.

**Output:**
{
  "period": "2024-Q4",
  "headline": "Technology sector leadership drives market gains amid rate uncertainty",
  "macro_drivers": [
    "Cloud computing revenue acceleration",
    "Elevated interest rate environment", 
    "Dollar strength pressuring international flows",
    "Corporate earnings concentration in tech"
  ],
  "key_stats": {
    "sp500_tr": 11.8,
    "ust10y_yield": 4.35,
    "vix_peak": 31.2,
    "dxy_chg": 1.7
  },
  "narrative": "The fourth quarter demonstrated concentrated market leadership as technology companies generated 11.8 percent total returns for the S&P 500. Cloud computing revenue acceleration provided fundamental support for the sector's outperformance. The interest rate environment remained restrictive with 10-year Treasury yields at 4.35 percent, creating headwinds for rate-sensitive sectors. Dollar strength of 1.7 percent pressured international capital flows and emerging market assets. Market volatility peaked at 31.2 during periods of monetary policy uncertainty, reflecting investor sensitivity to rate trajectory changes.",
  "sources": [
    "S&P 500 Index Data",
    "Treasury Department Yield Curves",
    "CBOE Volatility Index",
    "Federal Reserve Dollar Index"
  ]
}

## Example 2: Q1 2025 Mixed Signals

**Input Data:**
- Period: 2025-Q1  
- Key Stats: {"sp500_tr": 3.2, "inflation_rate": 3.8, "unemployment_rate": 4.2, "vix_peak": 27.9}
- Context: Markets showed modest gains despite persistent inflation. Labor market remained resilient. Consumer spending patterns indicated selective weakness.

**Output:**
{
  "period": "2025-Q1",
  "headline": "Modest market gains amid persistent inflation and labor resilience",
  "macro_drivers": [
    "Persistent inflation above target levels",
    "Resilient labor market conditions",
    "Selective consumer spending weakness",
    "Mixed corporate earnings results"
  ],
  "key_stats": {
    "sp500_tr": 3.2,
    "inflation_rate": 3.8, 
    "unemployment_rate": 4.2,
    "vix_peak": 27.9
  },
  "narrative": "The first quarter delivered modest equity gains with S&P 500 total returns of 3.2 percent against a backdrop of persistent inflation pressures. Consumer price inflation remained elevated at 3.8 percent, above Federal Reserve target levels and constraining monetary policy flexibility. Labor market resilience continued with unemployment at 4.2 percent, supporting consumer spending despite selective category weakness. Market volatility peaked at 27.9 during earnings season as investors assessed the durability of corporate profit margins amid cost pressures.",
  "sources": [
    "Bureau of Labor Statistics",
    "S&P Dow Jones Indices", 
    "Consumer Price Index Report",
    "CBOE Market Volatility Index"
  ]
}

**Key Success Patterns in Examples:**
- Factual, present/past tense language
- All key statistics integrated into narrative
- Avoidance of prohibited phrases (no "buy", "sell", "recommend", etc.)
- Proper JSON structure with all required fields
- Objective tone without speculation
- Numbers from key_stats appear exactly in narrative
- Headlines capture the essential market story
- Macro drivers explain market forces at work
- Sources provide credible data attribution

---

**Now generate a new market context report following these same patterns for the provided data.**