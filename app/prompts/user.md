# Generate Market Context for {period}

## Retrieved Context
{retrieved_context}

## Key Statistics (JSON)
{key_stats_json}

## Instructions

Generate a comprehensive market context report using ONLY the provided facts above. Follow these strict guidelines:

### Data Usage Rules
- Use ONLY numbers from the key statistics JSON provided
- If any required data is missing from the provided sources, state "data not available"
- Do NOT estimate, approximate, or infer missing values
- Every numerical claim in the narrative must be traceable to the key_stats

### Required Output Format
Return a valid JSON object with these exact fields:
- `period`: The time period being analyzed
- `headline`: Concise summary (8-12 words) of the dominant market theme
- `macro_drivers`: Array of 3-5 key factors driving market performance
- `key_stats`: Object containing the numerical data (copy from provided key_stats_json)
- `narrative`: 150-250 word analysis integrating all key statistics
- `sources`: Array of data sources referenced

### Content Requirements
1. **Headline**: Capture the primary market story without forward-looking language
2. **Macro Drivers**: Identify concrete factors affecting markets (no speculation)
3. **Narrative**: 
   - Integrate ALL numbers from key_stats naturally into the text
   - Follow the prescribed topic order: broad market → drivers → rates/FX/credit → earnings
   - Use past/present tense only
   - Maintain objective, factual tone

### Validation Checklist
- [ ] All key_stats numbers appear in the narrative
- [ ] No prohibited phrases ("outlook", "expect", "overweight", etc.)
- [ ] No forward-looking statements
- [ ] JSON format is valid and complete
- [ ] Word count within target range

Generate the market context report now using only the provided data.