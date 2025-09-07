#!/usr/bin/env python3
"""
Test suite for the validate node functionality.
"""

import asyncio
import json
import os
import sys
import pytest
from unittest.mock import patch, AsyncMock

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.nodes.validate import validate_node
from app.schemas.market_context import MarketContext


class TestValidateNode:
    """Test cases for the validate node."""
    
    def create_valid_draft_context(self):
        """Create a valid draft context for testing."""
        return {
            "period": "2024-Q3",
            "headline": "Markets show strong recovery with technology sector leading",
            "macro_drivers": [
                "Technology sector outperformance",
                "Federal Reserve policy stance",
                "Inflation concerns easing"
            ],
            "key_stats": {
                "sp500_tr": 12.3,
                "ust10y_yield": 4.25,
                "gdp_growth": 2.4,
                "inflation_rate": 3.2,
                "unemployment_rate": 4.1,
                "interest_rate": 5.25
            },
            "narrative": "The third quarter of 2024 demonstrated robust market performance, with the S&P 500 delivering strong returns of 12.3%. This recovery was largely driven by the technology sector, which benefited from continued innovation and strong earnings. The Federal Reserve maintained a cautious stance on interest rates, keeping them at 5.25% while monitoring inflation trends that showed signs of moderation.",
            "sources": [
                "S&P 500 Index Data",
                "Federal Reserve Economic Data",
                "Bureau of Labor Statistics",
                "Bureau of Economic Analysis"
            ]
        }
    
    def create_invalid_draft_context(self):
        """Create an invalid draft context for testing."""
        return {
            "period": "2024-Q3",
            "headline": "Markets show strong recovery",
            # Missing required fields: macro_drivers, key_stats, narrative, sources
        }
    
    def create_partially_invalid_draft_context(self):
        """Create a draft context with some invalid field types."""
        return {
            "period": "2024-Q3",
            "headline": "Markets show strong recovery with technology sector leading",
            "macro_drivers": "This should be a list, not a string",  # Wrong type
            "key_stats": {
                "sp500_tr": 12.3,
                "ust10y_yield": 4.25
            },
            "narrative": "The third quarter of 2024 demonstrated robust market performance.",
            "sources": [
                "S&P 500 Index Data",
                "Federal Reserve Economic Data"
            ]
        }
    
    @pytest.mark.asyncio
    async def test_validate_node_success(self):
        """Test successful validation with valid draft context."""
        print("\n=== Testing Validate Node Success ===")
        
        try:
            # Create valid state
            valid_draft = self.create_valid_draft_context()
            state = {
                "draft_context": valid_draft,
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "validated_context": None,
                "final_context": None,
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test validation
            result = await validate_node(state)
            
            # Verify success
            assert result.get("error") is None, f"Expected success, got error: {result.get('error')}"
            assert "validated_context" in result, "Missing validated_context in result"
            assert "final_context" in result, "Missing final_context in result"
            
            # Verify validated context is a MarketContext object
            validated_context = result["validated_context"]
            assert isinstance(validated_context, MarketContext), "Validated context should be MarketContext instance"
            
            # Verify all fields are present and correct
            assert validated_context.period == "2024-Q3", "Period mismatch"
            assert validated_context.headline == valid_draft["headline"], "Headline mismatch"
            assert validated_context.macro_drivers == valid_draft["macro_drivers"], "Macro drivers mismatch"
            assert validated_context.key_stats == valid_draft["key_stats"], "Key stats mismatch"
            assert validated_context.narrative == valid_draft["narrative"], "Narrative mismatch"
            assert validated_context.sources == valid_draft["sources"], "Sources mismatch"
            
            # Verify final_context is the same as validated_context
            assert result["final_context"] == validated_context, "Final context should match validated context"
            
            print("Validate node success test passed!")
            print(f"Validated period: {validated_context.period}")
            print(f"Validated headline: {validated_context.headline}")
            print(f"Validated macro drivers: {len(validated_context.macro_drivers)} items")
            print(f"Validated key stats: {len(validated_context.key_stats)} metrics")
            print(f"Validated narrative length: {len(validated_context.narrative)} characters")
            print(f"Validated sources: {len(validated_context.sources)} sources")
            
        except Exception as e:
            print(f"Validate node success test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    async def test_validate_node_missing_fields(self):
        """Test validation failure with missing required fields."""
        print("\n=== Testing Validate Node Missing Fields ===")
        
        try:
            # Create invalid state with missing fields
            invalid_draft = self.create_invalid_draft_context()
            state = {
                "draft_context": invalid_draft,
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "validated_context": None,
                "final_context": None,
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test validation
            result = await validate_node(state)
            
            # Verify failure
            assert "error" in result, "Expected error for invalid draft context"
            assert result.get("error") is not None, "Error should not be None"
            assert "Schema validation failed" in result["error"], "Should have schema validation error"
            
            # Verify no validated_context was created
            assert "validated_context" not in result or result.get("validated_context") is None, "Should not have validated context"
            assert "final_context" not in result or result.get("final_context") is None, "Should not have final context"
            
            print("Validate node missing fields test passed!")
            print(f"Error message: {result['error']}")
            
        except Exception as e:
            print(f"Validate node missing fields test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    async def test_validate_node_wrong_types(self):
        """Test validation failure with wrong field types."""
        print("\n=== Testing Validate Node Wrong Types ===")
        
        try:
            # Create state with wrong field types
            invalid_draft = self.create_partially_invalid_draft_context()
            state = {
                "draft_context": invalid_draft,
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "validated_context": None,
                "final_context": None,
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test validation
            result = await validate_node(state)
            
            # Verify failure
            assert "error" in result, "Expected error for invalid field types"
            assert result.get("error") is not None, "Error should not be None"
            assert "Schema validation failed" in result["error"], "Should have schema validation error"
            
            print("Validate node wrong types test passed!")
            print(f"Error message: {result['error']}")
            
        except Exception as e:
            print(f"Validate node wrong types test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    async def test_validate_node_empty_draft_context(self):
        """Test validation with empty draft context."""
        print("\n=== Testing Validate Node Empty Draft Context ===")
        
        try:
            # Create state with empty draft context
            state = {
                "draft_context": {},
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "validated_context": None,
                "final_context": None,
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test validation
            result = await validate_node(state)
            
            # Verify failure
            assert "error" in result, "Expected error for empty draft context"
            assert result.get("error") is not None, "Error should not be None"
            assert "Schema validation failed" in result["error"], "Should have schema validation error"
            
            print("Validate node empty draft context test passed!")
            print(f"Error message: {result['error']}")
            
        except Exception as e:
            print(f"Validate node empty draft context test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    async def test_validate_node_missing_draft_context(self):
        """Test validation with missing draft_context key."""
        print("\n=== Testing Validate Node Missing Draft Context Key ===")
        
        try:
            # Create state without draft_context
            state = {
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "validated_context": None,
                "final_context": None,
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test validation
            result = await validate_node(state)
            
            # Verify failure
            assert "error" in result, "Expected error for missing draft_context"
            assert result.get("error") is not None, "Error should not be None"
            assert "Schema validation failed" in result["error"], "Should have schema validation error"
            
            print("Validate node missing draft context key test passed!")
            print(f"Error message: {result['error']}")
            
        except Exception as e:
            print(f"Validate node missing draft context key test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    async def test_validate_node_edge_cases(self):
        """Test validation with edge cases."""
        print("\n=== Testing Validate Node Edge Cases ===")
        
        try:
            # Test with minimal valid data
            minimal_draft = {
                "period": "2024-Q3",
                "headline": "Market Update",
                "macro_drivers": ["Economic growth"],
                "key_stats": {"sp500_tr": 10.0},
                "narrative": "Market conditions improved during the quarter.",
                "sources": ["Market Data"]
            }
            
            state = {
                "draft_context": minimal_draft,
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "validated_context": None,
                "final_context": None,
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test validation
            result = await validate_node(state)
            
            # Verify success even with minimal data
            assert result.get("error") is None, f"Expected success with minimal data, got error: {result.get('error')}"
            assert "validated_context" in result, "Missing validated_context in result"
            
            validated_context = result["validated_context"]
            assert isinstance(validated_context, MarketContext), "Validated context should be MarketContext instance"
            assert validated_context.period == "2024-Q3", "Period should match"
            assert len(validated_context.macro_drivers) == 1, "Should have one macro driver"
            assert len(validated_context.key_stats) == 1, "Should have one key stat"
            
            print("Validate node edge cases test passed!")
            print(f"Minimal data validated successfully")
            print(f"Macro drivers: {validated_context.macro_drivers}")
            print(f"Key stats: {validated_context.key_stats}")
            
        except Exception as e:
            print(f"Validate node edge cases test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    async def test_validate_node_state_preservation(self):
        """Test that validate node preserves other state fields."""
        print("\n=== Testing Validate Node State Preservation ===")
        
        try:
            # Create state with additional fields
            valid_draft = self.create_valid_draft_context()
            original_state = {
                "draft_context": valid_draft,
                "period": "2024-Q3",
                "documents": ["doc1", "doc2"],
                "processed_data": {"key": "value"},
                "validated_context": None,
                "final_context": None,
                "formatted_context": "",
                "vectorstore": "some_vectorstore",
                "error": None,
                "custom_field": "custom_value"
            }
            
            # Test validation
            result = await validate_node(original_state)
            
            # Verify success
            assert result.get("error") is None, f"Expected success, got error: {result.get('error')}"
            
            # Verify original state fields are preserved
            assert result["period"] == original_state["period"], "Period should be preserved"
            assert result["documents"] == original_state["documents"], "Documents should be preserved"
            assert result["processed_data"] == original_state["processed_data"], "Processed data should be preserved"
            assert result["vectorstore"] == original_state["vectorstore"], "Vectorstore should be preserved"
            assert result["custom_field"] == original_state["custom_field"], "Custom field should be preserved"
            
            # Verify new fields are added
            assert "validated_context" in result, "Validated context should be added"
            assert "final_context" in result, "Final context should be added"
            
            print("Validate node state preservation test passed!")
            print(f"All original state fields preserved")
            print(f"Custom field preserved: {result['custom_field']}")
            
        except Exception as e:
            print(f"Validate node state preservation test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # For direct execution, run all tests
    async def run_all_tests():
        test_instance = TestValidateNode()
        
        print("VALIDATE NODE TEST SUITE")
        print("=" * 60)
        
        try:
            await test_instance.test_validate_node_success()
            await test_instance.test_validate_node_missing_fields()
            await test_instance.test_validate_node_wrong_types()
            await test_instance.test_validate_node_empty_draft_context()
            await test_instance.test_validate_node_missing_draft_context()
            await test_instance.test_validate_node_edge_cases()
            await test_instance.test_validate_node_state_preservation()
            
            print("\n" + "=" * 60)
            print("ALL VALIDATE NODE TESTS PASSED!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nTEST SUITE FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_all_tests())