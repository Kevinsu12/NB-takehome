#!/usr/bin/env python3
"""
Test suite for the revise node functionality.
"""

import asyncio
import json
import os
import sys
import pytest
from unittest.mock import patch, AsyncMock, mock_open

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.nodes.revise import revise_node
from app.schemas.market_context import MarketContext


class TestReviseNode:
    """Test cases for the revise node."""
    
    def create_validated_context(self):
        """Create a validated MarketContext for testing."""
        return MarketContext(
            period="2024-Q3",
            headline="Markets show strong recovery with technology sector leading",
            macro_drivers=[
                "Technology sector outperformance",
                "Federal Reserve policy stance",
                "Inflation concerns easing"
            ],
            key_stats={
                "sp500_tr": 12.3,
                "ust10y_yield": 4.25,
                "gdp_growth": 2.4,
                "inflation_rate": 3.2,
                "unemployment_rate": 4.1,
                "interest_rate": 5.25
            },
            narrative="The third quarter of 2024 demonstrated robust market performance, with the S&P 500 delivering strong returns of 12.3%. This recovery was largely driven by the technology sector, which benefited from continued innovation and strong earnings. The Federal Reserve maintained a cautious stance on interest rates, keeping them at 5.25% while monitoring inflation trends that showed signs of moderation.",
            sources=[
                "S&P 500 Index Data",
                "Federal Reserve Economic Data",
                "Bureau of Labor Statistics",
                "Bureau of Economic Analysis"
            ]
        )
    
    def create_revised_context_data(self):
        """Create mock revised context data that would come from LLM."""
        return {
            "period": "2024-Q3",
            "headline": "Technology-Driven Market Recovery Leads Q3 2024 Performance",
            "macro_drivers": [
                "Technology sector significantly outperformed broader market",
                "Federal Reserve maintained cautious monetary policy approach",
                "Inflation pressures showed signs of easing across key indicators"
            ],
            "key_stats": {
                "sp500_tr": 12.3,
                "ust10y_yield": 4.25,
                "gdp_growth": 2.4,
                "inflation_rate": 3.2,
                "unemployment_rate": 4.1,
                "interest_rate": 5.25
            },
            "narrative": "The third quarter of 2024 showcased exceptional market resilience, with the S&P 500 Total Return Index achieving a robust 12.3% gain. This performance was predominantly fueled by the technology sector's continued innovation and strong earnings momentum. The Federal Reserve's measured approach to monetary policy, maintaining the federal funds rate at 5.25%, provided market stability while inflation indicators showed encouraging signs of moderation.",
            "sources": [
                "S&P 500 Total Return Index",
                "Federal Reserve Economic Data (FRED)",
                "Bureau of Labor Statistics (BLS)",
                "Bureau of Economic Analysis (BEA)"
            ]
        }
    
    @pytest.mark.asyncio
    @patch("builtins.open", new_callable=mock_open)
    @patch("app.nodes.revise.create_llm_client")
    async def test_revise_node_success(self, mock_create_llm, mock_file):
        """Test successful revision with valid validated context."""
        print("\n=== Testing Revise Node Success ===")
        
        try:
            # Setup mock system prompt
            mock_file.return_value.read.return_value = "System prompt for revision"
            
            # Setup mock LLM client
            mock_llm_client = AsyncMock()
            mock_create_llm.return_value = mock_llm_client
            
            # Create mock revised response
            revised_data = self.create_revised_context_data()
            mock_llm_client.generate.return_value = json.dumps(revised_data)
            
            # Create state with validated context
            validated_context = self.create_validated_context()
            state = {
                "validated_context": validated_context,
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "draft_context": {},
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test revision
            result = await revise_node(state)
            
            # Verify success
            assert result.get("error") is None, f"Expected success, got error: {result.get('error')}"
            assert "final_context" in result, "Missing final_context in result"
            
            # Verify final_context is a MarketContext object
            final_context = result["final_context"]
            assert isinstance(final_context, MarketContext), "Final context should be MarketContext instance"
            
            # Verify revised content
            assert final_context.period == "2024-Q3", "Period should match"
            assert final_context.headline == revised_data["headline"], "Headline should be revised"
            assert final_context.macro_drivers == revised_data["macro_drivers"], "Macro drivers should be revised"
            assert final_context.narrative == revised_data["narrative"], "Narrative should be revised"
            assert final_context.sources == revised_data["sources"], "Sources should be revised"
            
            # Verify LLM was called correctly
            mock_llm_client.generate.assert_called_once()
            call_args = mock_llm_client.generate.call_args
            assert "Focus on clarity and consistency" in call_args[1]["system_prompt"], "System prompt should include revision focus"
            
            print("Revise node success test passed!")
            print(f"Original headline: {validated_context.headline}")
            print(f"Revised headline: {final_context.headline}")
            print(f"Original narrative length: {len(validated_context.narrative)} characters")
            print(f"Revised narrative length: {len(final_context.narrative)} characters")
            
        except Exception as e:
            print(f"Revise node success test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    @patch("builtins.open", new_callable=mock_open)
    @patch("app.nodes.revise.create_llm_client")
    async def test_revise_node_llm_failure(self, mock_create_llm, mock_file):
        """Test revision when LLM call fails - should fallback to validated context."""
        print("\n=== Testing Revise Node LLM Failure ===")
        
        try:
            # Setup mock system prompt
            mock_file.return_value.read.return_value = "System prompt for revision"
            
            # Setup mock LLM client to raise exception
            mock_llm_client = AsyncMock()
            mock_create_llm.return_value = mock_llm_client
            mock_llm_client.generate.side_effect = Exception("LLM API call failed")
            
            # Create state with validated context
            validated_context = self.create_validated_context()
            state = {
                "validated_context": validated_context,
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "draft_context": {},
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test revision
            result = await revise_node(state)
            
            # Verify fallback behavior
            assert result.get("error") is None, f"Expected fallback success, got error: {result.get('error')}"
            assert "final_context" in result, "Missing final_context in result"
            
            # Verify final_context is the original validated_context (fallback)
            final_context = result["final_context"]
            assert isinstance(final_context, MarketContext), "Final context should be MarketContext instance"
            assert final_context == validated_context, "Final context should be original validated context"
            assert final_context.headline == validated_context.headline, "Should use original headline"
            assert final_context.narrative == validated_context.narrative, "Should use original narrative"
            
            print("Revise node LLM failure test passed!")
            print(f"Fallback to original validated context successful")
            print(f"Final headline: {final_context.headline}")
            
        except Exception as e:
            print(f"Revise node LLM failure test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    @patch("builtins.open", new_callable=mock_open)
    @patch("app.nodes.revise.create_llm_client")
    async def test_revise_node_invalid_json_response(self, mock_create_llm, mock_file):
        """Test revision when LLM returns invalid JSON - should fallback to validated context."""
        print("\n=== Testing Revise Node Invalid JSON Response ===")
        
        try:
            # Setup mock system prompt
            mock_file.return_value.read.return_value = "System prompt for revision"
            
            # Setup mock LLM client to return invalid JSON
            mock_llm_client = AsyncMock()
            mock_create_llm.return_value = mock_llm_client
            mock_llm_client.generate.return_value = "This is not valid JSON"
            
            # Create state with validated context
            validated_context = self.create_validated_context()
            state = {
                "validated_context": validated_context,
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "draft_context": {},
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test revision
            result = await revise_node(state)
            
            # Verify fallback behavior
            assert result.get("error") is None, f"Expected fallback success, got error: {result.get('error')}"
            assert "final_context" in result, "Missing final_context in result"
            
            # Verify final_context is the original validated_context (fallback)
            final_context = result["final_context"]
            assert isinstance(final_context, MarketContext), "Final context should be MarketContext instance"
            assert final_context == validated_context, "Final context should be original validated context"
            
            print("Revise node invalid JSON response test passed!")
            print(f"Fallback to original validated context successful")
            
        except Exception as e:
            print(f"Revise node invalid JSON response test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    @patch("builtins.open", new_callable=mock_open)
    @patch("app.nodes.revise.create_llm_client")
    async def test_revise_node_invalid_schema_response(self, mock_create_llm, mock_file):
        """Test revision when LLM returns JSON that doesn't match schema - should fallback to validated context."""
        print("\n=== Testing Revise Node Invalid Schema Response ===")
        
        try:
            # Setup mock system prompt
            mock_file.return_value.read.return_value = "System prompt for revision"
            
            # Setup mock LLM client to return invalid schema JSON
            mock_llm_client = AsyncMock()
            mock_create_llm.return_value = mock_llm_client
            invalid_schema_data = {
                "period": "2024-Q3",
                "headline": "Revised headline",
                # Missing required fields: macro_drivers, key_stats, narrative, sources
            }
            mock_llm_client.generate.return_value = json.dumps(invalid_schema_data)
            
            # Create state with validated context
            validated_context = self.create_validated_context()
            state = {
                "validated_context": validated_context,
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "draft_context": {},
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test revision
            result = await revise_node(state)
            
            # Verify fallback behavior
            assert result.get("error") is None, f"Expected fallback success, got error: {result.get('error')}"
            assert "final_context" in result, "Missing final_context in result"
            
            # Verify final_context is the original validated_context (fallback)
            final_context = result["final_context"]
            assert isinstance(final_context, MarketContext), "Final context should be MarketContext instance"
            assert final_context == validated_context, "Final context should be original validated context"
            
            print("Revise node invalid schema response test passed!")
            print(f"Fallback to original validated context successful")
            
        except Exception as e:
            print(f"Revise node invalid schema response test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    @patch("builtins.open", new_callable=mock_open)
    @patch("app.nodes.revise.create_llm_client")
    async def test_revise_node_file_read_error(self, mock_create_llm, mock_file):
        """Test revision when system prompt file cannot be read - should fallback to validated context."""
        print("\n=== Testing Revise Node File Read Error ===")
        
        try:
            # Setup mock file to raise exception
            mock_file.side_effect = FileNotFoundError("System prompt file not found")
            
            # Setup mock LLM client
            mock_llm_client = AsyncMock()
            mock_create_llm.return_value = mock_llm_client
            
            # Create state with validated context
            validated_context = self.create_validated_context()
            state = {
                "validated_context": validated_context,
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "draft_context": {},
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test revision
            result = await revise_node(state)
            
            # Verify fallback behavior
            assert result.get("error") is None, f"Expected fallback success, got error: {result.get('error')}"
            assert "final_context" in result, "Missing final_context in result"
            
            # Verify final_context is the original validated_context (fallback)
            final_context = result["final_context"]
            assert isinstance(final_context, MarketContext), "Final context should be MarketContext instance"
            assert final_context == validated_context, "Final context should be original validated context"
            
            print("Revise node file read error test passed!")
            print(f"Fallback to original validated context successful")
            
        except Exception as e:
            print(f"Revise node file read error test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    async def test_revise_node_missing_validated_context(self):
        """Test revision when validated_context is missing from state."""
        print("\n=== Testing Revise Node Missing Validated Context ===")
        
        try:
            # Create state without validated_context
            state = {
                "period": "2024-Q3",
                "documents": [],
                "processed_data": {},
                "draft_context": {},
                "formatted_context": "",
                "vectorstore": None,
                "error": None
            }
            
            # Test revision - should raise KeyError
            try:
                result = await revise_node(state)
                assert False, "Expected KeyError for missing validated_context"
            except KeyError as e:
                assert "validated_context" in str(e), f"Expected KeyError for validated_context, got: {e}"
                print("Revise node missing validated context test passed!")
                print(f"Correctly raised KeyError for missing validated_context: {e}")
            
        except Exception as e:
            print(f"Revise node missing validated context test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    @patch("builtins.open", new_callable=mock_open)
    @patch("app.nodes.revise.create_llm_client")
    async def test_revise_node_state_preservation(self, mock_create_llm, mock_file):
        """Test that revise node preserves other state fields."""
        print("\n=== Testing Revise Node State Preservation ===")
        
        try:
            # Setup mock system prompt
            mock_file.return_value.read.return_value = "System prompt for revision"
            
            # Setup mock LLM client
            mock_llm_client = AsyncMock()
            mock_create_llm.return_value = mock_llm_client
            
            # Create mock revised response
            revised_data = self.create_revised_context_data()
            mock_llm_client.generate.return_value = json.dumps(revised_data)
            
            # Create state with additional fields
            validated_context = self.create_validated_context()
            original_state = {
                "validated_context": validated_context,
                "period": "2024-Q3",
                "documents": ["doc1", "doc2"],
                "processed_data": {"key": "value"},
                "draft_context": {"draft": "data"},
                "formatted_context": "",
                "vectorstore": "some_vectorstore",
                "error": None,
                "custom_field": "custom_value"
            }
            
            # Test revision
            result = await revise_node(original_state)
            
            # Verify success
            assert result.get("error") is None, f"Expected success, got error: {result.get('error')}"
            
            # Verify original state fields are preserved
            assert result["period"] == original_state["period"], "Period should be preserved"
            assert result["documents"] == original_state["documents"], "Documents should be preserved"
            assert result["processed_data"] == original_state["processed_data"], "Processed data should be preserved"
            assert result["draft_context"] == original_state["draft_context"], "Draft context should be preserved"
            assert result["vectorstore"] == original_state["vectorstore"], "Vectorstore should be preserved"
            assert result["custom_field"] == original_state["custom_field"], "Custom field should be preserved"
            
            # Verify new field is added
            assert "final_context" in result, "Final context should be added"
            
            print("Revise node state preservation test passed!")
            print(f"All original state fields preserved")
            print(f"Custom field preserved: {result['custom_field']}")
            
        except Exception as e:
            print(f"Revise node state preservation test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # For direct execution, run all tests
    async def run_all_tests():
        test_instance = TestReviseNode()
        
        print("REVISE NODE TEST SUITE")
        print("=" * 60)
        
        try:
            await test_instance.test_revise_node_success()
            await test_instance.test_revise_node_llm_failure()
            await test_instance.test_revise_node_invalid_json_response()
            await test_instance.test_revise_node_invalid_schema_response()
            await test_instance.test_revise_node_file_read_error()
            await test_instance.test_revise_node_missing_validated_context()
            await test_instance.test_revise_node_state_preservation()
            
            print("\n" + "=" * 60)
            print("ALL REVISE NODE TESTS PASSED!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nTEST SUITE FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_all_tests())