#!/usr/bin/env python3
"""
Test suite for the draft node functionality.
"""

import asyncio
import json
import os
import sys
import pytest
from datetime import datetime
from unittest.mock import patch, AsyncMock

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.nodes.draft import draft_node
from app.nodes.retrieve import retrieve_node
from app.nodes.ingest import ingest_node
from app.rag.vectorStore import VectorStore
from app.rag.pdfLoader import PDFLoader


class TestDraftNode:
    """Test cases for the draft node."""
    
    async def setup_test_data(self):
        """Set up test data by running retrieve and ingest nodes."""
        print("Setting up test data...")
        
        # Initialize vector store
        vectorstore = VectorStore()
        pdf_loader = PDFLoader()
        
        # Load and index documents
        documents, metadata = await pdf_loader.load_documents_with_metadata("data/pdf")
        if documents:
            await vectorstore.build_index(documents, metadata)
            print(f"Indexed {len(documents)} documents")
        
        # Create initial state
        period = "2024-Q3"
        initial_state = {
            "period": period,
            "documents": [],
            "processed_data": {},
            "draft_context": {},
            "validated_context": None,
            "final_context": None,
            "formatted_context": "",
            "vectorstore": vectorstore,
            "error": None
        }
        
        # Run retrieve node
        print("Running retrieve node...")
        retrieve_result = await retrieve_node(initial_state)
        print(f"Retrieved {len(retrieve_result['documents'])} documents")
        
        # Run ingest node
        print("Running ingest node...")
        ingest_result = await ingest_node(retrieve_result)
        print(f"Processed data keys: {list(ingest_result['processed_data'].keys())}")
        
        return ingest_result
    
    @pytest.mark.asyncio
    async def test_draft_node_success(self):
        """Test successful draft generation with real ingest output."""
        print("\n=== Testing Draft Node Success ===")
        
        try:
            # Set up test data
            state = await self.setup_test_data()
            
            # Test draft node
            result = await draft_node(state)
            
            # Verify success
            assert result.get("error") is None, f"Expected success, got error: {result.get('error')}"
            assert "draft_context" in result, "Draft context not found in result"
            
            # Verify draft context structure
            draft_context = result["draft_context"]
            assert isinstance(draft_context, dict), "Draft context should be a dictionary"
            assert len(draft_context) > 0, "Draft context should not be empty"
            
            # Verify required fields
            required_fields = ["period", "headline", "macro_drivers", "key_stats", "narrative", "sources"]
            for field in required_fields:
                assert field in draft_context, f"Missing required field: {field}"
            
            print("Draft node success test passed!")
            print(f"Period: {draft_context['period']}")
            print(f"Headline: {draft_context['headline']}")
            print(f"Macro drivers: {len(draft_context['macro_drivers'])} items")
            print(f"Key stats: {len(draft_context['key_stats'])} metrics")
            print(f"Narrative length: {len(draft_context['narrative'])} characters")
            print(f"Sources: {len(draft_context['sources'])} sources")
            
        except Exception as e:
            print(f"Draft node success test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @pytest.mark.asyncio
    async def test_draft_node_with_mock_data(self):
        """Test draft node with mock data to avoid API costs."""
        print("\n=== Testing Draft Node with Mock Data ===")
        
        # Create mock state with processed data
        mock_processed_data = {
            "period": "2024-Q3",
            "document_count": 2,
            "market_data": {
                "sp500_tr": 12.3,
                "ust10y_yield": 4.25,
                "gdp_growth": 2.4,
                "inflation_rate": 3.2,
                "unemployment_rate": 4.1,
                "interest_rate": 5.25
            },
            "raw_market": {
                "sp500_tr": 12.3,
                "ust10y_yield": 4.25,
                "gdp_growth": 2.4,
                "inflation_rate": 3.2,
                "unemployment_rate": 4.1,
                "interest_rate": 5.25
            },
            "key_themes": ["volatility", "technology", "inflation"],
            "processing_timestamp": "2024-01-01T00:00:00Z"
        }
        
        state = {
            "period": "2024-Q3",
            "documents": [
                "Market volatility increased during Q3 2024.",
                "Technology sector showed strong performance."
            ],
            "processed_data": mock_processed_data,
            "draft_context": {},
            "validated_context": None,
            "final_context": None,
            "formatted_context": "",
            "vectorstore": None,
            "error": None
        }
        
        # Mock the LLM client to avoid API calls
        with patch('app.nodes.draft.create_llm_client') as mock_create_llm:
            mock_llm = AsyncMock()
            mock_create_llm.return_value = mock_llm
            
            # Mock the LLM response
            mock_response = {
                "period": "2024-Q3",
                "headline": "Q3 2024 Market Analysis: Technology Drives Growth Amid Volatility",
                "macro_drivers": [
                    "Technology sector outperformance",
                    "Market volatility concerns",
                    "Inflation pressure easing"
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
            mock_llm.generate.return_value = json.dumps(mock_response)
            
            # Test draft node
            result = await draft_node(state)
            
            # Verify success
            assert result.get("error") is None, f"Expected success, got error: {result.get('error')}"
            assert "draft_context" in result, "Draft context not found in result"
            
            # Verify draft context matches mock response
            draft_context = result["draft_context"]
            assert draft_context["period"] == "2024-Q3", "Period should match"
            assert draft_context["headline"] == mock_response["headline"], "Headline should match"
            assert draft_context["macro_drivers"] == mock_response["macro_drivers"], "Macro drivers should match"
            assert draft_context["key_stats"] == mock_response["key_stats"], "Key stats should match"
            assert draft_context["narrative"] == mock_response["narrative"], "Narrative should match"
            assert draft_context["sources"] == mock_response["sources"], "Sources should match"
            
            print("Draft node with mock data test passed!")
            print(f"Mock LLM response correctly processed")
            print(f"All fields match expected values")
    
    @pytest.mark.asyncio
    async def test_draft_node_error_handling(self):
        """Test draft node error handling with invalid inputs."""
        print("\n=== Testing Draft Node Error Handling ===")
        
        # Test with missing processed_data
        invalid_state = {
            "period": "2024-Q3",
            "documents": [],
            "draft_context": {},
            "validated_context": None,
            "final_context": None,
            "formatted_context": "",
            "vectorstore": None,
            "error": None
        }
        
        try:
            result = await draft_node(invalid_state)
            assert False, "Expected KeyError for missing processed_data"
        except KeyError as e:
            assert "processed_data" in str(e), f"Expected KeyError for processed_data, got: {e}"
            print(f"Error handling test passed!")
            print(f"Correctly raised KeyError for missing processed_data: {e}")
    
    @pytest.mark.asyncio
    async def test_draft_node_deterministic(self):
        """Test that draft node produces consistent results across multiple runs."""
        print("\n=== Testing Draft Node Deterministic Behavior ===")
        
        # Set up test data
        state = await self.setup_test_data()
        
        # Mock the LLM client for deterministic testing
        with patch('app.nodes.draft.create_llm_client') as mock_create_llm:
            mock_llm = AsyncMock()
            mock_create_llm.return_value = mock_llm
            
            # Create a consistent mock response
            mock_response = {
                "period": "2024-Q3",
                "headline": "Q3 2024 Market Analysis: Technology Drives Growth",
                "macro_drivers": [
                    "Technology sector outperformance",
                    "Market volatility concerns",
                    "Inflation pressure easing"
                ],
                "key_stats": {
                    "sp500_tr": 12.3,
                    "ust10y_yield": 4.25,
                    "gdp_growth": 2.4,
                    "inflation_rate": 3.2,
                    "unemployment_rate": 4.1,
                    "interest_rate": 5.25
                },
                "narrative": "The third quarter of 2024 demonstrated robust market performance.",
                "sources": [
                    "S&P 500 Index Data",
                    "Federal Reserve Economic Data",
                    "Bureau of Labor Statistics"
                ]
            }
            mock_llm.generate.return_value = json.dumps(mock_response)
            
            # Run multiple times
            results = []
            for i in range(3):
                print(f"Run {i+1}/3...")
                result = await draft_node(state)
                results.append(result)
            
            # Verify all runs succeeded
            for i, result in enumerate(results):
                assert result.get("error") is None, f"Run {i+1} failed: {result.get('error')}"
                assert "draft_context" in result, f"Run {i+1} missing draft_context"
            
            # Verify structural consistency (focus on truly deterministic elements)
            first_draft = results[0]["draft_context"]
            for i, result in enumerate(results[1:], 1):
                draft = result["draft_context"]
                
                # Check truly deterministic fields
                assert draft["period"] == first_draft["period"], f"Period differs in run {i+1}"
                
                # Check that key stats values are identical (these should be deterministic)
                for key in first_draft["key_stats"]:
                    assert key in draft["key_stats"], f"Missing key stat {key} in run {i+1}"
                    assert draft["key_stats"][key] == first_draft["key_stats"][key], f"Key stat {key} differs in run {i+1}"
                
                # Check that all runs have reasonable structure
                assert len(draft["macro_drivers"]) >= 2, f"Too few macro drivers in run {i+1}"
                assert len(draft["sources"]) >= 3, f"Too few sources in run {i+1}"
                assert len(draft["key_stats"]) >= 6, f"Too few key stats in run {i+1}"
                assert len(draft["narrative"]) >= 50, f"Narrative too short in run {i+1}"
            
            print("Deterministic behavior test passed!")
            print(f"All runs produced structurally consistent results")
            print(f"Key stats values are identical across runs")
            print(f"All runs have reasonable content length")
    
    @pytest.mark.asyncio
    async def test_draft_node_content_quality(self):
        """Test the quality of generated draft content."""
        print("\n=== Testing Draft Node Content Quality ===")
        
        # Set up test data
        state = await self.setup_test_data()
        
        # Mock the LLM client
        with patch('app.nodes.draft.create_llm_client') as mock_create_llm:
            mock_llm = AsyncMock()
            mock_create_llm.return_value = mock_llm
            
            # Create a high-quality mock response
            mock_response = {
                "period": "2024-Q3",
                "headline": "Q3 2024 Market Analysis: Technology Drives Growth Amid Economic Uncertainty",
                "macro_drivers": [
                    "Technology sector significantly outperformed broader market indices",
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
            mock_llm.generate.return_value = json.dumps(mock_response)
            
            # Test draft node
            result = await draft_node(state)
            
            # Verify success
            assert result.get("error") is None, f"Expected success, got error: {result.get('error')}"
            draft_context = result["draft_context"]
            
            # Content quality checks
            print(f"\n=== Content Quality Analysis ===")
            
            # Headline quality
            headline = draft_context["headline"]
            assert len(headline) > 20, f"Headline too short: {len(headline)} chars"
            assert len(headline) < 100, f"Headline too long: {len(headline)} chars"
            print(f"Headline quality: {len(headline)} chars - good length")
            
            # Macro drivers quality
            macro_drivers = draft_context["macro_drivers"]
            assert len(macro_drivers) >= 2, f"Too few macro drivers: {len(macro_drivers)}"
            for i, driver in enumerate(macro_drivers):
                assert len(driver) > 10, f"Macro driver {i+1} too short: {len(driver)} chars"
                assert len(driver) < 100, f"Macro driver {i+1} too long: {len(driver)} chars"
            print(f"Macro drivers quality: {len(macro_drivers)} drivers, good length")
            
            # Narrative quality
            narrative = draft_context["narrative"]
            assert len(narrative) > 100, f"Narrative too short: {len(narrative)} chars"
            assert len(narrative) < 1000, f"Narrative too long: {len(narrative)} chars"
            print(f"Narrative quality: {len(narrative)} chars - good length")
            
            # Sources quality
            sources = draft_context["sources"]
            assert len(sources) >= 3, f"Too few sources: {len(sources)}"
            for i, source in enumerate(sources):
                assert len(source) > 5, f"Source {i+1} too short: {len(source)} chars"
            print(f"Sources quality: {len(sources)} sources, good length")
            
            # Key stats quality
            key_stats = draft_context["key_stats"]
            assert len(key_stats) >= 6, f"Too few key stats: {len(key_stats)}"
            for key, value in key_stats.items():
                assert isinstance(value, (int, float)), f"Key stat {key} should be numeric"
            print(f"Key stats quality: {len(key_stats)} metrics, all numeric")
            
            print("Content quality test passed!")
            print(f"All content meets quality standards")
            print(f"Headline: {headline}")
            print(f"Macro drivers: {len(macro_drivers)} items")
            print(f"Narrative: {len(narrative)} characters")
            print(f"Sources: {len(sources)} items")


if __name__ == "__main__":
    # For direct execution, run all tests
    async def run_all_tests():
        test_instance = TestDraftNode()
        
        print("DRAFT NODE TEST SUITE")
        print("=" * 60)
        
        try:
            await test_instance.test_draft_node_success()
            await test_instance.test_draft_node_with_mock_data()
            await test_instance.test_draft_node_error_handling()
            await test_instance.test_draft_node_deterministic()
            await test_instance.test_draft_node_content_quality()
            
            print("\n" + "=" * 60)
            print("ALL DRAFT NODE TESTS PASSED!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nTEST SUITE FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_all_tests())