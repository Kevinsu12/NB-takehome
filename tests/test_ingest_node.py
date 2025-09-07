#!/usr/bin/env python3
"""
Test suite for the ingest node.

This tests the ingest_node function which:
1. Fetches market data from mock API
2. Fetches economic data from mock API  
3. Normalizes the data into a standard format
4. Extracts key themes from documents
5. Returns processed_data with all the information

Since we're using mock APIs, we know exactly what data to expect.
"""

import asyncio
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, AsyncMock

# Add the project root to Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.nodes.ingest import ingest_node, normalize_market_data, extract_key_themes
from app.clients.api_clients import MarketDataClient


class TestIngestNode:
    """Test the ingest node functionality."""
    
    def get_sample_documents(self):
        """Sample documents for testing."""
        return [
            "Market volatility increased significantly during Q3 2024 due to geopolitical tensions and inflation concerns.",
            "Technology sector outperformed with strong earnings growth, while financials faced headwinds from rising interest rates."
        ]
    
    def get_sample_state(self):
        """Sample state for testing."""
        return {
            "period": "2024-Q3",
            "documents": self.get_sample_documents()
        }
    
    def get_temp_snapshot_dir(self):
        """Create temporary snapshot directory for testing."""
        temp_dir = tempfile.mkdtemp()
        snapshot_dir = Path(temp_dir) / "snapshots"
        snapshot_dir.mkdir()
        return str(snapshot_dir)
    
    @pytest.mark.asyncio
    async def test_ingest_node_with_mock_data(self):
        """Test ingest node with mock API data - verify exact values."""
        print("\n=== Testing Ingest Node with Mock Data ===")
        
        # Create state
        state = self.get_sample_state()
        print(f"Period: {state['period']}")
        print(f"Documents: {len(state['documents'])}")
        
        # Test ingest node
        result = await ingest_node(state)
        
        # Verify success
        assert "processed_data" in result, "Result should contain 'processed_data'"
        processed_data = result["processed_data"]
        
        print(f"\n=== Processed Data Analysis ===")
        print(f"Processed data keys: {list(processed_data.keys())}")
        
        # Verify period
        assert processed_data["period"] == "2024-Q3", f"Period should be '2024-Q3', got '{processed_data['period']}'"
        print(f"Period: {processed_data['period']}")
        
        # Verify document count
        assert processed_data["document_count"] == 2, f"Document count should be 2, got {processed_data['document_count']}"
        print(f"Document count: {processed_data['document_count']}")
        
        # Verify market data structure
        assert "market_data" in processed_data, "Should have market_data"
        market_data = processed_data["market_data"]
        print(f"Market data keys: {list(market_data.keys())}")
        
        # Verify specific mock values (we know these from the mock API)
        expected_values = {
            "sp500_tr": 12.3,
            "ust10y_yield": 4.25,
            "gdp_growth": 2.4,
            "inflation_rate": 3.2,
            "unemployment_rate": 4.1,
            "interest_rate": 5.25
        }
        
        print(f"\n=== Mock Data Verification ===")
        for key, expected_value in expected_values.items():
            assert key in market_data, f"Missing key: {key}"
            actual_value = market_data[key]
            assert actual_value == expected_value, f"{key}: expected {expected_value}, got {actual_value}"
            print(f"{key}: {actual_value}")
        
        # Verify raw market data is preserved
        assert "raw_market_data" in processed_data, "Should have raw_market_data"
        raw_market = processed_data["raw_market_data"]
        assert raw_market["sp500_tr"] == 12.3, "Raw market data should match"
        print(f"Raw market data preserved: {raw_market['sp500_tr']}")
        
        # Verify key themes
        assert "key_themes" in processed_data, "Should have key_themes"
        key_themes = processed_data["key_themes"]
        assert isinstance(key_themes, list), "Key themes should be a list"
        assert len(key_themes) > 0, "Should have some key themes"
        print(f"Detected themes: {key_themes}")
        
        # Check for expected themes based on sample documents
        expected_themes = ["volatility", "technology", "inflation", "interest rates"]
        found_themes = [theme for theme in key_themes if any(exp in theme.lower() for exp in expected_themes)]
        assert len(found_themes) > 0, f"Should find some expected themes, got: {key_themes}"
        print(f"Found expected themes: {found_themes}")
        
        # Verify processing timestamp
        assert "processing_timestamp" in processed_data, "Should have processing_timestamp"
        timestamp = processed_data["processing_timestamp"]
        assert isinstance(timestamp, (str, float)), "Timestamp should be string or float"
        print(f"Processing timestamp: {timestamp}")
        
        print(f"\n=== Success Metrics ===")
        print(f"All mock data values match expected values")
        print(f"Market data structure is correct")
        print(f"Key themes extracted successfully")
        print(f"Processing timestamp added")
        print("Ingest node with mock data test passed!")
    
    @pytest.mark.asyncio
    async def test_ingest_node_with_snapshot(self):
        """Test ingest node with existing snapshot data."""
        print("\n=== Testing Ingest Node with Snapshot ===")
        
        # Create temporary snapshot directory
        snapshot_dir = self.get_temp_snapshot_dir()
        
        try:
            # Create snapshot data
            snapshot_data = {
                "market_data": {
                    "sp500_tr": 15.0,
                    "ust10y_yield": 4.5,
                    "gdp_growth": 3.0,
                    "inflation_rate": 2.8,
                    "unemployment_rate": 3.8,
                    "interest_rate": 5.5
                },
                "economic_data": {
                    "gdp_growth": 3.0,
                    "inflation_rate": 2.8,
                    "unemployment_rate": 3.8
                },
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            # Save snapshot
            snapshot_path = Path(snapshot_dir) / "2024-Q3.json"
            import json
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot_data, f)
            
            print(f"Created snapshot at: {snapshot_path}")
            
            # Mock the snapshot loading
            with patch('app.nodes.ingest.load_snapshot') as mock_load, \
                 patch('app.nodes.ingest.save_snapshot') as mock_save:
                
                mock_load.return_value = snapshot_data
                
                # Create state
                state = self.get_sample_state()
                
                # Test ingest node
                result = await ingest_node(state)
                
                # Verify snapshot was used
                mock_load.assert_called_once_with("2024-Q3")
                print(f"Snapshot data used correctly:")
                
                # Verify the snapshot data was used
                processed_data = result["processed_data"]
                market_data = processed_data["market_data"]
                
                # Check that snapshot values were used
                assert market_data["sp500_tr"] == 15.0, "Should use snapshot value"
                assert market_data["ust10y_yield"] == 4.5, "Should use snapshot value"
                print(f"Snapshot S&P 500 TR: {market_data['sp500_tr']}")
                print(f"Snapshot 10Y Yield: {market_data['ust10y_yield']}")
                
                print("Ingest node with snapshot test passed!")
        
        finally:
            # Cleanup
            shutil.rmtree(snapshot_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_ingest_node_error_handling(self):
        """Test ingest node error handling."""
        print("\n=== Testing Ingest Node Error Handling ===")
        
        # Test with missing documents
        invalid_state = {"period": "2024-Q3"}
        
        try:
            result = await ingest_node(invalid_state)
            assert False, "Should have raised KeyError for missing documents"
        except KeyError as e:
            assert "documents" in str(e), f"Expected KeyError for documents, got: {e}"
            print(f"Error handling working: KeyError for missing 'documents': {e}")
        
        # Test with empty documents
        empty_state = {
            "period": "2024-Q3",
            "documents": []
        }
        
        result = await ingest_node(empty_state)
        assert "processed_data" in result, "Should handle empty documents gracefully"
        assert result["processed_data"]["document_count"] == 0, "Should count empty documents correctly"
        print(f"Empty documents handled gracefully: {result['processed_data']['document_count']} documents")
        
        print("Ingest node error handling test passed!")
    
    @pytest.mark.asyncio
    async def test_normalize_market_data(self):
        """Test the normalize_market_data function."""
        print("\n=== Testing Normalize Market Data Function ===")
        
        # Test data
        raw_data = {
            "sp500_tr": 12.3,
            "ust10y_yield": 4.25
        }
        
        economic_data = {
            "gdp_growth": 2.4,
            "inflation_rate": 3.2,
            "unemployment_rate": 4.1,
            "interest_rate": 5.25
        }
        
        period = "2024-Q3"
        
        # Test normalization
        normalized = normalize_market_data(period, raw_data, economic_data)
        
        # Verify structure
        print(f"Normalized data keys: {list(normalized.keys())}")
        assert "period" in normalized, "Should have period"
        assert normalized["period"] == period, f"Period should be {period}"
        
        # Verify all numeric fields are present
        market_fields = ["sp500_tr", "ust10y_yield"]
        economic_fields = ["gdp_growth", "inflation_rate", "unemployment_rate", "interest_rate"]
        
        for field in market_fields:
            assert field in normalized, f"Missing field: {field}"
            assert isinstance(normalized[field], (int, float)), f"{field} should be numeric"
            assert normalized[field] == raw_data[field], f"{field} value should match"
        
        for field in economic_fields:
            assert field in normalized, f"Missing field: {field}"
            assert isinstance(normalized[field], (int, float)), f"{field} should be numeric"
            assert normalized[field] == economic_data[field], f"{field} value should match"
        
        print(f"All expected fields present: {len(normalized)} fields")
        print(f"Period correctly set: {normalized['period']}")
        print(f"Market data values match: {all(normalized[k] == raw_data[k] for k in market_fields)}")
        print(f"Economic data values match: {all(normalized[k] == economic_data[k] for k in economic_fields)}")
        
        print("Normalize market data test passed!")
    
    @pytest.mark.asyncio
    async def test_extract_key_themes(self):
        """Test the extract_key_themes function."""
        print("\n=== Testing Extract Key Themes Function ===")
        
        # Test documents
        documents = [
            "Market volatility increased significantly during Q3 2024 due to geopolitical tensions and inflation concerns.",
            "Technology sector outperformed with strong earnings growth, while financials faced headwinds from rising interest rates.",
            "Federal Reserve maintained cautious stance on monetary policy amid economic uncertainty."
        ]
        
        # Test theme extraction
        themes = extract_key_themes(documents)
        
        # Verify results
        print(f"Extracted themes: {themes}")
        assert isinstance(themes, list), "Themes should be a list"
        assert len(themes) > 0, "Should extract some themes"
        
        # Check for expected themes
        expected_keywords = ["volatility", "technology", "inflation", "interest", "federal", "monetary"]
        found_keywords = [kw for kw in expected_keywords if any(kw in theme.lower() for theme in themes)]
        assert len(found_keywords) > 0, f"Should find some expected keywords, got: {themes}"
        
        print(f"Detected {len(themes)} themes")
        print(f"Found keywords: {found_keywords}")
        
        print("Extract key themes test passed!")
    
    @pytest.mark.asyncio
    async def test_ingest_node_deterministic(self):
        """Test that ingest node produces consistent results."""
        print("\n=== Testing Ingest Node Deterministic Behavior ===")
        
        # Create state
        state = self.get_sample_state()
        
        # Run multiple times
        results = []
        for i in range(3):
            print(f"Run {i+1}/3...")
            result = await ingest_node(state)
            results.append(result)
        
        # Verify all results are identical (excluding timestamps)
        print(f"\n=== Deterministic Analysis ===")
        first_result = results[0]["processed_data"]
        
        for i, result in enumerate(results[1:], 1):
            processed_data = result["processed_data"]
            
            # Check deterministic fields
            assert processed_data["period"] == first_result["period"], f"Period differs in run {i+1}"
            assert processed_data["document_count"] == first_result["document_count"], f"Document count differs in run {i+1}"
            
            # Compare market data excluding timestamp fields
            market_data_1 = {k: v for k, v in first_result["market_data"].items() if k != "timestamp"}
            market_data_2 = {k: v for k, v in processed_data["market_data"].items() if k != "timestamp"}
            assert market_data_2 == market_data_1, f"Market data differs in run {i+1}"
            
            assert processed_data["key_themes"] == first_result["key_themes"], f"Key themes differ in run {i+1}"
            
            # Timestamps should be different (exclude from comparison)
            assert "processing_timestamp" in processed_data, f"Missing timestamp in run {i+1}"
        
        print(f"All runs returned identical results (excluding timestamps)")
        print(f"Period: {first_result['period']}")
        print(f"Document count: {first_result['document_count']}")
        print(f"Market data keys: {list(first_result['market_data'].keys())}")
        print(f"Key themes: {first_result['key_themes']}")
        print("Deterministic behavior confirmed - identical results across runs")
        print("Ingest node deterministic test passed!")


if __name__ == "__main__":
    # For direct execution, run all tests
    async def main():
        test_instance = TestIngestNode()
        
        print("INGEST NODE TEST SUITE")
        print("=" * 60)
        
        try:
            await test_instance.test_ingest_node_with_mock_data()
            await test_instance.test_ingest_node_with_snapshot()
            await test_instance.test_ingest_node_error_handling()
            await test_instance.test_normalize_market_data()
            await test_instance.test_extract_key_themes()
            await test_instance.test_ingest_node_deterministic()
            
            print("\n" + "=" * 60)
            print("ALL INGEST NODE TESTS PASSED!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nTEST SUITE FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())