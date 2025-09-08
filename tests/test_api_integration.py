#!/usr/bin/env python3
"""
Integration tests for the FastAPI application.
These tests verify the actual API behavior with real pipeline calls.
"""

import asyncio
import os
import sys
import pytest
from unittest.mock import patch, AsyncMock

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app


class TestAPIIntegration:
    """Integration tests for the FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_health_endpoint_integration(self, client):
        """Test health endpoint with real pipeline state."""
        response = client.get("/health")
        
        # Should return 200 or 503 depending on pipeline state
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "status" in data
        
        # Check if service field exists (only in healthy responses)
        if response.status_code == 200:
            assert "service" in data
            assert data["service"] == "market-context-generator"
        else:
            # Unhealthy response should have reason
            assert "reason" in data
    
    @pytest.mark.asyncio
    async def test_market_context_endpoint_integration(self, client):
        """Test market context endpoint with mocked pipeline."""
        # Mock the pipeline to avoid actual API calls
        with patch('app.main.pipeline') as mock_pipeline:
            # Mock the run method to return a test response
            mock_pipeline.run = AsyncMock(return_value={
                "formatted_context": "Q3 2024 Market Analysis: Test integration response...",
                "draft_json": {"period": "2024-Q3", "headline": "Test headline"},
                "retrieved_chunks": []
            })
            
            response = client.post("/market-context?period=2024-Q3")
            
            # Should succeed with mocked pipeline
            assert response.status_code == 200
            
            data = response.json()
            assert data["period"] == "2024-Q3"
            assert "Test integration response" in data["formatted_context"]
            assert isinstance(data["formatted_context"], str)
    
    def test_invalid_period_validation_integration(self, client):
        """Test period validation with various invalid inputs."""
        invalid_periods = [
            "2024-Q5",  # Invalid quarter
            "24-Q3",    # Invalid year
            "2024-Q",   # Missing quarter number
            "invalid",  # Completely invalid
            "",         # Empty
        ]
        
        for period in invalid_periods:
            response = client.post(f"/market-context?period={period}")
            assert response.status_code == 400
            
            data = response.json()
            assert "Invalid period format" in data["detail"]
    
    def test_valid_period_formats_integration(self, client):
        """Test valid period formats with mocked pipeline."""
        with patch('app.main.pipeline') as mock_pipeline:
            valid_periods = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1"]
            
            for period in valid_periods:
                mock_pipeline.run = AsyncMock(return_value={
                    "formatted_context": "Valid period test response...",
                    "draft_json": {"period": period, "headline": "Test headline"},
                    "retrieved_chunks": []
                })
                response = client.post(f"/market-context?period={period}")
                assert response.status_code == 200
                
                data = response.json()
                assert data["period"] == period
                assert "Valid period test response" in data["formatted_context"]
    
    def test_api_documentation_access(self, client):
        """Test that API documentation is accessible."""
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        
        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        assert openapi_data["info"]["title"] == "Market Context Generator"
        assert openapi_data["info"]["version"] == "1.0.0"
    
    def test_cors_preflight_request(self, client):
        """Test CORS preflight request handling."""
        response = client.options("/market-context")
        # Should not return 404 (basic CORS support)
        assert response.status_code != 404
    
    def test_error_handling_integration(self, client):
        """Test error handling with various error scenarios."""
        with patch('app.main.pipeline') as mock_pipeline:
            # Test pipeline exception
            mock_pipeline.run = AsyncMock(side_effect=Exception("Test pipeline error"))
            
            response = client.post("/market-context?period=2024-Q3")
            assert response.status_code == 500
            
            data = response.json()
            assert "Internal server error" in data["detail"]
            assert "Test pipeline error" in data["detail"]
    
    def test_concurrent_requests_integration(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        with patch('app.main.pipeline') as mock_pipeline:
            mock_pipeline.run = AsyncMock(return_value={
                "formatted_context": "Concurrent test response...",
                "draft_json": {"period": "2024-Q3", "headline": "Test headline"},
                "retrieved_chunks": []
            })
            
            results = []
            errors = []
            
            def make_request():
                try:
                    response = client.post("/market-context?period=2024-Q3")
                    results.append(response.status_code)
                except Exception as e:
                    errors.append(str(e))
            
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # All requests should succeed
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert all(status == 200 for status in results), f"Not all requests succeeded: {results}"
            assert len(results) == 5
    
    def test_response_model_validation(self, client):
        """Test that response models are properly validated."""
        with patch('app.main.pipeline') as mock_pipeline:
            mock_pipeline.run = AsyncMock(return_value={
                "formatted_context": "Model validation test response...",
                "draft_json": {"period": "2024-Q3", "headline": "Test headline"},
                "retrieved_chunks": []
            })
            
            response = client.post("/market-context?period=2024-Q3")
            assert response.status_code == 200
            
            data = response.json()
            
            # Verify response model structure
            assert "formatted_context" in data
            assert "period" in data
            assert isinstance(data["formatted_context"], str)
            assert isinstance(data["period"], str)
            assert data["period"] == "2024-Q3"
    
    def test_missing_required_parameters(self, client):
        """Test handling of missing required parameters."""
        # Test without period parameter
        response = client.post("/market-context")
        assert response.status_code == 422  # Validation error
        
        data = response.json()
        assert "detail" in data
        assert any("period" in str(error) for error in data["detail"])
    
    def test_query_parameter_validation(self, client):
        """Test query parameter validation."""
        # Test with valid query parameters
        with patch('app.main.pipeline') as mock_pipeline:
            mock_pipeline.run = AsyncMock(return_value={
                "formatted_context": "Query parameter test...",
                "draft_json": {"period": "2024-Q3", "headline": "Test headline"},
                "retrieved_chunks": []
            })
            
            response = client.post("/market-context?period=2024-Q3")
            assert response.status_code == 200
        
        # Test with invalid query parameters
        response = client.post("/market-context?period=invalid&extra=param")
        assert response.status_code == 400  # Invalid period format


if __name__ == "__main__":
    # For direct execution, run all tests
    async def run_all_tests():
        print("API INTEGRATION TEST SUITE")
        print("=" * 60)
        
        # Note: These tests require the FastAPI TestClient
        # Run with: PYTHONPATH=. python -m pytest tests/test_api_integration.py -v
        print("Run with: PYTHONPATH=. python -m pytest tests/test_api_integration.py -v")
    
    asyncio.run(run_all_tests())
