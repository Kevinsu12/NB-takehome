#!/usr/bin/env python3
"""
Test suite for the FastAPI application.
"""

import asyncio
import json
import os
import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import status

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app, pipeline, MarketContextResponse
from app.schemas.market_context import MarketContext


class TestFastAPIApp:
    """Test cases for the FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_pipeline(self):
        """Mock the pipeline for testing."""
        with patch('app.main.pipeline') as mock:
            yield mock
    
    @pytest.fixture
    def mock_initialize(self):
        """Mock pipeline initialization."""
        with patch.object(pipeline, 'initialize', new_callable=AsyncMock) as mock:
            mock.return_value = None
            yield mock
    
    @pytest.fixture
    def mock_run(self):
        """Mock pipeline run method."""
        with patch.object(pipeline, 'run', new_callable=AsyncMock) as mock:
            yield mock
    
    def test_health_check_healthy(self, client, mock_pipeline):
        """Test health check endpoint when service is healthy."""
        # Mock pipeline as initialized
        mock_pipeline.graph = MagicMock()
        mock_pipeline.vectorstore = MagicMock()
        mock_pipeline.vectorstore.is_indexed.return_value = True
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "market-context-generator"
        assert data["vector_store"] == "available"
        assert data["version"] == "1.0.0"
    
    def test_health_check_unhealthy_pipeline_not_initialized(self, client, mock_pipeline):
        """Test health check when pipeline is not initialized."""
        # Mock pipeline as not initialized
        mock_pipeline.graph = None
        
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["reason"] == "Pipeline not initialized"
    
    def test_health_check_unhealthy_vectorstore_not_indexed(self, client, mock_pipeline):
        """Test health check when vectorstore is not indexed."""
        # Mock pipeline as initialized but vectorstore not indexed
        mock_pipeline.graph = MagicMock()
        mock_pipeline.vectorstore = MagicMock()
        mock_pipeline.vectorstore.is_indexed.return_value = False
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["vector_store"] == "not_indexed"
    
    def test_health_check_exception(self, client, mock_pipeline):
        """Test health check when an exception occurs."""
        # Mock pipeline to raise an exception
        mock_pipeline.graph = MagicMock()
        mock_pipeline.vectorstore = MagicMock()
        mock_pipeline.vectorstore.is_indexed.side_effect = Exception("Vector store error")
        
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["reason"] == "Vector store error"
        assert data["service"] == "market-context-generator"
    
    @pytest.mark.asyncio
    async def test_generate_market_context_success(self, client, mock_run):
        """Test successful market context generation."""
        # Mock successful pipeline run
        mock_run.return_value = "Q3 2024 Market Analysis: Technology sector shows strong performance..."
        
        response = client.post("/market-context?period=2024-Q3")
        
        assert response.status_code == 200
        data = response.json()
        assert data["period"] == "2024-Q3"
        assert "Technology sector" in data["formatted_context"]
        assert isinstance(data["formatted_context"], str)
        assert len(data["formatted_context"]) > 0
        
        # Verify pipeline was called with correct period
        mock_run.assert_called_once_with("2024-Q3")
    
    def test_generate_market_context_invalid_period_format(self, client):
        """Test market context generation with invalid period format."""
        # Test various invalid formats
        invalid_periods = [
            "2024-Q5",  # Invalid quarter
            "24-Q3",    # Invalid year format
            "2024-Q",   # Missing quarter number
            "2024Q3",   # Missing dash
            "Q3-2024",  # Wrong order
            "2024-3",   # Wrong format
            "invalid",  # Completely invalid
            "",         # Empty string
        ]
        
        for period in invalid_periods:
            response = client.post(f"/market-context?period={period}")
            
            assert response.status_code == 400
            data = response.json()
            assert "Invalid period format" in data["detail"]
            assert period in data["detail"]
    
    def test_generate_market_context_valid_period_formats(self, client, mock_run):
        """Test market context generation with valid period formats."""
        # Mock successful pipeline run
        mock_run.return_value = "Market analysis for the period..."
        
        valid_periods = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1"]
        
        for period in valid_periods:
            response = client.post(f"/market-context?period={period}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["period"] == period
            assert isinstance(data["formatted_context"], str)
    
    @pytest.mark.asyncio
    async def test_generate_market_context_pipeline_error(self, client, mock_run):
        """Test market context generation when pipeline fails."""
        # Mock pipeline to raise an exception
        mock_run.side_effect = Exception("Pipeline processing failed")
        
        response = client.post("/market-context?period=2024-Q3")
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]
        assert "Pipeline processing failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_generate_market_context_validation_error(self, client, mock_run):
        """Test market context generation with validation error."""
        # Mock pipeline to raise a ValueError (validation error)
        mock_run.side_effect = ValueError("Schema validation failed: Missing required field 'headline'")
        
        response = client.post("/market-context?period=2024-Q3")
        
        assert response.status_code == 400
        data = response.json()
        assert "Schema validation failed" in data["detail"]
        assert "Missing required field 'headline'" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_generate_market_context_business_rule_error(self, client, mock_run):
        """Test market context generation with business rule validation error."""
        # Mock pipeline to raise a ValueError with business rule error
        mock_run.side_effect = ValueError("Business rule validation failed: Period must be current or future")
        
        response = client.post("/market-context?period=2024-Q3")
        
        assert response.status_code == 400
        data = response.json()
        assert "Business rule validation failed" in data["detail"]
        assert "Period must be current or future" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_generate_market_context_debug_mode(self, client, mock_run):
        """Test market context generation with debug mode enabled."""
        # Mock pipeline to raise an exception
        mock_run.side_effect = Exception("Test error for debug mode")
        
        # Set debug mode
        with patch.dict(os.environ, {"DEBUG": "true"}):
            response = client.post("/market-context?period=2024-Q3")
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]
        assert "Test error for debug mode" in data["detail"]
        assert "Traceback:" in data["detail"]  # Should include traceback in debug mode
    
    def test_generate_market_context_missing_period(self, client):
        """Test market context generation without period parameter."""
        response = client.post("/market-context")
        
        assert response.status_code == 422  # Validation error for missing required parameter
        data = response.json()
        assert "detail" in data
        assert any("period" in str(error) for error in data["detail"])
    
    def test_generate_market_context_empty_period(self, client):
        """Test market context generation with empty period."""
        response = client.post("/market-context?period=")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid period format" in data["detail"]
    
    def test_period_validation_function(self):
        """Test the period validation function directly."""
        from app.main import _is_valid_period_format
        
        # Valid periods
        valid_periods = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1", "2030-Q4"]
        for period in valid_periods:
            assert _is_valid_period_format(period), f"Should be valid: {period}"
        
        # Invalid periods
        invalid_periods = [
            "2024-Q5", "2024-Q0", "2024-Q", "2024Q3", "Q3-2024", 
            "24-Q3", "2024-3", "invalid", "", "2024-Q10"
        ]
        for period in invalid_periods:
            assert not _is_valid_period_format(period), f"Should be invalid: {period}"
    
    def test_market_context_response_model(self):
        """Test the MarketContextResponse model."""
        # Test valid response
        response = MarketContextResponse(
            formatted_context="Test market analysis...",
            period="2024-Q3"
        )
        
        assert response.formatted_context == "Test market analysis..."
        assert response.period == "2024-Q3"
        
        # Test serialization
        data = response.model_dump()
        assert data["formatted_context"] == "Test market analysis..."
        assert data["period"] == "2024-Q3"
    
    def test_app_metadata(self, client):
        """Test FastAPI app metadata and documentation."""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        assert openapi_data["info"]["title"] == "Market Context Generator"
        assert openapi_data["info"]["version"] == "1.0.0"
        assert "/market-context" in openapi_data["paths"]
        assert "/health" in openapi_data["paths"]
    
    def test_docs_endpoints(self, client):
        """Test that documentation endpoints are accessible."""
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_startup_event_success(self, mock_initialize):
        """Test successful startup event."""
        # This would be tested by the startup event handler
        # In a real test, you'd need to trigger the startup event
        await mock_initialize()
        mock_initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_startup_event_failure(self, mock_initialize):
        """Test startup event failure."""
        # Mock initialization to raise an exception
        mock_initialize.side_effect = Exception("Initialization failed")
        
        # In a real test, this would be caught by the startup event handler
        with pytest.raises(Exception, match="Initialization failed"):
            await mock_initialize()
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set (if configured)."""
        # This test assumes CORS is configured
        response = client.options("/market-context")
        # The actual CORS behavior depends on FastAPI configuration
        # This is a placeholder for CORS testing if needed
    
    def test_rate_limiting(self, client, mock_run):
        """Test rate limiting if implemented."""
        # This is a placeholder for rate limiting tests
        # Rate limiting would need to be implemented first
        mock_run.return_value = "Test response"
        
        # Make multiple requests
        for i in range(5):
            response = client.post("/market-context?period=2024-Q3")
            # Without rate limiting, all should succeed
            assert response.status_code == 200
    
    def test_concurrent_requests(self, client, mock_run):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        mock_run.return_value = "Concurrent test response"
        
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
        for i in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(status == 200 for status in results), f"Not all requests succeeded: {results}"
        assert len(results) == 3


if __name__ == "__main__":
    # For direct execution, run all tests
    async def run_all_tests():
        print("FASTAPI APP TEST SUITE")
        print("=" * 60)
        
        # Note: These tests require the FastAPI TestClient
        # Run with: PYTHONPATH=. python -m pytest tests/test_fastapi_app.py -v
        print("Run with: PYTHONPATH=. python -m pytest tests/test_fastapi_app.py -v")
    
    asyncio.run(run_all_tests())
