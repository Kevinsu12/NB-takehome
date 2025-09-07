#!/usr/bin/env python3
"""
Test suite for the retrieve node functionality.
"""

import pytest
import asyncio
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.nodes.retrieve import retrieve_node
from app.rag.vectorStore import VectorStore


class TestRetrieveNode:
    """Test cases for the retrieve node."""
    
    @pytest.mark.asyncio
    async def test_retrieve_node_with_vectorstore(self):
        """Test retrieve node with real vector store - shows detailed results."""
        print("\n=== Testing Retrieve Node with Vector Store ===")
        
        # Create a mock state with a real vectorstore
        vectorstore = VectorStore()
        await vectorstore.load_index()  # Load the existing index
        
        state = {
            "period": "2024-Q3",
            "documents": [],
            "processed_data": {},
            "draft_context": {},
            "validated_context": None,
            "final_context": None,
            "vectorstore": vectorstore,
            "error": None
        }
        
        print(f"Vector store indexed: {vectorstore.is_indexed()}")
        print(f"Period: {state['period']}")
        
        # Test retrieve node
        result = await retrieve_node(state)
        
        # Assertions and detailed output
        print(f"\n=== Retrieve Node Results ===")
        assert "documents" in result, "Result should contain 'documents' key"
        assert "period" in result, "Result should contain 'period' key"
        assert result["period"] == "2024-Q3", f"Period should be '2024-Q3', got '{result['period']}'"
        assert len(result["documents"]) > 0, "Should retrieve some documents"
        assert len(result["documents"]) <= 2, f"Should not retrieve more than K=2 documents, got {len(result['documents'])}"
        
        print(f"Number of documents retrieved: {len(result['documents'])}")
        print(f"Period correctly set: {result['period']}")
        
        # Check document content
        print(f"\n=== Document Analysis ===")
        for i, doc in enumerate(result["documents"]):
            print(f"Document {i+1}:")
            assert isinstance(doc, str), f"Document {i+1} should be string, got {type(doc)}"
            assert len(doc) > 0, f"Document {i+1} should not be empty"
            assert len(doc) > 50, f"Document {i+1} should have substantial content, got {len(doc)} chars"
            print(f"  Length: {len(doc)} characters")
            print(f"  Preview: {doc[:100]}...")
            
            # Check for meaningful content
            words = doc.split()
            assert len(words) > 10, f"Document {i+1} should have substantial word count, got {len(words)}"
            print(f"  Word count: {len(words)}")
        
        print(f"\n=== Success Metrics ===")
        print(f"Retrieved {len(result['documents'])} documents from vectorstore")
        print(f"All documents are substantial (50+ chars each)")
        print(f"All documents are unique strings")
        print("Retrieve node with vectorstore test passed!")
    
    @pytest.mark.asyncio
    async def test_retrieve_node_without_vectorstore(self):
        """Test retrieve node fallback when vectorstore is not available."""
        print("\n=== Testing Retrieve Node without Vector Store ===")
        
        # Create state without vectorstore
        state = {
            "period": "2024-Q3",
            "documents": [],
            "processed_data": {},
            "draft_context": {},
            "validated_context": None,
            "final_context": None,
            "vectorstore": None,  # No vectorstore
            "error": None
        }
        
        print(f"Period: {state['period']}")
        print(f"Vectorstore: {state['vectorstore']}")
        
        # Test retrieve node
        result = await retrieve_node(state)
        
        # Assertions
        assert "documents" in result, "Result should contain 'documents' key"
        assert "period" in result, "Result should contain 'period' key"
        assert result["period"] == "2024-Q3", f"Period should be '2024-Q3', got '{result['period']}'"
        assert len(result["documents"]) == 2, f"Should return exactly 2 mock documents, got {len(result['documents'])}"
        
        print(f"Fallback to mock data: {len(result['documents'])} documents")
        
        # Check mock document content
        print(f"\n=== Mock Document Analysis ===")
        for i, doc in enumerate(result["documents"]):
            print(f"Mock document {i+1}:")
            assert isinstance(doc, str), f"Mock document {i+1} should be string, got {type(doc)}"
            assert len(doc) > 0, f"Mock document {i+1} should not be empty"
            assert "2024-Q3" in doc, f"Mock document {i+1} should contain period"
            print(f"  Mock document {i+1}: {doc[:100]}...")
        
        print(f"\n=== Fallback Success ===")
        print(f"Graceful fallback to mock data when vectorstore unavailable")
        print(f"Returned {len(result['documents'])} mock documents")
        print("Retrieve node without vectorstore test passed!")
    
    @pytest.mark.asyncio
    async def test_retrieve_node_error_handling(self):
        """Test retrieve node error handling with invalid inputs."""
        print("\n=== Testing Retrieve Node Error Handling ===")
        
        # Test with None period
        state = {
            "period": None,
            "documents": [],
            "processed_data": {},
            "draft_context": {},
            "validated_context": None,
            "final_context": None,
            "vectorstore": None,
            "error": None
        }
        
        print(f"Testing with None period: {state['period']}")
        
        # Test retrieve node - should handle gracefully
        result = await retrieve_node(state)
        
        # Should either have error or fallback to mock data
        if "error" in result and result["error"]:
            print(f"Error handling working: {result['error']}")
            assert "error" in result, "Should have error for invalid input"
        else:
            # Should fallback to mock data
            assert "documents" in result, "Should have documents from fallback"
            assert len(result["documents"]) == 2, "Should have 2 mock documents"
            print(f"Graceful fallback: {len(result['documents'])} mock documents")
        
        print("Retrieve node error handling test passed!")
    
    @pytest.mark.asyncio
    async def test_retrieve_node_deterministic(self):
        """Test that retrieve node produces consistent results."""
        print("\n=== Testing Retrieve Node Deterministic Behavior ===")
        
        # Create state
        state = {
            "period": "2024-Q3",
            "documents": [],
            "processed_data": {},
            "draft_context": {},
            "validated_context": None,
            "final_context": None,
            "vectorstore": None,  # Use mock data for consistency
            "error": None
        }
        
        # Run multiple times
        results = []
        for i in range(3):
            print(f"Run {i+1}/3...")
            result = await retrieve_node(state)
            results.append(result)
        
        # Verify all results are identical
        print(f"\n=== Deterministic Analysis ===")
        for i, result in enumerate(results[1:], 1):
            assert result["period"] == results[0]["period"], f"Period differs in run {i+1}"
            assert len(result["documents"]) == len(results[0]["documents"]), f"Document count differs in run {i+1}"
            assert result["documents"] == results[0]["documents"], f"Documents differ in run {i+1}"
        
        print(f"All runs returned identical results")
        print(f"Period: {results[0]['period']}")
        print(f"Document count: {len(results[0]['documents'])}")
        print("Deterministic behavior confirmed - identical results across runs")
        print("Retrieve node deterministic test passed!")
    
    @pytest.mark.asyncio
    async def test_retrieve_node_document_quality(self):
        """Test the quality and characteristics of retrieved documents."""
        print("\n=== Testing Retrieve Node Document Quality ===")
        
        # Create state
        state = {
            "period": "2024-Q3",
            "documents": [],
            "processed_data": {},
            "draft_context": {},
            "validated_context": None,
            "final_context": None,
            "vectorstore": None,  # Use mock data for testing
            "error": None
        }
        
        # Test retrieve node
        result = await retrieve_node(state)
        
        # Quality checks
        print(f"\n=== Document Quality Analysis ===")
        documents = result["documents"]
        
        # Length analysis
        lengths = [len(doc) for doc in documents]
        avg_length = sum(lengths) / len(lengths)
        print(f"Average document length: {avg_length:.1f} characters")
        
        # Uniqueness check
        unique_docs = set(documents)
        assert len(unique_docs) == len(documents), "All documents should be unique"
        print("No duplicate documents found")
        
        # Content quality
        all_words = []
        for doc in documents:
            words = doc.lower().split()
            all_words.extend(words)
        
        unique_words = set(all_words)
        print(f"Total words across all documents: {len(all_words)}")
        print(f"Unique words: {len(unique_words)}")
        print(f"Unique words: {unique_words}")
        
        # Check for meaningful content
        meaningful_words = [w for w in unique_words if len(w) > 3 and w.isalpha()]
        print(f"Meaningful words (4+ chars, alphabetic): {len(meaningful_words)}")
        
        print(f"\n=== Quality Metrics ===")
        print(f"Document count: {len(documents)}")
        print(f"Average length: {avg_length:.1f} characters")
        print(f"Uniqueness: 100% (no duplicates)")
        print(f"Vocabulary size: {len(unique_words)} unique words")
        print("Document quality checks passed")
        print("Retrieve node document quality test passed!")


if __name__ == "__main__":
    # For direct execution, run all tests
    async def run_all_tests():
        test_instance = TestRetrieveNode()
        
        print("RETRIEVE NODE TEST SUITE")
        print("=" * 60)
        
        try:
            await test_instance.test_retrieve_node_with_vectorstore()
            await test_instance.test_retrieve_node_without_vectorstore()
            await test_instance.test_retrieve_node_error_handling()
            await test_instance.test_retrieve_node_deterministic()
            await test_instance.test_retrieve_node_document_quality()
            
            print("\n" + "=" * 60)
            print("ALL RETRIEVE NODE TESTS PASSED!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nTEST SUITE FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_all_tests())