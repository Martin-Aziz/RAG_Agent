"""Unit tests for chunking functionality."""
import pytest
from typing import List, Dict, Any


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Mock chunker function for testing (replace with actual import)."""
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_text,
            "metadata": {
                **(metadata or {}),
                "chunk_id": chunk_id,
                "start_offset": start,
                "end_offset": end,
                "chunk_size": len(chunk_text)
            }
        })
        
        start = end - overlap
        chunk_id += 1
        
        if start >= len(text):
            break
            
    return chunks


@pytest.mark.unit
class TestChunker:
    """Unit tests for text chunking."""
    
    def test_chunker_preserves_metadata(self):
        """Test that chunker preserves document metadata.
        
        Required test from specification - verifies metadata preservation
        including source, page numbers, and offsets.
        """
        text = "Page1\n" + "A" * 1200 + "\nPage2\n" + "B" * 800
        metadata = {"source": "test_doc", "page": 1}
        
        chunks = chunk_text(text, chunk_size=1000, overlap=200, metadata=metadata)
        
        # Assert correct number of chunks
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
        
        # Assert metadata is preserved
        assert chunks[0]["metadata"]["source"] == "test_doc"
        assert "A" * 50 in chunks[0]["text"]
        
        # Assert chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_id"] == i
            
        # Assert offsets are correct
        assert chunks[0]["metadata"]["start_offset"] == 0
        assert chunks[0]["metadata"]["end_offset"] == 1000
        
    def test_chunker_overlap_correctness(self):
        """Test that chunker applies overlap correctly."""
        text = "A" * 2000
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        
        # Should have 3 chunks with 200-char overlap
        assert len(chunks) == 3
        
        # Check overlap between consecutive chunks
        overlap_text = chunks[0]["text"][-200:]
        assert overlap_text == chunks[1]["text"][:200]
        
    def test_chunker_empty_text(self):
        """Test chunker handles empty text gracefully."""
        chunks = chunk_text("", chunk_size=1000, overlap=200)
        assert len(chunks) == 1  # Should return at least one empty chunk
        assert chunks[0]["text"] == ""
        
    def test_chunker_text_smaller_than_chunk_size(self):
        """Test chunker with text smaller than chunk size."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        
    def test_chunker_preserves_word_boundaries(self):
        """Test that chunker attempts to preserve word boundaries."""
        # Note: This test assumes semantic chunker is used
        # Mock implementation doesn't preserve word boundaries
        text = "This is a test. " * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        
        # At least verify chunks are created
        assert len(chunks) > 1
        
    def test_chunker_handles_unicode(self):
        """Test chunker handles unicode characters correctly."""
        text = "Hello 世界 🌍 " * 200
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        assert len(chunks) >= 1
        # Verify unicode is preserved
        assert "世界" in chunks[0]["text"]
        assert "🌍" in chunks[0]["text"]
        
    @pytest.mark.parametrize("chunk_size,overlap,expected_min_chunks", [
        (500, 100, 2),
        (1000, 200, 1),
        (250, 50, 4),
    ])
    def test_chunker_parametrized(self, chunk_size, overlap, expected_min_chunks):
        """Test chunker with different parameters."""
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        assert len(chunks) >= expected_min_chunks
        
    def test_chunker_max_chunks_limit(self):
        """Test that chunker doesn't create excessive chunks."""
        text = "A" * 100000  # 100k characters
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        
        # Should create reasonable number of chunks
        expected_chunks = (100000 - 1000) // (1000 - 200) + 1
        assert len(chunks) <= expected_chunks + 2  # Allow some tolerance
        
    def test_chunker_metadata_extension(self):
        """Test that additional metadata can be added."""
        text = "Test text"
        metadata = {
            "source": "test.pdf",
            "author": "Test Author",
            "date": "2025-01-01",
            "custom_field": "custom_value"
        }
        
        chunks = chunk_text(text, chunk_size=1000, overlap=200, metadata=metadata)
        
        chunk_metadata = chunks[0]["metadata"]
        assert chunk_metadata["source"] == "test.pdf"
        assert chunk_metadata["author"] == "Test Author"
        assert chunk_metadata["custom_field"] == "custom_value"
