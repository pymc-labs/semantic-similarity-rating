"""
Tests for the compute module functions.
"""

import numpy as np
import pytest
from embeddings_similarity_rating.compute import (
    scale_pdf,
    response_embeddings_to_pdf,
)


class TestBasicFunctions:
    """Test basic utility functions."""

    def test_scale_pdf(self):
        """Test PDF temperature scaling."""
        pdf = np.array([0.1, 0.2, 0.3, 0.4])

        # Test temperature = 1 (no change)
        scaled = scale_pdf(pdf, temperature=1.0)
        assert np.allclose(scaled, pdf), "Temperature 1 should not change PDF"

        # Test temperature = 0 (one-hot)
        scaled = scale_pdf(pdf, temperature=0.0)
        assert np.isclose(scaled.sum(), 1.0), "Scaled PDF should sum to 1"
        assert scaled[np.argmax(pdf)] == 1.0, "Max element should become 1"
        assert np.sum(scaled > 0) == 1, "Only one element should be positive"

        # Test temperature > 1 (smoother)
        scaled = scale_pdf(pdf, temperature=2.0)
        assert np.isclose(scaled.sum(), 1.0), "Scaled PDF should sum to 1"

        # Test max_temp limit
        scaled = scale_pdf(pdf, temperature=20.0, max_temp=5.0)
        expected = scale_pdf(pdf, temperature=5.0)
        assert np.allclose(scaled, expected), "Should cap at max_temp"


class TestEmbeddingsToPDF:
    """Test the core response_embeddings_to_pdf function."""

    def test_response_embeddings_to_pdf(self):
        """Test conversion from embeddings to PDF."""
        # Create test data
        n_responses = 3
        n_dimensions = 10
        n_likert_points = 5

        # Generate random embeddings
        np.random.seed(42)
        response_embeddings = np.random.rand(n_responses, n_dimensions)
        likert_embeddings = np.random.rand(n_dimensions, n_likert_points)

        # Test the function
        result = response_embeddings_to_pdf(response_embeddings, likert_embeddings)

        # Check output shape
        expected_shape = (n_responses, n_likert_points)
        assert result.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {result.shape}"
        )

        # Check that each row is a valid PDF
        for i in range(n_responses):
            row_sum = result[i].sum()
            assert np.isclose(row_sum, 1.0), f"Row {i} should sum to 1, got {row_sum}"
            assert np.all(result[i] >= 0), f"Row {i} should have non-negative values"

        # Check that at least one element in each row is zero (due to min subtraction)
        for i in range(n_responses):
            assert np.any(result[i] == 0), (
                f"Row {i} should have at least one zero element"
            )

    def test_response_embeddings_to_pdf_edge_cases(self):
        """Test edge cases for response_embeddings_to_pdf."""
        # Test with identical embeddings
        n_dimensions = 5
        n_likert_points = 3

        # All response embeddings are identical
        response_embeddings = np.ones((2, n_dimensions))
        likert_embeddings = np.random.rand(n_dimensions, n_likert_points)

        result = response_embeddings_to_pdf(response_embeddings, likert_embeddings)

        # Both rows should be identical
        assert np.allclose(result[0], result[1]), (
            "Identical inputs should produce identical outputs"
        )

        # Each row should still be a valid PDF
        for i in range(2):
            assert np.isclose(result[i].sum(), 1.0), f"Row {i} should sum to 1"
            assert np.all(result[i] >= 0), f"Row {i} should have non-negative values"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
