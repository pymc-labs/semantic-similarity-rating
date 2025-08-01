"""
Tests for the compute module functions.
"""

import numpy as np
import pytest
from embeddings_similarity_rating.compute import (
    cos_to_pdf,
    cos_sim,
    scale_pdf,
    scale_pdf_no_max_temp,
    cos_sim_pdf,
    KS_sim_pdf,
    pdf_moment,
    mean,
    var,
    std,
    get_optimal_temperature_mean,
    get_optimal_temperature_KS_sim,
    response_embeddings_to_pdf,
)


class TestBasicFunctions:
    """Test basic utility functions."""

    def test_cos_to_pdf(self):
        """Test cosine similarity to PDF conversion."""
        cos = np.array([0.5, 0.8, 0.3, 0.9, 0.1])
        pdf = cos_to_pdf(cos)

        # Check that result is a valid PDF
        assert np.isclose(pdf.sum(), 1.0), "PDF should sum to 1"
        assert np.all(pdf >= 0), "PDF values should be non-negative"
        assert pdf.shape == cos.shape, "PDF should have same shape as input"

        # Check that minimum value becomes 0
        assert np.isclose(pdf.min(), 0.0), "Minimum PDF value should be 0"

    def test_cos_sim(self):
        """Test cosine similarity between embeddings."""
        # Test identical vectors
        emb1 = np.array([1, 2, 3])
        result = cos_sim(emb1, emb1)
        assert np.isclose(result, 1.0), "Identical vectors should have similarity 1"

        # Test orthogonal vectors
        emb1 = np.array([1, 0])
        emb2 = np.array([0, 1])
        result = cos_sim(emb1, emb2)
        assert np.isclose(result, 0.5), "Orthogonal vectors should have similarity 0.5"

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

    def test_scale_pdf_no_max_temp(self):
        """Test PDF scaling without max temperature limit."""
        pdf = np.array([0.1, 0.2, 0.3, 0.4])

        # Should be equivalent to scale_pdf with max_temp=inf
        scaled1 = scale_pdf_no_max_temp(pdf, temperature=2.0)
        scaled2 = scale_pdf(pdf, temperature=2.0, max_temp=np.inf)
        assert np.allclose(scaled1, scaled2), (
            "Should be equivalent to scale_pdf with max_temp=inf"
        )


class TestPDFSimilarity:
    """Test PDF similarity functions."""

    def test_cos_sim_pdf(self):
        """Test cosine similarity between PDFs."""
        pdf1 = np.array([0.2, 0.3, 0.5])

        # Test identical PDFs
        result = cos_sim_pdf(pdf1, pdf1)
        assert np.isclose(result, 1.0), "Identical PDFs should have similarity 1"

        # Test orthogonal PDFs
        pdf2 = np.array([0.0, 1.0, 0.0])
        pdf3 = np.array([1.0, 0.0, 0.0])
        result = cos_sim_pdf(pdf2, pdf3)
        assert np.isclose(result, 0.0), "Orthogonal PDFs should have similarity 0"

    def test_ks_sim_pdf(self):
        """Test Kolmogorov-Smirnov similarity between PDFs."""
        pdf1 = np.array([0.2, 0.3, 0.5])

        # Test identical PDFs
        result = KS_sim_pdf(pdf1, pdf1)
        assert np.isclose(result, 1.0), "Identical PDFs should have KS similarity 1"

        # Test completely different PDFs
        pdf2 = np.array([1.0, 0.0, 0.0])
        pdf3 = np.array([0.0, 0.0, 1.0])
        result = KS_sim_pdf(pdf2, pdf3)
        assert result < 1.0, "Different PDFs should have KS similarity < 1"


class TestPDFMoments:
    """Test PDF moment calculations."""

    def setup_method(self):
        """Set up test data."""
        self.x = np.array([1, 2, 3, 4, 5])
        self.pdf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    def test_pdf_moment(self):
        """Test general moment calculation."""
        # Test 0th moment (should be 1 for normalized PDF)
        m0 = pdf_moment(self.pdf, self.x, 0)
        assert np.isclose(m0, 1.0), "0th moment should be 1"

        # Test 1st moment (mean)
        m1 = pdf_moment(self.pdf, self.x, 1)
        expected_mean = np.sum(self.x * self.pdf)
        assert np.isclose(m1, expected_mean), "1st moment should equal weighted mean"

    def test_mean(self):
        """Test mean calculation."""
        result = mean(self.pdf, self.x)
        expected = np.sum(self.x * self.pdf)
        assert np.isclose(result, expected), (
            "Mean calculation should match expected value"
        )

    def test_var(self):
        """Test variance calculation."""
        result = var(self.pdf, self.x)

        # Calculate expected variance
        _mean = mean(self.pdf, self.x)
        expected = np.sum(self.pdf * (self.x - _mean) ** 2)

        assert np.isclose(result, expected), (
            "Variance calculation should match expected value"
        )
        assert result >= 0, "Variance should be non-negative"

    def test_std(self):
        """Test standard deviation calculation."""
        result = std(self.pdf, self.x)
        expected = np.sqrt(var(self.pdf, self.x))

        assert np.isclose(result, expected), "Std should be sqrt of variance"
        assert result >= 0, "Standard deviation should be non-negative"


class TestOptimization:
    """Test optimization functions."""

    def setup_method(self):
        """Set up test data."""
        self.x = np.arange(1, 6)
        self.pdf = np.array([0.1, 0.15, 0.05, 0.2, 0.5])

    def test_get_optimal_temperature_mean(self):
        """Test finding optimal temperature to match mean."""
        target_mean = 3.0

        T_opt, scaled_pdf = get_optimal_temperature_mean(self.x, self.pdf, target_mean)

        # Check that temperature is reasonable
        assert T_opt > 0, "Temperature should be positive"
        assert T_opt <= 10.0, "Temperature should be within bounds"

        # Check that resulting PDF has correct mean
        result_mean = mean(scaled_pdf, self.x)
        assert np.isclose(result_mean, target_mean, atol=1e-3), (
            "Should match target mean"
        )

        # Check that result is valid PDF
        assert np.isclose(scaled_pdf.sum(), 1.0), "Result should be valid PDF"
        assert np.all(scaled_pdf >= 0), "PDF values should be non-negative"

    def test_get_optimal_temperature_ks_sim(self):
        """Test finding optimal temperature to maximize KS similarity."""
        target_pdf = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform distribution

        T_opt, scaled_pdf = get_optimal_temperature_KS_sim(self.pdf, target_pdf)

        # Check that temperature is reasonable
        assert T_opt > 0, "Temperature should be positive"
        assert T_opt <= 10.0, "Temperature should be within bounds"

        # Check that result is valid PDF
        assert np.isclose(scaled_pdf.sum(), 1.0), "Result should be valid PDF"
        assert np.all(scaled_pdf >= 0), "PDF values should be non-negative"

        # Check that KS similarity improved
        original_sim = KS_sim_pdf(self.pdf, target_pdf)
        optimized_sim = KS_sim_pdf(scaled_pdf, target_pdf)
        assert optimized_sim >= original_sim, (
            "Optimization should improve KS similarity"
        )


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
