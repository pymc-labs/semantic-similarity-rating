"""
Tests for the compute module functions.
"""

import numpy as np
import pytest
from embeddings_similarity_rating.compute import (
    scale_pmf,
    response_embeddings_to_pmf,
)


class TestBasicFunctions:
    """Test basic utility functions."""

    def test_scale_pmf(self):
        """Test PMF temperature scaling."""
        pmf = np.array([0.1, 0.2, 0.3, 0.4])

        # Test temperature = 1 (no change)
        scaled = scale_pmf(pmf, temperature=1.0)
        assert np.allclose(scaled, pmf), "Temperature 1 should not change PMF"

        # Test temperature = 0 (one-hot)
        scaled = scale_pmf(pmf, temperature=0.0)
        assert np.isclose(scaled.sum(), 1.0), "Scaled PMF should sum to 1"
        assert scaled[np.argmax(pmf)] == 1.0, "Max element should become 1"
        assert np.sum(scaled > 0) == 1, "Only one element should be positive"

        # Test temperature > 1 (smoother)
        scaled = scale_pmf(pmf, temperature=2.0)
        assert np.isclose(scaled.sum(), 1.0), "Scaled PMF should sum to 1"

        # Test max_temp limit
        scaled = scale_pmf(pmf, temperature=20.0, max_temp=5.0)
        expected = scale_pmf(pmf, temperature=5.0)
        assert np.allclose(scaled, expected), "Should cap at max_temp"


class TestEmbeddingsToPMF:
    """Test the core response_embeddings_to_pmf function."""

    def test_response_embeddings_to_pmf(self):
        """Test conversion from embeddings to PMF."""
        # Create test data
        n_responses = 3
        n_dimensions = 10
        n_likert_points = 5

        # Generate random embeddings
        np.random.seed(42)
        response_embeddings = np.random.rand(n_responses, n_dimensions)
        likert_embeddings = np.random.rand(n_dimensions, n_likert_points)

        # Test the function
        result = response_embeddings_to_pmf(response_embeddings, likert_embeddings)

        # Check output shape
        expected_shape = (n_responses, n_likert_points)
        assert result.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {result.shape}"
        )

        # Check that each row is a valid PMF
        for i in range(n_responses):
            row_sum = result[i].sum()
            assert np.isclose(row_sum, 1.0), f"Row {i} should sum to 1, got {row_sum}"
            assert np.all(result[i] >= 0), f"Row {i} should have non-negative values"

        # Check that at least one element in each row is zero (due to min subtraction)
        for i in range(n_responses):
            assert np.any(result[i] == 0), (
                f"Row {i} should have at least one zero element"
            )

    def test_response_embeddings_to_pmf_edge_cases(self):
        """Test edge cases for response_embeddings_to_pmf."""
        # Test with identical embeddings
        n_dimensions = 5
        n_likert_points = 3

        # All response embeddings are identical
        response_embeddings = np.ones((2, n_dimensions))
        likert_embeddings = np.random.rand(n_dimensions, n_likert_points)

        result = response_embeddings_to_pmf(response_embeddings, likert_embeddings)

        # Both rows should be identical
        assert np.allclose(result[0], result[1]), (
            "Identical inputs should produce identical outputs"
        )

        # Each row should still be a valid PMF
        for i in range(2):
            assert np.isclose(result[i].sum(), 1.0), f"Row {i} should sum to 1"
            assert np.all(result[i] >= 0), f"Row {i} should have non-negative values"

    def test_response_embeddings_to_pmf_with_epsilon(self):
        """Test response_embeddings_to_pmf with non-zero epsilon parameter."""
        np.random.seed(42)  # For reproducible results

        n_dimensions = 4
        n_likert_points = 5
        n_responses = 3

        response_embeddings = np.random.rand(n_responses, n_dimensions)
        likert_embeddings = np.random.rand(n_dimensions, n_likert_points)

        epsilon_values = [0.0, 1e-6, 1e-3, 0.1]

        for epsilon in epsilon_values:
            result = response_embeddings_to_pmf(
                response_embeddings, likert_embeddings, epsilon=epsilon
            )

            # Each row should be a valid PMF regardless of epsilon
            for i in range(n_responses):
                assert np.isclose(result[i].sum(), 1.0, atol=1e-10), (
                    f"Row {i} should sum to 1 with epsilon={epsilon}"
                )
                assert np.all(result[i] >= 0), (
                    f"Row {i} should have non-negative values with epsilon={epsilon}"
                )

            # With positive epsilon, no values should be exactly zero
            if epsilon > 0:
                assert np.all(result > 0), (
                    f"All values should be positive with epsilon={epsilon}"
                )

    def test_response_embeddings_to_pmf_epsilon_effects(self):
        """Test specific mathematical effects of epsilon parameter."""
        np.random.seed(123)

        n_dimensions = 3
        n_likert_points = 4
        response_embeddings = np.random.rand(2, n_dimensions)
        likert_embeddings = np.random.rand(n_dimensions, n_likert_points)

        # Compare results with and without epsilon
        result_no_eps = response_embeddings_to_pmf(
            response_embeddings, likert_embeddings, epsilon=0.0
        )
        result_with_eps = response_embeddings_to_pmf(
            response_embeddings, likert_embeddings, epsilon=0.01
        )

        # Results should be different
        assert not np.allclose(result_no_eps, result_with_eps), (
            "Results should differ when epsilon is added"
        )

        # With epsilon, minimum positions should get boosted
        for i in range(result_no_eps.shape[0]):
            # Find positions that were zero (minimum) without epsilon
            zero_positions = result_no_eps[i] == 0
            if np.any(zero_positions):
                # These positions should now be positive with epsilon
                assert np.all(result_with_eps[i][zero_positions] > 0), (
                    "Previously zero positions should be positive with epsilon"
                )

    def test_response_embeddings_to_pmf_epsilon_kronecker_delta(self):
        """Test that epsilon correctly implements Kronecker delta behavior."""
        # Create response embedding
        response_embeddings = np.array([[1.0, 0.0]])

        # Create likert embeddings where the middle one will have lowest similarity
        likert_embeddings = np.array(
            [
                [1.0, 0.0, 0.5],  # High similarity to response
                [0.0, 1.0, 0.8],  # Low similarity to response for middle column
            ]
        )

        epsilon = 0.1
        result = response_embeddings_to_pmf(
            response_embeddings, likert_embeddings, epsilon=epsilon
        )

        # The result should be a valid PMF
        assert np.isclose(result.sum(), 1.0), "Result should sum to 1"
        assert np.all(result >= 0), "All values should be non-negative"

        # All positions should be positive due to epsilon
        assert np.all(result > 0), "All positions should be positive with epsilon"

    def test_response_embeddings_to_pmf_empty_input(self):
        """Test response_embeddings_to_pmf with empty input."""
        # Create empty response matrix but valid likert matrix
        empty_responses = np.empty((0, 4))  # 0 responses, 4 dimensions
        likert_embeddings = np.random.rand(4, 5)  # 4 dimensions, 5 Likert points

        result = response_embeddings_to_pmf(empty_responses, likert_embeddings)

        # Should return empty result with correct shape
        assert result.shape == (0, 5), "Should return (0, 5) for empty input"
        assert isinstance(result, np.ndarray), "Should return numpy array"

    def test_response_embeddings_to_pmf_epsilon_consistency(self):
        """Test that epsilon behavior is consistent across different scales."""
        np.random.seed(456)

        n_dimensions = 5
        n_likert_points = 5
        response_embeddings = np.random.rand(3, n_dimensions)
        likert_embeddings = np.random.rand(n_dimensions, n_likert_points)

        # Test with different epsilon values
        epsilons = [0.001, 0.01, 0.1]
        results = []

        for epsilon in epsilons:
            result = response_embeddings_to_pmf(
                response_embeddings, likert_embeddings, epsilon=epsilon
            )
            results.append(result)

            # Basic validation for each epsilon
            assert result.shape == (3, 5), "Shape should be preserved"
            for i in range(3):
                assert np.isclose(result[i].sum(), 1.0), (
                    f"Row {i} should sum to 1 with epsilon={epsilon}"
                )

        # Larger epsilon should generally lead to more uniform distributions
        # (entropy should increase with epsilon)
        for i in range(3):
            entropy_small = -np.sum(results[0][i] * np.log(results[0][i] + 1e-12))
            entropy_large = -np.sum(results[2][i] * np.log(results[2][i] + 1e-12))

            # Larger epsilon should generally increase entropy (more uniform)
            assert entropy_large >= entropy_small - 1e-10, (
                f"Entropy should not decrease significantly with larger epsilon for row {i}"
            )


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
