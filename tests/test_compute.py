"""
Tests for compute module - focused on mathematical properties and behavior.
"""

import numpy as np
import pytest
from semantic_similarity_rating.compute import (
    scale_pmf,
    response_embeddings_to_pmf,
)

# Test constants
EMBEDDING_DIM = 384  # Realistic embedding dimension
LIKERT_SIZE = 5


def assert_valid_pmf(pmf_array):
    """Assert array contains valid probability mass function(s)."""
    if pmf_array.ndim == 1:
        pmf_array = pmf_array.reshape(1, -1)

    for i, pmf in enumerate(pmf_array):
        assert np.isclose(pmf.sum(), 1.0, atol=1e-10), f"PMF {i} doesn't sum to 1"
        assert np.all(pmf >= 0), f"PMF {i} has negative probabilities"


def create_test_embeddings(n_responses=3, n_dimensions=EMBEDDING_DIM, seed=42):
    """Create deterministic test embeddings."""
    np.random.seed(seed)
    return np.random.randn(n_responses, n_dimensions)


def create_likert_embeddings(n_dimensions=EMBEDDING_DIM, n_points=LIKERT_SIZE, seed=42):
    """Create deterministic Likert reference embeddings."""
    np.random.seed(seed + 1)  # Different seed for diversity
    return np.random.randn(n_dimensions, n_points)


class TestScalePMF:
    """Test temperature scaling of probability distributions."""

    def test_temperature_identity(self):
        """Temperature of 1.0 should leave PMF unchanged."""
        pmf = np.array([0.1, 0.2, 0.3, 0.4])
        scaled = scale_pmf(pmf, temperature=1.0)
        assert np.allclose(scaled, pmf)

    def test_temperature_extremes(self):
        """Test behavior at temperature extremes."""
        pmf = np.array([0.1, 0.2, 0.3, 0.4])

        # Near-zero temperature should create one-hot distribution
        sharp = scale_pmf(pmf, temperature=0.01)
        assert_valid_pmf(sharp)
        assert sharp[np.argmax(pmf)] > 0.99  # Highest prob element dominates

        # High temperature should be more uniform
        smooth = scale_pmf(pmf, temperature=10.0)
        assert_valid_pmf(smooth)

        # Higher temperature should increase entropy
        sharp_entropy = -np.sum(sharp * np.log(sharp + 1e-12))
        smooth_entropy = -np.sum(smooth * np.log(smooth + 1e-12))
        assert smooth_entropy > sharp_entropy

    def test_temperature_capping(self):
        """Temperature should be capped at max_temp."""
        pmf = np.array([0.1, 0.6, 0.3])

        capped = scale_pmf(pmf, temperature=100.0, max_temp=5.0)
        expected = scale_pmf(pmf, temperature=5.0)
        assert np.allclose(capped, expected)


class TestEmbeddingsToPMF:
    """Test core embedding-to-PMF conversion function."""

    def test_basic_functionality(self):
        """Should convert embeddings to valid PMFs with correct shape."""
        response_embs = create_test_embeddings(n_responses=3)
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert result.shape == (3, LIKERT_SIZE)
        assert_valid_pmf(result)

    def test_deterministic_behavior(self):
        """Identical inputs should produce identical outputs."""
        # Create identical response embeddings
        response_embs = np.tile(create_test_embeddings(n_responses=1), (2, 1))
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert np.allclose(result[0], result[1])
        assert_valid_pmf(result)

    def test_epsilon_regularization(self):
        """Epsilon should prevent zero probabilities and affect distribution."""
        response_embs = create_test_embeddings(n_responses=2)
        likert_embs = create_likert_embeddings()

        no_eps = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.0)
        with_eps = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.1)

        assert_valid_pmf(no_eps)
        assert_valid_pmf(with_eps)
        assert not np.allclose(no_eps, with_eps)

        # With epsilon, all probabilities should be positive
        assert np.all(with_eps > 0)

    def test_epsilon_effect_on_uniformity(self):
        """Higher epsilon should generally create more uniform distributions."""
        response_embs = create_test_embeddings(n_responses=1)
        likert_embs = create_likert_embeddings()

        low_eps = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.001)[
            0
        ]
        high_eps = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.1)[
            0
        ]

        # Higher epsilon should increase entropy
        low_entropy = -np.sum(low_eps * np.log(low_eps + 1e-12))
        high_entropy = -np.sum(high_eps * np.log(high_eps + 1e-12))

        assert high_entropy >= low_entropy  # Should be more uniform

    def test_empty_input_handling(self):
        """Should handle empty response arrays gracefully."""
        empty_responses = np.empty((0, EMBEDDING_DIM))
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(empty_responses, likert_embs)

        assert result.shape == (0, LIKERT_SIZE)
        assert isinstance(result, np.ndarray)

    def test_similarity_ranking_preserved(self):
        """PMF should reflect embedding similarity ranking."""
        # Create response that's most similar to first Likert point
        likert_embs = create_likert_embeddings()
        response_embs = likert_embs[:, 0:1].T  # Transpose to make it (1, dim)

        result = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=0.01)

        # First Likert point should have highest probability
        assert np.argmax(result[0]) == 0


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_response_realistic_dimension(self):
        """Should work with single response and realistic embeddings."""
        response_embs = np.array([[1.0, 0.5, -0.2]])  # Non-zero, realistic
        likert_embs = np.array(
            [
                [1.0, 0.5, 0.0, -0.5, -1.0],
                [0.8, 0.2, 0.1, -0.3, -0.8],
                [0.6, 0.1, 0.0, -0.1, -0.6],
            ]
        )

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert result.shape == (1, 5)
        assert_valid_pmf(result)

    def test_small_but_nonzero_embeddings(self):
        """Should handle small but non-zero embeddings without errors."""
        response_embs = np.full((2, EMBEDDING_DIM), 1e-6)  # Small but not zero
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert result.shape == (2, LIKERT_SIZE)
        assert_valid_pmf(result)
        # Both responses should be identical since they're the same
        assert np.allclose(result[0], result[1])

    def test_extreme_similarity_values(self):
        """Should handle very high and very low similarity values."""
        # Create response that's very similar to one Likert point
        likert_embs = create_likert_embeddings()
        response_embs = likert_embs[:, 2:3].T * 1000  # Scale up for extreme similarity

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert_valid_pmf(result)
        # Should strongly prefer the similar point
        assert result[0, 2] > 0.8  # High probability for the similar point


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_large_embeddings(self):
        """Should handle large embedding values without numerical issues."""
        response_embs = create_test_embeddings() * 1000
        likert_embs = create_likert_embeddings() * 1000

        result = response_embeddings_to_pmf(response_embs, likert_embs)

        assert_valid_pmf(result)
        assert np.all(np.isfinite(result))

    def test_very_small_epsilon(self):
        """Should handle very small epsilon values."""
        response_embs = create_test_embeddings(n_responses=1)
        likert_embs = create_likert_embeddings()

        result = response_embeddings_to_pmf(response_embs, likert_embs, epsilon=1e-10)

        assert_valid_pmf(result)
        assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
