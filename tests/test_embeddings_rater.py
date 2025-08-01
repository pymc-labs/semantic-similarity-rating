"""
Tests for the EmbeddingsRater class.
"""

import numpy as np
import polars as po
import pytest
from embeddings_similarity_rating import EmbeddingsRater


class TestEmbeddingsRaterInitialization:
    """Test EmbeddingsRater initialization and validation."""

    def test_valid_initialization(self):
        """Test initialization with valid data."""
        # Create valid reference data
        df = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                "sentence": ["very bad", "bad", "neutral", "good", "very good"],
                "embedding_small": [np.random.rand(10).tolist() for _ in range(5)],
            }
        )

        rater = EmbeddingsRater(df, embeddings_column="embedding_small")

        # Check that reference matrices were created
        assert "set1" in rater.reference_matrices, "Reference matrix should be created"
        assert rater.reference_matrices["set1"].shape == (
            10,
            5,
        ), "Matrix should have correct shape"

        # Check that mean reference is available
        assert "mean" in rater.reference_sentences, "Mean reference should be available"

    def test_multiple_reference_sets(self):
        """Test initialization with multiple reference sets."""
        # Create data with multiple reference sets
        df = po.DataFrame(
            {
                "id": ["set1"] * 5 + ["set2"] * 5,
                "int_response": [1, 2, 3, 4, 5] * 2,
                "sentence": ["very bad", "bad", "neutral", "good", "very good"] * 2,
                "embedding_small": [np.random.rand(10).tolist() for _ in range(10)],
            }
        )

        rater = EmbeddingsRater(df, embeddings_column="embedding_small")

        # Check that both reference matrices were created
        assert "set1" in rater.reference_matrices, "Set1 should be created"
        assert "set2" in rater.reference_matrices, "Set2 should be created"
        assert len(rater.reference_matrices) == 2, (
            "Should have exactly 2 reference sets"
        )

    def test_invalid_dataframe_structure(self):
        """Test that invalid DataFrame structure raises errors."""
        # Missing required columns
        df_missing_cols = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                # Missing 'sentence' and 'embedding_small'
            }
        )

        with pytest.raises(ValueError, match="Expected reference-sentence data frame"):
            EmbeddingsRater(df_missing_cols, embeddings_column="embedding_small")

    def test_invalid_int_response_structure(self):
        """Test that invalid int_response structure raises errors."""
        # Missing response value (only 4 instead of 5)
        df_incomplete = po.DataFrame(
            {
                "id": ["set1"] * 4,
                "int_response": [1, 2, 3, 4],  # Missing 5
                "sentence": ["very bad", "bad", "neutral", "good"],
                "embedding_small": [np.random.rand(10).tolist() for _ in range(4)],
            }
        )

        with pytest.raises(AssertionError):
            EmbeddingsRater(df_incomplete, embeddings_column="embedding_small")


class TestEmbeddingsRaterResponsePMFs:
    """Test the get_response_pmfs method."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = po.DataFrame(
            {
                "id": ["set1"] * 5 + ["set2"] * 5,
                "int_response": [1, 2, 3, 4, 5] * 2,
                "sentence": ["very bad", "bad", "neutral", "good", "very good"] * 2,
                "embedding_small": [np.random.rand(10).tolist() for _ in range(10)],
            }
        )
        self.rater = EmbeddingsRater(self.df, embeddings_column="embedding_small")
        self.test_responses = np.random.rand(3, 10)

    def test_get_response_pmfs_specific_set(self):
        """Test getting PMFs for a specific reference set."""
        pmfs = self.rater.get_response_pmfs("set1", self.test_responses)

        # Check output shape
        assert pmfs.shape == (3, 5), "Should return 3 responses x 5 Likert points"

        # Check that each row is a valid PMF
        for i in range(3):
            assert np.isclose(pmfs[i].sum(), 1.0), f"Row {i} should sum to 1"
            assert np.all(pmfs[i] >= 0), f"Row {i} should have non-negative values"

    def test_get_response_pmfs_mean_set(self):
        """Test getting PMFs using mean across all reference sets."""
        pmfs = self.rater.get_response_pmfs("mean", self.test_responses)

        # Check output shape
        assert pmfs.shape == (3, 5), "Should return 3 responses x 5 Likert points"

        # Check that each row is a valid PMF
        for i in range(3):
            assert np.isclose(pmfs[i].sum(), 1.0), f"Row {i} should sum to 1"
            assert np.all(pmfs[i] >= 0), f"Row {i} should have non-negative values"

    def test_get_response_pmfs_with_temperature(self):
        """Test getting PMFs with temperature scaling."""
        # Test with temperature < 1 (sharper)
        pmfs_sharp = self.rater.get_response_pmfs(
            "set1", self.test_responses, temperature=0.5
        )

        # Test with temperature > 1 (smoother)
        pmfs_smooth = self.rater.get_response_pmfs(
            "set1", self.test_responses, temperature=2.0
        )

        # Test with temperature = 1 (baseline)
        pmfs_normal = self.rater.get_response_pmfs(
            "set1", self.test_responses, temperature=1.0
        )

        # All should be valid PMFs
        for pmfs in [pmfs_sharp, pmfs_smooth, pmfs_normal]:
            assert pmfs.shape == (3, 5), "Should have correct shape"
            for i in range(3):
                assert np.isclose(pmfs[i].sum(), 1.0), f"Row {i} should sum to 1"
                assert np.all(pmfs[i] >= 0), f"Row {i} should have non-negative values"

        # Sharp distribution should be more peaked than normal
        # (higher maximum values)
        for i in range(3):
            assert pmfs_sharp[i].max() >= pmfs_normal[i].max(), (
                "Sharp should be more peaked"
            )

    def test_get_response_pmfs_invalid_set(self):
        """Test that invalid reference set raises error."""
        with pytest.raises(KeyError):
            self.rater.get_response_pmfs("nonexistent_set", self.test_responses)

    def test_get_response_pmfs_with_epsilon(self):
        """Test that epsilon parameter is properly passed through in get_response_pmfs."""
        # Test with different epsilon values
        result_no_eps = self.rater.get_response_pmfs(
            "set1", self.test_responses, epsilon=0.0
        )
        result_with_eps = self.rater.get_response_pmfs(
            "set1", self.test_responses, epsilon=0.01
        )

        # Both should be valid PMFs
        for result in [result_no_eps, result_with_eps]:
            assert result.shape == (3, 5), "Should have correct shape"
            for i in range(3):
                assert np.isclose(result[i].sum(), 1.0), f"Row {i} should sum to 1"
                assert np.all(result[i] >= 0), (
                    f"Row {i} should have non-negative values"
                )

        # Results should be different due to epsilon
        assert not np.allclose(result_no_eps, result_with_eps), (
            "Results should differ when epsilon is used"
        )

        # With epsilon, no values should be exactly zero
        assert np.all(result_with_eps > 0), "All values should be positive with epsilon"

    def test_get_response_pmfs_with_epsilon_mean_reference(self):
        """Test epsilon parameter with mean reference set."""
        # Test with mean reference and epsilon
        result_no_eps = self.rater.get_response_pmfs(
            "mean", self.test_responses, epsilon=0.0
        )
        result_with_eps = self.rater.get_response_pmfs(
            "mean", self.test_responses, epsilon=0.005
        )

        # Both should be valid PMFs
        for result in [result_no_eps, result_with_eps]:
            assert result.shape == (3, 5), "Should have correct shape"
            for i in range(3):
                assert np.isclose(result[i].sum(), 1.0, atol=1e-10), (
                    f"Row {i} should sum to 1"
                )
                assert np.all(result[i] >= 0), (
                    f"Row {i} should have non-negative values"
                )

        # Results should be different due to epsilon
        assert not np.allclose(result_no_eps, result_with_eps), (
            "Results should differ when epsilon is used with mean reference"
        )

    def test_epsilon_and_temperature_interaction(self):
        """Test that epsilon and temperature parameters work together correctly."""
        # Test different combinations
        combinations = [
            (1.0, 0.0),  # No temperature scaling, no epsilon
            (1.0, 0.01),  # No temperature scaling, with epsilon
            (2.0, 0.0),  # Temperature scaling, no epsilon
            (2.0, 0.01),  # Both temperature scaling and epsilon
        ]

        results = []
        for temperature, epsilon in combinations:
            result = self.rater.get_response_pmfs(
                "set1", self.test_responses, temperature=temperature, epsilon=epsilon
            )
            results.append(result)

            # Each result should be a valid PMF
            assert result.shape == (3, 5), "Should have correct shape"
            for i in range(3):
                assert np.isclose(result[i].sum(), 1.0, atol=1e-10), (
                    f"Row {i} should sum to 1 with T={temperature}, ε={epsilon}"
                )
                assert np.all(result[i] >= 0), (
                    f"Row {i} should have non-negative values with T={temperature}, ε={epsilon}"
                )

        # All results should be different from each other
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert not np.allclose(results[i], results[j]), (
                    f"Results {i} and {j} should be different"
                )


class TestEmbeddingsRaterSurveyPMFs:
    """Test survey-level PMF methods."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                "sentence": ["very bad", "bad", "neutral", "good", "very good"],
                "embedding_small": [np.random.rand(10).tolist() for _ in range(5)],
            }
        )
        self.rater = EmbeddingsRater(self.df, embeddings_column="embedding_small")
        self.test_responses = np.random.rand(5, 10)

    def test_get_survey_response_pmf(self):
        """Test aggregating individual PMFs to survey-level PMF."""
        # Get individual PMFs
        individual_pmfs = self.rater.get_response_pmfs("set1", self.test_responses)

        # Get survey PMF
        survey_pmf = self.rater.get_survey_response_pmf(individual_pmfs)

        # Check that result is valid PMF
        assert survey_pmf.shape == (5,), "Survey PMF should have 5 elements"
        assert np.isclose(survey_pmf.sum(), 1.0), "Survey PMF should sum to 1"
        assert np.all(survey_pmf >= 0), "Survey PMF should have non-negative values"

        # Check that survey PMF is the mean of individual PMFs
        expected = individual_pmfs.mean(axis=0)
        assert np.allclose(survey_pmf, expected), "Should be mean of individual PMFs"

    def test_get_survey_response_pmf_by_reference_set_id(self):
        """Test convenience method for getting survey PMF."""
        # Test convenience method
        survey_pmf_conv = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses
        )

        # Test manual approach
        individual_pmfs = self.rater.get_response_pmfs("set1", self.test_responses)
        survey_pmf_manual = self.rater.get_survey_response_pmf(individual_pmfs)

        # Should be identical
        assert np.allclose(survey_pmf_conv, survey_pmf_manual), (
            "Convenience method should match manual approach"
        )

        # Check that result is valid PMF
        assert np.isclose(survey_pmf_conv.sum(), 1.0), "Survey PMF should sum to 1"
        assert np.all(survey_pmf_conv >= 0), (
            "Survey PMF should have non-negative values"
        )

    def test_get_survey_response_pmf_with_temperature(self):
        """Test convenience method with temperature scaling."""
        # Test with different temperatures
        survey_pmf_normal = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses, temperature=1.0
        )
        survey_pmf_sharp = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses, temperature=0.5
        )
        survey_pmf_smooth = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses, temperature=2.0
        )

        # All should be valid PMFs
        for pmf in [survey_pmf_normal, survey_pmf_sharp, survey_pmf_smooth]:
            assert np.isclose(pmf.sum(), 1.0), "PMF should sum to 1"
            assert np.all(pmf >= 0), "PMF should have non-negative values"

        # Sharp should be more peaked
        assert survey_pmf_sharp.max() >= survey_pmf_normal.max(), (
            "Sharp should be more peaked"
        )

    def test_get_survey_response_pmf_by_reference_set_id_with_epsilon(self):
        """Test that epsilon is properly handled in the convenience method."""
        # Test with epsilon
        result_no_eps = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses, temperature=1.0, epsilon=0.0
        )
        result_with_eps = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses, temperature=1.0, epsilon=0.02
        )

        # Both should be valid PMFs
        for result in [result_no_eps, result_with_eps]:
            assert result.shape == (5,), "Should return 1D array with 5 elements"
            assert np.isclose(result.sum(), 1.0), "Should sum to 1"
            assert np.all(result >= 0), "All values should be non-negative"

        # Results should be different due to epsilon
        assert not np.allclose(result_no_eps, result_with_eps), (
            "Results should differ when epsilon is used"
        )

    def test_epsilon_with_temperature_survey_level(self):
        """Test epsilon and temperature interaction at survey level."""
        # Test different combinations at survey level
        combinations = [
            (1.0, 0.0),  # Baseline
            (1.0, 0.01),  # Only epsilon
            (0.5, 0.0),  # Only temperature (sharp)
            (0.5, 0.01),  # Both (sharp with epsilon)
            (2.0, 0.01),  # Both (smooth with epsilon)
        ]

        survey_results = []
        for temperature, epsilon in combinations:
            result = self.rater.get_survey_response_pmf_by_reference_set_id(
                "set1", self.test_responses, temperature=temperature, epsilon=epsilon
            )
            survey_results.append(result)

            # Each result should be a valid PMF
            assert result.shape == (5,), "Should have correct shape"
            assert np.isclose(result.sum(), 1.0), (
                f"Should sum to 1 with T={temperature}, ε={epsilon}"
            )
            assert np.all(result >= 0), (
                f"Should have non-negative values with T={temperature}, ε={epsilon}"
            )

            # With epsilon, all values should be positive
            if epsilon > 0:
                assert np.all(result > 0), (
                    f"All values should be positive with epsilon={epsilon}"
                )

        # Different parameter combinations should yield different results
        for i in range(len(survey_results)):
            for j in range(i + 1, len(survey_results)):
                assert not np.allclose(survey_results[i], survey_results[j]), (
                    f"Survey results {i} and {j} should be different"
                )


class TestEmbeddingsRaterEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_response(self):
        """Test with single response."""
        df = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                "sentence": ["very bad", "bad", "neutral", "good", "very good"],
                "embedding_small": [np.random.rand(10).tolist() for _ in range(5)],
            }
        )
        rater = EmbeddingsRater(df, embeddings_column="embedding_small")

        # Single response
        single_response = np.random.rand(1, 10)
        pmfs = rater.get_response_pmfs("set1", single_response)

        assert pmfs.shape == (1, 5), "Should handle single response"
        assert np.isclose(pmfs[0].sum(), 1.0), "PMF should sum to 1"

    def test_large_number_of_responses(self):
        """Test with large number of responses."""
        df = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                "sentence": ["very bad", "bad", "neutral", "good", "very good"],
                "embedding_small": [np.random.rand(10).tolist() for _ in range(5)],
            }
        )
        rater = EmbeddingsRater(df, embeddings_column="embedding_small")

        # Large number of responses
        large_responses = np.random.rand(100, 10)
        pmfs = rater.get_response_pmfs("set1", large_responses)

        assert pmfs.shape == (100, 5), "Should handle large number of responses"

        # Check that all PMFs are valid
        for i in range(100):
            assert np.isclose(pmfs[i].sum(), 1.0), f"PMF {i} should sum to 1"
            assert np.all(pmfs[i] >= 0), f"PMF {i} should have non-negative values"

    def test_different_embedding_dimensions(self):
        """Test with different embedding dimensions."""
        # Test with different embedding dimension
        embedding_dim = 50
        df = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                "sentence": ["very bad", "bad", "neutral", "good", "very good"],
                "embedding_small": [
                    np.random.rand(embedding_dim).tolist() for _ in range(5)
                ],
            }
        )
        rater = EmbeddingsRater(df, embeddings_column="embedding_small")

        # Response with matching dimension
        responses = np.random.rand(3, embedding_dim)
        pmfs = rater.get_response_pmfs("set1", responses)

        assert pmfs.shape == (3, 5), "Should work with different embedding dimensions"
        for i in range(3):
            assert np.isclose(pmfs[i].sum(), 1.0), f"PMF {i} should sum to 1"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
