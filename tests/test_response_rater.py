"""
Tests for the ResponseRater class.
"""

import numpy as np
import polars as po
import pytest
from embeddings_similarity_rating import ResponseRater


class TestResponseRaterInitialization:
    """Test ResponseRater initialization and validation."""

    def test_valid_initialization(self):
        """Test initialization with valid data."""
        # Create valid reference data (text only)
        df = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                "sentence": ["very bad", "bad", "neutral", "good", "very good"],
            }
        )

        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        # Check that reference matrices were created
        assert "set1" in rater.reference_matrices, "Reference matrix should be created"

        # Check matrix dimensions (embeddings are transposed, so should be embedding_dim x 5)
        matrix_shape = rater.reference_matrices["set1"].shape
        assert matrix_shape[1] == 5, "Matrix should have 5 columns for 5 Likert points"
        assert matrix_shape[0] > 0, "Matrix should have positive embedding dimension"

        # Check that mean reference is available
        assert "mean" in rater.reference_sentences, "Mean reference should be available"

        # Check that actual sentences are stored
        assert "set1" in rater.reference_sentences, "Set1 sentences should be stored"
        assert len(rater.reference_sentences["set1"]) == 5, "Should have 5 sentences"

    def test_multiple_reference_sets(self):
        """Test initialization with multiple reference sets."""
        # Create data with multiple reference sets
        df = po.DataFrame(
            {
                "id": ["set1"] * 5 + ["set2"] * 5,
                "int_response": [1, 2, 3, 4, 5] * 2,
                "sentence": ["very bad", "bad", "neutral", "good", "very good"] * 2,
            }
        )

        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        # Check that both reference matrices were created
        assert "set1" in rater.reference_matrices, "Set1 should be created"
        assert "set2" in rater.reference_matrices, "Set2 should be created"
        assert len(rater.reference_matrices) == 2, (
            "Should have exactly 2 reference sets"
        )

        # Check that sentences are stored for both sets
        assert len(rater.reference_sentences["set1"]) == 5, (
            "Set1 should have 5 sentences"
        )
        assert len(rater.reference_sentences["set2"]) == 5, (
            "Set2 should have 5 sentences"
        )

    def test_custom_model(self):
        """Test initialization with different model."""
        df = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                "sentence": ["very bad", "bad", "neutral", "good", "very good"],
            }
        )

        # Use a different (smaller) model for testing
        rater = ResponseRater(df, model_name="all-MiniLM-L12-v2")

        # Check that model info is accessible
        model_info = rater.model_info
        assert "model_name" in model_info, "Model info should contain model name"
        assert "embedding_dimension" in model_info, (
            "Model info should contain embedding dimension"
        )

    def test_invalid_dataframe_structure(self):
        """Test that invalid DataFrame structure raises errors."""
        # Missing required columns
        df_missing_cols = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                # Missing 'sentence'
            }
        )

        with pytest.raises(ValueError, match="Expected reference-sentence data frame"):
            ResponseRater(df_missing_cols)

    def test_invalid_int_response_structure(self):
        """Test that invalid int_response structure raises errors."""
        # Missing response value (only 4 instead of 5)
        df_incomplete = po.DataFrame(
            {
                "id": ["set1"] * 4,
                "int_response": [1, 2, 3, 4],  # Missing 5
                "sentence": ["very bad", "bad", "neutral", "good"],
            }
        )

        with pytest.raises(AssertionError):
            ResponseRater(df_incomplete)


class TestResponseRaterResponsePMFs:
    """Test the get_response_pmfs method."""

    def setup_method(self):
        """Set up test data."""
        self.df = po.DataFrame(
            {
                "id": ["set1"] * 5 + ["set2"] * 5,
                "int_response": [1, 2, 3, 4, 5] * 2,
                "sentence": ["very bad", "bad", "neutral", "good", "very good"] * 2,
            }
        )
        self.rater = ResponseRater(self.df, model_name="all-MiniLM-L6-v2")
        self.test_responses = [
            "I completely agree",
            "This is okay",
            "I disagree strongly",
        ]

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

    def test_get_response_pmfs_with_epsilon(self):
        """Test that epsilon parameter works correctly."""
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

    def test_get_response_pmfs_invalid_set(self):
        """Test that invalid reference set raises error."""
        with pytest.raises(KeyError):
            self.rater.get_response_pmfs("nonexistent_set", self.test_responses)

    def test_single_response(self):
        """Test with single response text."""
        single_response = ["I agree completely"]
        pmfs = self.rater.get_response_pmfs("set1", single_response)

        assert pmfs.shape == (1, 5), "Should handle single response"
        assert np.isclose(pmfs[0].sum(), 1.0), "PMF should sum to 1"

    def test_empty_response_list(self):
        """Test with empty response list."""
        empty_responses = []
        pmfs = self.rater.get_response_pmfs("set1", empty_responses)

        assert pmfs.shape == (0, 5), "Should handle empty response list"


class TestResponseRaterSurveyPMFs:
    """Test survey-level PMF methods."""

    def setup_method(self):
        """Set up test data."""
        self.df = po.DataFrame(
            {
                "id": ["set1"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                "sentence": ["very bad", "bad", "neutral", "good", "very good"],
            }
        )
        self.rater = ResponseRater(self.df, model_name="all-MiniLM-L6-v2")
        self.test_responses = [
            "I completely agree with this",
            "This is somewhat okay",
            "I'm not sure about this",
            "I disagree with this",
            "This is absolutely wrong",
        ]

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

    def test_get_survey_response_pmf_with_temperature_and_epsilon(self):
        """Test convenience method with temperature and epsilon."""
        # Test with different parameter combinations
        survey_pmf_baseline = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses, temperature=1.0, epsilon=0.0
        )
        survey_pmf_temp = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses, temperature=0.5, epsilon=0.0
        )
        survey_pmf_eps = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses, temperature=1.0, epsilon=0.01
        )
        survey_pmf_both = self.rater.get_survey_response_pmf_by_reference_set_id(
            "set1", self.test_responses, temperature=0.5, epsilon=0.01
        )

        # All should be valid PMFs
        for pmf in [
            survey_pmf_baseline,
            survey_pmf_temp,
            survey_pmf_eps,
            survey_pmf_both,
        ]:
            assert np.isclose(pmf.sum(), 1.0), "PMF should sum to 1"
            assert np.all(pmf >= 0), "PMF should have non-negative values"

        # Different parameters should yield different results
        pmfs = [survey_pmf_baseline, survey_pmf_temp, survey_pmf_eps, survey_pmf_both]
        for i in range(len(pmfs)):
            for j in range(i + 1, len(pmfs)):
                assert not np.allclose(pmfs[i], pmfs[j]), (
                    f"PMFs {i} and {j} should be different"
                )


class TestResponseRaterUtilityMethods:
    """Test utility methods of ResponseRater."""

    def setup_method(self):
        """Set up test data."""
        self.df = po.DataFrame(
            {
                "id": ["set1"] * 5 + ["set2"] * 5,
                "int_response": [1, 2, 3, 4, 5] * 2,
                "sentence": [
                    "strongly disagree",
                    "disagree",
                    "neutral",
                    "agree",
                    "strongly agree",
                    "hate it",
                    "dislike it",
                    "ok",
                    "like it",
                    "love it",
                ],
            }
        )
        self.rater = ResponseRater(self.df, model_name="all-MiniLM-L6-v2")

    def test_encode_texts(self):
        """Test direct text encoding."""
        test_texts = ["This is a test", "Another test sentence"]
        embeddings = self.rater.encode_texts(test_texts)

        assert embeddings.shape[0] == 2, "Should have 2 embeddings"
        assert embeddings.shape[1] > 0, "Should have positive embedding dimension"
        assert isinstance(embeddings, np.ndarray), "Should return numpy array"

    def test_get_reference_sentences(self):
        """Test getting reference sentences."""
        sentences_set1 = self.rater.get_reference_sentences("set1")
        sentences_set2 = self.rater.get_reference_sentences("set2")

        assert len(sentences_set1) == 5, "Set1 should have 5 sentences"
        assert len(sentences_set2) == 5, "Set2 should have 5 sentences"
        assert sentences_set1 != sentences_set2, (
            "Different sets should have different sentences"
        )

        # Check that sentences are in the right order (by int_response)
        expected_set1 = [
            "strongly disagree",
            "disagree",
            "neutral",
            "agree",
            "strongly agree",
        ]
        assert sentences_set1 == expected_set1, "Sentences should be in correct order"

    def test_available_reference_sets(self):
        """Test getting available reference sets."""
        available_sets = self.rater.available_reference_sets

        assert "set1" in available_sets, "Should include set1"
        assert "set2" in available_sets, "Should include set2"
        assert len(available_sets) == 2, "Should have exactly 2 sets"

    def test_model_info(self):
        """Test getting model information."""
        model_info = self.rater.model_info

        required_keys = [
            "model_name",
            "max_seq_length",
            "embedding_dimension",
            "device",
        ]
        for key in required_keys:
            assert key in model_info, f"Model info should contain {key}"

        assert isinstance(model_info["embedding_dimension"], int), (
            "Embedding dimension should be an integer"
        )
        assert model_info["embedding_dimension"] > 0, (
            "Embedding dimension should be positive"
        )


class TestResponseRaterSemanticBehavior:
    """Test that ResponseRater produces semantically reasonable results."""

    def setup_method(self):
        """Set up test data with clear semantic differences."""
        self.df = po.DataFrame(
            {
                "id": ["sentiment"] * 5,
                "int_response": [1, 2, 3, 4, 5],
                "sentence": [
                    "This is terrible and awful",
                    "This is somewhat bad",
                    "This is okay and neutral",
                    "This is pretty good",
                    "This is excellent and amazing",
                ],
            }
        )
        self.rater = ResponseRater(self.df, model_name="all-MiniLM-L6-v2")

    def test_semantic_alignment(self):
        """Test that semantically similar responses get higher probabilities in expected categories."""
        # Positive response should align more with positive reference sentences
        positive_response = ["This is absolutely fantastic and wonderful"]
        positive_pmfs = self.rater.get_response_pmfs("sentiment", positive_response)

        # Should have higher probability mass on the positive end (indices 3, 4)
        positive_mass = (
            positive_pmfs[0][3] + positive_pmfs[0][4]
        )  # "good" + "excellent"
        negative_mass = positive_pmfs[0][0] + positive_pmfs[0][1]  # "terrible" + "bad"
        assert positive_mass > negative_mass, (
            "Positive response should align with positive references"
        )

        # Negative response should align more with negative reference sentences
        negative_response = ["This is horrible and disgusting"]
        negative_pmfs = self.rater.get_response_pmfs("sentiment", negative_response)

        negative_mass = negative_pmfs[0][0] + negative_pmfs[0][1]  # "terrible" + "bad"
        positive_mass = (
            negative_pmfs[0][3] + negative_pmfs[0][4]
        )  # "good" + "excellent"
        assert negative_mass > positive_mass, (
            "Negative response should align with negative references"
        )

    def test_neutral_response(self):
        """Test that neutral responses align with neutral references."""
        neutral_response = ["This is okay and average"]
        neutral_pmfs = self.rater.get_response_pmfs("sentiment", neutral_response)

        # Neutral response should have reasonable distribution (not too extreme)
        # Should not be overwhelmingly in the most extreme categories
        extreme_negative = neutral_pmfs[0][0]  # "terrible" category
        extreme_positive = neutral_pmfs[0][4]  # "excellent" category
        middle_mass = (
            neutral_pmfs[0][1] + neutral_pmfs[0][2] + neutral_pmfs[0][3]
        )  # middle categories

        assert middle_mass > 0.3, (
            "Neutral response should have significant mass in middle categories"
        )
        assert extreme_negative < 0.6, (
            "Neutral response should not be overwhelmingly negative"
        )
        assert extreme_positive < 0.6, (
            "Neutral response should not be overwhelmingly positive"
        )


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
