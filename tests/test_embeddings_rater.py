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


class TestEmbeddingsRaterResponsePDFs:
    """Test the get_response_pdfs method."""

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

    def test_get_response_pdfs_specific_set(self):
        """Test getting PDFs for a specific reference set."""
        pdfs = self.rater.get_response_pdfs("set1", self.test_responses)

        # Check output shape
        assert pdfs.shape == (3, 5), "Should return 3 responses x 5 Likert points"

        # Check that each row is a valid PDF
        for i in range(3):
            assert np.isclose(pdfs[i].sum(), 1.0), f"Row {i} should sum to 1"
            assert np.all(pdfs[i] >= 0), f"Row {i} should have non-negative values"

    def test_get_response_pdfs_mean_set(self):
        """Test getting PDFs using mean across all reference sets."""
        pdfs = self.rater.get_response_pdfs("mean", self.test_responses)

        # Check output shape
        assert pdfs.shape == (3, 5), "Should return 3 responses x 5 Likert points"

        # Check that each row is a valid PDF
        for i in range(3):
            assert np.isclose(pdfs[i].sum(), 1.0), f"Row {i} should sum to 1"
            assert np.all(pdfs[i] >= 0), f"Row {i} should have non-negative values"

    def test_get_response_pdfs_with_temperature(self):
        """Test getting PDFs with temperature scaling."""
        # Test with temperature < 1 (sharper)
        pdfs_sharp = self.rater.get_response_pdfs(
            "set1", self.test_responses, temperature=0.5
        )

        # Test with temperature > 1 (smoother)
        pdfs_smooth = self.rater.get_response_pdfs(
            "set1", self.test_responses, temperature=2.0
        )

        # Test with temperature = 1 (baseline)
        pdfs_normal = self.rater.get_response_pdfs(
            "set1", self.test_responses, temperature=1.0
        )

        # All should be valid PDFs
        for pdfs in [pdfs_sharp, pdfs_smooth, pdfs_normal]:
            assert pdfs.shape == (3, 5), "Should have correct shape"
            for i in range(3):
                assert np.isclose(pdfs[i].sum(), 1.0), f"Row {i} should sum to 1"
                assert np.all(pdfs[i] >= 0), f"Row {i} should have non-negative values"

        # Sharp distribution should be more peaked than normal
        # (higher maximum values)
        for i in range(3):
            assert pdfs_sharp[i].max() >= pdfs_normal[i].max(), (
                "Sharp should be more peaked"
            )

    def test_get_response_pdfs_invalid_set(self):
        """Test that invalid reference set raises error."""
        with pytest.raises(KeyError):
            self.rater.get_response_pdfs("nonexistent_set", self.test_responses)


class TestEmbeddingsRaterSurveyPDFs:
    """Test survey-level PDF methods."""

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

    def test_get_survey_response_pdf(self):
        """Test aggregating individual PDFs to survey-level PDF."""
        # Get individual PDFs
        individual_pdfs = self.rater.get_response_pdfs("set1", self.test_responses)

        # Get survey PDF
        survey_pdf = self.rater.get_survey_response_pdf(individual_pdfs)

        # Check that result is valid PDF
        assert survey_pdf.shape == (5,), "Survey PDF should have 5 elements"
        assert np.isclose(survey_pdf.sum(), 1.0), "Survey PDF should sum to 1"
        assert np.all(survey_pdf >= 0), "Survey PDF should have non-negative values"

        # Check that survey PDF is the mean of individual PDFs
        expected = individual_pdfs.mean(axis=0)
        assert np.allclose(survey_pdf, expected), "Should be mean of individual PDFs"

    def test_get_survey_response_pdf_by_reference_set_id(self):
        """Test convenience method for getting survey PDF."""
        # Test convenience method
        survey_pdf_conv = self.rater.get_survey_response_pdf_by_reference_set_id(
            "set1", self.test_responses
        )

        # Test manual approach
        individual_pdfs = self.rater.get_response_pdfs("set1", self.test_responses)
        survey_pdf_manual = self.rater.get_survey_response_pdf(individual_pdfs)

        # Should be identical
        assert np.allclose(survey_pdf_conv, survey_pdf_manual), (
            "Convenience method should match manual approach"
        )

        # Check that result is valid PDF
        assert np.isclose(survey_pdf_conv.sum(), 1.0), "Survey PDF should sum to 1"
        assert np.all(survey_pdf_conv >= 0), (
            "Survey PDF should have non-negative values"
        )

    def test_get_survey_response_pdf_with_temperature(self):
        """Test convenience method with temperature scaling."""
        # Test with different temperatures
        survey_pdf_normal = self.rater.get_survey_response_pdf_by_reference_set_id(
            "set1", self.test_responses, temperature=1.0
        )
        survey_pdf_sharp = self.rater.get_survey_response_pdf_by_reference_set_id(
            "set1", self.test_responses, temperature=0.5
        )
        survey_pdf_smooth = self.rater.get_survey_response_pdf_by_reference_set_id(
            "set1", self.test_responses, temperature=2.0
        )

        # All should be valid PDFs
        for pdf in [survey_pdf_normal, survey_pdf_sharp, survey_pdf_smooth]:
            assert np.isclose(pdf.sum(), 1.0), "PDF should sum to 1"
            assert np.all(pdf >= 0), "PDF should have non-negative values"

        # Sharp should be more peaked
        assert survey_pdf_sharp.max() >= survey_pdf_normal.max(), (
            "Sharp should be more peaked"
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
        pdfs = rater.get_response_pdfs("set1", single_response)

        assert pdfs.shape == (1, 5), "Should handle single response"
        assert np.isclose(pdfs[0].sum(), 1.0), "PDF should sum to 1"

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
        pdfs = rater.get_response_pdfs("set1", large_responses)

        assert pdfs.shape == (100, 5), "Should handle large number of responses"

        # Check that all PDFs are valid
        for i in range(100):
            assert np.isclose(pdfs[i].sum(), 1.0), f"PDF {i} should sum to 1"
            assert np.all(pdfs[i] >= 0), f"PDF {i} should have non-negative values"

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
        pdfs = rater.get_response_pdfs("set1", responses)

        assert pdfs.shape == (3, 5), "Should work with different embedding dimensions"
        for i in range(3):
            assert np.isclose(pdfs[i].sum(), 1.0), f"PDF {i} should sum to 1"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
