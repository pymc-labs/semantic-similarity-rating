"""
Module for rating and analyzing embeddings of survey responses against reference sentences.

This module provides functionality to:
- Validate reference sentence data structure
- Convert LLM response embeddings to probability distributions
- Calculate survey response PDFs using different reference sets
- Compare responses against mean or specific reference sets

The module is particularly useful for analyzing Likert scale responses from LLMs
by comparing their embeddings against reference sentence embeddings.
"""

import numpy as np
import polars as po

from . import compute


def _assert_reference_sentence_dataframe_structure(df, embeddings_column):
    """
    Validate the structure of a reference sentence dataframe.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing reference sentences and their embeddings
    embeddings_column : str
        Name of the column containing the embeddings

    Raises
    ------
    ValueError
        If the required columns are missing
    AssertionError
        If the response structure is invalid
    """
    if (
        embeddings_column not in df.columns
        or "id" not in df.columns
        or "int_response" not in df.columns
        or "sentence" not in df.columns
    ):
        raise ValueError(
            "Expected reference-sentence data frame to have columns "
            f' "{embeddings_column}", "id", "int_response", "sentence", '
            f"but it has columns: {df.columns}"
        )
    agg = df.group_by("id").agg(po.col("int_response")).sort("id")

    assert "mean" not in agg["id"]
    for i, int_resps in zip(agg["id"], agg["int_response"]):
        assert len(int_resps) == 5
        assert all([i + 1 == r for i, r in enumerate(sorted(int_resps))])


class EmbeddingsRater:
    """
    A class for rating and analyzing embeddings of survey responses.

    This class provides methods to convert LLM response embeddings into probability
    distributions by comparing them against reference sentence embeddings. It can
    use either specific reference sets or mean embeddings across all sets.

    Parameters
    ----------
    df_reference_sentences : polars.DataFrame
        DataFrame containing reference sentences and their embeddings
    embeddings_column : str, optional
        Name of the column containing the embeddings, by default 'embedding_small'

    Examples
    --------
    >>> import polars as po
    >>> import numpy as np
    >>>
    >>> # Create example reference sentences dataframe
    >>> df = po.DataFrame({
    ...     'id': ['set1', 'set1', 'set1', 'set1', 'set1',
    ...            'set2', 'set2', 'set2', 'set2', 'set2'],
    ...     'int_response': [1, 2, 3, 4, 5] * 2,
    ...     'sentence': [ 'Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree',
    ...                   'Disagree a lot', 'Kinda disagree', "Don't know", 'Kinda agree', 'Agree a lot'
    ...                 ],
    ...     'embedding_small': [np.random.rand(384).tolist() for _ in range(10)]
    ... })
    >>>
    >>> # Initialize rater
    >>> rater = EmbeddingsRater(df, embeddings_column='embedding_small')
    >>>
    >>> # Get PDFs for some LLM responses
    >>> llm_responses = np.random.rand(5, 384)  # 5 responses, each with 384-dim embedding
    >>> pdfs = rater.get_response_pdfs('set1', llm_responses)
    >>> survey_pdf = rater.get_survey_response_pdf(pdfs)
    """

    def __init__(
        self,
        df_reference_sentences: po.DataFrame,
        embeddings_column: str = "embedding_small",
    ):
        """
        Initialize the EmbeddingsRater with reference sentences.

        Parameters
        ----------
        df_reference_sentences : polars.DataFrame
            DataFrame containing reference sentences and their embeddings
        embeddings_column : str, optional
            Name of the column containing the embeddings, by default 'embedding_small'
        """
        df = df_reference_sentences

        _assert_reference_sentence_dataframe_structure(df, embeddings_column)

        # Initialize storage for reference matrices and sentences
        self.reference_matrices = {}
        self.reference_sentences = {"mean": ["1", "2", "3", "4", "5"]}

        # Process each unique sentence set
        unique_sentence_set_ids = df["id"].unique().sort()
        for sentence_set in unique_sentence_set_ids:
            this_set = df.filter(po.col("id") == sentence_set).sort(by="int_response")
            M = np.array(this_set[embeddings_column].to_list()).T
            self.reference_matrices[sentence_set] = M

    def get_response_pdfs(self, reference_set_id, llm_response_matrix, temperature=1.0):
        """
        Convert LLM response embeddings to PDFs using specified reference set.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set to use, or 'mean' to use average across all sets
        llm_response_matrix : numpy.ndarray
            Matrix of LLM response embeddings
            Shape: (n_responses, n_dimensions)
        temperature : float
            Get scaled pdf With temperature T:
            ``p_new[i] ~ p_old[i]^(1/T)``.

        Returns
        -------
        numpy.ndarray
            Probability distributions for each response
        """
        if isinstance(reference_set_id, str) and reference_set_id.lower() == "mean":
            # Calculate PDFs using mean over all reference sets
            llm_response_pdfs = np.array(
                [
                    compute.response_embeddings_to_pdf(llm_response_matrix, M)
                    for M in self.reference_matrices.values()
                ]
            ).mean(axis=0)
        else:
            # Calculate PDFs using specific reference set
            M = self.reference_matrices[reference_set_id]
            llm_response_pdfs = compute.response_embeddings_to_pdf(
                llm_response_matrix, M
            )

        if temperature != 1.0:
            llm_response_pdfs = np.array(
                [compute.scale_pdf(_pdf, temperature) for _pdf in llm_response_pdfs]
            )

        return llm_response_pdfs

    def get_survey_response_pdf(self, response_pdfs):
        """
        Calculate the overall survey response PDF by averaging individual response PDFs.

        Parameters
        ----------
        response_pdfs : numpy.ndarray
            Matrix of individual response PDFs

        Returns
        -------
        numpy.ndarray
            Average PDF representing the overall survey response
        """
        return response_pdfs.mean(axis=0)

    def get_survey_response_pdf_by_reference_set_id(
        self, reference_set_id, llm_response_matrix, temperature=1.0
    ):
        """
        Get the survey response PDF using a specific reference set.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set to use
        llm_response_matrix : numpy.ndarray
            Matrix of LLM response embeddings
            Shape: (n_responses, n_dimensions)
        temperature : float, default = 1.0
            Get scaled pdf With temperature T:
            ``p_new[i] ~ p_old[i]^(1/T)``.

        Returns
        -------
        numpy.ndarray
            Average PDF representing the overall survey response
        """
        return self.get_survey_response_pdf(
            self.get_response_pdfs(reference_set_id, llm_response_matrix)
        )
