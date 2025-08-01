"""
Module for rating and analyzing text responses against reference sentences using automatic embedding computation.

This module provides functionality to:
- Automatically compute embeddings for text sentences using sentence-transformers
- Validate reference sentence data structure
- Convert LLM text responses to probability distributions
- Calculate survey response PMFs using different reference sets
- Compare responses against mean or specific reference sets

The module is particularly useful for analyzing Likert scale responses from LLMs
by comparing their text against reference sentence text using semantic embeddings.
"""

import numpy as np
import polars as po
from sentence_transformers import SentenceTransformer

from . import compute


def _assert_reference_sentence_dataframe_structure_text(df):
    """
    Validate the structure of a reference sentence dataframe for text input.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing reference sentences (text)

    Raises
    ------
    ValueError
        If the required columns are missing
    AssertionError
        If the response structure is invalid
    """
    if (
        "id" not in df.columns
        or "int_response" not in df.columns
        or "sentence" not in df.columns
    ):
        raise ValueError(
            "Expected reference-sentence data frame to have columns "
            f'"id", "int_response", "sentence", '
            f"but it has columns: {df.columns}"
        )
    agg = df.group_by("id").agg(po.col("int_response")).sort("id")

    assert "mean" not in agg["id"]
    for i, int_resps in zip(agg["id"], agg["int_response"]):
        assert len(int_resps) == 5
        assert all([i + 1 == r for i, r in enumerate(sorted(int_resps))])


class ResponseRater:
    """
    A class for rating and analyzing text responses using automatic embedding computation.

    This class provides methods to convert LLM text responses into probability
    distributions by comparing them against reference sentence text. It automatically
    computes embeddings using sentence-transformers and can use either specific
    reference sets or mean embeddings across all sets.

    Parameters
    ----------
    df_reference_sentences : polars.DataFrame
        DataFrame containing reference sentences (text)
    model_name : str, optional
        Name of the sentence-transformer model to use, by default 'all-MiniLM-L6-v2'
    device : str, optional
        Device to run the model on ('cpu', 'cuda', etc.), by default None (auto-detect)

    Examples
    --------
    >>> import polars as po
    >>> import numpy as np
    >>> from embeddings_similarity_rating import ResponseRater
    >>>
    >>> # Create example reference sentences dataframe
    >>> df = po.DataFrame({
    ...     'id': ['set1', 'set1', 'set1', 'set1', 'set1',
    ...            'set2', 'set2', 'set2', 'set2', 'set2'],
    ...     'int_response': [1, 2, 3, 4, 5] * 2,
    ...     'sentence': ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree',
    ...                  'Disagree a lot', 'Kinda disagree', "Don't know", 'Kinda agree', 'Agree a lot']
    ... })
    >>>
    >>> # Initialize rater
    >>> rater = ResponseRater(df)
    >>>
    >>> # Get PMFs for some LLM text responses
    >>> llm_responses = ["I totally agree", "Not sure about this", "Completely disagree"]
    >>> pmfs = rater.get_response_pmfs('set1', llm_responses)
    >>> survey_pmf = rater.get_survey_response_pmf(pmfs)
    """

    def __init__(
        self,
        df_reference_sentences: po.DataFrame,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
    ):
        """
        Initialize the ResponseRater with reference sentences.

        Parameters
        ----------
        df_reference_sentences : polars.DataFrame
            DataFrame containing reference sentences (text)
        model_name : str, optional
            Name of the sentence-transformer model to use, by default 'all-MiniLM-L6-v2'
        device : str, optional
            Device to run the model on ('cpu', 'cuda', etc.), by default None (auto-detect)
        """
        df = df_reference_sentences

        _assert_reference_sentence_dataframe_structure_text(df)

        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name, device=device)

        # Initialize storage for reference matrices and sentences
        self.reference_matrices = {}
        self.reference_sentences = {"mean": ["1", "2", "3", "4", "5"]}

        # Process each unique sentence set
        unique_sentence_set_ids = df["id"].unique().sort()
        for sentence_set in unique_sentence_set_ids:
            this_set = df.filter(po.col("id") == sentence_set).sort(by="int_response")
            sentences = this_set["sentence"].to_list()

            # Store the actual sentences for reference
            self.reference_sentences[sentence_set] = sentences

            # Compute embeddings for the reference sentences
            embeddings = self.model.encode(sentences)
            M = embeddings.T  # Transpose to match expected format
            self.reference_matrices[sentence_set] = M

    def get_response_pmfs(
        self, reference_set_id, llm_response_texts, temperature=1.0, epsilon=0.0
    ):
        """
        Convert LLM text responses to PMFs using specified reference set.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set to use, or 'mean' to use average across all sets
        llm_response_texts : list of str
            List of LLM response texts
        temperature : float
            Get scaled pmf With temperature T:
            ``p_new[i] ~ p_old[i]^(1/T)``.
        epsilon : float, optional
            Small regularization parameter to prevent division by zero and add smoothing.
            Default is 0.0 (no regularization).

        Returns
        -------
        numpy.ndarray
            Probability distributions for each response
        """
        # Compute embeddings for the response texts
        llm_response_matrix = self.model.encode(llm_response_texts)

        if isinstance(reference_set_id, str) and reference_set_id.lower() == "mean":
            # Calculate PMFs using mean over all reference sets
            llm_response_pmfs = np.array(
                [
                    compute.response_embeddings_to_pmf(llm_response_matrix, M, epsilon)
                    for M in self.reference_matrices.values()
                ]
            ).mean(axis=0)
        else:
            # Calculate PMFs using specific reference set
            M = self.reference_matrices[reference_set_id]
            llm_response_pmfs = compute.response_embeddings_to_pmf(
                llm_response_matrix, M, epsilon
            )

        if temperature != 1.0:
            llm_response_pmfs = np.array(
                [compute.scale_pmf(_pmf, temperature) for _pmf in llm_response_pmfs]
            )

        return llm_response_pmfs

    def get_survey_response_pmf(self, response_pmfs):
        """
        Calculate the overall survey response PMF by averaging individual response PMFs.

        Parameters
        ----------
        response_pmfs : numpy.ndarray
            Matrix of individual response PMFs

        Returns
        -------
        numpy.ndarray
            Average PMF representing the overall survey response
        """
        return response_pmfs.mean(axis=0)

    def get_survey_response_pmf_by_reference_set_id(
        self, reference_set_id, llm_response_texts, temperature=1.0, epsilon=0.0
    ):
        """
        Get the survey response PMF using a specific reference set.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set to use
        llm_response_texts : list of str
            List of LLM response texts
        temperature : float, default = 1.0
            Get scaled pmf With temperature T:
            ``p_new[i] ~ p_old[i]^(1/T)``.
        epsilon : float, optional
            Small regularization parameter to prevent division by zero and add smoothing.
            Default is 0.0 (no regularization).

        Returns
        -------
        numpy.ndarray
            Average PMF representing the overall survey response
        """
        return self.get_survey_response_pmf(
            self.get_response_pmfs(
                reference_set_id, llm_response_texts, temperature, epsilon
            )
        )

    def encode_texts(self, texts):
        """
        Compute embeddings for a list of texts using the loaded model.

        Parameters
        ----------
        texts : list of str
            List of texts to encode

        Returns
        -------
        numpy.ndarray
            Matrix of embeddings, shape (n_texts, embedding_dim)
        """
        return self.model.encode(texts)

    def get_reference_sentences(self, reference_set_id):
        """
        Get the reference sentences for a specific reference set.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set

        Returns
        -------
        list of str
            List of reference sentences
        """
        return self.reference_sentences[reference_set_id]

    @property
    def available_reference_sets(self):
        """
        Get the list of available reference set IDs.

        Returns
        -------
        list of str
            List of available reference set IDs
        """
        return list(self.reference_matrices.keys())

    @property
    def model_info(self):
        """
        Get information about the loaded sentence transformer model.

        Returns
        -------
        dict
            Dictionary containing model information
        """
        return {
            "model_name": str(self.model),
            "max_seq_length": getattr(self.model, "max_seq_length", "Unknown"),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "device": str(self.model.device),
        }
