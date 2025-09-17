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


def _assert_reference_sentence_dataframe_structure(df, embeddings_column=None):
    """
    Validate the structure of a reference sentence dataframe.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing reference sentences and optionally embeddings
    embeddings_column : str, optional
        Name of the column containing embeddings (if provided)

    Raises
    ------
    ValueError
        If the required columns are missing
    AssertionError
        If the response structure is invalid
    """
    required_cols = ["id", "int_response", "sentence"]
    if embeddings_column:
        required_cols.append(embeddings_column)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected reference-sentence data frame to have columns {required_cols}, "
            f"but missing: {missing_cols}. Available columns: {df.columns}"
        )

    agg = df.group_by("id").agg(po.col("int_response")).sort("id")

    assert "mean" not in agg["id"]
    for i, int_resps in zip(agg["id"], agg["int_response"]):
        assert len(int_resps) == 5
        assert all([i + 1 == r for i, r in enumerate(sorted(int_resps))])


class ResponseRater:
    """
    Convert strings into PMFs over similarity with reference sentences.

    Practically, strings may be LLM responses to a survey question, and reference sentences may be
    some textual representations of each step on a Likert scale. In this case, the PMF gives the
    probabilities that each string represents each step on the Likert scale.

    It can work in two modes:

    1. **Embedding mode**: If `df_reference_sentences` contains an "embedding" column,
       it uses those user-supplied embeddings and expects embedding inputs.
    2. **Text mode**: If no "embedding" column is provided, it automatically computes
       embeddings using `sentence-transformers` and expects text inputs.

    Examples
    --------
    **Text mode (automatic embedding computation):**

    >>> import polars as po
    >>> from semantic_similarity_rating import ResponseRater
    >>>
    >>> # Create reference sentences dataframe (no embedding column)
    >>> df = po.DataFrame({
    ...     'id': ['set1'] * 5,
    ...     'int_response': [1, 2, 3, 4, 5],
    ...     'sentence': ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
    ... })
    >>>
    >>> rater = ResponseRater(df)
    >>> llm_responses = ["I totally agree", "Not sure about this"]  # Text input
    >>> pmfs = rater.get_response_pmfs('set1', llm_responses)

    **Embedding mode (pre-computed embeddings):**

    >>> import numpy as np
    >>> # Create reference sentences with embeddings
    >>> df = po.DataFrame({
    ...     'id': ['set1'] * 5,
    ...     'int_response': [1, 2, 3, 4, 5],
    ...     'sentence': ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],
    ...     'embedding': [np.random.rand(384).tolist() for _ in range(5)]
    ... })
    >>>
    >>> rater = ResponseRater(df)
    >>> llm_embeddings = np.random.rand(2, 384)  # Embedding input
    >>> pmfs = rater.get_response_pmfs('set1', llm_embeddings)
    """

    def __init__(
        self,
        df_reference_sentences: po.DataFrame,
        embeddings_column: str = "embedding",
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
    ):
        """
        Initialize the ResponseRater with reference sentences.

        Parameters
        ----------
        df_reference_sentences : polars.DataFrame
            DataFrame containing reference sentences and optionally pre-computed embeddings
        embeddings_column : str, optional
            Name of the column containing embeddings, by default 'embedding'.
            If this column exists, the rater operates in embedding mode.
        model_name : str, optional
            Name of the sentence-transformer model to use (text mode only), by default 'all-MiniLM-L6-v2'
        device : str, optional
            Device to run the model on ('cpu', 'cuda', etc.) (text mode only), by default None (auto-detect)
        """
        df = df_reference_sentences

        # Check if we're in embedding mode or text mode
        self.embedding_mode = embeddings_column in df.columns
        self.embeddings_column = embeddings_column if self.embedding_mode else None

        # Validate dataframe structure
        _assert_reference_sentence_dataframe_structure(df, self.embeddings_column)

        # Initialize sentence transformer model only in text mode
        self.model = None
        if not self.embedding_mode:
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

            if self.embedding_mode:
                # Use pre-computed embeddings
                embeddings = np.array(this_set[self.embeddings_column].to_list())
                M = embeddings.T  # Transpose to match expected format
            else:
                # Compute embeddings for the reference sentences
                embeddings = self.model.encode(sentences)
                M = embeddings.T  # Transpose to match expected format

            self.reference_matrices[sentence_set] = M

    def get_response_pmfs(
        self, reference_set_id, llm_responses, temperature=1.0, epsilon=0.0
    ):
        """
        Convert strings to PMFs using specified reference set.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set to use, or 'mean' to use average across all sets
        llm_responses : list of str or numpy.ndarray
            - In text mode: List of LLM response texts
            - In embedding mode: Matrix of LLM response embeddings (shape: n_responses x embedding_dim)
        temperature : float
            Get scaled pmf With temperature T:
            ``p_new[i] ~ p_old[i]^(1/T)``.
        epsilon : float, optional
            Small regularization parameter to prevent division by zero and add smoothing.
            Default is 0.0 (no regularization).

        Returns
        -------
        numpy.ndarray
            Probability mass functions for each response

        Raises
        ------
        ValueError
            If input type doesn't match the rater's mode (text vs embedding)
        """
        if self.embedding_mode:
            # Embedding mode: expect numpy array of embeddings
            if not isinstance(llm_responses, np.ndarray):
                raise ValueError(
                    "ResponseRater is in embedding mode (dataframe contains 'embedding' column). "
                    "Expected numpy array of embeddings, got: "
                    + str(type(llm_responses))
                )
            llm_response_matrix = llm_responses
        else:
            # Text mode: expect list of strings and compute embeddings
            if not isinstance(llm_responses, (list, tuple)):
                raise ValueError(
                    "ResponseRater is in text mode (no 'embedding' column in dataframe). "
                    "Expected list of text strings, got: " + str(type(llm_responses))
                )
            llm_response_matrix = self.model.encode(llm_responses)

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
        self, reference_set_id, llm_responses, temperature=1.0, epsilon=0.0
    ):
        """
        Get the survey response PMF using a specific reference set.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set to use
        llm_responses : list of str or numpy.ndarray
            - In text mode: List of LLM response texts
            - In embedding mode: Matrix of LLM response embeddings
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
                reference_set_id, llm_responses, temperature, epsilon
            )
        )

    def encode_texts(self, texts):
        """
        Compute embeddings for a list of texts using the loaded model.

        Note: This method is only available in text mode.

        Parameters
        ----------
        texts : list of str
            List of texts to encode

        Returns
        -------
        numpy.ndarray
            Matrix of embeddings, shape (n_texts, embedding_dim)

        Raises
        ------
        ValueError
            If called in embedding mode (no sentence transformer model loaded)
        """
        if self.embedding_mode:
            raise ValueError(
                "encode_texts() is not available in embedding mode. "
                "Embeddings should be pre-computed and provided directly."
            )
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
        Get information about the ResponseRater.

        Returns
        -------
        dict
            Dictionary containing model and mode information
        """
        info = {
            "mode": "embedding" if self.embedding_mode else "text",
            "embedding_dimension": list(self.reference_matrices.values())[0].shape[0]
            if self.reference_matrices
            else "Unknown",
        }

        if not self.embedding_mode and self.model:
            info.update(
                {
                    "model_name": str(self.model),
                    "max_seq_length": getattr(self.model, "max_seq_length", "Unknown"),
                    "device": str(self.model.device),
                }
            )

        return info
