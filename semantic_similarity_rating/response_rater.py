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

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class DecisionModel(Protocol):
    """
    Any model that can give choice probabilities over a list of options.

    Implement ``choice_probabilities`` with the model and prompt of your choice.
    Each context is whatever your model needs for one respondent, for example a
    respondent profile, a product description and the survey question. Ask the
    model which option this respondent would most likely say in reply.
    """

    def choice_probabilities(
        self, contexts: Sequence[Any], options: Sequence[str]
    ) -> np.ndarray:
        """Return an array of shape ``(len(contexts), len(options))`` whose rows sum to one."""
        ...


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
            llm_response_pmfs = compute.scale_pmfs(llm_response_pmfs, temperature)

        return llm_response_pmfs

    def compute_response_similarities(self, llm_responses):
        """
        Precompute per-reference-set cosine similarity statistics for responses.

        Use this together with :meth:`pmfs_from_similarities` when sweeping over
        a grid of ``(temperature, epsilon)`` values — the embedding encoding,
        normalization, and matmul run once; the inner loop becomes pure
        elementwise arithmetic on the cached row-wise reductions.

        Parameters
        ----------
        llm_responses : list of str or numpy.ndarray
            Text mode: list of strings. Embedding mode: ``(n_responses, dim)`` array.

        Returns
        -------
        dict
            Maps reference-set id (str) to a stats dict from
            :func:`compute.precompute_similarity_stats`. Keys in the dict are
            ``cos`` (``(n, 5)``), ``cos_min`` (``(n, 1)``), ``cos_sum`` (``(n, 1)``),
            and ``min_indices`` (``(n,)``).
        """
        if self.embedding_mode:
            if not isinstance(llm_responses, np.ndarray):
                raise ValueError(
                    "ResponseRater is in embedding mode (dataframe contains 'embedding' column). "
                    "Expected numpy array of embeddings, got: "
                    + str(type(llm_responses))
                )
            llm_response_matrix = llm_responses
        else:
            if not isinstance(llm_responses, (list, tuple)):
                raise ValueError(
                    "ResponseRater is in text mode (no 'embedding' column in dataframe). "
                    "Expected list of text strings, got: " + str(type(llm_responses))
                )
            llm_response_matrix = self.model.encode(llm_responses)

        stats_by_ref = {}
        for ref_id, M in self.reference_matrices.items():
            cos = compute.cosine_similarity_matrix(llm_response_matrix, M)
            stats_by_ref[ref_id] = compute.precompute_similarity_stats(cos)
        return stats_by_ref

    def pmfs_from_similarities(
        self, reference_set_id, similarity_stats, temperature=1.0, epsilon=0.0
    ):
        """
        Map precomputed similarity stats to PMFs — no encoding, no matmul.

        Parameters
        ----------
        reference_set_id : str
            Reference set id, or ``'mean'`` to average PMFs across all sets.
        similarity_stats : dict
            Output of :meth:`compute_response_similarities`.
        temperature : float, default 1.0
            Temperature for ``scale_pmfs`` (applied if != 1.0).
        epsilon : float, default 0.0
            Regularization passed to ``similarities_to_pmf``.

        Returns
        -------
        numpy.ndarray, shape (n_responses, n_likert_points)
        """
        if isinstance(reference_set_id, str) and reference_set_id.lower() == "mean":
            pmfs = np.array(
                [
                    compute.similarities_to_pmf(stats, epsilon)
                    for stats in similarity_stats.values()
                ]
            ).mean(axis=0)
        else:
            pmfs = compute.similarities_to_pmf(
                similarity_stats[reference_set_id], epsilon
            )

        if temperature != 1.0:
            pmfs = compute.scale_pmfs(pmfs, temperature)
        return pmfs

    def get_anchor_kernel(self, reference_set_id, temperature=1.0, epsilon=0.0):
        """
        Score each reference sentence of a set as if it were a response.

        Row ``j`` is the PMF that SSR returns for a response equal to reference
        sentence ``j``. The kernel depends only on the reference sentences, their
        embeddings and the SSR settings, never on survey responses.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set.
        temperature : float, default 1.0
            Temperature for ``scale_pmfs``, as in :meth:`get_response_pmfs`.
        epsilon : float, default 0.0
            Regularization, as in :meth:`get_response_pmfs`.

        Returns
        -------
        numpy.ndarray, shape (n_likert_points, n_likert_points)
        """
        M = self.reference_matrices[reference_set_id]
        kernel = compute.response_embeddings_to_pmf(M.T, M, epsilon)
        if temperature != 1.0:
            kernel = compute.scale_pmfs(kernel, temperature)
        return kernel

    def get_choice_pmfs(
        self, reference_set_id, choice_probabilities, temperature=1.0, epsilon=0.0
    ):
        """
        Convert probabilities over reference sentences into SSR PMFs.

        Use this when a model returns a probability for each reference sentence
        directly, for example a decision model asked which sentence a respondent
        would most likely say, instead of a free-text response. Each choice is
        scored with the same SSR mapping a free-text response equal to that
        sentence would get, so the output is on the same scale as
        :meth:`get_response_pmfs`. For one set, before temperature scaling:

        ``pmf = choice_probabilities @ get_anchor_kernel(reference_set_id)``

        As in :meth:`get_response_pmfs`, ``'mean'`` averages the per-set PMFs
        first, and temperature scaling is applied once at the end. A certain
        choice of sentence j therefore gives exactly the PMF that
        :meth:`get_response_pmfs` gives sentence j as a response.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set, or ``'mean'`` to average over all sets.
        choice_probabilities : numpy.ndarray or Mapping
            For one set: array of shape ``(n_responses, n_likert_points)`` whose
            columns follow the set's ``int_response`` order. For ``'mean'``: a
            mapping from every reference set ID to such an array, all with the
            same number of rows.
        temperature : float, default 1.0
            Temperature for ``scale_pmfs``, applied after any averaging.
        epsilon : float, default 0.0
            Regularization used for the kernel.

        Returns
        -------
        numpy.ndarray, shape (n_responses, n_likert_points)

        Raises
        ------
        ValueError
            If a probability array has the wrong shape, negative or non-finite
            values, or rows that do not sum to one, or if ``'mean'`` input does not
            cover exactly the reference sets or has arrays with different row counts.
        """
        if isinstance(reference_set_id, str) and reference_set_id.lower() == "mean":
            if not isinstance(choice_probabilities, Mapping) or set(
                choice_probabilities
            ) != set(self.reference_matrices):
                raise ValueError(
                    "For 'mean', choice_probabilities must map every reference set ID "
                    "to an array of choice probabilities."
                )
            per_set = [
                self._unscaled_choice_pmfs(set_id, probs, epsilon)
                for set_id, probs in choice_probabilities.items()
            ]
            if len({pmfs.shape[0] for pmfs in per_set}) != 1:
                raise ValueError(
                    "For 'mean', every set needs the same number of rows of choice probabilities."
                )
            pmfs = np.array(per_set).mean(axis=0)
        else:
            pmfs = self._unscaled_choice_pmfs(
                reference_set_id, choice_probabilities, epsilon
            )

        if temperature != 1.0:
            pmfs = compute.scale_pmfs(pmfs, temperature)
        return pmfs

    def _unscaled_choice_pmfs(self, reference_set_id, choice_probabilities, epsilon):
        kernel = self.get_anchor_kernel(reference_set_id, epsilon=epsilon)
        probs = np.asarray(choice_probabilities, dtype=float)
        if probs.ndim != 2 or probs.shape[1] != kernel.shape[0]:
            raise ValueError(
                f"Expected choice probabilities of shape (n_responses, {kernel.shape[0]}), "
                f"got {probs.shape}."
            )
        if (
            not np.all(np.isfinite(probs))
            or np.any(probs < 0)
            or not np.allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
        ):
            raise ValueError(
                "Each row of choice probabilities must be finite, non-negative and sum to one."
            )
        return probs @ kernel

    def get_decision_model_pmfs(
        self, reference_set_id, contexts, model, temperature=1.0, epsilon=0.0
    ):
        """
        Ask a decision model to choose among the reference sentences, then score the choices.

        The model is called once per reference set, with every context and a copy
        of that set's sentences in ``int_response`` order as the options. Its
        choice probabilities go through :meth:`get_choice_pmfs`.

        Parameters
        ----------
        reference_set_id : str
            ID of the reference set, or ``'mean'`` to use and average all sets.
        contexts : sequence
            One item per respondent, passed to the model unchanged.
        model : DecisionModel
            The decision model to use.
        temperature : float, default 1.0
            Passed to :meth:`get_choice_pmfs`.
        epsilon : float, default 0.0
            Passed to :meth:`get_choice_pmfs`.

        Returns
        -------
        numpy.ndarray, shape (len(contexts), n_likert_points)

        Raises
        ------
        TypeError
            If ``model`` has no callable ``choice_probabilities`` method.
        ValueError
            If the model returns a number of rows other than ``len(contexts)``, or
            probabilities that :meth:`get_choice_pmfs` rejects.
        """
        if not isinstance(model, DecisionModel) or not callable(
            model.choice_probabilities
        ):
            raise TypeError(
                "model must implement choice_probabilities(contexts, options)."
            )
        mean = isinstance(reference_set_id, str) and reference_set_id.lower() == "mean"
        set_ids = list(self.reference_matrices) if mean else [reference_set_id]
        probabilities = {}
        for set_id in set_ids:
            probs = np.asarray(
                model.choice_probabilities(
                    contexts, list(self.reference_sentences[set_id])
                ),
                dtype=float,
            )
            if probs.ndim != 2 or probs.shape[0] != len(contexts):
                raise ValueError(
                    f"The model returned {probs.shape[0] if probs.ndim else 0} rows "
                    f"for {len(contexts)} contexts."
                )
            probabilities[set_id] = probs
        if mean:
            return self.get_choice_pmfs("mean", probabilities, temperature, epsilon)
        return self.get_choice_pmfs(
            reference_set_id, probabilities[reference_set_id], temperature, epsilon
        )

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
