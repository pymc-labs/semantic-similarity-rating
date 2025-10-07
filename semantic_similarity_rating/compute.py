"""
Utility functions for computing and manipulating probability density functions (PMFs) and embeddings.

This module provides functions for:
- Converting between different similarity metrics (cosine, KS)
- Scaling PMFs using temperature parameters
- Computing statistical moments of PMFs
- Finding optimal temperature parameters for PMF scaling
- Converting response embeddings to PMFs

The module is particularly useful for working with Likert scale responses and their
embeddings, providing tools to analyze and transform the underlying probability
distributions.
"""

import numpy as np


def scale_pmf(pmf, temperature, max_temp=np.inf):
    """
    Scale a PMF using temperature scaling.

    Parameters
    ----------
    pmf : array_like
        Input probability density function
    temperature : float
        Temperature parameter for scaling (0 to max_temp)
    max_temp : float, optional
        Maximum temperature value, by default np.inf

    Returns
    -------
    numpy.ndarray
        Scaled PMF where all values sum to 1

    Notes
    -----
    - If temperature is 0, returns a one-hot vector at the maximum probability
    - If temperature > max_temp, uses max_temp for scaling
    - Otherwise uses the specified temperature for scaling
    """
    if temperature == 0.0:
        if np.all(pmf == pmf[0]):
            return pmf
        else:
            new_pmf = np.zeros_like(pmf)
            new_pmf[np.argmax(pmf)] = 1.0
            return new_pmf
    elif temperature > max_temp:
        hist = pmf ** (1 / max_temp)
    else:
        hist = pmf ** (1 / temperature)
    return hist / hist.sum()


def response_embeddings_to_pmf(matrix_responses, matrix_likert_sentences, epsilon=0.0):
    """
    Convert response embeddings and Likert sentence embeddings to a PMF.

    Parameters
    ----------
    matrix_responses : array_like
        Matrix of response embeddings
    matrix_likert_sentences : array_like
        Matrix of Likert sentence embeddings
    epsilon : float, optional
        Small regularization parameter to prevent division by zero and add smoothing.
        Default is 0.0 (no regularization).

    Returns
    -------
    numpy.ndarray
        Probability density function representing the response distribution

    Notes
    -----
    This implements the SSR equation:
    p_{c,i}(r) = [γ(σ_{r,i}, t_c̃) - γ(σ_ℓ,i, t_c̃) + ε δ_ℓ,r] /
                 [Σ_r γ(σ_{r,i}, t_c̃) - n_points * γ(σ_ℓ,i, t_c̃) + ε]
    where γ is the cosine similarity function, δ_ℓ,r is the Kronecker delta,
    and n_points is the number of Likert scale points.
    """
    M_left = matrix_responses
    M_right = matrix_likert_sentences

    # Handle empty input case
    if M_left.shape[0] == 0:
        return np.empty((0, M_right.shape[1]))

    # Normalize the right matrix (Likert sentences)
    norm_right = np.linalg.norm(M_right, axis=0)
    M_right = M_right / norm_right[None, :]

    # Normalize the left matrix (responses)
    norm_left = np.linalg.norm(M_left, axis=1)
    M_left = M_left / norm_left[:, None]

    # Calculate cosine similarities: γ(σ_{r,i}, t_c̃)
    cos = (1 + M_left.dot(M_right)) / 2

    # Find minimum similarity per row: γ(σ_ℓ,i, t_c̃)
    cos_min = cos.min(axis=1)[:, None]

    # Numerator: γ(σ_{r,i}, t_c̃) - γ(σ_ℓ,i, t_c̃) + ε δ_ℓ,r
    # The ε δ_ℓ,r term adds epsilon only to exactly one minimum similarity position per row
    numerator = cos - cos_min
    if epsilon > 0:
        # Add epsilon to the first position that achieves minimum in each row (Kronecker delta effect)
        min_indices = np.argmin(cos, axis=1)
        for i, min_idx in enumerate(min_indices):
            numerator[i, min_idx] += epsilon

    # Denominator: Σ_r γ(σ_{r,i}, t_c̃) - n_likert_points * γ(σ_ℓ,i, t_c̃) + ε
    # This is: sum of all similarities - n_likert_points * minimum similarity + epsilon
    n_likert_points = cos.shape[1]
    denominator = cos.sum(axis=1)[:, None] - n_likert_points * cos_min + epsilon

    # Calculate final PMF
    pmf = numerator / denominator

    return pmf
