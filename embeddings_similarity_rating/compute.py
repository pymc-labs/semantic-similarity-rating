"""
Utility functions for computing and manipulating probability density functions (PDFs) and embeddings.

This module provides functions for:
- Converting between different similarity metrics (cosine, KS)
- Scaling PDFs using temperature parameters
- Computing statistical moments of PDFs
- Finding optimal temperature parameters for PDF scaling
- Converting response embeddings to PDFs

The module is particularly useful for working with Likert scale responses and their
embeddings, providing tools to analyze and transform the underlying probability
distributions.
"""

import numpy as np


def scale_pdf(pdf, temperature, max_temp=np.inf):
    """
    Scale a PDF using temperature scaling.

    Parameters
    ----------
    pdf : array_like
        Input probability density function
    temperature : float
        Temperature parameter for scaling (0 to max_temp)
    max_temp : float, optional
        Maximum temperature value, by default np.inf

    Returns
    -------
    numpy.ndarray
        Scaled PDF where all values sum to 1

    Notes
    -----
    - If temperature is 0, returns a one-hot vector at the maximum probability
    - If temperature > max_temp, uses max_temp for scaling
    - Otherwise uses the specified temperature for scaling
    """
    if temperature == 0.0:
        if np.all(pdf == pdf[0]):
            return pdf
        else:
            new_pdf = np.zeros_like(pdf)
            new_pdf[np.argmax(pdf)] = 1.0
            return new_pdf
    elif temperature > max_temp:
        hist = pdf ** (1 / max_temp)
    else:
        hist = pdf ** (1 / temperature)
    return hist / hist.sum()


def response_embeddings_to_pdf(matrix_responses, matrix_likert_sentences):
    """
    Convert response embeddings and Likert sentence embeddings to a PDF.

    Parameters
    ----------
    matrix_responses : array_like
        Matrix of response embeddings
    matrix_likert_sentences : array_like
        Matrix of Likert sentence embeddings

    Returns
    -------
    numpy.ndarray
        Probability density function representing the response distribution
    """
    M_left = matrix_responses
    M_right = matrix_likert_sentences

    # Normalize the right matrix (Likert sentences)
    norm_right = np.linalg.norm(M_right, axis=0)
    M_right = M_right / norm_right[None, :]

    # Normalize the left matrix (responses)
    norm_left = np.linalg.norm(M_left, axis=1)
    M_left = M_left / norm_left[:, None]

    # Calculate cosine similarities and convert to PDF
    cos = (1 + M_left.dot(M_right)) / 2
    cos = cos - cos.min(axis=1)[:, None]
    sum_per_row = cos.sum(axis=1)
    pdf = cos / sum_per_row[:, None]

    return pdf
