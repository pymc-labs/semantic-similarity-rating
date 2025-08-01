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

Examples
--------
>>> x = np.arange(1,6)
>>> pdf = np.array([0.1,0.15,0.05,0.2,0.5])
>>> real_mean = 3.0
>>> T, scaled_pdf = get_optimal_temperature_mean(x, pdf, real_mean)
"""

import numpy as np
from scipy.optimize import minimize


def cos_to_pdf(cos):
    """
    Convert cosine similarities to a probability density function (PDF).

    Parameters
    ----------
    cos : array_like
        Array of cosine similarity values

    Returns
    -------
    numpy.ndarray
        Normalized PDF where all values sum to 1
    """
    hist = np.array(cos) - np.min(cos)
    return hist / hist.sum()


def cos_sim(emb1, emb2):
    """
    Calculate cosine similarity between two embeddings.

    Parameters
    ----------
    emb1 : array_like
        First embedding vector
    emb2 : array_like
        Second embedding vector

    Returns
    -------
    float
        Cosine similarity score between 0 and 1
    """
    return (1 + cos_sim_pdf(emb1, emb2)) / 2


def scale_pdf(pdf, temperature, max_temp=10):
    """
    Scale a PDF using temperature scaling.

    Parameters
    ----------
    pdf : array_like
        Input probability density function
    temperature : float
        Temperature parameter for scaling (0 to max_temp)
    max_temp : float, optional
        Maximum temperature value, by default 10

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


def scale_pdf_no_max_temp(pdf, temperature):
    """Calls ``scale_pdf(pdf, temperature, max_temp=np.inf)``"""
    return scale_pdf(pdf, temperature, max_temp=np.inf)


def cos_sim_pdf(pdf1, pdf2):
    """
    Calculate cosine similarity between two PDFs.

    Parameters
    ----------
    pdf1 : array_like
        First probability density function
    pdf2 : array_like
        Second probability density function

    Returns
    -------
    float
        Cosine similarity between the PDFs
    """
    return pdf1.dot(pdf2) / np.linalg.norm(pdf1) / np.linalg.norm(pdf2)


def KS_sim_pdf(pdf1, pdf2):
    """
    Calculate Kolmogorov-Smirnov similarity between two PDFs.

    Parameters
    ----------
    pdf1 : array_like
        First probability density function
    pdf2 : array_like
        Second probability density function

    Returns
    -------
    float
        KS similarity score between 0 and 1
    """
    return 1 - np.max(np.abs(np.cumsum(pdf1) - np.cumsum(pdf2)))


def pdf_moment(pdf, x, m):
    """
    Calculate the m-th moment of a PDF.

    Parameters
    ----------
    pdf : array_like
        Probability density function
    x : array_like
        Values corresponding to the PDF
    m : int
        Order of the moment to calculate

    Returns
    -------
    float
        The m-th moment of the PDF
    """
    return pdf.dot(x**m)


def mean(pdf, x):
    """
    Calculate the mean of a PDF.

    Parameters
    ----------
    pdf : array_like
        Probability density function
    x : array_like
        Values corresponding to the PDF

    Returns
    -------
    float
        Mean value of the PDF
    """
    return pdf_moment(pdf, x, m=1)


def var(pdf, x):
    """
    Calculate the variance of a PDF.

    Parameters
    ----------
    pdf : array_like
        Probability density function
    x : array_like
        Values corresponding to the PDF

    Returns
    -------
    float
        Variance of the PDF
    """
    _x_ = mean(pdf, x)
    _x2_ = pdf_moment(pdf, x, m=2)
    return _x2_ - _x_**2


def std(pdf, x):
    """
    Calculate the standard deviation of a PDF.

    Parameters
    ----------
    pdf : array_like
        Probability density function
    x : array_like
        Values corresponding to the PDF

    Returns
    -------
    float
        Standard deviation of the PDF
    """
    return np.sqrt(var(pdf, x))


def get_optimal_temperature_mean(x, pdf, real_mean):
    """
    Find the optimal temperature that matches the mean of a scaled PDF to a target mean.

    Parameters
    ----------
    x : array_like
        Values corresponding to the PDF
    pdf : array_like
        Input probability density function
    real_mean : float
        Target mean value

    Returns
    -------
    tuple
        (optimal_temperature, scaled_pdf)
    """

    def _obj(T):
        return (mean(scale_pdf(pdf, T), x) - real_mean) ** 2

    T0 = 1.0
    res = minimize(_obj, T0, bounds=[(0, 10.0)])

    T = res.x[0]
    pdf = scale_pdf(pdf, T)
    return T, pdf


def get_optimal_temperature_KS_sim(pdf, real_pdf):
    """
    Find the optimal temperature that maximizes KS similarity between scaled PDF and target PDF.

    Parameters
    ----------
    pdf : array_like
        Input probability density function
    real_pdf : array_like
        Target probability density function

    Returns
    -------
    tuple
        (optimal_temperature, scaled_pdf)
    """

    def _obj(T):
        return -KS_sim_pdf(scale_pdf(pdf, T), real_pdf)

    T0 = 1.0
    res = minimize(_obj, T0, bounds=[(0, 10.0)])

    T = res.x[0]
    pdf = scale_pdf(pdf, T)
    return T, pdf


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


if __name__ == "__main__":
    # Example usage with test data
    x = np.arange(1, 6)
    pdf = np.array([0.1, 0.15, 0.05, 0.2, 0.5])
    realpdf = np.array([0.1, 0.15, 0.5, 0.15, 0.1])
    real_mean = 3.0
    print(get_optimal_temperature_mean(x, pdf, real_mean))
    print(get_optimal_temperature_KS_sim(pdf, real_mean))
