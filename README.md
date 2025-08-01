# Embeddings-Similarity Rating (ESR)

A Python package implementing the Embeddings-Similarity Rating methodology for converting LLM textual responses to Likert scale probability distributions using semantic similarity against reference statements.

## Overview

The ESR methodology addresses the challenge of mapping rich textual responses from Large Language Models (LLMs) to structured Likert scale ratings. Instead of forcing a single numerical rating, ESR preserves the inherent uncertainty and nuance in textual responses by generating probability distributions over all possible Likert scale points.

This package provides a distilled, reusable implementation of the ESR methodology described in the paper "Measuring Synthetic Consumer Purchase Intent Using Embeddings-Similarity Ratings" by Maier & Aslak (2025).

## Installation

### Local Development
To install this package locally for development, run:
```bash
pip install -e .
```

### From GitHub Repository
To install this package into your own project from GitHub, run:
```bash
pip install git+https://github.com/pymc-labs/embeddings-similarity-rating.git
```

## Quick Start

```python
import numpy as np
import polars as po
from embeddings_similarity_rating import EmbeddingsRater

# Create reference sentences with embeddings
reference_data = po.DataFrame({
    'id': ['set1'] * 5,
    'int_response': [1, 2, 3, 4, 5],
    'sentence': [
        "It's very unlikely that I'd buy it.",
        "It's unlikely that I'd buy it.",
        "I might buy it or not. I don't know.",
        "It's somewhat possible I'd buy it.",
        "It's possible I'd buy it."
    ],
    'embedding_small': [np.random.rand(384).tolist() for _ in range(5)]
})

# Initialize the rater
rater = EmbeddingsRater(reference_data, embeddings_column='embedding_small')

# Convert LLM response embeddings to probability distributions
llm_responses = np.random.rand(10, 384)
pdfs = rater.get_response_pdfs('set1', llm_responses)

# Get overall survey distribution
survey_pdf = rater.get_survey_response_pdf(pdfs)
print(f"Survey distribution: {survey_pdf}")
```

## Methodology

The ESR methodology works by:
1. Defining reference statements for each Likert scale point
2. Computing cosine similarities between LLM response embeddings and reference statement embeddings
3. Converting similarities to probability distributions using minimum similarity subtraction and normalization
4. Optionally applying temperature scaling for distribution control

## Core Components

- `EmbeddingsRater`: Main class implementing the ESR methodology
- `response_embeddings_to_pdf()`: Core function for similarity-to-probability conversion
- `scale_pdf()` and `scale_pdf_no_max_temp()`: Temperature scaling functions

## Citation

```
Maier, B. F., & Aslak, U. (2025). Measuring Synthetic Consumer Purchase Intent Using Embeddings-Similarity Ratings.
```

## License

MIT License
