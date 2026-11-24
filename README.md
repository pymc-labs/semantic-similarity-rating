# Semantic-Similarity Rating (SSR)

A Python package implementing the Semantic-Similarity Rating methodology for converting LLM textual responses to Likert scale probability distributions using semantic similarity against reference statements.

## Overview

The SSR methodology addresses the challenge of mapping rich textual responses from Large Language Models (LLMs) to structured Likert scale ratings. Instead of forcing a single numerical rating, SSR preserves the inherent uncertainty and nuance in textual responses by generating probability distributions over all possible Likert scale points.

This package provides a distilled, reusable implementation of the SSR methodology described in the paper "Measuring Synthetic Consumer Purchase Intent Using Semantic-Similarity Ratings" (2025).

## Installation

### Local Development
To install this package locally for development, run:
```bash
pip install -e .
```

### From GitHub Repository
To install this package into your own project from GitHub, run:
```bash
pip install git+https://github.com/pymc-labs/semantic-similarity-rating.git
```

## Quick Start

```python
import polars as po
import numpy as np
from semantic_similarity_rating import ResponseRater

# Create example reference sentences dataframe
reference_set_1 = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]
reference_set_2 = [
    "Disagree a lot",
    "Kinda disagree",
    "Don't know",
    "Kinda agree",
    "Agree a lot",
]
df = po.DataFrame(
    {
        "id": ["set1"] * 5 + ["set2"] * 5,
        "int_response": [1, 2, 3, 4, 5] * 2,
        "sentence": reference_set_1 + reference_set_2,
    }
)

# Initialize rater
rater = ResponseRater(df)

# Create some example synthetic consumer responses
llm_responses = ["I totally agree", "Not sure about this", "Completely disagree"]

# Get PMFs for synthetic consumer responses
pmfs = rater.get_response_pmfs(
    reference_set_id="set1",      # Reference set to score against, or "mean"
    llm_responses=llm_responses,  # List of LLM responses to score
    temperature=1.0,              # Temperature for scaling the PMF
    epsilon=0.0,                  # Small regularization parameter to prevent division by zero and add smoothing
)

# Get survey response PMF
survey_pmf = rater.get_survey_response_pmf(pmfs)

print(survey_pmf)
```

## Choice probabilities from decision models

Some models return a probability for each option of a multiple-choice question instead of free text. With such a model, ask which reference sentence the respondent would most likely say in reply to the survey question, using each reference set as the options. `get_choice_pmfs` turns those probabilities into SSR PMFs:

```python
# One row per respondent; columns follow the set's int_response order.
choice_probabilities = np.array([[0.05, 0.10, 0.15, 0.40, 0.30]])
pmfs = rater.get_choice_pmfs("set1", choice_probabilities)

# Average over all reference sets, with one array per set.
pmfs = rater.get_choice_pmfs("mean", {"set1": probs_1, "set2": probs_2})
```

A chosen sentence is not read as its own scale point. It gets the PMF that SSR gives a free-text response equal to that sentence (`get_anchor_kernel`), and the choice probabilities weight those PMFs. The output is therefore on the same scale as `get_response_pmfs`, and nothing is fitted to survey data. The model still never states a number, and each reference set is scored separately and averaged, as in standard SSR.

## Methodology

The ESR methodology works by:
1. Defining reference statements for each Likert scale point
2. Computing cosine similarities between LLM response embeddings and reference statement embeddings
3. Converting similarities to probability distributions using minimum similarity subtraction and normalization
4. Optionally applying temperature scaling for distribution control

## Core Components

- `ResponseRater`: Main class implementing the SSR methodology
- `get_response_pmfs()`: Convert LLM response embeddings to PMFs using specified reference set

## Citation

```
Maier, B. F., Aslak, U., Fiaschi, L., Pappas, K., Wiecki, T. (2025). Measuring Synthetic Consumer Purchase Intent Using Embeddings-Similarity Ratings.
```

## License

MIT License
