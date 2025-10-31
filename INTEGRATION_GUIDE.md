# Semantic Similarity Rating (SSR) - Integration Guide for LLM Applications

## Overview

This guide explains how to integrate Semantic Similarity Rating (SSR) logic into applications that generate LLM responses and need to convert them into Likert scale probability distributions.

**What SSR Does**: Converts free-text LLM responses (e.g., "I somewhat agree") into probability distributions across Likert scale points (e.g., `[0.05, 0.15, 0.35, 0.35, 0.10]` for a 5-point scale from "Strongly Disagree" to "Strongly Agree").

**Why This Matters**: Instead of forcing LLMs to output single numeric ratings (1-5), SSR preserves uncertainty and nuance by quantifying the semantic similarity between the response and each scale point.

---

## Core Concepts

### 1. The SSR Equation

SSR computes a probability mass function (PMF) using cosine similarity between embeddings:

```
p[r] = (similarity[r] - min_similarity + ε) /
       (sum_all_similarities - n_points * min_similarity + ε)
```

Where:
- `similarity[r]`: Cosine similarity between response embedding and reference statement `r`
- `min_similarity`: Minimum similarity across all scale points (baseline subtraction)
- `n_points`: Number of Likert scale points (typically 5)
- `ε` (epsilon): Optional regularization parameter (default: 0.0)

**Key Insight**: Subtracting the minimum similarity creates a "relative similarity" that emphasizes distinctions between scale points.

### 2. Temperature Scaling (Optional)

After computing the base PMF, you can apply temperature scaling to control distribution sharpness:

```
p_scaled[i] = (p[i]^(1/T)) / sum(p[j]^(1/T) for all j)
```

- **T = 0**: One-hot encoding (argmax of probabilities)
- **T = 1**: No scaling (identity)
- **T > 1**: Softer distribution (more uniform)
- **T < 1**: Sharper distribution (more peaked)

---

## Implementation Components to Borrow

### 1. **Core Math Functions** (`compute.py`)

#### `response_embeddings_to_pmf(matrix_responses, matrix_likert_sentences, epsilon=0.0)`

**Purpose**: Converts embeddings to probability distributions using SSR equation.

**Input**:
- `matrix_responses`: numpy array of shape `(n_responses, embedding_dim)` - LLM response embeddings
- `matrix_likert_sentences`: numpy array of shape `(embedding_dim, n_scale_points)` - Reference embeddings (transposed)
- `epsilon`: Regularization parameter (default: 0.0)

**Output**:
- numpy array of shape `(n_responses, n_scale_points)` - Probability distributions

**Algorithm**:
```python
# 1. Normalize embeddings (L2 norm)
M_left = matrix_responses / ||matrix_responses||
M_right = matrix_likert_sentences / ||matrix_likert_sentences||

# 2. Compute cosine similarities (scaled to [0, 1])
cos = (1 + M_left @ M_right) / 2

# 3. Subtract minimum similarity per response
cos_min = min(cos, axis=1)
numerator = cos - cos_min + epsilon * kronecker_delta

# 4. Normalize to sum to 1
denominator = sum(cos, axis=1) - n_points * cos_min + epsilon
pmf = numerator / denominator
```

#### `scale_pmf(pmf, temperature, max_temp=inf)`

**Purpose**: Apply temperature scaling to control distribution sharpness.

**Input**:
- `pmf`: 1D array of probabilities
- `temperature`: Scaling parameter (0 to inf)
- `max_temp`: Optional ceiling on temperature

**Output**: Scaled PMF (still sums to 1)

---

### 2. **Orchestration Class** (`response_rater.py`)

The `ResponseRater` class provides a higher-level interface that manages:
- Multiple reference sets (different phrasings of Likert scales)
- Automatic embedding computation
- Reference set selection and averaging

**Key Features for Integration**:

#### Dual Operating Modes

**Text Mode** (recommended for most applications):
```python
# Automatically computes embeddings using sentence-transformers
rater = ResponseRater(df_references)  # No embedding column
pmfs = rater.get_response_pmfs('set1', ["I agree", "Not sure"])
```

**Embedding Mode** (for custom embedding pipelines):
```python
# Uses pre-computed embeddings
rater = ResponseRater(df_references_with_embeddings)
pmfs = rater.get_response_pmfs('set1', embedding_matrix)
```

#### Reference Set Management

```python
# Use specific reference set
pmfs = rater.get_response_pmfs('set1', responses)

# Average across all reference sets (more robust)
pmfs = rater.get_response_pmfs('mean', responses)

# Get survey-level aggregate (average PMFs)
survey_pmf = rater.get_survey_response_pmf(pmfs)
```

---

## Data Structures

### Reference Sentences DataFrame

**Required Structure** (Polars DataFrame or convert from pandas):

```python
import polars as po

df_references = po.DataFrame({
    'id': ['set1', 'set1', 'set1', 'set1', 'set1',      # Reference set ID
           'set2', 'set2', 'set2', 'set2', 'set2'],
    'int_response': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],     # Must be 1-5 for each set
    'sentence': [
        # Set 1 (formal phrasing)
        'Strongly disagree', 'Disagree', 'Neutral',
        'Agree', 'Strongly agree',
        # Set 2 (informal phrasing)
        'Disagree a lot', 'Kinda disagree', 'Don\'t know',
        'Kinda agree', 'Agree a lot'
    ]
})
```

**Validation Requirements**:
- Each reference set (`id`) must have exactly 5 sentences
- `int_response` must be [1, 2, 3, 4, 5] for each set
- Reserved ID: `'mean'` cannot be used (reserved for averaging)
- Optional `embedding` column for pre-computed embeddings

---

## Integration Workflow

### Minimal Integration (Just the Math)

If your app already has embeddings:

```python
import numpy as np
from semantic_similarity_rating.compute import response_embeddings_to_pmf, scale_pmf

# Your app generates these
llm_response_embeddings = np.array([[...], [...]])  # Shape: (n_responses, 384)
reference_embeddings = np.array([[...], [...], ...]).T  # Shape: (384, 5) - TRANSPOSED!

# Convert to PMFs
pmfs = response_embeddings_to_pmf(llm_response_embeddings, reference_embeddings)

# Optional: Apply temperature scaling
pmfs_scaled = np.array([scale_pmf(pmf, temperature=0.8) for pmf in pmfs])

# Get survey aggregate
survey_pmf = pmfs_scaled.mean(axis=0)
```

**Key Detail**: Reference embeddings must be **transposed** (shape: `embedding_dim x n_scale_points`).

---

### Full Integration (Text to PMF)

If your app generates text responses:

```python
import polars as po
from semantic_similarity_rating import ResponseRater

# 1. Set up reference sentences (one-time setup)
df_references = po.DataFrame({
    'id': ['likert_v1'] * 5,
    'int_response': [1, 2, 3, 4, 5],
    'sentence': [
        'Strongly disagree',
        'Disagree',
        'Neutral',
        'Agree',
        'Strongly agree'
    ]
})

# 2. Initialize rater (loads sentence-transformer model)
rater = ResponseRater(df_references, model_name='all-MiniLM-L6-v2')

# 3. Your app generates LLM responses
llm_responses = [
    "I completely agree with this",
    "I'm not really sure about this",
    "I strongly disagree"
]

# 4. Convert to PMFs
pmfs = rater.get_response_pmfs(
    reference_set_id='likert_v1',
    llm_responses=llm_responses,
    temperature=1.0,
    epsilon=0.0
)

# 5. Get aggregate survey response
survey_pmf = rater.get_survey_response_pmf(pmfs)

print("Individual PMFs:")
print(pmfs)  # Shape: (3, 5)
print("\nSurvey-level PMF:")
print(survey_pmf)  # Shape: (5,)
```

---

## Advanced: Multiple Reference Sets

Using multiple phrasings improves robustness:

```python
df_references = po.DataFrame({
    'id': ['formal'] * 5 + ['casual'] * 5 + ['academic'] * 5,
    'int_response': [1, 2, 3, 4, 5] * 3,
    'sentence': [
        # Formal
        'Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree',
        # Casual
        'Totally disagree', 'Kinda disagree', 'Meh', 'Kinda agree', 'Totally agree',
        # Academic
        'Reject entirely', 'Reject partially', 'Withhold judgment',
        'Accept partially', 'Accept entirely'
    ]
})

rater = ResponseRater(df_references)

# Use specific set
pmfs_formal = rater.get_response_pmfs('formal', responses)

# Average across all sets (recommended for robustness)
pmfs_averaged = rater.get_response_pmfs('mean', responses)
```

---

## Technical Considerations

### 1. **Embedding Model Selection**

**Default**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Pros**: Fast, lightweight (80MB), good general performance
- **Embedding dim**: 384
- **Max tokens**: 256

**Alternatives**:
- `all-mpnet-base-v2`: Better quality, slower (420MB, 768-dim)
- `paraphrase-multilingual-mpnet-base-v2`: Multilingual support
- Custom models via sentence-transformers or OpenAI/Cohere APIs

**Integration Tip**: If using custom embeddings (OpenAI, Cohere, etc.), use **embedding mode**:

```python
# Pre-compute embeddings with your provider
reference_embeddings = openai.embed([ref1, ref2, ref3, ref4, ref5])

df_with_embeddings = po.DataFrame({
    'id': ['set1'] * 5,
    'int_response': [1, 2, 3, 4, 5],
    'sentence': [...],
    'embedding': reference_embeddings  # List of lists or arrays
})

rater = ResponseRater(df_with_embeddings)  # Auto-detects embedding mode
```

### 2. **Performance Optimization**

- **Batch Processing**: `model.encode()` is vectorized - pass lists, not loops
- **GPU Acceleration**: Set `device='cuda'` in ResponseRater constructor
- **Caching**: Pre-compute reference embeddings once, reuse across requests
- **Embedding Size**: Smaller models (384-dim) are 2-3x faster than large (768-dim)

### 3. **Parameter Tuning**

**Epsilon (ε)**:
- **Default**: 0.0 (no regularization)
- **Use case**: Prevent numerical instability if all similarities are equal
- **Typical range**: 0.0 to 0.01
- **Effect**: Adds small uniform probability to all scale points

**Temperature (T)**:
- **Default**: 1.0 (no scaling)
- **T < 1**: Sharper distributions (more confident)
- **T > 1**: Softer distributions (less confident)
- **T = 0**: One-hot encoding (forces single choice)
- **Typical range**: 0.5 to 2.0

### 4. **Edge Cases**

**Empty responses**:
```python
# Returns empty array
pmfs = response_embeddings_to_pmf(np.empty((0, 384)), reference_matrix)
# Shape: (0, 5)
```

**Identical embeddings**:
- All similarities equal → PMF becomes uniform distribution
- Epsilon helps distinguish slightly: `[0.2, 0.2, 0.2, 0.2, 0.2]`

**Temperature = 0 with ties**:
- If multiple scale points have max probability → returns original PMF
- Otherwise → one-hot at argmax

---

## Example Application Integration

### Survey Application Flow

```python
class SurveyProcessor:
    def __init__(self, reference_sentences_df):
        self.rater = ResponseRater(
            reference_sentences_df,
            model_name='all-MiniLM-L6-v2',
            device='cpu'  # or 'cuda'
        )

    def process_survey_question(
        self,
        question: str,
        llm_responses: list[str],
        temperature: float = 1.0
    ) -> dict:
        """
        Process LLM responses for a single survey question.

        Returns:
            {
                'individual_pmfs': array of shape (n_responses, 5),
                'survey_pmf': array of shape (5,),
                'expected_value': float (1-5 scale),
                'distribution_entropy': float
            }
        """
        # Get PMFs (averaged across reference sets for robustness)
        pmfs = self.rater.get_response_pmfs(
            reference_set_id='mean',
            llm_responses=llm_responses,
            temperature=temperature
        )

        # Aggregate to survey level
        survey_pmf = self.rater.get_survey_response_pmf(pmfs)

        # Compute summary statistics
        scale_points = np.array([1, 2, 3, 4, 5])
        expected_value = np.dot(survey_pmf, scale_points)
        entropy = -np.sum(survey_pmf * np.log(survey_pmf + 1e-10))

        return {
            'individual_pmfs': pmfs,
            'survey_pmf': survey_pmf,
            'expected_value': expected_value,
            'distribution_entropy': entropy
        }

    def process_full_survey(
        self,
        questions: list[str],
        responses_per_question: list[list[str]]
    ) -> list[dict]:
        """Process all questions in a survey."""
        return [
            self.process_survey_question(q, responses)
            for q, responses in zip(questions, responses_per_question)
        ]
```

**Usage**:
```python
# Setup
processor = SurveyProcessor(df_references)

# Your app generates these
llm_responses = [
    "I think this is pretty good",
    "Not convinced about this",
    "Absolutely love it!"
]

# Process
results = processor.process_survey_question(
    question="How satisfied are you with the product?",
    llm_responses=llm_responses,
    temperature=1.0
)

print(f"Expected rating: {results['expected_value']:.2f}/5.0")
print(f"Distribution: {results['survey_pmf']}")
print(f"Uncertainty (entropy): {results['distribution_entropy']:.3f}")
```

---

## Dependencies

**Minimal** (just the math):
```
numpy>=1.24.0
scipy>=1.10.0
```

**Full** (with text embedding):
```
numpy>=1.24.0
scipy>=1.10.0
polars>=0.20.0
sentence-transformers>=2.2.0
beartype>=0.15.0
```

**Installation**:
```bash
pip install numpy scipy polars sentence-transformers beartype
```

Or install from this repository:
```bash
pip install git+https://github.com/pymc-labs/semantic-similarity-rating.git
```

---

## Key Files to Reference

- **`semantic_similarity_rating/compute.py`** (123 lines)
  Core mathematical functions - can be extracted as standalone module

- **`semantic_similarity_rating/response_rater.py`** (368 lines)
  Orchestration layer - adapt for your application's needs

- **`tests/test_compute.py`** (8KB)
  Test cases showing expected behavior and edge cases

- **`tests/test_response_rater.py`** (9KB)
  Integration test examples

---

## Citation

If you use this methodology, please cite:

```
Maier, B. F., Aslak, U., Fiaschi, L., Pappas, K., Wiecki, T. (2025).
Measuring Synthetic Consumer Purchase Intent Using Semantic-Similarity Ratings.
```

---

## License

MIT License - Free to use and modify for commercial and non-commercial applications.

---

## Quick Reference Card

| **Task** | **Code** |
|----------|----------|
| Convert text responses to PMFs | `rater.get_response_pmfs('set1', ["response1", "response2"])` |
| Use averaged reference sets | `rater.get_response_pmfs('mean', responses)` |
| Apply temperature scaling | `rater.get_response_pmfs('set1', responses, temperature=0.8)` |
| Get survey aggregate | `rater.get_survey_response_pmf(pmfs)` |
| Just the math (embeddings → PMF) | `compute.response_embeddings_to_pmf(resp_emb, ref_emb)` |
| Scale existing PMF | `compute.scale_pmf(pmf, temperature=0.5)` |
| Use custom embedding model | Include `'embedding'` column in DataFrame |
| Check available reference sets | `rater.available_reference_sets` |
| Get model info | `rater.model_info` |

---

## Common Pitfalls

1. **Reference embedding shape**: Must be `(embedding_dim, 5)` not `(5, embedding_dim)` - transpose if needed!
2. **Reference dataframe validation**: Each set needs exactly 5 sentences with `int_response` = [1, 2, 3, 4, 5]
3. **Mode confusion**: Text mode expects `list[str]`, embedding mode expects `np.ndarray`
4. **Temperature = 0**: Only use if you want hard classification (one-hot output)
5. **Polars vs Pandas**: Use `polars.DataFrame` or convert: `po.from_pandas(df)`

---

## Support

- **Original Repository**: https://github.com/pymc-labs/semantic-similarity-rating
- **Paper**: Maier et al. (2025)
- **License**: MIT
