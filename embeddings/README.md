# Embeddings Evaluator Module

This module provides functionality to generate embeddings using any Sentence Transformer model from HuggingFace, calculate similarities between posts and fact checks, and evaluate the results using success@10 metrics.

## Features

- Load and use any Sentence Transformer model from HuggingFace
- Generate embeddings for posts and fact checks
- Calculate cosine similarities and find top 10 most relevant fact checks
- Evaluate predictions using success@10 metric (overall and by language)
- Filter by specific language (optional)
- Configurable column names for data files
- Add prefixes to texts (e.g., "query: " for fact checks, "passage: " for posts)

## Usage

### Basic Usage

```python
from embeddings_evaluator import EmbeddingsEvaluator

# Initialize with any Sentence Transformer model
evaluator = EmbeddingsEvaluator('intfloat/multilingual-e5-small')

# Define column names for your data
columns = {
    'fact_checks_id': 'fact_check_id',
    'fact_checks_text': 'claim_title',
    'posts_id': 'post_id',
    'posts_text': 'text_ocr',
    'posts_language': 'language',
    'pairs_post_id': 'post_id',
    'pairs_fact_check_id': 'fact_check_id'
}

# Load your data (optionally specify a language and prefixes)
data_paths = {
    'fact_checks': 'data/transformed/fact_checks.csv',
    'posts': 'data/transformed/posts.csv',
    'pairs': 'data/transformed/pairs.csv'
}
evaluator.load_data(data_paths, columns, 
                   language='eng',  # optional
                   fact_checks_prefix="query: ",  # optional
                   posts_prefix="passage: ")  # optional

# Get some posts to evaluate
test_posts = evaluator.posts.index[:100].tolist()

# Get similarities and top 10 predictions
similarities, top_10_predictions = evaluator.get_similarities(test_posts)

# Evaluate predictions
general_score, by_language = evaluator.evaluate(test_posts, top_10_predictions)
```

### Using the Command Line Script

You can also use the provided script to run the evaluation:

```bash
python run_evaluation.py --model_name "intfloat/multilingual-e5-small" \
                        --num_posts 100 \
                        --language eng \
                        --fact_checks_prefix "query: " \
                        --posts_prefix "passage: "
```

Optional arguments:
- `--model_name`: Name of the HuggingFace model to use (default: 'intfloat/multilingual-e5-small')
- `--fact_checks_path`: Path to fact checks CSV file
- `--posts_path`: Path to posts CSV file
- `--pairs_path`: Path to pairs CSV file
- `--num_posts`: Number of posts to evaluate (default: 100)
- `--language`: Language to filter posts by (default: None, use all languages)
- `--fact_checks_prefix`: Prefix to add to fact check texts (default: None)
- `--posts_prefix`: Prefix to add to post texts (default: None)

Column name arguments (all optional with sensible defaults):
- `--fact_checks_id_col`: Column name for fact check IDs
- `--fact_checks_text_col`: Column name for fact check text to embed
- `--posts_id_col`: Column name for post IDs
- `--posts_text_col`: Column name for post text to embed
- `--posts_language_col`: Column name for language in posts
- `--pairs_post_id_col`: Column name for post IDs in pairs
- `--pairs_fact_check_id_col`: Column name for fact check IDs in pairs

## Input Data Format

The module expects CSV files with the following columns (column names are configurable):

1. fact_checks.csv:
   - ID column (default: 'fact_check_id')
   - Text column to embed (default: 'claim_title')

2. posts.csv:
   - ID column (default: 'post_id')
   - Text column to embed (default: 'text_ocr')
   - Language column (default: 'language')

3. pairs.csv:
   - Post ID column (default: 'post_id')
   - Fact check ID column (default: 'fact_check_id')

## Text Prefixes

Some models perform better when texts are prefixed with specific strings:
- Fact check texts can be prefixed (e.g., "query: " for E5 models)
- Post texts can be prefixed (e.g., "passage: " for E5 models)

The prefixes are added temporarily during processing and don't modify the original data.

## Output

The module will:
1. Generate embeddings for all texts (with prefixes if specified)
2. Calculate similarities between posts and fact checks
3. Find the top 10 most relevant fact checks for each post
4. Calculate and display success@10 metrics (overall and by language)

## Requirements

- sentence-transformers
- pandas
- numpy
- scikit-learn
- torch 