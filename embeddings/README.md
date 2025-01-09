# Embeddings Evaluator Module

This module provides functionality to generate embeddings using Sentence Transformer models (both base models from HuggingFace and fine-tuned versions), calculate similarities between posts and fact checks, and evaluate the results using success@10 metrics.

## Features

- Load and use any Sentence Transformer model:
  - Base models from HuggingFace
  - Locally saved base models
  - Fine-tuned models
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

# Initialize with a base model from HuggingFace
evaluator = EmbeddingsEvaluator('intfloat/multilingual-e5-small', model_type='base')

# Or load a fine-tuned model
evaluator = EmbeddingsEvaluator('models/fine-tuned-models/my-model', model_type='fine-tuned')

# Alternative way to load a pretrained model
evaluator = EmbeddingsEvaluator.from_pretrained('models/fine-tuned-models/my-model', model_type='fine-tuned')

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
# Using a base model from HuggingFace
python run_evaluation.py --model_name_or_path "intfloat/multilingual-e5-small" \
                        --model_type base \
                        --num_posts 100 \
                        --language eng \
                        --fact_checks_prefix "query: " \
                        --posts_prefix "passage: "

# Using a fine-tuned model
python run_evaluation.py --model_name_or_path "models/fine-tuned-models/my-model" \
                        --model_type fine-tuned \
                        --num_posts 100
```

Optional arguments:
- `--model_name_or_path`: Name of the HuggingFace model or path to local model
- `--model_type`: Type of model to load ('base' or 'fine-tuned')
- `--fact_checks_path`: Path to fact checks CSV file
- `--posts_path`: Path to posts CSV file
- `--pairs_path`: Path to pairs CSV file
- `--num_posts`: Number of posts to evaluate (default: 100)
- `--language`: Language to filter posts by (default: None, use all languages)
- `--fact_checks_prefix`: Prefix to add to fact check texts (default: None)
- `--posts_prefix`: Prefix to add to post texts (default: None)
- `--save_model`: Whether to save the base model locally (flag)

Column name arguments (all optional with sensible defaults):
- `--fact_checks_id_col`: Column name for fact check IDs
- `--fact_checks_text_col`: Column name for fact check text to embed
- `--posts_id_col`: Column name for post IDs
- `--posts_text_col`: Column name for post text to embed
- `--posts_language_col`: Column name for language in posts
- `--pairs_post_id_col`: Column name for post IDs in pairs
- `--pairs_fact_check_id_col`: Column name for fact check IDs in pairs

## Model Types and Paths

The module supports two types of models:

1. Base Models:
   - Can be loaded directly from HuggingFace
   - Can be saved locally in `models/base-models/{model_name}`
   - Example: `intfloat/multilingual-e5-small`

2. Fine-tuned Models:
   - Must be loaded from a local path
   - Should be in `models/fine-tuned-models/{model_name}`
   - Example: `models/fine-tuned-models/my-fine-tuned-model`

When loading a fine-tuned model, you can either provide the full path or just the model name (it will look in the fine-tuned-models directory).

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