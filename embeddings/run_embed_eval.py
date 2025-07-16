from embeddings_evaluator import EmbeddingsEvaluator


import os

os.chdir('/Users/santiagolares/Projects/semeval_2025')


val_data_paths = {
  'fact_checks': 'data/transformed/fact_checks_val.csv',
   'posts': 'data/transformed/posts_train_val.csv',
  'pairs': 'data/transformed/pairs_val.csv'
 }

# Define column names
columns = {
    'fact_checks_id': 'fact_check_id',
    'pairs_fact_check_id': 'fact_check_id',
    'fact_checks_text': 'claim_title_stripped',
    'posts_id': 'post_id',
    'posts_text': 'text_ocr_stripped',
    'posts_language': 'language',
    'pairs_post_id': 'post_id',
    'pairs_fact_check_id': 'fact_check_id'
    }

evaluator = EmbeddingsEvaluator("text-embedding-3-small", 
    model_type='openai')

evaluator.load_data(val_data_paths, columns, fact_checks_prefix="query: ",
                       posts_prefix="passage: ")
    
# Get some post IDs to test
posts = evaluator.posts.index.tolist()


similarities, top_10_predictions = evaluator.get_similarities(
    posts,
    same_language_only=True
)

# Evaluate predictions
general_score, by_language = evaluator.evaluate(posts, top_10_predictions) 