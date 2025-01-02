from embeddings_evaluator import EmbeddingsEvaluator
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate embeddings and evaluate fact-checking predictions')
    parser.add_argument('--model_name', type=str, default='intfloat/multilingual-e5-small',
                      help='Name of the HuggingFace model to use')
    parser.add_argument('--fact_checks_path', type=str, default='data/transformed/fact_checks.csv',
                      help='Path to fact checks CSV file')
    parser.add_argument('--posts_path', type=str, default='data/transformed/posts.csv',
                      help='Path to posts CSV file')
    parser.add_argument('--pairs_path', type=str, default='data/transformed/pairs.csv',
                      help='Path to pairs CSV file')
    parser.add_argument('--num_posts', type=int, default=100,
                      help='Number of posts to evaluate')
    parser.add_argument('--language', type=str, default=None,
                      help='Language to filter posts by (default: use all languages)')
    parser.add_argument('--fact_checks_prefix', type=str, default=None,
                      help='Prefix to add to fact check texts (e.g., "query: ")')
    parser.add_argument('--posts_prefix', type=str, default=None,
                      help='Prefix to add to post texts (e.g., "passage: ")')
    
    # Column names arguments
    parser.add_argument('--fact_checks_id_col', type=str, default='fact_check_id',
                      help='Column name for fact check IDs')
    parser.add_argument('--fact_checks_text_col', type=str, default='claim_title',
                      help='Column name for fact check text to embed')
    parser.add_argument('--posts_id_col', type=str, default='post_id',
                      help='Column name for post IDs')
    parser.add_argument('--posts_text_col', type=str, default='text_ocr',
                      help='Column name for post text to embed')
    parser.add_argument('--posts_language_col', type=str, default='language',
                      help='Column name for language in posts')
    parser.add_argument('--pairs_post_id_col', type=str, default='post_id',
                      help='Column name for post IDs in pairs')
    parser.add_argument('--pairs_fact_check_id_col', type=str, default='fact_check_id',
                      help='Column name for fact check IDs in pairs')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = EmbeddingsEvaluator(args.model_name)
    
    # Define column names
    columns = {
        'fact_checks_id': args.fact_checks_id_col,
        'fact_checks_text': args.fact_checks_text_col,
        'posts_id': args.posts_id_col,
        'posts_text': args.posts_text_col,
        'posts_language': args.posts_language_col,
        'pairs_post_id': args.pairs_post_id_col,
        'pairs_fact_check_id': args.pairs_fact_check_id_col
    }
    
    # Load data
    data_paths = {
        'fact_checks': args.fact_checks_path,
        'posts': args.posts_path,
        'pairs': args.pairs_path
    }
    evaluator.load_data(data_paths, columns, args.language,
                       fact_checks_prefix=args.fact_checks_prefix,
                       posts_prefix=args.posts_prefix)
    
    # Get posts to evaluate
    test_posts = evaluator.posts.index[:args.num_posts].tolist()
    
    # Get similarities and top 10 predictions
    similarities, top_10_predictions = evaluator.get_similarities(test_posts)
    
    # Evaluate predictions
    general_score, by_language = evaluator.evaluate(test_posts, top_10_predictions)

if __name__ == "__main__":
    main() 