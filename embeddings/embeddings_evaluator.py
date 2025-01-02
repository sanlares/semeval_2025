import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging
from typing import Optional, List
import os
import pickle

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)

class EmbeddingsEvaluator:
    def __init__(self, model_name: str, save_model: bool = True):
        """
        Initialize the evaluator with a SentenceTransformer model.
        
        Args:
            model_name: Name of the HuggingFace model to use (e.g., 'intfloat/multilingual-e5-small')
            save_model: Whether to save the model locally (default: True)
        """
        self.model_name = model_name
        model_path = os.path.join('models', 'base-models', model_name.split('/')[-1])
        
        if save_model and os.path.exists(model_path):
            logging.info(f"Loading model from local path: {model_path}")
            self.model = SentenceTransformer(model_path)
        else:
            logging.info(f"Loading model from HuggingFace: {model_name}")
            self.model = SentenceTransformer(model_name)
            if save_model:
                logging.info(f"Saving model to: {model_path}")
                os.makedirs(model_path, exist_ok=True)
                self.model.save(model_path)
                
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def load_data(self, 
                 data_paths: dict,
                 columns: dict,
                 language: Optional[str] = None,
                 fact_checks_prefix: Optional[str] = None,
                 posts_prefix: Optional[str] = None) -> None:
        """
        Load the necessary data files.
        
        Args:
            data_paths: Dictionary containing paths to required files:
                - fact_checks: path to fact checks CSV
                - posts: path to posts CSV
                - pairs: path to pairs CSV for evaluation
            columns: Dictionary containing column names to use:
                - fact_checks_id: column name for fact check IDs
                - fact_checks_text: column name for fact check text to embed
                - posts_id: column name for post IDs
                - posts_text: column name for post text to embed
                - posts_language: column name for language in posts
                - pairs_post_id: column name for post IDs in pairs
                - pairs_fact_check_id: column name for fact check IDs in pairs
            language: Optional language to filter posts by. If None, use all languages.
            fact_checks_prefix: Optional prefix to add to fact check texts (e.g., "query: ")
            posts_prefix: Optional prefix to add to post texts (e.g., "passage: ")
        """
        logging.info("Loading data files...")
        
        # Load fact checks
        self.fact_checks = pd.read_csv(data_paths['fact_checks'])
        if language:
            logging.info(f"Filtering fact checks for language: {language}")
            self.fact_checks = self.fact_checks[self.fact_checks[columns['posts_language']] == language]
            if len(self.fact_checks) == 0:
                raise ValueError(f"No fact checks found for language {language}")
                
        self.fact_checks = self.fact_checks.set_index(columns['fact_checks_id'])
        
        # Create temporary column with prefix if needed
        if fact_checks_prefix:
            self.fact_checks_text_col = f"{columns['fact_checks_text']}_with_prefix"
            self.fact_checks[self.fact_checks_text_col] = fact_checks_prefix + self.fact_checks[columns['fact_checks_text']].astype(str)
        else:
            self.fact_checks_text_col = columns['fact_checks_text']
        
        # Load posts
        self.posts = pd.read_csv(data_paths['posts'])
        if language:
            logging.info(f"Filtering posts for language: {language}")
            self.posts = self.posts[self.posts[columns['posts_language']] == language]
            if len(self.posts) == 0:
                raise ValueError(f"No posts found for language {language}")
                
        self.posts = self.posts.set_index(columns['posts_id'])
        
        # Create temporary column with prefix if needed
        if posts_prefix:
            self.posts_text_col = f"{columns['posts_text']}_with_prefix"
            self.posts[self.posts_text_col] = posts_prefix + self.posts[columns['posts_text']].astype(str)
        else:
            self.posts_text_col = columns['posts_text']
            
        self.posts_language_col = columns['posts_language']
        
        # Load pairs with correct column names
        self.pairs = pd.read_csv(data_paths['pairs'])
        self.pairs = self.pairs.rename(columns={
            columns['pairs_post_id']: 'post_id',
            columns['pairs_fact_check_id']: 'fact_check_id'
        })
        
        if language:
            # Filter pairs to only include posts and fact checks in the filtered language
            self.pairs = self.pairs[
                self.pairs['post_id'].isin(self.posts.index) & 
                self.pairs['fact_check_id'].isin(self.fact_checks.index)
            ]
        
        logging.info(f"Data loaded successfully. Posts shape: {self.posts.shape}, Fact checks shape: {self.fact_checks.shape}, Pairs shape: {self.pairs.shape}")
        
        # Log language distribution
        if not language:
            posts_lang_dist = self.posts[self.posts_language_col].value_counts()
            fact_checks_lang_dist = self.fact_checks[self.posts_language_col].value_counts()
            logging.info("\nPosts language distribution:")
            logging.info(posts_lang_dist)
            logging.info("\nFact checks language distribution:")
            logging.info(fact_checks_lang_dist)
        
    def generate_embeddings(self, texts: list, save_path: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to generate embeddings for
            save_path: Optional path to save the embeddings
            
        Returns:
            numpy array of embeddings
        """
        logging.info("Generating embeddings...")
        embeddings = self.model.encode(texts,
                                     batch_size=16,
                                     show_progress_bar=True,
                                     convert_to_numpy=True,
                                     normalize_embeddings=True)
        
        if save_path:
            logging.info(f"Saving embeddings to: {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(embeddings, f)
                
        return embeddings
        
    def get_similarities(self, 
                        posts_ids: list, 
                        same_language_only: bool = False,
                        save_embeddings: bool = True) -> tuple:
        """
        Calculate similarities between given posts and fact checks.
        
        Args:
            posts_ids: List of post IDs to process
            same_language_only: Whether to only consider fact checks in the same language as the post
            save_embeddings: Whether to save the generated embeddings
            
        Returns:
            Tuple containing:
            - similarities matrix (list of arrays, as lengths may vary with language filtering)
            - top 10 fact check IDs for each post
        """
        embeddings_dir = os.path.join('models', 'base-models', self.model_name.split('/')[-1], 'embeddings')
        fact_checks_emb_path = os.path.join(embeddings_dir, 'fact_checks.pkl')
        posts_emb_path = os.path.join(embeddings_dir, 'posts.pkl')
        
        # Generate or load fact checks embeddings
        if save_embeddings and os.path.exists(fact_checks_emb_path):
            logging.info(f"Loading fact checks embeddings from: {fact_checks_emb_path}")
            with open(fact_checks_emb_path, 'rb') as f:
                fact_checks_embeddings = pickle.load(f)
        else:
            fact_checks_texts = self.fact_checks[self.fact_checks_text_col].tolist()
            fact_checks_embeddings = self.generate_embeddings(
                fact_checks_texts,
                save_path=fact_checks_emb_path if save_embeddings else None
            )
        
        # Generate posts embeddings
        posts_texts = [str(self.posts.loc[post_id, self.posts_text_col]) for post_id in posts_ids]
        posts_embeddings = self.generate_embeddings(
            posts_texts,
            save_path=posts_emb_path if save_embeddings else None
        )
        
        # Initialize arrays for results
        all_top_10_ids = []
        all_similarities = []
        
        # Process each post
        for i, post_id in enumerate(posts_ids):
            post_embedding = posts_embeddings[i:i+1]  # Keep 2D shape
            
            if same_language_only:
                # Get post language
                post_language = self.posts.loc[post_id, self.posts_language_col]
                # Get fact checks in the same language
                same_lang_mask = self.fact_checks[self.posts_language_col] == post_language
                valid_fact_checks_embeddings = fact_checks_embeddings[same_lang_mask]
                valid_fact_checks_indices = np.where(same_lang_mask)[0]
                
                if len(valid_fact_checks_indices) == 0:
                    logging.warning(f"No fact checks found for language {post_language}, using all fact checks for post {post_id}")
                    valid_fact_checks_embeddings = fact_checks_embeddings
                    valid_fact_checks_indices = np.arange(len(fact_checks_embeddings))
            else:
                valid_fact_checks_embeddings = fact_checks_embeddings
                valid_fact_checks_indices = np.arange(len(fact_checks_embeddings))
            
            # Calculate similarities
            similarities = cosine_similarity(post_embedding, valid_fact_checks_embeddings)
            
            # Get top 10 indices (or less if not enough fact checks)
            num_results = min(10, len(valid_fact_checks_indices))
            top_k_local_indices = np.argsort(-similarities[0])[:num_results]
            # Convert to global indices
            top_k_global_indices = valid_fact_checks_indices[top_k_local_indices]
            # Get fact check IDs
            top_k_ids = [int(self.fact_checks.index[j]) for j in top_k_global_indices]
            
            all_top_10_ids.append(top_k_ids)
            all_similarities.append(similarities[0])
        
        return all_similarities, all_top_10_ids
        
    def evaluate(self, posts_ids: list, top_k_predictions: list) -> tuple:
        """
        Calculate success@10 metrics for the predictions.
        
        Args:
            posts_ids: List of post IDs that were processed
            top_k_predictions: List of lists containing top k fact check IDs for each post
            
        Returns:
            Tuple containing:
            - general_score: Overall success@10 score
            - by_language: DataFrame with success@10 scores broken down by language
        """
        logging.info("Computing success@10 metrics...")
        
        # Create DataFrame from predictions
        predictions_df = pd.DataFrame({
            'post_id': posts_ids,
            'top_k': top_k_predictions
        })
        
        # Add language information
        predictions_df = predictions_df.merge(
            self.posts[[self.posts_language_col]], 
            left_on='post_id', 
            right_index=True
        )
        
        # Get correct answers
        correct_answers = self.pairs.groupby('post_id')['fact_check_id'].apply(set).to_dict()
        
        # Calculate success scores
        success_scores = []
        for _, row in predictions_df.iterrows():
            post_id = row['post_id']
            predicted_ids = set(row['top_k'])
            success = 1 if post_id in correct_answers and predicted_ids & correct_answers[post_id] else 0
            success_scores.append(success)
            
        predictions_df['success_at_10'] = success_scores
        
        # Calculate metrics
        general_score = predictions_df['success_at_10'].mean()
        by_language = predictions_df.groupby(self.posts_language_col)['success_at_10'].agg(['mean', 'count']).round(3)
        
        logging.info(f"\nGeneral Success@10: {general_score:.3f}")
        logging.info("\nSuccess@10 by language:")
        logging.info(by_language)
        
        return general_score, by_language

    def get_similarities_from_embeddings(self, 
                                       embeddings_dir: str,
                                       posts_ids: list,
                                       same_language_only: bool = False) -> tuple:
        """
        Calculate similarities using pre-computed embeddings.
        
        Args:
            embeddings_dir: Directory containing the saved embeddings
            posts_ids: List of post IDs to process
            same_language_only: Whether to only consider fact checks in the same language as the post
            
        Returns:
            Tuple containing:
            - similarities matrix (list of arrays, as lengths may vary with language filtering)
            - top 10 fact check IDs for each post
        """
        fact_checks_emb_path = os.path.join(embeddings_dir, 'fact_checks.pkl')
        posts_emb_path = os.path.join(embeddings_dir, 'posts.pkl')
        
        # Load embeddings
        if not os.path.exists(fact_checks_emb_path) or not os.path.exists(posts_emb_path):
            raise ValueError(f"Embeddings not found in {embeddings_dir}")
            
        logging.info(f"Loading embeddings from: {embeddings_dir}")
        with open(fact_checks_emb_path, 'rb') as f:
            fact_checks_embeddings = pickle.load(f)
        with open(posts_emb_path, 'rb') as f:
            posts_embeddings = pickle.load(f)
            
        # Get indices for the requested post IDs
        post_indices = [self.posts.index.get_loc(post_id) for post_id in posts_ids]
        posts_embeddings = posts_embeddings[post_indices]
        
        # Initialize arrays for results
        all_top_10_ids = []
        all_similarities = []
        
        # Process each post
        for i, post_id in enumerate(posts_ids):
            post_embedding = posts_embeddings[i:i+1]  # Keep 2D shape
            
            if same_language_only:
                # Get post language
                post_language = self.posts.loc[post_id, self.posts_language_col]
                # Get fact checks in the same language
                same_lang_mask = self.fact_checks[self.posts_language_col] == post_language
                valid_fact_checks_embeddings = fact_checks_embeddings[same_lang_mask]
                valid_fact_checks_indices = np.where(same_lang_mask)[0]
                
                if len(valid_fact_checks_indices) == 0:
                    logging.warning(f"No fact checks found for language {post_language}, using all fact checks for post {post_id}")
                    valid_fact_checks_embeddings = fact_checks_embeddings
                    valid_fact_checks_indices = np.arange(len(fact_checks_embeddings))
            else:
                valid_fact_checks_embeddings = fact_checks_embeddings
                valid_fact_checks_indices = np.arange(len(fact_checks_embeddings))
            
            # Calculate similarities
            similarities = cosine_similarity(post_embedding, valid_fact_checks_embeddings)
            
            # Get top 10 indices (or less if not enough fact checks)
            num_results = min(10, len(valid_fact_checks_indices))
            top_k_local_indices = np.argsort(-similarities[0])[:num_results]
            # Convert to global indices
            top_k_global_indices = valid_fact_checks_indices[top_k_local_indices]
            # Get fact check IDs
            top_k_ids = [int(self.fact_checks.index[j]) for j in top_k_global_indices]
            
            all_top_10_ids.append(top_k_ids)
            all_similarities.append(similarities[0])
        
        return all_similarities, all_top_10_ids

# Example usage:
if __name__ == "__main__":
    # Initialize evaluator with model
    evaluator = EmbeddingsEvaluator('intfloat/multilingual-e5-small', save_model=True)
    
    # Define column names
    columns = {
        'fact_checks_id': 'fact_check_id',
        'fact_checks_text': 'claim_title',
        'posts_id': 'post_id',
        'posts_text': 'text_ocr',
        'posts_language': 'language',
        'pairs_post_id': 'post_id',
        'pairs_fact_check_id': 'fact_check_id'
    }
    
    # Load data with prefixes
    data_paths = {
        'fact_checks': 'data/transformed/fact_checks.csv',
        'posts': 'data/transformed/posts.csv',
        'pairs': 'data/transformed/pairs.csv'
    }
    evaluator.load_data(data_paths, columns, 
                       fact_checks_prefix="query: ",
                       posts_prefix="passage: ")
    
    # Example 1: Generate new embeddings and save them
    test_posts = evaluator.posts.index[:10].tolist()
    similarities, top_10_predictions = evaluator.get_similarities(
        test_posts,
        same_language_only=True,
        save_embeddings=True
    )
    
    # Example 2: Use pre-computed embeddings
    embeddings_dir = os.path.join('models', 'base-models', evaluator.model_name.split('/')[-1], 'embeddings')
    similarities, top_10_predictions = evaluator.get_similarities_from_embeddings(
        embeddings_dir,
        test_posts,
        same_language_only=True
    )
    
    # Evaluate predictions
    general_score, by_language = evaluator.evaluate(test_posts, top_10_predictions) 