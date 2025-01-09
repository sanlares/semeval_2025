import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)

class Predictor:
    def __init__(self, 
                 model_name_or_path: str,
                 json_paths: List[str],
                 data_paths: Dict[str, str],
                 predictions_dir: str,
                 model_type: str = 'base',
                 save_model: bool = True,
                 post_text_column: str = 'text_ocr',
                 fact_check_text_column: str = 'claim_title',
                 post_prefix: Optional[str] = None,
                 fact_check_prefix: Optional[str] = None,
                 post_additional_columns: Optional[List[str]] = None,
                 fact_check_additional_columns: Optional[List[str]] = None):
        """
        Initialize the predictor.
        
        Args:
            model_name_or_path: Name of the HuggingFace model or path to local model
            json_paths: List of paths to JSON files containing post IDs to predict
            data_paths: Dictionary with paths to required data files
            predictions_dir: Directory to save predictions
            model_type: Type of model to load ('base' or 'fine-tuned')
            save_model: Whether to save the base model locally
            post_text_column: Column name in posts dataframe to use for text content
            fact_check_text_column: Column name in fact checks dataframe to use for text content
            post_prefix: Optional prefix to add before post texts (e.g., 'passage: ')
            fact_check_prefix: Optional prefix to add before fact check texts (e.g., 'query: ')
            post_additional_columns: Optional list of column names from posts to include in text
            fact_check_additional_columns: Optional list of column names from fact checks to include in text
        """
        self.model_name = model_name_or_path
        self.json_paths = json_paths
        self.data_paths = data_paths
        self.predictions_dir = predictions_dir
        self.post_text_column = post_text_column
        self.fact_check_text_column = fact_check_text_column
        self.post_prefix = post_prefix
        self.fact_check_prefix = fact_check_prefix
        self.post_additional_columns = post_additional_columns or []
        self.fact_check_additional_columns = fact_check_additional_columns or []
        
        # Load the model
        if model_type == 'base':
            model_path = os.path.join('models', 'base-models', model_name_or_path.split('/')[-1])
            
            if save_model and os.path.exists(model_path):
                logging.info(f"Loading base model from local path: {model_path}")
                self.model = SentenceTransformer(model_path)
            else:
                logging.info(f"Loading base model from HuggingFace: {model_name_or_path}")
                self.model = SentenceTransformer(model_name_or_path)
                if save_model:
                    logging.info(f"Saving model to: {model_path}")
                    os.makedirs(model_path, exist_ok=True)
                    self.model.save(model_path)
        else:  # fine-tuned
            if not os.path.exists(model_name_or_path):
                # Try to find in fine-tuned-models directory
                alt_path = os.path.join('models', 'fine-tuned-models', model_name_or_path)
                if os.path.exists(alt_path):
                    model_name_or_path = alt_path
                else:
                    raise ValueError(f"Fine-tuned model not found at {model_name_or_path} or {alt_path}")
            
            logging.info(f"Loading fine-tuned model from: {model_name_or_path}")
            self.model = SentenceTransformer(model_name_or_path)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.load_data()

    def load_data(self):
        """Load and prepare the data for predictions."""
        logging.info("Loading data...")
        
        # Load fact checks
        self.fact_checks = pd.read_csv(self.data_paths['fact_checks'])
        self.fact_checks = self.fact_checks.set_index('fact_check_id')
        
        # Format fact checks texts with optional prefix and additional columns
        base_fact_check_text = self.fact_checks[self.fact_check_text_column].fillna('')
        additional_fact_check_text = ''
        if self.fact_check_additional_columns:
            for col in self.fact_check_additional_columns:
                if col in self.fact_checks.columns:
                    additional_fact_check_text += f" {col} {self.fact_checks[col].fillna('')}"
        
        if self.fact_check_prefix is not None:
            self.fact_checks['text_formatted'] = self.fact_check_prefix + additional_fact_check_text + ' ' + base_fact_check_text
        else:
            self.fact_checks['text_formatted'] = additional_fact_check_text + ' ' + base_fact_check_text
        
        # Load posts
        self.posts = pd.read_csv(self.data_paths['posts'])
        self.posts = self.posts.set_index('post_id')
        
        # Format posts texts with optional prefix and additional columns
        base_post_text = self.posts[self.post_text_column].fillna('')
        additional_post_text = ''
        if self.post_additional_columns:
            for col in self.post_additional_columns:
                if col in self.posts.columns:
                    additional_post_text += f" {col} {self.posts[col].fillna('')}"
        
        if self.post_prefix is not None:
            self.posts['text_formatted'] = self.post_prefix + additional_post_text + ' ' + base_post_text
        else:
            self.posts['text_formatted'] = additional_post_text + ' ' + base_post_text
        
        logging.info("Data loaded and formatted successfully")
        
        # Generate fact checks embeddings once
        fact_checks_texts = self.fact_checks['text_formatted'].tolist()
        self.fact_checks_embeddings = self.generate_embeddings(fact_checks_texts)

    def load_json_data(self, path: str) -> dict:
        """Load post IDs from JSON file."""
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        logging.info("Generating embeddings...")
        try:
            embeddings = self.model.encode(texts, 
                                         batch_size=16, 
                                         show_progress_bar=True,
                                         convert_to_numpy=True,
                                         normalize_embeddings=True)
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise

    def get_text_for_post_id(self, post_id: str) -> str:
        """Get the formatted text for a given post ID."""
        try:
            return str(self.posts.loc[int(post_id), 'text_formatted'])
        except KeyError:
            logging.warning(f"Post ID {post_id} not found in posts data")
            return f"Post {post_id} not found"

    def predict(self) -> dict:
        """Generate predictions for all JSON files."""
        predictions = {}
        
        for json_path in self.json_paths:
            logging.info(f"Processing {json_path}")
            data = self.load_json_data(json_path)
            post_ids = list(data.keys())
            
            # Get texts and generate embeddings for posts
            texts = [self.get_text_for_post_id(post_id) for post_id in post_ids]
            posts_embeddings = self.generate_embeddings(texts)
            
            # Compute similarities between posts and fact checks
            try:
                similarities = cosine_similarity(posts_embeddings, self.fact_checks_embeddings)
                top_10_indices = np.argsort(-similarities, axis=1)[:, :10]
            except ValueError as e:
                logging.error(f"Dimension mismatch - Posts shape: {posts_embeddings.shape}, Fact checks shape: {self.fact_checks_embeddings.shape}")
                raise
            
            # Create predictions dictionary
            for i, post_id in enumerate(post_ids):
                top_fact_check_ids = [int(self.fact_checks.index[j]) for j in top_10_indices[i]]
                predictions[post_id] = top_fact_check_ids
            
            # Save predictions
            if self.predictions_dir:
                json_filename = os.path.basename(json_path)
                output_path = os.path.join(self.predictions_dir, json_filename)
                os.makedirs(self.predictions_dir, exist_ok=True)
            else:
                output_path = json_path

            with open(output_path, 'w') as file:
                json.dump(predictions, file, indent=4)
            
            logging.info(f"Predictions saved to {output_path}")
            
        return predictions

# Example usage:
if __name__ == "__main__":
    # Example configuration
    data_paths = {
        'fact_checks': 'data/transformed/fact_checks.csv',
        'posts': 'data/transformed/posts.csv'
    }
    
    json_paths = [
        'data/original/monolingual_predictions.json',
        'data/original/crosslingual_predictions.json'
    ]
    
    predictor = Predictor(
        model_name_or_path='intfloat/multilingual-e5-small',
        json_paths=json_paths,
        data_paths=data_paths,
        predictions_dir='predictions',
        model_type='base',
        save_model=True,
        post_text_column='text_ocr',
        fact_check_text_column='claim_title',
        post_prefix="passage: ",
        fact_check_prefix="query: ",
        post_additional_columns=['language', 'domain'],  # Optional
        fact_check_additional_columns=['language']  # Optional
    )
    
    predictions = predictor.predict() 