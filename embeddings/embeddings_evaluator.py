import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging
from typing import Optional, List, Union, Dict
import os
import pickle
import json
import faiss
import time
import openai
import concurrent.futures
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm
import threading

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)


class OpenAIEmbedder:
    """Wrapper class to make OpenAI embeddings API compatible with SentenceTransformer interface"""
    
    def __init__(self, model_name: str = "text-embedding-3-small", cache_dir: Optional[str] = None,
                 prefix: str = "", batch_size: int = 16, max_workers: int = 10):
        """Initialize OpenAI embedder"""
        self.model_name = model_name
        self.prefix = prefix
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.client = openai.OpenAI()
        
        # Setup cache directory with model and prefix
        prefix_slug = prefix.strip().replace(" ", "_")[:30] if prefix else "no_prefix"
        self.cache_dir = Path(cache_dir) if cache_dir else Path("models/openai-cache")
        self.cache_dir = self.cache_dir / f"{model_name}_{prefix_slug}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats tracking
        self.stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "api_time": 0.0,
            "concurrent_max": 0
        }
        
        # Model dimensions
        self.embedding_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if model_name not in self.embedding_dims:
            raise ValueError(f"Unknown model: {model_name}. Supported models: {list(self.embedding_dims.keys())}")
            
        self.dimension = self.embedding_dims[model_name]
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.text_to_id = {}
        self.id_to_text = {}
        
        # Concurrency control
        self.lock = threading.Lock()
        self.active_requests = 0
        self.request_times = []
        
        logging.info(f"Initialized OpenAI embedder:")
        logging.info(f"  Model: {model_name}")
        logging.info(f"  Cache directory: {self.cache_dir}")
        logging.info(f"  Prefix: '{prefix}'")
        logging.info(f"  Batch size: {batch_size}")
        logging.info(f"  Max workers: {max_workers}")
        
    def _get_cache_path(self, text: str) -> Path:
        """Get cache file path for a given text"""
        # Use hash of text with prefix to ensure uniqueness
        text_with_prefix = f"{self.prefix}{text}"
        text_hash = str(hash(text_with_prefix))
        return self.cache_dir / f"{text_hash}.npy"
    
    def _log_stats(self):
        """Log current statistics"""
        avg_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        logging.info("\nEmbedding Statistics:")
        logging.info(f"  API calls: {self.stats['api_calls']}")
        logging.info(f"  Cache hits: {self.stats['cache_hits']}")
        logging.info(f"  Total tokens: {self.stats['total_tokens']}")
        logging.info(f"  Average API time: {avg_time:.2f}s")
        logging.info(f"  Max concurrent requests: {self.stats['concurrent_max']}")
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text with caching"""
        cache_path = self._get_cache_path(text)
        
        # Check disk cache
        if cache_path.exists():
            with self.lock:
                self.stats["cache_hits"] += 1
            return np.load(cache_path)
        
        # Track concurrent requests
        with self.lock:
            self.active_requests += 1
            self.stats["concurrent_max"] = max(self.stats["concurrent_max"], self.active_requests)
        
        try:
            start_time = time.time()
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(text.split())
            with self.lock:
                self.stats["total_tokens"] += estimated_tokens
            
            # Call OpenAI API (synchronously)
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            
            embedding = np.array(response.data[0].embedding)
            
            # Update stats
            api_time = time.time() - start_time
            with self.lock:
                self.stats["api_calls"] += 1
                self.request_times.append(api_time)
                self.stats["api_time"] += api_time
            
            # Save to disk cache
            np.save(cache_path, embedding)
            
            # Add to FAISS index
            with self.lock:
                if text not in self.text_to_id:
                    idx = len(self.text_to_id)
                    self.text_to_id[text] = idx
                    self.id_to_text[idx] = text
                    self.index.add(embedding.reshape(1, -1))
            
            return embedding
            
        except Exception as e:
            logging.error(f"Error getting embedding from OpenAI API: {str(e)}")
            raise
        finally:
            with self.lock:
                self.active_requests -= 1
    
    def encode(self, 
              sentences: Union[str, List[str]],
              batch_size: Optional[int] = None,
              show_progress_bar: bool = True,
              convert_to_numpy: bool = True,
              normalize_embeddings: bool = True,
              **kwargs) -> np.ndarray:
        """Generate embeddings for the given sentences using OpenAI's API"""
        if isinstance(sentences, str):
            sentences = [sentences]
            
        batch_size = batch_size or self.batch_size
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        
        all_embeddings = []
        with tqdm(total=len(sentences), desc="Processing texts", disable=not show_progress_bar) as pbar:
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                # Process batch in parallel using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    batch_embeddings = list(executor.map(self._get_embedding, batch))
                
                all_embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
                
                # Log intermediate stats every 5 batches
                if (i // batch_size + 1) % 5 == 0:
                    self._log_stats()
        
        embeddings = np.array(all_embeddings)
        if normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
        # Log final stats
        self._log_stats()
        return embeddings
    
    def search(self, query_embeddings: np.ndarray, k: int = 10) -> tuple:
        """
        Search for nearest neighbors in the FAISS index
        
        Args:
            query_embeddings: Query embeddings to search for
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
            
        with self.lock:
            # Ensure we don't request more neighbors than we have in the index
            actual_k = min(k, self.index.ntotal)
            if actual_k == 0:
                logging.warning("FAISS index is empty, no search results will be returned")
                return np.array([[]]), [[]]
                
            distances, indices = self.index.search(query_embeddings, actual_k)
            
            # Convert indices to texts
            texts = []
            for query_indices in indices:
                query_texts = []
                for idx in query_indices:
                    if idx in self.id_to_text:  # Check if index exists
                        query_texts.append(self.id_to_text[idx])
                texts.append(query_texts)
        
        return distances, texts
    
    def to(self, device):
        """Mock method for compatibility with PyTorch models"""
        pass

class EmbeddingsEvaluator:
    def __init__(self, model_name_or_path: str, model_type: str = 'base', save_model: bool = True):
        """
        Initialize the evaluator with a model.
        
        Args:
            model_name_or_path: Name of the model to use:
                - HuggingFace model name for 'base' type
                - Path to local model for 'fine-tuned' type
                - Directory with language models for 'language-specific' type
                - OpenAI model name for 'openai' type
            model_type: Type of model to load ('base', 'fine-tuned', 'language-specific', or 'openai')
            save_model: Whether to save the model locally (only applies to base models)
        """
        self.model_name = model_name_or_path
        self.model_type = model_type
        self._fact_checks_prefix = None
        self._posts_prefix = None
        
        # Handle OpenAI models
        if model_type == 'openai':
            logging.info(f"Initializing OpenAI embeddings model: {model_name_or_path}")
            cache_dir = os.path.join('models', 'openai-cache', model_name_or_path)
            # We'll initialize the model later when we have the prefixes
            self.model = None
            self.device = "cpu"  # OpenAI API runs remotely
            return
        
        # Only load a model if not using language-specific models
        if model_type != 'language-specific':
            # Determine model path based on type
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
                
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        else:
            # For language-specific models, verify the directory exists and contains model subdirectories
            if not os.path.exists(model_name_or_path):
                raise ValueError(f"Language models directory not found at {model_name_or_path}")
            
            # Check if directory contains language subdirectories
            lang_dirs = [d for d in os.listdir(model_name_or_path) 
                        if os.path.isdir(os.path.join(model_name_or_path, d))]
            if not lang_dirs:
                raise ValueError(f"No language model directories found in {model_name_or_path}")
                
            logging.info(f"Found language models for: {', '.join(lang_dirs)}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = None  # No base model for language-specific mode
        
    @classmethod
    def from_pretrained(cls, model_path: str, model_type: str = 'base') -> 'EmbeddingsEvaluator':
        """
        Load a model from a local path.
        
        Args:
            model_path: Path to the saved model
            model_type: Type of model to load ('base' or 'fine-tuned')
            
        Returns:
            EmbeddingsEvaluator instance
        """
        return cls(model_path, model_type=model_type, save_model=False)
        
    def load_data(self, 
                 data_paths: dict,
                 columns: dict,
                 mode: str = 'evaluation',  # 'evaluation' or 'prediction'
                 language: Optional[str] = None,
                 fact_checks_prefix: Optional[str] = None,
                 posts_prefix: Optional[str] = None) -> None:
        """
        Load the necessary data files.
        
        Args:
            data_paths: Dictionary containing paths to required files:
                - fact_checks: path to fact checks CSV
                - posts: path to posts CSV
                - pairs: path to pairs CSV for evaluation (only required in evaluation mode)
                - predictions_json: path to predictions JSON file (only required in prediction mode)
            columns: Dictionary containing column names to use:
                - fact_checks_id: column name for fact check IDs
                - fact_checks_text: column name for fact check text to embed
                - posts_id: column name for post IDs
                - posts_text: column name for post text to embed
                - posts_language: column name for language in posts
                - pairs_post_id: column name for post IDs in pairs (only for evaluation mode)
                - pairs_fact_check_id: column name for fact check IDs in pairs (only for evaluation mode)
            mode: Operation mode - 'evaluation' or 'prediction'
            language: Optional language to filter posts by. If None, use all languages.
            fact_checks_prefix: Optional prefix to add to fact check texts (e.g., "query: ")
            posts_prefix: Optional prefix to add to post texts (e.g., "passage: ")
        """
        # Store prefixes
        self._fact_checks_prefix = fact_checks_prefix or ""
        self._posts_prefix = posts_prefix or ""
        
        # Initialize OpenAI model if needed (now that we have prefixes)
        if self.model_type == 'openai' and self.model is None:
            cache_dir = os.path.join('models', 'openai-cache', self.model_name)
            
            # Create separate embedders for fact checks and posts
            self.fact_checks_embedder = OpenAIEmbedder(
                model_name=self.model_name,
                cache_dir=cache_dir,
                prefix=self._fact_checks_prefix
            )
            
            self.posts_embedder = OpenAIEmbedder(
                model_name=self.model_name,
                cache_dir=cache_dir,
                prefix=self._posts_prefix
            )
            
            # Use posts embedder as default for compatibility
            self.model = self.posts_embedder
        
        if mode not in ['evaluation', 'prediction']:
            raise ValueError("Mode must be either 'evaluation' or 'prediction'")
            
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
        if self._fact_checks_prefix:
            self.fact_checks_text_col = f"{columns['fact_checks_text']}_with_prefix"
            self.fact_checks[self.fact_checks_text_col] = self._fact_checks_prefix + self.fact_checks[columns['fact_checks_text']].astype(str)
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
        if self._posts_prefix:
            self.posts_text_col = f"{columns['posts_text']}_with_prefix"
            self.posts[self.posts_text_col] = self._posts_prefix + self.posts[columns['posts_text']].astype(str)
        else:
            self.posts_text_col = columns['posts_text']
            
        self.posts_language_col = columns['posts_language']
        
        # Load pairs or predictions based on mode
        if mode == 'evaluation':
            if 'pairs' not in data_paths:
                raise ValueError("pairs path is required in evaluation mode")
                
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
        else:  # prediction mode
            if 'predictions_json' not in data_paths:
                raise ValueError("predictions_json path is required in prediction mode")
                
            import json
            with open(data_paths['predictions_json'], 'r') as f:
                predictions_dict = json.load(f)
            self.post_ids_to_predict = list(map(int, predictions_dict.keys()))
        
        # Log data info
        logging.info(f"Data loaded successfully. Posts shape: {self.posts.shape}, Fact checks shape: {self.fact_checks.shape}")
        if mode == 'evaluation':
            logging.info(f"Pairs shape: {self.pairs.shape}")
        else:
            logging.info(f"Number of posts to predict: {len(self.post_ids_to_predict)}")
        
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
        try:
            embeddings = self.model.encode(texts,
                                         batch_size=16,
                                         show_progress_bar=True,
                                         convert_to_numpy=True,
                                         normalize_embeddings=True)
            
            if save_path:
                logging.info(f"Saving embeddings to: {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Change extension to .npy if it ends in .pkl
                if save_path.endswith('.pkl'):
                    save_path = save_path[:-4] + '.npy'
                    
                logging.info(f"Starting to save embeddings array of shape {embeddings.shape}")
                np.save(save_path, embeddings)
                logging.info("Finished saving embeddings")
                    
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise
        
    def get_similarities(self, 
                        posts_ids: Optional[List[int]] = None,
                        same_language_only: bool = False,
                        save_embeddings: bool = True,
                        predictions_output_path: Optional[str] = None,
                        force_regenerate: bool = True,
                        use_faiss: bool = False,
                        faiss_index_type: str = 'flat') -> tuple:
        """Calculate similarities between given posts and fact checks."""
        # If no posts_ids provided and in prediction mode, use post_ids_to_predict
        if posts_ids is None and hasattr(self, 'post_ids_to_predict'):
            posts_ids = self.post_ids_to_predict
        elif posts_ids is None:
            raise ValueError("posts_ids must be provided when not in prediction mode")
            
        # Special handling for OpenAI models that use built-in FAISS index
        if self.model_type == 'openai':
            all_similarities = []
            all_top_10_ids = []
            
            # First, process all fact checks to populate FAISS index
            logging.info("Processing fact checks...")
            fact_checks_by_lang = {}
            for lang in self.fact_checks[self.posts_language_col].unique():
                fact_checks_subset = self.fact_checks[self.fact_checks[self.posts_language_col] == lang]
                fact_checks_by_lang[lang] = fact_checks_subset
                
                # Generate embeddings for this language's fact checks
                texts = fact_checks_subset[self.fact_checks_text_col].tolist()
                logging.info(f"Generating embeddings for {len(texts)} fact checks in language '{lang}'")
                _ = self.fact_checks_embedder.encode(texts)  # This will populate FAISS index
                
            # Now process posts in batches
            batch_size = 32  # Process multiple posts at once
            num_batches = (len(posts_ids) + batch_size - 1) // batch_size
            
            logging.info(f"Processing {len(posts_ids)} posts in {num_batches} batches")
            with tqdm(total=len(posts_ids), desc="Processing posts") as pbar:
                for i in range(0, len(posts_ids), batch_size):
                    batch_post_ids = posts_ids[i:i+batch_size]
                    batch_texts = [str(self.posts.loc[pid, self.posts_text_col]) for pid in batch_post_ids]
                    batch_languages = [self.posts.loc[pid, self.posts_language_col] for pid in batch_post_ids]
                    
                    # Get post embeddings in parallel
                    post_embeddings = self.posts_embedder.encode(batch_texts)
                    
                    # Process each post in the batch
                    for j, (post_id, post_embedding, post_lang) in enumerate(zip(batch_post_ids, post_embeddings, batch_languages)):
                        # Get relevant fact checks
                        if same_language_only:
                            valid_fact_checks = fact_checks_by_lang.get(post_lang, pd.DataFrame())
                            if len(valid_fact_checks) == 0:
                                logging.warning(f"No fact checks found for language {post_lang}, using all fact checks for post {post_id}")
                                valid_fact_checks = self.fact_checks
                        else:
                            valid_fact_checks = self.fact_checks
                        
                        # Search in FAISS index
                        distances, texts = self.fact_checks_embedder.search(post_embedding)
                        
                        # Map texts back to fact check IDs
                        fact_check_ids = []
                        similarities = []
                        for text, distance in zip(texts[0], distances[0]):
                            mask = valid_fact_checks[self.fact_checks_text_col] == text
                            if mask.any():
                                fact_check_id = valid_fact_checks.index[mask][0]
                                fact_check_ids.append(int(fact_check_id))
                                similarities.append(float(distance))
                        
                        all_top_10_ids.append(fact_check_ids[:10])
                        all_similarities.append(similarities[:10])
                        pbar.update(1)
                    
                    # Log stats periodically
                    if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(posts_ids):
                        logging.info(f"Processed {min(i + batch_size, len(posts_ids))}/{len(posts_ids)} posts")
                        self.posts_embedder._log_stats()
                        self.fact_checks_embedder._log_stats()
            
            # Save predictions if needed
            if predictions_output_path:
                self.save_predictions(posts_ids, all_top_10_ids, predictions_output_path)
                
            # Log final stats
            logging.info("\nFinal Stats - Posts Embedder:")
            self.posts_embedder._log_stats()
            logging.info("\nFinal Stats - Fact Checks Embedder:")
            self.fact_checks_embedder._log_stats()
            
            return all_similarities, all_top_10_ids
            
        # Original implementation for other model types
        # ... rest of the existing implementation ...

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

    def save_predictions(self, post_ids: List[int], top_k_predictions: List[List[int]], output_path: str) -> None:
        """
        Save predictions to a JSON file in the required format.
        
        Args:
            post_ids: List of post IDs
            top_k_predictions: List of lists containing top k fact check IDs for each post
            output_path: Path where to save the predictions JSON file
        """
        predictions_dict = {str(post_id): pred_list for post_id, pred_list in zip(post_ids, top_k_predictions)}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save predictions
        with open(output_path, 'w') as f:
            json.dump(predictions_dict, f, indent=4)
        logging.info(f"Predictions saved to: {output_path}")

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