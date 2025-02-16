import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging
from typing import Optional, List
import os
import pickle
import json
import faiss

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)

class EmbeddingsEvaluator:
    def __init__(self, model_name_or_path: str, model_type: str = 'base', save_model: bool = True):
        """
        Initialize the evaluator with a SentenceTransformer model.
        
        Args:
            model_name_or_path: Name of the HuggingFace model, path to local model, or directory containing language-specific models
            model_type: Type of model to load ('base', 'fine-tuned', or 'language-specific')
            save_model: Whether to save the model locally (only applies to base models)
        """
        self.model_name = model_name_or_path
        self.model_type = model_type
        
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
        """
        Calculate similarities between given posts and fact checks.
        
        Args:
            posts_ids: List of post IDs to process. If None and in prediction mode, uses post_ids_to_predict
            same_language_only: Whether to only consider fact checks in the same language as the post
            save_embeddings: Whether to save the generated embeddings
            predictions_output_path: Path where to save predictions JSON file (only used in prediction mode)
            force_regenerate: If True, regenerate embeddings even if they exist in cache
            use_faiss: Whether to use FAISS for fast similarity search (recommended for large datasets)
            faiss_index_type: Type of FAISS index to use ('flat' for exact search, 'ivf' for approximate)
            
        Returns:
            Tuple containing:
            - similarities matrix (list of arrays, as lengths may vary with language filtering)
            - top 10 fact check IDs for each post
        """
        # If no posts_ids provided and in prediction mode, use post_ids_to_predict
        if posts_ids is None and hasattr(self, 'post_ids_to_predict'):
            posts_ids = self.post_ids_to_predict
        elif posts_ids is None:
            raise ValueError("posts_ids must be provided when not in prediction mode")
            
        # Initialize results containers
        all_top_10_ids = []
        all_similarities = []
        
        if self.model_type == 'language-specific':
            # Group posts by language
            posts_by_language = {}
            for post_id in posts_ids:
                lang = self.posts.loc[post_id, self.posts_language_col]
                if lang not in posts_by_language:
                    posts_by_language[lang] = []
                posts_by_language[lang].append(post_id)
            
            # Process each language separately
            for lang, lang_post_ids in posts_by_language.items():
                logging.info(f"Processing language: {lang}")
                
                # Load language-specific model
                lang_model_path = os.path.join(self.model_name, lang, lang)
                
                if not os.path.exists(lang_model_path):
                    raise ValueError(f"No model found for language {lang} at {lang_model_path}")
                
                logging.info(f"Loading language-specific model from: {lang_model_path}")
                lang_model = SentenceTransformer(lang_model_path)
                lang_model.to(self.device)
                
                try:
                    # Get fact checks for this language
                    fact_checks_subset = self.fact_checks[self.fact_checks[self.posts_language_col] == lang]
                    
                    if len(fact_checks_subset) == 0:
                        logging.warning(f"No fact checks found for language {lang}, skipping")
                        continue
                    
                    # Generate embeddings for fact checks
                    fact_checks_texts = fact_checks_subset[self.fact_checks_text_col].tolist()
                    fact_checks_embeddings = lang_model.encode(
                        fact_checks_texts,
                        batch_size=16,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    
                    # Generate embeddings for posts
                    posts_texts = [str(self.posts.loc[post_id, self.posts_text_col]) for post_id in lang_post_ids]
                    posts_embeddings = lang_model.encode(
                        posts_texts,
                        batch_size=16,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    
                    # Calculate similarities
                    similarities = cosine_similarity(posts_embeddings, fact_checks_embeddings)
                    
                    # Get top 10 for each post
                    for i, post_id in enumerate(lang_post_ids):
                        num_results = min(10, len(fact_checks_subset))
                        top_k_indices = np.argsort(-similarities[i])[:num_results]
                        top_k_ids = [int(fact_checks_subset.index[j]) for j in top_k_indices]
                        
                        all_top_10_ids.append(top_k_ids)
                        all_similarities.append(similarities[i])
                    
                    # Clear GPU memory
                    del lang_model
                    del fact_checks_embeddings
                    del posts_embeddings
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logging.error(f"Error processing language {lang}: {str(e)}")
                    del lang_model
                    torch.cuda.empty_cache()
                    raise
                    
            # Save predictions if needed
            if predictions_output_path and hasattr(self, 'post_ids_to_predict'):
                self.save_predictions(posts_ids, all_top_10_ids, predictions_output_path)
                
            return all_similarities, all_top_10_ids
        else:
            # Original implementation for non-language-specific processing
            # Determine embeddings directory based on model type and name
            if self.model_type == 'base':
                embeddings_dir = os.path.join('models', 'base-models', self.model_name.split('/')[-1], 'embeddings')
            else:
                model_name = os.path.basename(self.model_name)
                embeddings_dir = os.path.join('models', 'fine-tuned-models', model_name, 'embeddings')
                
            fact_checks_emb_path = os.path.join(embeddings_dir, 'fact_checks.pkl')
            posts_emb_path = os.path.join(embeddings_dir, 'posts.pkl')
            
            # Generate or load fact checks embeddings
            if not force_regenerate and save_embeddings and os.path.exists(fact_checks_emb_path):
                logging.info(f"Loading fact checks embeddings from: {fact_checks_emb_path}")
                with open(fact_checks_emb_path, 'rb') as f:
                    fact_checks_embeddings = pickle.load(f)
                    
                # Verify embeddings dimension matches current model
                sample_text = self.fact_checks[self.fact_checks_text_col].iloc[0]
                sample_embedding = self.generate_embeddings([sample_text])[0]
                if sample_embedding.shape[0] != fact_checks_embeddings[0].shape[0]:
                    logging.info(f"Saved embeddings dimension ({fact_checks_embeddings[0].shape[0]}) doesn't match current model ({sample_embedding.shape[0]}). Regenerating embeddings...")
                    fact_checks_texts = self.fact_checks[self.fact_checks_text_col].tolist()
                    fact_checks_embeddings = self.generate_embeddings(
                        fact_checks_texts,
                        save_path=fact_checks_emb_path if save_embeddings else None
                    )
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
            
            # After generating embeddings, choose search method
            if use_faiss:
                logging.info("Using FAISS for similarity search...")
                all_similarities = []
                all_top_10_ids = []
                
                # Process each language separately if same_language_only
                if same_language_only:
                    for i, post_id in enumerate(posts_ids):
                        post_language = self.posts.loc[post_id, self.posts_language_col]
                        fact_checks_subset = self.fact_checks[self.fact_checks[self.posts_language_col] == post_language]
                        
                        if len(fact_checks_subset) == 0:
                            logging.warning(f"No fact checks found for language {post_language}, using all fact checks for post {post_id}")
                            valid_fact_checks = fact_checks_embeddings
                            valid_indices = np.arange(len(fact_checks_embeddings))
                        else:
                            valid_indices = fact_checks_subset.index.map(lambda x: self.fact_checks.index.get_loc(x))
                            valid_fact_checks = fact_checks_embeddings[valid_indices]
                        
                        # Create FAISS index for this language
                        dimension = valid_fact_checks.shape[1]
                        if faiss_index_type == 'flat':
                            index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
                        else:  # 'ivf'
                            nlist = min(int(np.sqrt(len(valid_fact_checks))), 100)  # number of clusters
                            quantizer = faiss.IndexFlatIP(dimension)
                            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                            index.train(valid_fact_checks)
                            index.nprobe = min(20, nlist)  # number of clusters to visit during search
                        
                        # Add vectors to index
                        index.add(valid_fact_checks)
                        
                        # Search
                        post_emb = posts_embeddings[i].reshape(1, -1)
                        similarities, indices = index.search(post_emb, min(10, len(valid_fact_checks)))
                        
                        # Convert local indices to global fact check IDs
                        if len(fact_checks_subset) > 0:
                            top_k_ids = [int(fact_checks_subset.index[valid_indices[j]]) for j in indices[0]]
                        else:
                            top_k_ids = [int(self.fact_checks.index[j]) for j in indices[0]]
                        
                        all_top_10_ids.append(top_k_ids)
                        all_similarities.append(similarities[0])
                else:
                    # Create single FAISS index for all fact checks
                    dimension = fact_checks_embeddings.shape[1]
                    if faiss_index_type == 'flat':
                        index = faiss.IndexFlatIP(dimension)
                    else:  # 'ivf'
                        nlist = min(int(np.sqrt(len(fact_checks_embeddings))), 100)
                        quantizer = faiss.IndexFlatIP(dimension)
                        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                        index.train(fact_checks_embeddings)
                        index.nprobe = min(20, nlist)
                    
                    index.add(fact_checks_embeddings)
                    
                    # Search for all posts at once
                    similarities, indices = index.search(posts_embeddings, 10)
                    
                    # Convert indices to fact check IDs
                    for i in range(len(posts_ids)):
                        top_k_ids = [int(self.fact_checks.index[j]) for j in indices[i]]
                        all_top_10_ids.append(top_k_ids)
                        all_similarities.append(similarities[i])
                
                logging.info("Finished FAISS similarity search")
            else:
                # Original similarity computation code
                logging.info("Computing similarities using original method...")
                all_top_10_ids = []
                all_similarities = []
                
                for i, post_id in enumerate(posts_ids):
                    post_embedding = posts_embeddings[i].reshape(1, -1)
                    
                    if same_language_only:
                        post_language = self.posts.loc[post_id, self.posts_language_col]
                        fact_checks_subset = self.fact_checks[self.fact_checks[self.posts_language_col] == post_language]
                        
                        if len(fact_checks_subset) == 0:
                            logging.warning(f"No fact checks found for language {post_language}, using all fact checks for post {post_id}")
                            valid_fact_checks_embeddings = fact_checks_embeddings
                            valid_fact_checks_indices = np.arange(len(fact_checks_embeddings))
                        else:
                            valid_fact_checks_indices = fact_checks_subset.index.map(lambda x: self.fact_checks.index.get_loc(x))
                            valid_fact_checks_embeddings = fact_checks_embeddings[valid_fact_checks_indices]
                    else:
                        valid_fact_checks_embeddings = fact_checks_embeddings
                        valid_fact_checks_indices = np.arange(len(fact_checks_embeddings))
                    
                    try:
                        similarities = cosine_similarity(post_embedding, valid_fact_checks_embeddings)
                    except ValueError as e:
                        logging.error(f"Dimension mismatch - Post embedding shape: {post_embedding.shape}, Fact check embeddings shape: {valid_fact_checks_embeddings.shape}")
                        raise
                    
                    num_results = min(10, len(valid_fact_checks_indices))
                    top_k_local_indices = np.argsort(-similarities[0])[:num_results]
                    top_k_global_indices = valid_fact_checks_indices[top_k_local_indices]
                    top_k_ids = [int(self.fact_checks.index[j]) for j in top_k_global_indices]
                    
                    all_top_10_ids.append(top_k_ids)
                    all_similarities.append(similarities[0])
                
                logging.info("Finished computing similarities using original method")
            
            # Save predictions if needed
            if predictions_output_path and hasattr(self, 'post_ids_to_predict'):
                self.save_predictions(posts_ids, all_top_10_ids, predictions_output_path)
            
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