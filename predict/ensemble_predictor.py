import os
import json
import logging
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble prediction."""
    strategy: str = "frequency"  # Options: frequency, weighted_frequency, rank_based, language_weighted
    top_k: int = 10  # Number of predictions to return
    language_weights: Optional[Dict[str, float]] = None  # Weights for each language when using language_weighted
    model_weights: Optional[Dict[str, float]] = None  # Weights for each model in weighted strategies
    rank_decay: float = 0.95  # Decay factor for rank-based scoring (0.95 means each rank is worth 95% of the previous)

class EnsemblePredictor:
    def __init__(self, config: EnsembleConfig = None):
        """
        Initialize the ensemble predictor.
        
        Args:
            config: Configuration for ensemble prediction
        """
        self.config = config or EnsembleConfig()
        
    def load_predictions(self, prediction_files: List[str], model_name_mapping: Optional[Dict[str, str]] = None) -> Dict[str, List[Dict[str, List[int]]]]:
        """
        Load predictions from multiple JSON files.
        
        Args:
            prediction_files: List of paths to prediction JSON files
            model_name_mapping: Optional dictionary to map file paths to model names.
                              If not provided, will use the parent directory name.
                              Example: {
                                  "models/.../model1/pred.json": "multilingual-e5-large",
                                  "models/.../model2/pred.json": "multilingual-e5-base"
                              }
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        predictions = {}
        for file_path in prediction_files:
            if model_name_mapping and file_path in model_name_mapping:
                model_name = model_name_mapping[file_path]
            else:
                # Use parent directory name as model name
                model_name = os.path.basename(os.path.dirname(file_path))
                
            try:
                with open(file_path, 'r') as f:
                    predictions[model_name] = json.load(f)
                logging.info(f"Loaded predictions from {file_path} as model '{model_name}'")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
                continue
                
        # Log weights being used for each model
        if self.config.model_weights:
            logging.info("\nModel weights:")
            for model in predictions.keys():
                weight = self._get_model_weight(model)
                logging.info(f"  {model}: {weight}")
        
        return predictions
    
    def _get_model_weight(self, model_name: str) -> float:
        """Get weight for a specific model."""
        if self.config.model_weights and model_name in self.config.model_weights:
            return self.config.model_weights[model_name]
        return 1.0
    
    def _get_language_weight(self, language: str) -> float:
        """Get weight for a specific language."""
        if self.config.language_weights and language in self.config.language_weights:
            return self.config.language_weights[language]
        return 1.0
    
    def _frequency_based_ensemble(self, predictions_list: List[List[int]]) -> List[int]:
        """Simple frequency-based ensemble."""
        # Count occurrences of each fact check ID
        counter = Counter()
        for preds in predictions_list:
            counter.update(preds)
        
        # Get top k most common
        return [fact_id for fact_id, _ in counter.most_common(self.config.top_k)]
    
    def _weighted_frequency_ensemble(self, predictions_list: List[List[int]], 
                                   model_names: List[str]) -> List[int]:
        """Weighted frequency-based ensemble."""
        counter = Counter()
        for preds, model_name in zip(predictions_list, model_names):
            weight = self._get_model_weight(model_name)
            for fact_id in preds:
                counter[fact_id] += weight
        
        return [fact_id for fact_id, _ in counter.most_common(self.config.top_k)]
    
    def _rank_based_ensemble(self, predictions_list: List[List[int]], 
                           model_names: List[str]) -> List[int]:
        """Rank-based ensemble with exponential decay."""
        scores = Counter()
        for preds, model_name in zip(predictions_list, model_names):
            weight = self._get_model_weight(model_name)
            for rank, fact_id in enumerate(preds):
                # Score decreases exponentially with rank
                score = weight * (self.config.rank_decay ** rank)
                scores[fact_id] += score
        
        return [fact_id for fact_id, _ in scores.most_common(self.config.top_k)]
    
    def _language_weighted_ensemble(self, predictions_list: List[List[int]], 
                                  model_names: List[str],
                                  language: str) -> List[int]:
        """Language-weighted ensemble."""
        counter = Counter()
        lang_weight = self._get_language_weight(language)
        
        for preds, model_name in zip(predictions_list, model_names):
            model_weight = self._get_model_weight(model_name)
            weight = model_weight * lang_weight
            for fact_id in preds:
                counter[fact_id] += weight
        
        return [fact_id for fact_id, _ in counter.most_common(self.config.top_k)]
    
    def ensemble_predictions(self, 
                           predictions: Dict[str, Dict[str, List[int]]],
                           posts_languages: Optional[Dict[str, str]] = None) -> Dict[str, List[int]]:
        """
        Create ensemble predictions from multiple models.
        
        Args:
            predictions: Dictionary mapping model names to their predictions
            posts_languages: Optional dictionary mapping post IDs to their languages
            
        Returns:
            Dictionary mapping post IDs to ensembled predictions
        """
        ensembled = {}
        model_names = list(predictions.keys())
        
        # Get all unique post IDs
        all_post_ids = set()
        for model_preds in predictions.values():
            all_post_ids.update(model_preds.keys())
        
        for post_id in all_post_ids:
            # Get predictions from all models for this post
            post_predictions = []
            available_models = []
            
            for model_name in model_names:
                if post_id in predictions[model_name]:
                    post_predictions.append(predictions[model_name][post_id])
                    available_models.append(model_name)
            
            if not post_predictions:
                logging.warning(f"No predictions found for post {post_id}")
                continue
            
            # Apply ensemble strategy
            if self.config.strategy == "frequency":
                ensembled[post_id] = self._frequency_based_ensemble(post_predictions)
            elif self.config.strategy == "weighted_frequency":
                ensembled[post_id] = self._weighted_frequency_ensemble(post_predictions, available_models)
            elif self.config.strategy == "rank_based":
                ensembled[post_id] = self._rank_based_ensemble(post_predictions, available_models)
            elif self.config.strategy == "language_weighted" and posts_languages:
                language = posts_languages.get(post_id)
                if language:
                    ensembled[post_id] = self._language_weighted_ensemble(
                        post_predictions, available_models, language
                    )
                else:
                    ensembled[post_id] = self._frequency_based_ensemble(post_predictions)
            else:
                ensembled[post_id] = self._frequency_based_ensemble(post_predictions)
        
        return ensembled
    
    def analyze_ensemble(self, 
                        predictions: Dict[str, Dict[str, List[int]]],
                        ground_truth: Optional[Dict[str, List[int]]] = None) -> pd.DataFrame:
        """
        Analyze the agreement between different models and optionally compare with ground truth.
        
        Args:
            predictions: Dictionary mapping model names to their predictions
            ground_truth: Optional dictionary mapping post IDs to correct fact check IDs
            
        Returns:
            DataFrame with analysis metrics
        """
        model_names = list(predictions.keys())
        analysis_data = []
        
        # Get all unique post IDs
        all_post_ids = set()
        for model_preds in predictions.values():
            all_post_ids.update(model_preds.keys())
        
        for post_id in all_post_ids:
            row = {'post_id': post_id}
            
            # Get predictions from all models
            post_predictions = {
                model: preds[post_id] if post_id in preds else []
                for model, preds in predictions.items()
            }
            
            # Calculate agreement metrics
            all_predictions = [pred for pred in post_predictions.values() if pred]
            if len(all_predictions) > 1:
                # Calculate overlap between models
                common_predictions = set.intersection(*[set(p) for p in all_predictions])
                row['num_common_predictions'] = len(common_predictions)
                
                # Calculate average overlap between pairs of models
                pair_overlaps = []
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        if post_id in predictions[model_names[i]] and post_id in predictions[model_names[j]]:
                            overlap = len(set(predictions[model_names[i]][post_id]) & 
                                       set(predictions[model_names[j]][post_id]))
                            pair_overlaps.append(overlap)
                
                if pair_overlaps:
                    row['avg_pairwise_overlap'] = np.mean(pair_overlaps)
                    row['min_pairwise_overlap'] = min(pair_overlaps)
                    row['max_pairwise_overlap'] = max(pair_overlaps)
            
            # Compare with ground truth if available
            if ground_truth and post_id in ground_truth:
                correct_ids = set(ground_truth[post_id])
                for model_name, preds in post_predictions.items():
                    if preds:
                        hits = len(set(preds) & correct_ids)
                        row[f'{model_name}_hits'] = hits
            
            analysis_data.append(row)
        
        return pd.DataFrame(analysis_data)
    
    def save_predictions(self, predictions: Dict[str, List[int]], output_path: str):
        """Save ensemble predictions to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=4)
        logging.info(f"Saved ensemble predictions to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Example configuration with model weights
    config = EnsembleConfig(
        strategy="weighted_frequency",
        model_weights={
            "multilingual-e5-large-instruct": 1.2,  # Boost predictions from this model
            "multilingual-e5-base": 0.8,  # Lower weight for this model
            "multilingual-e5-large": 1.0   # Neutral weight
        }
    )
    
    predictor = EnsemblePredictor(config)
    
    # Example prediction files with explicit model name mapping
    prediction_files = [
        "models/fine-tuned-models/multilingual-e5-large-instruct/predictions/monolingual_predictions.json",
        "models/fine-tuned-models/multilingual-e5-base/predictions/monolingual_predictions.json",
        "models/fine-tuned-models/multilingual-e5-large/predictions/monolingual_predictions.json"
    ]
    
    # Optional: explicitly map files to model names
    model_mapping = {
        "models/fine-tuned-models/multilingual-e5-large-instruct/predictions/monolingual_predictions.json": "multilingual-e5-large-instruct",
        "models/fine-tuned-models/multilingual-e5-base/predictions/monolingual_predictions.json": "multilingual-e5-base",
        "models/fine-tuned-models/multilingual-e5-large/predictions/monolingual_predictions.json": "multilingual-e5-large"
    }
    
    # Load predictions with explicit model names
    predictions = predictor.load_predictions(prediction_files, model_mapping)
    
    # Example posts languages (in real usage, load from your data)
    posts_languages = {
        "1": "tha",
        "2": "eng"
    }
    
    ensembled = predictor.ensemble_predictions(predictions, posts_languages)
    
    # Analyze ensemble
    analysis = predictor.analyze_ensemble(predictions)
    print("\nEnsemble Analysis:")
    print(analysis)
    
    # Save results
    predictor.save_predictions(ensembled, "predictions/ensemble_predictions.json") 