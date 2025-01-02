import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch

class Predictor:
    def __init__(self, model_path, json_paths, data_paths, predictions_dir):
        print(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.json_paths = json_paths
        self.data_paths = data_paths
        self.predictions_dir = predictions_dir
        self.load_data()

    def load_data(self):
        print("Loading data...")
        self.fact_checks = pd.read_csv(self.data_paths['fact_checks'])
        self.posts = pd.read_csv(self.data_paths['posts'])
        print("Data loaded successfully")
        
        # Format texts as required by the model
        self.posts = self.posts.set_index('post_id')
        self.fact_checks = self.fact_checks.set_index('fact_check_id')
              
        # Generate fact checks embeddings once
        self.fact_checks_texts = self.fact_checks['claim_title'].tolist()
        self.fact_checks_embeddings = self.generate_embeddings(self.fact_checks_texts)


    def load_json_data(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def generate_embeddings(self, texts):
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, 
                                     batch_size=16, 
                                     show_progress_bar=True,
                                     convert_to_numpy=True,
                                     normalize_embeddings=True)
        return embeddings

    def get_text_for_post_id(self, post_id):
        try:
            return self.posts.loc[int(post_id), 'text_ocr']
        except KeyError:
            print(f"Warning: post_id {post_id} not found in posts data")
            return f"passage: Post {post_id} not found"

    def predict(self):
        predictions = {}
        for json_path in self.json_paths:
            print(f"Processing {json_path}")
            data = self.load_json_data(json_path)
            post_ids = list(data.keys())
            
            # Get texts and generate embeddings for posts
            texts = [self.get_text_for_post_id(post_id) for post_id in post_ids]
            posts_embeddings = self.generate_embeddings(texts)
            
            # Compute similarities between posts and fact checks
            similarities = cosine_similarity(posts_embeddings, self.fact_checks_embeddings)
            
            # Get top 10 fact check indices for each post
            top_10_indices = np.argsort(-similarities, axis=1)[:, :10]
            
            # Create predictions dictionary
            for i, post_id in enumerate(post_ids):
                top_fact_check_ids = [int(self.fact_checks.index[j]) for j in top_10_indices[i]]
                predictions[post_id] = top_fact_check_ids
            
            # Modificar la ruta de guardado si se especificó predictions_dir
            if self.predictions_dir:
                # Obtener solo el nombre del archivo del json_path
                json_filename = os.path.basename(json_path)
                # Crear la nueva ruta en predictions_dir
                output_path = os.path.join(self.predictions_dir, json_filename)
                # Asegurar que el directorio existe
                os.makedirs(self.predictions_dir, exist_ok=True)
            else:
                output_path = json_path

            # Guardar predicciones en la nueva ruta
            with open(output_path, 'w') as file:
                json.dump(predictions, file, indent=4)
            
            print(f"Predictions saved to {output_path}")
            
        return predictions

# Example usage:
# data_paths = {
#     'fact_checks': 'data/fact_checks.csv',
#     'posts': 'data/posts.csv'
# }
# json_paths = ['data/original/monolingual_predictions.json', 'data/original/crosslingual_predictions.json']
# predictor = Predictor('intfloat/multilingual-e5-small', json_paths, data_paths)
# predictions = predictor.predict() 