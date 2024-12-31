import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity

class Evaluation:
    def __init__(self, model_path, val_data_paths):
        print(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.val_data_paths = val_data_paths

    def load_data(self):
        print("Loading data...")
        self.fact_checks_val = pd.read_csv(self.val_data_paths['fact_checks_val'])
        self.posts_val = pd.read_csv(self.val_data_paths['posts_val'])
        self.pairs_val = pd.read_csv(self.val_data_paths['pairs_val'])
        print("Data loaded successfully")

    def prepare_data(self):
        print("Preparing data...")
        self.posts_val = self.posts_val.set_index('post_id')
        self.fact_checks_val = self.fact_checks_val.set_index('fact_check_id')
        
        # Format texts as required by the model
        self.posts_val['text_formatted'] = 'passage: ' + self.posts_val['text_ocr'].fillna('')
        self.fact_checks_val['text_formatted'] = 'query: ' + self.fact_checks_val['claim_title'].fillna('')
        print("Data prepared successfully")

    def generate_embeddings(self, texts):
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, 
                                     batch_size=16, 
                                     show_progress_bar=True,
                                     convert_to_numpy=True,
                                     normalize_embeddings=True)
        return embeddings

    def evaluate(self):
        self.load_data()
        self.prepare_data()
        
        print("Processing texts...")
        posts_texts = self.posts_val['text_formatted'].tolist()
        fact_checks_texts = self.fact_checks_val['text_formatted'].tolist()
        
        posts_embeddings = self.generate_embeddings(posts_texts)
        fact_checks_embeddings = self.generate_embeddings(fact_checks_texts)
        
        print("Computing similarities...")
        similarities = cosine_similarity(posts_embeddings, fact_checks_embeddings)
        top_10_indices = np.argsort(-similarities, axis=1)[:, :10]
        
        predictions = []
        for i, indices in enumerate(top_10_indices):
            predictions.append({
                "post_id": self.posts_val.index[i],
                "top_k": [{"fact_check_id": self.fact_checks_val.index[j]} for j in indices]
            })
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.merge(self.posts_val[['language']], left_on='post_id', right_index=True)
        
        return self.success_at_10(predictions_df)

    def success_at_10(self, predictions_df):
        print("Computing success@10 metrics...")
        correct_answers = self.pairs_val.groupby('post_id')['fact_check_id'].apply(set).to_dict()
        success_scores = []
        
        for _, row in predictions_df.iterrows():
            post_id = row['post_id']
            predicted_ids = set(int(item['fact_check_id']) for item in row['top_k'])
            success = 1 if post_id in correct_answers and predicted_ids & correct_answers[post_id] else 0
            success_scores.append(success)
        
        predictions_df['success_at_10'] = success_scores
        general_score = predictions_df['success_at_10'].mean()
        by_language = predictions_df.groupby('language')['success_at_10'].agg(['mean', 'count']).round(3)
        
        print(f"\nGeneral Success@10: {general_score:.3f}")
        print("\nSuccess@10 by language:")
        print(by_language)
        
        return general_score, by_language

class EmbeddingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_tokens=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, truncation=True, max_length=self.max_tokens, padding="max_length", return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze()
        }

# Example usage:
# val_data_paths = {
#     'fact_checks_val': 'data/transformed/fact_checks_val.csv',
#     'posts_val': 'data/transformed/posts_train_val.csv',
#     'pairs_val': 'data/transformed/pairs_val.csv'
# }
# evaluator = Evaluation('intfloat/multilingual-e5-small', val_data_paths)
# general_score, score_by_language = evaluator.evaluate()
# print(f"General Success@10: {general_score}")
# print(f"Success@10 by language: {score_by_language}") 