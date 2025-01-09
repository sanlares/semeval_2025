import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity

class Evaluation:
    def __init__(self, model_path, val_data_paths, task_type='monolingual', 
                 post_text_column='text_ocr', fact_check_text_column='claim_title',
                 post_prefix=None, fact_check_prefix=None,
                 post_additional_columns=None, fact_check_additional_columns=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the model to use for evaluation
            val_data_paths: Dictionary with paths to validation data files
            task_type: Type of task to evaluate ('monolingual' or 'crosslingual')
            post_text_column: Column name in posts dataframe to use for text content
            fact_check_text_column: Column name in fact checks dataframe to use for text content
            post_prefix: Optional prefix to add before post texts (e.g., 'passage: ')
            fact_check_prefix: Optional prefix to add before fact check texts (e.g., 'query: ')
            post_additional_columns: Optional list of column names from posts to include in text
            fact_check_additional_columns: Optional list of column names from fact checks to include in text
        """
        print(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.val_data_paths = val_data_paths
        self.task_type = task_type
        self.post_text_column = post_text_column
        self.fact_check_text_column = fact_check_text_column
        self.post_prefix = post_prefix
        self.fact_check_prefix = fact_check_prefix
        self.post_additional_columns = post_additional_columns or []
        self.fact_check_additional_columns = fact_check_additional_columns or []

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
        
        # Format posts texts with optional prefix and additional columns
        base_post_text = self.posts_val[self.post_text_column].fillna('')
        additional_post_text = ''
        if self.post_additional_columns:
            for col in self.post_additional_columns:
                if col in self.posts_val.columns:
                    additional_post_text += f" {col} {self.posts_val[col].fillna('')}"
        
        if self.post_prefix is not None:
            self.posts_val['text_formatted'] = self.post_prefix + additional_post_text + ' ' + base_post_text
        else:
            self.posts_val['text_formatted'] = additional_post_text + ' ' + base_post_text
        
        # Format fact checks texts with optional prefix and additional columns
        base_fact_check_text = self.fact_checks_val[self.fact_check_text_column].fillna('')
        additional_fact_check_text = ''
        if self.fact_check_additional_columns:
            for col in self.fact_check_additional_columns:
                if col in self.fact_checks_val.columns:
                    additional_fact_check_text += f" {col} {self.fact_checks_val[col].fillna('')}"
        
        if self.fact_check_prefix is not None:
            self.fact_checks_val['text_formatted'] = self.fact_check_prefix + additional_fact_check_text + ' ' + base_fact_check_text
        else:
            self.fact_checks_val['text_formatted'] = additional_fact_check_text + ' ' + base_fact_check_text
            
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
        
        # Only add language info for monolingual task
        if self.task_type == 'monolingual':
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
        
        print(f"\nGeneral Success@10: {general_score:.3f}")
        
        # Only compute language-specific metrics for monolingual task
        if self.task_type == 'monolingual':
            by_language = predictions_df.groupby('language')['success_at_10'].agg(['mean', 'count']).round(3)
            print("\nSuccess@10 by language:")
            print(by_language)
            return general_score, by_language
        else:
            return general_score, None

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
# evaluator = Evaluation('intfloat/multilingual-e5-small', 
#                       val_data_paths, 
#                       task_type='monolingual',
#                       post_text_column='text_ocr',
#                       fact_check_text_column='claim_title',
#                       post_prefix='passage: ',
#                       fact_check_prefix='query: ',
#                       post_additional_columns=['language', 'domain'],  # Optional
#                       fact_check_additional_columns=['language'])  # Optional
# general_score, score_by_language = evaluator.evaluate()
# print(f"General Success@10: {general_score}")
# if score_by_language is not None:
#     print(f"Success@10 by language: {score_by_language}") 