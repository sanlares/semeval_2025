from sentence_transformers import SentenceTransformer, models, losses, InputExample, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader, Dataset
import torch
import os
import pandas as pd
from typing import List, Dict, Set
import logging
from dataclasses import dataclass
from transformers import AutoTokenizer
import numpy as np
from tqdm.auto import tqdm
import random
from transformers.integrations import TensorBoardCallback

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)

@dataclass
class TrainingConfig:
    """Configuration for training the model."""
    model_name: str = "intfloat/multilingual-e5-small"
    output_path: str = "models/sentence-transformer-fact-check"
    max_seq_length: int = 512
    pooling_mode: str = "mean"  # Options: mean, max, cls
    embedding_dim: int = 128
    batch_size: int = 4  # Further reduced batch size
    gradient_accumulation_steps: int = 4  # Added gradient accumulation
    num_epochs: int = 3
    temperature: float = 0.05
    learning_rate: float = 2e-5
    use_amp: bool = True  # Use automatic mixed precision
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_path, exist_ok=True)

class FineTuneContrastiveDataset(Dataset):
    def __init__(self, pairs_data, posts, fact_checks, tokenizer, max_tokens=512):
        """
        pairs_data: List of tuples (post_id, fact_check_id, label).
        posts: DataFrame with texts associated with post_id.
        fact_checks: DataFrame with texts associated with fact_check_id.
        tokenizer: Hugging Face tokenizer.
        max_tokens: Maximum allowed length of texts.
        """
        self.pairs_data = pairs_data
        self.posts = posts  # Already indexed by post_id
        self.fact_checks = fact_checks  # Already indexed by fact_check_id
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.pairs_data)

    def __getitem__(self, idx):
        post_id, fact_check_id, label = self.pairs_data[idx]
        post_text = str(self.posts.loc[post_id]["ocr_text"])
        fact_check_text = str(self.fact_checks.loc[fact_check_id]["claim_title"])

        return InputExample(texts=[post_text, fact_check_text], label=label)

class ImprovedE5Retrieval:
    """Improved E5 model for fact-checking retrieval using SentenceTransformers."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the model with the given configuration."""
        self.config = config
        self.model = self._create_model()
        self.train_loss = losses.MultipleNegativesRankingLoss(
            model=self.model,
            scale=1.0 / self.config.temperature,
            similarity_fct=self.custom_similarity
        )
    
    def custom_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Custom similarity function optimized for fact-checking retrieval."""
        return util.cos_sim(a, b).float()
    
    def _create_model(self) -> SentenceTransformer:
        """Create and configure the SentenceTransformer model."""
        # Initialize the base model
        word_embedding_model = models.Transformer(
            self.config.model_name,
            max_seq_length=self.config.max_seq_length
        )
        
        # Add pooling layer
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode=self.config.pooling_mode
        )
        
        # Add dense layer for dimensionality reduction
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=self.config.embedding_dim,
            activation_function=torch.nn.GELU()
        )
        
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    
    def train(self, train_dataset: FineTuneContrastiveDataset, val_dataset: FineTuneContrastiveDataset, pairs_val: pd.DataFrame, posts_val: pd.DataFrame, fact_checks_val: pd.DataFrame):
        """Train the model using the provided datasets."""
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_examples = [
            InputExample(texts=[posts_val.loc[row['post_id'], 'ocr_text'], fact_checks_val.loc[row['fact_check_id'], 'claim_title']], label=1.0)
            for _, row in pairs_val.iterrows()
        ]

        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples,
            name='fact-check-validation'
        )
        
        warmup_steps = int(len(train_dataloader) * self.config.num_epochs * 0.1)
        
        self.model.gradient_checkpointing_enable()
        
        # Crear el callback de TensorBoard
        tensorboard_callback = TensorBoardCallback()
        
        self.model.fit(
            train_objectives=[(train_dataloader, self.train_loss)],
            evaluator=evaluator,
            epochs=self.config.num_epochs,
            evaluation_steps=100,
            warmup_steps=warmup_steps,
            output_path=self.config.output_path,
            show_progress_bar=True,
            use_amp=self.config.use_amp,
            callback=tensorboard_callback
        )
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches with optional caching."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    def save(self, path: str):
        """Save the model and configuration."""
        self.model.save(path)
        
    @classmethod
    def load(cls, path: str, config: TrainingConfig) -> 'ImprovedE5Retrieval':
        """Load a saved model."""
        instance = cls(config)
        instance.model = SentenceTransformer(path)
        return instance
    
    def split_train_val(self, posts_df: pd.DataFrame, fact_checks_df: pd.DataFrame, pairs_df: pd.DataFrame):
        from sklearn.model_selection import train_test_split

        # Create ids columns from index
        posts_df["post_id"] = posts_df.index
        fact_checks_df["fact_check_id"] = fact_checks_df.index
        pairs_df["post_id"] = pairs_df["post_id"].astype(int)
        pairs_df["fact_check_id"] = pairs_df["fact_check_id"].astype(int)

        # Split posts
        posts_train, posts_val = train_test_split(posts_df, test_size=0.2, random_state=42)
        
        # Split pairs based on post_id
        pairs_train = pairs_df[pairs_df['post_id'].isin(posts_train.index)]
        pairs_val = pairs_df[pairs_df['post_id'].isin(posts_val.index)]

        # Split fact checks based on pairs
        fact_checks_train = fact_checks_df[fact_checks_df.index.isin(pairs_train['fact_check_id'])]
        fact_checks_val = fact_checks_df[fact_checks_df.index.isin(pairs_val['fact_check_id'])]
        
        # Verify the presence of the new columns
        logging.info(f"Columns in posts_train: {posts_train.columns}")
        logging.info(f"Columns in fact_checks_train: {fact_checks_train.columns}")
        logging.info(f"Columns in posts_val: {posts_val.columns}")
        logging.info(f"Columns in fact_checks_val: {fact_checks_val.columns}")
        
        return posts_train, posts_val, fact_checks_train, fact_checks_val, pairs_train, pairs_val

def combine_ocr_text(row):
    ocr = row['ocr']
    text = row['text']
    if ocr and isinstance(ocr, list) and len(ocr) > 0 and len(ocr[0]) > 0:
        return 'passage: '+ str(ocr[0][0]) + ' ' + (str(text[0]) if text else '')
    elif text:
        return 'passage: '+ str(text[0])
    else:
        return ''


def combine_claim_text(row):
    claim = row['claim']
    title = row['title']
    if claim and isinstance(claim, list) and len(claim) > 0 and len(claim[0]) > 0:
        return 'query: '+ str(claim[0][0]) + ' ' + (str(title[0]) if title else '')
    elif title:
        return 'query: '+ str(title[0])
    else:
        return ''

def main(sample_size: int = None):
    # Load your datasets here
    from utils.load import LoadDataCSV
    import logging
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Load data
    data_loader = LoadDataCSV()
    logging.info("Loading data...")
    df_fact_checks, df_posts, pairs_df = data_loader.load_data()
    logging.info(f"Loaded {len(df_posts)} posts, {len(df_fact_checks)} fact checks, and {len(pairs_df)} pairs")
    
    # Sample data if sample_size is provided
    if sample_size is not None:
        logging.info(f"Sampling {sample_size} posts...")
        df_posts = df_posts.sample(n=min(sample_size, len(df_posts)), random_state=42)
        pairs_df = pairs_df[pairs_df['post_id'].isin(df_posts.index)]
        df_fact_checks = df_fact_checks[df_fact_checks.index.isin(pairs_df['fact_check_id'])]
        logging.info(f"After sampling: {len(df_posts)} posts, {len(df_fact_checks)} fact checks, and {len(pairs_df)} pairs")
    
    # Apply the functions to create new columns before splitting
    df_posts['ocr_text'] = df_posts.apply(combine_ocr_text, axis=1)
    df_fact_checks['claim_title'] = df_fact_checks.apply(combine_claim_text, axis=1)

    # Create train/val splits
    model = ImprovedE5Retrieval(config)
    logging.info("Creating train/val splits...")
    posts_train, posts_val, fact_checks_train, fact_checks_val, pairs_train, pairs_val = model.split_train_val(df_posts, df_fact_checks, pairs_df)
    logging.info(f"Train set: {len(posts_train)} posts, {len(fact_checks_train)} fact checks, {len(pairs_train)} pairs")
    logging.info(f"Val set: {len(posts_val)} posts, {len(fact_checks_val)} fact checks, {len(pairs_val)} pairs")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Apply the functions to create new columns before creating datasets
    posts_train['ocr_text'] = posts_train.apply(combine_ocr_text, axis=1)
    fact_checks_train['claim_title'] = fact_checks_train.apply(combine_claim_text, axis=1)

    posts_val['ocr_text'] = posts_val.apply(combine_ocr_text, axis=1)
    fact_checks_val['claim_title'] = fact_checks_val.apply(combine_claim_text, axis=1)

    train_dataset = FineTuneContrastiveDataset(
        pairs_data=[(row['post_id'], row['fact_check_id'], 1) for _, row in pairs_train.iterrows()],
        posts=posts_train,
        fact_checks=fact_checks_train,
        tokenizer=tokenizer
    )

    val_dataset = FineTuneContrastiveDataset(
        pairs_data=[(row['post_id'], row['fact_check_id'], 1) for _, row in pairs_val.iterrows()],
        posts=posts_val,
        fact_checks=fact_checks_val,
        tokenizer=tokenizer
    )
    
    # Train model
    logging.info("Starting training...")
    model.train(train_dataset, val_dataset, pairs_val, posts_val, fact_checks_val)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, help="Number of posts to sample. If not provided, uses full dataset")
    args = parser.parse_args()
    main(args.sample_size) 