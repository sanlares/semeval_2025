import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia advertencias de TensorFlow

from sentence_transformers import SentenceTransformer, models, losses, InputExample, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader, Dataset
import torch
import os
import pandas as pd
from typing import List, Dict, Set
import logging
from dataclasses import dataclass, field
from transformers import AutoTokenizer
import numpy as np
from tqdm.auto import tqdm
import random
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import time

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)

def check_cuda():
    """Verify CUDA availability and print device info"""
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"CUDA available. Found {torch.cuda.device_count()} GPU(s):")
        for i in range(torch.cuda.device_count()):
            logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = "cpu"
        logging.info("CUDA not available. Using CPU.")
    return device

@dataclass
class TrainingConfig:
    """Configuration for training the model."""
    model_name: str = "intfloat/multilingual-e5-small"
    output_path: str = "models/sentence-transformer-fact-check"
    max_seq_length: int = 512
    pooling_mode: str = "mean"  # Options: mean, max, cls
    embedding_dim: int = 128
    batch_size: int = 16  # Aumentado ya que tienes suficiente memoria GPU
    gradient_accumulation_steps: int = 2  # Reducido ya que aumentamos el batch_size
    num_epochs: int = 3
    temperature: float = 0.05
    learning_rate: float = 2e-5
    use_amp: bool = True  # Use automatic mixed precision
    device: str = field(default_factory=check_cuda)  # Añadir device
    
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
        self.posts = posts
        self.fact_checks = fact_checks
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        logging.info(f"Dataset initialized with {len(pairs_data)} pairs")

    def __len__(self):
        return len(self.pairs_data)

    def __getitem__(self, idx):
        try:
            post_id, fact_check_id, label = self.pairs_data[idx]
            post_text = str(self.posts.loc[post_id]["ocr_text"])
            fact_check_text = str(self.fact_checks.loc[fact_check_id]["claim_title"])
            return InputExample(texts=[post_text, fact_check_text], label=label)
        except Exception as e:
            logging.error(f"Error in __getitem__ at idx {idx}: {str(e)}")
            raise

class ImprovedE5Retrieval:
    """Improved E5 model for fact-checking retrieval using SentenceTransformers."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the model with the given configuration."""
        self.config = config
        # Asegurar que estamos usando el dispositivo correcto desde el inicio
        self.device = torch.device(self.config.device)
        self.model = self._create_model()
        # Mover el modelo al dispositivo correcto
        self.model = self.model.to(self.device)
        self.train_loss = losses.MultipleNegativesRankingLoss(
            model=self.model,
            scale=1.0 / self.config.temperature,
            similarity_fct=self.custom_similarity
        )
        self.last_pos_sim = None
        self.last_neg_sim = None
    
    def custom_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Custom similarity function optimized for fact-checking retrieval."""
        a = a.to(self.device)
        b = b.to(self.device)
        sim = util.cos_sim(a, b).float()
        
        # Guardar similitudes para logging
        if a.size(0) == b.size(0):  # Son pares positivos
            self.last_pos_sim = sim
        else:  # Son pares negativos
            self.last_neg_sim = sim
        
        return sim
    
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
    
    def train(self, train_dataset, val_dataset, pairs_val, posts_val, fact_checks_val):
        """Train the model using the provided datasets."""
        logging.info(f"Training on device: {self.device}")
        
        # Verificar memoria GPU disponible
        if torch.cuda.is_available():
            logging.info(f"GPU Memory before training:")
            logging.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            logging.info(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Configurar el DataLoader sin workers en paralelo por ahora
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Cambiado a 0 para evitar problemas de multiprocessing
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Setup evaluator con ejemplos positivos y negativos
        val_examples = []
        
        # Ejemplos positivos (pares correctos)
        for _, row in pairs_val.iterrows():
            val_examples.append(InputExample(
                texts=[
                    posts_val.loc[row['post_id'], 'ocr_text'],
                    fact_checks_val.loc[row['fact_check_id'], 'claim_title']
                ],
                label=1.0
            ))
        
        # Generar algunos ejemplos negativos
        for _, row in pairs_val.iterrows():
            # Seleccionar un fact check aleatorio diferente al correcto
            wrong_fact_check_id = random.choice(list(set(fact_checks_val.index) - {row['fact_check_id']}))
            val_examples.append(InputExample(
                texts=[
                    posts_val.loc[row['post_id'], 'ocr_text'],
                    fact_checks_val.loc[wrong_fact_check_id, 'claim_title']
                ],
                label=0.0
            ))

        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples,
            name='fact-check-validation'
        )
        
        # Setup TensorBoard
        tb_writer = SummaryWriter(log_dir=os.path.join(self.config.output_path, 'logs'))
        tensorboard_callback = EnhancedTensorBoardCallback(
            tb_writer, 
            self.model, 
            self.config.batch_size
        )

        # Training
        warmup_steps = int(len(train_dataloader) * self.config.num_epochs * 0.1)
        
        self.model.fit(
            train_objectives=[(train_dataloader, self.train_loss)],
            evaluator=evaluator,
            epochs=self.config.num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=self.config.output_path,
            show_progress_bar=True,
            use_amp=self.config.use_amp,
            callback=tensorboard_callback
        )
        
        tb_writer.close()
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches with optional caching."""
        # Asegurar que el modelo está en el dispositivo correcto
        self.model.to(self.device)
        
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
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
    
    def split_train_val(self, posts_df, fact_checks_df, pairs_df):
        """Split data into train and validation sets."""
        from sklearn.model_selection import train_test_split
        
        # Create copies to avoid SettingWithCopyWarning
        posts_df = posts_df.copy()
        fact_checks_df = fact_checks_df.copy()
        
        # Create ids columns
        posts_df["post_id"] = posts_df.index
        fact_checks_df["fact_check_id"] = fact_checks_df.index
        
        # Split posts
        posts_train, posts_val = train_test_split(posts_df, test_size=0.2, random_state=42)
        
        # Split pairs
        pairs_train = pairs_df[pairs_df['post_id'].isin(posts_train.index)].copy()
        pairs_val = pairs_df[pairs_df['post_id'].isin(posts_val.index)].copy()
        
        # Split fact checks
        fact_checks_train = fact_checks_df[fact_checks_df.index.isin(pairs_train['fact_check_id'])].copy()
        fact_checks_val = fact_checks_df[fact_checks_df.index.isin(pairs_val['fact_check_id'])].copy()
        
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

def print_gpu_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        logging.info(f"GPU Memory Usage:")
        logging.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logging.info(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def clear_gpu_memory():
    """Clear unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("GPU memory cache cleared")

class EnhancedTensorBoardCallback(TensorBoardCallback):
    """Enhanced TensorBoard callback with additional metrics."""
    
    def __init__(self, tb_writer, model, batch_size):
        super().__init__()
        self.tb_writer = tb_writer
        self.model = model
        self.batch_size = batch_size
        self.global_step = 0
        self.prev_time = time.time()
    
    def __call__(self, score, epoch, steps):
        """Called by SentenceTransformer when training"""
        # Obtener loss actual (score es el negativo de la loss)
        loss = -score
        
        # Métricas básicas de entrenamiento
        self.tb_writer.add_scalar('Loss/train', loss, self.global_step)
        
        # Learning rate
        if hasattr(self.model, 'optimizer'):
            self.tb_writer.add_scalar('Learning Rate', 
                                    self.model.optimizer.param_groups[0]['lr'], 
                                    self.global_step)
        
        # Similitudes (si están disponibles en el modelo)
        if hasattr(self.model, 'last_pos_sim') and hasattr(self.model, 'last_neg_sim'):
            self.tb_writer.add_scalar('Similarity/positive', 
                                    self.model.last_pos_sim.mean().item(), 
                                    self.global_step)
            self.tb_writer.add_scalar('Similarity/negative', 
                                    self.model.last_neg_sim.mean().item(), 
                                    self.global_step)
        
        # Norma del gradiente
        if hasattr(self.model, 'model'):
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            self.tb_writer.add_scalar('Gradients/total_norm', total_norm, self.global_step)
        
        # Métricas de rendimiento
        current_time = time.time()
        step_time = current_time - self.prev_time
        self.tb_writer.add_scalar('Performance/step_time', step_time, self.global_step)
        self.tb_writer.add_scalar('Performance/samples_per_second', 
                                self.batch_size / step_time if step_time > 0 else 0, 
                                self.global_step)
        
        self.prev_time = current_time
        self.global_step += 1

def main(sample_size: int = None):
    from utils.load import LoadDataCSV
    
    # Configurar logging para mostrar información sobre CUDA
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
        # Configurar para usar determinismo en CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
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