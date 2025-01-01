import os
import logging
import torch
import pandas as pd
from dataclasses import dataclass, field
from typing import List
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
import random
import time
from utils.general import custom_similarity

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)

@dataclass
class TrainingConfig:
    """Configuration for training the model."""
    model_name: str
    output_path: str
    max_seq_length: int = 512
    pooling_mode: str = "mean"  # Options: mean, max, cls
    embedding_dim: int = 384
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_epochs: int = 6
    temperature: float = 0.05
    learning_rate: float = 2e-5
    use_amp: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        os.makedirs(self.output_path, exist_ok=True)

class ContrastiveDataset(Dataset):
    def __init__(self, pairs_data, posts, fact_checks):
        self.pairs_data = pairs_data
        self.posts = posts
        self.fact_checks = fact_checks
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

class EnhancedTensorBoardCallback:
    def __init__(self, tb_writer, model, batch_size):
        self.tb_writer = tb_writer
        self.model = model
        self.batch_size = batch_size
        self.global_step = 0
        self.prev_time = time.time()
    
    def __call__(self, score, epoch, steps):
        loss = -score
        self.tb_writer.add_scalar('Loss/train', loss, self.global_step)
        
        if hasattr(self.model, 'optimizer'):
            self.tb_writer.add_scalar('Learning Rate', 
                                    self.model.optimizer.param_groups[0]['lr'], 
                                    self.global_step)
        
        current_time = time.time()
        step_time = current_time - self.prev_time
        self.tb_writer.add_scalar('Performance/step_time', step_time, self.global_step)
        self.tb_writer.add_scalar('Performance/samples_per_second', 
                                self.batch_size / step_time if step_time > 0 else 0, 
                                self.global_step)
        
        self.prev_time = current_time
        self.global_step += 1

class ContrastiveRetrieval:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(self.config.device)
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        self.train_loss = losses.MultipleNegativesRankingLoss(
            model=self.model,
            scale=1.0 / self.config.temperature,
            similarity_fct=lambda a, b: custom_similarity(a, b, self.device)
        )
    
    def _create_model(self) -> SentenceTransformer:
        word_embedding_model = models.Transformer(
            self.config.model_name,
            max_seq_length=self.config.max_seq_length
        )
        
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode=self.config.pooling_mode
        )
        
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=self.config.embedding_dim,
            activation_function=torch.nn.GELU()
        )
        
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    
    def train(self, train_dataset, val_dataset, pairs_val, posts_val, fact_checks_val):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_examples = []
        
        for _, row in pairs_val.iterrows():
            val_examples.append(InputExample(
                texts=[
                    posts_val.loc[row['post_id'], 'ocr_text'],
                    fact_checks_val.loc[row['fact_check_id'], 'claim_title']
                ],
                label=1.0
            ))
        
        for _, row in pairs_val.iterrows():
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
        
        tb_writer = SummaryWriter(log_dir=os.path.join(self.config.output_path, 'logs'))
        tensorboard_callback = EnhancedTensorBoardCallback(
            tb_writer, 
            self.model, 
            self.config.batch_size
        )

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
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
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
        self.model.save(path)
        
    @classmethod
    def load(cls, path: str, config: TrainingConfig) -> 'ContrastiveRetrieval':
        instance = cls(config)
        instance.model = SentenceTransformer(path)
        return instance

def train_model(
    base_model_name: str,
    posts_train_path: str,
    fact_checks_train_path: str,
    pairs_train_path: str,
    output_path: str,
    **kwargs
) -> ContrastiveRetrieval:
    """
    Train a contrastive retrieval model.
    
    Args:
        base_model_name: Name of the base model from HuggingFace
        posts_train_path: Path to the posts training CSV
        fact_checks_train_path: Path to the fact checks training CSV
        pairs_train_path: Path to the pairs training CSV
        output_path: Path where to save the fine-tuned model
        **kwargs: Additional arguments to override default TrainingConfig values
    
    Returns:
        Trained ContrastiveRetrieval model
    """
    # Load training data
    posts_train = pd.read_csv(posts_train_path, index_col='post_id')
    fact_checks_train = pd.read_csv(fact_checks_train_path, index_col='fact_check_id')
    pairs_train = pd.read_csv(pairs_train_path)
    
    # Create config
    config_args = {'model_name': base_model_name, 'output_path': output_path}
    config_args.update(kwargs)
    config = TrainingConfig(**config_args)
    
    # Initialize model
    model = ContrastiveRetrieval(config)
    
    # Create dataset
    train_dataset = ContrastiveDataset(
        pairs_data=[(row['post_id'], row['fact_check_id'], 1) for _, row in pairs_train.iterrows()],
        posts=posts_train,
        fact_checks=fact_checks_train
    )
    
    # Train model
    model.train(train_dataset, None, pairs_train, posts_train, fact_checks_train)
    
    return model 