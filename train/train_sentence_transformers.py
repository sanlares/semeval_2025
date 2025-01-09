import os
import logging
import torch
import pandas as pd
import wandb
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Any
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
import random
import time
import numpy as np
from utils.general import custom_similarity
from datetime import datetime

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
    batch_size: int = 4
    eval_batch_size: int = 16  # Separate batch size for evaluation
    gradient_accumulation_steps: int = 2
    num_epochs: Optional[int] = None  # Made optional to be set by train_model
    temperature: float = 0.05
    learning_rate: float = 2e-5
    use_amp: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    model_source: str = "huggingface"  # can be "huggingface" or "local"
    use_wandb: bool = True  # Whether to use Weights & Biases logging
    project_name: str = "semeval-2025"  # Project name for W&B
    max_validation_samples: int = 1000  # Maximum number of validation samples to use
    wandb_run: Optional[Any] = None  # Store wandb run instance
    
    def __post_init__(self):
        os.makedirs(self.output_path, exist_ok=True)
        if self.num_epochs is None:
            self.num_epochs = 3  # Default value if not specified

class ContrastiveDataset(Dataset):
    def __init__(self, pairs_data: List[tuple], posts: pd.DataFrame, fact_checks: pd.DataFrame, columns: Dict[str, str]):
        """
        Initialize the dataset.
        
        Args:
            pairs_data: List of tuples (post_id, fact_check_id, label)
            posts: DataFrame containing posts
            fact_checks: DataFrame containing fact checks
            columns: Dictionary with column names to use
        """
        self.pairs_data = pairs_data
        self.posts = posts
        self.fact_checks = fact_checks
        self.columns = columns
        logging.info(f"Dataset initialized with {len(pairs_data)} pairs")

    def __len__(self):
        return len(self.pairs_data)

    def __getitem__(self, idx):
        try:
            post_id, fact_check_id, label = self.pairs_data[idx]
            post_text = str(self.posts.loc[post_id][self.columns['posts_text']])
            fact_check_text = str(self.fact_checks.loc[fact_check_id][self.columns['fact_checks_text']])
            return InputExample(texts=[post_text, fact_check_text], label=label)
        except Exception as e:
            logging.error(f"Error in __getitem__ at idx {idx}: {str(e)}")
            raise

class EnhancedTensorBoardCallback:
    def __init__(self, tb_writer, model, batch_size, val_evaluator=None, use_wandb=False):
        self.tb_writer = tb_writer
        self.model = model
        self.batch_size = batch_size
        self.val_evaluator = val_evaluator
        self.use_wandb = use_wandb
        self.global_step = 0
        self.prev_time = time.time()
        self.best_val_score = -float('inf')
        self.train_dataloader_len = None  # Will be set during first call
        
    def log_metrics(self, metrics_dict, step):
        """Log metrics to both TensorBoard and W&B if enabled."""
        for name, value in metrics_dict.items():
            self.tb_writer.add_scalar(name, value, step)
        if self.use_wandb:
            wandb.log(metrics_dict, step=step)
    
    def log_parameter_stats(self):
        """Log parameter statistics."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats = {
                    f"parameters/{name}/mean": param.data.mean().item(),
                    f"parameters/{name}/std": param.data.std().item(),
                    f"parameters/{name}/norm": param.data.norm().item(),
                }
                if param.grad is not None:
                    stats.update({
                        f"gradients/{name}/mean": param.grad.data.mean().item(),
                        f"gradients/{name}/std": param.grad.std().item(),
                        f"gradients/{name}/norm": param.grad.norm().item(),
                    })
                self.log_metrics(stats, self.global_step)
    
    def log_attention_stats(self):
        """Log attention statistics if available."""
        if hasattr(self.model, "modules"):
            for module in self.model.modules():
                if isinstance(module, torch.nn.MultiheadAttention):
                    # Log attention weights statistics
                    if hasattr(module, "attn_weights"):
                        attn_stats = {
                            "attention/mean": module.attn_weights.mean().item(),
                            "attention/std": module.attn_weights.std().item(),
                        }
                        self.log_metrics(attn_stats, self.global_step)
    
    def __call__(self, score, epoch, steps):
        # Log training metrics
        train_loss = -score  # Convert score to loss
        metrics = {
            'train/loss': train_loss,
            'train/epoch': epoch,
            'train/global_step': self.global_step,
        }
        
        # Log validation metrics if evaluator is available
        if self.val_evaluator is not None:
            val_scores = self.val_evaluator(self.model, output_path=None)
            if isinstance(val_scores, dict):
                metrics.update({
                    f'val/{k}': v for k, v in val_scores.items()
                })
                # Use cosine similarity score as the main validation metric
                if 'cosine_pearson' in val_scores:
                    metrics['val/score'] = val_scores['cosine_pearson']
            else:
                val_loss = -val_scores
                metrics.update({
                    'val/loss': val_loss,
                    'val/score': val_scores,
                })
            
            # Track best validation score
            if metrics.get('val/score', -float('inf')) > self.best_val_score:
                self.best_val_score = metrics['val/score']
                metrics['val/best_score'] = self.best_val_score
        
        # Log learning rate
        if hasattr(self.model, 'optimizer'):
            current_lr = self.model.optimizer.param_groups[0]['lr']
            metrics['train/learning_rate'] = current_lr
        
        # Log performance metrics
        current_time = time.time()
        step_time = current_time - self.prev_time
        samples_per_second = self.batch_size / step_time if step_time > 0 else 0
        
        # Store train_dataloader_len on first call
        if self.train_dataloader_len is None and hasattr(self.model, '_train_objectives'):
            self.train_dataloader_len = len(self.model._train_objectives[0][0])
            
        metrics.update({
            'performance/step_time': step_time,
            'performance/samples_per_second': samples_per_second,
        })
        
        # Only add epoch_progress if we have the dataloader length
        if self.train_dataloader_len is not None:
            metrics['performance/epoch_progress'] = steps / self.train_dataloader_len
        
        # Log all metrics
        self.log_metrics(metrics, self.global_step)
        
        # Log parameter and gradient statistics periodically
        if self.global_step % 100 == 0:  # Reduce frequency of parameter logging
            self.log_parameter_stats()
            self.log_attention_stats()
        
        self.prev_time = current_time
        self.global_step += 1

class ExperimentTracker:
    def __init__(self, project_name="semeval-2025"):
        """Initialize experiment tracking"""
        self.project_name = project_name
        self.run = None

    def start_run(self, config: TrainingConfig, run_name=None):
        """Start a new experiment run"""
        # Convert config to dict and add additional metadata
        config_dict = asdict(config)
        config_dict.update({
            "git_commit": self._get_git_commit(),
            "dataset_version": self._get_dataset_version(),
            "timestamp": datetime.now().isoformat()
        })
        
        # Initialize W&B run
        self.run = wandb.init(
            project=self.project_name,
            name=run_name,
            config=config_dict,
            reinit=True
        )
        
        # Create model artifacts directory
        os.makedirs("model_artifacts", exist_ok=True)
        
        return self.run

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics to W&B"""
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_model(self, model_path: str, name: str):
        """Log model as artifact"""
        if self.run is not None:
            artifact = wandb.Artifact(name=name, type='model')
            artifact.add_dir(model_path)
            self.run.log_artifact(artifact)

    def finish(self):
        """End the experiment run"""
        if self.run is not None:
            self.run.finish()

    @staticmethod
    def _get_git_commit():
        """Get current git commit hash"""
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha
        except:
            return None

    @staticmethod
    def _get_dataset_version():
        """Get dataset version/hash"""
        try:
            import hashlib
            data_path = "data/transformed"
            files_hash = hashlib.md5()
            
            for filename in sorted(os.listdir(data_path)):
                if filename.endswith('.csv'):
                    with open(os.path.join(data_path, filename), 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b''):
                            files_hash.update(chunk)
            
            return files_hash.hexdigest()
        except:
            return None

class ContrastiveRetrieval:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(self.config.device)
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        self.train_loss = losses.MultipleNegativesRankingLoss(
            model=self.model,
            scale=1.0 / self.config.temperature
        )
        
        # Initialize experiment tracker only if no wandb run is provided
        self.tracker = ExperimentTracker()
        if not hasattr(config, 'wandb_run') or config.wandb_run is None:
            self.run = self.tracker.start_run(
                config=config,
                run_name=f"{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        else:
            self.run = config.wandb_run

    def _create_model(self) -> SentenceTransformer:
        if hasattr(self.config, 'model_source') and self.config.model_source == "local":
            # Load local SentenceTransformers model
            return SentenceTransformer(self.config.model_name)
        else:
            # Create new model from HuggingFace
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
    
    def train(self, train_dataset, evaluator, pairs_train, posts_train, fact_checks_train):
        """
        Train the model.
        
        Args:
            train_dataset: ContrastiveDataset for training
            evaluator: EmbeddingSimilarityEvaluator for validation
            pairs_train: DataFrame containing training pairs
            posts_train: DataFrame containing posts
            fact_checks_train: DataFrame containing fact checks
        """
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        tb_writer = SummaryWriter(log_dir=os.path.join(self.config.output_path, 'logs'))
        tensorboard_callback = EnhancedTensorBoardCallback(
            tb_writer, 
            self.model, 
            self.config.batch_size,
            evaluator,
            self.config.use_wandb
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
        
        # Don't finish wandb run here as it's managed by train_model function
    
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
    task: str = "monolingual",  # can be "monolingual" or "crosslingual"
    model_source: str = "huggingface",  # can be "huggingface" or "local"
    train_by_language: bool = False,  # whether to train separate models for each language
    target_language: Optional[str] = None,  # specific language to train for
    use_wandb: bool = True,
    project_name: str = "semeval-2025",
    columns: dict = None,
    num_epochs: Optional[int] = None,  # Added num_epochs parameter
    **kwargs
) -> Union[ContrastiveRetrieval, Dict[str, ContrastiveRetrieval]]:
    """
    Train a contrastive retrieval model.
    
    Args:
        base_model_name: Name/path of the base model
        posts_train_path: Path to the posts training CSV
        fact_checks_train_path: Path to the fact checks training CSV
        pairs_train_path: Path to the pairs training CSV
        output_path: Path where to save the fine-tuned model
        task: Type of task, either "monolingual" or "crosslingual" (default: "monolingual")
        model_source: Source of the base model, either "huggingface" or "local" (default: "huggingface")
        train_by_language: Whether to train separate models for each language (default: False)
        target_language: Specific language to train for (e.g., "tha", "eng"). Cannot be used with train_by_language.
        use_wandb: Whether to use Weights & Biases logging (default: True)
        project_name: Project name for W&B (default: "semeval-2025")
        columns: Dictionary containing column names to use
        num_epochs: Number of training epochs (default: None, will use TrainingConfig default)
        **kwargs: Additional arguments to override default TrainingConfig values
    
    Returns:
        If train_by_language is False and target_language is None: Single trained model for all languages
        If train_by_language is True: Dictionary mapping language codes to trained models
        If target_language is provided: Single trained model for the target language
    """
    # Initialize wandb only once at the start
    wandb_run = None
    if use_wandb:
        run_name = f"{base_model_name.split('/')[-1]}"
        if target_language:
            run_name += f"_{target_language}"
        elif train_by_language:
            run_name += "_multilingual"
        wandb_run = wandb.init(project=project_name, name=run_name, config=kwargs, reinit=True)
        
    if task not in ["monolingual", "crosslingual"]:
        raise ValueError("Task must be either 'monolingual' or 'crosslingual'")
        
    if model_source not in ["huggingface", "local"]:
        raise ValueError("model_source must be either 'huggingface' or 'local'")
        
    if train_by_language and target_language is not None:
        raise ValueError("Cannot use both train_by_language and target_language. Choose one approach.")

    try:
        # Default column names
        default_columns = {
            'fact_checks_id': 'fact_check_id',
            'fact_checks_text': 'claim_title',
            'posts_id': 'post_id',
            'posts_text': 'text_ocr',
            'pairs_post_id': 'post_id',
            'pairs_fact_check_id': 'fact_check_id',
            'posts_language': 'language',  # Always include language column
            'fact_checks_language': 'language'  # Language column for fact checks
        }
        
        # Use provided columns or defaults
        columns = {**default_columns, **(columns or {})}
        
        # Load training data
        posts_train = pd.read_csv(posts_train_path)
        fact_checks_train = pd.read_csv(fact_checks_train_path)
        pairs_train = pd.read_csv(pairs_train_path)
        
        # Set default kwargs for GPU memory efficiency
        default_kwargs = {
            'batch_size': 4,  # Smaller batch size
            'eval_batch_size': 16,  # Separate eval batch size
            'gradient_accumulation_steps': 4,  # Increase gradient accumulation
            'max_validation_samples': 1000,  # Limit validation samples
        }
        
        # Add num_epochs to kwargs if provided
        if num_epochs is not None:
            default_kwargs['num_epochs'] = num_epochs
            
        # Only update if not provided in kwargs
        for k, v in default_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
        
        # If target_language is provided, only train for that language
        if target_language is not None:
            logging.info(f"Training model for language: {target_language}")
            
            # Filter data for target language
            lang_posts = posts_train[posts_train[columns['posts_language']] == target_language].copy()
            lang_fact_checks = fact_checks_train[fact_checks_train[columns['fact_checks_language']] == target_language].copy()
            
            if len(lang_posts) == 0 or len(lang_fact_checks) == 0:
                raise ValueError(f"No data found for language {target_language}")
            
            # Filter pairs to only include posts and fact checks in target language
            lang_pairs = pairs_train[
                pairs_train[columns['pairs_post_id']].isin(lang_posts[columns['posts_id']]) &
                pairs_train[columns['pairs_fact_check_id']].isin(lang_fact_checks[columns['fact_checks_id']])
            ].copy()
            
            if len(lang_pairs) == 0:
                raise ValueError(f"No training pairs found for language {target_language}")
                
            # Optimize training parameters for single language
            dataset_size = len(lang_pairs)
            if 'batch_size' not in kwargs:
                # Increase batch size for smaller datasets
                kwargs['batch_size'] = min(32, max(8, dataset_size // 100))
            if 'eval_batch_size' not in kwargs:
                kwargs['eval_batch_size'] = min(64, max(16, dataset_size // 50))
            if 'gradient_accumulation_steps' not in kwargs:
                # Reduce gradient accumulation for smaller datasets
                kwargs['gradient_accumulation_steps'] = max(1, 4 * 1000 // dataset_size)
            if 'max_validation_samples' not in kwargs:
                # Use smaller validation set for smaller datasets
                kwargs['max_validation_samples'] = min(1000, max(100, dataset_size // 10))
                
            logging.info(f"Optimized training parameters for {target_language}: {kwargs}")
            
            # Set indices
            lang_posts = lang_posts.set_index(columns['posts_id'])
            lang_fact_checks = lang_fact_checks.set_index(columns['fact_checks_id'])
            
            # Create language-specific output path
            lang_output_path = os.path.join(output_path, target_language)
            os.makedirs(lang_output_path, exist_ok=True)
            
            # Create config
            config_args = {
                'model_name': base_model_name, 
                'output_path': lang_output_path,
                'model_source': model_source,
                'use_wandb': use_wandb,
                'project_name': project_name,
                'wandb_run': wandb_run  # Add wandb_run to config
            }
            config_args.update(kwargs)
            config = TrainingConfig(**config_args)
            
            # Initialize model
            model = ContrastiveRetrieval(config)
            
            # Create datasets
            train_dataset = ContrastiveDataset(
                pairs_data=[
                    (row[columns['pairs_post_id']], row[columns['pairs_fact_check_id']], 1) 
                    for _, row in lang_pairs.iterrows()
                ],
                posts=lang_posts,
                fact_checks=lang_fact_checks,
                columns=columns
            )
            
            # Create validation examples
            val_pairs = lang_pairs.sample(n=min(len(lang_pairs), config.max_validation_samples))
            val_examples = []
            for _, row in val_pairs.iterrows():
                val_examples.append(InputExample(
                    texts=[
                        lang_posts.loc[row[columns['pairs_post_id']], columns['posts_text']],
                        lang_fact_checks.loc[row[columns['pairs_fact_check_id']], columns['fact_checks_text']]
                    ],
                    label=1.0
                ))
                # Add negative example
                wrong_fact_check_id = random.choice(list(set(lang_fact_checks.index) - {row[columns['pairs_fact_check_id']]}))
                val_examples.append(InputExample(
                    texts=[
                        lang_posts.loc[row[columns['pairs_post_id']], columns['posts_text']],
                        lang_fact_checks.loc[wrong_fact_check_id, columns['fact_checks_text']]
                    ],
                    label=0.0
                ))
            
            # Create evaluator
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                val_examples,
                name=f'fact-check-validation-{target_language}',
                batch_size=config.eval_batch_size
            )
            
            # Train model
            model.train(train_dataset, evaluator, lang_pairs, lang_posts, lang_fact_checks)
            
            if use_wandb:
                wandb.finish()
            
            return model
        
        elif train_by_language:
            # Get unique languages and reorder to start with 'tha'
            languages = posts_train[columns['posts_language']].unique().tolist()
            if 'tha' in languages:
                languages.remove('tha')
                languages = ['tha'] + languages
            logging.info(f"Training separate models for languages in order: {languages}")
            
            models = {}
            for lang in languages:
                logging.info(f"\nTraining model for language: {lang}")
                torch.cuda.empty_cache()  # Clear GPU memory between languages
                
                # Filter data for current language
                lang_posts = posts_train[posts_train[columns['posts_language']] == lang].copy()
                lang_fact_checks = fact_checks_train[fact_checks_train[columns['fact_checks_language']] == lang].copy()
                
                # Filter pairs to only include posts and fact checks in current language
                lang_pairs = pairs_train[
                    pairs_train[columns['pairs_post_id']].isin(lang_posts[columns['posts_id']]) &
                    pairs_train[columns['pairs_fact_check_id']].isin(lang_fact_checks[columns['fact_checks_id']])
                ].copy()
                
                if len(lang_pairs) == 0:
                    logging.warning(f"No training pairs found for language {lang}, skipping...")
                    continue
                    
                # Set indices
                lang_posts = lang_posts.set_index(columns['posts_id'])
                lang_fact_checks = lang_fact_checks.set_index(columns['fact_checks_id'])
                
                # Create language-specific output path
                lang_output_path = os.path.join(output_path, f"{lang}")
                os.makedirs(lang_output_path, exist_ok=True)
                
                # Create config
                config_args = {
                    'model_name': base_model_name, 
                    'output_path': lang_output_path,
                    'model_source': model_source,
                    'use_wandb': use_wandb,
                    'project_name': project_name,
                    'wandb_run': wandb_run  # Add wandb_run to config
                }
                config_args.update(kwargs)
                config = TrainingConfig(**config_args)
                
                # Initialize model
                model = ContrastiveRetrieval(config)
                
                # Create datasets
                train_dataset = ContrastiveDataset(
                    pairs_data=[
                        (row[columns['pairs_post_id']], row[columns['pairs_fact_check_id']], 1) 
                        for _, row in lang_pairs.iterrows()
                    ],
                    posts=lang_posts,
                    fact_checks=lang_fact_checks,
                    columns=columns
                )
                
                # Create validation examples for current language
                val_pairs = lang_pairs.sample(n=min(len(lang_pairs), config.max_validation_samples))
                val_examples = []
                for _, row in val_pairs.iterrows():
                    val_examples.append(InputExample(
                        texts=[
                            lang_posts.loc[row[columns['pairs_post_id']], columns['posts_text']],
                            lang_fact_checks.loc[row[columns['pairs_fact_check_id']], columns['fact_checks_text']]
                        ],
                        label=1.0
                    ))
                    # Add negative example
                    wrong_fact_check_id = random.choice(list(set(lang_fact_checks.index) - {row[columns['pairs_fact_check_id']]}))
                    val_examples.append(InputExample(
                        texts=[
                            lang_posts.loc[row[columns['pairs_post_id']], columns['posts_text']],
                            lang_fact_checks.loc[wrong_fact_check_id, columns['fact_checks_text']]
                        ],
                        label=0.0
                    ))
                
                # Create evaluator
                evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                    val_examples,
                    name=f'fact-check-validation-{lang}',
                    batch_size=config.eval_batch_size
                )
                
                # Train model
                model.train(train_dataset, evaluator, lang_pairs, lang_posts, lang_fact_checks)
                
                # Store model
                models[lang] = model
                
                logging.info(f"Finished training model for language: {lang}")
                
                # Clean up GPU memory
                torch.cuda.empty_cache()
            
            if use_wandb:
                wandb.finish()
            
            return models
            
        else:
            # Set indices using specified column names
            posts_train = posts_train.set_index(columns['posts_id'])
            fact_checks_train = fact_checks_train.set_index(columns['fact_checks_id'])
            
            # Create config with wandb settings
            config_args = {
                'model_name': base_model_name, 
                'output_path': output_path,
                'model_source': model_source,
                'use_wandb': use_wandb,
                'project_name': project_name,
                'wandb_run': wandb_run  # Add wandb_run to config
            }
            config_args.update(kwargs)
            config = TrainingConfig(**config_args)
            
            # Initialize model
            model = ContrastiveRetrieval(config)
            
            # Create datasets
            train_dataset = ContrastiveDataset(
                pairs_data=[
                    (row[columns['pairs_post_id']], row[columns['pairs_fact_check_id']], 1) 
                    for _, row in pairs_train.iterrows()
                ],
                posts=posts_train,
                fact_checks=fact_checks_train,
                columns=columns
            )
            
            # Create validation examples
            val_pairs = pairs_train.sample(n=min(len(pairs_train), config.max_validation_samples))
            val_examples = []
            for _, row in val_pairs.iterrows():
                val_examples.append(InputExample(
                    texts=[
                        posts_train.loc[row[columns['pairs_post_id']], columns['posts_text']],
                        fact_checks_train.loc[row[columns['pairs_fact_check_id']], columns['fact_checks_text']]
                    ],
                    label=1.0
                ))
                # Add negative example
                wrong_fact_check_id = random.choice(list(set(fact_checks_train.index) - {row[columns['pairs_fact_check_id']]}))
                val_examples.append(InputExample(
                    texts=[
                        posts_train.loc[row[columns['pairs_post_id']], columns['posts_text']],
                        fact_checks_train.loc[wrong_fact_check_id, columns['fact_checks_text']]
                    ],
                    label=0.0
                ))
            
            # Create evaluator
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                val_examples,
                name='fact-check-validation',
                batch_size=config.eval_batch_size
            )
            
            # Train model
            model.train(train_dataset, evaluator, pairs_train, posts_train, fact_checks_train)
            
            if use_wandb:
                wandb.finish()
            
            return model

    finally:
        # Ensure wandb run is finished properly
        if wandb_run is not None:
            wandb_run.finish()