configurations = [
    # Baseline configuration
    {
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
        'gradient_checkpointing': True,
        'max_validation_samples': 500
    },
    # Architecture variations
    {
        'pooling_mode': 'weightedmean',
        'add_pooling_dropout': True,
        'pooling_dropout': 0.1,
    },
    {
        'dense_dropout': 0.2,
        'dense_activation': 'selu',
    },
    # Loss function variations
    {
        'loss_type': 'triplet',
        'triplet_margin': 5.0,
        'negative_mining': 'hard',
    },
    # Training strategy variations
    {
        'scheduler': 'warmupcosine',
        'warmup_ratio': 0.2,
        'weight_decay': 0.01,
        'early_stopping_patience': 5
    },
    # Heavy regularization
    {
        'dense_dropout': 0.3,
        'label_smoothing': 0.1,
        'weight_decay': 0.1,
        'early_stopping_patience': 5,
    }
]

for i, config in enumerate(configurations):
    print(f"\nTraining configuration {i+1}/{len(configurations)}")
    train_model(
        base_model_name='intfloat/multilingual-e5-large-instruct',
        posts_train_path='data/transformed/posts_train_train.csv',
        fact_checks_train_path='data/transformed/fact_checks_train.csv',
        pairs_train_path='data/transformed/pairs_train.csv',
        output_path=f'models/fine-tuned-models/config_{i+1}',
        task='monolingual',
        columns=custom_columns,
        train_by_language=True,
        use_wandb=True,
        project_name='semeval-2025',
        **config
    )
