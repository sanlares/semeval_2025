from evaluation import Evaluation
import os

val_data_paths = {
    'fact_checks_val': 'data/transformed/fact_checks_val.csv',
    'posts_val': 'data/transformed/posts_train_val.csv',
    'pairs_val': 'data/transformed/pairs_val.csv'
}

model_path = 'models/sentence-transformer-fact-check'

print(f"Current working directory: {os.getcwd()}")
print(f"Checking if model exists: {os.path.exists(model_path)}")
print(f"Checking if data files exist:")
for key, path in val_data_paths.items():
    print(f"- {key}: {os.path.exists(path)}")

print("\nInitializing evaluation...")
evaluator = Evaluation(model_path, val_data_paths)
print("\nStarting evaluation...")
general_score, score_by_language = evaluator.evaluate() 