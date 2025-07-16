# SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval

This repository contains code, experiments, and documentation for participation in [SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval](https://semeval.github.io/SemEval2025/tasks), a shared task organized by the [DisAI Center](https://disai.eu/) and the European Union. The competition addresses the challenge of retrieving relevant fact-checked claims for social media posts in multiple languages, supporting research in automated fact-checking and misinformation detection.

## About the Competition

The rapid spread of online disinformation is a global issue. This task focuses on retrieving fact-checked claims that match new claims found in social media posts, across a variety of languages. The competition includes two main tracks:
- **Monolingual**: Posts and fact-checks are in the same language.
- **Crosslingual**: Posts and fact-checks may be in different languages.

For more details, see the [official task description](https://arxiv.org/abs/2505.10740) and the [SemEval-2025 website](https://semeval.github.io/SemEval2025/tasks).

## Repository Structure

This repository is organized for exploration and research, and includes several modules:

- **`embeddings/`**: Tools for generating and evaluating text embeddings using Sentence Transformers. Includes scripts and notebooks for running embedding-based retrieval and evaluation.
- **`evaluation/`**: Scripts and notebooks for evaluating retrieval results, including metrics such as success@10.
- **`train/`**: Code for training and fine-tuning sentence transformer models on the competition data.
- **`predict/`**: Scripts for generating predictions using trained models, as well as ensemble methods for combining results from multiple models.
- **`eda/`**: Notebooks for exploratory data analysis, including data loading and inspection.
- **`data/`**: Contains data files and subfolders for original, transformed, and prediction data. See the `README.md` files in the data subfolders for details on data formats.
- **`utils/`**: Utility scripts for data loading, saving, and other helper functions.
- **`notebooks/`**: Orchestration and exploratory notebooks for running end-to-end experiments.
- **`commands/`**: Contains the `install.sh` script for setting up the environment.

## Installation

To set up the environment and install all required dependencies, run:

```bash
bash commands/install.sh
```

This will install PyTorch, xformers, and all Python dependencies listed in `requirements.txt`.

## Usage

The repository is modular and intended for research and experimentation. Typical workflows include:

- **Generating Embeddings and Evaluating Retrieval**:
  - Use scripts in `embeddings/` to generate embeddings for posts and fact-checks, compute similarities, and evaluate retrieval performance.
  - Example: `python embeddings/run_evaluation.py --model_name_or_path <model> --model_type base --num_posts 100`

- **Training Models**:
  - Use scripts in `train/` to fine-tune sentence transformer models on the provided data.
  - Example: `python train/train_sentence_transformers.py --config <config_file>`

- **Making Predictions**:
  - Use scripts in `predict/` to generate predictions for new data or competition test sets.
  - Example: `python predict/predict.py --model_name_or_path <model> --data_paths <paths> --predictions_dir <output_dir>`
  - Ensemble methods are available in `predict/ensemble_predictor.py`.

- **Evaluation**:
  - Use scripts in `evaluation/` to compute metrics such as success@10 on your predictions.

- **Exploratory Data Analysis**:
  - Notebooks in `eda/` and `notebooks/` provide examples for loading and inspecting the data.

## Data

The `data/` directory contains all necessary files for training, validation, and testing, including:
- `fact_checks.csv`: Fact-checked claims in multiple languages.
- `posts.csv`: Social media posts.
- `pairs.csv`: Mappings between posts and relevant fact-checks.
- Submission templates and additional documentation are provided in subfolders.

See the `README.md` files in `data/original/`, `data/test_original/`, and `data/SemEval_Task7_Test_Phase/` for more information about the data format and usage.

## Notes

- This repository is intended for research and exploration. The codebase is modular, and users are encouraged to adapt scripts and notebooks for their own experiments.
- For questions about the competition, refer to the [official task page](https://semeval.github.io/SemEval2025/tasks) or the [DisAI Center](https://disai.eu/).

## References
- [SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval (arXiv)](https://arxiv.org/abs/2505.10740)
- [SemEval-2025 Official Website](https://semeval.github.io/SemEval2025/tasks)
- [DisAI Center](https://disai.eu/) 