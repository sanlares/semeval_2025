import pandas as pd
from utils.load import LoadDataCSV
import os

def test_data_splits():
    """
    Test the data splitting functionality and verify data quality
    """
    # Initialize loader
    loader = LoadDataCSV()
    
    # Load and split data
    split_data = loader.split_data()
    
    # Get individual datasets
    fact_checks = split_data["fact_checks"]
    posts_train = split_data["posts_train"] 
    posts_dev = split_data["posts_dev"]
    pairs = split_data["pairs"]
    
    posts_train_train = split_data["posts_train_train"]
    posts_train_val = split_data["posts_train_val"]
    fact_checks_train = split_data["fact_checks_train"]
    fact_checks_val = split_data["fact_checks_val"]
    pairs_train = split_data["pairs_train"]
    pairs_val = split_data["pairs_val"]

    # Print dataset sizes
    print("\nDataset sizes:")
    print(f"Fact checks: {len(fact_checks)}")
    print(f"Posts train: {len(posts_train)}")
    print(f"Posts dev: {len(posts_dev)}")
    print(f"Pairs: {len(pairs)}")
    print(f"\nTrain/Val splits:")
    print(f"Posts train-train: {len(posts_train_train)}")
    print(f"Posts train-val: {len(posts_train_val)}")
    print(f"Fact checks train: {len(fact_checks_train)}")
    print(f"Fact checks val: {len(fact_checks_val)}")
    print(f"Pairs train: {len(pairs_train)}")
    print(f"Pairs val: {len(pairs_val)}")

    # Verify no overlap between train and val
    train_posts = set(posts_train_train.index)
    val_posts = set(posts_train_val.index)
    overlap = train_posts & val_posts
    print(f"\nOverlap between train and val posts: {len(overlap)}")

    # Check text combinations
    print("\nChecking text combinations...")
    
    # Check posts
    empty_text_ocr = posts_train[posts_train['text_ocr'].isna() | (posts_train['text_ocr'] == '')].index
    print(f"Posts with empty text_ocr: {len(empty_text_ocr)}")
    if len(empty_text_ocr) > 0:
        print("Sample empty post IDs:", list(empty_text_ocr)[:5])

    # Check fact checks  
    empty_claim_title = fact_checks[fact_checks['claim_title'].isna() | (fact_checks['claim_title'] == '')].index
    print(f"Fact checks with empty claim_title: {len(empty_claim_title)}")
    if len(empty_claim_title) > 0:
        print("Sample empty fact check IDs:", list(empty_claim_title)[:5])

    # Verify pairs consistency
    train_pairs_posts = set(pairs_train['post_id'])
    train_posts_ids = set(posts_train_train.index)
    missing_posts = train_pairs_posts - train_posts_ids
    print(f"\nMissing posts in train pairs: {len(missing_posts)}")

    val_pairs_posts = set(pairs_val['post_id']) 
    val_posts_ids = set(posts_train_val.index)
    missing_val_posts = val_pairs_posts - val_posts_ids
    print(f"Missing posts in val pairs: {len(missing_val_posts)}")

    # Check languages
    print("\nLanguage distribution in training:")
    print(posts_train_train['language'].value_counts())

if __name__ == "__main__":
    test_data_splits() 