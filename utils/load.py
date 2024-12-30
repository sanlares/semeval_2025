"""
This is just a quick script that is able to load the files. Just using pandas can be tricky because of the newline characters in the text data. Here it is handled via the `parse_col` method.
"""

import ast
import os
import json

import pandas as pd

from sklearn.model_selection import train_test_split

class LoadDataCSV:

    def __init__(self):

        self.our_dataset_path = 'data/'
    
    def load_data(self):

        posts_path = os.path.join(self.our_dataset_path, 'posts.csv')
        fact_checks_path = os.path.join(self.our_dataset_path, 'fact_checks.csv')
        fact_check_post_mapping_path = os.path.join(self.our_dataset_path, 'pairs.csv')

        for path in [posts_path, fact_checks_path, fact_check_post_mapping_path]:
            assert os.path.isfile(path)

        # We need to apply t = t.replace('\n', '\\n') for text fields before using `ast.literal_eval`.
        # `ast.literal_eval` has problems when there are new lines in the text, e.g.:
        # `ast.literal_eval('("\n")')` effectively tries to interpret the following code:

        # ```
        # ("
        # ")
        # ```

        # This raises a SyntaxError exception. By escaping new lines we are able to force it to interpret it properly. There might
        # be some other way to do this more systematically, but it is a workable fix for now.

        parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s

        df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')
        for col in ['claim', 'instances', 'title']:
            df_fact_checks[col] = df_fact_checks[col].apply(parse_col)


        df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
        for col in ['instances', 'ocr', 'verdicts', 'text']:
            df_posts[col] = df_posts[col].apply(parse_col)


        df_fact_check_post_mapping = pd.read_csv(fact_check_post_mapping_path) 

        return df_fact_checks, df_posts, df_fact_check_post_mapping

    def combine_ocr_text(row):
        ocr = row['ocr']
        text = row['text']

        result = 'passage: '

        # Handle OCR
        if isinstance(ocr, list) and len(ocr) > 0:
            # If OCR is not empty, add first OCR text
            result += str(ocr[0][0] if isinstance(ocr[0], tuple) else ocr[0])

        # Handle text
        if isinstance(text, tuple):
            # Add text content if it exists
            result += ' ' + str(text[0])
        elif isinstance(text, str) and text.startswith('('):
            # Handle string representation of tuple
            try:
                # Extract content between first parentheses
                text_content = text.split("'")[1] if "'" in text else text[1:-1]
                result += ' ' + text_content
            except:
                pass

        return result.strip()
    
    def combine_claim_text(row):
        claim = row['claim']
        title = row['title']

        result = 'query: '

        # Handle claim
        if isinstance(claim, tuple):
            # Extract first element from tuple
            result += str(claim[0])
        elif isinstance(claim, str) and claim.startswith('('):
            # Handle string representation of tuple
            try:
                # Extract content between first quotes
                claim_content = claim.split('"')[1] if '"' in claim else claim.split("'")[1]
                result += claim_content
            except:
                pass

        # Handle title
        if isinstance(title, tuple):
            result += ' ' + str(title[0])
        elif isinstance(title, str) and title.startswith('('):
            try:
                # Extract content between first quotes
                title_content = title.split('"')[1] if '"' in title else title.split("'")[1]
                result += ' ' + title_content
            except:
                pass

        return result.strip()
    
    def split_data(self):
        """
        Splits the data into fact checks, posts_train, and posts_dev datasets for each language in tasks.json.
        Adds a 'language' column to each dataset and includes unique identifiers for each dataset:
            - 'fact_check_id' for fact_checks
            - 'post_id' for posts_train
            - 'post_id' for posts_dev

        Returns:
            dict: A dictionary containing the datasets for all languages with the format:
                {
                    "fact_checks": DataFrame,
                    "posts_train": DataFrame,
                    "posts_dev": DataFrame
                }
        """
        # Load the data
        df_fact_checks, df_posts, df_pairs = self.load_data()

        # Load tasks.json
        with open(os.path.join(self.our_dataset_path, 'tasks.json'), 'r') as f:
            tasks = json.load(f)

        # Initialize dictionaries for storing datasets
        fact_checks_all = []
        posts_train_all = []
        posts_dev_all = []
        pairs_all = []

        posts_train_train = []
        posts_train_val = []
        fact_checks_train = []
        fact_checks_val = []
        pairs_train = []
        pairs_val = []

        # Iterate through all languages in monolingual data
        for language, language_data in tasks["monolingual"].items():
            # Extract IDs for the current language
            fact_check_ids = language_data.get("fact_checks", [])
            posts_train_ids = language_data.get("posts_train", [])
            posts_dev_ids = language_data.get("posts_dev", [])

            # Filter and annotate the dataframes
            fact_checks = df_fact_checks[df_fact_checks.index.isin(fact_check_ids)].copy()
            fact_checks['language'] = language
            fact_checks['fact_check_id'] = fact_checks.index
            fact_checks['claim_title'] = fact_checks.apply(self.combine_claim_text, axis=1)

            posts_train = df_posts[df_posts.index.isin(posts_train_ids)].copy()
            posts_train['language'] = language
            posts_train['post_id'] = posts_train.index
            posts_train['text_ocr'] = posts_train.apply(self.combine_ocr_text, axis=1)

            posts_dev = df_posts[df_posts.index.isin(posts_dev_ids)].copy()
            posts_dev['language'] = language
            posts_dev['post_id'] = posts_dev.index
            posts_dev['text_ocr'] = posts_dev.apply(self.combine_ocr_text, axis=1)

            posts_train_train, posts_train_val = train_test_split(posts_train, test_size=0.2, random_state=42)

            fact_checks_train=fact_checks[fact_checks.index.isin(df_pairs[df_pairs['post_id'].isin(posts_train_train.index)]['fact_check_id'])]
            fact_checks_val=fact_checks[fact_checks.index.isin(df_pairs[df_pairs['post_id'].isin(posts_train_val.index)]['fact_check_id'])]


            pairs_df=df_pairs[df_pairs['post_id'].isin(posts_train.index)]
            pairs_train=df_pairs[df_pairs['post_id'].isin(posts_train_train.index)]
            pairs_val=df_pairs[df_pairs['post_id'].isin(posts_train_val.index)]

            # Append to the lists
            fact_checks_all.append(fact_checks)
            posts_train_all.append(posts_train)
            posts_dev_all.append(posts_dev)

            fact_checks_train.append(fact_checks_train)
            fact_checks_val.append(fact_checks_val)
            pairs_train.append(pairs_train)
            pairs_val.append(pairs_val)
            pairs_all.append(pairs_df)

        # Concatenate all language-specific dataframes
        fact_checks_df = pd.concat(fact_checks_all, ignore_index=True)
        posts_train_df = pd.concat(posts_train_all, ignore_index=True)
        posts_dev_df = pd.concat(posts_dev_all, ignore_index=True)
        pairs_df = pd.concat(pairs_all, ignore_index=True)

        posts_train_train_df = pd.concat(posts_train_train, ignore_index=True)
        posts_train_val_df = pd.concat(posts_train_val, ignore_index=True)
        fact_checks_train_df = pd.concat(fact_checks_train, ignore_index=True)
        fact_checks_val_df = pd.concat(fact_checks_val, ignore_index=True)
        pairs_train_df = pd.concat(pairs_train, ignore_index=True)
        pairs_val_df = pd.concat(pairs_val, ignore_index=True)

        # Return as a dictionary
        return {
            "fact_checks": fact_checks_df,
            "posts_train": posts_train_df,
            "posts_dev": posts_dev_df,
            "pairs": pairs_df,
            "posts_train_train": posts_train_train_df,
            "posts_train_val": posts_train_val_df,
            "fact_checks_train": fact_checks_train_df,
            "fact_checks_val": fact_checks_val_df,
            "pairs_train": pairs_train_df,
            "pairs_val": pairs_val_df
        }

    

    

