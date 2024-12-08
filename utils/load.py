"""
This is just a quick script that is able to load the files. Just using pandas can be tricky because of the newline characters in the text data. Here it is handled via the `parse_col` method.
"""

import ast
import os
import json

import pandas as pd

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
        df_fact_checks, df_posts, df_fact_check_post_mapping = self.load_data()

        # Load tasks.json
        with open(os.path.join(self.our_dataset_path, 'tasks.json'), 'r') as f:
            tasks = json.load(f)

        # Initialize dictionaries for storing datasets
        fact_checks_dfs = []
        posts_train_dfs = []
        posts_dev_dfs = []

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

            posts_train = df_posts[df_posts.index.isin(posts_train_ids)].copy()
            posts_train['language'] = language
            posts_train['post_id'] = posts_train.index

            posts_dev = df_posts[df_posts.index.isin(posts_dev_ids)].copy()
            posts_dev['language'] = language
            posts_dev['post_id'] = posts_dev.index

            # Append to the lists
            fact_checks_dfs.append(fact_checks)
            posts_train_dfs.append(posts_train)
            posts_dev_dfs.append(posts_dev)

        # Concatenate all language-specific dataframes
        fact_checks_df = pd.concat(fact_checks_dfs, ignore_index=True)
        posts_train_df = pd.concat(posts_train_dfs, ignore_index=True)
        posts_dev_df = pd.concat(posts_dev_dfs, ignore_index=True)

        # Return as a dictionary
        return {
            "fact_checks": fact_checks_df,
            "posts_train": posts_train_df,
            "posts_dev": posts_dev_df
        }

    

