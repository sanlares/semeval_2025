"""Utils for embeddings generation and loading."""
import ast
import numpy as np
import pandas as pd

import numpy as np
import ast

class Embeddings:

    def restore_embeddings(df, embedding_columns):
        """
        Restores the embeddings from a DataFrame that has been saved as a CSV file.

        Args:
            df (pd.DataFrame): DataFrame with the embeddings.
            embedding_columns (list): List of columns that contain the embeddings.

        Returns:
            pd.DataFrame: DataFrame with the embeddings restored.
        """
        for col in embedding_columns:
            def safe_eval(x):
                try:
                    if isinstance(x, np.ndarray):  # Ya es un array de numpy
                        return x.astype('float32')
                    elif isinstance(x, list):  # Es una lista estándar
                        return np.array(x, dtype='float32')
                    elif isinstance(x, str):  # Es una cadena que necesita evaluación
                        return np.array(ast.literal_eval(x), dtype='float32')
                    else:
                        return np.nan  # Cualquier otro tipo es inválido
                except Exception as e:
                    print(f"Error evaluando fila: {x}, Error: {e}")
                    return np.nan
            
            # Aplicar conversión segura
            df[col] = df[col].apply(safe_eval)
        return df
    
    def prepare_df_for_saving(df, embedding_columns):
        # Create a copy to avoid modifying original
        df_to_save = df.copy()

        for col in embedding_columns:
        
            # Convert numpy arrays to list format
            df_to_save[col] = df_to_save[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
        
        return df_to_save

    def success_at_10(predictions_df, pairs_df):
        """
        Calcula la métrica Success-at-10.

        Args:
            predictions_df (DataFrame): Contiene las predicciones con columnas 'post_id' y 'predicted_fact_check_ids'.
            pairs_df (DataFrame): Contiene las respuestas correctas con columnas 'post_id' y 'fact_check_id'.

        Returns:
            DataFrame: DataFrame con una nueva columna 'success_at_10' indicando si la predicción fue correcta.
        """
        # Crear un diccionario de respuestas correctas: {post_id: set(fact_check_ids)}
        correct_answers = pairs_df.groupby('post_id')['fact_check_id'].apply(set).to_dict()

        # Evaluar cada post_id en las predicciones
        success_scores = []

        for _, row in predictions_df.iterrows():
            post_id = row['post_id']
            predicted_ids = set(int(item['fact_check_id']) for item in row['top_k'])

            # Verificar si alguna predicción es correcta
            success = 1 if post_id in correct_answers and predicted_ids & correct_answers[post_id] else 0
            success_scores.append(success)
        
        # Añadir la columna 'success_at_10' al DataFrame
        predictions_df['success_at_10'] = success_scores

        return predictions_df