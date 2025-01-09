import os
import zipfile
import logging
from typing import List, Union

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)

def create_zip(file_paths: Union[str, List[str]], output_path: str, flatten: bool = True) -> None:
    """
    Create a zip file containing one or more files.
    
    Args:
        file_paths: Single file path or list of file paths to include in the zip
        output_path: Path where to save the zip file
        flatten: If True, remove directory structure in the zip file (default: True)
    
    Example:
        >>> create_zip('data/predictions/monolingual_predictions.json', 'models/predictions.zip')
        >>> create_zip(['data/pred1.json', 'data/pred2.json'], 'models/all_predictions.zip')
    """
    # Convert single path to list
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logging.warning(f"File not found: {file_path}")
                    continue
                    
                # Determine arcname (name inside zip)
                if flatten:
                    arcname = os.path.basename(file_path)
                else:
                    arcname = file_path
                
                logging.info(f"Adding {file_path} to zip as {arcname}")
                zipf.write(file_path, arcname)
                
        logging.info(f"Zip file created successfully at: {output_path}")
        
    except Exception as e:
        logging.error(f"Error creating zip file: {str(e)}")
        raise

def extract_zip(zip_path: str, output_dir: str = None) -> None:
    """
    Extract a zip file to a directory.
    
    Args:
        zip_path: Path to the zip file to extract
        output_dir: Directory where to extract files (default: same directory as zip)
    
    Example:
        >>> extract_zip('models/predictions.zip', 'data/extracted/')
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    # If no output directory specified, use same directory as zip
    if output_dir is None:
        output_dir = os.path.dirname(zip_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            logging.info(f"Extracting {zip_path} to {output_dir}")
            zipf.extractall(output_dir)
            logging.info("Extraction completed successfully")
            
    except Exception as e:
        logging.error(f"Error extracting zip file: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Create a zip file with a single file
    create_zip(
        'data/predictions/monolingual_predictions.json',
        'models/predictions.zip'
    )
    
    # Create a zip file with multiple files
    create_zip(
        [
            'data/predictions/monolingual_predictions.json',
            'data/predictions/crosslingual_predictions.json'
        ],
        'models/all_predictions.zip'
    )
    
    # Extract a zip file
    extract_zip('models/predictions.zip', 'data/extracted/') 