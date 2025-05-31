import pandas as pd
import os

def load_gym_data():
    """
    Load the gym dataset from the archive directory.
    Returns:
        pandas.DataFrame: The loaded dataset
    Raises:
        FileNotFoundError: If the dataset file is not found
        pd.errors.EmptyDataError: If the dataset file is empty
    """
    try:
        # Get the absolute path to the dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, '..', '..', 'archive', 'megaGymDataset.csv')
        
        # Load the dataset
        data = pd.read_csv(dataset_path)
        
        # Basic validation
        if data.empty:
            raise pd.errors.EmptyDataError("The dataset is empty")
            
        # Ensure required columns exist
        required_columns = ['Exercise Name', 'BodyPart']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The dataset file is empty")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")
