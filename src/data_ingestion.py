import pandas as pd
from typing import Tuple, Dict, Any
import os
import logging
import yaml
class DataIngestion:
    def __init__(self):
        self.config = self.load_config()
    def load_config(self):
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            return config
    def load_data(self) -> Tuple[pd.DataFrame]:
        """
        Load data from the specified source in the config file.
        Returns a tuple of DataFrames (train, test).
        """
        data_path = self.config["data"]["data_path"]
        if not os.path.exists(data_path):
            logging.error(f"Data path {data_path} does not exist.")
            raise FileNotFoundError(f"Data path {data_path} does not exist.")
        
        try:
            df = pd.read_csv(data_path)
            logging.info("Data loaded successfully.")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise e

