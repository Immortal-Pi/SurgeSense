import joblib 
import numpy as np 
import pandas as pd 
from pathlib import Path 
from SurgeSense.constants import * 
from SurgeSense.utils.common import read_yaml


class PredictionPipeline: 
    def __init__(self, config=CONFIG_FILE_PATH):
        self.config=read_yaml(config)
        self.model=joblib.load(Path(self.config.model_evaluation.model_path))

    def predict(self,data):
        prediction=self.model.predict(data)
        return prediction
