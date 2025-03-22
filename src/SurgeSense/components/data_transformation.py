import os 
from SurgeSense import logger
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import os 
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split 
from SurgeSense.config.configuration import DataTransformationConfig

class DataTransformation: 
    def __init__(self,config:DataTransformationConfig):
        self.config=config


    def transform_data_pipeline(self):
        # data=pd.read_csv(self.config.data_path)
        categorical_columns=self.config.categorical_columns
        numerical_columns=self.config.numerical_columns
        numerical_preprocessor=Pipeline(
            steps=[
                ('imputation_menu',SimpleImputer(missing_values=np.nan,strategy='median')),
                ('scalar',StandardScaler())
            ]
        )

        categorical_preprocessor=Pipeline(
            steps=[
                ('imputation_constant',SimpleImputer(strategy='most_frequent')),
                ('encode',OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        preprocessor=ColumnTransformer(
            transformers=[
                ('categorical_columns',categorical_preprocessor,categorical_columns),
                ('numerical_columns',numerical_preprocessor,numerical_columns)
            ]
        )

        pipe=Pipeline(
            steps=[
                ('preprocessor',preprocessor)
            ]
        )
        return pipe 
    
    def train_test_spliting(self,pipe: Pipeline):
        data=pd.read_csv(self.config.data_path)
        data_transformed=pipe.fit_transform(data)
        logger.info('Transforming the data')
        train,test=train_test_split(pd.DataFrame(data_transformed))
        train.to_csv(os.path.join(self.config.root_dir,'train.csv'),index=False)
        test.to_csv(os.path.join(self.config.root_dir,'test.csv'),index=False)
        logger.info('splitting the data into train and test set')
        logger.info(f'training set shape: {train.shape}')
        logger.info(f'testing set shape: {test.shape}')