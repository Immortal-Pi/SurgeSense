import pandas as pd 
import os 
from SurgeSense import logger
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd 
import numpy as np 
from SurgeSense.config.configuration import ModelTrainConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainConfig):
        self.config=config

    def create_pipeline(self):
        
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

        if self.config.select_model=='XGBoostRegressor':
            pipe=Pipeline(
                steps=[
                    ('preprocessor',preprocessor),
                    ('model', XGBRegressor(
                        n_estimators=self.config.n_estimators,
                        learning_rate= self.config.learning_rate,
                        max_depth=self.config.max_depth
                    ))
                ]
            )
        elif self.config.select_model=='GRADIENT_BOOSTING':
            pipe=Pipeline(
                steps=[
                    ('preprocessor',preprocessor),
                    ('model', GradientBoostingRegressor(
                        n_estimators=self.config.n_estimators,
                        learning_rate= self.config.learning_rate,
                        max_depth=self.config.max_depth
                    ))
                ]
            )
        elif self.config.select_model=='RANDOM_FOREST':
            pipe=Pipeline(
                steps=[
                    ('preprocessor',preprocessor),
                    ('model', RandomForestRegressor(
                        n_estimators=self.config.n_estimators,
                        learning_rate= self.config.learning_rate,
                        max_depth=self.config.max_depth
                    ))
                ]
            )

        return pipe

    def train(self, pipe: Pipeline):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data=pd.read_csv(self.config.test_data_path)

        train_x=train_data.drop([self.config.target_column]+self.config.drop_columns,axis=1)
        test_x=test_data.drop([self.config.target_column]+self.config.drop_columns,axis=1)
        train_y=train_data[[self.config.target_column]]
        test_y=test_data[[self.config.target_column]]
        # print(test_x.columns)
        pipe.fit(train_x,train_y)
        joblib.dump(pipe,os.path.join(self.config.root_dir,self.config.model_name))
