from hyperopt import STATUS_OK, hp, fmin, tpe, Trials
import dagshub
from functools import partial 
import mlflow
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error,r2_score,mean_squared_error
from SurgeSense.config.configuration import HyperOptParamsGradientBoosting

class hyperOptTraining:
    def __init__(self,config=HyperOptParamsGradientBoosting):
          self.config=config


    def evaluation_metrics(actual,pred):
        rmse=np.sqrt(mean_squared_error(actual,pred))
        mae=mean_absolute_error(actual,pred)
        r2=r2_score(actual,pred)
        return rmse, mae, r2 


    def create_pipeline(self):
        
        categorical_columns=['cab_type','destination','source','name']
        numerical_columns=['distance','surge_multiplier','temp','clouds','pressure','rain','humidity','wind','day','hour','month']

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
                ('preprocessor',preprocessor),
                ('model', GradientBoostingRegressor())
            ]
        )
        

        return pipe

    def evaluation_metrics(self,actual,pred):
        rmse=np.sqrt(mean_squared_error(actual,pred))
        mae=mean_absolute_error(actual,pred)
        r2=r2_score(actual,pred)
        return rmse, mae, r2

    def objective(self,params,xtrain,ytrain,xtest,ytest):
        with mlflow.start_run():
            mlflow.set_tag('model','gradientboost')

            pipe=self.create_pipeline()
            model=pipe.set_params(**params)
            model.fit(xtrain,ytrain)
            ypred=model.predict(xtest)
            mlflow.log_params(model.get_params())
            rmse,mae,r2=self.evaluation_metrics(ytest,ypred)
            mlflow.log_metrics({'rmse':rmse,'mse': mae, 'r2':r2})
        return {'loss':rmse, 'status':STATUS_OK, 'model':model}


    def train(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data=pd.read_csv(self.config.test_data_path)

        xtrain=train_data.drop([self.config.target_column],axis=1)
        xtest=test_data.drop([self.config.target_column],axis=1)
        ytrain=train_data[[self.config.target_column]]
        ytest=test_data[[self.config.target_column]]

        search_space={
            'model__n_estimators':hp.uniformint('n_estimators',self.config.n_estimators[0],self.config.n_estimators[1]),
            'model__max_depth':hp.uniformint('max_depth',self.config.max_depth[0],self.config.max_depth[1]),
            'model__learning_rate':hp.uniform('learning_rate',self.config.learning_rate[0],self.config.learning_rate[1])
        }
        dagshub.init(repo_owner='Immortal-Pi',repo_name='SurgeSense',mlflow=True)
        experiment_name='hyperopt_test_gradient_boosting'
        existing_experiment=mlflow.get_experiment_by_name(experiment_name)

        if existing_experiment is None:
                experiment_id = mlflow.create_experiment(name=experiment_name,artifact_location='hyperopt-test')
        else:
            experiment_id = existing_experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id) 

        trials=Trials()
        best_results=fmin(
            fn=partial(
                self.objective,
                xtrain=xtrain[:1000],
                ytrain=ytrain[:1000],
                xtest=xtest[:1000],
                ytest=ytest[:1000]
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=10, # change to config
            trials=trials
        )
        return best_results,trials

    def register_best_model(self,best_results,trials):
        best_index=np.argmin([trial['result']['loss'] for trial in trials.trials])
        best_model=trials.trials[best_index]['result']['model']

        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(best_model,artifact_path='model')
            mlflow.log_params(trials.trials[best_index]['misc']['vals'])
            model_uri=f'runs:/{run.info.run_id}/best_model'
            mlflow.register_model(model_uri=model_uri,name='best_model')


