import  os 
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from urllib.parse import urlparse
import mlflow 
import mlflow.sklearn 
import numpy as np 
import joblib 
import dagshub 
from SurgeSense.config.configuration import ModelEvaluationConfig
from SurgeSense.utils.common import * 


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config=config

    def eval_metrics(self,actual,pred):
        rmse=np.sqrt(mean_squared_error(actual,pred))
        mae=mean_squared_error(actual,pred)
        r2=r2_score(actual,pred)
        return rmse,mae,r2 
    
    def log_into_mlflow(self):
        test_data=pd.read_csv(self.config.test_data_path)
        model=joblib.load(self.config.model_path)

        test_x=test_data.drop([self.config.target_column],axis=1)
        test_y=test_data[[self.config.target_column]]

        dagshub.init(repo_name=self.config.repo_name, repo_owner=self.config.repo_owner,mlflow=True)
        tracking_uri_type_store=urlparse(mlflow.get_registry_uri()).scheme

        with mlflow.start_run():
            predicted_quantities=model.predict(test_x)
            (rmse,mae,r2)=self.eval_metrics(test_y,predicted_quantities)

            # saving metrics as local 
            score={'rmse':rmse, 'r2_score':r2, 'mae':mae}
            save_json(path=Path(self.config.metric_file_name),data=score)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(score)

            # if tracking_uri_type_store!='file':
            #     if self.config.all_params.algo=='XGBoostRegressor':
            #         mlflow.xgboost.log_model(model,'model',registered_model_name='XGBoostRegressor')
            #     elif self.config.all_params.algo=='GRADIENT_BOOSTING':
            #         mlflow.sklearn.log_model(model,'model',registered_model_name='GRADIENT_BOOSTING')
            #     elif self.config.all_params.algo=='RANDOM_FOREST':
            #         mlflow.sklearn.log_model(model,'model',registered_model_name='RANDOM_FOREST')
                    
