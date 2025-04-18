from SurgeSense.constants import * 
from SurgeSense.utils.common import read_yaml,create_directories
from SurgeSense.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainConfig, ModelEvaluationConfig, HyperOptParamsGradientBoosting, HyperOptParamsRandomForest, HyperOptParamsXGBoost

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.param=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config


    def get_data_validation_config(self)-> DataValidationConfig:
        config=self.config.data_validation
        schema=self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config=DataValidationConfig(
            unzip_data_dir=config.unzip_data_dir,
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema 
        )
        return data_validation_config
    

    def get_data_transformation_config(self)->DataTransformationConfig:
        config=self.config.data_transformation 
        schema=self.schema.TRANSFORM
        create_directories([config.root_dir])

        data_transformation_config=DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            categorical_columns=schema.CATEGORICAL_DATA,
            numerical_columns=schema.NUMERICAL_DATA
        )
        return data_transformation_config
    
    def get_model_train_config(self)->ModelTrainConfig:
        config=self.config.model_trainer
        params=self.param.select_model 
        schema=self.schema
        create_directories([config.root_dir])

        model_train_config=ModelTrainConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            learning_rate=params.learning_rate,
            select_model=params.algo,
            target_column=schema.TARGET_COLUMN.name,
            categorical_columns=schema.TRANSFORM.CATEGORICAL_DATA,
            numerical_columns=schema.TRANSFORM.NUMERICAL_DATA,
            drop_columns=schema.DROP_COLUMNS

        )
        return model_train_config
    
    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        config=self.config.model_evaluation
        params=self.param.select_model
        schema=self.schema.TARGET_COLUMN
        mlflow_tracking=self.config.mlflow_tracking
        create_directories([config.root_dir])


        model_evlution_config=ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name,
            repo_owner=mlflow_tracking.repo_owner,
            repo_name=mlflow_tracking.repo_name
        )

        return model_evlution_config
    
    def get_hyperopt_config_XGBoost(self)->HyperOptParamsXGBoost:
        config=self.config.model_trainer 
        params=self.param.Hyperopt_params.XGBoostRegressor
        schema=self.schema.TARGET_COLUMN

        hypoeropt_config=HyperOptParamsXGBoost(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            target_column=schema.name
        )
        return hypoeropt_config
    
    def get_hyperopt_config_GradientBoosting(self)->HyperOptParamsGradientBoosting:
        config=self.config.model_trainer 
        params=self.param.Hyperopt_params.GRADIENT_BOOSTING
        schema=self.schema.TARGET_COLUMN

        hypoeropt_config=HyperOptParamsGradientBoosting(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            target_column=schema.name
        )
        return hypoeropt_config
    
    def get_hyperopt_config_RandomForest(self)->HyperOptParamsRandomForest:
        config=self.config.model_trainer 
        params=self.param.Hyperopt_params.RANDOM_FOREST
        schema=self.schema.TARGET_COLUMN

        hypoeropt_config=HyperOptParamsRandomForest(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            target_column=schema.name
        )
        return hypoeropt_config