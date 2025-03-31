from dataclasses import dataclass
from pathlib import Path 

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str 
    local_data_file: Path
    unzip_dir: Path 

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path 
    STATUS_FILE: str 
    unzip_data_dir: Path 
    all_schema: dict
 
@dataclass
class DataTransformationConfig:
    root_dir: Path 
    data_path: Path
    categorical_columns: list
    numerical_columns: list 


@dataclass
class ModelTrainConfig:
    root_dir: Path 
    train_data_path: Path 
    test_data_path: Path 
    model_name: str 
    n_estimators: int 
    max_depth: int 
    min_samples_split: int 
    learning_rate: int
    select_model: str
    target_column: str 
    categorical_columns:list
    numerical_columns:list
    drop_columns:list 
 

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path 
    test_data_path: Path 
    model_path: Path 
    all_params: dict 
    metric_file_name: Path 
    target_column: str 
    repo_owner: str 
    repo_name: str

@dataclass(frozen=True)
class HyperOptParamsXGBoost:
    root_dir: Path
    train_data_path: Path 
    test_data_path: Path
    model_name: str
    n_estimators: list 
    max_depth: list 
    learning_rate: str 
    target_column: str

@dataclass(frozen=True)
class HyperOptParamsGradientBoosting:
    root_dir: Path
    train_data_path: Path 
    test_data_path: Path
    model_name: str
    n_estimators: list 
    max_depth: list 
    learning_rate: str 
    target_column: str

@dataclass(frozen=True)
class HyperOptParamsRandomForest:
    root_dir: Path
    train_data_path: Path 
    test_data_path: Path
    model_name: str
    n_estimators: list 
    max_depth: list 
    min_samples_split: int 
    target_column: str