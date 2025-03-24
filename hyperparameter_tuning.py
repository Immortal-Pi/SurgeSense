from SurgeSense import logger
from SurgeSense.pipeline.xgboost_tuning import ModelTrainerTrainingPipelineXGBoost
from SurgeSense.pipeline.gradient_boosting_tuning import ModelTrainerTrainingPipelineGradientBoosting
from SurgeSense.pipeline.random_forest_tuning import ModelTrainerTrainingPipelineRandomForest
import sys


if sys.argv[1]=='XGBoost':

    STAGE_NAME='XGBoost Tuning'
    try:
        logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
        data_ingestion=ModelTrainerTrainingPipelineXGBoost()
        data_ingestion.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<< \n ')
    except Exception as e:
        logger.exception(e)
        raise e 

elif sys.argv[1]=='RandomForest tuning':
    STAGE_NAME='RandomForest'
    try:
        logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
        data_ingestion=ModelTrainerTrainingPipelineRandomForest()
        data_ingestion.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<< \n ')
    except Exception as e:
        logger.exception(e)
        raise e 

elif sys.argv[1]=='GradientBoosting':
    STAGE_NAME='Gradient Boosting Tuning'
    try:
        logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
        data_ingestion=ModelTrainerTrainingPipelineGradientBoosting()
        data_ingestion.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<< \n ')
    except Exception as e:
        logger.exception(e)
        raise e 