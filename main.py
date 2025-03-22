from SurgeSense import logger
from SurgeSense.pipeline.data_ingestion_stage import DataIngestionTrainingPipeline
from SurgeSense.pipeline.data_validation_stage import DataValidationTrainingPipeline
from SurgeSense.pipeline.data_transform_stage import DataTransformationTrainingPipeline

logger.info('look up in the sky! its a bird? its a plane? no its immortalpi in the main function')

STAGE_NAME='DATA INGESTION STAGE'
try:
    logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<< \n ')
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME='DATA VALIDATION STAGE'
try:
    logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
    obj=DataValidationTrainingPipeline()
    obj.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<< \n\n x==========x')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME='MODEL TRANSFORMATION STAGE'
try:
    logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
    data_transformation=DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<\n\n x==========x')
except Exception as e:
    logger.exception(e)
    raise e