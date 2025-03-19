from SurgeSense import logger
from SurgeSense.pipeline.data_ingestion_stage import DataIngestionTrainingPipeline

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