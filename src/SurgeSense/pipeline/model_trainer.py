from SurgeSense.config.configuration import ConfigurationManager
from SurgeSense.components.model_training import ModelTrainer
from SurgeSense import logger 

STAGE_NAME='Model Trainer Stage'


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_train_config()
        model_trainer=ModelTrainer(config=model_trainer_config)
        pipe=model_trainer.create_pipeline()
        model_trainer.train(pipe)


if __name__=='__main__':
    try:
        logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
        obj=ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x')
    except Exception as e :
        logger.exception(e)
        raise e 